"""
The agentic loop: analyze -> retrieve -> grade -> [conditional] -> rewrite (loops
back to retrieve) OR generate OR fallback.

Deliberately built by ORCHESTRATING Phase 1's existing components (guardrails,
query_processing, retrieval_qdrant, rerank, generate), not rebuilding them.
The only genuinely new logic here is the grader and rewriter, plus the graph
structure itself -- everything else is the same functions Phase 1 already
verified, just called from graph nodes instead of directly from FastAPI.

This is what makes it agentic rather than a longer straight line: the
router node is deterministic logic reading state (score, retry_count), not
free-form LLM reasoning about what to do next -- same distinction your
Guidewire doc draws between this pattern and ReAct.
"""

import os
import sys
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from langgraph.graph import StateGraph, END

from agentic_state import AgenticRAGState
from guardrails import redact_pii, check_injection
from query_processing import expand_acronyms, route_product
from retrieval_qdrant import QdrantHybridRetriever
from retrieval_grader import grade_retrieval
from query_rewriter import rewrite_query
from answer_grader import grade_answer
from generate import generate_answer

RELEVANCE_THRESHOLD = 0.7  # matches the Guidewire doc's documented threshold
FALLBACK_SCORE_FLOOR = 0.3  # below this, even with chunks present, escalate rather than let the generator say "I don't know"
FAITHFULNESS_THRESHOLD = 0.8  # matches Phase 1's RAGAS CI-gate threshold -- same bar, now checked live instead of only offline

# Built once, reused across every graph invocation -- rebuilding the Qdrant
# collection per-query would be wasteful and pointless, same reasoning as
# app/main.py building _retriever once at startup.
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = QdrantHybridRetriever()
    return _retriever


# ============================================================
# NODES
# ============================================================

def analyze_node(state: AgenticRAGState) -> dict:
    if check_injection(state["query"]):
        raise ValueError("Request blocked: possible prompt injection")

    redacted = redact_pii(state["query"])
    expanded = expand_acronyms(redacted)
    product = route_product(expanded)

    return {
        "redacted_query": redacted,
        "product": product,
        "route_history": state.get("route_history", []) + ["analyze"],
    }


def retrieve_node(state: AgenticRAGState) -> dict:
    # Use the rewritten query if the rewriter has run at least once this
    # session, otherwise the expanded original -- this is the mechanism
    # that makes the retry loop actually different each time instead of
    # repeating the identical failed search.
    query_for_search = state.get("rewritten_query") or expand_acronyms(state["redacted_query"])

    retriever = _get_retriever()
    candidates = [c for c, _ in retriever.hybrid_search(query_for_search, k=10, product_filter=state.get("product"))]

    from rerank_gemini import rerank_with_gemini
    top_chunks = rerank_with_gemini(state["redacted_query"], candidates, top_n=5)

    return {
        "chunks": top_chunks,
        "route_history": state["route_history"] + ["retrieve"],
    }


def grade_node(state: AgenticRAGState) -> dict:
    score = grade_retrieval(state["redacted_query"], state["chunks"])
    return {
        "relevance_score": score,
        "route_history": state["route_history"] + [f"grade({score:.2f})"],
    }


def rewrite_node(state: AgenticRAGState) -> dict:
    source_query = state.get("rewritten_query") or state["redacted_query"]
    rewritten = rewrite_query(source_query)
    return {
        "rewritten_query": rewritten,
        "retry_count": state.get("retry_count", 0) + 1,
        "route_history": state["route_history"] + [f"rewrite -> {rewritten!r}"],
    }


def generate_node(state: AgenticRAGState) -> dict:
    answer = generate_answer(state["redacted_query"], state["chunks"])
    sources = list({f"{c['metadata']['product']}/{c['metadata']['section']}" for c in state["chunks"]})
    return {
        "generation": answer,
        "sources": sources,
        "route_history": state["route_history"] + ["generate"],
    }


def grade_answer_node(state: AgenticRAGState) -> dict:
    score = grade_answer(state["generation"], state["chunks"])
    return {
        "faithfulness_score": score,
        "route_history": state["route_history"] + [f"grade_answer({score:.2f})"],
    }


def fallback_node(state: AgenticRAGState) -> dict:
    # Two different paths can reach fallback now: retrieval never found
    # anything usable (no generation attempted yet), or generation was
    # attempted but repeatedly failed the faithfulness check. Different
    # situations, different honest message -- don't claim "couldn't find
    # information" when an answer was actually generated and rejected.
    if state.get("generation") and state.get("faithfulness_score", 0.0) > 0:
        message = (
            f"I found information related to \"{state['query']}\", but couldn't "
            f"generate an answer I could verify as fully grounded in the source "
            f"material after {state.get('answer_retry_count', 0)} attempt(s).\n\n"
            f"This has been escalated for human review rather than risk an "
            f"inaccurate answer."
        )
    else:
        message = (
            f"I couldn't find reliable information to answer: \"{state['query']}\"\n\n"
            f"This has been escalated for human review. In the meantime, try "
            f"rephrasing with more specific terms."
        )
    return {
        "generation": message,
        "sources": [],
        "route_history": state["route_history"] + ["fallback"],
    }


def answer_retry_bump_node(state: AgenticRAGState) -> dict:
    """A generated answer failed the faithfulness check. Bump the SEPARATE
    answer-retry counter (not retry_count, which tracks retrieval-grade
    retries -- these are different failure modes with different budgets),
    then loop back through rewrite -> retrieve -> ... -> generate for a
    fresh attempt, on the theory that different/better context is more
    likely to fix a hallucination than regenerating from the identical
    chunks that already produced one."""
    return {
        "answer_retry_count": state.get("answer_retry_count", 0) + 1,
        "route_history": state["route_history"] + ["answer_retry_bump"],
    }


# ============================================================
# ROUTER (conditional edge -- deterministic, reads state, not an LLM call)
# ============================================================

def route_after_grade(state: AgenticRAGState) -> Literal["generate", "rewrite", "fallback"]:
    """Real bug found on live data: Qdrant's similarity search almost never
    returns truly empty results -- it returns the k nearest vectors
    regardless of how irrelevant they are. A genuinely rock-bottom query
    (grade 0.00, topic completely absent from the corpus) still had
    non-empty `chunks`, so the old `chunks else fallback` check never
    fired -- it routed to generate and relied on the LLM to say "I don't
    know" instead of actually escalating. Fixed by checking the SCORE at
    the retry-exhausted point, not just whether chunks exist."""
    score = state.get("relevance_score", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    chunks = state.get("chunks", [])

    if score >= RELEVANCE_THRESHOLD and chunks:
        return "generate"
    if retry_count < max_retries:
        return "rewrite"
    # Out of retries -- decide between honest best-effort generation and
    # human escalation based on how bad the score actually is, not just
    # whether Qdrant happened to return any results at all.
    if not chunks or score < FALLBACK_SCORE_FLOOR:
        return "fallback"
    return "generate"


def route_after_answer_grade(state: AgenticRAGState) -> Literal["end", "retry_answer", "fallback"]:
    """New this round: closes the loop on the GENERATION side, not just
    retrieval. A hallucinated answer can happen even with a perfect
    relevance_score -- retrieval being good doesn't guarantee the
    generator stayed faithful to it. Separate retry budget from the
    retrieval loop, since these are genuinely different failure modes."""
    score = state.get("faithfulness_score", 0.0)
    answer_retry_count = state.get("answer_retry_count", 0)
    max_answer_retries = state.get("max_answer_retries", 1)

    if score >= FAITHFULNESS_THRESHOLD:
        return "end"
    if answer_retry_count < max_answer_retries:
        return "retry_answer"
    return "fallback"  # exhausted answer retries, still not faithful -- don't return a possibly-hallucinated answer


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_agentic_graph():
    workflow = StateGraph(AgenticRAGState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_answer", grade_answer_node)
    workflow.add_node("answer_retry_bump", answer_retry_bump_node)
    workflow.add_node("fallback", fallback_node)

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade", route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"},
    )

    workflow.add_edge("rewrite", "retrieve")  # the retrieval-side loop

    # Generation no longer goes straight to END -- it's graded for
    # faithfulness first, closing the loop on the generation side too.
    workflow.add_edge("generate", "grade_answer")
    workflow.add_conditional_edges(
        "grade_answer", route_after_answer_grade,
        {"end": END, "retry_answer": "answer_retry_bump", "fallback": "fallback"},
    )
    workflow.add_edge("answer_retry_bump", "rewrite")  # the answer-side loop, via fresh retrieval context

    workflow.add_edge("fallback", END)

    return workflow.compile()


def run_agentic_query(query: str, max_retries: int = 2, max_answer_retries: int = 1) -> AgenticRAGState:
    app = build_agentic_graph()
    initial_state: AgenticRAGState = {
        "query": query,
        "redacted_query": "",
        "rewritten_query": "",
        "product": None,
        "chunks": [],
        "relevance_score": 0.0,
        "retry_count": 0,
        "max_retries": max_retries,
        "generation": "",
        "faithfulness_score": 0.0,
        "answer_retry_count": 0,
        "max_answer_retries": max_answer_retries,
        "sources": [],
        "route_history": [],
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    result = run_agentic_query("how does CC handle FNOL for a BI claim?")
    print("Route taken:", " -> ".join(result["route_history"]))
    print("\nAnswer:", result["generation"])
    print("\nSources:", result["sources"])
