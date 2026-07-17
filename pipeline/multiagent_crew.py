"""
Phase 3: genuinely multi-agent, not another single-pipeline loop like Phase
2. Three specialist agents (ClaimCenter, BillingCenter, PolicyCenter), each
scoped to ONLY their product's documentation, coordinated by a manager that
decides which specialist(s) to consult per query -- a single-product
question invokes one agent, a cross-product question (e.g. the
renewal-affects-open-claim case) invokes multiple agents and the manager
synthesizes their answers.

This is a different architecture from Phase 2 on purpose, not a renamed
version of it -- see the module docstring comparison in the interview doc.
Phase 2 is one agent, self-correcting via a fixed graph. Phase 3 is
multiple agent identities, each with its own scope, coordinated by a
manager that reasons about delegation -- genuinely closer to free-form
agent reasoning than Phase 2's deterministic router.

Deliberately reuses Phase 1's retrieval/rerank components as CrewAI tools
instead of rebuilding retrieval logic a third time -- the tools below are
thin wrappers around retrieval_qdrant.QdrantHybridRetriever and
rerank_gemini, not new retrieval code.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from guardrails import redact_pii, check_injection
from retrieval_qdrant import QdrantHybridRetriever
from rerank_gemini import rerank_with_gemini

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = QdrantHybridRetriever()
    return _retriever


def _make_product_tool(product: str):
    """Builds a scoped retrieval tool for exactly one product. Each
    specialist agent gets its OWN tool instance bound to its product via
    closure -- this is what actually enforces "ClaimCenter agent can only
    see ClaimCenter docs," not just a naming convention in the agent's
    role description."""

    @tool(f"search_{product.lower()}_docs")
    def search_docs(query: str) -> str:
        """Search this product's documentation for information relevant to the query."""
        retriever = _get_retriever()
        candidates = [c for c, _ in retriever.hybrid_search(query, k=10, product_filter=product)]
        if not candidates:
            return f"No relevant {product} documentation found for this query."
        top_chunks = rerank_with_gemini(query, candidates, top_n=3)
        return "\n\n".join(
            f"[{c['metadata']['product']} / {c['metadata']['section']}]\n{c['text']}"
            for c in top_chunks
        )

    return search_docs


def build_crew(manager_llm: LLM | None = None) -> Crew:
    llm = LLM(model=f"gemini/{GEMINI_MODEL}", api_key=os.environ.get("GEMINI_API_KEY"))

    claim_agent = Agent(
        role="ClaimCenter Documentation Specialist",
        goal="Answer questions about claims, FNOL, deductibles, and claim reserves using ONLY ClaimCenter documentation",
        backstory="You are an expert in Guidewire ClaimCenter, the claims-handling system. "
                   "You only answer from ClaimCenter documentation -- if a question is about "
                   "billing or policy underwriting, say so rather than guessing.",
        tools=[_make_product_tool("ClaimCenter")],
        llm=llm,
        verbose=False,
    )

    billing_agent = Agent(
        role="BillingCenter Documentation Specialist",
        goal="Answer questions about premiums, invoices, payments, and the no claims discount using ONLY BillingCenter documentation",
        backstory="You are an expert in Guidewire BillingCenter, the billing system. "
                   "You only answer from BillingCenter documentation.",
        tools=[_make_product_tool("BillingCenter")],
        llm=llm,
        verbose=False,
    )

    policy_agent = Agent(
        role="PolicyCenter Documentation Specialist",
        goal="Answer questions about policy renewal, underwriting, and endorsements using ONLY PolicyCenter documentation",
        backstory="You are an expert in Guidewire PolicyCenter, the underwriting system. "
                   "You only answer from PolicyCenter documentation.",
        tools=[_make_product_tool("PolicyCenter")],
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=(
            "Answer this insurance documentation question: {query}\n\n"
            "If the question spans multiple products (e.g. how a policy renewal "
            "affects an open claim), consult ALL relevant specialists and "
            "synthesize their answers into one coherent response with citations. "
            "If a specialist has no relevant information, say so rather than "
            "inventing an answer."
        ),
        expected_output="A grounded answer with citations to the specific product/section it came from.",
        agent=claim_agent,  # placeholder assignment; manager reassigns dynamically in hierarchical mode
    )

    return Crew(
        agents=[claim_agent, billing_agent, policy_agent],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=False,
    )


def run_multiagent_query(query: str) -> dict:
    if check_injection(query):
        raise ValueError("Request blocked: possible prompt injection")
    redacted = redact_pii(query)

    crew = build_crew()
    result = crew.kickoff(inputs={"query": redacted})

    return {
        "query": query,
        "answer": str(result),
    }


if __name__ == "__main__":
    result = run_multiagent_query("how does the deductible work when a policy renews while a claim is open?")
    print("Answer:", result["answer"])
