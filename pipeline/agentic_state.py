"""
Shared state for the agentic RAG graph. Every node reads from this and
writes back into it -- this is what LangGraph passes between nodes, and
what makes the retry loop possible without losing context (contrast with
Phase 1's linear pipeline, which had no shared state to loop against at all).

Mirrors the state schema in your Guidewire interview doc's section 16.3,
built for real this time instead of illustrative-only.
"""

from typing import TypedDict


class AgenticRAGState(TypedDict):
    query: str                  # original user input, never mutated
    redacted_query: str         # after PII redaction + injection check
    rewritten_query: str        # query rewriter's output; "" until first rewrite
    product: str | None         # routed product (ClaimCenter/BillingCenter/PolicyCenter/None)

    chunks: list[dict]          # current retrieved+reranked chunks
    relevance_score: float      # retrieval grader's score for `chunks`, 0.0-1.0

    retry_count: int
    max_retries: int

    generation: str             # generator's answer text
    faithfulness_score: float   # answer grader's score, 0.0-1.0
    answer_retry_count: int     # separate from retry_count -- retrieval retries vs. hallucination retries are different failure modes
    max_answer_retries: int
    sources: list[str]

    route_history: list[str]    # every routing decision made this run, for debugging/tracing
