"""
LLM-based reranking using Gemini's free tier, as an alternative to
downloading cross-encoder/ms-marco-MiniLM-L-6-v2 (rerank.py).

Be precise about what this is in an interview: this is NOT the same
technique as a trained cross-encoder. A cross-encoder is a small model
trained specifically to score (query, document) relevance pairs -- fast,
cheap, purpose-built. This is a general-purpose LLM being PROMPTED to act
as a relevance judge (this is sometimes called "LLM-as-reranker" or
pointwise LLM reranking, related to RankGPT-style listwise approaches).
It works, and it's genuinely used in production when a dedicated rerank
API isn't available or add-on cost isn't justified -- but it's higher
latency and less proven than a purpose-built cross-encoder or Cohere/
Voyage/Jina's dedicated rerank endpoints. Know the distinction; don't
present this as "the same thing but free."

Also note: as of mid-2026, Gemini's free tier is Flash/Flash-Lite only
(Pro was moved to paid-only in April 2026) at roughly 15-30 RPM depending
on model -- fine for interview-prep-scale usage, not for real production
QPS. Grok's API does NOT have a comparably reliable free tier -- their
free credits are promotional and tied to a data-sharing opt-in, so Gemini
is the better choice here.

RUN ON YOUR LAPTOP with a GEMINI_API_KEY environment variable set
(free key from Google AI Studio, no card required).
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()

RERANK_PROMPT = """Score how relevant this document chunk is to the query, on a scale of 0.0 to 1.0.
Output ONLY the number, nothing else.

Query: {query}

Document: {document}

Relevance score (0.0-1.0):"""


def rerank_with_gemini(query: str, candidates: list[dict], top_n: int = 5,
                        model: str = "gemini-2.5-flash") -> list[dict]:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (free, from Google AI Studio) before running this.")

    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(model)

    scored = []
    for c in candidates:
        prompt = RERANK_PROMPT.format(query=query, document=c["text"])
        response = llm.generate_content(prompt)
        match = re.search(r"[\d.]+", response.text.strip())
        score = float(match.group()) if match else 0.0
        scored.append((c, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _score in scored[:top_n]]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from retrieval import load_children, HybridRetriever
    from query_processing import expand_acronyms

    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY to actually run this against Gemini's free tier.")
        print("Get one at https://aistudio.google.com/apikey -- no card needed.")
    else:
        children = load_children()
        retriever = HybridRetriever(children)

        query = "how does CC handle FNOL for a BI claim?"
        expanded = expand_acronyms(query)
        candidates = [c for c, _ in retriever.hybrid_search(expanded, k=10)]

        reranked = rerank_with_gemini(query, candidates, top_n=3)
        for c in reranked:
            print(f"{c['metadata']['product']} / {c['metadata']['section']}: {c['text'][:80]}...")
