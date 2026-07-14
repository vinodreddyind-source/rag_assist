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
from retry import retry_with_backoff, is_retryable_gemini_error

load_dotenv()

RERANK_PROMPT = """Score how relevant each numbered document is to the query, on a scale of 0.0 to 1.0.
Return ONLY a JSON array of scores in the same order as the documents, e.g. [0.8, 0.2, 0.5]
No other text, no explanation, just the JSON array.

Query: {query}

Documents:
{documents}

Scores (JSON array):"""


def rerank_with_gemini(query: str, candidates: list[dict], top_n: int = 5,
                        model: str | None = None) -> list[dict]:
    """Listwise reranking: ONE API call scores all candidates together,
    instead of one call per candidate. This isn't just a rate-limit
    workaround -- fewer, larger calls is the standard efficiency pattern
    for LLM-based reranking in production, and it's also a fairer
    comparison across candidates since the model sees them side by side
    rather than scoring each in isolation."""
    import json
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (free, from Google AI Studio) before running this.")

    # See generate.py's _generate_gemini for the full story: "latest" and
    # "2.5-flash" both turned out to be worse choices than this, for
    # different reasons (retired-for-new-users vs. surprisingly low quota).
    # Check https://aistudio.google.com/ for your project's live numbers
    # rather than trusting any hardcoded default, including this one.
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

    client = genai.Client(api_key=api_key)

    documents_block = "\n".join(f"{i+1}. {c['text']}" for i, c in enumerate(candidates))
    prompt = RERANK_PROMPT.format(query=query, documents=documents_block)

    response = retry_with_backoff(
        lambda: client.models.generate_content(model=model, contents=prompt),
        retryable_check=is_retryable_gemini_error,
    )

    scores = _parse_score_array(response.text, expected_len=len(candidates))
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _score in scored[:top_n]]


def _parse_score_array(text: str, expected_len: int) -> list[float]:
    """Parses the model's JSON array response. Falls back to all-zero
    scores (preserves original order) if parsing fails, rather than
    crashing the whole request -- a malformed rerank response shouldn't
    take down generation, which is the actually-important step."""
    import json
    import re
    match = re.search(r"\[[\d.,\s]+\]", text)
    if not match:
        print(f"WARNING: could not parse rerank scores from response: {text[:200]!r}")
        return [0.0] * expected_len
    try:
        scores = json.loads(match.group())
        if len(scores) != expected_len:
            print(f"WARNING: got {len(scores)} scores for {expected_len} candidates -- padding/truncating")
            scores = (scores + [0.0] * expected_len)[:expected_len]
        return [float(s) for s in scores]
    except (json.JSONDecodeError, ValueError):
        print(f"WARNING: could not parse rerank scores from response: {text[:200]!r}")
        return [0.0] * expected_len


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
