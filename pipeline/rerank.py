"""
RUN ON YOUR LAPTOP -- downloads cross-encoder/ms-marco-MiniLM-L-6-v2 from
Hugging Face on first run. This is the open-source alternative your
Guidewire doc names for "no external API dependency" scenarios.

Cross-encoders score (query, document) PAIRS jointly through one forward
pass, which is why they're more accurate than cosine similarity for final
top-k selection -- but too slow to run over an entire corpus, which is why
they only rerank the ~20-40 candidates that hybrid search already narrowed
down, not the whole index.
"""

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """candidates: list of chunk dicts (with a 'text' field), already
    narrowed down by hybrid_search. Returns top_n reranked, best first."""
    model = CrossEncoder(MODEL_NAME)
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _score in scored[:top_n]]


if __name__ == "__main__":
    import sys, os, json
    sys.path.insert(0, os.path.dirname(__file__))
    from retrieval import load_children, HybridRetriever

    children = load_children()
    retriever = HybridRetriever(children)  # BM25 only fallback is fine for this demo

    query = "how does CC handle FNOL for a BI claim?"
    from query_processing import expand_acronyms
    expanded = expand_acronyms(query)
    print(f"Expanded query: {expanded}")

    candidates_with_scores = retriever.hybrid_search(expanded, k=10)
    candidates = [c for c, _ in candidates_with_scores]

    reranked = rerank(query, candidates, top_n=3)
    print("\nTop 3 after cross-encoder rerank:")
    for c in reranked:
        print(f"  {c['metadata']['product']} / {c['metadata']['section']}: {c['text'][:80]}...")
