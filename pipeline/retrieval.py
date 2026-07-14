"""
Hybrid retrieval: BM25 (keyword) + vector (semantic), fused with
Reciprocal Rank Fusion (RRF).

RRF is implemented BY HAND here (not via LangChain's EnsembleRetriever) so
you can actually explain the mechanics in an interview: for each ranked
list, a document's score is 1 / (k + rank). Scores from multiple lists are
summed, then re-sorted. k=60 is the standard default from the original RRF
paper -- it dampens the influence of any single very-high rank.

The vector half needs a real embedding model (sentence-transformers,
downloaded from Hugging Face) which this sandbox can't reach -- run this
file's __main__ block on your own machine after `pip install -r
requirements.txt`. The BM25 half needs nothing external and is fully
testable right here.
"""

import json
import os
from rank_bm25 import BM25Okapi

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_children():
    """Only children get searched -- parents are looked up afterward for
    generation context. Mirrors ParentDocumentRetriever's search-small,
    return-big pattern."""
    children = []
    with open(os.path.join(DATA_DIR, "chunks.jsonl")) as f:
        for line in f:
            c = json.loads(line)
            if not c["is_parent"]:
                children.append(c)
    return children


def tokenize(text: str) -> list[str]:
    return text.lower().replace(",", " ").replace(".", " ").split()


class HybridRetriever:
    def __init__(self, children: list[dict], embed_fn=None, vectors=None):
        """
        embed_fn: callable(str) -> vector, used to embed the query at search time
        vectors: precomputed list of chunk vectors, same order as `children`,
                 produced offline by embed_pipeline.py (run locally)
        """
        self.children = children
        self.corpus_tokens = [tokenize(c["text"]) for c in children]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.embed_fn = embed_fn
        self.vectors = vectors

    def bm25_search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """Returns [(chunk_index, bm25_score), ...] sorted best-first."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def vector_search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """Cosine similarity against precomputed chunk vectors.
        Requires embed_fn + vectors to be set (i.e. run locally, not in this
        sandbox)."""
        if self.embed_fn is None or self.vectors is None:
            raise RuntimeError(
                "Vector search needs a local embedding model. Run this on "
                "your machine after generating vectors with embed_pipeline.py"
            )
        import numpy as np
        q_vec = self.embed_fn(query)
        sims = np.dot(self.vectors, q_vec) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q_vec) + 1e-8
        )
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    @staticmethod
    def rrf_fuse(*ranked_lists: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion. Each ranked_list is [(chunk_index, score), ...]
        already sorted best-first. Returns fused [(chunk_index, rrf_score), ...]
        sorted best-first. Ignores the original scores entirely -- RRF only
        cares about RANK POSITION, which is exactly what makes it robust to
        combining two differently-scaled scoring systems (BM25 scores and
        cosine similarities live on completely different numeric ranges)."""
        fused_scores: dict[int, float] = {}
        for ranked_list in ranked_lists:
            for rank, (chunk_idx, _score) in enumerate(ranked_list):
                fused_scores[chunk_idx] = fused_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def hybrid_search(self, query: str, k: int = 5, bm25_k: int = 20, vector_k: int = 20):
        bm25_ranked = self.bm25_search(query, bm25_k)
        try:
            vector_ranked = self.vector_search(query, vector_k)
            fused = self.rrf_fuse(bm25_ranked, vector_ranked)
        except RuntimeError:
            # Sandbox fallback: BM25-only so the fusion function itself is
            # still exercised end-to-end, just with one list.
            fused = self.rrf_fuse(bm25_ranked)
        top = fused[:k]
        return [(self.children[idx], score) for idx, score in top]


if __name__ == "__main__":
    children = load_children()
    retriever = HybridRetriever(children)

    test_queries = [
        "how does CC handle FNOL for a BI claim?",
        "what happens to NCD after an at-fault claim?",
        "deductible when a policy renews mid-claim",
    ]

    for q in test_queries:
        print(f"\nQUERY: {q}")
        results = retriever.hybrid_search(q, k=3)
        for chunk, score in results:
            print(f"  [{score:.4f}] {chunk['metadata']['product']} / "
                  f"{chunk['metadata']['section']}: {chunk['text'][:80]}...")
