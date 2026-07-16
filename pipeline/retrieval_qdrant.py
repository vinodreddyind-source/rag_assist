"""
Qdrant hybrid retrieval: dense (semantic) + sparse (keyword-ish) vectors,
fused server-side with Qdrant's native RRF -- one query call instead of the
two-query-plus-manual-fuse dance pgvector needs.

Why this replaces the earlier pgvector plan: pgvector is ONLY a vector-
similarity extension. It has no keyword/BM25 capability built in -- getting
hybrid search out of Postgres means pairing pgvector with Postgres's
separate native full-text search (tsvector/tsquery) and fusing the two
result sets yourself, which is exactly what retrieval.py's hand-rolled RRF
already did. Qdrant does the same fusion, just as one native query, and
runs as a single Docker container with no extension to compile.

Sparse vectors here use a simple feature-hashed term-frequency encoding --
NOT full BM25/SPLADE. This is deliberate: it proves the dense+sparse+fusion
plumbing works without needing to download a sparse encoder model. Swap in
Qdrant's `Qdrant/bm25` fastembed model on your laptop for the production-
quality version -- the query/upsert shape barely changes, only how the
sparse vector gets built.
"""

"""
Qdrant hybrid retrieval: dense (semantic) + sparse (keyword-ish) vectors,
fused server-side with Qdrant's native RRF -- one query call instead of the
two-query-plus-manual-fuse dance pgvector needs.

Why this replaces the earlier pgvector plan: pgvector is ONLY a vector-
similarity extension. It has no keyword/BM25 capability built in -- getting
hybrid search out of Postgres means pairing pgvector with Postgres's
separate native full-text search (tsvector/tsquery) and fusing the two
result sets yourself, which is exactly what retrieval.py's hand-rolled RRF
already did. Qdrant does the same fusion, just as one native query, and
runs embedded (no server) or as a single Docker container.

Sparse vectors here use a simple feature-hashed term-frequency encoding --
NOT full BM25/SPLADE. This is deliberate: it proves the dense+sparse+fusion
plumbing works without needing to download a separate sparse encoder
model. Swap in Qdrant's `Qdrant/bm25` fastembed model for the production-
quality version -- the query/upsert shape barely changes, only how the
sparse vector gets built.
"""

import json
import os
import hashlib
import numpy as np
from qdrant_client import QdrantClient, models

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
COLLECTION = "guidewire_docs"
SPARSE_VOCAB_SIZE = 4096  # feature-hashing bucket count
DENSE_DIM = 384  # all-MiniLM-L6-v2 output size


def load_children():
    children = []
    with open(os.path.join(DATA_DIR, "chunks.jsonl")) as f:
        for line in f:
            c = json.loads(line)
            if not c["is_parent"]:
                children.append(c)
    return children


def load_dense_vectors():
    """Loads the REAL vectors produced by embed_pipeline.py. Raises a clear
    error if you haven't run that yet, rather than silently falling back to
    randomness like the sandbox test path did."""
    vec_path = os.path.join(DATA_DIR, "chunk_vectors.npy")
    idx_path = os.path.join(DATA_DIR, "chunk_vector_index.json")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(
            "data/chunk_vectors.npy not found -- run `python pipeline/embed_pipeline.py` first."
        )
    vectors = np.load(vec_path)
    with open(idx_path) as f:
        chunk_ids = json.load(f)
    return vectors, chunk_ids


def sparse_vector(text: str) -> models.SparseVector:
    """Feature-hashed term frequency -- a stand-in for a real BM25/SPLADE
    sparse encoder. Good enough to test the Qdrant plumbing; not what you'd
    ship. Say this plainly if asked."""
    tokens = text.lower().replace(",", " ").replace(".", " ").split()
    counts: dict[int, int] = {}
    for tok in tokens:
        bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % SPARSE_VOCAB_SIZE
        counts[bucket] = counts.get(bucket, 0) + 1
    indices = list(counts.keys())
    values = [float(v) for v in counts.values()]
    return models.SparseVector(indices=indices, values=values)


def get_client(path: str | None = "./qdrant_data") -> QdrantClient:
    """Default: embedded on-disk mode -- no server process, ~150MB RAM,
    persists between runs. Right choice for an 8GB laptop.

    path=None -> pure in-memory, nothing persisted (what the sandbox uses
    to test this file, since even a disk write isn't needed just to prove
    the query mechanics work).

    On a machine with RAM to spare, you can instead run a real Qdrant
    server (`docker run -p 6333:6333 qdrant/qdrant`) and connect with
    QdrantClient(url="http://localhost:6333") -- functionally identical,
    just able to serve multiple processes/replicas at once, which the
    embedded mode can't."""
    if path is None:
        return QdrantClient(":memory:")
    return QdrantClient(path=path)


class QdrantHybridRetriever:
    """Single entry point for main.py -- builds/loads the collection from
    REAL embeddings on init, embeds queries with the same model at search
    time, so index-time and query-time vectors are guaranteed comparable."""

    def __init__(self, qdrant_path: str = "./qdrant_data", force_rebuild: bool = False):
        self.children = load_children()
        self.client = get_client(qdrant_path)

        needs_build = force_rebuild or not self._collection_ready()
        if needs_build:
            dense_vectors, chunk_ids = load_dense_vectors()
            # Sanity check: embed_pipeline's order must match load_children()'s
            # order here, or vectors get attached to the wrong chunk.
            actual_ids = [c["chunk_id"] for c in self.children]
            if chunk_ids != actual_ids:
                raise ValueError(
                    "Vector/chunk order mismatch -- re-run embed_pipeline.py "
                    "after any change to chunking.py or the raw corpus."
                )
            self._build_collection(dense_vectors)

        # Lazy-loaded so importing this module doesn't require sentence-transformers
        # to already be installed/downloaded (e.g. the sandbox never has it).
        self._embed_model = None

    def _collection_ready(self) -> bool:
        try:
            info = self.client.get_collection(COLLECTION)
            return info.points_count == len(self.children)
        except Exception:
            return False

    def _build_collection(self, dense_vectors: np.ndarray):
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        points = [
            models.PointStruct(
                id=i,
                vector={"dense": dvec.tolist(), "sparse": sparse_vector(chunk["text"])},
                payload=chunk,
            )
            for i, (chunk, dvec) in enumerate(zip(self.children, dense_vectors))
        ]
        self.client.upsert(collection_name=COLLECTION, points=points)

    def _embed_query(self, query: str) -> np.ndarray:
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed_model.encode([query])[0]

    def hybrid_search(self, query: str, k: int = 5, product_filter: str | None = None):
        """product_filter: restrict to one product's chunks (from route_product()).
        Falls back to an unfiltered search if the filtered result set is too
        thin (fewer than k results) -- a routing misfire should degrade
        gracefully to "search everything", not silently starve retrieval of
        results. This is the actual fix for the bug flagged earlier: routing
        was computed but never affected retrieval, making it purely cosmetic.
        Now it does affect retrieval, with a safety net for when it's wrong."""
        query_vec = self._embed_query(query)

        query_filter = None
        if product_filter:
            query_filter = models.Filter(
                must=[models.FieldCondition(
                    key="metadata.product",
                    match=models.MatchValue(value=product_filter),
                )]
            )

        results = self.client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                models.Prefetch(query=query_vec.tolist(), using="dense", limit=20, filter=query_filter),
                models.Prefetch(query=sparse_vector(query), using="sparse", limit=20, filter=query_filter),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=k,
        )
        hits = [(point.payload, point.score) for point in results.points]

        if product_filter and len(hits) < k:
            print(f"WARNING: filtered search for product={product_filter!r} only "
                  f"returned {len(hits)}/{k} results -- falling back to unfiltered "
                  f"search. Either routing misfired, or that product genuinely has "
                  f"few matching chunks.")
            return self.hybrid_search(query, k=k, product_filter=None)

        return hits


if __name__ == "__main__":
    # Sandbox-safe smoke test: uses random placeholder vectors of the right
    # shape (no real embed model here) just to prove the Qdrant plumbing
    # itself works. On your laptop, this file's __main__ isn't what you run --
    # use it via QdrantHybridRetriever from app/main.py instead, which uses
    # your REAL embed_pipeline.py vectors.
    children = load_children()
    client = get_client(path=None)  # in-memory for this smoke test only
    rng = np.random.default_rng(42)
    fake_vectors = rng.normal(size=(len(children), DENSE_DIM))

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    points = [
        models.PointStruct(id=i, vector={"dense": v.tolist(), "sparse": sparse_vector(c["text"])}, payload=c)
        for i, (c, v) in enumerate(zip(children, fake_vectors))
    ]
    client.upsert(collection_name=COLLECTION, points=points)

    test_query = "deductible when a policy renews mid-claim"
    query_vec = rng.normal(size=DENSE_DIM)
    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=query_vec.tolist(), using="dense", limit=20),
            models.Prefetch(query=sparse_vector(test_query), using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=3,
    )
    print(f"QUERY: {test_query}")
    for point in results.points:
        print(f"  [{point.score:.4f}] {point.payload['metadata']['product']} / "
              f"{point.payload['metadata']['section']}: {point.payload['text'][:80]}...")
