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

import json
import os
import hashlib
from qdrant_client import QdrantClient, models

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
COLLECTION = "guidewire_docs"
SPARSE_VOCAB_SIZE = 4096  # feature-hashing bucket count


def load_children():
    children = []
    with open(os.path.join(DATA_DIR, "chunks.jsonl")) as f:
        for line in f:
            c = json.loads(line)
            if not c["is_parent"]:
                children.append(c)
    return children


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


def build_collection(client: QdrantClient, children: list[dict], dense_vectors=None):
    """dense_vectors: optional numpy array aligned with `children`, produced
    by embed_pipeline.py on your laptop. If None (sandbox run), random
    vectors stand in just to prove the upsert/query mechanics work."""
    import numpy as np

    dim = 384  # all-MiniLM-L6-v2 output size
    if dense_vectors is None:
        rng = np.random.default_rng(42)
        dense_vectors = rng.normal(size=(len(children), dim))

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    points = []
    for i, (chunk, dvec) in enumerate(zip(children, dense_vectors)):
        points.append(models.PointStruct(
            id=i,
            vector={
                "dense": dvec.tolist(),
                "sparse": sparse_vector(chunk["text"]),
            },
            payload=chunk,
        ))
    client.upsert(collection_name=COLLECTION, points=points)
    return dense_vectors  # so query-time embedding uses the same random fallback path


def hybrid_search(client: QdrantClient, query: str, query_dense_vec, k: int = 5):
    """One query, native RRF fusion server-side -- this replaces the
    two-search-plus-manual-RRF dance in retrieval.py's HybridRetriever."""
    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=query_dense_vec.tolist(), using="dense", limit=20),
            models.Prefetch(query=sparse_vector(query), using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
    )
    return [(point.payload, point.score) for point in results.points]


if __name__ == "__main__":
    import numpy as np

    children = load_children()
    client = get_client()  # in-memory, sandbox-safe
    dense_vectors = build_collection(client, children)  # random placeholder vectors

    test_query = "deductible when a policy renews mid-claim"
    # In the sandbox we don't have a real embedding model, so we reuse one of
    # the random placeholder vectors just to prove the query mechanics run
    # end-to-end. On your laptop, replace this line with:
    #   query_vec = embed_model.encode([test_query])[0]
    query_vec = np.random.default_rng(7).normal(size=384)

    print(f"QUERY: {test_query}")
    results = hybrid_search(client, test_query, query_vec, k=3)
    for payload, score in results:
        print(f"  [{score:.4f}] {payload['metadata']['product']} / "
              f"{payload['metadata']['section']}: {payload['text'][:80]}...")
