"""
8GB-laptop-friendly semantic cache: same lookup-by-cosine-similarity idea as
semantic_cache.py, but backed by `diskcache` (a local SQLite-based cache)
instead of a Redis server. No Docker, no separate process, ~0 extra RAM
beyond the Python process itself.

This is the right tool for LOCAL DEV on constrained hardware. It is NOT
what you'd say in an interview as your production answer -- your Guidewire
doc's Redis/ElastiCache story is still the honest production design (multi-
instance deployments need cache state shared across replicas, which a local
SQLite file can't do). Say plainly if asked: "for local development on
limited hardware I used a disk-backed cache; production uses Redis/
ElastiCache specifically because it's shared across replicas, which a
single-machine cache isn't."
"""

import numpy as np
from diskcache import Cache


class DiskSemanticCache:
    def __init__(self, embed_fn, cache_dir="./cache_data", similarity_threshold=0.95):
        self.embed_fn = embed_fn
        self.threshold = similarity_threshold
        self.cache = Cache(cache_dir)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def lookup(self, query: str) -> str | None:
        query_vec = self.embed_fn(query)
        for key in self.cache:
            entry = self.cache[key]
            sim = self._cosine(query_vec, np.array(entry["vector"]))
            if sim >= self.threshold:
                return entry["answer"]
        return None

    def store(self, query: str, answer: str):
        query_vec = self.embed_fn(query)
        self.cache[str(hash(query))] = {"vector": query_vec.tolist(), "answer": answer}


if __name__ == "__main__":
    def fake_embed(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(size=384)

    cache = DiskSemanticCache(fake_embed, cache_dir="/tmp/test_semcache")
    cache.store("how does CC handle FNOL for a BI claim?", "Cached answer here.")
    print("Same query hit:", cache.lookup("how does CC handle FNOL for a BI claim?"))
    print("Different query hit:", cache.lookup("what is the NCD policy?"))
