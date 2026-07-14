"""
RUN ON YOUR LAPTOP (needs a real Redis instance: `docker run -p 6379:6379 redis`)

Semantic cache: embed the incoming query, compare against previously-answered
queries stored in Redis, return the cached answer if similarity clears the
threshold -- skips the entire retrieval+rerank+generation pipeline on a
cache hit. This is the Section 6/16.3a pattern from your Guidewire doc,
actually implemented instead of just described.

Design notes worth stating in an interview:
- Cache on the REDACTED query (after PII removal), not raw user input --
  so two queries differing only in a redacted name still hit the same
  cache entry, and no PII sits in the cache itself.
- Cosine similarity threshold ~0.95 is deliberately conservative -- a false
  cache hit returns a WRONG answer with high confidence, which is worse
  than a cache miss (which just costs latency, not correctness).
- Cache entries need invalidating whenever the underlying docs re-index,
  or you serve stale answers after a documentation update. This module
  doesn't implement invalidation -- flag that as a known gap if asked "how
  do you handle staleness," since it's a real, honest answer.
"""

import json
import numpy as np
import redis


class SemanticCache:
    def __init__(self, embed_fn, redis_host="localhost", redis_port=6379,
                 similarity_threshold=0.95, key_prefix="semcache:"):
        self.embed_fn = embed_fn
        self.threshold = similarity_threshold
        self.key_prefix = key_prefix
        self.r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def lookup(self, query: str) -> str | None:
        """Linear scan over cached entries -- fine at hundreds of entries,
        would move to RediSearch's native vector KNN index (as in your doc's
        16.3a) at real production scale."""
        query_vec = self.embed_fn(query)
        for key in self.r.scan_iter(f"{self.key_prefix}*"):
            entry = json.loads(self.r.get(key))
            cached_vec = np.array(entry["vector"])
            sim = self._cosine(query_vec, cached_vec)
            if sim >= self.threshold:
                return entry["answer"]
        return None

    def store(self, query: str, answer: str):
        query_vec = self.embed_fn(query)
        key = f"{self.key_prefix}{hash(query)}"
        self.r.set(key, json.dumps({"vector": query_vec.tolist(), "answer": answer}))


if __name__ == "__main__":
    # Fake embed_fn so this file's logic is testable without a real model --
    # on your laptop, pass the real sentence-transformers .encode function.
    def fake_embed(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(size=384)

    try:
        cache = SemanticCache(fake_embed)
        cache.store("how does CC handle FNOL for a BI claim?", "Cached answer here.")
        # Same query -> should hit
        print("Same query hit:", cache.lookup("how does CC handle FNOL for a BI claim?"))
        # Different query -> should miss (fake embeddings are random per-string)
        print("Different query hit:", cache.lookup("what is the NCD policy?"))
    except redis.exceptions.ConnectionError:
        print("No local Redis running -- this is expected in the sandbox. "
              "Run `docker run -p 6379:6379 redis` on your laptop and re-run this file.")
