"""
Simple in-process token-bucket rate limiter, keyed by client IP.

Production would use Redis (shared state across multiple app instances)
or an API-gateway-level limiter (AWS API Gateway usage plans). This
in-memory version is the right thing to actually run and understand first --
same algorithm, just not distributed. Say that distinction plainly if asked
"how would this work with 3 replicas behind a load balancer" -- the honest
answer is "the bucket needs to move to Redis so all replicas share state,
this version doesn't."
"""

import time
import threading


class TokenBucket:
    def __init__(self, rate_per_minute: int, burst: int | None = None):
        self.rate_per_second = rate_per_minute / 60.0
        self.capacity = burst or rate_per_minute
        self.tokens = float(self.capacity)
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_second)
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class RateLimiter:
    """One bucket per client key (e.g. IP address)."""

    def __init__(self, rate_per_minute: int = 30, burst: int | None = None):
        self.rate_per_minute = rate_per_minute
        self.burst = burst
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, client_key: str) -> bool:
        with self._lock:
            if client_key not in self._buckets:
                self._buckets[client_key] = TokenBucket(self.rate_per_minute, self.burst)
            bucket = self._buckets[client_key]
        return bucket.allow()


# Shared instance for app.main to import
rate_limiter = RateLimiter(rate_per_minute=30, burst=10)


if __name__ == "__main__":
    limiter = RateLimiter(rate_per_minute=60, burst=5)  # 1/sec sustained, burst of 5
    allowed = [limiter.allow("client_1") for _ in range(8)]
    print(f"8 rapid requests, burst=5: {allowed}")
    print(f"Expected: first 5 True (burst), rest False until tokens refill")
