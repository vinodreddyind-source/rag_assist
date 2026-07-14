"""
Production RAG systems get judged on these numbers as much as on RAGAS
scores. This module is deliberately dependency-light (no Prometheus/Grafana
stack) so you can run and understand it directly -- the interview answer is
"I instrumented p95 latency, TTFT, RPM and TPM and would wire this into
CloudWatch/Prometheus in production," and this is the thing that actually
computes those numbers so that sentence isn't just a buzzword you memorized.

Terms, if you want the one-line definition ready:
- p95 latency: 95% of requests finished faster than this value. More
  useful than average latency because average hides a slow tail that
  users actually feel.
- TTFT (time to first token): for streaming responses, how long before the
  FIRST token appears -- matters for perceived responsiveness even if
  total generation time is the same.
- RPM / TPM: requests-per-minute / tokens-per-minute -- what you're
  actually rate-limited against by most LLM providers (see Gemini's free
  tier: 15 RPM, 1M TPM for Flash as of mid-2026).
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RequestRecord:
    timestamp: float
    total_latency_s: float
    ttft_s: float | None
    prompt_tokens: int
    completion_tokens: int


class Monitor:
    """Thread-safe sliding-window monitor. Keeps the last `window_s` seconds
    of requests and computes percentiles / rates on demand."""

    def __init__(self, window_s: int = 300):
        self.window_s = window_s
        self._records: deque[RequestRecord] = deque()
        self._lock = threading.Lock()

    def record(self, total_latency_s: float, ttft_s: float | None = None,
               prompt_tokens: int = 0, completion_tokens: int = 0):
        now = time.time()
        with self._lock:
            self._records.append(RequestRecord(now, total_latency_s, ttft_s,
                                                 prompt_tokens, completion_tokens))
            self._evict_old(now)

    def _evict_old(self, now: float):
        cutoff = now - self.window_s
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()

    def _percentile(self, values: list[float], pct: float) -> float | None:
        if not values:
            return None
        values = sorted(values)
        idx = int(len(values) * pct)
        idx = min(idx, len(values) - 1)
        return values[idx]

    def snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            self._evict_old(now)
            records = list(self._records)

        if not records:
            return {"window_s": self.window_s, "request_count": 0}

        latencies = [r.total_latency_s for r in records]
        ttfts = [r.ttft_s for r in records if r.ttft_s is not None]
        total_prompt_tokens = sum(r.prompt_tokens for r in records)
        total_completion_tokens = sum(r.completion_tokens for r in records)
        minutes = self.window_s / 60

        return {
            "window_s": self.window_s,
            "request_count": len(records),
            "rpm": round(len(records) / minutes, 1),
            "tpm": round((total_prompt_tokens + total_completion_tokens) / minutes, 1),
            "latency_p50_s": self._percentile(latencies, 0.50),
            "latency_p95_s": self._percentile(latencies, 0.95),
            "latency_p99_s": self._percentile(latencies, 0.99),
            "ttft_p50_s": self._percentile(ttfts, 0.50) if ttfts else None,
            "ttft_p95_s": self._percentile(ttfts, 0.95) if ttfts else None,
        }


# Module-level singleton so app.main can import and use one shared monitor
monitor = Monitor(window_s=300)


if __name__ == "__main__":
    import random
    # Simulate 50 requests with realistic-ish jitter to prove percentiles work
    for _ in range(50):
        latency = random.gauss(1.2, 0.4)
        latency = max(latency, 0.1)
        ttft = latency * random.uniform(0.1, 0.3)
        monitor.record(latency, ttft, prompt_tokens=random.randint(400, 900),
                        completion_tokens=random.randint(80, 250))
    print(monitor.snapshot())
