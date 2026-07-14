"""
Retry-with-backoff for transient upstream failures (503 UNAVAILABLE, 429
rate limits, connection resets). This is a real production-RAG concern any
external LLM/rerank API call needs -- free-tier Gemini in particular
returns 503 under load fairly often, and a single unretried failure
shouldn't take down the whole query.

Exponential backoff with jitter: each retry waits roughly double the
previous wait, plus a small random amount (jitter) so many clients retrying
at once don't all hammer the server in lockstep -- a real failure mode
called the "thundering herd" problem.
"""

import time
import random
import functools


def retry_with_backoff(max_attempts: int = 4, base_delay: float = 1.0,
                        max_delay: float = 20.0, retryable_exceptions: tuple = (Exception,)):
    """Decorator. Retries the wrapped function on the given exception types,
    with exponential backoff + jitter, up to max_attempts total tries."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exc = e
                    if attempt == max_attempts - 1:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay += random.uniform(0, delay * 0.25)  # jitter
                    print(f"[retry] attempt {attempt + 1}/{max_attempts} failed "
                          f"({type(e).__name__}: {e}); retrying in {delay:.1f}s")
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


if __name__ == "__main__":
    # Simulate a flaky function that fails twice, then succeeds -- proves
    # the retry/backoff mechanics work without needing a real API.
    call_count = {"n": 0}

    @retry_with_backoff(max_attempts=4, base_delay=0.1, max_delay=1.0)
    def flaky():
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError(f"simulated transient failure #{call_count['n']}")
        return "success"

    result = flaky()
    print(f"Result: {result} (took {call_count['n']} attempts)")
    assert result == "success" and call_count["n"] == 3
    print("Retry/backoff logic verified.")
