"""
Generic retry-with-exponential-backoff for any LLM API call. This is the
direct fix for the Gemini 503 "high demand" error -- that error was
transient (the request itself was correct, the server was just temporarily
overloaded), and transient errors are exactly what a retry-with-backoff
loop is for. This is standard production practice for any external API
call, not specific to Gemini.

Deliberately NOT using the `tenacity` library even though it's already a
transitive dependency of google-genai -- writing this by hand means you can
actually explain the backoff math in an interview instead of naming a
decorator you didn't write.
"""

import time
import random


def retry_with_backoff(fn, max_attempts: int = 4, base_delay: float = 1.0,
                        max_delay: float = 20.0, retryable_check=None):
    """Calls fn() and retries on failure with exponential backoff + jitter.

    base_delay doubles each attempt (1s, 2s, 4s, 8s...), capped at max_delay.
    Jitter (randomized +/-20%) avoids many clients retrying in lockstep and
    hammering an already-overloaded server at the same instant -- the
    "thundering herd" problem.

    retryable_check(exception) -> bool: lets the caller decide which errors
    are worth retrying (503/429/timeouts) vs which should fail immediately
    (e.g. a 400 for a malformed request will never succeed on retry).
    Defaults to retrying everything, which is fine for a demo but not
    something you'd want in real production code.
    """
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if retryable_check is not None and not retryable_check(e):
                raise  # non-retryable error -- fail fast, don't waste time
            if attempt == max_attempts - 1:
                break  # last attempt, fall through to raise below
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = delay * random.uniform(-0.2, 0.2)
            sleep_time = max(0.1, delay + jitter)
            print(f"Attempt {attempt + 1}/{max_attempts} failed ({type(e).__name__}: {e}). "
                  f"Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)
    raise last_exception


def is_retryable_llm_error(exception: Exception) -> bool:
    """Retry on rate limits and transient server errors. Don't retry on
    auth failures or malformed requests -- those need a code/config fix,
    not a delay."""
    error_str = str(exception).lower()
    retryable_signals = ["503", "429", "unavailable", "rate limit",
                          "timeout", "overloaded", "high demand"]
    return any(signal in error_str for signal in retryable_signals)


if __name__ == "__main__":
    # Fully testable without any network call -- simulate a flaky function
    # that fails twice with a "retryable" error, then succeeds.
    attempts = {"count": 0}

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("503 UNAVAILABLE: high demand, simulated")
        return "success"

    result = retry_with_backoff(flaky_call, max_attempts=4, base_delay=0.1,
                                 retryable_check=is_retryable_llm_error)
    print(f"Result: {result} after {attempts['count']} attempts")

    # Now prove non-retryable errors fail immediately, no wasted retries
    attempts2 = {"count": 0}

    def bad_request_call():
        attempts2["count"] += 1
        raise RuntimeError("400 Bad Request: malformed prompt, simulated")

    try:
        retry_with_backoff(bad_request_call, max_attempts=4, base_delay=0.1,
                            retryable_check=is_retryable_llm_error)
    except RuntimeError as e:
        print(f"Correctly failed fast after {attempts2['count']} attempt(s): {e}")
