"""
RUN ON YOUR LAPTOP -- needs an OPENROUTER_API_KEY (free, no card, from
https://openrouter.ai/keys).

OpenRouter is an OpenAI-compatible gateway in front of 300+ models,
including ~25 genuinely free ones (IDs ending in :free). Two things make
it worth having as a third backend alongside Ollama and direct Gemini:

1. One API shape for many providers -- switching models is a string
   change, not a new SDK.
2. Built-in provider fallback: you can pass a list of backup models, and
   OpenRouter retries against the next one if the first is rate-limited or
   down. This is the platform-level version of the retry.py module in
   this same folder -- that module retries the SAME model after a delay;
   OpenRouter's fallback list retries a DIFFERENT model immediately. Good
   to know the difference if asked "how would you make this more
   resilient."

Free tier limits (verified mid-2026): 20 requests/minute, 50/day (rises to
1000/day permanently after ever purchasing $10 in credits -- never
required). Free-model roster rotates; check https://openrouter.ai/models
filtered to $0 pricing before assuming a specific model ID still works.
"""

import os
import requests
from dotenv import load_dotenv
from retry import retry_with_backoff, is_retryable_openrouter_error

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Rotates -- verify at https://openrouter.ai/models before relying on this
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
FALLBACK_MODELS = ["google/gemma-3-12b-it:free", "qwen/qwen-2.5-7b-instruct:free"]


def _call_openrouter(prompt: str, model: str = DEFAULT_MODEL,
                      fallback_models: list[str] | None = None) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY -- see .env.example")

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if fallback_models:
        body["models"] = [model] + fallback_models  # OpenRouter's own fallback list

    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    if response.status_code == 429:
        # Surface the actual reason instead of a bare HTTPError -- OpenRouter's
        # 429 body/headers distinguish "wait a few seconds" (per-minute cap)
        # from "come back tomorrow" (daily cap), and short retries only make
        # sense for the former. Without this, a daily-cap 429 looks identical
        # to a transient one and just wastes retry attempts against a limit
        # that won't clear for hours.
        reset_header = response.headers.get("X-RateLimit-Reset", "unknown")
        try:
            body_detail = response.json().get("error", {}).get("message", response.text[:200])
        except Exception:
            body_detail = response.text[:200]
        raise requests.exceptions.HTTPError(
            f"429 rate limited. Reset: {reset_header}. Detail: {body_detail}",
            response=response,
        )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_openrouter(prompt: str) -> str:
    return _call_openrouter(prompt, fallback_models=FALLBACK_MODELS)


def rerank_prompt(query: str, documents_block: str) -> str:
    return f"""Score how relevant each numbered document is to the query, on a scale of 0.0 to 1.0.
Return ONLY a JSON array of scores in the same order as the documents, e.g. [0.8, 0.2, 0.5]
No other text, no explanation, just the JSON array.

Query: {query}

Documents:
{documents_block}

Scores (JSON array):"""


def rerank_with_openrouter(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """Listwise: ONE call scores all candidates, not one call per candidate.
    Matters even more here than for Gemini -- OpenRouter's free tier is
    50 requests/DAY, not per-minute, so a pointwise loop over 10 candidates
    could burn a fifth of the whole day's budget on a single query."""
    from rerank_gemini import _parse_score_array

    documents_block = "\n".join(f"{i+1}. {c['text']}" for i, c in enumerate(candidates))
    prompt = rerank_prompt(query, documents_block)

    response_text = retry_with_backoff(
        lambda: _call_openrouter(prompt, fallback_models=FALLBACK_MODELS),
        retryable_check=is_retryable_openrouter_error,
    )

    scores = _parse_score_array(response_text, expected_len=len(candidates))
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _score in scored[:top_n]]


if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY to test this -- get one free at https://openrouter.ai/keys")
    else:
        answer = generate_openrouter("Say 'OpenRouter is working' and nothing else.")
        print(f"Response: {answer}")
