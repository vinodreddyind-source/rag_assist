"""
Query rewriter: called when the retrieval grader scores below threshold.
Rewrites the query to be more specific/retrievable -- expanding vague
phrasing, adding likely-missing keywords -- then the graph loops back to
retrieval with the rewritten query instead of the original.

This node is genuinely new versus Phase 1's acronym expansion, which is a
fixed, deterministic transform applied once. This is adaptive: it only
fires when retrieval actually scored poorly, and it can see WHY (via the
low relevance score) even though it doesn't get per-chunk detail -- a
real, deliberate simplification versus passing the full grading detail in.
"""

import os
from dotenv import load_dotenv

load_dotenv()

REWRITE_BACKEND = os.environ.get("GEN_BACKEND", "gemini")

REWRITE_PROMPT = """The following search query didn't retrieve good results from an insurance
documentation system (covering ClaimCenter, BillingCenter, PolicyCenter).

Rewrite it to be more specific and likely to match relevant documentation.
Consider: expanding vague terms, adding likely-missing keywords, rephrasing
to match how technical documentation is usually written.

Output ONLY the rewritten query, nothing else.

Original query: {query}

Rewritten query:"""


def rewrite_query(query: str) -> str:
    prompt = REWRITE_PROMPT.format(query=query)
    result = _call_rewrite_backend(prompt)
    return result.strip().strip('"')  # models sometimes wrap output in quotes


def _call_rewrite_backend(prompt: str) -> str:
    from retry import retry_with_backoff, is_retryable_gemini_error, is_retryable_openrouter_error

    if REWRITE_BACKEND == "openrouter":
        from llm_openrouter import _call_openrouter, FALLBACK_MODELS
        return retry_with_backoff(
            lambda: _call_openrouter(prompt, fallback_models=FALLBACK_MODELS),
            retryable_check=is_retryable_openrouter_error,
        )
    if REWRITE_BACKEND == "ollama":
        import requests
        from generate import OLLAMA_URL, OLLAMA_MODEL
        response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
        response.raise_for_status()
        return response.json()["response"]

    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY -- see .env.example")
    client = genai.Client(api_key=api_key)
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    response = retry_with_backoff(
        lambda: client.models.generate_content(model=model, contents=prompt),
        retryable_check=is_retryable_gemini_error,
    )
    return response.text


if __name__ == "__main__":
    prompt = REWRITE_PROMPT.format(query="what about the discount thing after a claim")
    print("Prompt builds correctly:")
    print(prompt)
    print("\n(Real rewriting needs a live API call -- test on your laptop.)")
