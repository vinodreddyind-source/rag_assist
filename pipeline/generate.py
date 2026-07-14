"""
RUN ON YOUR LAPTOP. Backends, tried in order with automatic fallback:

  GEN_BACKEND=gemini      -- free API, ~0 local RAM. Default primary.
  GEN_BACKEND=openrouter  -- free API, ~0 local RAM, own provider fallback.
  GEN_BACKEND=ollama      -- fully local, needs ~1-3GB RAM for a small model.

GEN_FALLBACK_TO_OLLAMA=true (default) means: if the primary backend fails
with a genuine quota/service wall (not a transient error already handled by
retry.py), automatically fall back to a local Ollama model instead of
failing the whole request. This is the actual fix for hitting daily quota
caps mid-demo -- no manual .env editing needed once it's set up.

Recommended small Ollama models for 8GB laptops (pull whichever you want,
set OLLAMA_MODEL to match):
  deepseek-r1:1.5b   -- ~1.1GB, DeepSeek's smallest distill, reasoning-
                        capable despite the size, good fallback choice
  qwen2.5:1.5b       -- ~1GB, fast, solid general quality
  llama3.2:3b        -- ~2GB, slightly better quality, slightly more RAM

Kimi K2 is NOT a realistic local option -- it's a ~1 trillion parameter MoE
model with no meaningful small distill, unlike DeepSeek which specifically
publishes small distilled versions for exactly this use case.

    uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

This is Phase 1: LINEAR RAG (Boeing-style). Query -> retrieve -> rerank ->
generate. No grading, no retry loop, no LangGraph -- that comes in Phase 2.
"""

import os
import requests
from dotenv import load_dotenv
from retry import retry_with_backoff, is_retryable_gemini_error, is_retryable_openrouter_error

load_dotenv()  # reads .env into os.environ -- without this, .env does nothing

GEN_BACKEND = os.environ.get("GEN_BACKEND", "gemini")  # gemini | openrouter | ollama
FALLBACK_TO_OLLAMA = os.environ.get("GEN_FALLBACK_TO_OLLAMA", "true").lower() == "true"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:1.5b")  # ~1.1GB, fits 8GB laptops easily

GENERATION_PROMPT = """You are a helpful assistant answering questions using ONLY the context below.
If the context doesn't contain enough information, say so clearly. Cite which
section each fact comes from.

Context:
{context}

Question: {query}

Answer:"""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        meta = c["metadata"]
        parts.append(f"[Source: {meta['product']} / {meta['section']}]\n{c['text']}")
    return "\n\n".join(parts)


def _generate_ollama(prompt: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    })
    response.raise_for_status()
    return response.json()["response"]


def _generate_gemini(prompt: str) -> str:
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY -- see .env.example")
    client = genai.Client(api_key=api_key)
    # Model choice history, worth knowing: gemini-2.5-flash was the FIRST
    # model tried here and was already retired for new users (404). Then
    # "gemini-flash-latest" pointed at gemini-3.5-flash, which turned out to
    # have only a 20-requests/DAY free quota for this account (confirmed
    # directly from a real 429 response) -- much stricter than 2.5-flash's
    # documented ~1,500/day, apparently because newer/flagship models get
    # tighter introductory free-tier limits than established "lite" variants.
    # gemini-3.1-flash-lite is the current-generation, cost-tier model most
    # likely to have a generous free quota. That said: free-tier quotas on
    # Gemini have changed at least twice just during this project. Don't
    # trust this comment either -- check the live number for your own
    # project at https://aistudio.google.com/ (Usage & billing) before
    # assuming any figure, including this one.
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def generate_answer(query: str, chunks: list[dict]) -> str:
    context = format_context(chunks)
    prompt = GENERATION_PROMPT.format(context=context, query=query)

    if GEN_BACKEND == "ollama":
        return _generate_ollama(prompt)  # already local -- nothing to fall back to

    if GEN_BACKEND == "openrouter":
        from llm_openrouter import generate_openrouter
        try:
            return retry_with_backoff(lambda: generate_openrouter(prompt),
                                       retryable_check=is_retryable_openrouter_error)
        except Exception as e:
            return _fallback_or_raise(e, prompt, "OpenRouter")

    # default: gemini
    try:
        return retry_with_backoff(lambda: _generate_gemini(prompt),
                                   retryable_check=is_retryable_gemini_error)
    except Exception as e:
        return _fallback_or_raise(e, prompt, "Gemini")


def _fallback_or_raise(original_error: Exception, prompt: str, backend_name: str) -> str:
    """Called when the primary cloud backend's retries are exhausted or it
    failed fast on a non-retryable error (e.g. a daily quota cap). If
    fallback is enabled, try local Ollama instead of failing the whole
    request. If Ollama itself isn't running, this raises the ORIGINAL error
    (not a confusing Ollama connection error), since that's the actually
    useful thing to see in the logs."""
    if not FALLBACK_TO_OLLAMA:
        raise original_error
    print(f"WARNING: {backend_name} failed ({type(original_error).__name__}: {original_error}). "
          f"Falling back to local Ollama ({OLLAMA_MODEL})...")
    try:
        return _generate_ollama(prompt)
    except requests.exceptions.ConnectionError:
        print(f"Ollama fallback also failed -- is it running? (ollama serve, and "
              f"`ollama pull {OLLAMA_MODEL}` if you haven't). Raising the original "
              f"{backend_name} error instead of the Ollama connection error, since "
              f"that's the more useful one to see.")
        raise original_error


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from retrieval import load_children, HybridRetriever
    from query_processing import expand_acronyms
    from rerank import rerank

    children = load_children()
    retriever = HybridRetriever(children)

    query = "how does CC handle FNOL for a BI claim?"
    expanded = expand_acronyms(query)
    candidates = [c for c, _ in retriever.hybrid_search(expanded, k=10)]
    top_chunks = rerank(query, candidates, top_n=5)

    answer = generate_answer(query, top_chunks)
    print(f"QUERY: {query}\n")
    print(f"ANSWER:\n{answer}")
