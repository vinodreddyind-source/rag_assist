"""
RUN ON YOUR LAPTOP. Two backends -- pick based on your RAM:

  GEN_BACKEND=ollama  -- fully local, needs ~2.5-6GB RAM depending on model
                          size (see MODEL below). Better fit for 16GB+ laptops.
  GEN_BACKEND=gemini   -- free API, ~0 local RAM. Recommended default for
                          8GB laptops, since it removes the single biggest
                          RAM cost in this whole pipeline.

    uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

This is Phase 1: LINEAR RAG (Boeing-style). Query -> retrieve -> rerank ->
generate. No grading, no retry loop, no LangGraph -- that comes in Phase 2.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ -- without this, .env does nothing

GEN_BACKEND = os.environ.get("GEN_BACKEND", "gemini")  # default: RAM-friendly

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"  # smaller than 8b -- if you do use Ollama on 8GB, this is the one to pull
GEMINI_MODEL = "gemini-2.5-flash"

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
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY -- see .env.example")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text


def generate_answer(query: str, chunks: list[dict]) -> str:
    context = format_context(chunks)
    prompt = GENERATION_PROMPT.format(context=context, query=query)

    if GEN_BACKEND == "gemini":
        return _generate_gemini(prompt)
    return _generate_ollama(prompt)


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
