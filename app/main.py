"""
RUN ON YOUR LAPTOP.
    uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

POST /query  {"query": "how does CC handle FNOL for a BI claim?"}
GET  /health
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

from dotenv import load_dotenv
load_dotenv()  # must happen before any os.environ.get() below, including RERANK_BACKEND

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retrieval_qdrant import QdrantHybridRetriever
from query_processing import expand_acronyms, route_product
from generate import generate_answer
from monitoring import monitor
from rate_limit import rate_limiter
from guardrails import redact_pii, check_injection
from demo_gate import passcode_required, check_passcode
from semantic_cache_lite import DiskSemanticCache

# Matches GEN_BACKEND's pattern: default to Gemini so an 8GB laptop never
# needs to download the local cross-encoder alongside the embedding model.
RERANK_BACKEND = os.environ.get("RERANK_BACKEND", "gemini")
RERANK_FALLBACK_TO_OLLAMA = os.environ.get("GEN_FALLBACK_TO_OLLAMA", "true").lower() == "true"

if RERANK_BACKEND == "gemini":
    from rerank_gemini import rerank_with_gemini as _primary_rerank
elif RERANK_BACKEND == "openrouter":
    from llm_openrouter import rerank_with_openrouter as _primary_rerank
elif RERANK_BACKEND == "ollama":
    from rerank_ollama import rerank_with_ollama as _primary_rerank
else:
    from rerank import rerank as _primary_rerank


def rerank(query: str, candidates: list, top_n: int = 5):
    if RERANK_BACKEND == "ollama":
        return _primary_rerank(query, candidates, top_n=top_n)  # already local
    try:
        return _primary_rerank(query, candidates, top_n=top_n)
    except Exception as e:
        if not RERANK_FALLBACK_TO_OLLAMA:
            raise
        print(f"WARNING: {RERANK_BACKEND} reranker failed ({type(e).__name__}: {e}). "
              f"Falling back to local Ollama reranker...")
        from rerank_ollama import rerank_with_ollama
        try:
            return rerank_with_ollama(query, candidates, top_n=top_n)
        except Exception:
            raise e  # surface the ORIGINAL error, same reasoning as generate.py's fallback

app = FastAPI(title="Insurance Docs RAG (Advanced/Linear)")

# Single source of truth for "which paths are real query endpoints" -- used
# by the passcode gate, rate limiter, and monitoring below. Real bug found
# fixing this: each of those three checks was independently hardcoded to
# just "/query", so /query/agentic and /query/multiagent silently bypassed
# rate limiting and monitoring entirely, and /query/multiagent also bypassed
# the passcode gate. One shared constant means adding a 4th endpoint later
# can't reintroduce the same gap by accident.
QUERY_PATHS = ("/query", "/query/agentic", "/query/multiagent")

# Built once at startup: loads real embeddings from data/chunk_vectors.npy
# (run pipeline/embed_pipeline.py first if this raises FileNotFoundError)
_retriever = QdrantHybridRetriever()

# Separate cache per endpoint, deliberately -- sharing one cache across
# /query, /query/agentic, and /query/multiagent would mean the first
# endpoint queried "wins" and the others silently serve its cached answer
# instead of actually running their own pipeline, defeating the entire
# point of comparing the three approaches on the same question.
_cache = DiskSemanticCache(embed_fn=_retriever._embed_query, cache_dir="./cache_data")
_cache_agentic = DiskSemanticCache(embed_fn=_retriever._embed_query, cache_dir="./cache_data_agentic")
_cache_multiagent = DiskSemanticCache(embed_fn=_retriever._embed_query, cache_dir="./cache_data_multiagent")


@app.middleware("http")
async def rate_limit_and_timing(request: Request, call_next):
    # Passcode gate: only active if DEMO_PASSCODE is set in .env. Checked via
    # a query param (?passcode=...) so a single shareable link works in a
    # plain browser -- no login form, no session, deliberately lightweight
    # for a "share one link during one demo" use case, not real auth.
    if passcode_required() and request.url.path in QUERY_PATHS:
        provided = request.query_params.get("passcode")
        if not check_passcode(provided):
            raise HTTPException(status_code=401, detail="Missing or incorrect passcode")

    client_key = request.client.host if request.client else "unknown"
    if request.url.path in QUERY_PATHS and not rate_limiter.allow(client_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded, try again shortly")

    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start

    if request.url.path in QUERY_PATHS:
        # Real token counts would come from the LLM response's usage field;
        # this is a rough word-count proxy since Ollama's /api/generate
        # response here isn't parsed for token counts in generate.py yet.
        monitor.record(total_latency_s=elapsed)

    return response


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    expanded_query: str
    routed_product: str | None
    answer: str
    sources: list[str]
    cache_hit: bool


class AgenticQueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    retry_count: int
    relevance_score: float
    route_history: list[str]
    cache_hit: bool


class MultiAgentQueryResponse(BaseModel):
    query: str
    answer: str
    cache_hit: bool


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    if check_injection(payload.query):
        raise HTTPException(status_code=400, detail="Request blocked: possible prompt injection")

    redacted_query = redact_pii(payload.query)

    # Cache on the REDACTED query, not raw input -- so two queries differing
    # only in a redacted name still hit the same entry, and no PII sits in
    # the cache. Skips the entire retrieve->rerank->generate pipeline (and
    # every external API call in it) on a hit.
    cached_answer = _cache.lookup(redacted_query)
    if cached_answer is not None:
        expanded = expand_acronyms(redacted_query)
        product = route_product(expanded)
        return QueryResponse(
            query=payload.query,
            expanded_query=expanded,
            routed_product=product,
            answer=cached_answer,
            sources=[],  # not re-derived on a cache hit -- the cached answer already has its sources embedded in the text
            cache_hit=True,
        )

    expanded = expand_acronyms(redacted_query)
    product = route_product(expanded)

    candidates = [c for c, _ in _retriever.hybrid_search(expanded, k=10, product_filter=product)]
    top_chunks = rerank(redacted_query, candidates, top_n=5)

    answer = generate_answer(redacted_query, top_chunks)
    sources = list({f"{c['metadata']['product']}/{c['metadata']['section']}" for c in top_chunks})

    _cache.store(redacted_query, answer)

    return QueryResponse(
        query=payload.query,
        expanded_query=expanded,
        routed_product=product,
        answer=answer,
        sources=sources,
        cache_hit=False,
    )


@app.post("/query/agentic", response_model=AgenticQueryResponse)
async def agentic_query_endpoint(payload: QueryRequest):
    """The LangGraph version -- same underlying components as /query
    (guardrails, retrieval, rerank, generation), but with a self-correcting
    retrieve-grade-rewrite loop instead of a single linear pass. Directly
    comparable to /query on the same corpus -- run both on the same tricky
    query and compare route_history/retry_count against the linear
    endpoint's single-pass result."""
    if check_injection(payload.query):
        raise HTTPException(status_code=400, detail="Request blocked: possible prompt injection")
    redacted_query = redact_pii(payload.query)

    # A cache hit here skips the ENTIRE graph -- potentially several LLM
    # calls (grade, rewrite, generate, grade_answer, possibly looped) --
    # not just one generation call like the linear endpoint. The most
    # valuable place in this whole app for caching to actually pay off.
    cached_answer = _cache_agentic.lookup(redacted_query)
    if cached_answer is not None:
        return AgenticQueryResponse(
            query=payload.query, answer=cached_answer, sources=[],
            retry_count=0, relevance_score=0.0, route_history=["cache_hit"],
            cache_hit=True,
        )

    from agentic_graph import run_agentic_query

    try:
        result = run_agentic_query(payload.query)  # analyze_node redacts internally
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _cache_agentic.store(redacted_query, result["generation"])

    return AgenticQueryResponse(
        query=payload.query,
        answer=result["generation"],
        sources=result["sources"],
        retry_count=result["retry_count"],
        relevance_score=result["relevance_score"],
        route_history=result["route_history"],
        cache_hit=False,
    )


@app.post("/query/multiagent", response_model=MultiAgentQueryResponse)
async def multiagent_query_endpoint(payload: QueryRequest):
    """Phase 3 -- genuinely different from /query/agentic, not a renamed
    version of it. Three specialist agents (one per product), coordinated
    by a manager that decides delegation -- multiple agent identities,
    not one self-correcting pipeline. Directly comparable to the other two
    endpoints on the same corpus."""
    if check_injection(payload.query):
        raise HTTPException(status_code=400, detail="Request blocked: possible prompt injection")
    redacted_query = redact_pii(payload.query)

    # Likely the most expensive endpoint per call (manager + up to 3
    # specialists, each a separate LLM interaction) -- a cache hit here
    # saves the most, proportionally, of any endpoint in this app.
    cached_answer = _cache_multiagent.lookup(redacted_query)
    if cached_answer is not None:
        return MultiAgentQueryResponse(query=payload.query, answer=cached_answer, cache_hit=True)

    from multiagent_crew import run_multiagent_query

    try:
        result = run_multiagent_query(payload.query)  # run_multiagent_query redacts internally
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _cache_multiagent.store(redacted_query, result["answer"])

    return MultiAgentQueryResponse(query=payload.query, answer=result["answer"], cache_hit=False)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(_retriever.children)}


@app.get("/metrics")
def metrics():
    """p95/p99 latency, RPM, TPM over a rolling 5-minute window.
    In production this would feed CloudWatch/Prometheus instead of being
    polled directly, but the numbers underneath are the same."""
    return monitor.snapshot()


# Mounted LAST and deliberately: Starlette matches routes in registration
# order, and a Mount at "/" matches every path as a prefix. Registering it
# before /health or /metrics would have swallowed those requests into a 404
# from the static file lookup instead of ever reaching the real endpoints --
# a genuine bug I hit while wiring this up, worth mentioning if asked about
# real debugging you've done with FastAPI routing.
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")
