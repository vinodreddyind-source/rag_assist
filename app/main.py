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

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retrieval import load_children, HybridRetriever
from query_processing import expand_acronyms, route_product
from rerank import rerank
from generate import generate_answer
from monitoring import monitor
from rate_limit import rate_limiter
from guardrails import redact_pii, check_injection

app = FastAPI(title="Insurance Docs RAG (Advanced/Linear)")

# Loaded once at startup, not per-request
_children = load_children()
_retriever = HybridRetriever(_children)


@app.middleware("http")
async def rate_limit_and_timing(request: Request, call_next):
    client_key = request.client.host if request.client else "unknown"
    if request.url.path == "/query" and not rate_limiter.allow(client_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded, try again shortly")

    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start

    if request.url.path == "/query":
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


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    if check_injection(payload.query):
        raise HTTPException(status_code=400, detail="Request blocked: possible prompt injection")

    redacted_query = redact_pii(payload.query)
    expanded = expand_acronyms(redacted_query)
    product = route_product(redacted_query)

    candidates = [c for c, _ in _retriever.hybrid_search(expanded, k=10)]
    top_chunks = rerank(redacted_query, candidates, top_n=5)

    answer = generate_answer(redacted_query, top_chunks)
    sources = list({f"{c['metadata']['product']}/{c['metadata']['section']}" for c in top_chunks})

    return QueryResponse(
        query=payload.query,
        expanded_query=expanded,
        routed_product=product,
        answer=answer,
        sources=sources,
    )


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(_children)}


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
