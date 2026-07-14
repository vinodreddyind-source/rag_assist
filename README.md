# Insurance Docs RAG — Phase 1: Linear (Advanced, non-agentic)

Hands-on build mirroring your Boeing BCS project's shape (linear RAG, no
retry loop) on a Guidewire-style insurance domain, so it reinforces the
same interview story instead of competing with it.

## Vector DB decision: Qdrant, not pgvector

pgvector is a vector-similarity extension only — it has no built-in
keyword/BM25 search. True hybrid search on Postgres means pairing pgvector
with Postgres's separate native full-text search and fusing both result
sets yourself (which `retrieval.py`'s hand-rolled RRF already does — that
part transfers directly either way). Qdrant does dense+sparse+RRF fusion
as one native query and runs as a single Docker container, no extension to
compile. **Interview framing:** your Guidewire documented story still says
pgvector (that's the honest production account) — this hands-on build uses
Qdrant because it's dramatically easier to actually run and learn from
locally. Keep the two straight: "in production I used pgvector + Postgres
full-text with manual RRF; I also built and compared against Qdrant's
native hybrid fusion" is a stronger, truer answer than picking one and
pretending the other doesn't exist.

## What's already built AND verified in this sandbox (no external models/servers needed)

| Component | File | Verified how |
|---|---|---|
| Synthetic corpus + golden set (now with expected_answer) | `generate_synthetic_docs.py` | Ran — 12 docs, 8 QA pairs |
| Parent/child chunking | `chunking.py` | Ran — 28 chunks (14/14) |
| BM25 + hand-rolled RRF | `retrieval.py` | Ran on 3 real queries |
| Acronym expansion + routing | `query_processing.py` | Ran — **found a real bug**: "at-fault claim" false-matches "claim" and misroutes an NCD question to ClaimCenter. Left in deliberately — good interview material on why production uses an LLM classifier instead of a keyword dict. |
| Qdrant hybrid (dense+sparse+native RRF) | `retrieval_qdrant.py` | Ran in-memory, real query mechanics confirmed (placeholder random dense vectors since no HF download here — swap in real embeddings on your laptop) |
| Redis semantic cache | `semantic_cache.py` | **Ran against a real local Redis instance** — cache hit/miss logic confirmed working |
| Rate limiting (token bucket) | `rate_limit.py` | Ran — burst-then-throttle behavior confirmed exactly as designed |
| Monitoring (p95/p99 latency, TTFT, RPM, TPM) | `monitoring.py` | Ran — percentile math confirmed against simulated traffic |
| LCEL Runnable composition | `lcel_demo.py` | Ran — pipe-operator mechanics confirmed (real chain needs Ollama, see file) |
| FastAPI app (wired with monitoring + rate limiting) | `app/main.py` | Compiles clean — **and caught a real routing bug**: mounting the static frontend at `/` before `/health`/`/metrics` would have shadowed both endpoints with 404s. Fixed by mounting last. |
| Minimal web frontend | `app/static/index.html` | Single HTML file, no build step, calls `/query` |

## What needs YOUR laptop (Hugging Face / Ollama / Gemini API access this sandbox can't reach)

```bash
cd insurance_rag
pip install -r requirements.txt

# 1. Start Qdrant + Redis
docker compose up -d

# 2. Embed the chunks (downloads all-MiniLM-L6-v2 from Hugging Face, ~90MB)
python pipeline/embed_pipeline.py

# 3. Rebuild the Qdrant collection with REAL vectors instead of the sandbox's
#    random placeholders — swap embed_pipeline's vectors into build_collection()

# 4. Install Ollama (https://ollama.com), pull a model
ollama pull llama3.1:8b        # or llama3.2:3b if CPU-only and 8b is slow

# 5. Reranking — two options, pick one:
python pipeline/rerank.py                 # local cross-encoder, downloads ~90MB from HF
# OR, to avoid any download:
export GEMINI_API_KEY=your_free_key_here  # https://aistudio.google.com/apikey, no card
python pipeline/rerank_gemini.py

# 6. RAGAS CI gate (now includes context_precision/context_recall, not just
#    faithfulness/answer_relevancy)
python pipeline/ragas_eval.py

# 7. Serve it (monitoring + rate limiting active)
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
# open http://localhost:8080 in a browser for the UI, or:
curl -X POST localhost:8080/query -H "Content-Type: application/json" \
  -d '{"query": "how does CC handle FNOL for a BI claim?"}'
curl localhost:8080/metrics    # p95 latency, RPM, TPM

# 8. Containerize
docker build -t insurance-rag .
```

## Reranker choice: Cohere vs local cross-encoder vs Gemini free tier

Verified (mid-2026): **Gemini's free tier is real and usable** — Flash/
Flash-Lite, no card, 1,500 requests/day, 1M tokens/minute (Pro moved to
paid-only in April 2026). **Grok/xAI does NOT have a comparably reliable
free tier** — their docs don't guarantee one; the "free credits" people
cite are promotional and tied to opting into a data-sharing program where
xAI can train on your traffic. Use Gemini if you want to avoid a model
download.

Be precise about the technique difference if asked: a cross-encoder
(`rerank.py`) is a small model *trained specifically* to score (query, doc)
pairs — fast, cheap, purpose-built. `rerank_gemini.py` is a general LLM
*prompted* to act as a relevance judge — works, genuinely used in
production when a dedicated rerank API isn't justified, but higher latency
and less proven than a trained cross-encoder or a dedicated rerank API
(Cohere/Voyage/Jina). Don't present it as "the same thing but free."

## Production-topic coverage — honest scorecard

| Topic | Status |
|---|---|
| Guardrails (PII/injection) | Documented in your Guidewire doc, not yet coded here — next batch |
| RAGAS: faithfulness, answer relevancy, context precision, context recall | All four now wired in `ragas_eval.py` |
| Redis semantic cache | Built and verified |
| p95 latency, TTFT, RPM, TPM | Built and verified |
| Pydantic | Used throughout (`QueryRequest`/`QueryResponse`, RAG state) |
| LCEL | Demonstrated with reasoning on why we don't use it for the Phase 2 loop |
| Rate limiting | Built and verified (in-memory token bucket — note this doesn't share state across replicas; Redis-backed would be the production fix) |
| Web application | Minimal frontend built — this is intentionally thin, the point was proving the API layer, not a polished UI |
| Fallback | Real version needs Phase 2's grader to know *when* to fall back — coming with the agentic loop |
| Model routing/tiering | Coming properly in Phase 2 (nano/mini/full per node) — this build's reranker choice (local vs Gemini) is a preview of the same tradeoff |

## Next: Phase 2 — Agentic RAG (LangGraph)
Query analyzer node, retrieval grader, rewrite-and-retry conditional edge,
answer/faithfulness grader, PII redaction (Presidio), prompt-injection
check — turning this exact pipeline into the Guidewire-style agentic loop.

## Laptop setup — venv, Gemini key, git repo

### 1. Virtual environment (Windows PowerShell)
```powershell
cd insurance_rag
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
If PowerShell blocks the activation script: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then retry.
(Windows cmd.exe instead: `venv\Scripts\activate.bat`)

### 2. Gemini API key (free, no card)
1. Go to https://aistudio.google.com/apikey
2. Sign in with a Google account, click "Create API key"
3. Copy it, then locally: `copy .env.example .env` and paste the key into `.env`
4. Never commit `.env` — it's already in `.gitignore`

### 3. Git repo
```powershell
cd insurance_rag
git init
git add .
git commit -m "Phase 1: linear RAG pipeline, hybrid retrieval, guardrails, monitoring"
```
Then on GitHub: create a new empty repo (no README/license, so it doesn't
conflict with what you just committed), then:
```powershell
git remote add origin https://github.com/<your-username>/insurance-docs-rag.git
git branch -M main
git push -u origin main
```
Matches the pattern from your Trip Fuel repo
(`vinodreddyind-source/trip_fuel_prediciton`) — same account, new repo.

## 8GB laptop path (no Docker, minimal local RAM)

Docker Desktop's WSL2 baseline alone costs ~1.5-2GB before any container
even starts, and a local LLM costs another 2.5-6GB. On 8GB total system
RAM, running Docker + Ollama + everything else simultaneously won't
comfortably fit alongside Windows and a browser. This path avoids both:

| Piece | Default in this path | RAM |
|---|---|---|
| Vector DB | Qdrant **embedded mode** (`retrieval_qdrant.get_client()` default, `./qdrant_data` on disk, no server) | ~150MB |
| Semantic cache | `semantic_cache_lite.py` (diskcache, no Redis/Docker) | ~0 extra |
| Reranker | `rerank_gemini.py` (free API) instead of `rerank.py` (local model) | ~0 local |
| Generation | `generate.py` with `GEN_BACKEND=gemini` (now the default) instead of Ollama | ~0 local |
| Embeddings | Still local (`all-MiniLM-L6-v2`) -- this is the one worth keeping local for genuine hands-on value | ~800MB-1GB |

```powershell
# .env
GEMINI_API_KEY=your_free_key_here
GEN_BACKEND=gemini
```

Total actively used: roughly **3-4GB**, comfortable on 8GB with an IDE and
browser also open. Docker/Redis/Qdrant-server/Ollama are all still in the
codebase and fully documented -- switch to them any time you're on
better hardware, or specifically to demo "I know both the self-hosted and
the API-based tradeoffs," which is a genuinely good thing to say out loud
in an interview.

**Honest interview framing for this:** "For local development on my own
8GB laptop, I used Qdrant's embedded mode and Gemini's API for generation
and reranking to fit the hardware. In production at Guidewire-scale, the
answer is different -- RDS/pgvector or a real Qdrant cluster, and either
Bedrock or a dedicated GPU-backed self-hosted model, because you need
shared state across replicas and consistent latency under real load,
neither of which a single laptop's disk-backed cache or embedded vector
store can give you." That's a stronger answer than either extreme alone.

## Third backend: OpenRouter (free, with automatic provider fallback)

Added as a resilience option after hitting a real Gemini 503 ("high demand")
mid-build. Two independent fixes now handle that:

1. **`pipeline/retry.py`** -- retries the SAME model after exponential
   backoff + jitter. Wraps every Gemini call now (both generation and
   reranking).
2. **`pipeline/llm_openrouter.py`** -- a DIFFERENT model/provider entirely,
   via OpenRouter's free tier (`meta-llama/llama-3.3-70b-instruct:free` by
   default, with its own fallback list). Set `GEN_BACKEND=openrouter` and/or
   `RERANK_BACKEND=openrouter` in `.env` to use it instead of Gemini.

Get a free key (no card) at https://openrouter.ai/keys. Verified limits
mid-2026: 20 requests/minute, 50/day (permanently rising to 1000/day after
ever purchasing $10 in credits -- optional, never required).

**Grok was deliberately not wired in** -- it's reachable through OpenRouter,
but at standard paid xAI rates, not free. Not worth a separate integration
for this project.

**Interview framing for this whole detour:** external API churn (a model
getting retired, a provider having a bad day) is a real production risk,
and retry-with-backoff plus a fallback provider are the standard mitigations
-- not a one-off fix for this specific error. Good thing to mention
unprompted if asked about production reliability.

## Real Gemini model/quota lesson (worth remembering for interviews)

Three different failures, same underlying cause -- chasing "the newest/most
convenient model name" instead of checking your actual project's live quota:

1. `gemini-2.5-flash` (first model tried) -- retired for new API users, 404.
2. `gemini-flash-latest` (the "avoid staleness" fix) -- silently resolved to
   `gemini-3.5-flash`, whose free-tier daily quota turned out to be only 20
   requests/day for this account (confirmed directly from a real 429 body),
   far stricter than older Flash models.
3. Now pinned to `gemini-3.1-flash-lite`, configurable via `GEMINI_MODEL` in
   `.env` -- but even this may not hold. **Check your live quota at
   https://aistudio.google.com/ (Usage & billing) rather than trusting any
   number in this README, including this one.**

Also fixed: `pipeline/retry.py` now distinguishes a Gemini 429 caused by the
**daily** cap (fails fast -- the error body literally contains
`GenerateRequestsPerDayPerProjectPerModel`, so there's no ambiguity) from a
per-minute rate limit (still worth a short backoff). Retrying a daily cap
for a few seconds was pure wasted time before this fix.

**Interview framing:** this is a real, current example of why production
systems pin explicit model versions with monitored quotas rather than
following "latest" aliases, and why retry logic needs to actually parse
the error body instead of treating every 429 the same.

## Automatic fallback to local DeepSeek when cloud quota is hit

`generate.py` now falls back to a local Ollama model automatically when
Gemini or OpenRouter fail with a real quota/service error (not just a
transient blip already handled by retry.py). No manual `.env` editing mid-
demo required.

**Kimi K2 is not usable locally** -- it's a ~1 trillion parameter MoE model
with no meaningful small distill available. **DeepSeek is the right choice**
-- DeepSeek-R1 publishes genuinely small distilled versions specifically for
constrained hardware:

| Model | Size | Fit for 8GB laptop |
|---|---|---|
| `deepseek-r1:1.5b` (default here) | ~1.1GB | Comfortable |
| `qwen2.5:1.5b` | ~1GB | Comfortable |
| `llama3.2:3b` | ~2GB | Fine, slightly better quality |

Setup:
```powershell
# Install Ollama from https://ollama.com, then:
ollama pull deepseek-r1:1.5b
```

That's it -- as long as Ollama is running (`ollama serve`, or it may already
run as a background service after install) and the model is pulled,
`generate.py` will automatically drop to it if Gemini/OpenRouter hit a wall.
If Ollama ISN'T running when a fallback is attempted, you'll see the
*original* cloud error in the logs, not a confusing Ollama connection
error -- that's deliberate, since the cloud error is the one that's
actually useful to see.

Set `GEN_FALLBACK_TO_OLLAMA=false` in `.env` to disable this and just fail
normally if you want to see raw cloud errors without the fallback masking
them (useful when you're specifically debugging the cloud backend itself,
like we were earlier today).

## Heads-up for Phase 2: a real version conflict is coming

RAGAS 0.2.15 (the version pinned above, chosen because it's the last one
that imports cleanly) needs `langchain-core<1.0`. But **LangGraph 1.x**
(needed for Phase 2's actual agentic loop) needs `langchain-core>=1.2`.
Those two requirements can't both be satisfied in one environment.

Not a problem today -- ragas_eval.py doesn't use LangGraph, and nothing in
Phase 1 does either. But when Phase 2 starts, expect to need either:
- A second virtual environment specifically for running `ragas_eval.py`
  (kept on the old langchain-core), separate from the one running the
  LangGraph app itself, or
- A newer ragas release that's fixed its LangChain-ecosystem imports by
  then -- worth checking `pip index versions ragas` again before assuming
  today's pin is still the right one.

This is itself a legitimate interview point: fast-moving ecosystems
(RAGAS, LangChain, LangGraph, Gemini's SDK/model names -- all four broke
at least once during this project) mean pinned, tested dependency sets
are not optional for anything meant to actually run reliably, and CI needs
to catch version drift before it reaches anyone relying on the tool.
