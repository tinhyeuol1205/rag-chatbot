# RAG latency benchmark and TTFT review

## What the benchmark measures

`scripts/benchmark_rag_latency.py` has two complementary modes:

- `in-process`: executes the real retriever and reports inclusive latency for
  condensation, Multi-Query, HyDE LLM/embedding, each hybrid search, Qdrant,
  RRF, reranking, parent lookup, context assembly, final stream/provider TTFT,
  total retrieval/generation TTFT and total RAG time. `generation.stream_ttft`
  includes final-call pacing; `generation.provider_ttft` starts after the LLM
  permit is granted.
- `http`: calls `POST /chat/stream` and measures the user-observed response
  headers, first status, worker-start status, first answer token and stream end.
  This includes API, Redis queue/admission, worker and SSE transport overhead.

The report intentionally excludes the question, history content, answer,
retrieved context, tokens and credentials. Raw spans contain only timings,
counts and non-sensitive stage metadata.

Inclusive spans overlap. In particular, Multi-Query and HyDE run in parallel,
and Qdrant/embedding spans are children of hybrid-search spans. Do not add all
rows to estimate total latency; use `rag.in_process_ttft` or `rag.http_ttft`.

## Commands

Profile modules and the direct RAG critical path using the active `.env`:

```bash
PYTHONPATH=src rag/bin/python scripts/benchmark_rag_latency.py \
  --mode in-process \
  --query "Chính sách nghỉ phép của công ty là gì?" \
  --runs 3 \
  --output artifacts/rag-latency-modules.json
```

Measure production-like End-to-End TTFT after API, worker and model server are
running:

```bash
RAG_BENCHMARK_API_KEY="..." \
PYTHONPATH=src rag/bin/python scripts/benchmark_rag_latency.py \
  --mode http \
  --api-url http://127.0.0.1:8080 \
  --query "Chính sách nghỉ phép của công ty là gì?" \
  --runs 5 \
  --output artifacts/rag-latency-e2e.json
```

Omit `RAG_BENCHMARK_API_KEY` when API authentication is disabled. `--mode both`
runs both measurements. Warmup loads local models without a paid LLM request;
`--warmup-runs N` adds `N` complete, paid, unreported requests.

In-process mode defaults to `--pacing shared`, so it uses the configured
admission scheduler and does not silently exceed a provider quota shared with a
worker. `--pacing isolated` is useful in an isolated development environment,
but must not target a provider account receiving production traffic.

For multi-turn behavior, pass a JSON file whose shape is:

```json
[["Câu hỏi trước", "Câu trả lời trước"]]
```

## Bottleneck review of the current pipeline

### 1. LLM quota pacing creates a deterministic TTFT floor

The current configuration reserves three provider calls for a question without
history: Multi-Query, HyDE and final generation. With 15 calls per 60 seconds,
`RateScheduler` spaces calls by about 4 seconds. Therefore the final call cannot
start earlier than roughly 8 seconds after the first transform call. A request
with history adds condensation and raises this floor to roughly 12 seconds.

Although Multi-Query and HyDE use two threads, both call the same paced provider
reservation. The scheduler consequently serializes their provider boundaries.
The threads still overlap non-provider work, but they cannot remove the 4-second
quota gap. Under load, each three-call reservation also advances the shared
schedule by about 12 seconds before the next reservation can start.

The benchmark exposes this as `admission.llm_pacing_wait` and reports the
configured lower bound separately from provider latency.

### 2. Retrieval performs up to five sequential searches

With three expansion variants, the retriever performs one direct hybrid search,
one HyDE dense search and three variant hybrid searches. Four query embeddings
and all five Qdrant requests happen sequentially after expansion/HyDE. Remote
embedding and reranker adapters also use module-level `httpx.post`, which opens
a short-lived client path rather than reusing a long-lived connection pool.

Measure `embedding.search_query`, `qdrant.hybrid_query`,
`qdrant.dense_query`, `search.hybrid_total` and their `calls_per_run` before
changing search breadth.

### 3. Reranking and final prompt prefill can be large

The reranker accepts up to 30 candidates at a maximum sequence length of 512,
then the assembler may send up to 24,000 characters of parent context to the
final LLM. Both costs grow with candidate/context size. Compare `rerank.total`
and `generation.provider_ttft`; a large provider TTFT with small pacing wait is
often prompt-prefill or provider queue time.

### 4. Inline SSE buffers the complete answer

In inline API mode, `_run_stream_job` converts `chat_stream_events(...)` to a
list before the endpoint emits its first token. Client-observed TTFT therefore
includes the entire answer generation in this mode. Redis-worker mode publishes
tokens incrementally and does not have this buffering behavior. Always use HTTP
mode to confirm the client-visible number for the deployed execution mode.

## Recommended optimization order

1. **Reduce online LLM calls.** Start with an ablation using direct hybrid
   retrieval only, then add either Multi-Query or HyDE only when a cheap query
   classifier predicts that expansion is needed. Validate recall/faithfulness
   on the evaluation set. Removing both optional calls changes the normal path
   from three provider calls to one and removes the 8-second intra-request
   pacing floor.
2. **Set quota from the real provider limit.** If 15 calls/minute is only a demo
   default rather than the account's actual rolling limit, configure the real
   value. Do not raise it beyond the provider contract. Track queue-to-worker
   time separately so quota backlog is visible.
3. **Batch retrieval work.** Embed the original and expansion queries in one
   remote `/embed` request and issue Qdrant searches concurrently or through a
   batch/query API while preserving one sparse vote per distinct query. Cap the
   number of variants using measured marginal recall.
4. **Reuse HTTP clients.** Give embedding and reranker services persistent,
   lifecycle-managed `httpx.Client` instances to reuse connections. Keep
   request timeouts and shutdown cleanup explicit.
5. **Tune rerank/context budgets by evidence.** Benchmark 10/20/30 rerank
   candidates and smaller context budgets; select the smallest values that
   retain evaluation quality. Deduplicate/trim parent chunks before final
   generation.
6. **Fix inline streaming if it is user-facing.** Relay a thread-backed iterator
   or async queue instead of materializing `list(chat_stream_events(...))`.
7. **Add latency SLO telemetry.** Persist p50/p95 for queue wait, retrieval,
   provider TTFT and E2E TTFT. Compare cold start separately from warmed steady
   state and run at both concurrency 1 and expected production concurrency.

Use quality gates for every retrieval ablation. A lower TTFT is not an
improvement if it materially reduces answer recall, faithfulness or citation
coverage.
