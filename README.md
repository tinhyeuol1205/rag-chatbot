# 🤖 RAG Chatbot — Internal Knowledge Base Assistant

> Advanced RAG chatbot for querying internal company documents (PDF, DOCX, Markdown).  
> Built as a portfolio project showcasing production-grade RAG techniques.

## ✨ RAG Techniques Implemented

| # | Technique | Description |
|---|---|---|
| 1 | **Hybrid Search (Dense + BM25 + RRF)** | Combines semantic search with keyword matching for best of both worlds |
| 2 | **Cross-Encoder Reranking** | Two-stage retrieval: fast recall → precise reranking with `bge-reranker-v2-m3` |
| 3 | **Multi-Query Expansion** | LLM generates query variants to overcome vocabulary mismatch |
| 4 | **HyDE** | Hypothetical Document Embedding — answer-to-document matching |
| 5 | **Parent-Child Retrieval** | Search on small chunks (precision), return large chunks (full context) |

**Bonus:** Lost-in-Middle reordering, source citation, RAG Triad evaluation (RAGAS).

## 🏗️ Architecture

```
Documents (PDF/DOCX/MD)
        │
        ▼
┌── Ingestion Pipeline ──┐
│ Stream → Chunk → Embed │──────► Qdrant staging generation
│ (Parent-Child strategy) │       (atomic retrieval aliases)
└─────────────────────────┘
                                         │
User Query                               │
    │                                    │
    ▼                                    ▼
┌── Retrieval Pipeline (Advanced RAG) ───────────────────────┐
│ Multi-Query Expansion → HyDE → Hybrid Search → Reranking  │
│ → Parent Resolution → Context Assembly → LLM Generation    │
└────────────────────────────────────────────────────────────┘
    │
    ▼
  Answer + Sources
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI-compatible (OpenAI / NVIDIA NIM / vLLM) **hoặc** Google Gemini |
| Embeddings | `BAAI/bge-m3` (1024d, self-host GPU in production) |
| Reranker | `BAAI/bge-reranker-v2-m3` (self-host GPU in production) |
| Vector DB | Qdrant (native BM25 + dense RRF) |
| Admission | Redis Streams + atomic 15 calls/minute reservation |
| Backend | FastAPI + SSE streaming |
| Frontend | Gradio |
| Evaluation | RAGAS (RAG Triad metrics) |
| Dependency | uv (`pyproject.toml` + `uv.lock`) |

## 🚀 Quick Start

> ⚠️ Yêu cầu **Python ≥3.10, <3.13** (constraint trong `pyproject.toml`; môi trường `rag/` dùng 3.12).

```bash
# 1. Clone & install (uv sync --locked đọc pyproject.toml + uv.lock → tạo môi trường rag/)
git clone <repo-url>
cd rag-chatbot
cp .env.example .env          # Chọn LLM_PROVIDER + điền API key
make install-dev              # hoặc make install

# 2. Start Redis + Qdrant (Docker)
make local-start

# 3. Ingest sample documents
make ingest

# 4a. Start API backend (FastAPI, port 8080; inline dev mode)
make run-api

# 4b. Start chatbot UI (Gradio, port 7860)
make run-ui

# Production: set RAG_EXECUTION_MODE=redis_worker and run the worker separately
make run-worker
```

**Ghi chú:** production API/worker dùng Redis admission; chỉ worker sau khi
reserve quota mới chạy embedding → Qdrant → reranking → LLM. `inline` là adapter
dev/test và không tạo backlog phân tán.

## 📁 Project Structure

```
src/
├── core/                   # Shared utilities (config, logging, DB connector)
├── ingestion/              # Parse → Chunk → Embed → Store pipeline
│   ├── parsers/            # PDF, Markdown, DOCX parsers (Strategy pattern)
│   ├── chunking/           # Recursive + Parent-Child chunking
│   ├── embeddings.py       # BGE-M3 local adapter / GPU HTTP client
│   ├── batching.py          # point/byte bounded request batches
│   ├── manifest.py          # SQLite source manifest + checkpoints
│   └── pipeline.py         # Orchestrator
├── retrieval/              # Advanced RAG retrieval pipeline
│   ├── query_transform/    # Multi-Query Expansion + HyDE
│   ├── search/             # Dense + Sparse + Hybrid (RRF)
│   ├── reranking/          # Cross-Encoder reranker
│   ├── context/            # Parent resolution + Lost-in-Middle
│   └── retriever.py        # Main orchestrator
├── evaluation/             # RAGAS evaluation pipeline
├── api/                    # FastAPI backend + Gradio UI
└── workers/                # Redis Streams RAG worker
```

## 📊 Evaluation

Run RAG Triad evaluation:

```bash
make evaluate
```

Metrics:
- **Context Relevance** — Are retrieved chunks relevant to the query?
- **Faithfulness** — Is the answer grounded in the context?
- **Answer Relevance** — Does the answer address the question?

## 🔌 API

Backend chạy ở `http://127.0.0.1:8080` (`make run-api`). Tài liệu OpenAPI tại
`/docs`.

### Auth

- Nếu `API_KEY` rỗng (mặc định, dev mode) → không cần auth.
- Nếu `API_KEY` được set → `/chat` và `/chat/stream` phải kèm header
  `X-API-Key: <API_KEY>`; `/health` vẫn public cho health check.
- ⚠️ Không expose server ra ngoài khi `API_KEY` rỗng — chỉ chạy localhost.

```bash
curl -X POST http://127.0.0.1:8080/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"query": "What is the password policy?"}'
```

### Response schema

`POST /chat` trả JSON:

```json
{
  "answer": "All passwords must be at least 12 characters... [1]",
  "sources": [
    {"citation_id": 1, "file_name": "company_policy.md",
     "section_title": "Security Policy", "page_number": 2}
  ]
}
```

`POST /chat/stream` trả SSE. Các event types:

| Event | Data | Ý nghĩa |
|---|---|---|
| `status` | `{"message": "..."}` | Request đã được nhận, đang retrieval |
| `token` | `{"text": "..."}` | Token của câu trả lời |
| `sources` | `{"sources": [...]}` | Sources sau khi generation xong |
| `error` | `{"code", "message"}` | Lỗi an toàn (không leak nội bộ) |
| `end` | `{}` | Kết thúc stream |

### Ingestion sync

`make ingest-sync` đồng bộ Qdrant với source directory — file bị xoá khỏi thư mục
cũng được xoá khỏi DB, scoped theo `INGEST_DATASET_ID`. Lệnh từ chối source rỗng
theo mặc định để tránh xóa nhầm khi volume mount sai; chỉ dùng
`python -m ingestion.main --sync --allow-empty-source` khi đã xác minh nguồn.
Dùng `--dry-run` để xem kế hoạch trước khi mutate. Ingestion có file lỗi sẽ ghi
summary JSON và trả exit code khác 0 để scheduler/CI không đánh dấu job thành công.
PR14 dùng native BM25 trong Qdrant nên generation mới thấy được ngay sau alias
switch, không cần restart API để rebuild sparse index.

Sau khi nâng cấp từ dữ liệu trước PR 8, cần đặt `INGEST_DATASET_ID` rồi chạy full
`make ingest` một lần để tạo payload/IDs mới. Legacy points thiếu `dataset_id` không
bị sync tự động. PR11 ghi generation mới vào các collection dạng
`child_chunks_active__<generation>`/`parent_chunks_active__<generation>`, rồi switch
hai alias `child_chunks_active` và `parent_chunks_active` atomically. Dữ liệu cũ cần
backfill qua staging; không mutate collection production đang active.

Manifest mặc định ở `data/ingest_runs/manifest.sqlite3` và phải đặt trên volume bền
vững khi chạy nhiều container. Có thể resume một job cụ thể:

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir data/real-docs --sync --job-id nightly-2026-08-12
```

Xem trước mà không ghi Qdrant:

```bash
PYTHONPATH=src rag/bin/python -m ingestion.main \
  --data-dir data/real-docs --sync --dry-run --summary-path /tmp/ingest-plan.json
```

Rollback về generation đã giữ lại:

```bash
PYTHONPATH=src rag/bin/python - <<'PY'
from ingestion.pipeline import IngestionPipeline
IngestionPipeline().rollback("<generation-id>")
PY
```

Các generation đã commit cũ hơn `INGEST_GENERATION_RETENTION` sẽ được dọn sau
khi activation thành công; generation lỗi vẫn được giữ để điều tra.

Xem bảng thay đổi PR11 tại
[docs/pr11-changes-and-deployment.md](docs/pr11-changes-and-deployment.md) và
runbook triển khai PR14 tại
[docs/pr14-deployment.md](docs/pr14-deployment.md).

## 🗺️ Port Map

| Service | Port | Ghi chú |
|---|---|---|
| Qdrant | 6333 / 6334 | docker-compose |
| Redis | 6379 | Streams admission + rate reservations |
| LLM server (vLLM / NIM) | 8000 | trỏ bởi `OPENAI_BASE_URL` |
| Embedding GPU service | 8080 | `EMBEDDING_BASE_URL`, BGE-M3 |
| Reranker GPU service | 8080 | `RERANKER_BASE_URL`, BGE-Reranker-v2-m3 |
| FastAPI backend | 8080 | `make run-api` |
| Gradio UI | 7860 | `make run-ui` |

## ⚠️ Limitations

- **Ngôn ngữ:** BM25 tokenizer hỗ trợ Unicode — hoạt động tốt với ngôn ngữ có dấu
  cách phân từ (Anh, Việt kể cả chữ có dấu). CJK (Trung/Nhật/Hàn) chưa hỗ trợ —
  cần tokenizer riêng. Dense search thì đa ngôn ngữ bình thường.
- **Latency:** mỗi câu hỏi có thể dùng tối đa 4 LLM calls (condense khi có history,
  Multi-Query, HyDE và final) cùng cross-encoder rerank; GPU services và Redis
  reservation giúp giới hạn tải nhưng SLO vẫn phải đo trên corpus thật.
- **Multi-turn:** hỗ trợ cơ bản qua Query Condensation (viết lại follow-up thành câu hỏi
  độc lập). Giới hạn: chỉ nhìn 3 lượt gần nhất, tốn thêm 1 LLM call mỗi câu hỏi có history.
- **Hybrid index:** dense + native BM25 chạy và fuse server-side trong Qdrant,
  dùng cùng dataset filter; API không còn load corpus vào RAM hay cần restart
  để refresh sparse index.
- **Scale:** Redis queue được bound bởi `RAG_QUEUE_MAX_OUTSTANDING`; quota LLM
  được reserve atomically trước pipeline (mặc định 15 calls/60s). Context vẫn
  dùng character cap `MAX_CONTEXT_CHARS` của baseline, không có tokenizer/budget
  accounting trong PR14.
- **Provider Gemini:** đã smoke-test end-to-end với `google-genai 2.15.0`
  (Interactions API, model `gemini-2.5-flash`) — xem `review/standalone.md` P0-2.

## 📝 License

MIT
