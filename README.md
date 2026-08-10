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
│ Parse → Chunk → Embed  │──────► Qdrant Vector DB
│ (Parent-Child strategy) │       (child_chunks + parent_chunks)
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
| Embeddings | `BAAI/bge-small-en-v1.5` (local, free) |
| Reranker | `BAAI/bge-reranker-v2-m3` (local, free) |
| Vector DB | Qdrant |
| Backend | FastAPI + SSE streaming |
| Frontend | Gradio |
| Evaluation | RAGAS (RAG Triad metrics) |
| Dependency | uv (`pyproject.toml` + `uv.lock`) |

## 🚀 Quick Start

> ⚠️ Yêu cầu **Python ≥3.10** (môi trường `rag/` dùng 3.12).

```bash
# 1. Clone & install (uv sync đọc pyproject.toml + uv.lock → tạo môi trường rag/)
git clone <repo-url>
cd rag-chatbot
cp .env.example .env          # Chọn LLM_PROVIDER + điền API key
make install-dev              # hoặc make install

# 2. Start Qdrant (Docker)
make local-start

# 3. Ingest sample documents
make ingest

# 4a. Start API backend (FastAPI, port 8080)
make run-api

# 4b. Start chatbot UI (Gradio, port 7860)
make run-ui
```

**Ghi chú:** UI (`run-ui`) gọi retriever **trực tiếp trong process** (không qua FastAPI). API (`run-api`) dành cho client riêng qua HTTP/SSE.

## 📁 Project Structure

```
src/
├── core/                   # Shared utilities (config, logging, DB connector)
├── ingestion/              # Parse → Chunk → Embed → Store pipeline
│   ├── parsers/            # PDF, Markdown, DOCX parsers (Strategy pattern)
│   ├── chunking/           # Recursive + Parent-Child chunking
│   ├── embeddings.py       # bge-small-en embedding service
│   └── pipeline.py         # Orchestrator
├── retrieval/              # Advanced RAG retrieval pipeline
│   ├── query_transform/    # Multi-Query Expansion + HyDE
│   ├── search/             # Dense + Sparse + Hybrid (RRF)
│   ├── reranking/          # Cross-Encoder reranker
│   ├── context/            # Parent resolution + Lost-in-Middle
│   └── retriever.py        # Main orchestrator
├── evaluation/             # RAGAS evaluation pipeline
└── api/                    # FastAPI backend + Gradio UI
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

## 🗺️ Port Map

| Service | Port | Ghi chú |
|---|---|---|
| Qdrant | 6333 / 6334 | docker-compose |
| LLM server (vLLM / NIM) | 8000 | trỏ bởi `OPENAI_BASE_URL` |
| FastAPI backend | 8080 | `make run-api` |
| Gradio UI | 7860 | `make run-ui` |

## ⚠️ Limitations

- **Ngôn ngữ:** BM25 tokenizer chỉ hoạt động tốt với ngôn ngữ có dấu cách phân từ
  (Anh, Việt). CJK chưa hỗ trợ. Dense search thì đa ngôn ngữ bình thường.
- **Latency:** trên CPU, mỗi câu hỏi mất vài giây đến vài chục giây (2 LLM call +
  cross-encoder rerank). Xem `RERANKER_MODEL_ID` trong `.env` để đổi sang model nhẹ hơn.
- **Multi-turn:** hỗ trợ cơ bản qua Query Condensation (viết lại follow-up thành câu hỏi
  độc lập). Giới hạn: chỉ nhìn 3 lượt gần nhất, tốn thêm 1 LLM call mỗi câu hỏi có history.
- **BM25 index:** build 1 lần trong process. Nếu ingest ở terminal khác với server đang
  chạy, **phải restart server** để BM25 thấy dữ liệu mới. Dense search thì thấy ngay.
- **Scale:** BM25 giữ toàn bộ corpus trong RAM. Không phù hợp với > ~100k chunk.
- **Provider Gemini:** chưa được test end-to-end — xem `review/standalone.md`.

## 📝 License

MIT
