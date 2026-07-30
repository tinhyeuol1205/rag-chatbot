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
| LLM | OpenAI GPT-4o-mini |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, free) |
| Reranker | `BAAI/bge-reranker-v2-m3` (local, free) |
| Vector DB | Qdrant |
| Backend | FastAPI + SSE streaming |
| Frontend | Gradio |
| Evaluation | RAGAS (RAG Triad metrics) |

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone <repo-url>
cd rag-chatbot
cp .env.example .env          # Fill in OPENAI_API_KEY
make install

# 2. Start Qdrant
make local-start

# 3. Ingest sample documents
make ingest

# 4. Start chatbot
make run-ui
```

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

## 📝 License

MIT
