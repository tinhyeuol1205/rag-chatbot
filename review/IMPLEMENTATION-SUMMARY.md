# 📋 Báo cáo tổng kết — Triển khai toàn bộ Code Review

> **Ngày:** 2026-08-10 (vòng 1) — cập nhật 2026-08-11 (vòng 2, PR 9)
> **Branch:** `dev` — đã merge qua 5 PR (vòng 1) + PR 6-8 (vòng 2) + PR 9
> **Trạng thái:** ✅ Vòng 1 + vòng 2 (PR 6-9) hoàn tất, **114 test xanh**

---

## 1. Tổng quan

Bộ tài liệu review trong `review/` (PR 1-5 + standalone) gồm **4 blocker, 9 high, 11 medium, 7 low** — tổng 31 issues. Toàn bộ đã được xử lý, kèm **2 phát hiện quan trọng** ngoài dự đoán của review (chi tiết ở mục 4).

**Môi trường:** chuyển từ `.venv/` (Python 3.9) → `rag/` (uv venv, Python 3.12.13), sinh `uv.lock`, làm sạch 6 package rác langgraph.

---

## 2. Tình trạng xử lý từng PR

| PR | Nội dung | Issues | Status |
|---|---|---|---|
| **1** | Bug làm sai kết quả im lặng | P0-1, P1-4, P1-5, P1-7 | ✅ Hoàn tất |
| **2** | Chất lượng retrieval & citation | P1-3, P2-1, P2-8, P2-9, P2-10, P3-4, P3-5 | ✅ Hoàn tất |
| **3** | Ingestion, metadata, ID scheme | P1-2, P1-6, P2-5 | ✅ Hoàn tất + **re-ingest** |
| **4** | API async, model loading, hiệu năng | P0-3, P1-1, P1-8, P1-9, P2-3, P2-4, P2-6 | ✅ Hoàn tất |
| **5** | Evaluation, dependencies, docs, tests | P0-4, P2-2, P2-7, P2-11, P3-1, P3-2, P3-3, P3-7 | ✅ Hoàn tất |
| **Standalone** | Gemini (P0-2), Multi-turn (P3-6) | P0-2, P3-6 | ✅ Hoàn tất |

---

## 3. Chi tiết từng bug đã sửa

### PR 1 — Correctness (4 issues)

| ID | Mức | Fix |
|---|---|---|
| **P0-1** | 🔴 | `parent_resolver.py`: fallback về child chunk khi parent mất trong DB; log riêng `missing_parents`/`deduped` |
| **P1-4** | 🟠 | `llm.py`: xử lý `content=None` (content_filter, tool_call-only) → trả `""` |
| **P1-5** | 🟠 | `llm.py`: bỏ qua usage-only chunk khi stream; thêm `max_tokens` vào signature |
| **P1-7** | 🟠 | `multi_query`/`hyde`: graceful degradation (lỗi LLM → dùng query gốc); `sparse.py`: bọc `RetrievalError` |

### PR 2 — Retrieval quality (7 issues)

| ID | Mức | Fix |
|---|---|---|
| **P1-3** | 🟠 | `hybrid.py`: tách `rrf_fusion` module-level; `retriever.py`: fuse TẤT CẢ lists, **cộng dồn** điểm (thay `max`) |
| **P2-1** | 🟡 | `assembler.py`: build context + sources từ CÙNG list reordered, số `[N]` khớp nhau |
| **P2-8** | 🟡 | RRF `enumerate(start=1)` — đúng công thức paper |
| **P2-9** | 🟡 | `sparse.py`: `tokenize()` regex giữ mã hiệu dính dấu câu ("TC-456." → `tc-456`) |
| **P2-10** | 🟡 | `assembler.py`: token budget `MAX_CONTEXT_CHARS=24_000`, bỏ cả block; short-circuit context rỗng |
| **P3-4** | 🔵 | `cross_encoder.py`: copy rồi sort, không mutate input |
| **P3-5** | 🔵 | `multi_query.py`: `_parse_variants` bỏ preamble/numbering/trùng query |

### PR 3 — Ingestion & schema (3 issues)

| ID | Mức | Fix |
|---|---|---|
| **P1-2** | 🟠 | `parent_child.py`: chunk theo TỪNG document giữ `section_title`/`page_number`; `markdown_parser.py`: split `###` + bỏ qua code fence; `page_number` vào payload |
| **P1-6** | 🟠 | `qdrant.py`: `delete_by_file_name` + `create_payload_index`; `pipeline.py`: delete-by-file trước khi ghi, key relative path, `is_file()`, try/except parser |
| **P2-5** | 🟡 | `models.py`: `chunk_id` → **UUIDv5** deterministic (bỏ MD5 prefix collision); xoá hack `.replace("-","")` |

### PR 4 — API & hiệu năng (7 issues)

| ID | Mức | Fix |
|---|---|---|
| **P0-3** | 🔴 | `main.py`: bỏ `async` /chat (threadpool); `iterate_in_threadpool` /chat/stream |
| **P1-1** | 🟠 | `embeddings.py`: module-level `lru_cache _load_model` (trước: load 2 lần) |
| **P1-8** | 🟠 | `cross_encoder.py`: cache model, device, `max_length=512`, cap `RERANK_CANDIDATES=30`, lock predict |
| **P1-9** | 🟠 | `sparse.py`: index dùng chung + `invalidate_bm25_index()`; `qdrant.py`: `scroll_all` phân trang |
| **P2-3** | 🟡 | CORS theo config; `max_length=2000`; `API_HOST/API_PORT=8080` |
| **P2-4** | 🟡 | `chat.py`: `logger.exception`, `USER_FACING_ERROR`, `chat_or_raise`; API trả 503/500 |
| **P2-6** | 🟡 | Port 8080; `.env.example` đúng; `model_validator` provider |

**Bonus:** Qdrant singleton thread-safe (Lock + double-checked locking, `timeout=30`).

### PR 5 — Evaluation & docs (8 issues)

| ID | Mức | Fix |
|---|---|---|
| **P0-4** | 🔴 | `retriever.py`: `RAGResult` + `retrieve()`/`query_with_context()` — contexts là context THẬT; `evaluate.py`: lưu JSON |
| **P2-2** | 🟡 | `metrics.py`: RAGAS 0.2 API (`SingleTurnSample`), bắt exception rộng, judge LLM từ config |
| **P2-7** | 🟡 | `pyproject.toml` → PEP 621 `[project]`; `uv.lock` sinh; `requirements.txt` chốt version |
| **P2-11** | 🟡 | `tests/`: 12 file test (81 cases tổng) |
| **P3-1** | 🔵 | **32 file** docstring di chuyển lên trên `from __future__` |
| **P3-2** | 🔵 | `logger.py`: `_configure()` 1 lần, level filter, stderr |
| **P3-3** | 🔵 | Xoá import rác; `recursive_chunk` ghi rõ demo |
| **P3-7** | 🔵 | README: LLM 2 provider, Port map, Limitations |

### Standalone

| ID | Mức | Kết quả |
|---|---|---|
| **P0-2** | 🔴 | **Review SAI** — Gemini code dùng đúng API (`interactions.create`). Verify + smoke test thật (chi tiết mục 4) |
| **P3-6** | 🔵 | `condense.py`: Query Condensation; truyền history qua layers; **sửa bug search_query không xuống generate** |

---

## 4. ⚠️ Hai phát hiện quan trọng ngoài dự đoán của review

### 4a. P0-2 Gemini — review đã SAI

Review dự đoán `interactions.create` là "API surface không tồn tại". **Thực tế (verify bằng key thật + docs):**

```python
# Đúng chính xác — theo docs hiện tại của Gemini API
interaction = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input="Say OK",
    system_instruction="Reply with exactly one word.",
    generation_config={"temperature": 0.1, "max_output_tokens": 10},
)
print(interaction.output_text)   # → 'OK'
```

- `google-genai` 2.15.0: `interactions.create` **tồn tại và hoạt động**
- `generate()`, `generate_stream()` đều chạy end-to-end (stream lọc `thought_signature` bằng `delta.type == 'text'`)
- **Vẫn áp dụng cải tiến:** `generate_stream` thêm `max_output_tokens`; `get_llm_service()` raise `ConfigurationError`

### 4b. Multi-turn — bug thật nằm trong code, không phải Gemini

Triệu chứng: turn 2 trả `"Could you please clarify what 'it' refers to"` dù condense resolve đúng.

**Nguyên nhân gốc:** `retrieve()` condense thành `search_query`, nhưng `query()` truyền `user_query` **gốc** (chứa "it") vào `_generate()`.

**Fix:** `retrieve()` trả thêm `search_query`; `query()`/`query_with_context()` dùng `search_query` cho generate.

**Kết quả sau fix (Gemini thật):**
- Q1 "What is the company password policy?" → A1 trả lời
- Q2 "How often do employees have to change **it**?" → ✅ **"Employees must change their company passwords every 90 days [1]"** (trước: hỏi lại "it")

---

## 5. Kiểm chứng cuối cùng

| Hạng mục | Kết quả |
|---|---|
| **Test suite** | ✅ **81/81 passed** (12 file test) |
| **Module docstring** | ✅ 0 module thiếu (trước: 32) |
| **Embedding load** | ✅ 1 lần (trước: 2) |
| **Event loop không block** | ✅ `/health` 16ms khi query nặng chạy song song |
| **Re-ingest** | ✅ Idempotent (2 lần → số point không đổi), UUIDv5, section title thật |
| **Log level** | ✅ `LOG_LEVEL=WARNING` filter info |
| **uv pip check** | ✅ All compatible (trước: 7 incompatibility) |

---

## 6. Files chính đã thay đổi

**Đã commit qua 5 PR:**
- `src/core/`: `llm.py`, `config.py`, `logger.py`, `errors.py`, `db/qdrant.py`, `db/__init__.py`
- `src/ingestion/`: `models.py`, `pipeline.py`, `embeddings.py`, `chunking/parent_child.py`, `chunking/recursive.py`, `parsers/markdown_parser.py`
- `src/retrieval/`: `retriever.py`, `search/hybrid.py`, `search/sparse.py`, `search/dense.py`, `context/assembler.py`, `context/parent_resolver.py`, `reranking/cross_encoder.py`, `query_transform/multi_query.py`, `query_transform/hyde.py`, `query_transform/condense.py` (mới), `prompts.py`
- `src/evaluation/`: `evaluate.py`, `metrics.py`
- `src/api/`: `main.py`, `chat.py`, `ui.py`
- `tests/`: **12 file test**
- `pyproject.toml`, `requirements.txt`, `uv.lock`, `Makefile`, `.env.example`, `README.md`

---

## 7. Còn lại (ngoài phạm vi review)

| Mục | Ghi chú |
|---|---|
| `docs/rag_master.md` | Reference chết trong docstring — nên thêm hoặc đổi thành paper gốc |
| Sparse vector (BM42/SPLADE) | Thay BM25 phía client để scale tốt hơn |
| Model nhẹ cho reranker | `ms-marco-MiniLM-L-6-v2` nếu muốn latency < 1s trên CPU |

---

## 8. 🔄 Round 2 (PR 6-9) — phát hiện sau khi merge vòng 1

Vòng 2 trong `review/ROUND2-README.md` gồm **7 issues chức năng + 1 nhóm hygiene**, chia 4 PR.

| PR | Nội dung | Issues | Status |
|---|---|---|---|
| **6** | Auth đồng nhất, không leak lỗi nội bộ | P1-10, P2-12 | ✅ Hoàn tất (merge PR #6) |
| **7** | Unicode BM25, citation contract, evaluation đúng context | P1-11, P1-12, P1-13 | ✅ Hoàn tất (merge PR #7) |
| **8** | Safe replace, sync file đã xoá, dataset isolation | P1-14, P2-13, P2-14→ | ✅ Hoàn tất (merge PR #8) + **re-ingest thêm dataset_id** |
| **9** | Cold-start concurrency, dependency source, lint/docs | P2-14, P3-8, P3-9 | ✅ Hoàn tất (PR này) |

### PR 9 — chi tiết

**P2-14 Cold-start concurrency** 🔵
- `chat.py`: `get_retriever()` → lock + double-checked (publish chỉ sau init thành công)
- `llm.py`: `get_llm_service()` → lock + double-checked
- `embeddings.py` / `cross_encoder.py`: bỏ `lru_cache` → explicit global + lock
  (lru_cache không single-flight khi 2 thread cùng miss cache)
- FastAPI `lifespan`: warmup retriever + models trước khi nhận request, không block
  event loop (`run_in_threadpool`); đóng Qdrant khi shutdown
- `retriever.warmup()`: load model sample-free, KHÔNG gọi paid LLM
- Gradio `main()`: gọi `get_retriever().warmup()` trước launch (fail-fast)
- Test mới `tests/test_singleton_concurrency.py`: 4 tests, 20 thread × 100 calls
  → chỉ tạo 1 instance (retriever, LLM, embedding, reranker)

**P3-8 Ruff quality gate** 🔵
- `ruff check src tests` đã sạch (trước: 17 lỗi). Autofix + sửa tay:
  - `qdrant.py` `create_payload_index`: chỉ catch `UnexpectedResponse` status 409
    (index đã tồn tại); network/auth lỗi propagate (trước: catch mọi Exception)
  - `BLE001` chủ đích (condense/hyde/multi_query) → `# noqa` dòng + giải thích invariant
  - `PYI034` `__new__` → `Self`; `PYI063` → PEP 570 `/`; `SIM102`, `C401`, `I001`, `PIE790`
- Make targets mới: `make lint`, `make check`
- CI mới: `.github/workflows/ci.yml` (ruff + pytest + `uv lock --check`, `--locked`)

**P3-9 Dependency & docs drift** 🔵
- Xoá `requirements.txt` — một nguồn sự thật duy nhất = `pyproject.toml` + `uv.lock`
- Pin Qdrant `qdrant/qdrant:latest` → `qdrant/qdrant:v1.18.3` (version đang chạy,
  khớp qdrant-client 1.19.0)
- README: Python range `>=3.10,<3.13`; `uv sync --locked`; API section mới (auth
  `X-API-Key`, response schema, SSE event types, `make ingest-sync`); Gemini đã test
  end-to-end; BM25 Unicode hỗ trợ tiếng Việt có dấu
- `main.py` docstring: port 8000 → 8080

**Definition of Done PR 9:**

| Hạng mục | Kết quả |
|---|---|
| 20 thread cold-start → 1 retriever/LLM/model | ✅ `test_singleton_concurrency.py` 4/4 pass |
| API lifespan warmup trước request | ✅ `/health` OK sau "Application ready" |
| `ruff check src tests` | ✅ All checks passed |
| `make test` | ✅ **114 passed** (81 + 33 mới từ PR 6-8) |
| `uv lock --check` | ✅ Resolved 249 packages |
| `uv pip check` | ✅ 203 packages compatible |
| Một dependency source | ✅ `pyproject.toml` + `uv.lock` (xoá `requirements.txt`) |
| README khớp thực tế | ✅ API/schema/SSE/Python range/ingestion sync |
| `/chat` end-to-end | ✅ 200 + real sources (citation 1-5, section title thật) |
