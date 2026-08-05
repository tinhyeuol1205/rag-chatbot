# Code Review — RAG Chatbot

Review commit `10e392a`. Phạm vi: toàn bộ `src/` (44 file Python), `pyproject.toml`,
`requirements.txt`, `Makefile`, `docker-compose.yml`, `.env` / `.env.example`, `README.md`.

**Giới hạn của review:** Đây là review **tĩnh** (đọc code). Các thư viện chưa được cài
trong môi trường nên **không chạy được** code để xác nhận runtime. Những kết luận phụ
thuộc phiên bản thư viện được đánh dấu ⚠️ **CẦN KIỂM CHỨNG** kèm command để tự xác minh.

---

## Cách dùng bộ tài liệu này

Mỗi file dưới đây là **một PR độc lập, tự chứa** — đọc riêng vẫn đủ ngữ cảnh để thực thi.
Làm **theo thứ tự** vì có phụ thuộc giữa các PR.

| PR | File | Nội dung | Cần re-ingest? |
|---|---|---|---|
| **1** | [`pr1-correctness.md`](pr1-correctness.md) | Bug làm sai kết quả im lặng | Không |
| **2** | [`pr2-retrieval-quality.md`](pr2-retrieval-quality.md) | Chất lượng retrieval & citation | Không |
| **3** | [`pr3-ingestion-schema.md`](pr3-ingestion-schema.md) | Ingestion, metadata, ID scheme | ⚠️ **CÓ** |
| **4** | [`pr4-api-performance.md`](pr4-api-performance.md) | API async, model loading, hiệu năng | Không |
| **5** | [`pr5-eval-deps-docs.md`](pr5-eval-deps-docs.md) | Evaluation, dependencies, docs, tests | Không |
| — | [`standalone.md`](standalone.md) | Gemini provider (bị chặn) + multi-turn (feature) | Không |

Tổng: **4 blocker, 9 high, 11 medium, 7 low**.

---

## Tóm tắt điều hành

Kiến trúc tổng thể **tốt**: phân lớp rõ (core / ingestion / retrieval / evaluation / api),
Strategy + Dispatcher pattern cho parsers, Strategy cho LLM provider, tách biệt
search / rerank / context assembly. 5 kỹ thuật RAG trong README đều có code thật, không
phải vỏ rỗng. Docstring giải thích thuật toán (RRF, HyDE, Lost-in-Middle) rất chi tiết —
đây là điểm mạnh của project portfolio.

Tuy nhiên có **4 bug làm sai kết quả im lặng** (không crash, chỉ trả lời kém đi — loại
bug nguy hiểm nhất trong RAG), **1 bug chặn hoàn toàn provider Gemini**, **1 bug làm API
treo dưới tải đồng thời**, và **metrics evaluation hiện đang đo sai pipeline** nên số đo
không dùng được.

**Không có bug nào cần viết lại kiến trúc.** Bốn P0 đều là fix cục bộ trong 1 file. Sau
PR 1-4 thì hệ thống sẽ chạy đúng và đủ nhanh để dùng thật; PR 5 làm cho số đo evaluation
tin được và bộ dependency reproducible.

---

## Bảng ưu tiên đầy đủ

| ID | Mức | Vấn đề | File | PR |
|---|---|---|---|---|
| **P0-1** | 🔴 Blocker | ParentResolver **âm thầm xoá** documents → context rỗng → bot trả lời "I don't have enough information" | `retrieval/context/parent_resolver.py:71-89` | 1 |
| **P0-2** | 🔴 Blocker | Gemini provider dùng API surface không đúng SDK → `LLM_PROVIDER=gemini` chết | `core/llm.py:135-180` | — |
| **P0-3** | 🔴 Blocker | Sync code nặng chạy trong `async def` → block event loop, API treo | `api/main.py:56-73` | 4 |
| **P0-4** | 🔴 Blocker | Evaluation đo **pipeline khác** với pipeline sinh answer → metrics vô nghĩa | `evaluation/evaluate.py:40-44` | 5 |
| **P1-1** | 🟠 High | Embedding model bị load **2 lần** (bug class-attr vs instance-attr) | `ingestion/embeddings.py:34-46` | 4 |
| **P1-2** | 🟠 High | Metadata thật (section title, page number) bị **ghi đè** → citation vô dụng | `ingestion/chunking/parent_child.py:44-71` | 3 |
| **P1-3** | 🟠 High | Multi-Query fusion dùng `max` thay vì `sum` RRF → mất tín hiệu chính của Multi-Query | `retrieval/retriever.py:137-151` | 2 |
| **P1-4** | 🟠 High | `.strip()` trên `content` có thể là `None` → AttributeError | `core/llm.py:89` | 1 |
| **P1-5** | 🟠 High | `chunk.choices[0]` crash khi provider gửi usage-only chunk (vLLM/NIM) | `core/llm.py:108-110` | 1 |
| **P1-6** | 🟠 High | Re-ingest để lại **chunk rác** vĩnh viễn (không delete-by-file) | `ingestion/pipeline.py:43-80` | 3 |
| **P1-7** | 🟠 High | Không có try/except nào trong retrieval → 1 lỗi LLM giết cả query | `retrieval/**` | 1 |
| **P1-8** | 🟠 High | Reranker chấm ~80 cặp trên CPU với model 568M params → 30-90s/query | `retrieval/reranking/cross_encoder.py:71` | 4 |
| **P1-9** | 🟠 High | BM25 index **stale** sau re-ingest + `scroll_all` cắt im lặng ở 10k | `retrieval/search/sparse.py:88-116`, `core/db/qdrant.py:117-125` | 4 |
| **P2-1** | 🟡 Medium | Số `[Source N]` trong context **không khớp** danh sách Sources | `retrieval/context/assembler.py:46-77` | 2 |
| **P2-2** | 🟡 Medium | RAGAS gọi theo API 0.1 nhưng requirements cho phép cài 0.2+ | `evaluation/metrics.py:60-87` | 5 |
| **P2-3** | 🟡 Medium | CORS `*` + bind `0.0.0.0` + không auth + query không giới hạn độ dài | `api/main.py:29-34`, `Makefile:34` | 4 |
| **P2-4** | 🟡 Medium | Lỗi nội bộ (có thể chứa API key / URL) bị trả thẳng ra UI | `api/chat.py:47-49` | 4 |
| **P2-5** | 🟡 Medium | `chunk_id` = MD5(200 ký tự đầu) → collision làm mất chunk | `ingestion/models.py:58-63` | 3 |
| **P2-6** | 🟡 Medium | `.env` trỏ LLM vào `localhost:8000` — **cùng port** với FastAPI | `.env` | 4 |
| **P2-7** | 🟡 Medium | `pyproject.toml` và `requirements.txt` lệch nhau; pyproject không cài được | cả 2 file | 5 |
| **P2-8** | 🟡 Medium | RRF `rank` bắt đầu từ 0 thay vì 1 (lệch công thức paper) | `retrieval/search/hybrid.py:118` | 2 |
| **P2-9** | 🟡 Medium | BM25 tokenizer `.split()` — thất bại đúng ca dùng mà docstring quảng cáo | `retrieval/search/sparse.py:61,111` | 2 |
| **P2-10** | 🟡 Medium | Không có token budget cho context | `retrieval/context/assembler.py` | 2 |
| **P2-11** | 🟡 Medium | Không có test nào, `make test` fail | `Makefile:51` | 5 |
| **P3-1** | 🔵 Low | **32/44 file** mất module docstring (`__future__` đặt trước docstring) | toàn bộ `src/` | 5 |
| **P3-2** | 🔵 Low | `structlog.configure()` gọi lại mỗi lần `get_logger()`; không có log level | `core/logger.py:24-35` | 5 |
| **P3-3** | 🔵 Low | Dead code: `errors.py` (0 usage), `recursive_chunk` (0 usage), import rác | nhiều file | 5 |
| **P3-4** | 🔵 Low | Reranker mutate list của caller; bỏ qua `EMBEDDING_DEVICE` | `retrieval/reranking/cross_encoder.py:50,78` | 2 |
| **P3-5** | 🔵 Low | Multi-query parse text thô → dễ lẫn "Here are 3 versions:" vào query | `retrieval/query_transform/multi_query.py:65` | 2 |
| **P3-6** | 🔵 Low | Chat history nhận vào nhưng bị bỏ qua → không có multi-turn | `api/ui.py:34-48` | — |
| **P3-7** | 🔵 Low | README / docstring drift so với code thật | `README.md`, `api/ui.py:52` | 5 |

---

## Nguyên tắc chung khi thực hiện

1. **Sửa docstring kèm code trong cùng commit.** Docstring của project này là tài sản
   chính (portfolio) — docstring sai còn tệ hơn không có docstring, vì người đọc tin nó.
   Danh sách drift đầy đủ ở [PR 5, mục P3-7](pr5-eval-deps-docs.md#p3-7--readme--docstring-drift).
2. **Không merge code chưa gọi thật lần nào.** Áp dụng đặc biệt cho P0-2 (Gemini).
3. **Mỗi PR phải chạy được `make test` xanh** trước khi merge (PR 5 tạo `tests/`; các PR
   trước thêm test tương ứng vào cùng thư mục — xem mục "Test cần thêm" của từng PR).

---

## Ghi nhận điểm mạnh

Những điểm nên **giữ nguyên**, đừng "sửa" trong lúc refactor:

1. **Phân lớp module rất sạch.** `core` / `ingestion` / `retrieval` / `evaluation` / `api`
   với dependency đi một chiều. Không có circular import. Dễ test từng phần.
2. **Strategy + Dispatcher cho parsers** (`parsers/dispatcher.py`) — thêm format mới chỉ
   cần 1 class + 1 dòng registry. Đúng pattern, đúng chỗ.
3. **Strategy cho LLM provider** (`llm.py`) — abstraction hợp lý, `BaseLLMService` đúng
   interface tối thiểu. Vấn đề nằm ở implementation Gemini, không phải ở thiết kế.
4. **Docstring giải thích *tại sao*, không chỉ *cái gì*.** `hybrid.py:12-31` (RRF có ví dụ
   số cụ thể), `cross_encoder.py:11-25` (bi-encoder vs cross-encoder có sơ đồ),
   `assembler.py:6-23` (Lost-in-Middle có hình minh hoạ), `metrics.py:23-26` (bảng chẩn
   đoán khi metric cao/thấp) — chất lượng cao hơn hầu hết project cùng loại. Đây là lý do
   P3-1 (docstring bị vô hiệu hoá) đáng sửa dù chỉ là "low severity".
5. **5 kỹ thuật RAG đều được implement thật**, không phải wrapper mỏng quanh LangChain.
   Có Parent-Child với 2 collection riêng, có RRF tự viết, có HyDE tách riêng.
6. **Config tập trung qua pydantic-settings** với default cho mọi biến → chạy được ngay
   không cần `.env` đầy đủ.
7. **Eval dataset viết tay từ chính sample docs** (`dataset.py`) với ground truth chi
   tiết, cụ thể, kiểm chứng được. Nhiều project bỏ hẳn bước này.
