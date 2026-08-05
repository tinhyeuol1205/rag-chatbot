# PR 4 — API async, model loading, hiệu năng

> **Mục tiêu:** Làm API chịu được request đồng thời, giảm latency, và siết cấu hình
> mạng/lỗi.
>
> **Phụ thuộc:** [PR 1](pr1-correctness.md) (P2-4 dựa trên `RetrievalError` được raise ở
> P1-7). Nên làm sau [PR 2](pr2-retrieval-quality.md) vì P1-8 sửa cùng file
> `cross_encoder.py` với P3-4.
> **Cần re-ingest:** ❌ không.
> **Issues:** P0-3, P1-1, P1-8, P1-9, P2-3, P2-4, P2-6
>
> **File bị sửa:**
> - `src/api/main.py` (bỏ `async`, threadpool, CORS, `max_length`)
> - `src/api/chat.py` (`logger.exception`, không leak error)
> - `src/ingestion/embeddings.py` (`lru_cache` module-level)
> - `src/retrieval/reranking/cross_encoder.py` (cache, device, cap, lock)
> - `src/retrieval/search/sparse.py` (index dùng chung + invalidate)
> - `src/core/db/qdrant.py` (`scroll_all` phân trang, thread-safe singleton, timeout)
> - `src/retrieval/retriever.py` (song song hoá LLM call)
> - `src/retrieval/search/hybrid.py` (song song hoá dense/sparse)
> - `src/core/config.py` (nhiều setting mới)
> - `Makefile`, `.env.example`
>
> **Definition of Done:**
> - [ ] Log `"Loading embedding model"` xuất hiện **đúng 1 lần** (trước: 2)
> - [ ] Gửi 2 request đồng thời → `/health` vẫn phản hồi ngay
> - [ ] Latency 1 query giảm rõ rệt (đo trước/sau)
> - [ ] `make run-api` bind `127.0.0.1:8080`, không phải `0.0.0.0:8000`

---

## P0-3 🔴 Sync code nặng trong `async def` → block event loop

**File:** `src/api/main.py:56-73`

```python
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    answer = chat(request.query)          # ❌ sync, mất 30-90s, block toàn bộ event loop
    return ChatResponse(answer=answer)

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    async def generate():
        for token in chat_stream(request.query):   # ❌ sync generator trong async generator
            yield {"data": token}
    return EventSourceResponse(generate())
```

**Triệu chứng:** Có 2 user hỏi cùng lúc → user thứ 2 phải chờ user thứ 1 xong hoàn toàn.
`GET /health` cũng treo. Uvicorn 1 worker → server "chết" trong lúc chạy pipeline.

**Nguyên nhân gốc:** `chat()` là hàm sync gọi network (LLM) + inference CPU (embedding,
cross-encoder). Trong `async def`, FastAPI **không** đưa nó vào threadpool — nó chạy ngay
trên event loop. Cross-encoder trên CPU còn giữ GIL rất lâu.

**Fix cho `/chat`** — bỏ `async`, để FastAPI tự đẩy vào threadpool:

```python
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):       # ← def, KHÔNG async def
    answer = chat(request.query)
    return ChatResponse(answer=answer)
```

**Fix cho `/chat/stream`** — dùng `iterate_in_threadpool`:

```python
from starlette.concurrency import iterate_in_threadpool

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    async def generate():
        yield {"event": "status", "data": "Đang tìm kiếm tài liệu..."}
        async for token in iterate_in_threadpool(chat_stream(request.query)):
            yield {"data": token}
        yield {"event": "end", "data": ""}     # tín hiệu kết thúc cho client
    return EventSourceResponse(generate())
```

**Sửa docstring `api/main.py:64-68`** — đang ghi:

> Client nhận từng token realtime, giảm thời gian chờ đợi.
> Time to First Token (TTFT) thường < 500ms.

Sai. `chat_stream` là sync generator nên toàn bộ retrieval (LLM expand + HyDE + search +
rerank) chạy ở **token đầu tiên** → TTFT thực tế 30-90s (giảm sau P1-8 nhưng vẫn tính bằng
giây, không phải ms). Sửa thành:

```python
    """Chat streaming endpoint — trả về SSE (Server-Sent Events).

    Lưu ý: retrieval pipeline (multi-query + HyDE + search + rerank) chạy TRƯỚC
    token đầu tiên, nên TTFT ≈ thời gian retrieval (vài giây), không phải < 500ms.
    Event "status" được gửi ngay để client biết request đã được nhận.
    """
```

**Liên quan:** vì `_retriever` là singleton dùng chung mà giờ có nhiều thread → xem P1-8
(cross-encoder không thread-safe khi predict song song, cần lock) và P3-3 ở
[PR 5](pr5-eval-deps-docs.md#p3-3--dead-code) (Qdrant singleton `__new__` race).
**Lock cho cross-encoder nằm trong PR này** (P1-8), lock cho Qdrant singleton cũng nên làm
luôn ở đây thay vì đợi PR 5 — xem cuối file.

---

## P1-1 🟠 Embedding model bị load 2 lần (bug class-attr / instance-attr)

**File:** `src/ingestion/embeddings.py:31-46` (pattern y hệt ở `cross_encoder.py:43-52`)

```python
class EmbeddingService:
    """Singleton-like embedding service. Load model 1 lần, dùng mãi."""
    _model: SentenceTransformer | None = None     # ← class attribute

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(...)  # ❌ tạo INSTANCE attribute!
        return self._model
```

**Nguyên nhân gốc:** `self._model = ...` **không** ghi vào class attribute — nó tạo một
instance attribute mới che (shadow) class attribute. Class attribute vẫn là `None` mãi.
→ **Mỗi instance `EmbeddingService()` load lại model từ đầu.** Docstring "Singleton-like...
load 1 lần" là **sai**.

**Số lần load thực tế** trong `RAGRetriever.__init__`:
- `HyDEGenerator()` → `EmbeddingService()` #1 → load bge-small
- `HybridSearcher()` → `DenseSearcher()` → `EmbeddingService()` #2 → load bge-small **lần 2**

Thêm `IngestionPipeline` → lần 3 (khi chạy ingest). Mỗi lần ~130MB RAM + thời gian load.
Với model to hơn (bge-base/large) hoặc GPU, đây thành lỗi OOM.

**Fix — dùng module-level cache thật:**

```python
# src/ingestion/embeddings.py
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from core import get_logger
from core.config import settings

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load 1 lần duy nhất cho cả process (lru_cache ở module level)."""
    logger.info("Loading embedding model", model=settings.EMBEDDING_MODEL_ID)
    model = SentenceTransformer(
        settings.EMBEDDING_MODEL_ID,
        device=settings.EMBEDDING_DEVICE,
    )
    logger.info("Embedding model loaded", dimensions=settings.EMBEDDING_SIZE)
    return model


class EmbeddingService:
    """Wrapper mỏng quanh model đã cache ở module level.

    Tạo bao nhiêu instance cũng chỉ load model 1 lần (cache nằm ở _load_model).
    """

    @property
    def model(self) -> SentenceTransformer:
        return _load_model()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed danh sách text → danh sách vectors."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,   # ★ xem ghi chú dưới
            batch_size=32,
        )
        return vectors.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Embed 1 text duy nhất → 1 vector."""
        return self.embed([text])[0]
```

### ★ Ghi chú quan trọng về `normalize_embeddings`

Qdrant collection dùng `Distance.COSINE` (`core/db/qdrant.py:70`) nên Qdrant tự normalize
khi so sánh — về mặt *ranking* thì không normalize cũng cho kết quả đúng. Nhưng model BGE
được **train và benchmark với vector đã normalize**, và giá trị `score` trả về sẽ dễ
đọc/dễ đặt threshold hơn khi normalize.

Nếu bật, **phải bật ở cả ingest và query**. Vì cả hai đều đi qua `EmbeddingService` nên chỉ
cần sửa 1 chỗ này là nhất quán. Nhưng chú ý: **vector đã có trong Qdrant từ lần ingest
trước là chưa normalize.** Ranking vẫn đúng (cosine bất biến với scale) nên **không bắt
buộc re-ingest**, chỉ là score sẽ lệch thang giữa point cũ và mới. Nếu muốn sạch hoàn
toàn thì re-ingest — hoặc gộp thay đổi này vào [PR 3](pr3-ingestion-schema.md) (PR đó đã
re-ingest sẵn).

**Áp dụng y hệt cho reranker** — xem P1-8 ngay dưới.

---

## P1-8 🟠 Reranker chấm ~80 cặp trên CPU với model 568M params

**File:** `src/retrieval/reranking/cross_encoder.py:50,71`

```python
self._model = CrossEncoder(settings.RERANKER_MODEL_ID)   # không device, không max_length
...
scores = self.model.predict(pairs)                        # không batch_size
```

**Phân tích số lượng:** `EXPAND_N_QUERY=3` → 4 query (gốc + 3 variant) + 1 lượt HyDE = 5
lượt search. Mỗi lượt trả `TOP_K=20` → **~100 rows**, sau fusion còn ~50-80 unique.
`bge-reranker-v2-m3` là XLM-RoBERTa-large (~568M params). Trên CPU:
**~0.3-1s mỗi cặp** ở độ dài 512 token → **25-80 giây chỉ riêng reranking**, mỗi query.

Cộng thêm 2 LLM call tuần tự (expand + HyDE) trước đó. Tổng latency thực tế mỗi câu hỏi
ước lượng **30-90s**.

**Fix — 5 thay đổi, áp dụng cả 5** (đã bao gồm fix P3-4 "không mutate input" của PR 2):

```python
from functools import lru_cache
from threading import Lock

from sentence_transformers import CrossEncoder

from core import get_logger
from core.config import settings

logger = get_logger(__name__)

# CrossEncoder không thread-safe khi predict song song → cần lock (liên quan P0-3)
_predict_lock = Lock()


@lru_cache(maxsize=1)
def _load_reranker() -> CrossEncoder:
    logger.info("Loading reranker model", model=settings.RERANKER_MODEL_ID)
    model = CrossEncoder(
        settings.RERANKER_MODEL_ID,
        device=settings.EMBEDDING_DEVICE,   # ★ (2) config này đang bị bỏ qua hoàn toàn
        max_length=512,                     # ★ (3) chặn input dài → chậm bất định
    )
    logger.info("Reranker loaded")
    return model


class CrossEncoderReranker:
    """Rerank kết quả search bằng Cross-Encoder model."""

    @property
    def model(self) -> CrossEncoder:
        return _load_reranker()             # ★ (1) cache module-level, không reload

    def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        if not documents:
            return []

        # ★ (4) Cap số candidate — documents đã sort theo RRF nên cắt là an toàn
        candidates = documents[:settings.RERANK_CANDIDATES]
        if len(documents) > len(candidates):
            logger.info("Capped rerank candidates",
                        total=len(documents), kept=len(candidates))

        pairs = [(query, doc["content"]) for doc in candidates]
        with _predict_lock:
            scores = self.model.predict(pairs, batch_size=settings.RERANK_BATCH_SIZE)

        # ★ (5) KHÔNG mutate list/dict của caller (P3-4)
        scored = [{**doc, "rerank_score": float(s)} for doc, s in zip(candidates, scores)]
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        top_docs = scored[:settings.KEEP_TOP_K]

        logger.info("Reranking done",
                    input_count=len(candidates), output_count=len(top_docs),
                    top_score=round(top_docs[0]["rerank_score"], 4) if top_docs else 0)
        return top_docs
```

Thêm vào `src/core/config.py`:

```python
    # --- Reranker Model ---
    RERANKER_MODEL_ID: str = "BAAI/bge-reranker-v2-m3"
    RERANK_CANDIDATES: int = 30    # Số candidate tối đa đưa vào cross-encoder
    RERANK_BATCH_SIZE: int = 16
```

### Nếu vẫn quá chậm trên CPU — cân nhắc đổi model

Ghi rõ trade-off vào README thay vì im lặng:

| Model | Params | CPU latency (30 cặp, ước lượng) | Chất lượng |
|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` | 568M | ~15-30s | tốt nhất, đa ngôn ngữ |
| `BAAI/bge-reranker-base` | 278M | ~5-10s | tốt |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | **~0.5s** | khá (chỉ tiếng Anh) |

Với demo/portfolio chạy CPU, `ms-marco-MiniLM-L-6-v2` là lựa chọn hợp lý; giữ
`bge-reranker-v2-m3` làm cấu hình "quality mode" qua `.env`.

### Giảm latency thêm — song song hoá các bước độc lập

**(a) 2 LLM call ở đầu pipeline.** `expand()` và `generate_embedding()` không phụ thuộc
nhau (`retriever.py:76,79`) nhưng đang chạy tuần tự → tiết kiệm được thời gian của 1 LLM
call:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as pool:
    f_expand = pool.submit(self.expander.expand, user_query)
    f_hyde = pool.submit(self.hyde.generate_embedding, user_query)
    expanded_queries = f_expand.result()
    hyde_vector = f_hyde.result()
```

(Cả 2 đã có try/except nội bộ sau [PR 1 / P1-7](pr1-correctness.md#p1-7--không-có-tryexcept-nào-trong-retrieval--1-lỗi-llm-giết-cả-query)
nên `.result()` không raise.)

**(b) Dense + Sparse trong `HybridSearcher`.** Docstring `hybrid.py:71` ghi "Bước 1: Chạy
**song song** 2 search engines" nhưng code chạy **tuần tự** (dòng 74-80). Hoặc sửa
docstring, hoặc làm cho đúng:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as pool:
    if hyde_vector:
        f_dense = pool.submit(self.dense.search_by_vector, hyde_vector, top_k)
    else:
        f_dense = pool.submit(self.dense.search, query, top_k)
    f_sparse = pool.submit(self.sparse.search, query, top_k)
    dense_results, sparse_results = f_dense.result(), f_sparse.result()
```

Dense là I/O-bound (chờ Qdrant) nên thread giúp thật; BM25 giữ GIL nên lợi ích một phần.
Cần `_index_lock` của P1-9 để an toàn.

**(c) 5 lượt search trong `retriever`** cũng độc lập → có thể song song hoá tương tự.
Nhưng cẩn thận: mỗi lượt lại spawn 2 thread con → 10 thread. Đặt `max_workers` hợp lý và
đo trước khi làm.

---

## P1-9 🟠 BM25 index stale + `scroll_all` cắt im lặng ở 10k

**File:** `src/retrieval/search/sparse.py:33-36, 88-116` và `core/db/qdrant.py:117-125`

**Ba vấn đề riêng biệt:**

**(a) Index không bao giờ invalidate.** `_index` là instance attribute, build 1 lần ở
query đầu tiên. Server FastAPI dùng `_retriever` singleton → sau khi `make ingest` thêm
tài liệu mới, **BM25 vẫn dùng index cũ** cho tới khi restart server. Dense search thì
thấy tài liệu mới ngay (query trực tiếp Qdrant) → **hybrid search bất đối xứng**, khó
debug.

**(b) `scroll_all` chỉ đọc trang đầu:**

```python
def scroll_all(self, collection_name: str, limit: int = 10000) -> list:
    points, _ = self.client.scroll(collection_name=..., limit=limit, ...)
    return points          # ❌ bỏ next_page_offset → mất mọi point sau #10000
```

Quá 10k chunk → phần còn lại **biến mất khỏi BM25** hoàn toàn, không log, không lỗi.
Với `CHILD_CHUNK_SIZE=400`, 10k chunk ≈ 4MB text ≈ vài trăm trang — đạt được dễ dàng.

**(c) Toàn bộ corpus nằm trong RAM** (`self._documents` giữ full content) → không scale.
Không fix trong PR này, chỉ ghi nhận giới hạn vào README.

**Fix (b) — phân trang thật:**

```python
# src/core/db/qdrant.py
def scroll_all(self, collection_name: str, batch_size: int = 1000,
               max_points: int | None = None) -> list:
    """Đọc TẤT CẢ points trong collection (phân trang đúng cách)."""
    all_points, offset = [], None
    while True:
        points, offset = self.client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(points)
        if offset is None:
            break
        if max_points is not None and len(all_points) >= max_points:
            logger.warning("scroll_all hit max_points cap — dữ liệu bị cắt",
                           collection=collection_name, cap=max_points,
                           fetched=len(all_points))
            break
    logger.info("Scrolled collection", collection=collection_name, total=len(all_points))
    return all_points
```

**Fix (a) — index dùng chung + invalidate được** (đã bao gồm `tokenize` của
[PR 2 / P2-9](pr2-retrieval-quality.md#p2-9--bm25-tokenizer-thất-bại-đúng-ca-dùng-mà-docstring-quảng-cáo)):

```python
# src/retrieval/search/sparse.py
import heapq
import re
from threading import Lock

from rank_bm25 import BM25Okapi

from core import get_logger
from core.config import settings
from core.db import QdrantConnector
from core.errors import RetrievalError

logger = get_logger(__name__)

_index_lock = Lock()
_shared: dict = {"index": None, "documents": None, "version": 0}

_TOKEN_RE = re.compile(r"[0-9a-z]+(?:[-_][0-9a-z]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenizer giữ được mã kiểu 'TC-456' kể cả khi dính dấu câu."""
    return _TOKEN_RE.findall(text.lower())


def invalidate_bm25_index() -> None:
    """Gọi sau khi ingest xong để BM25 nạp lại corpus."""
    with _index_lock:
        _shared["index"] = None
        _shared["documents"] = None
        _shared["version"] += 1
    logger.info("BM25 index invalidated", version=_shared["version"])


class SparseSearcher:
    """Tìm kiếm bằng BM25 keyword matching.

    Index được chia sẻ ở module level: tạo bao nhiêu instance cũng chỉ build 1 lần.
    Gọi invalidate_bm25_index() sau khi ingest để nạp lại.
    """

    def __init__(self):
        self.qdrant = QdrantConnector()

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or settings.TOP_K
        index, documents = self._ensure_index()
        if not documents or index is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            logger.info("BM25 search skipped — query has no usable tokens")
            return []

        scores = index.get_scores(query_tokens)
        # top-k bằng heap thay vì sort toàn bộ corpus (O(n log k) thay vì O(n log n))
        top_idx = heapq.nlargest(top_k, range(len(scores)), key=scores.__getitem__)

        results = []
        for i in top_idx:
            if scores[i] <= 0:              # chỉ lấy docs match ít nhất 1 từ
                continue
            results.append({**documents[i], "score": float(scores[i]), "source": "sparse"})

        logger.info("BM25 search done", query=query[:50], results=len(results))
        return results

    def _ensure_index(self):
        with _index_lock:
            if _shared["index"] is None and _shared["documents"] is None:
                self._build_index_locked()
            return _shared["index"], _shared["documents"] or []

    def _build_index_locked(self) -> None:
        """Load tất cả child chunks từ Qdrant → build BM25 index."""
        logger.info("Building BM25 index...")
        try:
            points = self.qdrant.scroll_all(settings.CHILD_COLLECTION)
        except Exception as e:
            raise RetrievalError(
                f"Không đọc được collection '{settings.CHILD_COLLECTION}'. "
                f"Đã chạy 'make ingest' chưa? Lỗi gốc: {e}"
            ) from e

        documents, corpus = [], []
        for point in points:
            payload = point.payload or {}
            content = payload.get("content", "")
            if not content:
                continue
            documents.append({
                "chunk_id": point.id,
                "content": content,
                "parent_id": payload.get("parent_id"),
                "file_name": payload.get("file_name", ""),
                "section_title": payload.get("section_title", ""),
                "page_number": payload.get("page_number"),
            })
            corpus.append(tokenize(content))

        _shared["documents"] = documents
        _shared["index"] = BM25Okapi(corpus) if corpus else None
        logger.info("BM25 index built", total_documents=len(documents))
```
Rồi gọi `invalidate_bm25_index()` ở cuối `IngestionPipeline.run()`:

```python
# src/ingestion/pipeline.py — cuối hàm run()
        logger.info("Ingestion pipeline complete")

        # BM25 index (nếu có process nào đang giữ) cần nạp lại corpus mới
        try:
            from retrieval.search.sparse import invalidate_bm25_index
            invalidate_bm25_index()
        except ImportError:
            pass    # ingestion không bắt buộc phụ thuộc retrieval
```

**Ghi rõ giới hạn vào README:** invalidate chỉ có tác dụng **trong cùng process**. Nếu
ingest ở terminal khác với server đang chạy, **vẫn phải restart server** để BM25 thấy dữ
liệu mới. (Cách xử lý đúng về dài hạn: chuyển sang sparse vector của Qdrant — BM42/SPLADE
— để không cần index phía client. Ngoài scope PR này.)

---

## P2-3 🟡 CORS `*` + bind `0.0.0.0` + không auth + query không giới hạn

**File:** `src/api/main.py:29-34, 39-40`, `Makefile:34`

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
class ChatRequest(BaseModel):
    query: str                      # ❌ không giới hạn độ dài
```

```makefile
run-api: uvicorn api.main:app --host 0.0.0.0 --port 8000   # ❌ mở ra mọi interface
```

**Rủi ro:** Đây là chatbot cho **tài liệu nội bộ công ty**. Hiện tại bất kỳ ai truy cập
được port 8000 đều query được toàn bộ knowledge base — không auth, không rate limit.
Commit `10e392a` đã siết Gradio về `localhost` + `share=False` — nhưng FastAPI vẫn mở
`0.0.0.0`, tức là **cửa trước vẫn mở** dù cửa sau đã đóng.

`query: str` không giới hạn → gửi query 1MB: nó đi vào prompt của multi-query + HyDE +
rerank → tốn token/chi phí, và cross-encoder xử lý input khổng lồ → treo CPU (DoS bằng
1 request).

**Fix:**

```makefile
# Makefile — mặc định chỉ localhost, cần mở thì override qua biến
API_HOST ?= 127.0.0.1
API_PORT ?= 8080          # ★ 8000 đang bị vLLM dùng — xem P2-6

run-api: ## Chạy FastAPI backend
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m uvicorn api.main:app \
		--host $(API_HOST) --port $(API_PORT) --reload
```

```python
# src/api/main.py
from pydantic import BaseModel, Field

from core.config import settings

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,        # ★ list cụ thể, không "*"
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)   # ★ chặn query khổng lồ
```

```python
# src/core/config.py
    # --- API ---
    CORS_ORIGINS: list[str] = ["http://localhost:7860", "http://127.0.0.1:7860"]
    API_KEY: str = ""     # để trống = tắt auth (dev); set giá trị = bật auth
```

Auth tối thiểu (nếu deploy ra ngoài localhost):

```python
from fastapi import Depends, Header, HTTPException, status


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.API_KEY:          # dev mode: bỏ qua
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat_endpoint(request: ChatRequest): ...
```

**Ghi chú về mức độ:** đây là project portfolio/demo, không phải hệ production đang phục
vụ dữ liệu thật — nên đây là "hardening", không phải sự cố. Nhưng `--host 0.0.0.0` trên
**server dùng chung** thì đúng là nên sửa ngay.

---

## P2-4 🟡 Lỗi nội bộ bị trả thẳng ra UI

**File:** `src/api/chat.py:47-49, 68-70`

```python
except Exception as e:
    logger.error("Chat error", error=str(e))      # ❌ mất traceback
    return f"❌ Error: {str(e)}"                   # ❌ leak nội dung exception
```

**Hai vấn đề:**

1. **`logger.error` + `str(e)` mất traceback** → không debug được. Với structlog nên dùng
   `logger.exception(...)` (tự thêm `exc_info`).
2. **Leak thông tin.** Exception của OpenAI SDK thường chứa `base_url`, tên model, đôi khi
   một phần header. Với `.env` hiện tại, `OPENAI_BASE_URL=http://localhost:8000/v1` sẽ
   hiện ra UI. Message của Qdrant chứa host/port nội bộ. Ngoài ra API trả **HTTP 200** kèm
   body `{"answer": "❌ Error: ..."}` → client không phân biệt được thành công/thất bại,
   monitoring không đếm được error rate.

**Fix — log đầy đủ ở server, trả message chung cho user:**

```python
# src/api/chat.py
from core.errors import RAGChatbotError

USER_FACING_ERROR = (
    "Xin lỗi, hệ thống đang gặp sự cố khi xử lý câu hỏi. "
    "Vui lòng thử lại sau ít phút."
)


def chat(query: str) -> str:
    """Xử lý câu hỏi và trả về câu trả lời (non-streaming). Không raise."""
    if not query.strip():
        return "Please enter a question."
    try:
        return get_retriever().query(query, stream=False)
    except RAGChatbotError as e:
        logger.exception("Chat failed (known error)")
        return f"⚠️ {e}"          # lỗi mình tự raise → message đã an toàn, hữu ích
    except Exception:
        logger.exception("Chat failed (unexpected)")   # ★ full traceback vào log
        return USER_FACING_ERROR                        # ★ không leak ra ngoài


def chat_or_raise(query: str) -> str:
    """Bản cho API layer — raise để FastAPI trả HTTP status đúng."""
    if not query.strip():
        return "Please enter a question."
    return get_retriever().query(query, stream=False)


def chat_stream(query: str):
    """Xử lý câu hỏi và trả về câu trả lời (streaming). Không raise."""
    if not query.strip():
        yield "Please enter a question."
        return
    try:
        yield from get_retriever().query(query, stream=True)
    except RAGChatbotError as e:
        logger.exception("Chat stream failed (known error)")
        yield f"⚠️ {e}"
    except Exception:
        logger.exception("Chat stream failed (unexpected)")
        yield USER_FACING_ERROR
```

Và ở API layer, trả status code đúng:

```python
# src/api/main.py
from fastapi import HTTPException

from api.chat import USER_FACING_ERROR, chat_or_raise, chat_stream
from core.errors import RAGChatbotError


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        return ChatResponse(answer=chat_or_raise(request.query))
    except RAGChatbotError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail=USER_FACING_ERROR)
```

`api/ui.py` tiếp tục dùng `chat_stream` (bản không raise) — Gradio cần string để hiển thị,
không xử lý được exception.

**Cần thêm logger vào `api/main.py`** — file này hiện chưa có:

```python
from core import get_logger

logger = get_logger(__name__)
```

---

## P2-6 🟡 `.env` trỏ LLM vào `localhost:8000` — cùng port với FastAPI

**File:** `.env` (không commit — chỉ tồn tại local)

```
OPENAI_API_KEY=EMPTY
OPENAI_BASE_URL=http://localhost:8000/v1     # ← LLM server (vLLM?) ở port 8000
```

`Makefile:34`: `uvicorn api.main:app --host 0.0.0.0 --port 8000` ← **cũng** port 8000.

**Hệ quả:** chạy `make run-api` khi vLLM đã chiếm 8000 → uvicorn fail `Address already in
use`. Hoặc nếu uvicorn lên trước → mọi LLM call đi vào **chính FastAPI app** →
`404 Not Found` cho `/v1/chat/completions`, hoặc tệ hơn là request loop.

**Fix:** đổi port API sang 8080 (đã có trong `Makefile` ở P2-3), và ghi rõ vào README:

```markdown
### Port map
| Service | Port | Ghi chú |
|---|---|---|
| Qdrant | 6333 / 6334 | docker-compose |
| LLM server (vLLM / NIM) | 8000 | trỏ bởi `OPENAI_BASE_URL` |
| FastAPI backend | 8080 | `make run-api` |
| Gradio UI | 7860 | `make run-ui` |
```
### Vấn đề thứ 2 — `.env` và `.env.example` lệch nhau (drift)

| Biến | `.env.example` | `.env` thực tế |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | **thiếu** → mặc định `openai` |
| `GEMINI_API_KEY` | có | **thiếu** |
| `EMBEDDING_DEVICE` | **thiếu** (dù `config.py:51` có) | thiếu |

`.env.example` đã được cập nhật cho commit `a0a4828` (LLM abstraction) nhưng `.env` local
vẫn ở dạng cũ. Nghĩa là: nếu ai làm theo `.env.example` (`LLM_PROVIDER=gemini`) thì sẽ
đâm ngay vào [P0-2](standalone.md#p0-2--gemini-provider-gọi-api-surface-không-tồn-tại)
(Gemini không chạy).

**Khuyến nghị:** để `LLM_PROVIDER=openai` trong `.env.example` cho tới khi P0-2 được xác
minh xong, và bổ sung `EMBEDDING_DEVICE` + các setting mới của PR này:

```bash
# --- LLM Provider (chọn 1 trong 2: "openai" hoặc "gemini") ---
# LƯU Ý: provider "gemini" chưa được test end-to-end — xem review/standalone.md
LLM_PROVIDER=openai

# --- Embedding Model (chạy local, không tốn tiền) ---
EMBEDDING_MODEL_ID=BAAI/bge-small-en-v1.5
EMBEDDING_SIZE=384
EMBEDDING_DEVICE=cpu              # cpu | cuda | mps

# --- Reranker Model (chạy local) ---
RERANKER_MODEL_ID=BAAI/bge-reranker-v2-m3
RERANK_CANDIDATES=30
RERANK_BATCH_SIZE=16

# --- Context budget ---
MAX_CONTEXT_CHARS=24000

# --- API ---
API_KEY=                          # để trống = tắt auth (dev)
```

**Thêm validation trong config** — fail sớm với message rõ, thay vì lỗi 401 khó hiểu:

```python
# src/core/config.py
from pydantic import model_validator

from core.errors import ConfigurationError     # ★ dùng exception đang bị bỏ không

    @model_validator(mode="after")
    def _check_provider_credentials(self):
        provider = self.LLM_PROVIDER.lower()
        if provider not in {"openai", "gemini"}:
            raise ConfigurationError(
                f"LLM_PROVIDER='{self.LLM_PROVIDER}' không hợp lệ. Dùng 'openai' hoặc 'gemini'."
            )
        # base_url tự host (vLLM) thường dùng key giả "EMPTY" → chấp nhận
        if provider == "openai" and not self.OPENAI_API_KEY and not self.OPENAI_BASE_URL:
            raise ConfigurationError("LLM_PROVIDER=openai nhưng thiếu OPENAI_API_KEY.")
        if provider == "gemini" and not self.GEMINI_API_KEY:
            raise ConfigurationError("LLM_PROVIDER=gemini nhưng thiếu GEMINI_API_KEY.")
        return self
```

Đồng thời `get_llm_service()` (`llm.py:206`) nên raise `ConfigurationError` thay vì
`ValueError` để nhất quán với `errors.py`.

> ⚠️ **Cẩn thận circular import:** `config.py` giờ import `core.errors`. `errors.py` không
> import gì → không có cycle. Nhưng **đừng** để `config.py` import `core.logger`
> ([PR 5 / P3-2](pr5-eval-deps-docs.md#p3-2--structlogconfigure-gọi-lại-mỗi-lần-get_logger-không-có-log-level)
> làm `logger.py` import `config.py`, nên chiều ngược lại sẽ tạo cycle).

---

## Bonus trong PR này: Qdrant singleton thread-safe

Sau P0-3, request chạy trong threadpool → nhiều thread cùng gọi `client` lần đầu là tình
huống **thật**, không phải giả định. `QdrantConnector` (`qdrant.py:30-55`) có `__new__` và
lazy init đều không thread-safe:

```python
from threading import Lock


class QdrantConnector:
    """Singleton connector cho Qdrant vector database."""

    _instance: "QdrantConnector | None" = None
    _client: QdrantClient | None = None
    _lock = Lock()

    def __new__(cls) -> "QdrantConnector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def client(self) -> QdrantClient:
        """Lazy init: chỉ tạo connection khi thực sự cần dùng."""
        if self._client is None:
            with self._lock:
                if self._client is None:      # double-check sau khi lấy lock
                    self._client = QdrantClient(
                        host=settings.QDRANT_HOST,
                        port=settings.QDRANT_PORT,
                        timeout=30,           # ★ mặc định có thể treo rất lâu
                    )
                    logger.info("Connected to Qdrant",
                                host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        return self._client
```

> **Ghi nhận:** pattern class-attr/instance-attr ở đây giống P1-1 nhưng **không phải bug**
> — vì `__new__` luôn trả về cùng một instance nên chỉ có 1 chỗ để gán. Chỉ cần biết là nó
> hoạt động vì lý do khác với những gì code trông như đang làm.

---

## Kiểm chứng

**1. Model chỉ load 1 lần:**

```bash
PYTHONPATH=src python -c "
from retrieval.retriever import RAGRetriever
r = RAGRetriever()
" 2>&1 | grep -c "Loading embedding model"
# phải in: 1        (trước fix: 2)
```

**2. Event loop không bị block** — chạy `make run-api` rồi:

```bash
# terminal 1: gửi 1 request nặng
curl -s -X POST localhost:8080/chat -H 'Content-Type: application/json' \
  -d '{"query":"What is the password policy?"}' &

# terminal 2 (ngay lập tức): health phải trả về NGAY, không chờ
time curl -s localhost:8080/health
# phải < 100ms. Trước fix: chờ hết request kia (30-90s)
```

**3. Đo latency trước/sau:**

```bash
time curl -s -X POST localhost:8080/chat -H 'Content-Type: application/json' \
  -d '{"query":"How many days of annual leave do employees get?"}' > /dev/null
```

Ghi lại số trước khi làm PR và sau khi làm. Kỳ vọng giảm đáng kể nhờ cap candidates
(80 → 30 cặp) + song song hoá 2 LLM call.

**4. Query quá dài bị chặn:**

```bash
python -c "print('x'*3000)" | xargs -0 -I{} curl -s -o /dev/null -w '%{http_code}\n' \
  -X POST localhost:8080/chat -H 'Content-Type: application/json' \
  -d "{\"query\":\"{}\"}"
# phải trả 422 (validation error), không phải 200
```

**5. Lỗi không leak** — tạm đổi `OPENAI_BASE_URL` sang URL sai, hỏi 1 câu:
- UI phải hiện message chung ("hệ thống đang gặp sự cố"), **không** chứa `localhost` hay
  tên model
- Log server phải có full traceback
- `curl /chat` phải trả HTTP 500/503, không phải 200

**6. BM25 invalidate:**

```bash
PYTHONPATH=src python -c "
from retrieval.search.sparse import SparseSearcher, invalidate_bm25_index
s1, s2 = SparseSearcher(), SparseSearcher()
s1.search('policy')            # build index
invalidate_bm25_index()
s2.search('policy')            # rebuild
" 2>&1 | grep -c "BM25 index built"
# phải in: 2  (chứng tỏ invalidate hoạt động, và index dùng chung giữa 2 instance)
```
