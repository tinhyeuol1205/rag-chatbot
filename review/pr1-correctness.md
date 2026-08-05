# PR 1 — Sửa bug làm sai kết quả im lặng

> **Mục tiêu:** Loại bỏ các bug khiến bot trả lời sai/không trả lời được **mà không báo
> lỗi**. Đây là loại bug nguy hiểm nhất trong RAG vì không ai phát hiện được từ log.
>
> **Phụ thuộc:** không có — làm đầu tiên.
> **Cần re-ingest:** ❌ không. Không đổi schema, không đổi ID.
> **Issues:** P0-1, P1-4, P1-5, P1-7
>
> **File bị sửa:**
> - `src/retrieval/context/parent_resolver.py`
> - `src/core/llm.py`
> - `src/retrieval/query_transform/multi_query.py`
> - `src/retrieval/query_transform/hyde.py`
> - `src/retrieval/search/sparse.py` (chỉ phần error message)
> - `tests/test_parent_resolver.py` (mới)
>
> **Definition of Done:**
> - [ ] Chạy 1 query thật, không còn "I don't have enough information" giả
> - [ ] `test_parent_resolver.py` xanh
> - [ ] Log có `missing_parents` khi parent bị thiếu (thay vì im lặng)

---

## P0-1 🔴 ParentResolver âm thầm xoá documents → context rỗng

**File:** `src/retrieval/context/parent_resolver.py:71-89`

```python
for doc in child_results:
    pid = str(doc.get("parent_id", "")).replace("-", "")

    if pid and pid in parent_map and pid not in seen_parent_ids:
        resolved.append({...})           # ✅ có parent
        seen_parent_ids.add(pid)
    elif not pid:
        resolved.append(doc)             # ✅ không có parent → giữ child
    # ❌ KHÔNG CÓ else: pid có nhưng không tìm thấy trong parent_map → BỊ XOÁ
```

**Triệu chứng:** Bot trả lời `"I don't have enough information in the company documents
to answer this question."` dù retrieval đã tìm đúng chunk. Không có log lỗi, không có
exception — bug im lặng hoàn toàn.

**Nguyên nhân gốc:** Nhánh `pid` truthy nhưng `pid not in parent_map` không được xử lý.
Kịch bản xảy ra rất thực tế:

1. **Sau re-ingest** (kết hợp với P1-6 ở PR 3): child chunk cũ còn trong DB trỏ tới
   `parent_id` đã không còn tồn tại → toàn bộ những child này bị xoá khỏi kết quả.
2. **Parent collection bị xoá/chưa tạo** nhưng child collection còn → `get_by_ids` trả
   `[]` → `parent_map` rỗng → `resolved` rỗng → context rỗng.
3. Qdrant `retrieve` chỉ trả về những ID tồn tại (không báo lỗi ID thiếu) → thiếu âm thầm.

Kèm theo: log ở dòng 91-96 báo `deduplicated=len(child_results) - len(resolved)` — trộn
lẫn "dedupe hợp lệ" với "bị mất do lỗi", nên đọc log cũng không phát hiện được.

**Fix:** luôn có đường lùi về child chunk, và log riêng số bị miss.

```python
resolved = []
seen_parent_ids: set[str] = set()
missing_parents = 0
deduped = 0

for doc in child_results:
    pid = str(doc.get("parent_id") or "").replace("-", "")

    if not pid:
        resolved.append(doc)                       # child không có parent
        continue

    if pid in seen_parent_ids:
        deduped += 1                               # 2 child cùng parent → bỏ 1
        continue

    parent_payload = parent_map.get(pid)
    if parent_payload is None:
        # ★ FALLBACK: parent mất trong DB → dùng child, KHÔNG được xoá
        missing_parents += 1
        resolved.append(doc)
        continue

    seen_parent_ids.add(pid)
    resolved.append({
        "chunk_id": pid,
        "content": parent_payload.get("content") or doc["content"],
        "score": doc.get("rerank_score", doc.get("score", 0)),
        "file_name": parent_payload.get("file_name") or doc.get("file_name", ""),
        "section_title": parent_payload.get("section_title") or doc.get("section_title", ""),
        "page_number": parent_payload.get("page_number") or doc.get("page_number"),
        "is_parent": True,
    })

if missing_parents:
    logger.warning(
        "Parent chunks missing in DB — fell back to child chunks. "
        "Có thể do re-ingest để lại chunk rác, cần chạy lại ingest sạch.",
        missing=missing_parents,
    )

logger.info(
    "Parent resolution done",
    children_in=len(child_results),
    docs_out=len(resolved),
    deduplicated=deduped,
    missing_parents=missing_parents,
)
```

> **Ghi chú về `page_number`:** field này chưa có trong payload Qdrant hiện tại (sẽ được
> thêm ở PR 3 / P1-2). Viết sẵn `.get("page_number")` là an toàn — trả `None` cho tới khi
> PR 3 xong.

**Ghi chú về hack `.replace("-", "")` (dòng 59-65, 72):** comment trong code nói đúng —
Qdrant chấp nhận MD5 hex 32 ký tự không dấu gạch (parser UUID của Rust nhận cả dạng
"simple") rồi trả về dạng canonical **có** dấu gạch. Nên hack này *đang hoạt động*, không
phải bug crash. Nhưng nó phụ thuộc vào hành vi normalize của Qdrant. Sửa gốc ở
[PR 3 / P2-5](pr3-ingestion-schema.md#p2-5--chunk_id--md5200-ký-tự-đầu--collision-làm-mất-chunk)
(sinh UUID thật) sẽ bỏ được hack này. **Trong PR này giữ nguyên `.replace()`.**

---

## P1-4 🟠 `.strip()` trên `content` có thể `None`

**File:** `src/core/llm.py:89`

```python
return response.choices[0].message.content.strip()
```

**Triệu chứng:** `AttributeError: 'NoneType' object has no attribute 'strip'`.

**Khi nào xảy ra:** `message.content` là `None` khi
`finish_reason == "content_filter"`, khi model chỉ trả tool_call, hoặc với một số
reasoning model chỉ điền `reasoning_content`. Với vLLM/NIM (`.env` đang trỏ tới endpoint
OpenAI-compatible) trường hợp này khá phổ biến.

**Fix:**

```python
def generate(self, user_prompt, system_prompt="", temperature=0.1, max_tokens=1024) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    response = self._client.chat.completions.create(
        model=self._model, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    if not response.choices:
        logger.warning("LLM returned no choices")
        return ""
    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        logger.warning("LLM returned empty content", finish_reason=choice.finish_reason)
        return ""
    return content.strip()
```

---

## P1-5 🟠 `chunk.choices[0]` crash khi provider gửi usage-only chunk

**File:** `src/core/llm.py:108-110`

```python
for chunk in stream:
    if chunk.choices[0].delta.content:    # ❌ IndexError khi choices == []
        yield chunk.choices[0].delta.content
```

**Triệu chứng:** `IndexError: list index out of range` giữa lúc đang stream — câu trả lời
bị cắt ngang, user thấy `❌ Error: list index out of range`.

**Khi nào xảy ra:** vLLM/NIM/Azure gửi chunk cuối chỉ chứa `usage` với `choices: []`
(khi bật `stream_options={"include_usage": True}`, hoặc mặc định ở một số phiên bản
vLLM). `.env` đang dùng `OPENAI_BASE_URL` local → khả năng cao là vLLM → bug này sẽ
xuất hiện thật.

**Fix — kèm luôn `max_tokens` đang bị thiếu:**

```python
def generate_stream(
    self,
    user_prompt: str,
    system_prompt: str = "",
    temperature: float = 0.1,
    max_tokens: int = 1024,          # ★ THÊM: hiện streaming KHÔNG giới hạn output
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    stream = self._client.chat.completions.create(
        model=self._model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,       # ★ THÊM
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:        # ★ usage-only chunk → bỏ qua
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
```

**Sửa kèm ở abstract base** (`llm.py:47-54`) — thêm `max_tokens` vào signature
`generate_stream` để 2 provider nhất quán:

```python
    @abstractmethod
    def generate_stream(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,       # ★ THÊM
    ):
        """Gọi LLM và trả về generator (streaming)."""
```

`GeminiLLMService.generate_stream` (dòng 157-180) cũng đang thiếu `max_output_tokens`
→ Gemini stream không giới hạn độ dài. Vì toàn bộ class đó đang được xử lý riêng ở
[`standalone.md` / P0-2](standalone.md#p0-2--gemini-provider-gọi-api-surface-không-tồn-tại),
trong PR này chỉ cần **thêm `max_tokens` vào signature** cho khớp abstract base, đừng sửa
sâu vào body.

---

## P1-7 🟠 Không có try/except nào trong retrieval → 1 lỗi LLM giết cả query

**File:** toàn bộ `src/retrieval/**` (đã grep: **0** occurrence của `except`)

**Triệu chứng:** LLM rate-limit / timeout ở bước Multi-Query Expansion (bước ①) → toàn bộ
query fail, user thấy `❌ Error: Rate limit exceeded`. Trong khi thực tế **vẫn có thể trả
lời được** bằng query gốc — expansion và HyDE chỉ là bước *tăng cường*, không bắt buộc.

Tệ hơn: `errors.py` định nghĩa sẵn 5 exception class (`RetrievalError`, `IngestionError`,
`ParsingError`, `ConfigurationError`) nhưng **không file nào dùng** (đã grep: 0 usage
ngoài chính `errors.py`) — abstraction chết.

**Fix — graceful degradation cho các bước tăng cường:**

```python
# src/retrieval/query_transform/multi_query.py
def expand(self, query: str) -> list[str]:
    logger.info("Expanding query", original=query, n_variants=settings.EXPAND_N_QUERY)
    try:
        raw_output = self.llm.generate(
            user_prompt=MULTI_QUERY_PROMPT.format(n=settings.EXPAND_N_QUERY, query=query),
            temperature=0.7, max_tokens=300,
        )
    except Exception as e:
        # ★ Degrade: không có variant vẫn search được bằng query gốc
        logger.warning("Multi-query expansion failed, using original query only",
                       error=str(e))
        return [query]

    variants = [q.strip() for q in raw_output.split("\n") if q.strip()]
    all_queries = [query] + variants[:settings.EXPAND_N_QUERY]
    logger.info("Query expanded", total_queries=len(all_queries), variants=variants)
    return all_queries
```

> Việc **làm sạch** `variants` (bỏ "Here are 3 versions:", numbering) thuộc
> [PR 2 / P3-5](pr2-retrieval-quality.md#p3-5--multi-query-parse-text-thô). PR này chỉ
> thêm try/except, giữ nguyên logic parse.

```python
# src/retrieval/query_transform/hyde.py
def generate_embedding(self, query: str) -> list[float] | None:
    """Trả về None nếu HyDE fail → caller sẽ dùng dense search thường."""
    try:
        hypothetical = self._generate_hypothetical(query)
    except Exception as e:
        logger.warning("HyDE generation failed, skipping HyDE", error=str(e))
        return None
    if not hypothetical.strip():
        logger.warning("HyDE returned empty text, skipping HyDE")
        return None
    vector = self.embedder.embed_single(hypothetical)
    logger.info("HyDE embedding generated",
                query=query[:50], hypothetical=hypothetical[:80])
    return vector
```

`HybridSearcher.search` đã xử lý `hyde_vector=None` đúng rồi (`hybrid.py:74` — `if
hyde_vector:`), nên không cần sửa thêm. Nhưng **phải cập nhật type hint** ở
`retriever.py` và docstring `hyde.py:54-55` (đang ghi "Returns: Vector 384d") cho khớp
việc giờ có thể trả `None`.

Với bước **bắt buộc** (search, rerank) thì nên bọc bằng `RetrievalError` để `chat.py`
phân biệt được:

```python
# src/retrieval/search/sparse.py — collection chưa tồn tại là lỗi hay gặp nhất
from core.errors import RetrievalError

def _build_index(self) -> None:
    logger.info("Building BM25 index...")
    try:
        points = self.qdrant.scroll_all(settings.CHILD_COLLECTION)
    except Exception as e:
        raise RetrievalError(
            f"Không đọc được collection '{settings.CHILD_COLLECTION}'. "
            f"Đã chạy 'make ingest' chưa? Lỗi gốc: {e}"
        ) from e
    ...
```

> Việc `chat.py` **bắt** `RetrievalError` và trả message riêng thuộc
> [PR 4 / P2-4](pr4-api-performance.md#p2-4--lỗi-nội-bộ-bị-trả-thẳng-ra-ui). Trong PR này
> `RetrievalError` sẽ đi qua `except Exception` hiện có ở `chat.py` — vẫn tốt hơn trạng
> thái cũ vì message đã hữu ích.

---

## Test cần thêm

Tạo `tests/conftest.py` (nếu PR này chạy trước PR 5):

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

```python
# tests/test_parent_resolver.py
from retrieval.context.parent_resolver import ParentResolver


class _FakeQdrant:
    """Giả lập Qdrant trả về subset ID (hành vi thật của retrieve)."""
    def __init__(self, store): self.store = store
    def get_by_ids(self, collection_name, ids):
        return [type("P", (), {"id": i, "payload": self.store[i]})()
                for i in ids if i in self.store]


def test_missing_parent_falls_back_to_child():
    """★ Bug P0-1: parent mất trong DB thì phải giữ child, KHÔNG được xoá."""
    resolver = ParentResolver()
    resolver.qdrant = _FakeQdrant({})            # parent collection rỗng
    children = [{"chunk_id": "c1", "content": "child text",
                 "parent_id": "p1", "file_name": "a.md"}]
    out = resolver.resolve(children)
    assert len(out) == 1                          # trước fix: len == 0
    assert out[0]["content"] == "child text"


def test_two_children_same_parent_dedupe():
    resolver = ParentResolver()
    resolver.qdrant = _FakeQdrant({"p1": {"content": "parent text", "file_name": "a.md"}})
    children = [
        {"chunk_id": "c1", "content": "x", "parent_id": "p1", "file_name": "a.md"},
        {"chunk_id": "c2", "content": "y", "parent_id": "p1", "file_name": "a.md"},
    ]
    out = resolver.resolve(children)
    assert len(out) == 1
    assert out[0]["content"] == "parent text"


def test_child_without_parent_id_kept():
    resolver = ParentResolver()
    resolver.qdrant = _FakeQdrant({})
    children = [{"chunk_id": "c1", "content": "orphan", "parent_id": None}]
    out = resolver.resolve(children)
    assert len(out) == 1
    assert out[0]["content"] == "orphan"
```

---

## Kiểm chứng

```bash
make test                      # test_parent_resolver.py xanh
make local-start && make ingest
make run-ui                    # hỏi 1 câu có trong sample docs
```

**Dấu hiệu thành công:**
- Bot trả lời có nội dung, không phải "I don't have enough information"
- Log **không** có dòng `missing_parents` (nếu có → còn chunk rác, cần PR 3)
- Log có `docs_out=` > 0

**Test degradation thủ công** — tạm đặt `OPENAI_API_KEY` sai rồi hỏi:
- Trước fix: `❌ Error: ...` (query chết hoàn toàn)
- Sau fix: vẫn chết ở bước generate (đúng — đó là bước bắt buộc), nhưng log phải có
  `"Multi-query expansion failed, using original query only"` và
  `"HyDE generation failed, skipping HyDE"` → chứng tỏ degradation hoạt động
