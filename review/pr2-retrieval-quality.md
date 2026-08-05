# PR 2 — Chất lượng retrieval & citation

> **Mục tiêu:** Sửa các lỗi thuật toán làm giảm độ chính xác xếp hạng, và làm citation
> truy vết được.
>
> **Phụ thuộc:** [PR 1](pr1-correctness.md) (P1-7 đã thêm try/except vào `multi_query.py`
> — P3-5 dưới đây sửa tiếp phần parse trong cùng hàm).
> **Cần re-ingest:** ❌ không.
> **Issues:** P1-3, P2-1, P2-8, P2-9, P2-10, P3-4, P3-5
>
> **File bị sửa:**
> - `src/retrieval/search/hybrid.py` (tách `rrf_fusion` ra module-level)
> - `src/retrieval/retriever.py` (fuse tất cả list, xoá `_deduplicate`)
> - `src/retrieval/context/assembler.py` (citation nhất quán + token budget)
> - `src/retrieval/search/sparse.py` (tokenizer)
> - `src/retrieval/reranking/cross_encoder.py` (không mutate input)
> - `src/retrieval/query_transform/multi_query.py` (parse phòng thủ)
> - `src/core/config.py` (`MAX_CONTEXT_CHARS`)
> - `tests/test_fusion.py`, `tests/test_assembler.py`, `tests/test_tokenize.py` (mới)
>
> **Definition of Done:**
> - [ ] Chunk được nhiều query cùng tìm thấy xếp trên chunk chỉ 1 query tìm thấy
> - [ ] Số trong `[Source N]` khớp số trong danh sách `### Sources`
> - [ ] `tokenize("See TC-456.")` chứa `"tc-456"`
> - [ ] `make evaluate` chạy được, so sánh `avg_keyword_overlap` trước/sau

---

## P1-3 🟠 Multi-Query fusion dùng `max` thay vì `sum` RRF

**File:** `src/retrieval/retriever.py:82-95` + `_deduplicate` 137-151

```python
all_results = []
hyde_results = self.searcher.search(user_query, hyde_vector=hyde_vector)
all_results.extend(hyde_results)
for eq in expanded_queries:
    all_results.extend(self.searcher.search(eq))

unique = self._deduplicate(all_results)   # ❌ giữ bản có rrf_score CAO NHẤT
```

```python
existing_score = seen[cid].get("rrf_score", seen[cid].get("score", 0))
new_score = doc.get("rrf_score", doc.get("score", 0))
if new_score > existing_score:            # ❌ MAX, không phải SUM
    seen[cid] = doc
```

**Nguyên nhân gốc:** RRF được áp dụng **riêng lẻ trong từng** `HybridSearcher.search()`
(fuse dense+sparse của 1 query), rồi kết quả của 4 query khác nhau chỉ được **dedupe bằng
max**. Điều này phá bỏ chính lợi ích của Multi-Query Expansion:

| Chunk | Xuất hiện ở | Nên xếp hạng | Hiện tại (`max`) |
|---|---|---|---|
| A | cả 4 query, rank ~5 | **cao** (đồng thuận mạnh) | `1/(60+5) = 0.0154` |
| B | 1 query duy nhất, rank 1 | thấp hơn (có thể do query lỗi) | `1/(60+1) = 0.0164` → **thắng A** |

→ Chunk B (chỉ 1 query tìm được) đứng trên chunk A (cả 4 query đều tìm được). Sai hoàn
toàn về ý nghĩa. Multi-Query Expansion về bản chất là một **ensemble** — tín hiệu quan
trọng nhất là "bao nhiêu query đồng ý", và `max` xoá đúng tín hiệu đó.

Ảnh hưởng thực tế bị *che mờ* bởi việc reranker chạy sau đó (cross-encoder sẽ sửa phần
nào thứ tự), nhưng vẫn sai ở chỗ: `unique` bị cắt ngầm theo thứ tự sai khi ta cap số
candidate ([PR 4 / P1-8](pr4-api-performance.md#p1-8--reranker-chấm-80-cặp-trên-cpu-với-model-568m-params)),
nên chunk tốt có thể bị loại **trước khi** reranker kịp thấy.

**Fix — fuse tất cả list cùng lúc, cộng dồn điểm.** Tách hàm ra module-level để
`retriever.py` dùng lại được:

```python
# src/retrieval/search/hybrid.py
def rrf_fusion(*result_lists: list[dict], k: int = RRF_K) -> list[dict]:
    """RRF trên N danh sách. Điểm được CỘNG DỒN qua mọi list chứa doc."""
    rrf_scores: dict[str, float] = {}
    doc_store: dict[str, dict] = {}
    hit_counts: dict[str, int] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):    # ★ start=1, xem P2-8
            cid = doc["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            hit_counts[cid] = hit_counts.get(cid, 0) + 1
            if cid not in doc_store:
                doc_store[cid] = doc

    merged = []
    for cid in sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True):
        doc = doc_store[cid].copy()
        doc["rrf_score"] = rrf_scores[cid]
        doc["n_hits"] = hit_counts[cid]      # hữu ích để debug/log
        merged.append(doc)
    return merged
```

`HybridSearcher._rrf_fusion` giờ chỉ delegate (giữ để không phá interface cũ):

```python
    def _rrf_fusion(self, *result_lists: list[dict]) -> list[dict]:
        return rrf_fusion(*result_lists)
```

Rồi trong `retriever.py`:

```python
from retrieval.search.hybrid import rrf_fusion

# ③ Search — thu từng list RIÊNG BIỆT, không extend chung
result_lists = [self.searcher.search(user_query, hyde_vector=hyde_vector)]
for eq in expanded_queries:
    result_lists.append(self.searcher.search(eq))

# ④ Fuse TẤT CẢ cùng lúc — điểm cộng dồn qua mọi query
unique = rrf_fusion(*result_lists)
logger.info(
    "Fusion done",
    n_lists=len(result_lists),
    total_rows=sum(len(r) for r in result_lists),
    unique=len(unique),
    top_n_hits=unique[0]["n_hits"] if unique else 0,
)
```

Sau đó **xoá `_deduplicate`** (dòng 137-151) — nó không còn cần thiết vì fusion đã dedupe.

**Cập nhật docstring** `retriever.py:22-23` — sơ đồ đang ghi "③ Hybrid Search cho MỖI
query → ④ Deduplicate kết quả từ tất cả queries". Đổi thành "④ RRF Fusion trên tất cả
queries (điểm cộng dồn)".

**Cân nhắc nâng cao (optional, không bắt buộc trong PR này):** cho HyDE list trọng số cao
hơn vì nó thường chính xác nhất — `rrf_fusion` nhận thêm `weights: list[float]` và cộng
`w / (k + rank)`.

---

## P2-8 🟡 RRF `rank` bắt đầu từ 0 thay vì 1

**File:** `src/retrieval/search/hybrid.py:113-118`

```python
for rank, doc in enumerate(result_list):       # rank = 0, 1, 2...
    rrf_scores[chunk_id] = ... + 1.0 / (RRF_K + rank)
```

Công thức paper Cormack et al. là `1/(k + rank)` với **rank ≥ 1**. Ở đây rank 0 cho
`1/60` thay vì `1/61`.

**Tác động:** rất nhỏ (thứ tự tương đối gần như không đổi vì mọi doc đều lệch cùng chiều)
— nhưng docstring dòng 20-26 tính ví dụ bằng `1/(60+1)`, `1/(60+5)`, tức là **tài liệu
mô tả rank-từ-1 còn code làm rank-từ-0**. Không khớp nhau.

**Fix:** `enumerate(result_list, start=1)` — đã bao gồm trong đoạn `rrf_fusion` ở P1-3.

**Nhớ cập nhật comment dòng 116-117:**

```python
# TRƯỚC:
# Công thức RRF: 1 / (k + rank)
# rank bắt đầu từ 0, nên rank 0 = vị trí #1

# SAU:
# Công thức RRF: 1 / (k + rank), rank tính từ 1 (đúng paper Cormack et al.)
```

---

## P2-1 🟡 Số `[Source N]` không khớp danh sách Sources

**File:** `src/retrieval/context/assembler.py:46-77`

```python
reordered = self._lost_in_middle_reorder(documents)      # thứ tự A
for i, doc in enumerate(reordered, 1):
    label = f"[Source {i}: {source}...]"                 # số theo thứ tự A
...
for doc in documents:                                    # ❌ thứ tự B (chưa reorder)
    ... sources.append(f"- {fname}")                      # không có số
```

**Vấn đề:** `SYSTEM_PROMPT` rule 3 yêu cầu LLM cite nguồn, và prompt đưa vào cả block
`### Context` (có nhãn `[Source 1..N]` theo thứ tự đã zigzag) lẫn block `### Sources`
(bullet list, **không số**, theo thứ tự relevance gốc). LLM viết "theo Source 3" nhưng
người đọc đối chiếu xuống danh sách Sources thì thấy thứ tự khác → citation không truy
vết được.

**Fix — đánh số nhất quán, build cả 2 block từ CÙNG list đã reorder** (đã tích hợp luôn
token budget của P2-10):

```python
def assemble(self, documents: list[dict]) -> tuple[str, str]:
    if not documents:
        return "", ""

    reordered = self._lost_in_middle_reorder(documents)

    context_parts, sources, seen = [], [], set()
    used_chars, dropped = 0, 0

    for i, doc in enumerate(reordered, 1):
        label = self._format_label(i, doc)
        block = f"{label}\n{doc['content']}"

        # ★ P2-10: token budget — bỏ cả block, không cắt giữa câu
        if used_chars + len(block) > settings.MAX_CONTEXT_CHARS and context_parts:
            dropped += 1
            continue

        context_parts.append(block)
        used_chars += len(block)

        key = self._source_key(doc)
        if key not in seen:
            seen.add(key)
            sources.append(f"[{i}] {self._format_source(doc)}")   # ★ CÙNG số i

    if dropped:
        logger.warning("Context truncated by budget",
                       dropped_docs=dropped, kept=len(context_parts),
                       budget=settings.MAX_CONTEXT_CHARS, used=used_chars)

    context_text = "\n\n---\n\n".join(context_parts)
    sources_text = "\n".join(sources)
    logger.info("Context assembled", chunks=len(context_parts), total_chars=len(context_text))
    return context_text, sources_text


@staticmethod
def _format_source(doc: dict) -> str:
    parts = [doc.get("file_name") or "Unknown"]
    if doc.get("section_title"):
        parts.append(doc["section_title"])
    if doc.get("page_number") is not None:
        parts.append(f"p.{doc['page_number']}")
    return " → ".join(parts)

@staticmethod
def _source_key(doc: dict) -> str:
    return f"{doc.get('file_name','')}:{doc.get('section_title','')}:{doc.get('page_number')}"


def _format_label(self, i: int, doc: dict) -> str:
    return f"[Source {i}: {self._format_source(doc)}]"
```

Và sửa `SYSTEM_PROMPT` (`src/retrieval/prompts.py:18`) cho rõ ràng hơn:

```python
3. Always cite sources inline using the bracket number from the context, e.g. "[2]".
   Only cite numbers that actually appear in the provided context.
```

**Ghi chú về `_lost_in_middle_reorder`** (dòng 82-99): logic đúng như docstring mô tả
(`[1,2,3,4,5]` → `[1,3,5,4,2]`). **Không phải bug, đừng sửa.** Chỉ lưu ý: hiệu quả của kỹ
thuật này với context ngắn (5 parent × 2000 chars ≈ 2.5k token) là **không đáng kể** —
paper "Lost in the Middle" đo trên context 4k-16k+ token. Nên giữ (đúng về mặt showcase
kỹ thuật) nhưng đừng kỳ vọng cải thiện đo được ở scale này.

---

## P2-10 🟡 Không có token budget cho context

**File:** `src/retrieval/context/assembler.py`

Hiện tại: `KEEP_TOP_K=5` × parent 2000 chars = ~10k chars ≈ 2.5k token → an toàn với mọi
model. **Nhưng không có guard nào**: tăng `KEEP_TOP_K` lên 10 hoặc `PARENT_CHUNK_SIZE`
lên 4000 qua `.env` → context 40k chars ≈ 10k token → vượt context window của model nhỏ,
hoặc tốn tiền bất ngờ. Lỗi sẽ là 400 từ API, không phải message hữu ích.

**Fix:** phần cắt theo budget đã nằm trong đoạn `assemble()` ở P2-1. Chỉ cần thêm config:

```python
# src/core/config.py
    MAX_CONTEXT_CHARS: int = 24_000     # ~6k token, an toàn cho model 8k+
```

**Liên quan — short-circuit khi context rỗng.** `retriever.query()` (dòng 104-110) vẫn
gọi LLM dù `context_text == ""`. LLM sẽ trả "I don't have enough information" (đúng nhờ
`SYSTEM_PROMPT` rule 2) nhưng ta vừa tốn 1 LLM call vô ích. Trả thẳng message cho nhanh:

```python
# src/retrieval/retriever.py
NO_CONTEXT_MSG = ("I don't have enough information in the company documents "
                  "to answer this question.")

# trong query(), sau bước ⑦:
if not context_text.strip():
    logger.warning("Empty context — skipping LLM call", query=user_query[:80])
    return iter([NO_CONTEXT_MSG]) if stream else NO_CONTEXT_MSG
```

---

## P2-9 🟡 BM25 tokenizer thất bại đúng ca dùng mà docstring quảng cáo

**File:** `src/retrieval/search/sparse.py:61, 111`

```python
query_tokens = query.lower().split()      # dòng 61
corpus.append(content.lower().split())    # dòng 111
```

Docstring dòng 12-13 khẳng định:

> Ưu điểm: Chính xác với từ khóa, mã sản phẩm, tên riêng
> `"TC-456"` → BM25 tìm CHÍNH XÁC document chứa `"TC-456"`

**Nhưng `.split()` không tách dấu câu**, nên:

| Trong document | Token thực tế | Query `"TC-456"` khớp? |
|---|---|---|
| `ticket TC-456 was closed` | `tc-456` | ✅ |
| `see TC-456.` | `tc-456.` | ❌ **không khớp** |
| `(TC-456)` | `(tc-456)` | ❌ |
| `TC-456,` | `tc-456,` | ❌ |

Trong văn bản thật, mã hiệu gần như luôn dính dấu câu → tính năng chủ đạo của BM25 fail
im lặng. Thêm nữa: tiếng Việt / CJK không có space phân từ → `.split()` cho token vô
nghĩa, dù `SYSTEM_PROMPT` rule 6 hứa "Answer in the same language as the question".

**Fix — tokenizer regex giữ nguyên mã hiệu, loại dấu câu:**

```python
# src/retrieval/search/sparse.py
import re

_TOKEN_RE = re.compile(r"[0-9a-z]+(?:[-_][0-9a-z]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenizer giữ được mã kiểu 'TC-456' kể cả khi dính dấu câu."""
    return _TOKEN_RE.findall(text.lower())
```

Dùng ở cả 2 chỗ:

```python
query_tokens = tokenize(query)          # thay dòng 61
if not query_tokens:                    # ★ query toàn stopword/dấu câu → khỏi scoring
    return []
...
corpus.append(tokenize(content))        # thay dòng 111
```

**Cập nhật docstring** dòng 12-13 — hoặc giữ nguyên (giờ nó đúng rồi sau fix), hoặc thêm
dòng giới hạn:

```
Giới hạn: chỉ hoạt động tốt với ngôn ngữ có space phân từ (Anh, Việt có dấu cách).
CJK (Trung/Nhật/Hàn) cần tokenizer riêng — chưa hỗ trợ.
```

**Nếu cần hỗ trợ tiếng Việt/CJK thật** — ghi rõ giới hạn vào README, hoặc dùng dense
search làm chủ lực và đổi sang sparse embedding học được (SPLADE / BM42 của Qdrant) thay
cho BM25 thuần. Đừng để docstring hứa nhiều hơn code làm.

**Ghi chú về `if score > 0` (dòng 74):** filter này khiến `sparse_results` có thể ngắn
hơn `dense_results` rất nhiều (query toàn stopword → rỗng). RRF xử lý được list ngắn, nên
**không phải bug** — nhưng nên log để biết khi sparse "im lặng".

---

## P3-4 🔵 Reranker mutate list của caller

**File:** `src/retrieval/reranking/cross_encoder.py:74-78`

```python
for doc, score in zip(documents, scores):
    doc["rerank_score"] = float(score)      # mutate dict của caller
documents.sort(key=..., reverse=True)        # mutate LIST của caller
```

`documents` chính là `unique` trong `retriever.query()` — sau khi gọi `rerank()`, biến
`unique` ở caller đã bị **sắp xếp lại và thêm field**. Hiện chưa gây bug vì caller không
dùng lại `unique`, nhưng đây là bẫy: ai đó thêm log "top RRF chunk" **sau** dòng rerank sẽ
nhận số sai mà không hiểu tại sao (và PR 2 vừa thêm đúng loại log đó ở P1-3).

**Fix — copy rồi mới sort:**

```python
    def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        if not documents:
            return []

        pairs = [(query, doc["content"]) for doc in documents]
        scores = self.model.predict(pairs)

        # ★ KHÔNG mutate list/dict của caller
        scored = [{**doc, "rerank_score": float(s)} for doc, s in zip(documents, scores)]
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        top_docs = scored[:settings.KEEP_TOP_K]

        logger.info("Reranking done",
                    input_count=len(documents), output_count=len(top_docs),
                    top_score=round(top_docs[0]["rerank_score"], 4) if top_docs else 0)
        return top_docs
```

> Phần **hiệu năng** của reranker (cap candidates, `device`, `max_length`, `batch_size`,
> lock, `lru_cache`) thuộc [PR 4 / P1-8](pr4-api-performance.md#p1-8--reranker-chấm-80-cặp-trên-cpu-với-model-568m-params).
> PR này chỉ sửa phần mutate. Đoạn code ở PR 4 đã bao gồm cả fix này — nếu làm PR 4 sau
> thì merge tự nhiên, không xung đột.

---

## P3-5 🔵 Multi-query parse text thô

**File:** `src/retrieval/query_transform/multi_query.py:65-68`

```python
variants = [q.strip() for q in raw_output.split("\n") if q.strip()]
all_queries = [query] + variants[:settings.EXPAND_N_QUERY]
```

Prompt đã yêu cầu "Do NOT number the questions" / "Do NOT include the original question",
nhưng LLM (nhất là model nhỏ, hoặc `temperature=0.7`) thường vẫn trả:

```
Here are 3 alternative versions:
1. What is the company policy on laptop equipment?
2. How do employees receive their work computer?
```

→ `variants[0]` = `"Here are 3 alternative versions:"` → được **đưa thẳng vào search** như
một query thật, chiếm 1 trong 3 slot variant. Vừa tốn 1 lượt hybrid search (2 lần query
Qdrant + BM25 scoring toàn corpus), vừa bơm nhiễu vào RRF fusion.

**Fix — parse phòng thủ:**

```python
import re

_NUMBERING_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s*")
_PREAMBLE_RE = re.compile(
    r"^(here (are|is)|sure|certainly|below are|alternative|these are)\b", re.IGNORECASE
)


class MultiQueryExpander:
    ...

    def _parse_variants(self, raw_output: str, exclude: str) -> list[str]:
        """Làm sạch output LLM: bỏ numbering, preamble, dòng quá ngắn, trùng query gốc."""
        seen = {exclude.strip().lower()}
        variants = []
        for line in raw_output.splitlines():
            line = _NUMBERING_RE.sub("", line.strip()).strip().strip('"')
            if len(line) < 8:                    # quá ngắn để là câu hỏi
                continue
            if _PREAMBLE_RE.match(line):         # "Here are 3 versions:"
                continue
            if line.endswith(":"):               # dòng tiêu đề
                continue
            key = line.lower()
            if key in seen:                      # trùng query gốc hoặc variant trước
                continue
            seen.add(key)
            variants.append(line)
        return variants
```

Rồi trong `expand()` (thay dòng 65, giữ nguyên try/except đã thêm ở PR 1):

```python
    variants = self._parse_variants(raw_output, exclude=query)
    all_queries = [query] + variants[:settings.EXPAND_N_QUERY]
```

**Cân nhắc dài hạn:** dùng structured output (JSON mode / `response_format`) thay vì parse
text — nhưng phải kiểm tra endpoint OpenAI-compatible local có hỗ trợ không.

---

## Test cần thêm

```python
# tests/test_fusion.py
from retrieval.search.hybrid import rrf_fusion


def _doc(cid: str) -> dict:
    return {"chunk_id": cid, "content": f"content-{cid}"}


def test_rrf_sums_across_lists():
    """★ Bug P1-3: chunk ở NHIỀU list phải thắng chunk chỉ ở 1 list."""
    list_a = [_doc("A"), _doc("B")]
    list_b = [_doc("B"), _doc("C")]
    merged = rrf_fusion(list_a, list_b)
    assert merged[0]["chunk_id"] == "B"          # B ở cả 2 list
    assert merged[0]["n_hits"] == 2


def test_rrf_rank_starts_at_one():
    """★ Bug P2-8: công thức phải là 1/(k+1) cho hit đầu tiên."""
    merged = rrf_fusion([_doc("A")], k=60)
    assert merged[0]["rrf_score"] == 1.0 / 61


def test_rrf_empty_input():
    assert rrf_fusion() == []
    assert rrf_fusion([], []) == []


def test_rrf_does_not_mutate_input():
    list_a = [_doc("A")]
    rrf_fusion(list_a)
    assert "rrf_score" not in list_a[0]
```

```python
# tests/test_assembler.py
import re

from retrieval.context.assembler import ContextAssembler


def test_lost_in_middle_zigzag():
    docs = [{"chunk_id": str(i), "content": str(i), "file_name": "f"} for i in range(1, 6)]
    out = ContextAssembler()._lost_in_middle_reorder(docs)
    assert [d["content"] for d in out] == ["1", "3", "5", "4", "2"]


def test_reorder_short_lists_unchanged():
    for n in (0, 1, 2):
        docs = [{"content": str(i)} for i in range(n)]
        assert ContextAssembler()._lost_in_middle_reorder(docs) == docs


def test_source_numbers_match_context_labels():
    """★ Bug P2-1: số trong [Source N] phải khớp số trong danh sách Sources."""
    docs = [{"chunk_id": str(i), "content": f"c{i}",
             "file_name": f"f{i}.md", "section_title": f"S{i}"} for i in range(1, 6)]
    context, sources = ContextAssembler().assemble(docs)
    ctx_nums = re.findall(r"\[Source (\d+):", context)
    src_nums = re.findall(r"^\[(\d+)\]", sources, re.MULTILINE)
    assert ctx_nums == src_nums


def test_context_budget_drops_whole_blocks(monkeypatch):
    """★ P2-10: vượt budget thì bỏ cả block, không cắt giữa câu."""
    from core.config import settings
    monkeypatch.setattr(settings, "MAX_CONTEXT_CHARS", 200)
    docs = [{"chunk_id": str(i), "content": "x" * 150, "file_name": "f.md"}
            for i in range(5)]
    context, _ = ContextAssembler().assemble(docs)
    assert len(context) < 400            # chỉ giữ được 1 block
    assert "x" * 150 in context          # block được giữ nguyên vẹn


def test_empty_documents():
    assert ContextAssembler().assemble([]) == ("", "")
```

```python
# tests/test_tokenize.py
from retrieval.search.sparse import tokenize


def test_keeps_product_codes_with_punctuation():
    """★ Bug P2-9: mã hiệu dính dấu câu vẫn phải khớp."""
    assert "tc-456" in tokenize("See TC-456.")
    assert "tc-456" in tokenize("(TC-456), closed")
    assert tokenize("See TC-456.") == tokenize("see tc-456")


def test_empty_and_punctuation_only():
    assert tokenize("") == []
    assert tokenize("...!?") == []
```

```python
# tests/test_multi_query.py
from retrieval.query_transform.multi_query import MultiQueryExpander


def test_parse_variants_strips_preamble_and_numbering():
    """★ Bug P3-5: bỏ preamble và numbering khỏi variant."""
    exp = MultiQueryExpander.__new__(MultiQueryExpander)   # không cần LLM
    raw = ("Here are 3 alternative versions:\n"
           "1. What is the laptop policy?\n"
           "2. How do employees get a work computer?\n"
           "- Which devices does the company issue?\n")
    out = exp._parse_variants(raw, exclude="laptop policy")
    assert out == [
        "What is the laptop policy?",
        "How do employees get a work computer?",
        "Which devices does the company issue?",
    ]


def test_parse_variants_excludes_original():
    exp = MultiQueryExpander.__new__(MultiQueryExpander)
    out = exp._parse_variants("What is the laptop policy?\nSomething else entirely",
                              exclude="What is the laptop policy?")
    assert out == ["Something else entirely"]
```

---

## Kiểm chứng

```bash
make test                               # 4 file test mới xanh
make evaluate 2>&1 | tail -30           # ghi lại avg_keyword_overlap
```

**So sánh trước/sau:** chạy `make evaluate` **trước** khi làm PR 2, lưu lại
`avg_keyword_overlap` và `has_context_ratio`, rồi so với sau. Kỳ vọng: bằng hoặc tốt hơn.
Nếu tệ đi đáng kể → khả năng cao `rrf_fusion` bị gọi sai chỗ (ví dụ fuse cả list đã fuse).

**Kiểm tra citation bằng mắt** — hỏi 1 câu qua `make run-ui`, xem answer có cite `[2]` và
số đó có trong danh sách Sources không.

**Kiểm tra fusion hoạt động** — bật `LOG_LEVEL=DEBUG` (nếu đã làm PR 5) hoặc đọc log
`"Fusion done"`: `top_n_hits` phải > 1 với query thường (chứng tỏ nhiều query cùng tìm
được chunk đó). Nếu luôn `= 1` → các expanded query đang trả về tập kết quả rời rạc hoàn
toàn, đáng nghi.
