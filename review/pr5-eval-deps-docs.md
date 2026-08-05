# PR 5 — Evaluation, dependencies, docs, tests

> **Mục tiêu:** Làm số đo evaluation tin được, bộ dependency reproducible, và tài liệu
> khớp code.
>
> **Phụ thuộc:** [PR 1-4](README.md) — P0-4 cần `retriever` đã ổn định; các con số đo
> chỉ có ý nghĩa sau khi PR 1-3 xong.
> **Cần re-ingest:** ❌ không.
> **Issues:** P0-4, P2-2, P2-7, P2-11, P3-1, P3-2, P3-3, P3-7
>
> **File bị sửa:**
> - `src/retrieval/retriever.py` (`RAGResult`, tách `retrieve()`)
> - `src/evaluation/evaluate.py` (dùng context thật, lưu JSON)
> - `src/evaluation/metrics.py` (ragas 0.2 API, judge LLM)
> - `requirements.txt`, `pyproject.toml` (chốt version)
> - `tests/` (tạo mới, gom test của PR 1-4)
> - **32 file** trong `src/` (di chuyển docstring)
> - `src/core/logger.py`, `src/core/config.py`
> - `README.md`
>
> **Definition of Done:**
> - [ ] `make evaluate` chạy xong không crash, in ra RAG Triad thật (không phải fallback)
> - [ ] `contexts` trong eval là parent chunk (~2000 chars), không phải child (~400)
> - [ ] `pip install -r requirements.txt` sạch trong venv mới
> - [ ] `make test` xanh với toàn bộ test của PR 1-5
> - [ ] Script kiểm docstring in `none ✅`

---

## P0-4 🔴 Evaluation đo sai pipeline → metrics vô nghĩa

**File:** `src/evaluation/evaluate.py:36-51`

```python
answer = retriever.query(question, stream=False)

expanded = retriever.expander.expand(question)      # ❌ biến KHÔNG dùng, tốn 1 LLM call
search_results = retriever.searcher.search(question) # ❌ pipeline KHÁC hoàn toàn
contexts = [r["content"] for r in search_results[:5]]
```

**Ba vấn đề chồng nhau:**

1. **`expanded` là dead variable** → mỗi câu hỏi tốn thêm 1 LLM call vô ích (10 câu = 10 call).
2. **`contexts` không phải context thật đã đưa vào LLM.** `answer` được sinh từ:
   multi-query → HyDE → hybrid → **rerank** → **parent resolution** → assemble.
   Còn `contexts` chỉ là: hybrid search 1 query, **không rerank, không parent**.
   → `contexts` là **child chunk 400 ký tự**, trong khi LLM thật nhận **parent chunk 2000 ký tự**.
3. **Hệ quả:** `faithfulness` (answer có grounded trong context không) và `context_precision`
   đo trên context mà LLM **chưa từng thấy** → điểm thấp giả tạo. Số đo không dùng để
   ra quyết định được, mà đây lại chính là mục đích của module evaluation.

**Fix — cho retriever trả về context thật.** Thêm dataclass kết quả:

```python
# src/retrieval/retriever.py
from dataclasses import dataclass, field


@dataclass
class RAGResult:
    """Kết quả đầy đủ của 1 lượt RAG — dùng cho evaluation và debug."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    sources: str = ""
    expanded_queries: list[str] = field(default_factory=list)
    num_candidates: int = 0
```

Rồi tách `query()` thành 2 phần — retrieve (trả về context) và generate:

```python
    def retrieve(self, user_query: str) -> tuple[list[dict], str, str, list[str]]:
        """Chạy toàn bộ retrieval, trả về (resolved_docs, context_text, sources_text, expanded)."""
        # ① + ② — song song hoá ở PR 4
        expanded_queries = self.expander.expand(user_query)
        hyde_vector = self.hyde.generate_embedding(user_query)

        # ③ Hybrid search cho HyDE + mỗi expanded query
        result_lists = [self.searcher.search(user_query, hyde_vector=hyde_vector)]
        for eq in expanded_queries:
            result_lists.append(self.searcher.search(eq))

        # ④ RRF fusion trên tất cả (PR 2)
        unique = rrf_fusion(*result_lists)

        # ⑤ Rerank → ⑥ Parent resolution → ⑦ Assemble
        top_chunks = self.reranker.rerank(user_query, unique)
        resolved = self.parent_resolver.resolve(top_chunks)
        context_text, sources_text = self.assembler.assemble(resolved)
        return resolved, context_text, sources_text, expanded_queries

    def query(self, user_query: str, stream: bool = False):
        """Giữ nguyên interface cũ cho api/chat.py."""
        logger.info("RAG query started", query=user_query[:80])
        _, context, sources, _ = self.retrieve(user_query)

        if not context.strip():                        # PR 2 / P2-10
            logger.warning("Empty context — skipping LLM call", query=user_query[:80])
            return iter([NO_CONTEXT_MSG]) if stream else NO_CONTEXT_MSG

        if stream:
            return self._generate_stream(user_query, context, sources)
        return self._generate(user_query, context, sources)

    def query_with_context(self, user_query: str) -> RAGResult:
        """Dùng cho evaluation — trả về ĐÚNG context đã đưa vào LLM."""
        resolved, context, sources, expanded = self.retrieve(user_query)
        answer = (self._generate(user_query, context, sources)
                  if context.strip() else NO_CONTEXT_MSG)
        return RAGResult(
            answer=answer,
            contexts=[d["content"] for d in resolved],   # ★ context THẬT
            sources=sources,
            expanded_queries=expanded,
            num_candidates=len(resolved),
        )
```

`evaluate.py` gọn lại, chỉ chạy pipeline **1 lần**:

```python
def main():
    logger.info("Starting RAG evaluation", num_questions=len(EVAL_DATASET))

    retriever = RAGRetriever()
    results: list[EvalResult] = []

    for i, sample in enumerate(EVAL_DATASET, 1):
        question, ground_truth = sample["question"], sample["ground_truth"]
        logger.info(f"[{i}/{len(EVAL_DATASET)}] Evaluating", question=question[:60])
        try:
            result = retriever.query_with_context(question)
            results.append(EvalResult(
                question=question,
                answer=result.answer,
                ground_truth=ground_truth,
                contexts=result.contexts,
            ))
            logger.info(f"[{i}/{len(EVAL_DATASET)}] Done",
                        answer_preview=result.answer[:80],
                        num_contexts=len(result.contexts))
        except Exception as e:
            logger.exception(f"[{i}/{len(EVAL_DATASET)}] Failed")
            results.append(EvalResult(
                question=question, answer=f"ERROR: {e}",
                ground_truth=ground_truth, contexts=[],
            ))

    scores = evaluate_with_ragas(results)
    _print_report(results, scores)
    _save_results(results, scores)
```

**Bonus — ghi kết quả ra file** để so sánh giữa các lần tune (hiện chỉ `print`, không lưu
gì, không so sánh được trước/sau):

```python
import json
from pathlib import Path


def _save_results(results: list[EvalResult], scores: dict,
                  out_dir: str = "data/eval_runs") -> None:
    """Lưu kết quả để so sánh giữa các lần chạy."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "scores": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                   for k, v in scores.items()},
        "samples": [
            {"question": r.question, "answer": r.answer,
             "ground_truth": r.ground_truth, "num_contexts": len(r.contexts),
             "contexts": r.contexts}
            for r in results
        ],
    }
    path = Path(out_dir) / "latest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Eval results saved", path=str(path))
```

> Nhớ thêm `data/eval_runs/` vào `.gitignore` nếu không muốn commit kết quả chạy.

**Kiểm chứng nhanh** — sau fix, mỗi `contexts[i]` phải dài ~2000 ký tự (parent), không
phải ~400 (child):

```bash
make evaluate
python -c "
import json
d = json.load(open('data/eval_runs/latest.json'))
lens = [len(c) for s in d['samples'] for c in s['contexts']]
print('n_contexts:', len(lens), 'avg_len:', sum(lens)//max(len(lens),1))
"
# avg_len phải ~1500-2000 (parent chunk). Nếu ~400 → vẫn đang lấy child, fix chưa đúng.
```

---

## P2-2 🟡 RAGAS gọi theo API 0.1 nhưng requirements cho phép 0.2+

**File:** `src/evaluation/metrics.py:60-87`, `requirements.txt:36`

```python
data = {"question": [...], "answer": [...], "ground_truth": [...], "contexts": [...]}
dataset = Dataset.from_dict(data)
scores = evaluate(dataset=dataset, metrics=[context_precision, faithfulness, answer_relevancy])
return dict(scores)
```

⚠️ **CẦN KIỂM CHỨNG** (ragas chưa cài trong môi trường). Ba rủi ro:

1. **Tên cột đổi từ ragas 0.2:** `question`→`user_input`, `answer`→`response`,
   `contexts`→`retrieved_contexts`, `ground_truth`→`reference`. Với
   `requirements.txt: ragas>=0.1` (không chặn trên) pip sẽ cài bản mới nhất → schema
   validation fail.
2. **`except ImportError` không bắt được lỗi này.** Lỗi sẽ là `ValidationError` /
   `KeyError` / `TypeError` → **exception thoát ra**, giết luôn `make evaluate` sau khi
   đã chạy hết 10 câu hỏi (mất toàn bộ công sức, kể cả `_print_report` không kịp chạy).
3. **RAGAS mặc định dùng OpenAI làm judge**, đọc `OPENAI_API_KEY` từ env của **nó**, không
   biết `settings.OPENAI_BASE_URL`. Với `LLM_PROVIDER=gemini` → RAGAS vẫn cần OpenAI key.
   Với `.env` hiện tại (`OPENAI_API_KEY=EMPTY`, base_url local) → RAGAS gọi ra
   `api.openai.com` với key `"EMPTY"` → 401.

Thêm: `requirements.txt:38` pin `langchain-community<0.2.0` — pin rất cũ, gần như chắc
chắn xung đột với ragas hiện đại. Và `datasets>=2.0`: ragas 0.2 đã bỏ phụ thuộc `datasets`.

**Bước 0 — xác minh version trước khi sửa:**

```bash
pip install "ragas>=0.2,<0.3"
python -c "
import ragas; print('ragas', ragas.__version__)
from ragas import EvaluationDataset, SingleTurnSample
print('SingleTurnSample fields:', list(SingleTurnSample.model_fields))
from ragas.metrics import Faithfulness; print('Faithfulness OK')
"
```
**Fix — pin version + bắt exception rộng + inject judge LLM:**

```python
def evaluate_with_ragas(results: list[EvalResult]) -> dict:
    """Chạy RAGAS evaluation trên tập kết quả, fallback nếu thất bại."""
    if not results:
        return {}
    try:
        return _ragas_evaluate(results)
    except ImportError as e:
        logger.warning("RAGAS not installed, using simple fallback", error=str(e))
    except Exception:
        # ★ Bắt RỘNG: schema mismatch, judge API fail, timeout... đều phải fallback
        logger.exception("RAGAS evaluation failed, using simple fallback")
    return _simple_evaluate(results)


def _ragas_evaluate(results: list[EvalResult]) -> dict:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

    samples = [
        SingleTurnSample(
            user_input=r.question,
            response=r.answer,
            retrieved_contexts=r.contexts,
            reference=r.ground_truth,
        )
        for r in results
    ]
    dataset = EvaluationDataset(samples=samples)

    logger.info("Running RAGAS evaluation", num_samples=len(results))
    scores = evaluate(
        dataset=dataset,
        metrics=[ContextPrecision(), Faithfulness(), AnswerRelevancy()],
        llm=_build_judge_llm(),            # ★ dùng ĐÚNG provider của project
        embeddings=_build_judge_embeddings(),
    )
    logger.info("RAGAS evaluation complete")
    if hasattr(scores, "_repr_dict"):
        return {k: float(v) for k, v in scores._repr_dict.items()}
    return dict(scores)
```

**Judge LLM dùng chính config của project** (tránh phụ thuộc ngầm vào OpenAI):

```python
def _build_judge_llm():
    """Judge LLM = đúng provider trong .env. Trả None → để RAGAS dùng default."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    from core.config import settings

    if settings.LLM_PROVIDER.lower() != "openai":
        logger.warning(
            "RAGAS judge chỉ được cấu hình cho provider 'openai'. "
            "Đang dùng provider khác → RAGAS sẽ dùng default (cần OPENAI_API_KEY riêng).",
            provider=settings.LLM_PROVIDER,
        )
        return None
    kwargs = {"model": settings.OPENAI_MODEL_ID, "api_key": settings.OPENAI_API_KEY,
              "temperature": 0}
    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL
    return LangchainLLMWrapper(ChatOpenAI(**kwargs))


def _build_judge_embeddings():
    """answer_relevancy cần embeddings — dùng model local, khỏi tốn API."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from core.config import settings

    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_ID,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
        )
    )
```

**Về `_simple_evaluate` (dòng 90-116):** fallback này hoạt động đúng nhưng metric
`avg_keyword_overlap` là recall từ khoá thô — **không** đo được faithfulness hay
hallucination. Đừng để nó xuất hiện trong report dưới nhãn "RAG Triad":

```python
def _simple_evaluate(results: list[EvalResult]) -> dict:
    ...
    return {
        "_mode": "SIMPLE_FALLBACK (không phải RAG Triad — chỉ đo keyword overlap)",
        "num_samples": total,
        "avg_answer_length": sum(len(r.answer) for r in results) / total,
        "avg_keyword_overlap": sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0,
        "has_context_ratio": sum(1 for r in results if r.contexts) / total,
    }
```

Nếu quyết định **giữ ragas 0.1** thì phải pin chặt `ragas>=0.1,<0.2` và giữ code hiện
tại — nhưng ragas 0.1 đã EOL, không khuyến nghị.

---

## P2-7 🟡 `pyproject.toml` ⟷ `requirements.txt` lệch nhau; pyproject không cài được

**Bảng đối chiếu:**

| Package | `pyproject.toml` | `requirements.txt` | Code thực tế |
|---|---|---|---|
| `google-genai` | ❌ **thiếu** | `==2.15.0` | `llm.py:125` import |
| `FlagEmbedding` | `^1.2` | ❌ thiếu | **không dùng** (grep: 0 hit) |
| `gradio` | `^4.0` | `>=4.0` | docstring `ui.py:52` nói "Gradio 6.x API" |
| `ragas` | `^0.1` | `>=0.1` (không chặn trên) | code viết cho 0.1 |
| `langchain-community` | ❌ thiếu | `<0.2.0` | không import trực tiếp |
| `datasets` | `^2.0` | `>=2.0` | chỉ dùng trong metrics.py |

### Vấn đề nghiêm trọng nhất — `poetry install` sẽ FAIL

```toml
[tool.poetry]
name = "rag-chatbot"        # → poetry tìm package `rag_chatbot/` ở root
# ❌ không có `packages = [...]`, mà code nằm trong src/{core,ingestion,...}
```

Poetry không tìm thấy `rag_chatbot/` → lỗi. Project chạy được **chỉ nhờ** `PYTHONPATH=src`
trong Makefile, tức `pyproject.toml` hiện là **file trang trí**, chưa từng được dùng.

**Fix — chọn 1 trong 2, đừng giữ cả hai làm nguồn sự thật:**

**Phương án A (khuyến nghị cho project này) — giữ `requirements.txt`, đặt pyproject
non-package:**

```toml
[tool.poetry]
name = "rag-chatbot"
version = "0.1.0"
description = "Internal RAG Chatbot — Advanced RAG techniques showcase"
authors = ["Binh DV"]
readme = "README.md"
package-mode = false        # ★ project là app, không phải library để publish

[tool.ruff]
line-length = 120
target-version = "py310"    # ★ khớp python = ">=3.10" thay vì py311
```

Xoá phần `[tool.poetry.dependencies]` và `[tool.poetry.group.dev.dependencies]` để tránh
2 nguồn lệch nhau (dev deps đã có trong `requirements-dev.txt`).

**Chốt version trong `requirements.txt`:**

```
# Core utilities
pydantic>=2.7,<3
pydantic-settings>=2.2,<3
structlog>=24.1,<26

# LLM
openai>=1.30,<2
google-genai>=2.15,<3          # ★ khớp với llm.py

# Embeddings + Reranker (local)
sentence-transformers>=3.0,<6

# Vector Database
qdrant-client>=1.9,<2

# Document parsing
unstructured[pdf,docx,md]>=0.14,<0.19

# Text splitting
langchain-text-splitters>=0.2,<0.4

# Sparse search
rank-bm25>=0.2,<0.3

# API
fastapi>=0.111,<1
uvicorn[standard]>=0.30,<1
sse-starlette>=2.0,<4

# Chat UI
gradio>=4.0,<7

# Evaluation
ragas>=0.2,<0.3
langchain-openai>=0.2,<1
langchain-huggingface>=0.1,<1
```

**Bỏ hẳn:**
- `FlagEmbedding` — không import ở đâu (code dùng `sentence_transformers.CrossEncoder`)
- `langchain-community<0.2.0` — pin cũ gây xung đột với ragas
- `datasets` — ragas 0.2 không cần
- `python-dotenv` — pydantic-settings tự đọc `.env`
- `python-multipart` — không có endpoint upload file nào

**Ghi chú về `unstructured[pdf,docx,md]`:** đây là dependency **rất nặng** (kéo theo
onnxruntime, pdfminer, và tuỳ extras có thể cả detectron2) trong khi code chỉ dùng
`strategy="fast"` (`pdf_parser.py:22`) — tức không dùng layout model. Cân nhắc thay bằng
`pypdf` + `python-docx` (nhẹ hơn hàng trăm MB) nếu không cần OCR/table extraction. Nếu
giữ `unstructured`, ghi rõ vào README rằng `make install` sẽ tải rất nhiều.

**Phương án B — dùng Poetry thật:** thêm `packages = [{include = "core", from = "src"}, ...]`
cho từng package, bỏ `requirements.txt`, commit `poetry.lock`, sửa Makefile dùng
`poetry run`. Nhiều việc hơn, chỉ nên chọn nếu muốn CI reproducible.

**Kiểm chứng version thật:**

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt 2>&1 | tail -20
pip list | grep -Ei "gradio|ragas|qdrant|google-genai|sentence-transformers"
python -c "import gradio; print('gradio', gradio.__version__)"
```

**Về `gradio`:** docstring `ui.py:52` viết "Gradio 6.x API" nhưng dependency cho phép 4.x.
`gr.ChatInterface(fn, title, description, examples, cache_examples)` với `respond(message,
history)` hoạt động trên cả 4/5/6 — nhưng **format của `history`** khác nhau giữa các bản
(tuples ở 4.x mặc định, `messages` dict ở 5.x+). Code hiện bỏ qua `history` hoàn toàn
([P3-6](standalone.md#p3-6--chat-history-nhận-vào-nhưng-bị-bỏ-qua)) nên chưa vỡ. Nếu sau
này implement multi-turn, **phải** khai báo rõ `gr.ChatInterface(..., type="messages")`
và pin gradio.

---

## P2-11 🟡 Không có test nào, `make test` fail

**File:** `Makefile:50-51` — `pytest tests/ -v`, nhưng **`tests/` không tồn tại**
(đã kiểm tra) → exit code 4, `ERROR: file or directory not found: tests/`.

Toàn bộ logic thuật toán trong project là **hàm thuần, không cần network** — rất dễ test,
và chính là phần dễ sai nhất (P1-3, P2-8, P0-1 đều là bug logic thuần).

**Tạo `tests/conftest.py`:**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

**Gom test từ các PR trước** (nếu chưa tạo):

| File test | Từ PR | Kiểm cái gì |
|---|---|---|
| `test_parent_resolver.py` | [1](pr1-correctness.md#test-cần-thêm) | P0-1 fallback về child |
| `test_fusion.py` | [2](pr2-retrieval-quality.md#test-cần-thêm) | P1-3 sum RRF, P2-8 rank từ 1 |
| `test_assembler.py` | [2](pr2-retrieval-quality.md#test-cần-thêm) | P2-1 citation khớp, P2-10 budget |
| `test_tokenize.py` | [2](pr2-retrieval-quality.md#test-cần-thêm) | P2-9 mã hiệu dính dấu câu |
| `test_multi_query.py` | [2](pr2-retrieval-quality.md#test-cần-thêm) | P3-5 parse phòng thủ |
| `test_chunk_id.py` | [3](pr3-ingestion-schema.md#test-cần-thêm) | P2-5 UUID, không collision |
| `test_parent_child_chunking.py` | [3](pr3-ingestion-schema.md#test-cần-thêm) | P1-2 giữ metadata |
| `test_markdown_parser.py` | [3](pr3-ingestion-schema.md#test-cần-thêm) | `###`, code fence |

**Thêm test riêng cho PR này:**
```python
# tests/test_metrics_fallback.py
from evaluation.metrics import EvalResult, _simple_evaluate


def test_simple_evaluate_marks_itself_as_fallback():
    """★ P2-2: fallback KHÔNG được trông giống RAG Triad thật."""
    results = [EvalResult(question="q", answer="a b c", ground_truth="a b",
                          contexts=["ctx"])]
    out = _simple_evaluate(results)
    assert "_mode" in out
    assert "FALLBACK" in out["_mode"]


def test_simple_evaluate_empty():
    assert _simple_evaluate([]) == {}


def test_keyword_overlap_computed():
    results = [EvalResult(question="q", answer="alpha beta", ground_truth="alpha",
                          contexts=[])]
    out = _simple_evaluate(results)
    assert out["avg_keyword_overlap"] == 1.0      # "alpha" khớp hết ground_truth
```

```python
# tests/test_module_docstrings.py
"""★ P3-1: bảo vệ regression — docstring phải nằm TRƯỚC from __future__."""
import importlib
import pkgutil

import pytest

PACKAGES = ["core", "ingestion", "retrieval", "evaluation"]


def _iter_modules():
    for pkg in PACKAGES:
        for m in pkgutil.walk_packages([pkg], prefix=f"{pkg}."):
            yield m.name


@pytest.mark.parametrize("module_name", list(_iter_modules()))
def test_module_has_docstring(module_name):
    mod = importlib.import_module(module_name)
    assert (mod.__doc__ or "").strip(), (
        f"{module_name} mất docstring — kiểm tra 'from __future__ import annotations' "
        f"có bị đặt TRƯỚC docstring không (xem review/pr5-eval-deps-docs.md P3-1)"
    )
```

> `api` không có trong `PACKAGES` vì import `api.main` sẽ khởi tạo FastAPI app và có thể
> kéo theo model loading. Nếu muốn bao gồm, mock trước.

**Makefile** giữ nguyên — chỉ cần thực sự tạo `tests/`:

```makefile
test: ## Chạy unit tests
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v
```

---

## P3-1 🔵 32/44 file mất module docstring

**Pattern lặp ở 32 file** (đã đếm bằng script), ví dụ `core/config.py:1-14`:

```python
from __future__ import annotations       # ← dòng 1

"""
Central configuration — Tất cả settings đọc từ file .env
...
"""
```

**Vấn đề:** Python chỉ coi string literal là docstring khi nó là **statement đầu tiên**
của module. Ở đây statement đầu tiên là `from __future__ import annotations`, nên string
phía sau chỉ là một expression statement vô tác dụng → **`module.__doc__ is None`**.

Hệ quả: `help(core.config)` không hiện gì, Sphinx/pdoc/mkdocstrings không sinh được doc,
IDE không hiện tooltip. Với project portfolio mà docstring là điểm mạnh nhất (giải thích
RRF, HyDE, Lost-in-Middle rất kỹ) thì đây là mất mát đáng kể — công viết doc bị vô hiệu.

**Fix:** docstring **trước**, future import **sau**:

```python
"""
Central configuration — Tất cả settings đọc từ file .env
...
"""

from __future__ import annotations

from pathlib import Path
...
```

**Tìm các file bị ảnh hưởng:**

```bash
for f in $(find src -name "*.py"); do
  head -1 "$f" | grep -q "from __future__" && echo "$f"
done
```

**Kiểm chứng sau khi sửa:**

```bash
cd src && python -c "
import importlib, pkgutil, sys
sys.path.insert(0, '.')
bad = []
for pkg in ['core', 'ingestion', 'retrieval', 'evaluation']:
    for m in pkgutil.walk_packages([pkg], prefix=pkg + '.'):
        try:
            mod = importlib.import_module(m.name)
            if not (mod.__doc__ or '').strip():
                bad.append(m.name)
        except Exception as e:
            print('SKIP', m.name, type(e).__name__)
print('Modules without docstring:', bad or 'none ✅')
"
```

**Lưu ý:** đây là sửa cơ học trên 32 file. Có thể script hoá, nhưng **phải kiểm tra** file
nào docstring đã đúng vị trí sẵn (`ingestion/parsers/__init__.py` đúng rồi, không có
future import) và file nào có comment ở đầu. Đề xuất: sửa bán tự động rồi `git diff`
review từng file, đừng regex mù toàn repo. `test_module_docstrings.py` ở trên sẽ chặn
regression về sau.

---

## P3-2 🔵 `structlog.configure()` gọi lại mỗi lần `get_logger()`; không có log level

**File:** `src/core/logger.py:21-37`

```python
def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    structlog.configure(...)        # ❌ gọi LẠI mỗi lần import module mới
    return structlog.get_logger().bind(module=name)
```

**Ba vấn đề:**

1. **Configure lặp lại** — mỗi module import gọi `get_logger(__name__)` ở top-level → với
   ~20 module là ~20 lần `configure()`. Kết hợp `cache_logger_on_first_use=True` thì các
   lần sau không có tác dụng nhưng vẫn tốn công, và nếu code khác muốn config riêng thì bị
   ghi đè bất ngờ.
2. **Không có log level filtering** → `logger.debug()` luôn in ra. Không có cách tắt log
   khi chạy production hoặc khi chạy eval (log ngập terminal).
3. **`PrintLoggerFactory` in ra stdout** → trộn lẫn với output của Gradio/uvicorn và với
   `print()` trong `_print_report`. Nên đẩy log sang **stderr**, giữ stdout cho output
   chương trình.

**Fix:**

```python
"""
Structured logging với structlog.

Tại sao dùng structlog thay vì logging tiêu chuẩn?
- Log có cấu trúc key=value, dễ parse bằng máy (ELK, Datadog)
- Tự động thêm context (module name, log level)
- Output đẹp hơn trong terminal (có màu)

Usage:
    from core import get_logger
    logger = get_logger(__name__)
    logger.info("Processing document", file="report.pdf", chunks=5)
"""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def _configure() -> None:
    """Configure structlog ĐÚNG 1 LẦN cho cả process."""
    global _configured
    if _configured:
        return

    from core.config import settings

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.LOG_JSON
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),   # ★ timestamp để debug latency
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,          # ★ để logger.exception() in traceback
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),   # ★ level filtering
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr), # ★ stderr
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str):
    """Tạo logger instance, bind với tên module để biết log từ đâu."""
    _configure()
    return structlog.get_logger().bind(module=name)
```

Thêm vào `src/core/config.py`:

```python
    # --- Logging ---
    LOG_LEVEL: str = "INFO"     # DEBUG / INFO / WARNING / ERROR
    LOG_JSON: bool = False      # True → JSON output (cho ELK/Datadog)
```

> **`format_exc_info` là bắt buộc** cho `logger.exception()` dùng ở
> [PR 4 / P2-4](pr4-api-performance.md#p2-4--lỗi-nội-bộ-bị-trả-thẳng-ra-ui) — không có
> processor này thì traceback không được render.

**⚠️ Cẩn thận circular import:** `logger.py` giờ import `core.config`, mà `config.py` (sau
[PR 4 / P2-6](pr4-api-performance.md#vấn-đề-thứ-2--env-và-envexample-lệch-nhau-drift))
import `core.errors`. `errors.py` không import gì → **không có cycle**. Nhưng **đừng** để
`config.py` import `logger.py`. Nếu cần log trong config, dùng `warnings.warn`.

**Bỏ type annotation sai:** `-> structlog.stdlib.BoundLogger` không đúng kể cả ở code hiện
tại (`structlog.get_logger()` với `PrintLoggerFactory` không trả `stdlib.BoundLogger`), và
càng không đúng sau khi dùng `make_filtering_bound_logger`. Bỏ hẳn annotation.

---

## P3-3 🔵 Dead code

Đã grep xác nhận **0 usage**:

| Thứ | File | Xử lý |
|---|---|---|
| `RAGChatbotError`, `ConfigurationError`, `ParsingError`, `IngestionError`, `RetrievalError` | `core/errors.py` (cả file) | **Dùng** — [PR 1/P1-7](pr1-correctness.md#p1-7--không-có-tryexcept-nào-trong-retrieval--1-lỗi-llm-giết-cả-query), [PR 4/P2-4, P2-6](pr4-api-performance.md) |
| `recursive_chunk()` | `ingestion/chunking/recursive.py` | Giữ làm demo, ghi rõ trong docstring (xem dưới) |
| `FlagEmbedding` | `pyproject.toml:36` | **Xoá** (P2-7) |
| `from qdrant_client.models import models` | `core/db/qdrant.py:108` | **Xoá** — import rác trong hàm, không dùng |
| `expanded = retriever.expander.expand(...)` | `evaluation/evaluate.py:42` | **Xoá** (P0-4) |
| `Chunk.is_parent` | `ingestion/models.py:50` | Được set nhưng không bao giờ đọc — giữ để tự tài liệu hoá, hoặc xoá |
| `python-multipart`, `python-dotenv` | requirements | **Xoá** (P2-7) |

`errors.py` là trường hợp đáng chú ý nhất: file được viết công phu với 5 exception class
và docstring giải thích "tại sao cần custom exception", nhưng **không dòng code nào raise
hay catch chúng**. Sau PR 1 và PR 4 thì abstraction này mới có nghĩa.

**Ghi rõ `recursive_chunk` là demo, không phải dead code bị lãng quên:**

```python
"""
Recursive Character Chunking — Chiến lược chunking cơ bản.

⚠️ KHÔNG dùng trong ingestion pipeline — pipeline dùng parent_child_chunk().
Giữ lại để minh hoạ chiến lược chunking cơ bản (đối chiếu với Parent-Child).
...
"""
```
**Về `core/db/qdrant.py:98-115`** — ngoài import rác, `search()` còn nên nhận thêm
`query_filter` để về sau filter theo `file_name`/`file_type` (payload index đã có sau
[PR 3 / P1-6](pr3-ingestion-schema.md#p1-6--re-ingest-để-lại-chunk-rác-vĩnh-viễn)):

```python
def search(
    self,
    collection_name: str,
    query_vector: list[float],
    limit: int = 10,
    query_filter=None,               # ★ qdrant_client.models.Filter
    score_threshold: float | None = None,
) -> list:
    """Tìm kiếm vector tương đồng (cosine similarity).

    qdrant-client v1.18+: dùng query_points() thay vì search().
    """
    result = self.client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return result.points
```

---

## P3-7 🔵 README / docstring drift

| Nơi | Nội dung hiện tại | Thực tế |
|---|---|---|
| `README.md:45` | "LLM: OpenAI GPT-4o-mini" | Đã hỗ trợ **2 provider** (OpenAI + Gemini) — commit `a0a4828` chưa được phản ánh |
| `README.md:59` | `cp .env.example .env  # Fill in OPENAI_API_KEY` | `.env.example` giờ mặc định `LLM_PROVIDER=gemini` → hướng dẫn dẫn người dùng vào P0-2 |
| `README.md:53-70` | Quick Start không có bước `make run-api` | Chỉ có `run-ui`; không nói rõ UI gọi retriever **trực tiếp** (in-process), không qua FastAPI |
| `README.md` | không có mục Limitations | Nên ghi rõ (xem dưới) |
| `README.md` | không có Port map | Xem [PR 4 / P2-6](pr4-api-performance.md#p2-6--env-trỏ-llm-vào-localhost8000--cùng-port-với-fastapi) |
| `api/main.py:67` | "TTFT thường < 500ms" | Sai — retrieval chạy trước token đầu → tính bằng giây (PR 4 / P0-3) |
| `retriever.py:33` | "⑧ LLM Generation (GPT-4o-mini)" | Provider-agnostic rồi |
| `retriever.py:22-23` | "④ Deduplicate kết quả từ tất cả queries" | Giờ là RRF fusion cộng dồn (PR 2 / P1-3) |
| `ui.py:52` | "Gradio 6.x API" | Dependency cho phép `>=4.0` (P2-7) |
| `embeddings.py:32` | "Singleton-like... Load model 1 lần" | Sai — load nhiều lần (PR 4 / P1-1) |
| `models.py:60` | "idempotent khi re-ingest" | Chỉ đúng khi content không đổi (PR 3 / P1-6) |
| `sparse.py:12-13` | `"TC-456"` → tìm CHÍNH XÁC | Fail khi dính dấu câu (PR 2 / P2-9) |
| `sparse.py:41-42` | "Lần đầu gọi sẽ build index... các lần sau cached" | Thiếu: **không invalidate** sau ingest (PR 4 / P1-9) |
| `hybrid.py:71` | "Bước 1: Chạy **song song** 2 search engines" | Chạy **tuần tự** (dòng 74-80) — PR 4 sửa cho đúng |
| `hybrid.py:20-26` | ví dụ tính `1/(60+1)` | Code dùng rank từ 0 (PR 2 / P2-8) |
| `parent_child.py:20`, `hybrid.py:32`, v.v. | "Tham khảo: rag_master.md" | File `rag_master.md` **không có trong repo** → reference chết |

**Mục Limitations đề xuất thêm vào README:**

```markdown
## ⚠️ Limitations

- **Ngôn ngữ:** BM25 tokenizer chỉ hoạt động tốt với ngôn ngữ có dấu cách phân từ
  (Anh, Việt). CJK chưa hỗ trợ. Dense search thì đa ngôn ngữ bình thường.
- **Latency:** trên CPU, mỗi câu hỏi mất vài giây đến vài chục giây (2 LLM call +
  cross-encoder rerank). Xem `RERANKER_MODEL_ID` trong `.env` để đổi sang model nhẹ hơn.
- **Multi-turn:** chưa hỗ trợ. Mỗi câu hỏi là một phiên độc lập — câu hỏi follow-up dùng
  đại từ ("cái đó", "them") sẽ không được resolve.
- **BM25 index:** build 1 lần trong process. Nếu ingest ở terminal khác với server đang
  chạy, **phải restart server** để BM25 thấy dữ liệu mới. Dense search thì thấy ngay.
- **Scale:** BM25 giữ toàn bộ corpus trong RAM. Không phù hợp với > ~100k chunk.
- **Provider Gemini:** chưa được test end-to-end — xem `review/standalone.md`.
```

**Nguyên tắc:** khi sửa code theo report này, **sửa docstring kèm theo trong cùng commit**.
Docstring của project này là tài sản chính (portfolio) — docstring sai còn tệ hơn không có
docstring, vì người đọc tin nó.

`hybrid.py:71` là ví dụ điển hình: comment nói "song song" nên người đọc tưởng đã tối ưu,
thực tế tuần tự. PR 4 làm cho nó song song thật — nếu không làm thì phải sửa comment.

**Xử lý reference `rag_master.md`:** file này được nhắc ở nhiều docstring nhưng không có
trong repo. Hoặc (a) thêm file đó vào repo, hoặc (b) đổi thành reference tới paper gốc:

```python
# TRƯỚC:
Tham khảo: rag_master.md — Module 4, mục 4.2

# SAU:
Tham khảo: Cormack et al. (2009), "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" — https://dl.acm.org/doi/10.1145/1571941.1572114
```

---

## Kiểm chứng

```bash
# 1. Dependency sạch trong venv mới
python -m venv /tmp/verify-venv && . /tmp/verify-venv/bin/activate
pip install -r requirements.txt && pip install -r requirements-dev.txt
pip check                       # không có conflict

# 2. Test toàn bộ
make test                       # tất cả test của PR 1-5 xanh

# 3. Docstring
cd src && python -c "
import importlib, pkgutil, sys
sys.path.insert(0, '.')
bad = []
for pkg in ['core','ingestion','retrieval','evaluation']:
    for m in pkgutil.walk_packages([pkg], prefix=pkg+'.'):
        try:
            mod = importlib.import_module(m.name)
            if not (mod.__doc__ or '').strip(): bad.append(m.name)
        except Exception as e: print('SKIP', m.name, type(e).__name__)
print('Modules without docstring:', bad or 'none ✅')
"

# 4. Evaluation chạy thật
make evaluate 2>&1 | tail -40
```

**Dấu hiệu evaluation đã đúng:**
- Report in ra `context_precision`, `faithfulness`, `answer_relevancy` (số RAGAS thật),
  **không** có `_mode: SIMPLE_FALLBACK`
- `data/eval_runs/latest.json` tồn tại
- `avg_len` của contexts ~1500-2000 (parent chunk), không phải ~400 (child)
- Số LLM call giảm: trước là 3/câu (expand + hyde + generate) + 1 dead = 4; sau là 3

**Log level hoạt động:**

```bash
LOG_LEVEL=WARNING make evaluate 2>&1 | grep -c "\[info\]"
# phải in 0 — các dòng info bị filter
```
