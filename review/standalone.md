# Hai việc đứng riêng ngoài PR 1-5

Hai issue này không xếp vào PR nào, vì lý do khác nhau:

| Issue | Lý do đứng riêng |
|---|---|
| [**P0-2**](#p0-2--gemini-provider-gọi-api-surface-không-tồn-tại) — Gemini provider | **Bị chặn**: phải cài được `google-genai` để xác minh signature trước khi sửa. Không thể sửa "mù". |
| [**P3-6**](#p3-6--chat-history-nhận-vào-nhưng-bị-bỏ-qua) — Multi-turn | **Feature mới**, không phải fix bug. Chỉ làm sau khi PR 1-5 xong. |

---

## P0-2 🔴 Gemini provider gọi API surface không tồn tại

> **Mức:** Blocker cho `LLM_PROVIDER=gemini`. **Không** ảnh hưởng `LLM_PROVIDER=openai`
> (mặc định) — nên không chặn PR 1-5.
>
> **Trạng thái:** ⚠️ **CẦN KIỂM CHỨNG** — không cài được `google-genai` trong môi trường
> review nên không chạy thử được. Kết luận dưới đây dựa trên đọc code + đối chiếu naming
> convention của 2 SDK Google.

**File:** `src/core/llm.py:135-180`

```python
interaction = self._client.interactions.create(
    model=..., input=user_prompt,
    system_instruction=system_prompt,
    generation_config={"temperature": ..., "max_output_tokens": ...},
)
return interaction.output_text
```

```python
# generate_stream, dòng 165-180
stream = self._client.interactions.create(..., stream=True)
for event in stream:
    if event.event_type == "step.delta":
        if event.delta.type == "text":
            yield event.delta.text
```

### Bốn dấu hiệu code này sai

1. **`generation_config` (dict) là naming của SDK cũ** `google-generativeai`. SDK
   `google-genai` dùng `config=types.GenerateContentConfig(...)` — typed object, không phải
   dict với key `generation_config`.
2. **`system_instruction` trong `google-genai` nằm BÊN TRONG config**, không phải kwarg
   top-level.
3. **Streaming trong `google-genai` là method riêng** (`generate_content_stream`), không
   phải `create(stream=True)`.
4. **`interaction.output_text` / `event.event_type == "step.delta"` / `event.delta.type`**
   — không khớp shape response của `generate_content`.

### Dấu hiệu thứ 5 — code chưa từng được chạy

`requirements.txt:9` pin `google-genai==2.15.0` nhưng **`pyproject.toml` không khai báo
`google-genai` gì cả** (xem [PR 5 / P2-7](pr5-eval-deps-docs.md#p2-7--pyprojecttoml--requirementstxt-lệch-nhau-pyproject-không-cài-được)).
Kết hợp với `.env` thực tế **không có** `LLM_PROVIDER` (mặc định về `openai`) và không có
`GEMINI_API_KEY` → nhánh Gemini gần như chắc chắn chưa từng thực thi.

---

### Bước 1 — Xác minh TRƯỚC khi sửa

```bash
pip install "google-genai==2.15.0"
python -c "
from google import genai
import inspect
c = genai.Client(api_key='dummy-key-for-introspection')
print('has interactions:', hasattr(c, 'interactions'))
print('has models     :', hasattr(c, 'models'))
if hasattr(c, 'interactions'):
    print('interactions.create:', inspect.signature(c.interactions.create))
print('models.generate_content:', inspect.signature(c.models.generate_content))
print('models has stream method:',
      [m for m in dir(c.models) if 'stream' in m])
"
```

Ghi lại output — nó quyết định đi nhánh 2a hay 2b.

---

### Bước 2a — Nếu `interactions` KHÔNG tồn tại (khả năng cao)

Viết lại theo API chuẩn `client.models.generate_content`:

```python
class GeminiLLMService(BaseLLMService):
    """LLM provider dùng Google Gemini qua google-genai SDK.

    Khác biệt so với OpenAI:
      - system_prompt → system_instruction (nằm trong GenerateContentConfig)
      - user_prompt → contents
      - Response text: resp.text
    """

    def __init__(self):
        from google import genai

        kwargs = {}
        if settings.GEMINI_API_KEY:
            kwargs["api_key"] = settings.GEMINI_API_KEY

        self._client = genai.Client(**kwargs)
        self._model = settings.GEMINI_MODEL_ID
        logger.info("Gemini LLM initialized", model=self._model)

    def _build_config(self, system_prompt: str, temperature: float,
                      max_tokens: int | None):
        from google.genai import types

        cfg = {"temperature": temperature}
        if max_tokens is not None:
            cfg["max_output_tokens"] = max_tokens
        if system_prompt:
            cfg["system_instruction"] = system_prompt
        return types.GenerateContentConfig(**cfg)

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        resp = self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=self._build_config(system_prompt, temperature, max_tokens),
        )
        text = resp.text
        if not text:
            # ★ Cùng lý do như P1-4 ở PR 1: safety filter / MAX_TOKENS → text rỗng
            logger.warning("Gemini returned empty text",
                           candidates=len(resp.candidates or []))
            return ""
        return text.strip()

    def generate_stream(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,      # ★ bản gốc THIẾU — stream không giới hạn độ dài
    ):
        stream = self._client.models.generate_content_stream(
            model=self._model,
            contents=user_prompt,
            config=self._build_config(system_prompt, temperature, max_tokens),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
```

**Xoá luôn** dòng docstring `llm.py:121` ("Bonus: Multi-turn tự động bằng
previous_interaction_id") — tính năng đó thuộc Interactions API, không áp dụng cho
`generate_content`. Multi-turn được xử lý riêng ở [P3-6](#p3-6--chat-history-nhận-vào-nhưng-bị-bỏ-qua)
bằng query condensation.

Sửa docstring đầu file `llm.py:8, 11-12` — đang ghi "Gemini Interactions API".

---

### Bước 2b — Nếu `interactions` CÓ tồn tại

Giữ nó, nhưng **sửa cho khớp signature thật** mà bước 1 in ra. Cụ thể phải kiểm:

- [ ] Tên kwarg cho config: `generation_config` (dict) hay `config` (typed object)?
- [ ] `system_instruction` là top-level kwarg hay nằm trong config?
- [ ] Attribute lấy text: `output_text`, `text`, hay `outputs[-1].text`?
      (docstring dòng 119 ghi `interaction.outputs[-1].text` nhưng code dòng 155 dùng
      `interaction.output_text` — **hai chỗ trong cùng file đã không khớp nhau**, dấu hiệu
      rõ ràng là chưa chạy thử)
- [ ] Streaming: `create(stream=True)` hay method riêng?
- [ ] Shape của stream event: `event.event_type` / `event.delta.type` / `event.delta.text`?

**Bắt buộc:** phải chạy thành công smoke test ở bước 3 trước khi coi là xong.

---

### Bước 3 — Smoke test cho cả 2 provider

```python
# tests/manual_llm_smoke.py — chạy TAY, không vào CI (cần API key thật)
"""Smoke test LLM provider. Chạy: PYTHONPATH=src python tests/manual_llm_smoke.py"""

from core.config import settings
from core.llm import get_llm_service


def main():
    print(f"Provider: {settings.LLM_PROVIDER}")
    llm = get_llm_service()

    print("\n--- generate() ---")
    out = llm.generate("Say OK", system_prompt="Reply with exactly one word.")
    print(repr(out))
    assert out, "generate() trả về rỗng"

    print("\n--- generate_stream() ---")
    chunks = list(llm.generate_stream("Count from 1 to 5.", max_tokens=50))
    print(f"{len(chunks)} chunks:", repr("".join(chunks)))
    assert chunks, "generate_stream() không yield gì"

    print("\n✅ OK")


if __name__ == "__main__":
    main()
```

Chạy với cả 2 provider:

```bash
LLM_PROVIDER=openai PYTHONPATH=src python tests/manual_llm_smoke.py
LLM_PROVIDER=gemini PYTHONPATH=src python tests/manual_llm_smoke.py
```

> `LLM_PROVIDER` là env var nên pydantic-settings đọc được, override giá trị trong `.env`.
> Nhưng lưu ý `get_llm_service()` cache singleton (`llm.py:187`) — trong 1 process chỉ tạo
> được 1 provider. Chạy 2 lệnh riêng, đừng gộp.

---

### Trong lúc chưa xác minh được

**Đừng merge code Gemini "trông có vẻ đúng".** Thay vào đó:

1. `.env.example` để `LLM_PROVIDER=openai` (đã có trong
   [PR 4 / P2-6](pr4-api-performance.md#p2-6--env-trỏ-llm-vào-localhost8000--cùng-port-với-fastapi))
2. Thêm cảnh báo vào README (đã có trong
   [PR 5 / P3-7](pr5-eval-deps-docs.md#p3-7--readme--docstring-drift) mục Limitations)
3. Thêm cảnh báo runtime để không ai dùng nhầm mà tưởng nó hoạt động:

```python
# src/core/llm.py — trong get_llm_service()
        if provider == "gemini":
            logger.warning(
                "Provider 'gemini' CHƯA được test end-to-end. "
                "Xem review/standalone.md (P0-2) trước khi dùng trong production."
            )
            _llm_instance = GeminiLLMService()
```

**Vẫn làm được ngay trong PR 1** (không cần xác minh SDK): thêm `max_tokens` vào signature
`generate_stream` của `GeminiLLMService` cho khớp abstract base — xem
[PR 1 / P1-5](pr1-correctness.md#p1-5--chunkchoices0-crash-khi-provider-gửi-usage-only-chunk).

---
## P3-6 🔵 Chat history nhận vào nhưng bị bỏ qua

> **Mức:** Low. Đây là **feature mới**, không phải fix bug. Làm sau PR 1-5.

**File:** `src/api/ui.py:34-48`

```python
def respond(message: str, chat_history: list):
    response = ""
    for token in chat_stream(message):     # ❌ chat_history không được dùng
        response += token
        yield response
```

**Triệu chứng:** Mỗi câu hỏi là một phiên độc lập. Hội thoại tự nhiên bị vỡ:

```
User: How many days of annual leave do employees get?
Bot:  15 days per year...
User: Can I carry them over?              ← "them" = gì? Bot không biết
Bot:  I don't have enough information...
```

Đây là hạn chế **thiết kế**, không phải bug — `api/chat.py` docstring dòng 7 đã ghi
"Quản lý conversation history (nếu cần)" như việc chưa làm. Nhưng với một chatbot thì
follow-up question là kịch bản dùng cơ bản nhất, và README mô tả sản phẩm là "chatbot".

### Fix — Query Condensation

Kỹ thuật chuẩn cho conversational RAG: dùng LLM viết lại câu hỏi follow-up thành câu hỏi
độc lập **trước khi** vào retrieval.

```python
# src/retrieval/query_transform/condense.py (file mới)
"""
Query Condensation — Viết lại câu hỏi follow-up thành câu hỏi độc lập.

Vấn đề: "Can I carry them over?" — retrieval không hiểu "them" là gì.
Giải pháp: dùng LLM + history → "Can employees carry over unused annual leave?"

Chạy TRƯỚC Multi-Query Expansion trong pipeline:
  history + question → [Condense] → standalone question → [Multi-Query] → ...
"""

from __future__ import annotations

from core import get_logger
from core.llm import get_llm_service

logger = get_logger(__name__)

CONDENSE_PROMPT = """Given the conversation history and a follow-up question,
rewrite the follow-up question to be a standalone question that makes sense
without the history. Resolve all pronouns and references.

If the follow-up question is already standalone, return it UNCHANGED.
Return ONLY the rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""

MAX_HISTORY_TURNS = 3


class QueryCondenser:
    """Viết lại câu hỏi follow-up thành câu hỏi độc lập."""

    def __init__(self):
        self.llm = get_llm_service()

    def condense(self, question: str, history: list[tuple[str, str]]) -> str:
        if not history:
            return question

        recent = history[-MAX_HISTORY_TURNS:]
        history_text = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in recent)
        try:
            rewritten = self.llm.generate(
                user_prompt=CONDENSE_PROMPT.format(history=history_text, question=question),
                temperature=0.0, max_tokens=150,
            ).strip()
        except Exception as e:
            # Cùng nguyên tắc graceful degradation như PR 1 / P1-7
            logger.warning("Condensation failed, using original question", error=str(e))
            return question

        if not rewritten or len(rewritten) > 500:
            logger.warning("Condensation output rejected, using original",
                           length=len(rewritten))
            return question
        logger.info("Query condensed", original=question[:60], rewritten=rewritten[:60])
        return rewritten
```

### Truyền history qua các layer

```
ui.respond(message, chat_history)
  → chat_stream(message, history)
    → retriever.query(query, history=...)
      → condenser.condense(query, history)   ← TRƯỚC bước ①
```

Chữ ký cần đổi:

```python
# src/retrieval/retriever.py
    def retrieve(self, user_query: str, history: list[tuple[str, str]] | None = None):
        # ⓿ Condense — resolve đại từ/tham chiếu trước khi retrieval
        search_query = self.condenser.condense(user_query, history or [])
        expanded_queries = self.expander.expand(search_query)
        hyde_vector = self.hyde.generate_embedding(search_query)
        ...
        # ★ LƯU Ý: dùng search_query cho retrieval, nhưng user_query cho generate
        #   (để LLM trả lời đúng câu người dùng vừa hỏi, giữ ngữ cảnh tự nhiên)
```

```python
# src/api/chat.py
def chat_stream(query: str, history: list | None = None):
    ...
    yield from get_retriever().query(query, stream=True,
                                     history=_normalize_history(history or []))
```

### Chuẩn hoá format history của Gradio

Format khác nhau giữa các version (tuples ở 4.x, `messages` dict ở 5.x+) — xem
[PR 5 / P2-7](pr5-eval-deps-docs.md#p2-7--pyprojecttoml--requirementstxt-lệch-nhau-pyproject-không-cài-được):

```python
# src/api/ui.py
def _normalize_history(chat_history: list) -> list[tuple[str, str]]:
    """Gradio 4.x: [[user, bot], ...] | Gradio 5.x+: [{'role','content'}, ...]"""
    if not chat_history:
        return []
    if isinstance(chat_history[0], dict):
        pairs, pending = [], None
        for msg in chat_history:
            if msg.get("role") == "user":
                pending = msg.get("content", "")
            elif msg.get("role") == "assistant" and pending is not None:
                pairs.append((pending, msg.get("content", "")))
                pending = None
        return pairs
    return [(u, a) for u, a in chat_history if u and a]


def respond(message: str, chat_history: list):
    """Xử lý message từ user, trả về streaming response."""
    history = _normalize_history(chat_history)
    response = ""
    for token in chat_stream(message, history=history):
        response += token
        yield response
```

**Và khai báo rõ format trong `create_ui()`** để không phụ thuộc default của version:

```python
    demo = gr.ChatInterface(
        fn=respond,
        type="messages",            # ★ chốt format
        title="🤖 RAG Chatbot — Internal Knowledge Base",
        description=...,
        examples=EXAMPLE_QUESTIONS,
        cache_examples=False,
    )
```

### Test

```python
# tests/test_condense.py
from retrieval.query_transform.condense import QueryCondenser


class _FakeLLM:
    def __init__(self, out): self.out = out
    def generate(self, **kw): return self.out


def _condenser(out: str) -> QueryCondenser:
    c = QueryCondenser.__new__(QueryCondenser)
    c.llm = _FakeLLM(out)
    return c


def test_no_history_returns_original():
    c = _condenser("SHOULD NOT BE USED")
    assert c.condense("What is X?", []) == "What is X?"


def test_resolves_pronoun():
    c = _condenser("Can employees carry over unused annual leave?")
    out = c.condense("Can I carry them over?",
                     [("How many days of annual leave?", "15 days per year.")])
    assert "annual leave" in out


def test_llm_failure_falls_back_to_original():
    class _Boom:
        def generate(self, **kw): raise RuntimeError("rate limit")
    c = QueryCondenser.__new__(QueryCondenser)
    c.llm = _Boom()
    assert c.condense("Can I carry them over?", [("q", "a")]) == "Can I carry them over?"


def test_absurdly_long_output_rejected():
    c = _condenser("x" * 600)
    assert c.condense("short q", [("q", "a")]) == "short q"
```

```python
# tests/test_ui_history.py
from api.ui import _normalize_history


def test_gradio_5_messages_format():
    hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    assert _normalize_history(hist) == [("q1", "a1")]


def test_gradio_4_tuples_format():
    assert _normalize_history([["q1", "a1"]]) == [("q1", "a1")]


def test_empty():
    assert _normalize_history([]) == []


def test_dangling_user_message_ignored():
    """User vừa gửi, bot chưa trả lời → chưa thành 1 turn."""
    hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}]
    assert _normalize_history(hist) == [("q1", "a1")]
```

### Chi phí phải cân nhắc

Thêm **1 LLM call nữa** vào mỗi query (thành 4: condense + expand + hyde + generate). Với
latency đã 30-90s trên CPU ([PR 4 / P1-8](pr4-api-performance.md#p1-8--reranker-chấm-80-cặp-trên-cpu-với-model-568m-params)),
nên:

- Chỉ condense khi **có** history (đã làm — `if not history: return question`)
- Cân nhắc bỏ qua condense khi câu hỏi đã đủ dài và không có đại từ (heuristic rẻ trước khi
  gọi LLM)
- `temperature=0.0` + `max_tokens=150` → nhanh và deterministic

**Cập nhật README** — bỏ dòng "Multi-turn: chưa hỗ trợ" khỏi mục Limitations sau khi làm
xong.
