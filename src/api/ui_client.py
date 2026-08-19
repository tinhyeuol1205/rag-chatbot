"""Thin HTTP/SSE client used by the Gradio UI.

The UI process deliberately knows only the FastAPI contract.  It must not import
the retriever, Qdrant client, Redis admission queue, or model runtimes because
those boundaries are different between the development and product demo modes.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from core.config import settings

_ALLOWED_SOURCE_FIELDS = ("citation_id", "file_name", "section_title", "page_number")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class UIClientError(RuntimeError):
    """Safe error raised when the UI cannot use the FastAPI contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "ui_api_error",
        status_code: int | None = None,
        retry_after: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass(frozen=True)
class ReadyState:
    """Subset of ``/ready`` that the UI is allowed to inspect."""

    status: str
    app_env: str
    execution_mode: str
    embedding_runtime: str
    reranker_runtime: str
    worker_available: bool

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ReadyState:
        return cls(
            status=str(payload.get("status", "")),
            app_env=str(payload.get("app_env", "")),
            execution_mode=str(payload.get("execution_mode", "")),
            embedding_runtime=str(payload.get("embedding_runtime", "")),
            reranker_runtime=str(payload.get("reranker_runtime", "")),
            worker_available=bool(payload.get("worker_available", False)),
        )


def normalize_history(
    history: Iterable[Any] | None,
    *,
    max_turns: int | None = None,
) -> list[tuple[str, str]]:
    """Convert Gradio tuple/message history into bounded user/assistant pairs.

    Incomplete turns are dropped because they cannot be used safely for query
    condensation.  Keeping only the most recent turns bounds every request sent
    from the UI, regardless of how much state a browser retains.
    """

    if not history:
        return []
    rows = list(history)
    pairs: list[tuple[str, str]] = []
    if isinstance(rows[0], Mapping):
        pending: str | None = None
        for message in rows:
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            content = _clean_text(message.get("content", ""))
            if role == "user" and content:
                pending = content
            elif role == "assistant" and content and pending is not None:
                pairs.append((pending, content))
                pending = None
    else:
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            user, assistant = _clean_text(row[0]), _clean_text(row[1])
            if user and assistant:
                pairs.append((user, assistant))
    if max_turns is not None and max_turns >= 0:
        return pairs[-max_turns:] if max_turns else []
    return pairs


def sanitize_sources(sources: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Allowlist citation fields and remove control characters before rendering."""

    safe: list[dict[str, Any]] = []
    for source in sources or []:
        if not isinstance(source, Mapping):
            continue
        file_name = _clean_text(source.get("file_name", ""))
        if not file_name:
            continue
        item: dict[str, Any] = {
            "citation_id": _safe_int(source.get("citation_id")),
            "file_name": file_name,
        }
        section = _clean_text(source.get("section_title", ""))
        if section:
            item["section_title"] = section
        page = _safe_int(source.get("page_number"))
        if page is not None:
            item["page_number"] = page
        safe.append({key: item[key] for key in _ALLOWED_SOURCE_FIELDS if key in item})
    return safe


class UIAPIClient:
    """Small synchronous client suitable for a Gradio callback thread."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        connect_timeout: float | None = None,
        request_timeout: float | None = None,
        sse_idle_timeout: float | None = None,
        client: httpx.Client | None = None,
    ):
        url = (base_url or settings.UI_API_BASE_URL).strip().rstrip("/")
        if not url:
            raise UIClientError("UI_API_BASE_URL is empty", code="ui_configuration_error")
        self.base_url = url
        self.api_key = settings.UI_API_KEY if api_key is None else api_key
        self.connect_timeout = connect_timeout or settings.UI_CONNECT_TIMEOUT_SECONDS
        self.request_timeout = request_timeout or settings.UI_REQUEST_TIMEOUT_SECONDS
        self.sse_idle_timeout = sse_idle_timeout or settings.UI_SSE_IDLE_TIMEOUT_SECONDS
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers=self._headers(),
            follow_redirects=False,
            timeout=httpx.Timeout(
                self.request_timeout,
                connect=self.connect_timeout,
                read=self.request_timeout,
            ),
        )

    def _headers(self, idempotency_key: str | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key.strip():
            headers["X-API-Key"] = self.api_key
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def ready(self) -> ReadyState:
        """Read and validate the safe readiness payload from FastAPI."""

        response = self._request("GET", "/ready")
        return ReadyState.from_payload(response)

    def preflight(self, expected_mode: str | None = None) -> ReadyState:
        """Fail before Gradio listens when the API topology is wrong."""

        mode = expected_mode or settings.UI_DEMO_MODE
        state = self.ready()
        if state.status != "ready":
            raise UIClientError("FastAPI chưa sẵn sàng phục vụ", code="ui_api_not_ready")
        expected_execution = "redis_worker" if mode == "product" else "inline"
        if state.execution_mode != expected_execution:
            raise UIClientError(
                f"UI profile {mode} cần FastAPI execution_mode={expected_execution}",
                code="ui_topology_mismatch",
            )
        if mode == "product":
            if state.embedding_runtime != "remote" or state.reranker_runtime != "remote":
                raise UIClientError(
                    "Product demo yêu cầu embedding và reranker runtime=remote",
                    code="ui_runtime_mismatch",
                )
            if not state.worker_available:
                raise UIClientError("Redis worker chưa có heartbeat", code="ui_worker_unavailable")
        return state

    def chat(
        self,
        query: str,
        history: Iterable[Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Call non-streaming `/chat` and return its JSON payload."""

        payload = self._chat_payload(query, history)
        return self._request(
            "POST",
            "/chat",
            json=payload,
            headers=self._headers(idempotency_key or uuid.uuid4().hex),
        )

    def stream(
        self,
        query: str,
        history: Iterable[Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield typed SSE events while the API/worker produces the answer."""

        payload = self._chat_payload(query, history)
        key = idempotency_key or uuid.uuid4().hex
        timeout = httpx.Timeout(
            self.sse_idle_timeout,
            connect=self.connect_timeout,
            read=self.sse_idle_timeout,
        )
        try:
            with self._client.stream(
                "POST",
                "/chat/stream",
                json=payload,
                headers=self._headers(key),
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise self._response_error(response)
                yield from _parse_sse(response.iter_lines())
        except httpx.TimeoutException as exc:
            raise UIClientError(
                "API không trả dữ liệu kịp thời",
                code="ui_timeout",
                status_code=504,
            ) from exc
        except httpx.HTTPError as exc:
            raise UIClientError(
                "Không thể kết nối tới FastAPI",
                code="ui_api_unavailable",
                status_code=503,
            ) from exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _chat_payload(self, query: str, history: Iterable[Any] | None) -> dict[str, Any]:
        cleaned_query = _clean_text(query)
        if not cleaned_query:
            raise UIClientError("Câu hỏi không được để trống", code="invalid_query")
        if len(cleaned_query) > settings.UI_MAX_INPUT_CHARS:
            raise UIClientError("Câu hỏi vượt quá giới hạn cho phép", code="invalid_query")
        messages: list[dict[str, str]] = []
        for user, assistant in normalize_history(
            history,
            max_turns=settings.UI_HISTORY_MAX_TURNS,
        ):
            messages.extend(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            )
        return {"query": cleaned_query, "history": messages}

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
            if response.status_code >= 400:
                raise self._response_error(response)
            payload = response.json()
            if not isinstance(payload, dict):
                raise UIClientError("FastAPI trả response không hợp lệ", code="ui_invalid_response")
            return payload
        except UIClientError:
            raise
        except httpx.TimeoutException as exc:
            raise UIClientError(
                "API không trả dữ liệu kịp thời",
                code="ui_timeout",
                status_code=504,
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise UIClientError(
                "Không thể kết nối hoặc đọc response từ FastAPI",
                code="ui_api_unavailable",
                status_code=503,
            ) from exc

    @staticmethod
    def _response_error(response: httpx.Response) -> UIClientError:
        code = "ui_api_error"
        message = "FastAPI từ chối yêu cầu"
        try:
            payload = response.json()
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
            if isinstance(detail, Mapping):
                code = str(detail.get("code", code))
                message = str(detail.get("message", message))
            elif isinstance(detail, str):
                message = detail
        except (ValueError, json.JSONDecodeError):
            pass
        return UIClientError(
            message,
            code=code,
            status_code=response.status_code,
            retry_after=response.headers.get("Retry-After"),
        )


def _parse_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Parse the small SSE subset emitted by FastAPI/SSE-Starlette."""

    event_name = "message"
    data_lines: list[str] = []
    event_id: str | None = None

    def flush() -> dict[str, Any] | None:
        nonlocal event_name, data_lines, event_id
        if not data_lines:
            event_name = "message"
            event_id = None
            return None
        raw = "\n".join(data_lines)
        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError:
            data = raw
        result: dict[str, Any] = {"event": event_name, "data": data}
        if event_id:
            result["id"] = event_id
        event_name = "message"
        data_lines = []
        event_id = None
        return result

    for line in lines:
        if line == "":
            event = flush()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value or "message"
        elif field == "data":
            data_lines.append(value)
        elif field == "id":
            event_id = value
    event = flush()
    if event is not None:
        yield event


def _clean_text(value: Any) -> str:
    return _CONTROL_CHARS.sub("", str(value or "")).strip()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
