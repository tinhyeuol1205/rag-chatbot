#!/usr/bin/env python3
"""Probe 9Router models through its OpenAI-compatible Chat Completions API.

The safe default checks only model IDs containing ``free``.  The script never
prints or writes the API key, prompt, or generated response content.

Examples:

    NINEROUTER_API_KEY=9router-local \
      PYTHONPATH=src rag/bin/python scripts/check_9router_models.py

    NINEROUTER_API_KEY=9router-local \
      PYTHONPATH=src rag/bin/python scripts/check_9router_models.py \
      --model oc/deepseek-v4-flash-free \
      --model bzl/auto:free

    NINEROUTER_API_KEY=9router-local \
      PYTHONPATH=src rag/bin/python scripts/check_9router_models.py \
      --match 'deepseek.*v4.*flash' --output artifacts/9router-model-check.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:20128/v1"
DEFAULT_MATCH = "free"
PROBE_PROMPT = "Reply with exactly: OK"


@dataclass(frozen=True)
class ModelCheck:
    model: str
    status: str
    http_status: int | None
    latency_ms: float
    error: str


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _safe_error(response: httpx.Response) -> str:
    """Extract a short provider error without retaining response content."""
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return f"non-JSON response ({response.reason_phrase})"

    value: Any = body
    if isinstance(body, dict) and "error" in body:
        value = body["error"]
    if isinstance(value, dict):
        message = value.get("message") or value.get("type") or value.get("code")
    else:
        message = value
    compact = " ".join(str(message or response.reason_phrase).split())
    return compact[:300]


def _model_ids(client: httpx.Client, base_url: str) -> list[str]:
    response = client.get(_endpoint(base_url, "/models"))
    response.raise_for_status()
    body = response.json()
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise TypeError("/models response does not contain a data list")
    model_ids = {
        item["id"]
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
    }
    return sorted(model_ids)


def _has_usable_content(body: Any) -> bool:
    """Match the response contract consumed by src/core/llm.py."""
    if not isinstance(body, dict):
        return False
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    message = first.get("message")
    return isinstance(message, dict) and isinstance(message.get("content"), str) and bool(message["content"].strip())


def _probe_model(
    client: httpx.Client,
    base_url: str,
    model: str,
    *,
    max_tokens: int,
) -> ModelCheck:
    started = time.perf_counter_ns()
    try:
        response = client.post(
            _endpoint(base_url, "/chat/completions"),
            json={
                "model": model,
                "messages": [{"role": "user", "content": PROBE_PROMPT}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        if not response.is_success:
            return ModelCheck(model, "http_error", response.status_code, round(latency_ms, 2), _safe_error(response))
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            return ModelCheck(model, "invalid_json", response.status_code, round(latency_ms, 2), "invalid JSON")
        if not _has_usable_content(body):
            return ModelCheck(
                model,
                "empty_content",
                response.status_code,
                round(latency_ms, 2),
                "response has no choices[0].message.content",
            )
        return ModelCheck(model, "usable", response.status_code, round(latency_ms, 2), "")
    except httpx.RequestError as exc:
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        return ModelCheck(model, "request_error", None, round(latency_ms, 2), type(exc).__name__)


def _select_models(args: argparse.Namespace, catalog: list[str], parser: argparse.ArgumentParser) -> list[str]:
    if args.model:
        requested = list(dict.fromkeys(args.model))
        missing = [model for model in requested if model not in catalog]
        if missing:
            parser.error(f"model IDs are not listed by /models: {', '.join(missing)}")
        return requested

    if args.all:
        selected = catalog
    else:
        try:
            pattern = re.compile(args.match, re.IGNORECASE)
        except re.error as exc:
            parser.error(f"invalid --match regular expression: {exc}")
        selected = [model for model in catalog if pattern.search(model)]

    if not selected:
        parser.error("no models matched the selection")
    if len(selected) > args.limit:
        parser.error(
            f"selection contains {len(selected)} models, above --limit={args.limit}; "
            "narrow --match or explicitly raise --limit"
        )
    return selected


def _print_results(results: list[ModelCheck]) -> None:
    print(f"{'status':14} {'http':>5} {'latency ms':>11}  model")
    print("-" * 90)
    for result in results:
        http_status = "-" if result.http_status is None else str(result.http_status)
        print(f"{result.status:14} {http_status:>5} {result.latency_ms:>11.2f}  {result.model}")
        if result.error:
            print(f"  error: {result.error}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check which 9Router models satisfy this project's Chat Completions contract",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("NINEROUTER_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="NINEROUTER_API_KEY",
        help="environment variable containing the router API key",
    )
    parser.add_argument("--model", action="append", help="exact model ID; may be repeated")
    parser.add_argument("--match", default=DEFAULT_MATCH, help="case-insensitive model-ID regular expression")
    parser.add_argument("--all", action="store_true", help="probe the complete catalog; may call paid models")
    parser.add_argument("--limit", type=int, default=100, help="safety cap on the number of provider calls")
    parser.add_argument("--max-tokens", type=int, default=16, help="maximum output tokens for each probe")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds")
    parser.add_argument("--delay", type=float, default=0.1, help="delay between model probes in seconds")
    parser.add_argument("--output", help="optional redacted JSON report path")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.limit <= 0 or args.max_tokens <= 0 or args.timeout <= 0 or args.delay < 0:
        parser.error("--limit, --max-tokens and --timeout must be positive; --delay cannot be negative")

    api_key = os.getenv(args.api_key_env, "")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(headers=headers, timeout=args.timeout, follow_redirects=False) as client:
            catalog = _model_ids(client, args.base_url)
            selected = _select_models(args, catalog, parser)
            print(f"Catalog: {len(catalog)} models; probing: {len(selected)}")
            results: list[ModelCheck] = []
            for index, model in enumerate(selected):
                if index and args.delay:
                    time.sleep(args.delay)
                results.append(
                    _probe_model(
                        client,
                        args.base_url,
                        model,
                        max_tokens=args.max_tokens,
                    )
                )
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        print(f"Cannot read 9Router catalog: {type(exc).__name__}", file=sys.stderr)
        return 2

    _print_results(results)
    usable = sum(result.status == "usable" for result in results)
    print(f"\nUsable: {usable}/{len(results)}")

    if args.output:
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "selected_models": len(results),
            "usable_models": usable,
            "privacy": "API key, prompt and generated response content are omitted.",
            "results": [asdict(result) for result in results],
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report: {output.resolve()}")

    return 0 if usable == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
