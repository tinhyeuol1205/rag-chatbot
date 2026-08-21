#!/usr/bin/env python3
"""Small concurrent smoke/load harness for the PR17 model server.

It intentionally reports transport latency and status codes only; it never
prints document contents.  Run it after warmup against the actual Mac process,
then use the server's ``/metrics`` and OS memory tools for the full gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx


async def _run(args: argparse.Namespace) -> dict:
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    statuses: dict[str, int] = {}

    async def one(client: httpx.AsyncClient, index: int) -> None:
        async with semaphore:
            if args.endpoint == "embed":
                payload = {
                    "inputs": [f"load probe {index} Vietnamese policy text"],
                    "priority": "online",
                }
                path = "/embed"
            else:
                payload = {
                    "query": "điều kiện nghỉ phép",
                    "documents": [f"tài liệu kiểm thử số {index}"],
                    "priority": "online",
                }
                path = "/rerank"
            started = time.perf_counter()
            try:
                response = await client.post(path, json=payload)
                statuses[str(response.status_code)] = statuses.get(str(response.status_code), 0) + 1
            except Exception:
                statuses["transport_error"] = statuses.get("transport_error", 0) + 1
            finally:
                latencies.append((time.perf_counter() - started) * 1000)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        await asyncio.gather(*(one(client, index) for index in range(args.requests)))
    ordered = sorted(latencies)

    def percentile(value: float) -> float:
        if not ordered:
            return 0.0
        position = min(len(ordered) - 1, int(len(ordered) * value))
        return round(ordered[position], 2)

    return {
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "status_counts": statuses,
        "latency_ms": {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Load-test the PR17 BGE model server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--endpoint", choices=("embed", "rerank"), default="embed")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        parser.error("--requests and --concurrency must be positive")
    report = asyncio.run(_run(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

