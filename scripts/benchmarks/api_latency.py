#!/usr/bin/env python3
"""Measure basic API latency for a deployed service endpoint."""

from __future__ import annotations

import argparse
import statistics
import time
import urllib.request


def request_once(url: str, timeout: float) -> float:
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as response:
        response.read()
    return (time.perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HTTP latency.")
    parser.add_argument("--url", required=True, help="Endpoint to benchmark.")
    parser.add_argument("--requests", type=int, default=20, help="Number of requests to send.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds.")
    args = parser.parse_args()

    latencies = [request_once(args.url, args.timeout) for _ in range(args.requests)]
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]

    print(f"requests={len(latencies)}")
    print(f"avg_ms={statistics.mean(latencies):.2f}")
    print(f"p50_ms={statistics.median(latencies):.2f}")
    print(f"p95_ms={p95:.2f}")
    print(f"max_ms={max(latencies):.2f}")


if __name__ == "__main__":
    main()
