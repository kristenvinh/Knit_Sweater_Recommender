#!/usr/bin/env python3
import json
import statistics
import sys
from typing import Any

TIMING_KEYS = [
    "total_request_sec",
    "feature_extraction_sec",
    "index_query_sec",
    "ravelry_fetch_sec",
    "xai_sec",
]


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = int(round(0.95 * (len(values) - 1)))
    index = max(0, min(index, len(values) - 1))
    return values[index]


def load_rows(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_metric(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        body = row.get("body")
        if not isinstance(body, dict):
            continue
        timings = body.get("timings")
        if not isinstance(timings, dict):
            continue
        value = timings.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 bench/summarize.py <benchmark.jsonl>")
        return 1

    path = sys.argv[1]
    rows = load_rows(path)

    if not rows:
        print("No benchmark rows found.")
        return 1

    ok_rows = [row for row in rows if row.get("status") == 200]
    error_rows = [row for row in rows if row.get("status") != 200]

    print(f"Results file: {path}")
    print(f"Total requests: {len(rows)}")
    print(f"Successful (200): {len(ok_rows)}")
    print(f"Errors (!200): {len(error_rows)}")

    if error_rows:
        counts: dict[int, int] = {}
        for row in error_rows:
            status = int(row.get("status", -1))
            counts[status] = counts.get(status, 0) + 1
        print("Error counts by status:", counts)

    print("")
    print("Timing metrics")
    for key in TIMING_KEYS:
        values = extract_metric(ok_rows, key)
        if not values:
            print(f"- {key}: no data")
            continue
        med = statistics.median(values)
        high = p95(values)
        print(f"- {key}: median={med:.4f}s p95={high:.4f}s n={len(values)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
