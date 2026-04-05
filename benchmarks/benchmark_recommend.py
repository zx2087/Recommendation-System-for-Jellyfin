"""
Benchmark script for the Jellyfin recommender serving API.

Payload: loaded directly from contracts/recommender_input.sample.json
Format : POST /recommend  body = List[ScoringItem]
         Response         = List[ScoringResult]

Usage (from repo root):
    BASE_URL=http://<IP>:8000 CONCURRENCY=4 TOTAL_REQUESTS=500 python -m benchmarks.benchmark_recommend

Environment variables:
    BASE_URL          API base URL            (default: http://127.0.0.1:8000)
    TOTAL_REQUESTS    Number of requests      (default: 300)
    CONCURRENCY       Parallel threads        (default: 1)
    TIMEOUT_SECONDS   Per-request timeout (s) (default: 30)
"""

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL        = os.getenv("BASE_URL", "http://127.0.0.1:8000")
ENDPOINT        = f"{BASE_URL}/recommend"
TOTAL_REQUESTS  = int(os.getenv("TOTAL_REQUESTS", "300"))
CONCURRENCY     = int(os.getenv("CONCURRENCY", "1"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))

REPO_ROOT    = Path(__file__).resolve().parent.parent
SAMPLE_PATH  = REPO_ROOT / "contracts" / "recommender_input.sample.json"


# ── Load payload from the agreed contract sample ──────────────────────────────
def _load_payload() -> list:
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    # Sample is a list of ScoringItems — send as-is
    if not isinstance(data, list):
        raise ValueError(f"{SAMPLE_PATH} must be a JSON array, got {type(data)}")
    print(f"[benchmark] Loaded payload from {SAMPLE_PATH}  ({len(data)} item(s))")
    return data


BASE_PAYLOAD: list = _load_payload()


# ── Benchmark helpers ─────────────────────────────────────────────────────────
def send_one_request(session: requests.Session, request_index: int) -> dict:
    start = time.perf_counter()
    try:
        response = session.post(ENDPOINT, json=BASE_PAYLOAD, timeout=TIMEOUT_SECONDS)
        elapsed_ms = (time.perf_counter() - start) * 1000
        ok = response.status_code == 200
        return {
            "ok": ok,
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
            "error": None if ok else response.text[:300],
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"ok": False, "status_code": None, "latency_ms": elapsed_ms, "error": str(e)}


def percentile(sorted_values: list, p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def warmup(session: requests.Session, n: int = 10) -> None:
    for _ in range(n):
        try:
            session.post(ENDPOINT, json=BASE_PAYLOAD, timeout=TIMEOUT_SECONDS)
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Benchmark configuration ===")
    print(f"Endpoint      : {ENDPOINT}")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Concurrency   : {CONCURRENCY}")
    print(f"Timeout       : {TIMEOUT_SECONDS}s")
    print()

    with requests.Session() as session:
        print("Warming up (10 requests)...")
        warmup(session)

        overall_start = time.perf_counter()
        results: list[dict] = []

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = [
                executor.submit(send_one_request, session, i)
                for i in range(TOTAL_REQUESTS)
            ]
            for future in as_completed(futures):
                results.append(future.result())

    overall_elapsed = time.perf_counter() - overall_start

    success_results   = [r for r in results if r["ok"]]
    failed_results    = [r for r in results if not r["ok"]]
    success_latencies = sorted(r["latency_ms"] for r in success_results)

    total         = len(results)
    success_count = len(success_results)
    success_rate  = success_count / total if total else 0.0
    throughput    = total / overall_elapsed if overall_elapsed > 0 else 0.0

    print("=== Benchmark results ===")
    print(f"Total requests  : {total}")
    print(f"Successful      : {success_count}")
    print(f"Failed          : {len(failed_results)}")
    print(f"Success rate    : {success_rate:.2%}")
    print(f"Throughput      : {throughput:.2f} req/s")
    print(f"Total wall time : {overall_elapsed:.2f}s")

    if success_latencies:
        avg = statistics.mean(success_latencies)
        p50 = percentile(success_latencies, 0.50)
        p95 = percentile(success_latencies, 0.95)
        p99 = percentile(success_latencies, 0.99)
        print(f"Latency avg (ms): {avg:.2f}")
        print(f"Latency p50 (ms): {p50:.2f}")
        print(f"Latency p95 (ms): {p95:.2f}")
        print(f"Latency p99 (ms): {p99:.2f}")
    else:
        print("No successful requests — latency stats unavailable.")

    if failed_results:
        print("\n=== Sample failures ===")
        for idx, item in enumerate(failed_results[:5], start=1):
            print(f"[{idx}] status={item['status_code']} error={item['error']}")


if __name__ == "__main__":
    main()