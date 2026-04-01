import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests



BASE_URL = "http://129.114.27.23:8000"
ENDPOINT = f"{BASE_URL}/recommend"

TOTAL_REQUESTS = 100
CONCURRENCY = 1
TIMEOUT_SECONDS = 10

INPUT_JSON_PATH = Path(__file__).resolve().parent.parent / "contracts" / "input_sample.json"



lock = threading.Lock()


def load_payload():
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def send_one_request(session: requests.Session, request_index: int):
    payload = load_payload()

    payload["request_id"] = f"bench_req_{request_index}"

    start = time.perf_counter()
    try:
        response = session.post(ENDPOINT, json=payload, timeout=TIMEOUT_SECONDS)
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
        return {
            "ok": False,
            "status_code": None,
            "latency_ms": elapsed_ms,
            "error": str(e),
        }


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def main():
    print("=== Benchmark configuration ===")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Timeout: {TIMEOUT_SECONDS}s")
    print()

    overall_start = time.perf_counter()
    results = []

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = [
                executor.submit(send_one_request, session, i)
                for i in range(TOTAL_REQUESTS)
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

    overall_elapsed = time.perf_counter() - overall_start

    success_results = [r for r in results if r["ok"]]
    failed_results = [r for r in results if not r["ok"]]
    success_latencies = sorted(r["latency_ms"] for r in success_results)

    success_count = len(success_results)
    fail_count = len(failed_results)
    success_rate = success_count / len(results) if results else 0.0
    throughput = len(results) / overall_elapsed if overall_elapsed > 0 else 0.0

    avg_latency = statistics.mean(success_latencies) if success_latencies else None
    p50_latency = percentile(success_latencies, 0.50)
    p95_latency = percentile(success_latencies, 0.95)

    print("=== Benchmark result ===")
    print(f"Total requests: {len(results)}")
    print(f"Successful requests: {success_count}")
    print(f"Failed requests: {fail_count}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Throughput (req/s): {throughput:.2f}")

    if avg_latency is not None:
        print(f"Average latency (ms): {avg_latency:.2f}")
        print(f"P50 latency (ms): {p50_latency:.2f}")
        print(f"P95 latency (ms): {p95_latency:.2f}")
    else:
        print("No successful requests, so latency stats are unavailable.")

    if failed_results:
        print("\n=== Sample failures ===")
        for idx, item in enumerate(failed_results[:5], start=1):
            print(f"[{idx}] status={item['status_code']} error={item['error']}")


if __name__ == "__main__":
    main()