from __future__ import annotations

from collections import defaultdict
from threading import Lock


class MetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self.requests_total: dict[tuple[str, str], int] = defaultdict(int)
        self.fallback_total: dict[str, int] = defaultdict(int)
        self.candidate_counts: list[int] = []
        self.return_counts: list[int] = []
        self.latency_sums: dict[str, float] = defaultdict(float)
        self.latency_counts: dict[str, int] = defaultdict(int)

    def observe_request(self, mode: str, status: str) -> None:
        with self._lock:
            self.requests_total[(mode, status)] += 1

    def observe_fallback(self, reason: str) -> None:
        with self._lock:
            self.fallback_total[reason] += 1

    def observe_candidate_count(self, count: int) -> None:
        with self._lock:
            self.candidate_counts.append(int(count))

    def observe_return_count(self, count: int) -> None:
        with self._lock:
            self.return_counts.append(int(count))

    def observe_latency(self, mode: str, seconds: float) -> None:
        with self._lock:
            self.latency_sums[mode] += float(seconds)
            self.latency_counts[mode] += 1

    @staticmethod
    def _avg(values: list[int]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    def render_prometheus_text(self) -> str:
        with self._lock:
            lines: list[str] = []

            lines.append("# HELP recommend_requests_total Total recommend endpoint requests")
            lines.append("# TYPE recommend_requests_total counter")
            for (mode, status), count in sorted(self.requests_total.items()):
                lines.append(
                    f'recommend_requests_total{{mode="{mode}",status="{status}"}} {count}'
                )

            lines.append("# HELP fallback_used_total Number of requests served using fallback logic")
            lines.append("# TYPE fallback_used_total counter")
            for reason, count in sorted(self.fallback_total.items()):
                lines.append(f'fallback_used_total{{reason="{reason}"}} {count}')

            lines.append("# HELP recommend_candidate_count_avg Average candidate count per request")
            lines.append("# TYPE recommend_candidate_count_avg gauge")
            lines.append(f"recommend_candidate_count_avg {self._avg(self.candidate_counts):.6f}")

            lines.append("# HELP recommend_return_count_avg Average recommendations returned per request")
            lines.append("# TYPE recommend_return_count_avg gauge")
            lines.append(f"recommend_return_count_avg {self._avg(self.return_counts):.6f}")

            lines.append("# HELP recommend_request_latency_seconds_avg Average end-to-end latency")
            lines.append("# TYPE recommend_request_latency_seconds_avg gauge")
            for mode in sorted(set(self.latency_sums) | set(self.latency_counts)):
                cnt = self.latency_counts.get(mode, 0)
                avg = (self.latency_sums.get(mode, 0.0) / cnt) if cnt else 0.0
                lines.append(
                    f'recommend_request_latency_seconds_avg{{mode="{mode}"}} {avg:.6f}'
                )

            return "\n".join(lines) + "\n"


METRICS = MetricsStore()
