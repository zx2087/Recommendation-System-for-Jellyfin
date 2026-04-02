import hashlib
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort

from .schemas import RecommendRequest, RecommendationItem, RecommendResponse


MODEL_VERSION = "toy-onnx-v1"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "toy_scorer.onnx"


def stable_float_from_text(text: str, salt: str = "") -> float:
    h = hashlib.md5(f"{salt}:{text}".encode("utf-8")).hexdigest()
    value = int(h[:8], 16) / 0xFFFFFFFF
    return float(value)


def build_candidate_features(request: RecommendRequest, movie_id: str) -> List[float]:
    completed_count = sum(1 for e in request.recent_events if e.watch_state == "completed")
    stopped_count = sum(1 for e in request.recent_events if e.watch_state == "stopped")
    started_count = sum(1 for e in request.recent_events if e.watch_state == "started")
    total_watch = sum(e.watch_duration_sec for e in request.recent_events)

    max_watch = max([e.watch_duration_sec for e in request.recent_events], default=1)
    avg_watch = total_watch / max(len(request.recent_events), 1)

    return [
        stable_float_from_text(movie_id, "movie"),
        stable_float_from_text(request.user_id, "user"),
        min(completed_count / 10.0, 1.0),
        min(stopped_count / 10.0, 1.0),
        min(started_count / 10.0, 1.0),
        min(avg_watch / 3600.0, 1.0),
        min(max_watch / 3600.0, 1.0),
        stable_float_from_text(movie_id + request.user_id, "pair"),
    ]


SESSION = ort.InferenceSession(
    MODEL_PATH.as_posix(),
    providers=["CPUExecutionProvider"],
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def recommend(request: RecommendRequest) -> RecommendResponse:
    start = time.perf_counter()

    candidate_ids = request.candidate_movie_ids or [
        "movie_popular_001",
        "movie_popular_002",
        "movie_popular_003",
        "movie_popular_004",
        "movie_popular_005",
    ]

    candidate_ids = candidate_ids[: max(request.request_k, len(candidate_ids))]

    feature_rows = np.array(
        [build_candidate_features(request, mid) for mid in candidate_ids],
        dtype=np.float32,
    )

    raw_scores = SESSION.run(["scores"], {"features": feature_rows})[0]
    probs = sigmoid(raw_scores).tolist()

    scored = list(zip(candidate_ids, probs))
    scored.sort(key=lambda item: item[1], reverse=True)
    top_items = scored[: request.request_k]

    recommendations = [
        RecommendationItem(
            rank=i + 1,
            movie_id=movie_id,
            score=round(float(score), 4),
            reason="toy_onnx_score",
        )
        for i, (movie_id, score) in enumerate(top_items)
    ]

    latency_ms = (time.perf_counter() - start) * 1000

    return RecommendResponse(
        request_id=request.request_id,
        user_id=request.user_id,
        timestamp=request.timestamp,
        model_version=MODEL_VERSION,
        fallback_used=False,
        latency_ms=round(latency_ms, 3),
        recommendations=recommendations,
    )