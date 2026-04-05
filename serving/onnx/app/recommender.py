import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort

from .schemas import RecommendRequest, RecommendationItem, RecommendResponse

EMBEDDING_DIM = 384
FEATURE_DIM = EMBEDDING_DIM * 2 + 3

MODEL_VERSION = "mlp-best-onnx-v2"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model_mlp_best.onnx"

SESSION = ort.InferenceSession(
    MODEL_PATH.as_posix(),
    providers=["CPUExecutionProvider"],
)


def build_feature_matrix(request: RecommendRequest) -> tuple[list[str], np.ndarray]:
    if not request.candidate_movies:
        raise ValueError("candidate_movies cannot be empty")

    user_emb = np.asarray(request.user_embedding, dtype=np.float32)
    movie_ids: List[str] = []
    rows: List[np.ndarray] = []

    user_norm = np.linalg.norm(user_emb) + 1e-8

    for item in request.candidate_movies:
        movie_emb = np.asarray(item.movie_embedding, dtype=np.float32)

        movie_norm = np.linalg.norm(movie_emb) + 1e-8
        cosine_sim = float(np.sum(user_emb * movie_emb) / (user_norm * movie_norm))
        dot_product = float(np.sum(user_emb * movie_emb))
        l2_dist = float(np.linalg.norm(user_emb - movie_emb))

        features = np.hstack([
            user_emb,
            movie_emb,
            np.array([cosine_sim, dot_product, l2_dist], dtype=np.float32),
        ]).astype(np.float32)

        movie_ids.append(item.movie_id)
        rows.append(features)

    X = np.stack(rows)
    if X.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected feature dim {FEATURE_DIM}, got {X.shape[1]}")

    return movie_ids, X


def recommend(request: RecommendRequest) -> RecommendResponse:
    start = time.perf_counter()

    movie_ids, X = build_feature_matrix(request)
    raw_scores = SESSION.run(["scores"], {"features": X})[0]
    raw_scores = np.asarray(raw_scores).reshape(-1).tolist()

    scored = list(zip(movie_ids, raw_scores))
    scored.sort(key=lambda item: item[1], reverse=True)
    top_items = scored[: request.request_k]

    recommendations = [
        RecommendationItem(
            rank=i + 1,
            movie_id=movie_id,
            score=round(float(score), 6),
            reason="mlp_onnx_score",
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