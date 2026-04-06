from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

from .schemas import RecommendRequest, RecommendResponse, RecommendationItem

EMBEDDING_DIM = 384
FEATURE_DIM = EMBEDDING_DIM * 2 + 3  # 771


def _resolve_model_version() -> str:
    p = Path(__file__).resolve().as_posix()
    if "onnx_multiworker" in p:
        return "mlp-best-onnx-multiworker-v2"
    if "multiworker" in p:
        return "mlp-best-onnx-multiworker-v2"
    return "mlp-best-onnx-v2"


MODEL_VERSION = _resolve_model_version()
MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "model_mlp_best.onnx"

SESSION = ort.InferenceSession(
    MODEL_PATH.as_posix(),
    providers=["CPUExecutionProvider"],
)


def _build_features(req: RecommendRequest) -> tuple[np.ndarray, list[str]]:
    if not req.candidates:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []

    user = np.asarray(req.user_embedding, dtype=np.float32)  # (384,)
    movie_ids = [c.movie_id for c in req.candidates]
    movies = np.asarray([c.movie_embedding for c in req.candidates], dtype=np.float32)  # (N, 384)

    users = np.broadcast_to(user, movies.shape).astype(np.float32)

    user_norms = np.linalg.norm(users, axis=1) + 1e-8
    movie_norms = np.linalg.norm(movies, axis=1) + 1e-8

    dots = np.einsum("ij,ij->i", users, movies).astype(np.float32)
    cosines = (dots / (user_norms * movie_norms)).astype(np.float32)
    l2s = np.linalg.norm(users - movies, axis=1).astype(np.float32)

    extra = np.stack([cosines, dots, l2s], axis=1).astype(np.float32)
    features = np.concatenate([users, movies, extra], axis=1).astype(np.float32)

    return features, movie_ids


def score_request(req: RecommendRequest) -> RecommendResponse:
    started = perf_counter()

    if not req.candidates:
        latency_ms = (perf_counter() - started) * 1000
        return RecommendResponse(
            request_id=req.request_id,
            user_id=req.user_id,
            timestamp=req.timestamp,
            model_version=MODEL_VERSION,
            fallback_used=False,
            latency_ms=round(latency_ms, 3),
            recommendations=[],
        )

    X, movie_ids = _build_features(req)
    scores = SESSION.run(["scores"], {"features": X})[0].reshape(-1)

    top_k = min(req.request_k, len(movie_ids))
    top_indices = np.argsort(scores)[::-1][:top_k]

    recommendations = [
        RecommendationItem(
            rank=rank,
            movie_id=movie_ids[idx],
            score=round(float(scores[idx]), 8),
            reason="ranked_by_onnx_model",
        )
        for rank, idx in enumerate(top_indices, start=1)
    ]

    latency_ms = (perf_counter() - started) * 1000

    return RecommendResponse(
        request_id=req.request_id,
        user_id=req.user_id,
        timestamp=req.timestamp,
        model_version=MODEL_VERSION,
        fallback_used=False,
        latency_ms=round(latency_ms, 3),
        recommendations=recommendations,
    )