import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

from .metrics import METRICS
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


def resolve_model_path(filename: str) -> Path:
    here = Path(__file__).resolve()

    candidates = [
        here.parents[1] / "models" / filename,  # Docker: /app/models/...
        Path.cwd() / "models" / filename,  # local repo root when running from repo root
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find model {filename}. Tried: " + ", ".join(str(p) for p in candidates)
    )


def _normalize_mode(value: str) -> str:
    mode = (value or "model").strip().lower()
    if mode not in {"model", "fallback"}:
        return "model"
    return mode


def _fallback_version(pool_size: int) -> str:
    return f"fallback-popular-{pool_size}-v1"


def _resolve_pool_size(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 100
    return max(1, min(500, parsed))


def _resolve_fallback_ids_path() -> Path:
    here = Path(__file__).resolve()
    env_override = os.getenv("FALLBACK_IDS_FILE", "")
    if env_override:
        return Path(env_override)

    return here.parent / "fallback_popular_movies.json"


def _load_fallback_ids(path: Path) -> list[str]:
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(v) for v in data if v is not None]

    if isinstance(data, dict):
        values = data.get("movie_ids") or data.get("item_ids") or []
        if isinstance(values, list):
            return [str(v) for v in values if v is not None]

    return []


MODEL_VERSION = _resolve_model_version()
MODEL_PATH = resolve_model_path("model_mlp_best.onnx")
SERVING_MODE = _normalize_mode(os.getenv("SERVING_MODE", "model"))
FALLBACK_POOL_SIZE = _resolve_pool_size(os.getenv("FALLBACK_POOL_SIZE", "100"))
FALLBACK_IDS_PATH = _resolve_fallback_ids_path()
FALLBACK_IDS = _load_fallback_ids(FALLBACK_IDS_PATH)

SESSION = ort.InferenceSession(
    MODEL_PATH.as_posix(),
    providers=["CPUExecutionProvider"],
)


def get_runtime_flags() -> dict:
    return {
        "serving_mode": SERVING_MODE,
        "fallback_pool_size": FALLBACK_POOL_SIZE,
        "fallback_ids_file": str(FALLBACK_IDS_PATH),
        "fallback_ids_count": len(FALLBACK_IDS),
    }


def _build_features(req: RecommendRequest) -> tuple[np.ndarray, list[str]]:
    if not req.candidates:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []

    user = np.asarray(req.user_embedding, dtype=np.float32)
    movie_ids = [c.movie_id for c in req.candidates]
    movies = np.asarray([c.movie_embedding for c in req.candidates], dtype=np.float32)

    users = np.broadcast_to(user, movies.shape).astype(np.float32)

    user_norms = np.linalg.norm(users, axis=1) + 1e-8
    movie_norms = np.linalg.norm(movies, axis=1) + 1e-8

    dots = np.einsum("ij,ij->i", users, movies).astype(np.float32)
    cosines = (dots / (user_norms * movie_norms)).astype(np.float32)
    l2s = np.linalg.norm(users - movies, axis=1).astype(np.float32)

    extra = np.stack([cosines, dots, l2s], axis=1).astype(np.float32)
    features = np.concatenate([users, movies, extra], axis=1).astype(np.float32)

    return features, movie_ids


def _fallback_movie_ids(req: RecommendRequest) -> list[str]:
    if FALLBACK_IDS:
        return FALLBACK_IDS[:FALLBACK_POOL_SIZE]

    if req.candidates:
        return [c.movie_id for c in req.candidates][:FALLBACK_POOL_SIZE]

    return []


def _build_fallback_response(req: RecommendRequest, started: float, reason: str) -> RecommendResponse:
    movie_ids = _fallback_movie_ids(req)
    top_k = min(req.request_k, len(movie_ids))

    recommendations = [
        RecommendationItem(
            rank=rank,
            movie_id=movie_ids[idx],
            score=0.0,
            reason=reason,
        )
        for rank, idx in enumerate(range(top_k), start=1)
    ]

    latency_ms = (perf_counter() - started) * 1000

    METRICS.observe_fallback(reason)
    METRICS.observe_return_count(len(recommendations))
    METRICS.observe_request(mode="fallback", status="ok")
    METRICS.observe_latency(mode="fallback", seconds=latency_ms / 1000.0)

    return RecommendResponse(
        request_id=req.request_id,
        user_id=req.user_id,
        timestamp=req.timestamp,
        model_version=_fallback_version(FALLBACK_POOL_SIZE),
        fallback_used=True,
        latency_ms=round(latency_ms, 3),
        recommendations=recommendations,
    )


def score_request(req: RecommendRequest) -> RecommendResponse:
    started = perf_counter()
    METRICS.observe_candidate_count(len(req.candidates))

    if SERVING_MODE == "fallback":
        return _build_fallback_response(req, started, reason="fallback_mode_enabled")

    if not req.candidates:
        METRICS.observe_request(mode="model", status="ok")
        latency_ms = (perf_counter() - started) * 1000
        METRICS.observe_latency(mode="model", seconds=latency_ms / 1000.0)
        METRICS.observe_return_count(0)
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

    try:
        scores = SESSION.run(["scores"], {"features": X})[0].reshape(-1)
    except Exception:
        return _build_fallback_response(req, started, reason="fallback_inference_error")

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

    METRICS.observe_request(mode="model", status="ok")
    METRICS.observe_latency(mode="model", seconds=latency_ms / 1000.0)
    METRICS.observe_return_count(len(recommendations))

    return RecommendResponse(
        request_id=req.request_id,
        user_id=req.user_id,
        timestamp=req.timestamp,
        model_version=MODEL_VERSION,
        fallback_used=False,
        latency_ms=round(latency_ms, 3),
        recommendations=recommendations,
    )
