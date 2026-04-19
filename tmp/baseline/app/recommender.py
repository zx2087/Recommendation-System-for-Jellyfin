from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .model import RecommenderMLP, FEATURE_DIM
from .schemas import RecommendRequest, RecommendResponse, RecommendationItem

EMBEDDING_DIM = 384


def _resolve_model_version() -> str:
    p = Path(__file__).resolve().as_posix()
    if "torch_multiworker" in p:
        return "mlp-best-pt-multiworker-v2"
    if "torch_model" in p:
        return "mlp-best-pt-v2"
    return "mlp-best-pt-v2"


def resolve_model_path(filename: str) -> Path:
    here = Path(__file__).resolve()

    candidates = [
        here.parents[1] / "models" / filename,  # Docker: /app/models/...
        Path.cwd() / "models" / filename,       # local repo root when running from repo root
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find model {filename}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


MODEL_VERSION = _resolve_model_version()
MODEL_PATH = resolve_model_path("model_mlp_best.pt")


def _normalize_state_dict(state: dict) -> dict:
    if not state:
        return state
    sample_key = next(iter(state))
    if sample_key.startswith("net."):
        return state
    return {f"net.{k}": v for k, v in state.items()}


def _load_model() -> torch.nn.Module:
    model = RecommenderMLP(FEATURE_DIM)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(_normalize_state_dict(state))
    model.eval()
    return model


_MODEL = _load_model()


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
    x = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        scores = _MODEL(x).cpu().numpy().reshape(-1)

    top_k = min(req.request_k, len(movie_ids))
    top_indices = np.argsort(scores)[::-1][:top_k]

    recommendations = [
        RecommendationItem(
            rank=rank,
            movie_id=movie_ids[idx],
            score=round(float(scores[idx]), 8),
            reason="ranked_by_torch_model",
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