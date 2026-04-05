from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort

from .schemas import ScoringItem, ScoringResult

EMBEDDING_DIM = 384
FEATURE_DIM   = EMBEDDING_DIM * 2 + 3  # 771

MODEL_VERSION = "mlp-best-onnx-v2"
MODEL_PATH    = Path(__file__).resolve().parent.parent / "models" / "model_mlp_best.onnx"

SESSION = ort.InferenceSession(
    MODEL_PATH.as_posix(),
    providers=["CPUExecutionProvider"],
)


def _build_features(items: List[ScoringItem]) -> np.ndarray:
    rows = []
    for item in items:
        u = np.asarray(item.user_embedding, dtype=np.float32)
        m = np.asarray(item.movie_embedding, dtype=np.float32)
        u_norm = np.linalg.norm(u) + 1e-8
        m_norm = np.linalg.norm(m) + 1e-8
        cosine  = float(np.dot(u, m) / (u_norm * m_norm))
        dot     = float(np.dot(u, m))
        l2      = float(np.linalg.norm(u - m))
        rows.append(np.hstack([u, m, np.array([cosine, dot, l2], dtype=np.float32)]))
    return np.stack(rows).astype(np.float32)


def score_batch(items: List[ScoringItem]) -> List[ScoringResult]:
    X      = _build_features(items)
    scores = SESSION.run(["scores"], {"features": X})[0].reshape(-1).tolist()
    return [
        ScoringResult(user_id=item.user_id, movie_id=item.movie_id, predicted_score=round(float(s), 8))
        for item, s in zip(items, scores)
    ]