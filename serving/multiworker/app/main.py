from typing import List
from fastapi import FastAPI
from .schemas import ScoringItem, ScoringResult
from .recommender import score_batch, MODEL_VERSION

app = FastAPI(
    title="Jellyfin Recommender ONNX Multi-Worker API",
    description="ONNX Runtime scorer — 4 gunicorn workers for high throughput",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/recommend", response_model=List[ScoringResult])
def recommend_endpoint(items: List[ScoringItem]):
    return score_batch(items)