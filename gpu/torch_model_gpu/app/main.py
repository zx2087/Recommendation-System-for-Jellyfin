from typing import List
from fastapi import FastAPI
from .schemas import ScoringItem, ScoringResult
from .recommender import score_batch, MODEL_VERSION, DEVICE

app = FastAPI(
    title="Jellyfin Recommender Torch GPU API",
    description="PyTorch scorer service (GPU)",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "device": DEVICE}


@app.post("/recommend", response_model=List[ScoringResult])
def recommend_endpoint(items: List[ScoringItem]):
    return score_batch(items)