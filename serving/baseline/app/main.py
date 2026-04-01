from fastapi import FastAPI

from .schemas import RecommendRequest, RecommendResponse
from .recommender import recommend, MODEL_VERSION

app = FastAPI(
    title="Jellyfin Recommender Baseline API",
    description="Baseline FastAPI CPU service for movie recommendation",
    version="0.1.0"
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(request: RecommendRequest):
    return recommend(request)