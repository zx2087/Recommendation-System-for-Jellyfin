from fastapi import FastAPI

from .recommender import MODEL_VERSION, score_request
from .schemas import RecommendRequest, RecommendResponse

app = FastAPI(
    title="Jellyfin Recommender Torch API",
    description="PyTorch scorer service",
    version="0.2.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: RecommendRequest):
    return score_request(req)