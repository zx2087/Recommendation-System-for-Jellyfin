from fastapi import FastAPI

from .recommender import recommend, MODEL_VERSION, DEVICE
from .schemas import RecommendRequest, RecommendResponse

app = FastAPI(
    title="Jellyfin Recommender Torch GPU API",
    description="PyTorch scorer service with GPU acceleration",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "device": DEVICE,
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(request: RecommendRequest):
    return recommend(request)
