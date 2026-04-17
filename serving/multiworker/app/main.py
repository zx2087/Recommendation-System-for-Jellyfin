from fastapi import FastAPI, Response

from .metrics import METRICS
from .recommender import MODEL_VERSION, get_runtime_flags, score_request
from .schemas import RecommendRequest, RecommendResponse

app = FastAPI(
    title="Jellyfin Recommender Multiworker API",
    description="ONNX Runtime scorer — multi-worker deployment",
    version="0.3.0",
)


@app.get("/health")
def health():
    runtime = get_runtime_flags()
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        **runtime,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=METRICS.render_prometheus_text(), media_type="text/plain; version=0.0.4")


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: RecommendRequest):
    return score_request(req)
