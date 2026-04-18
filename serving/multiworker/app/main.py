from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from .metrics import collect_metrics
from .recommender import STATE, score_request
from .schemas import RecommendRequest, RecommendResponse

app = FastAPI(
    title="Jellyfin Recommender Multiworker API",
    description="ONNX Runtime scorer — multi-worker deployment with Prometheus monitoring",
    version="0.4.0",
)


@app.get("/health")
def health():
    return {"status": "ok", **STATE.get_runtime_flags()}


@app.get("/metrics")
def metrics() -> Response:
    body, content_type = collect_metrics()
    return Response(content=body, media_type=content_type)


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: RecommendRequest):
    return score_request(req)


class SetModeRequest(BaseModel):
    mode: str


class RollbackRequest(BaseModel):
    model_path: str
    model_version: str | None = None


@app.post("/admin/set-mode")
def set_mode(body: SetModeRequest):
    STATE.set_mode(body.mode)
    return {"status": "ok", "mode": STATE.mode}


@app.post("/admin/rollback")
def rollback(body: RollbackRequest):
    try:
        STATE.reload_model(body.model_path, body.model_version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "status": "ok",
        "model_version": STATE.model_version,
        "serving_mode": STATE.mode,
    }
