from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class RecentEvent(BaseModel):
    movie_id: str = Field(..., description="Movie identifier")
    event_time: str = Field(..., description="ISO timestamp of the event")
    watch_duration_sec: int = Field(..., ge=0, description="Watch duration in seconds")
    watch_state: Literal["started", "stopped", "completed"]


class ClientContext(BaseModel):
    surface: Optional[str] = Field(default="homepage")
    device_type: Optional[str] = Field(default="web")
    app_version: Optional[str] = Field(default="0.1.0")


class RecommendRequest(BaseModel):
    request_id: str
    user_id: str
    timestamp: str
    request_k: int = Field(default=10, ge=1, le=50)
    recent_events: List[RecentEvent] = Field(default_factory=list)
    candidate_movie_ids: List[str] = Field(default_factory=list)
    client_context: Optional[ClientContext] = Field(default_factory=ClientContext)


class RecommendationItem(BaseModel):
    rank: int = Field(..., ge=1)
    movie_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str


class RecommendResponse(BaseModel):
    request_id: str
    user_id: str
    timestamp: str
    model_version: str
    fallback_used: bool
    latency_ms: float = Field(..., ge=0.0)
    recommendations: List[RecommendationItem]