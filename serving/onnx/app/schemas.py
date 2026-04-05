from typing import List
from pydantic import BaseModel, Field, field_validator

EMBEDDING_DIM = 384


class ScoringItem(BaseModel):
    user_id: int
    movie_id: int
    user_embedding: List[float] = Field(..., description="User embedding vector (384-dim)")
    movie_embedding: List[float] = Field(..., description="Movie embedding vector (384-dim)")

    @field_validator("user_embedding", "movie_embedding")
    @classmethod
    def validate_embedding_dim(cls, v):
        if len(v) != EMBEDDING_DIM:
            raise ValueError(f"embedding must have length {EMBEDDING_DIM}, got {len(v)}")
        return v


class ScoringResult(BaseModel):
    user_id: int
    movie_id: int
    predicted_score: float