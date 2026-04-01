import time
from typing import List

from .schemas import RecommendRequest, RecommendationItem, RecommendResponse


MODEL_VERSION = "baseline-mock-v1"


def mock_rank_candidates(candidate_movie_ids: List[str], top_k: int) -> List[RecommendationItem]:
    """
    A placeholder recommender.
    It does not use a real model yet.
    It simply returns the first top_k candidates with decreasing mock scores.
    """
    if not candidate_movie_ids:
        candidate_movie_ids = [
            "movie_popular_001",
            "movie_popular_002",
            "movie_popular_003",
            "movie_popular_004",
            "movie_popular_005",
        ]

    selected = candidate_movie_ids[:top_k]
    recommendations: List[RecommendationItem] = []

    base_score = 0.95
    for idx, movie_id in enumerate(selected, start=1):
        score = max(0.5, base_score - (idx - 1) * 0.03)

        if idx == 1:
            reason = "similar_to_recent_completed"
        elif idx == 2:
            reason = "high_popularity"
        elif idx == 3:
            reason = "co_watch_signal"
        else:
            reason = "popular_backup"

        recommendations.append(
            RecommendationItem(
                rank=idx,
                movie_id=movie_id,
                score=round(score, 4),
                reason=reason,
            )
        )

    return recommendations


def recommend(request: RecommendRequest) -> RecommendResponse:
    start = time.perf_counter()

    recommendations = mock_rank_candidates(
        candidate_movie_ids=request.candidate_movie_ids,
        top_k=request.request_k,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return RecommendResponse(
        request_id=request.request_id,
        user_id=request.user_id,
        timestamp=request.timestamp,
        model_version=MODEL_VERSION,
        fallback_used=False,
        latency_ms=round(latency_ms, 3),
        recommendations=recommendations,
    )