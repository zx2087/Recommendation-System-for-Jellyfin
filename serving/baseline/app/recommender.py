import time
from typing import List

from .schemas import RecommendRequest, RecommendationItem, RecommendResponse

MODEL_VERSION = "baseline-mock-v2"


def mock_rank_candidates(request: RecommendRequest) -> List[RecommendationItem]:
    if not request.candidate_movies:
        fallback_ids = [
            "movie_popular_001",
            "movie_popular_002",
            "movie_popular_003",
            "movie_popular_004",
            "movie_popular_005",
        ]
        selected_ids = fallback_ids[: request.request_k]
    else:
        selected_ids = [item.movie_id for item in request.candidate_movies[: request.request_k]]

    recommendations: List[RecommendationItem] = []
    base_score = 0.95

    for idx, movie_id in enumerate(selected_ids, start=1):
        score = max(0.5, base_score - (idx - 1) * 0.03)
        if idx == 1:
            reason = "baseline_rank_1"
        elif idx == 2:
            reason = "baseline_rank_2"
        else:
            reason = "baseline_mock_order"

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
    recommendations = mock_rank_candidates(request)
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