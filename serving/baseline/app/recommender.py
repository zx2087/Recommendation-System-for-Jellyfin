from typing import List
from .schemas import ScoringItem, ScoringResult

MODEL_VERSION = "baseline-mock-v2"


def score_batch(items: List[ScoringItem]) -> List[ScoringResult]:
    """Mock scorer: returns a fixed score for each pair."""
    return [
        ScoringResult(
            user_id=item.user_id,
            movie_id=item.movie_id,
            predicted_score=0.5,
        )
        for item in items
    ]