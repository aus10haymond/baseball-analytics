"""
Matchup Machine - MLB Batter-Pitcher Matchup Predictions

Core ML models trained on Statcast data for outcome prediction.
"""

__version__ = "0.1.0"

# Export key functions for convenience
from .fantasy_inference import (
    load_artifacts,
    find_player_id,
    estimate_batter_outcome_probs_from_history,
)
from .fantasy_scoring import expected_hitter_points_per_pa
from .build_dataset import OUTCOME_LABELS

__all__ = [
    "load_artifacts",
    "find_player_id",
    "estimate_batter_outcome_probs_from_history",
    "expected_hitter_points_per_pa",
    "OUTCOME_LABELS",
]
