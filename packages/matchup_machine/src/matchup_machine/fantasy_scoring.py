# src/fantasy_scoring.py

from __future__ import annotations

from typing import Dict
from build_dataset import OUTCOME_LABELS

"""
Fantasy scoring rules (hitting only) based on your league:

Batting
- Runs Scored (R)        : +1
- Total Bases (TB)       : +1 per base
- Runs Batted In (RBI)   : +1
- Walks (BB)             : +1
- Strikeouts (K)         : -1
- Stolen Bases (SB)      : +1   (not modeled yet in outcome model)

We map outcome classes (single, double, HR, etc.)
to an approximate fantasy point value per plate appearance.
"""


# Points assigned per outcome label (per PA)
# These labels must exactly match OUTCOME_LABELS from build_dataset.py:
# ["single", "double", "triple", "home_run", "walk",
#  "strikeout", "ball_in_play_out", "other"]
HITTER_OUTCOME_POINTS: Dict[str, float] = {
    "single": 1.0,       # 1 TB
    "double": 2.0,       # 2 TB
    "triple": 3.0,       # 3 TB
    # HR ≈ 4 TB + 1 R + 1 RBI => 6 total points (approximation)
    "home_run": 6.0,
    # Walk = 1 BB
    "walk": 1.0,
    # Strikeout = -1 K
    "strikeout": -1.0,
    # Ball in play out = 0
    "ball_in_play_out": 0.0,
    # "other" (HBP, sac, CI, etc.) – treat roughly like a walk
    "other": 1.0,
}


def expected_hitter_points_per_pa(
    outcome_probs: Dict[str, float],
) -> float:
    """
    Given a mapping {outcome_label -> probability}, compute expected
    fantasy points per plate appearance for a hitter.

    outcome_probs should come from the multiclass model, with keys
    matching OUTCOME_LABELS.
    """
    ev = 0.0
    for label in OUTCOME_LABELS:
        prob = float(outcome_probs.get(label, 0.0))
        pts = HITTER_OUTCOME_POINTS.get(label, 0.0)
        ev += prob * pts
    return ev


def hitter_points_breakdown(
    outcome_probs: Dict[str, float],
) -> Dict[str, float]:
    """
    Optional helper: return a breakdown of expected points contribution
    by outcome type. For example, you can use this if you want to show
    how much of the EV comes from HR vs walks vs singles.

    Returns: {label -> expected_points_from_that_label_per_PA}
    """
    breakdown: Dict[str, float] = {}
    for label in OUTCOME_LABELS:
        prob = float(outcome_probs.get(label, 0.0))
        pts = HITTER_OUTCOME_POINTS.get(label, 0.0)
        breakdown[label] = prob * pts
    return breakdown
