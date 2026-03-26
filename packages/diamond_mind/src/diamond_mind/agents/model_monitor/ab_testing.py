"""
A/B testing framework for champion/challenger model evaluation.

Tracks per-variant predictions and outcomes, then uses a two-sample t-test
to determine whether the challenger significantly outperforms the champion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats


@dataclass
class VariantStats:
    """Accumulated statistics for one A/B variant (champion or challenger)."""

    name: str
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.actuals)

    @property
    def errors(self) -> np.ndarray:
        return np.abs(np.array(self.predictions) - np.array(self.actuals))

    @property
    def mean_absolute_error(self) -> Optional[float]:
        if self.n == 0:
            return None
        return float(self.errors.mean())

    @property
    def accuracy(self) -> Optional[float]:
        """Binary accuracy: fraction of predictions within 0.5 of actual."""
        if self.n == 0:
            return None
        return float((self.errors < 0.5).mean())

    def record(self, prediction: float, actual: float) -> None:
        self.predictions.append(prediction)
        self.actuals.append(actual)


class ABTest:
    """
    Champion/challenger A/B test.

    Usage::

        test = ABTest(test_id="exp_001", min_samples=50, significance=0.05)
        variant = test.assign(entity_id="player_123")   # "champion" or "challenger"
        test.record(variant, prediction=0.285, actual=0.301)
        if test.has_sufficient_data():
            result = test.test_significance()
    """

    CHAMPION = "champion"
    CHALLENGER = "challenger"

    def __init__(
        self,
        test_id: str,
        min_samples: int = 50,
        significance: float = 0.05,
        challenger_traffic_pct: float = 0.2,
    ):
        self.test_id = test_id
        self.min_samples = min_samples
        self.significance = significance
        self.challenger_traffic_pct = challenger_traffic_pct

        self.champion = VariantStats(name=self.CHAMPION)
        self.challenger = VariantStats(name=self.CHALLENGER)
        self._rng = np.random.default_rng()

    def assign(self, entity_id: Any = None) -> str:
        """
        Route an entity to champion or challenger.

        If entity_id is provided, uses a deterministic hash so the same entity
        always gets the same variant.  Otherwise random.
        """
        if entity_id is not None:
            bucket = hash(str(entity_id)) % 100
            return self.CHALLENGER if bucket < self.challenger_traffic_pct * 100 else self.CHAMPION
        return (
            self.CHALLENGER
            if self._rng.random() < self.challenger_traffic_pct
            else self.CHAMPION
        )

    def record(self, variant: str, prediction: float, actual: float) -> None:
        """Record a prediction/outcome pair for a variant."""
        if variant == self.CHAMPION:
            self.champion.record(prediction, actual)
        else:
            self.challenger.record(prediction, actual)

    def has_sufficient_data(self) -> bool:
        """Return True when both variants have >= min_samples observations."""
        return self.champion.n >= self.min_samples and self.challenger.n >= self.min_samples

    def test_significance(self) -> Dict[str, Any]:
        """
        Run a two-sample Welch t-test on absolute errors.

        Returns a dict with keys: p_value, t_stat, champion_mae, challenger_mae,
        challenger_wins, significant.
        """
        if not self.has_sufficient_data():
            raise RuntimeError(
                f"Insufficient data: champion={self.champion.n}, "
                f"challenger={self.challenger.n}, required={self.min_samples}"
            )

        champ_errors = self.champion.errors
        chal_errors = self.challenger.errors

        t_stat, p_value = stats.ttest_ind(champ_errors, chal_errors, equal_var=False)

        champ_mae = float(champ_errors.mean())
        chal_mae = float(chal_errors.mean())
        significant = bool(p_value < self.significance)
        challenger_wins = significant and chal_mae < champ_mae

        return {
            "test_id": self.test_id,
            "t_stat": round(float(t_stat), 6),
            "p_value": round(float(p_value), 6),
            "champion_mae": round(champ_mae, 6),
            "challenger_mae": round(chal_mae, 6),
            "champion_n": self.champion.n,
            "challenger_n": self.challenger.n,
            "significant": significant,
            "challenger_wins": challenger_wins,
        }

    def should_promote_challenger(self) -> bool:
        """Return True if challenger has significantly lower error than champion."""
        if not self.has_sufficient_data():
            return False
        result = self.test_significance()
        return result["challenger_wins"]

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary dict for logging / result_data."""
        return {
            "test_id": self.test_id,
            "champion": {
                "n": self.champion.n,
                "mae": self.champion.mean_absolute_error,
                "accuracy": self.champion.accuracy,
            },
            "challenger": {
                "n": self.challenger.n,
                "mae": self.challenger.mean_absolute_error,
                "accuracy": self.challenger.accuracy,
            },
            "has_sufficient_data": self.has_sufficient_data(),
            "should_promote": self.should_promote_challenger() if self.has_sufficient_data() else None,
        }
