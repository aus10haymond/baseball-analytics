"""
Drift detection utilities for the Model Monitor Agent.

Provides Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) based
feature drift detection with no external ML dependencies.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

from shared.schemas import DriftDetectionResult


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index between two distributions.

    PSI interpretation:
      < 0.1  — no significant change
      0.1–0.2 — moderate change; monitor
      > 0.2  — significant change; action needed

    Args:
        expected: Baseline/reference values.
        actual:   Current/incoming values.
        bins:     Number of histogram bins.

    Returns:
        PSI score (non-negative float).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Derive bin edges from the combined range
    combined = np.concatenate([expected, actual])
    bin_edges = np.linspace(combined.min(), combined.max(), bins + 1)
    # Ensure last edge is inclusive
    bin_edges[-1] += 1e-10

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to proportions, smoothing zeros to avoid log(0)
    exp_pct = np.where(exp_counts == 0, 1e-6, exp_counts / len(expected))
    act_pct = np.where(act_counts == 0, 1e-6, act_counts / len(actual))

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 6)


def run_ks_test(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """
    Run two-sample Kolmogorov-Smirnov test.

    Args:
        expected: Baseline values.
        actual:   Current values.

    Returns:
        (ks_statistic, p_value) — p < 0.05 suggests significant drift.
    """
    result = stats.ks_2samp(expected, actual)
    return float(result.statistic), float(result.pvalue)


def detect_feature_drift(
    baseline_df,
    current_df,
    psi_threshold: float = 0.2,
    ks_p_threshold: float = 0.05,
) -> DriftDetectionResult:
    """
    Compare numeric feature distributions between baseline and current DataFrames.

    A feature is flagged as drifted if PSI >= psi_threshold OR KS p-value < ks_p_threshold.

    Args:
        baseline_df:     Reference/training distribution (pandas DataFrame).
        current_df:      Current/production distribution (pandas DataFrame).
        psi_threshold:   PSI threshold for flagging drift.
        ks_p_threshold:  KS p-value threshold (smaller → more sensitive).

    Returns:
        DriftDetectionResult schema instance.
    """
    import pandas as pd  # local import to keep module importable without pandas at module level

    numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
    common_cols = [c for c in numeric_cols if c in current_df.columns]

    psi_scores: Dict[str, float] = {}
    ks_statistics: Dict[str, float] = {}
    affected_features: List[str] = []

    for col in common_cols:
        base_vals = baseline_df[col].dropna().values
        curr_vals = current_df[col].dropna().values

        if len(base_vals) < 2 or len(curr_vals) < 2:
            continue

        psi = calculate_psi(base_vals, curr_vals)
        ks_stat, ks_p = run_ks_test(base_vals, curr_vals)

        psi_scores[col] = psi
        ks_statistics[col] = ks_stat

        if psi >= psi_threshold or ks_p < ks_p_threshold:
            affected_features.append(col)

    drift_detected = len(affected_features) > 0

    if not drift_detected:
        drift_score = 0.0
        drift_type = "none"
        recommendation = "No drift detected. Continue monitoring."
    else:
        drift_score = float(np.mean([psi_scores[f] for f in affected_features]))
        affected_frac = len(affected_features) / max(len(common_cols), 1)
        drift_type = "data_drift"
        if affected_frac > 0.5:
            recommendation = (
                "Widespread feature drift detected across >50% of features. "
                "Consider triggering model retraining."
            )
        else:
            recommendation = (
                f"Feature drift detected in: {', '.join(affected_features)}. "
                "Monitor closely and consider retraining if performance degrades."
            )

    return DriftDetectionResult(
        drift_detected=drift_detected,
        drift_score=round(drift_score, 6),
        drift_type=drift_type,
        affected_features=affected_features,
        psi_scores=psi_scores,
        ks_statistics=ks_statistics,
        recommendation=recommendation,
    )
