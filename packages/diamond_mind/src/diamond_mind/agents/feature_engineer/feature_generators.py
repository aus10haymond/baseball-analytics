"""
Feature generation library for the Feature Engineer Agent.

Provides pure functions for every generator category:
  - Rolling/window statistics
  - Pairwise interaction features (arithmetic)
  - Polynomial expansions
  - Lag features
  - Domain-specific baseball features

All functions accept pandas Series or DataFrame inputs and return a new
Series (or DataFrame for ``generate_baseball_features``).  No side-effects.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rolling / window features
# ---------------------------------------------------------------------------

WindowFunc = Literal["mean", "std", "max", "min", "sum"]

_WINDOW_FUNCS: Dict[str, callable] = {
    "mean": lambda r: r.mean(),
    "std": lambda r: r.std(),
    "max": lambda r: r.max(),
    "min": lambda r: r.min(),
    "sum": lambda r: r.sum(),
}


def rolling_stat(series: pd.Series, window: int, func: str = "mean") -> pd.Series:
    """
    Compute a rolling statistic over ``series``.

    Args:
        series: Input data.
        window: Rolling window size.
        func:   Aggregation function name ("mean", "std", "max", "min", "sum").

    Returns:
        Rolling statistic series (NaN for the first ``window-1`` positions).
    """
    if func not in _WINDOW_FUNCS:
        raise ValueError(f"Unknown window func {func!r}. Choose from {list(_WINDOW_FUNCS)}")
    roller = series.rolling(window=window, min_periods=1)
    result = _WINDOW_FUNCS[func](roller)
    result.name = f"{series.name}_roll{window}_{func}"
    return result


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

InteractionOp = Literal["mul", "div", "add", "sub"]


def interaction(s1: pd.Series, s2: pd.Series, operation: str = "mul") -> pd.Series:
    """
    Compute a pairwise interaction between two series.

    Args:
        s1, s2:    Input series (must be aligned by index).
        operation: One of "mul", "div", "add", "sub".

    Returns:
        Interaction feature series.
    """
    ops = {
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b.replace(0, np.nan),
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
    }
    if operation not in ops:
        raise ValueError(f"Unknown operation {operation!r}. Choose from {list(ops)}")
    n1 = s1.name or "feat1"
    n2 = s2.name or "feat2"
    result = ops[operation](s1, s2)
    result.name = f"{n1}_{operation}_{n2}"
    return result


# ---------------------------------------------------------------------------
# Polynomial features
# ---------------------------------------------------------------------------


def polynomial(series: pd.Series, degree: int = 2) -> pd.Series:
    """
    Raise ``series`` to ``degree`` (e.g. quadratic or cubic expansion).

    Args:
        series: Input data.
        degree: Exponent (2 or 3).

    Returns:
        Series of ``series ** degree``.
    """
    if degree < 2:
        raise ValueError("Degree must be >= 2")
    result = series ** degree
    result.name = f"{series.name}_pow{degree}"
    return result


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------


def lag_feature(series: pd.Series, lag: int = 1) -> pd.Series:
    """
    Shift ``series`` back by ``lag`` periods.

    Args:
        series: Input data.
        lag:    Number of periods to shift.

    Returns:
        Lagged series (NaN at the start).
    """
    if lag < 1:
        raise ValueError("Lag must be >= 1")
    result = series.shift(lag)
    result.name = f"{series.name}_lag{lag}"
    return result


# ---------------------------------------------------------------------------
# Baseball-domain features
# ---------------------------------------------------------------------------


def iso_power(slg: pd.Series, avg: pd.Series) -> pd.Series:
    """Isolated Power = SLG - AVG."""
    result = slg - avg
    result.name = "iso_power"
    return result


def babip(
    hits: pd.Series,
    home_runs: pd.Series,
    at_bats: pd.Series,
    strikeouts: pd.Series,
    sac_flies: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Batting Average on Balls In Play.

    BABIP = (H - HR) / (AB - K - HR + SF)
    """
    sf = sac_flies if sac_flies is not None else pd.Series(0, index=hits.index)
    denom = (at_bats - strikeouts - home_runs + sf).replace(0, np.nan)
    result = (hits - home_runs) / denom
    result.name = "babip"
    return result


def k_pct(strikeouts: pd.Series, plate_appearances: pd.Series) -> pd.Series:
    """Strikeout percentage = K / PA."""
    denom = plate_appearances.replace(0, np.nan)
    result = strikeouts / denom
    result.name = "k_pct"
    return result


def bb_pct(walks: pd.Series, plate_appearances: pd.Series) -> pd.Series:
    """Walk percentage = BB / PA."""
    denom = plate_appearances.replace(0, np.nan)
    result = walks / denom
    result.name = "bb_pct"
    return result


def contact_rate(
    at_bats: pd.Series, strikeouts: pd.Series
) -> pd.Series:
    """Contact rate = (AB - K) / AB."""
    denom = at_bats.replace(0, np.nan)
    result = (at_bats - strikeouts) / denom
    result.name = "contact_rate"
    return result


def obp_to_slg_ratio(obp: pd.Series, slg: pd.Series) -> pd.Series:
    """OBP-to-SLG ratio (useful for identifying contact vs. power hitters)."""
    denom = slg.replace(0, np.nan)
    result = obp / denom
    result.name = "obp_slg_ratio"
    return result


# ---------------------------------------------------------------------------
# Bulk generation helpers
# ---------------------------------------------------------------------------

_ROLLING_WINDOWS = [3, 5, 10]
_ROLLING_FUNCS = ["mean", "std", "max"]
_INTERACTION_OPS = ["mul", "div", "add", "sub"]
_POLY_DEGREES = [2, 3]
_LAG_PERIODS = [1, 3, 5]


def generate_rolling_candidates(
    df: pd.DataFrame,
    numeric_cols: List[str],
    windows: List[int] = _ROLLING_WINDOWS,
    funcs: List[str] = _ROLLING_FUNCS,
) -> List[Tuple[str, pd.Series]]:
    """Generate all rolling-stat feature candidates."""
    candidates: List[Tuple[str, pd.Series]] = []
    for col in numeric_cols:
        for w in windows:
            for fn in funcs:
                feat = rolling_stat(df[col], window=w, func=fn)
                candidates.append((str(feat.name), feat))
    return candidates


def generate_interaction_candidates(
    df: pd.DataFrame,
    numeric_cols: List[str],
    ops: List[str] = _INTERACTION_OPS,
) -> List[Tuple[str, pd.Series]]:
    """Generate pairwise interaction candidates (all unique column pairs)."""
    candidates: List[Tuple[str, pd.Series]] = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1 :]:
            for op in ops:
                try:
                    feat = interaction(df[c1], df[c2], operation=op)
                    if not feat.isnull().all():
                        candidates.append((str(feat.name), feat))
                except Exception:
                    pass
    return candidates


def generate_polynomial_candidates(
    df: pd.DataFrame,
    numeric_cols: List[str],
    degrees: List[int] = _POLY_DEGREES,
) -> List[Tuple[str, pd.Series]]:
    """Generate polynomial expansion candidates."""
    candidates: List[Tuple[str, pd.Series]] = []
    for col in numeric_cols:
        for deg in degrees:
            feat = polynomial(df[col], degree=deg)
            candidates.append((str(feat.name), feat))
    return candidates


def generate_lag_candidates(
    df: pd.DataFrame,
    numeric_cols: List[str],
    lags: List[int] = _LAG_PERIODS,
) -> List[Tuple[str, pd.Series]]:
    """Generate lag feature candidates."""
    candidates: List[Tuple[str, pd.Series]] = []
    for col in numeric_cols:
        for lag in lags:
            feat = lag_feature(df[col], lag=lag)
            candidates.append((str(feat.name), feat))
    return candidates


def generate_baseball_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to compute all baseball domain features that can be derived
    from columns present in ``df``.

    Silently skips features whose required columns are missing.
    Returns a DataFrame containing only the successfully computed features.
    """
    result: Dict[str, pd.Series] = {}
    cols = set(df.columns)

    def _has(*required: str) -> bool:
        return all(c in cols for c in required)

    if _has("ops", "batting_avg"):
        # SLG ≈ OPS - OBP; approximate ISO from OPS and batting_avg
        result["iso_approx"] = df["ops"] * 0.6 - df["batting_avg"]
        result["iso_approx"].name = "iso_approx"

    if _has("slugging", "batting_avg"):
        result["iso_power"] = iso_power(df["slugging"], df["batting_avg"])

    if _has("hits", "home_runs", "at_bats", "strikeouts"):
        result["babip"] = babip(df["hits"], df["home_runs"], df["at_bats"], df["strikeouts"])

    if _has("strikeouts", "plate_appearances"):
        result["k_pct"] = k_pct(df["strikeouts"], df["plate_appearances"])

    if _has("walks", "plate_appearances"):
        result["bb_pct"] = bb_pct(df["walks"], df["plate_appearances"])

    if _has("at_bats", "strikeouts"):
        result["contact_rate"] = contact_rate(df["at_bats"], df["strikeouts"])

    if _has("obp", "slugging"):
        result["obp_slg_ratio"] = obp_to_slg_ratio(df["obp"], df["slugging"])

    if _has("batting_avg", "ops"):
        # Power factor: ops relative to batting average
        denom = df["batting_avg"].replace(0, np.nan)
        pf = df["ops"] / denom
        pf.name = "power_factor"
        result["power_factor"] = pf

    if not result:
        return pd.DataFrame(index=df.index)

    return pd.DataFrame(result, index=df.index)


def generate_all_candidates(
    df: pd.DataFrame,
    max_per_type: int = 20,
    include_baseball: bool = True,
) -> List[Tuple[str, pd.Series]]:
    """
    Generate feature candidates from all generator types.

    Args:
        df:              Source DataFrame.
        max_per_type:    Cap candidates per generator category.
        include_baseball: Include baseball-domain features.

    Returns:
        List of (feature_name, feature_series) tuples.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return []

    all_candidates: List[Tuple[str, pd.Series]] = []

    for bucket in [
        generate_rolling_candidates(df, numeric_cols),
        generate_interaction_candidates(df, numeric_cols),
        generate_polynomial_candidates(df, numeric_cols),
        generate_lag_candidates(df, numeric_cols),
    ]:
        all_candidates.extend(bucket[:max_per_type])

    if include_baseball:
        baseball_df = generate_baseball_features(df)
        for col in baseball_df.columns:
            all_candidates.append((col, baseball_df[col]))

    # Deduplicate by name (keep first occurrence)
    seen: set = set()
    unique: List[Tuple[str, pd.Series]] = []
    for name, feat in all_candidates:
        if name not in seen:
            seen.add(name)
            unique.append((name, feat))

    return unique
