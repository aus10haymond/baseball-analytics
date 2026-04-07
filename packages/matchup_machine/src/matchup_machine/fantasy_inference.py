# src/fantasy_inference.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from . import config
from .build_dataset import OUTCOME_LABELS
from .train_hit_model import fill_missing_values  # reuse your imputer


def load_artifacts() -> Tuple[
    object,          # model
    list[str],       # feature_cols
    pd.DataFrame,    # pitcher_profiles
    Optional[pd.DataFrame],  # batter_profiles (may be None)
    pd.DataFrame,    # player_index
    pd.DataFrame,    # pa_proj
    pd.DataFrame,    # matchups
]:
    """
    Load:
      - XGBoost outcome model
      - training feature columns
      - pitcher / (optional) batter profiles
      - player index
      - projected PA for 2026
      - matchups.parquet (full PA-level dataset)

    Paths are resolved at call time (not import time) so that
    MM_ARTIFACTS_DIR env var set by the dashboard takes effect.
    """
    # Reload config so MM_ARTIFACTS_DIR env var is picked up if set
    # (env var may be set after the module was first imported)
    import importlib
    import matchup_machine.config as _config_module
    importlib.reload(_config_module)
    # Refresh the local reference so path constants below use the reloaded values
    from matchup_machine import config as config  # noqa: F811

    model_path = config.MODELS_DIR / "xgb_outcome_model.joblib"
    features_path = config.MODELS_DIR / "outcome_feature_cols.json"
    pitcher_profiles_path = config.PITCHER_PROFILES_DIR / "pitcher_profiles.parquet"
    batter_profiles_path = config.DATA_DIR / "batter_profiles.parquet"
    player_index_path = config.DATA_DIR / "player_index.csv"
    pa_proj_path = config.DATA_DIR / "batter_pa_projection_2026.parquet"
    matchups_path = config.MODELING_DIR / "matchups.parquet"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature cols JSON not found at {features_path}")
    if not pitcher_profiles_path.exists():
        raise FileNotFoundError(f"Pitcher profiles not found at {pitcher_profiles_path}")
    if not player_index_path.exists():
        raise FileNotFoundError(f"Player index CSV not found at {player_index_path}")
    if not pa_proj_path.exists():
        raise FileNotFoundError(f"PA projection parquet not found at {pa_proj_path}")
    if not matchups_path.exists():
        raise FileNotFoundError(f"matchups.parquet not found at {matchups_path}")

    model = joblib.load(model_path)

    with open(features_path, "r") as f:
        feature_cols = json.load(f)

    pitcher_profiles = pd.read_parquet(pitcher_profiles_path)

    if batter_profiles_path.exists():
        batter_profiles: Optional[pd.DataFrame] = pd.read_parquet(batter_profiles_path)
    else:
        batter_profiles = None

    player_index = pd.read_csv(player_index_path)
    pa_proj = pd.read_parquet(pa_proj_path)
    matchups = pd.read_parquet(matchups_path)

    return model, feature_cols, pitcher_profiles, batter_profiles, player_index, pa_proj, matchups


def _normalize(s: str) -> str:
    """Strip accent marks and lowercase for fuzzy name matching."""
    import unicodedata
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()


def find_player_id(player_index: pd.DataFrame, name_query: str) -> int:
    """
    Fuzzy lookup: find a player ID whose name contains the query (case-insensitive).
    Normalizes unicode (e.g. José → Jose) so accent-free input still matches.
    Ignores "Unknown ######" rows so only real-named players are considered.
    """
    known = player_index[~player_index["player_name"].astype(str).str.startswith("Unknown")].copy()

    normalized_query = _normalize(name_query)
    known["_norm"] = known["player_name"].astype(str).apply(_normalize)

    matches = known[known["_norm"].str.contains(normalized_query, na=False)]

    if matches.empty:
        raise ValueError(f"No player found matching {name_query!r}")

    if len(matches) == 1:
        return int(matches.iloc[0]["player_id"])

    print("Multiple player matches found:")
    for _, row in matches.iterrows():
        pid = row["player_id"]
        pname = row["player_name"]
        print(f"  {pid}: {pname}")

    choice = input("Enter player_id from above: ").strip()
    return int(choice)


def lookup_projected_pa(
    batter_id: int,
    pa_proj: pd.DataFrame,
    default_pa: int = 400,
) -> int:
    """
    Look up projected PA for 2026 for a given batter.
    Falls back to default_pa if not found.
    """
    row = pa_proj[pa_proj["batter"] == batter_id]
    if row.empty:
        return default_pa
    return int(row["projected_pa"].iloc[0])


def estimate_batter_outcome_probs_from_history(
    model,
    feature_cols: list[str],
    matchups: pd.DataFrame,
    batter_id: int,
    min_pas: int = 200,
    recent_only: bool = True,
    recent_start_year: int = 2024,
) -> Dict[str, float]:
    """
    Use the trained multiclass outcome model on the batter's *real* plate appearances
    to estimate an average outcome probability distribution.

    - filters to rows where outcome_id is not null (terminal PAs)
    - optionally restricts to recent seasons (e.g., 2024–2025)
    - runs model.predict_proba on those rows
    - returns the mean probability vector across all PAs

    This avoids synthetic "neutral" rows and uses actual contexts the batter saw.
    """

    df = matchups.copy()
    df = df[df["batter"] == batter_id]
    df = df[df["outcome_id"].notna()]

    if recent_only:
        df = df[df["date"].dt.year >= recent_start_year]

    if df.empty:
        raise ValueError(f"No historical plate appearances found for batter_id={batter_id}")

    if len(df) < min_pas:
        print(f"Warning: only {len(df)} PAs for batter_id={batter_id} (min_pas={min_pas})")

    # Align to feature cols used in training
    X = df.reindex(columns=feature_cols, fill_value=0)
    X = fill_missing_values(X)

    probs = model.predict_proba(X)  # shape: (num_samples, num_classes)
    avg_probs = probs.mean(axis=0)

    return {label: float(p) for label, p in zip(OUTCOME_LABELS, avg_probs)}


# (Optional) Keep these around if you want to use synthetic rows or direct per-matchup inference later.
def predict_outcome_probs_for_matchup(
    model,
    X: pd.DataFrame,
) -> Dict[str, float]:
    """
    Run the multiclass outcome model and map the probabilities to OUTCOME_LABELS.
    Returns: {label -> probability}
    """
    probs = model.predict_proba(X)[0]  # shape: (num_classes,)
    return {label: float(p) for label, p in zip(OUTCOME_LABELS, probs)}
