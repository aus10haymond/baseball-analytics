# src/build_batter_pa_projection.py

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import config


def load_terminal_pas() -> pd.DataFrame:
    """
    Load matchups.parquet and filter to terminal pitches (rows with outcome_id).
    Each row here represents one plate appearance.
    """
    path = config.MODELING_DIR / "matchups.parquet"
    df = pd.read_parquet(path)
    df = df[df["outcome_id"].notna()].copy()
    df["season"] = df["date"].dt.year
    return df


def compute_pa_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PA counts per batter per season.
    Returns columns: [batter, season, pa].
    """
    pa_counts = (
        df.groupby(["batter", "season"], as_index=False)
          .size()
          .rename(columns={"size": "pa"})
    )
    return pa_counts


def project_pa_for_next_season(pa_history: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """
    For each batter, compute a projected PA for target_season using the last
    1–3 seasons of PA history with recency weighting.

    Example weighting if 3 seasons:
      oldest: 0.1, middle: 0.3, latest: 0.6
    """
    proj_rows = []

    for batter, grp in pa_history.groupby("batter"):
        grp = grp.sort_values("season")

        seasons = grp["season"].tolist()
        pas = grp["pa"].tolist()
        n = len(pas)

        if n == 0:
            continue

        if n >= 3:
            # use last 3 seasons
            pas = pas[-3:]
            n = 3
            base_weights = np.array([0.1, 0.3, 0.6])
            weights = base_weights[-n:]
        elif n == 2:
            # 40% older, 60% latest
            weights = np.array([0.4, 0.6])
        else:  # n == 1
            weights = np.array([1.0])

        weights = weights / weights.sum()
        proj_pa = int(round(float(np.dot(pas, weights))))

        proj_rows.append(
            {
                "batter": batter,
                "projected_season": target_season,
                "projected_pa": proj_pa,
            }
        )

    proj_df = pd.DataFrame(proj_rows)
    return proj_df


def main():
    print("Loading terminal plate appearances...")
    df = load_terminal_pas()

    print("Computing PA history per batter...")
    pa_history = compute_pa_history(df)

    # Your historical seasons are 2023–2025; we're projecting 2026:
    target_season = 2026
    print(f"Projecting PA for season {target_season}...")
    proj_df = project_pa_for_next_season(pa_history, target_season)

    out_path = config.DATA_DIR / f"batter_pa_projection_{target_season}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proj_df.to_parquet(out_path, index=False)
    print(f"Saved batter PA projections to {out_path} ({len(proj_df):,} batters)")


if __name__ == "__main__":
    main()
