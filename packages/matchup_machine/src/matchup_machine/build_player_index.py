# src/build_player_index.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pybaseball import playerid_reverse_lookup

from . import config


def load_matchups() -> pd.DataFrame:
    """
    Load the modeling dataset that contains batter/pitcher IDs.
    """
    path = config.MODELING_DIR / "matchups.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"matchups.parquet not found at {path}. "
            "Run build_dataset.py first."
        )
    df = pd.read_parquet(path)
    return df


def extract_unique_player_ids(df: pd.DataFrame):
    """
    Extract unique batter and pitcher IDs from the matchups dataset.
    Returns: (batter_ids, pitcher_ids, all_ids)
    """
    batter_ids = (
        df["batter"]
        .dropna()
        .astype("int64")
        .unique()
        .tolist()
    )
    pitcher_ids = (
        df["pitcher"]
        .dropna()
        .astype("int64")
        .unique()
        .tolist()
    )

    all_ids = sorted(set(batter_ids) | set(pitcher_ids))

    print(f"Unique batters:  {len(batter_ids):,}")
    print(f"Unique pitchers: {len(pitcher_ids):,}")
    print(f"Total unique players: {len(all_ids):,}")

    return batter_ids, pitcher_ids, all_ids


def lookup_player_metadata(all_ids: list[int]) -> pd.DataFrame:
    """
    Use pybaseball.playerid_reverse_lookup to fetch names & metadata
    for MLBAM IDs.
    """
    if not all_ids:
        raise ValueError("No player IDs provided for lookup.")

    print("Looking up player metadata via pybaseball...")
    # key_type="mlbam" because Statcast uses MLBAM IDs
    meta = playerid_reverse_lookup(all_ids, key_type="mlbam")

    # Typical columns include:
    # key_mlbam, name_first, name_last, bats, throws, mlb_played_first, mlb_played_last, ...
    meta = meta.rename(columns={"key_mlbam": "player_id"})
    meta["player_id"] = meta["player_id"].astype("int64")

    # Build full name
    meta["player_name"] = (
        meta["name_first"].fillna("") + " " + meta["name_last"].fillna("")
    ).str.strip()

    return meta


def build_player_index(
    batter_ids: list[int],
    pitcher_ids: list[int],
    all_ids: list[int],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine raw ID sets and metadata into a player_index table.

    Columns:
      - player_id
      - player_name
      - is_batter (bool)
      - is_pitcher (bool)
      - role (one of "batter", "pitcher", "two-way")
      - bats (if available)
      - throws (if available)
    """
    index_df = pd.DataFrame({"player_id": all_ids})
    index_df["player_id"] = index_df["player_id"].astype("int64")

    # Merge metadata (left join so all IDs are preserved)
    index_df = index_df.merge(meta, on="player_id", how="left")

    # Basic flags
    batter_set = set(batter_ids)
    pitcher_set = set(pitcher_ids)

    index_df["is_batter"] = index_df["player_id"].isin(batter_set)
    index_df["is_pitcher"] = index_df["player_id"].isin(pitcher_set)

    def role_row(row) -> str:
        if row["is_batter"] and row["is_pitcher"]:
            return "two-way"
        if row["is_batter"]:
            return "batter"
        if row["is_pitcher"]:
            return "pitcher"
        return "unknown"

    index_df["role"] = index_df.apply(role_row, axis=1)

    # Ensure we always have some name string
    # Fall back to "Unknown <id>" if name is missing
    index_df["player_name"] = index_df["player_name"].fillna(
        index_df["player_id"].apply(lambda x: f"Unknown {x}")
    )

    # Keep only useful columns
    cols_to_keep = [
        "player_id",
        "player_name",
        "is_batter",
        "is_pitcher",
        "role",
    ]

    # Add bats/throws if available from pybaseball metadata
    for col in ["bats", "throws"]:
        if col in index_df.columns:
            cols_to_keep.append(col)

    index_df = index_df[cols_to_keep].sort_values("player_name").reset_index(drop=True)

    return index_df


def main():
    print("Loading matchups dataset...")
    df = load_matchups()

    print("Extracting unique player IDs...")
    batter_ids, pitcher_ids, all_ids = extract_unique_player_ids(df)

    print("Looking up player metadata...")
    meta = lookup_player_metadata(all_ids)

    print("Building player index table...")
    player_index = build_player_index(batter_ids, pitcher_ids, all_ids, meta)

    out_path = config.DATA_DIR / "player_index.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    player_index.to_csv(out_path, index=False)
    print(f"Saved player index to {out_path} ({len(player_index):,} players)")


if __name__ == "__main__":
    main()
