from pathlib import Path
from typing import List
import pandas as pd

import config

def list_clean_month_files() -> List[Path]:
    """Return sorted list of cleaned month parquet files."""
    return sorted(config.PROCESSED_DIR.glob("statcast_clean_*.parquet"))

def load_all_clean() -> pd.DataFrame:
    """Load, concatenate, and sort all cleaned Statcast months."""
    files = list_clean_month_files()
    if not files:
        raise FileNotFoundError("No cleaned month files found. Run clean_month.py first.")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["date", "game_pk", "at_bat_number", "pitch_number"])
    return df.reset_index(drop=True)

def add_hit_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary hit indicator based on Statcast event string."""
    hit_events = {"single", "double", "triple", "home_run"}
    df["is_hit"] = df["event"].isin(hit_events).astype("int8")
    return df

OUTCOME_LABELS = [
    "single",
    "double",
    "triple",
    "home_run",
    "walk",
    "strikeout",
    "ball_in_play_out",
    "other",
]

OUTCOME_TO_ID = {label: i for i, label in enumerate(OUTCOME_LABELS)}


def add_outcome_label(df: pd.DataFrame) -> pd.DataFrame:
    def map_event(ev: str):
        # If no event recorded, this is a non-terminal pitch: no outcome label
        if pd.isna(ev):
            return None

        ev = str(ev).lower()

        if ev == "single":
            return "single"
        if ev == "double":
            return "double"
        if ev == "triple":
            return "triple"
        if ev == "home_run":
            return "home_run"

        if ev in {"walk", "intent_walk"}:
            return "walk"

        if "strikeout" in ev:
            return "strikeout"

        if ev in {
            "field_out",
            "grounded_into_double_play",
            "force_out",
            "double_play",
            "field_error",
            "sac_fly",
            "sac_bunt",
        }:
            return "ball_in_play_out"

        if ev in {"hit_by_pitch", "catcher_interf"}:
            return "other"

        return "other"

    df["outcome"] = df["event"].map(map_event)
    df["outcome_id"] = df["outcome"].map(OUTCOME_TO_ID).astype("Int8")

    return df


def load_pitcher_profiles() -> pd.DataFrame:
    """Load aggregated pitcher profiles."""
    path = config.PITCHER_PROFILES_DIR / "pitcher_profiles.parquet"
    if not path.exists():
        raise FileNotFoundError("Pitcher profiles not found. Run build_pitcher_tendencies.py first.")

    return pd.read_parquet(path)

def merge_pitcher_profiles(df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Left-join pitcher profiles onto pitch-level data."""
    return df.merge(profiles, on="pitcher", how="left")

def add_batter_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["batter", "date", "game_pk", "at_bat_number", "pitch_number"])

    if "launch_speed" in df.columns:
        df["rolling_launch_speed"] = (
            df.groupby("batter")["launch_speed"]
              .rolling(50, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )

    if "launch_angle" in df.columns:
        df["rolling_launch_angle"] = (
            df.groupby("batter")["launch_angle"]
              .rolling(50, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )

    df["rolling_hit_rate"] = (
        df.groupby("batter")["is_hit"]
          .rolling(100, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )

    return df

def add_matchup_handedness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add numeric features for batter/pitcher handedness matchup.
      - is_same_hand: 1 if batter and pitcher use same hand, else 0
      - is_lefty_batter: 1 if batter stands left
      - is_lefty_pitcher: 1 if pitcher throws left
    """
    # Make sure we’re working with strings (batter_stand and pitcher_hand are categories)
    df["batter_stand"] = df["batter_stand"].astype(str)
    df["pitcher_hand"] = df["pitcher_hand"].astype(str)

    df["is_same_hand"] = (df["batter_stand"] == df["pitcher_hand"]).astype("int8")
    df["is_lefty_batter"] = (df["batter_stand"] == "L").astype("int8")
    df["is_lefty_pitcher"] = (df["pitcher_hand"] == "L").astype("int8")

    return df

def add_pitch_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add count/baserunner context flags."""
    df["is_two_strike"] = (df["strikes"] == 2).astype("int8")
    df["is_two_ball"] = (df["balls"] == 2).astype("int8")
    df["is_full_count"] = ((df["balls"] == 3) & (df["strikes"] == 2)).astype("int8")

    df["has_runner_on"] = (
        df[["on_1b", "on_2b", "on_3b"]].notna().any(axis=1).astype("int8")
    )

    return df

FINAL_COLUMNS = [
    # IDs
    "date", "game_pk", "at_bat_number", "pitch_number",
    "batter", "pitcher",

    # Batter features
    "rolling_launch_speed",
    "rolling_launch_angle",
    "rolling_hit_rate",

    # Pitcher profile features
    "total_pitches",
    "avg_release_vel",
    "avg_spin_rate",
    "avg_plate_x",
    "avg_plate_z",

    # Handedness features
    "is_same_hand",
    "is_lefty_batter",
    "is_lefty_pitcher",

    # Context
    "inning", "balls", "strikes", "outs_when_up", "has_runner_on",
    "is_two_strike", "is_two_ball", "is_full_count",

    # Pitch info
    "pitch_type", "release_vel", "spin_rate", "plate_x", "plate_z",

    # Targets
    "is_hit",
    "outcome",
    "outcome_id",
]

def finalize_and_save(df: pd.DataFrame) -> None:
    """Keep modeling columns, append pitch-mix pct columns, and write parquet."""
    cols = [c for c in FINAL_COLUMNS if c in df.columns] + \
           [c for c in df.columns if c.endswith("_pct")]  # pitch mix

    final = df[cols].copy()
    final_path = config.MODELING_DIR / "matchups.parquet"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(final_path, index=False)
    print(f"Saved final dataset: {final_path} ({len(final):,} rows)")

def main():
    config.ensure_directories()

    print("Loading cleaned data...")
    df = load_all_clean()

    print("Adding hit label...")
    df = add_hit_label(df)

    print("Adding hit label...")
    df = add_hit_label(df)

    print("Adding multiclass outcome label...")
    df = add_outcome_label(df)

    print("Adding pitch context features...")
    df = add_pitch_context(df)

    print("Loading pitcher profiles...")
    profiles = load_pitcher_profiles()
    df = merge_pitcher_profiles(df, profiles)

    print("Adding handedness features...")
    df = add_matchup_handedness(df)

    print("Computing rolling batter features...")
    df = add_batter_rolling(df)

    print("Saving final dataset...")
    finalize_and_save(df)

    print("Done!")


if __name__ == "__main__":
    main()
