from pathlib import Path
from typing import List

import pandas as pd

from . import config

# File patterns / names
CLEAN_PATTERN = "statcast_clean_*.parquet"
PARTIALS_NAME = "pitcher_partials_{:04d}_{:02d}.parquet"
PROFILES_NAME = "pitcher_profiles.parquet"


def list_clean_month_files() -> List[Path]:
    """
    Return a sorted list of all cleaned month Parquet files in data/processed/.
    """
    return sorted(config.PROCESSED_DIR.glob(CLEAN_PATTERN))


def compute_monthly_pitcher_aggregates(clean_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(clean_path)

    stem_parts = clean_path.stem.split("_")
    if len(stem_parts) >= 4:
        year, month = int(stem_parts[2]), int(stem_parts[3])
    else:
        first_date = pd.to_datetime(df["date"]).iloc[0]
        year, month = int(first_date.year), int(first_date.month)

    # Ensure core fields are present.
    df = df.dropna(subset=["pitcher", "pitch_type"])

    # FIX 1 — enforce clean dtypes for grouping keys
    df["pitcher"] = df["pitcher"].astype("int32")
    df["pitch_type"] = df["pitch_type"].astype(str)

    # FIX 2 — enforce clean numeric types before aggregation
    numeric_cols = ["release_vel", "spin_rate", "plate_x", "plate_z"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # Now grouping is safe
    grouped = (
        df.groupby(["pitcher", "pitch_type"], as_index=False)
        .agg(
            pitch_count=("pitch_type", "size"),
            avg_release_vel=("release_vel", "mean"),
            avg_spin_rate=("spin_rate", "mean"),
            avg_plate_x=("plate_x", "mean"),
            avg_plate_z=("plate_z", "mean"),
        )
    )

    grouped["pitch_type"] = grouped["pitch_type"].astype(str)
    grouped["year"] = year
    grouped["month"] = month

    return grouped[
        [
            "pitcher",
            "pitch_type",
            "year",
            "month",
            "pitch_count",
            "avg_release_vel",
            "avg_spin_rate",
            "avg_plate_x",
            "avg_plate_z",
        ]
    ]

def save_partial_pitcher_aggregates(df: pd.DataFrame, year: int, month: int) -> None:
    """
    Save monthly pitcher aggregates to data/pitcher_profiles/pitcher_partials_YYYY_MM.parquet.
    """
    output_path = config.PITCHER_PROFILES_DIR / PARTIALS_NAME.format(year, month)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved partials for {year}-{month:02d} to {output_path} ({len(df):,} rows)")


def build_all_monthly_partials() -> None:
    """
    For each cleaned month file:
      - compute_monthly_pitcher_aggregates
      - save_partial_pitcher_aggregates
      - skip if partial already exists
    """
    for clean_path in list_clean_month_files():
        print(f"Building partials for {clean_path.name}...")

        stem_parts = clean_path.stem.split("_")
        if len(stem_parts) < 4:
            print(f"  Skipping malformed filename: {clean_path.name}")
            continue

        # statcast_clean_YYYY_MM → ["statcast", "clean", "YYYY", "MM"]
        year, month = int(stem_parts[2]), int(stem_parts[3])

        partials_path = config.PITCHER_PROFILES_DIR / PARTIALS_NAME.format(year, month)
        if partials_path.exists():
            print(f"  Skipping existing partials: {partials_path.name}")
            continue

        monthly_df = compute_monthly_pitcher_aggregates(clean_path)
        if monthly_df.empty:
            print(f"  No pitcher data for {year}-{month:02d}, skipping save.")
            continue

        save_partial_pitcher_aggregates(monthly_df, year, month)


def list_partials_files() -> List[Path]:
    """
    Return a sorted list of pitcher partials parquet files.
    """
    return sorted(config.PITCHER_PROFILES_DIR.glob("pitcher_partials_*.parquet"))


def load_all_partials() -> pd.DataFrame:
    """
    Load all monthly partials and concatenate into one DataFrame.
    """
    partial_paths = list_partials_files()
    if not partial_paths:
        raise FileNotFoundError(
            "No pitcher partial files found. Run build_all_monthly_partials() first."
        )

    dfs = [pd.read_parquet(p) for p in partial_paths]
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(partial_paths)} partial files with {len(combined):,} rows")
    return combined


def aggregate_pitcher_profiles(partials: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly pitcher partials into a single pitcher profile table.
    For each pitcher, compute:
      - total_pitch_count
      - pitch_mix (pitch_type percentages pivoted into columns)
      - overall average release_vel, spin_rate, plate_x, plate_z

    Return one row per pitcher with numeric features.
    """
    if partials.empty:
        return pd.DataFrame()

    # Total pitches per pitcher.
    pitcher_totals = (
        partials.groupby("pitcher", as_index=False)["pitch_count"]
        .sum()
        .rename(columns={"pitch_count": "total_pitches"})
    )

    # Pitch-type totals for pitch mix.
    type_totals = partials.groupby(["pitcher", "pitch_type"], as_index=False)["pitch_count"].sum()

    mix = type_totals.pivot(index="pitcher", columns="pitch_type", values="pitch_count").fillna(0)

    # Avoid division by zero if a row somehow sums to 0.
    row_sums = mix.sum(axis=1).replace(0, 1)
    mix_pct = mix.div(row_sums, axis=0).add_suffix("_pct")
    mix_pct.reset_index(inplace=True)

    # Weighted averages of velocity/spin/location across pitch types.
    weighted = partials.copy()
    metrics = ["avg_release_vel", "avg_spin_rate", "avg_plate_x", "avg_plate_z"]

    # Ensure metric columns are float for safe arithmetic.
    for col in metrics:
        weighted[col] = weighted[col].astype("float32")

    for col in metrics:
        weighted[col] = weighted[col] * weighted["pitch_count"]

    weighted_sum = weighted.groupby("pitcher", as_index=False)[metrics + ["pitch_count"]].sum()
    for col in metrics:
        weighted_sum[col] = weighted_sum[col] / weighted_sum["pitch_count"]
    weighted_sum = weighted_sum.drop(columns=["pitch_count"])

    # Merge everything into a single DataFrame keyed by 'pitcher'.
    profiles = pitcher_totals.merge(weighted_sum, on="pitcher", how="left")
    profiles = profiles.merge(mix_pct, on="pitcher", how="left")

    return profiles


def save_pitcher_profiles(df: pd.DataFrame) -> None:
    """
    Save the final pitcher profiles table to data/pitcher_profiles/pitcher_profiles.parquet.
    """
    output_path = config.PITCHER_PROFILES_DIR / PROFILES_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved pitcher profiles to {output_path} ({len(df):,} pitchers)")


def main() -> None:
    """
    Orchestrate pitcher tendencies build:
      - build monthly partials if needed
      - load all partials
      - aggregate into pitcher profiles
      - save result
    """
    config.ensure_directories()
    build_all_monthly_partials()
    partials = load_all_partials()
    profiles = aggregate_pitcher_profiles(partials)
    save_pitcher_profiles(profiles)


if __name__ == "__main__":
    main()
