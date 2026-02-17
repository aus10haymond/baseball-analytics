from pathlib import Path
from typing import Iterable

import pandas as pd
from . import config

RAW_PATTERN = "statcast_*.parquet"
CLEAN_NAME = "statcast_clean_{:04d}_{:02d}.parquet"

COLUMN_RENAMES = {
    "game_date": "date",
    "events": "event",
    "stand": "batter_stand",
    "p_throws": "pitcher_hand",
    "release_speed": "release_vel",
    "release_spin_rate": "spin_rate",
    "release_extension": "extension",
}

COLUMNS_TO_KEEP = [
    "date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "batter_stand",
    "pitcher_hand",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "balls",
    "strikes",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "pitch_type",
    "type",
    "description",
    "event",
    "zone",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "release_vel",
    "spin_rate",
    "extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "pfx_x",
    "pfx_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "effective_speed",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "estimated_woba_using_speedangle",
    "bb_type",
    "hc_x",
    "hc_y",
]

REQUIRED_FOR_VALID = [
    "batter",
    "pitcher",
    "pitch_type",
    "plate_x",
    "plate_z",
]

def list_raw_month_files() -> list[Path]:
    return sorted(config.RAW_DIR.glob(RAW_PATTERN))

def load_raw_month(path: Path):
    return pd.read_parquet(path)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_RENAMES)
    keep = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    df = df.loc[:, keep].copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float", "Float32", "Float64"]).columns
    int_cols = df.select_dtypes(include=["int", "Int64", "int64"]).columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    
    for col in float_cols:
        df[col] = df[col].astype("float32")
    for col in int_cols:
        df[col] = df[col].astype("Int32")
    for col in obj_cols:
        df[col] = df[col].astype("category")
    return df

def filter_valid_pitches(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=[c for c in REQUIRED_FOR_VALID if c in df.columns]).reset_index(drop=True)

def save_clean_month(df: pd.DataFrame, year: int, month: int) -> None:
    output_path = config.PROCESSED_DIR / CLEAN_NAME.format(year, month)
    df.to_parquet(output_path, index=False)
    print(f"Saved clean file: {output_path} ({len(df):,} rows)")

def clean_all_months() -> None:
    for raw_path in list_raw_month_files():
        print(f"Cleaning {raw_path.name}...")
        
        stem_parts = raw_path.stem.split("_")
        if len(stem_parts) < 3:
            print(f"Skipping malformed filename: {raw_path.name}")
            continue

        year, month = int(stem_parts[1]), int(stem_parts[2])
        output_path = config.PROCESSED_DIR / CLEAN_NAME.format(year, month)

        # Skip if already cleaned
        if output_path.exists():
            print(f"  Skipping existing clean file: {output_path.name}")
            continue

        # Load raw
        df = load_raw_month(raw_path)

        # Normalize
        df = normalize_columns(df)

        # Downcast
        df = downcast_dtypes(df)

        # Filter invalid pitches
        df = filter_valid_pitches(df)

        print(f"  Rows after cleaning: {len(df):,}")

        if df.empty:
            print(f"  No valid rows after cleaning {raw_path.name}; skipping.")
            continue

        # Save
        save_clean_month(df, year, month)
        print(f"  Saved clean file: {output_path}")


if __name__ == "__main__":
    clean_all_months()
