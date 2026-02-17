from calendar import monthrange
from datetime import date, timedelta
from typing import List, Tuple
from pybaseball import statcast
import pandas as pd
import time

import config

def month_ranges(start: date, end: date) -> List[Tuple[date, date]]:
    if end < start:
        raise ValueError("end date must be on or after start date")

    ranges: List[Tuple[date, date]] = []
    current_start = start

    while current_start <= end:
        last_day = monthrange(current_start.year, current_start.month)[1]
        current_end = date(current_start.year, current_start.month, last_day)
        if current_end > end:
            current_end = end

        ranges.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)

    return ranges


def fetch_statcast_for_range(start: date, end: date):
    for attempt in range(3):
        try:
            df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            print(f"Error fetching {start} → {end}: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

    print(f"Failed to fetch {start} → {end} after 3 attempts. Returning empty DataFrame.")
    return pd.DataFrame()


def save_raw_month(df, year: int, month: int) -> None:
    output_path = config.RAW_DIR / f"statcast_{year:04d}_{month:02d}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {output_path} ({len(df):,} rows)")


def collect_all_months() -> None:
    for start_dt, end_dt in month_ranges(config.STATCAST_START, config.STATCAST_END):
        year, month = start_dt.year, start_dt.month
        output_path = config.RAW_DIR / f"statcast_{year:04d}_{month:02d}.parquet"

        if output_path.exists():
            print(f"Skipping {output_path.name} (already exists)")
            continue

        print(f"Fetching {year}-{month:02d} ({start_dt} to {end_dt})...")
        df = fetch_statcast_for_range(start_dt, end_dt)

        if df.empty:
            print(f"No data for {year}-{month:02d}, skipping save.")
            continue

        save_raw_month(df, year, month)

def collect_test_week() -> None:
    test_start = date(2023, 4, 1)
    test_end = date(2023, 4, 7)

    for start_dt, end_dt in month_ranges(test_start, test_end):
        year, month = start_dt.year, start_dt.month
        output_path = config.RAW_DIR / f"statcast_{year:04d}_{month:02d}.parquet"

        if output_path.exists():
            print(f"Skipping {output_path.name} (already exists)")
            continue

        print(f"Fetching {year}-{month:02d} ({start_dt} to {end_dt})...")
        df = fetch_statcast_for_range(start_dt, end_dt)

        if df.empty:
            print(f"No data for {year}-{month:02d}, skipping save.")
            continue

        save_raw_month(df, year, month)

def main() -> None:
    print("Starting data collection...")
    config.ensure_directories()
    collect_all_months()
    #collect_test_week()

if __name__ == "__main__":
    main()