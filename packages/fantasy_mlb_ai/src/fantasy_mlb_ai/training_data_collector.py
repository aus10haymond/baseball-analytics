import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from pybaseball import statcast_batter, playerid_lookup, batting_stats
from pybaseball import cache
import argparse

# Enable pybaseball's request caching
cache.enable()

# === Paths ===
DATA_DIR = "data/model_training"
os.makedirs(DATA_DIR, exist_ok=True)

DATA_OUT_PATH = os.path.join(DATA_DIR, "collected_data.csv")
PROCESSED_LOG_PATH = os.path.join(DATA_DIR, "processed_log.csv")
PLAYER_ID_CACHE_PATH = os.path.join(DATA_DIR, "player_id_cache.json")

# === Fantasy Point Scoring ===
FANTASY_SCORING = {
    "H": 1,
    "2B": 1,
    "3B": 2,
    "HR": 4,
    "RBI": 1,
    "R": 1,
    "BB": 1,
    "SB": 2
}

# === Utilities ===
def calculate_fantasy_points(row):
    return (
        row.get("H", 0) * FANTASY_SCORING["H"]
        + row.get("2B", 0) * FANTASY_SCORING["2B"]
        + row.get("3B", 0) * FANTASY_SCORING["3B"]
        + row.get("HR", 0) * FANTASY_SCORING["HR"]
        + row.get("RBI", 0) * FANTASY_SCORING["RBI"]
        + row.get("R", 0) * FANTASY_SCORING["R"]
        + row.get("BB", 0) * FANTASY_SCORING["BB"]
        + row.get("SB", 0) * FANTASY_SCORING["SB"]
    )

def load_processed_log():
    if os.path.exists(PROCESSED_LOG_PATH):
        return set(pd.read_csv(PROCESSED_LOG_PATH).apply(lambda row: f"{row['WeekStart']}|{row['Name']}", axis=1))
    return set()

def save_processed_entry(week_start, name):
    file_exists = os.path.exists(PROCESSED_LOG_PATH)
    with open(PROCESSED_LOG_PATH, "a") as f:
        if not file_exists:
            f.write("WeekStart,Name\n")
        f.write(f"{week_start},{name}\n")

def load_player_id_cache():
    if os.path.exists(PLAYER_ID_CACHE_PATH):
        with open(PLAYER_ID_CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_player_id_cache(cache_dict):
    with open(PLAYER_ID_CACHE_PATH, "w") as f:
        json.dump(cache_dict, f)

def get_player_id(name, cache_dict):
    if name in cache_dict:
        return cache_dict[name]
    try:
        last = name.split()[-1]
        first = " ".join(name.split()[:-1])
        result = playerid_lookup(last, first)
        if not result.empty:
            mlbam_id = int(result.iloc[0]['key_mlbam'])
            cache_dict[name] = mlbam_id
            return mlbam_id
    except Exception:
        pass
    return None

# === Main Collector ===
def collect_weekly_data(year, month, day):
    week_start = datetime(year, month, day)
    week_end = week_start + timedelta(days=6)
    week_str = week_start.strftime("%Y-%m-%d")
    print(f"Collecting data for week: {week_str} to {week_end.strftime('%Y-%m-%d')}")

    processed = load_processed_log()
    player_id_cache = load_player_id_cache()
    players = batting_stats(year, qual=0)['Name'].dropna().unique().tolist()

    for name in players:
        key = f"{week_str}|{name}"
        if key in processed:
            continue

        player_id = get_player_id(name, player_id_cache)
        if not player_id:
            continue

        try:
            df = statcast_batter(start_dt=week_str, end_dt=week_end.strftime("%Y-%m-%d"), player_id=player_id)
            if df.empty:
                save_processed_entry(week_str, name)
                continue

            df['game_date'] = pd.to_datetime(df['game_date'])
            grouped = df.groupby(df['game_date'].dt.date)

            for game_date, group in grouped:
                ab = group['ab'].sum()
                if ab < 2:
                    continue

                hits = group['events'].isin(["single", "double", "triple", "home_run"]).sum()
                doubles = group['events'].eq("double").sum()
                triples = group['events'].eq("triple").sum()
                homers = group['events'].eq("home_run").sum()
                rbis = group['rbi'].sum()
                runs = group['run'].sum()
                walks = group['bb'].sum()
                sbs = group['events'].eq("stolen_base").sum()

                points = calculate_fantasy_points({
                    "H": hits,
                    "2B": doubles,
                    "3B": triples,
                    "HR": homers,
                    "RBI": rbis,
                    "R": runs,
                    "BB": walks,
                    "SB": sbs
                })

                row = {
                    "Date": game_date,
                    "Year": year,
                    "Name": name,
                    "AB": ab,
                    "H": hits,
                    "2B": doubles,
                    "3B": triples,
                    "HR": homers,
                    "RBI": rbis,
                    "R": runs,
                    "BB": walks,
                    "SB": sbs,
                    "FantasyPoints": points
                }

                pd.DataFrame([row]).to_csv(DATA_OUT_PATH, mode='a', header=not os.path.exists(DATA_OUT_PATH), index=False)

            save_processed_entry(week_str, name)
            time.sleep(0.8)
        except Exception as e:
            print(f"{name} during week of {week_str}: {e}")

    save_player_id_cache(player_id_cache)
    print("Week complete.")

# === Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Statcast data for one week.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--day", type=int, required=True)  # start day of the week
    args = parser.parse_args()

    collect_weekly_data(args.year, args.month, args.day)

