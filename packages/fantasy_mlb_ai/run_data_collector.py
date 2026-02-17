import json
from datetime import datetime, timedelta
import subprocess
from pathlib import Path

TRACKER_PATH = Path("data/model_training/week_tracker.json")
LOG_PATH = Path("data/model_training/collection_log.txt")
COLLECT_SCRIPT = "training_data_collector.py"

START_YEAR, END_YEAR = 2022, 2024
FINAL_MONTH, FINAL_DAY = 9, 30

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(LOG_PATH, "a") as f:
        f.write(f"{timestamp} {msg}\n")

def load_tracker():
    with open(TRACKER_PATH, "r") as f:
        return json.load(f)

def save_tracker(year, month, day):
    with open(TRACKER_PATH, "w") as f:
        json.dump({"year": year, "month": month, "day": day}, f)

def advance_week(year, month, day):
    current = datetime(year, month, day)
    next_week = current + timedelta(days=7)
    return next_week.year, next_week.month, next_week.day

def main():
    tracker = load_tracker()
    year, month, day = tracker["year"], tracker["month"], tracker["day"]

    # Stop condition
    if (year > END_YEAR) or (year == END_YEAR and month > FINAL_MONTH):
        log("All weeks from Apr 2022 to Sep 2024 collected.")
        return

    log(f"Running collector for {year}-{month:02d}-{day:02d}")
    try:
        result = subprocess.Popen(
        ["python", COLLECT_SCRIPT, "--year", str(year), "--month", str(month), "--day", str(day)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True)

        for line in result.stdout:
            print(line, end='')  # stream live to console
            log(line.strip())    # also log each line if you want
        result.wait()

        if result.returncode == 0:
            log(f"Success for {year}-{month:02d}-{day:02d}")
        else:
            log(f"Error for {year}-{month:02d}-{day:02d}: {result.stderr.strip()}")
            return
    except Exception as e:
        log(f"Exception during run: {e}")
        return

    # Advance to next week
    next_year, next_month, next_day = advance_week(year, month, day)
    save_tracker(next_year, next_month, next_day)

if __name__ == "__main__":
    main()
