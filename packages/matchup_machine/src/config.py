from pathlib import Path
from datetime import date

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELING_DIR = DATA_DIR / "modeling"
PITCHER_PROFILES_DIR = DATA_DIR / "pitcher_profiles"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Date ranges
STATCAST_START = date(2023, 4, 1)
STATCAST_END   = date(2025, 9, 28)

# Train/val/test cutoffs
TRAIN_END = date(2024, 9, 30)
VAL_START = date(2025, 4, 1)
VAL_END   = date(2025, 7, 31)
TEST_START = date(2025, 8, 1)

def ensure_directories() -> None:
    """Create all project directories if they don't exist."""
    for d in [
        DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELING_DIR,
        PITCHER_PROFILES_DIR, MODELS_DIR, RESULTS_DIR
    ]:
        d.mkdir(parents=True, exist_ok=True)
