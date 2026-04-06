# matchup_machine

XGBoost model that predicts plate appearance outcomes for MLB batter–pitcher matchups using Statcast pitch-by-pitch data. Produces the model artifacts consumed by `fantasy_mlb_ai` and `diamond_mind`.

---

## What It Produces

| Artifact | Location | Size | Used by |
|---|---|---|---|
| `xgb_outcome_model.joblib` | `models/` | ~50 MB | fantasy_mlb_ai, diamond_mind |
| `outcome_feature_cols.json` | `models/` | small | fantasy_mlb_ai |
| `matchups.parquet` | `data/modeling/` | ~500 MB | fantasy_mlb_ai |
| `pitcher_profiles.parquet` | `data/pitcher_profiles/` | ~5 MB | fantasy_mlb_ai |
| `player_index.csv` | `data/` | ~1 MB | fantasy_mlb_ai |
| `batter_pa_projection_2026.parquet` | `data/` | small | fantasy_mlb_ai |

All of these are gitignored. You must generate them locally by running the pipeline below.

---

## Prerequisites

```bash
# Install the package in editable mode from the workspace root
uv pip install -e packages/matchup_machine

# Verify installation
python -c "import matchup_machine; print('OK')"
```

**Python 3.10+** is required. The pipeline downloads data from Baseball Savant via `pybaseball` — no API key needed, but it rate-limits aggressively. Expect the data collection step to take **2–4 hours** for a full season.

---

## Directory Layout

After running the full pipeline, the package directory looks like:

```
packages/matchup_machine/
├── src/matchup_machine/          # Source code
│   ├── config.py                 # Date ranges, paths
│   ├── collect_data.py           # Step 1: download Statcast
│   ├── clean_month.py            # Step 2: clean raw parquet files
│   ├── build_dataset.py          # Step 3: build matchups.parquet
│   ├── build_pitcher_tendencies.py  # Step 4a: pitcher profiles
│   ├── build_batter_pa_projection.py  # Step 4b: PA projections
│   ├── build_player_index.py    # Step 4c: name → ID lookup
│   ├── train_outcome_model.py   # Step 5: train XGBoost
│   ├── train_hit_model.py       # Step 5: (alt) binary hit model
│   ├── fantasy_inference.py     # Inference entry point
│   └── fantasy_scoring.py       # Fantasy points scoring weights
├── data/
│   ├── raw/                     # statcast_YYYY_MM.parquet (gitignored)
│   ├── processed/               # statcast_clean_YYYY_MM.parquet (gitignored)
│   ├── modeling/                # matchups.parquet (gitignored)
│   ├── pitcher_profiles/        # pitcher_profiles.parquet (gitignored)
│   ├── player_index.csv         (gitignored)
│   └── batter_pa_projection_2026.parquet  (gitignored)
└── models/
    ├── xgb_outcome_model.joblib  (gitignored)
    └── outcome_feature_cols.json (gitignored)
```

---

## Full Pipeline (First Time)

Run these steps **in order** from the workspace root.

### Step 1 — Collect Statcast Data

Downloads pitch-by-pitch data from Baseball Savant month by month. Already-downloaded months are skipped automatically.

```bash
python -m matchup_machine.collect_data
```

- **Time:** 2–4 hours (one API call per month, rate-limited)
- **Output:** `data/raw/statcast_YYYY_MM.parquet` (one file per month)
- **Date range:** controlled by `STATCAST_START` / `STATCAST_END` in `config.py` (currently 2023-04-01 to 2025-09-28)
- If a month fails, re-run the command — it resumes where it left off

### Step 2 — Clean Raw Data

Standardises column names, drops pitches missing critical fields, downcasts dtypes to save memory.

```bash
python -m matchup_machine.clean_month
```

- **Time:** 5–15 minutes
- **Output:** `data/processed/statcast_clean_YYYY_MM.parquet`
- Already-cleaned months are skipped

### Step 3 — Build Modeling Dataset

Assembles all clean months into a single `matchups.parquet` and adds outcome labels.

**Outcome classes** (8 total):

| Class | Fantasy pts/PA |
|---|---|
| `single` | +1 |
| `double` | +2 |
| `triple` | +3 |
| `home_run` | +6 |
| `walk` | +1 |
| `strikeout` | −1 |
| `ball_in_play_out` | 0 |
| `other` | +1 |

```bash
python -m matchup_machine.build_dataset
```

- **Time:** 10–20 minutes
- **Output:** `data/modeling/matchups.parquet` (~500 MB)

### Step 4 — Build Supporting Artifacts

These three steps can be run in any order, but all require `matchups.parquet` from Step 3.

**4a — Pitcher tendencies / profiles**
```bash
python -m matchup_machine.build_pitcher_tendencies
```
Output: `data/pitcher_profiles/pitcher_profiles.parquet`

**4b — Batter PA projections**
```bash
python -m matchup_machine.build_batter_pa_projection
```
Output: `data/batter_pa_projection_2026.parquet`

**4c — Player name → ID index**
```bash
python -m matchup_machine.build_player_index
```
Output: `data/player_index.csv`

> `build_player_index` calls pybaseball's `playerid_reverse_lookup` which hits a remote endpoint. It may take a few minutes.

### Step 5 — Train the Model

Trains an XGBoost multiclass classifier (8 outcome classes) with a time-based train/val/test split:

| Split | Date range |
|---|---|
| Train | 2023-04-01 → 2024-09-30 |
| Val | 2025-04-01 → 2025-07-31 |
| Test | 2025-08-01 → 2025-09-28 |

```bash
python -m matchup_machine.train_outcome_model
```

- **Time:** 10–30 minutes (depends on hardware)
- **Output:** `models/xgb_outcome_model.joblib` and `models/outcome_feature_cols.json`
- Prints accuracy, macro F1, confusion matrix, and per-class classification report on completion
- Target: **~0.80 AUC** on held-out test set

---

## Updating (Incremental)

At the start of each season or when new data is available:

### 1. Extend the date range

Edit `config.py`:
```python
STATCAST_END = date(2026, 9, 28)  # push end date forward
```

### 2. Re-run collection (only new months are fetched)

```bash
python -m matchup_machine.collect_data
```

### 3. Re-run cleaning (only new months are cleaned)

```bash
python -m matchup_machine.clean_month
```

### 4. Rebuild the dataset and artifacts

```bash
python -m matchup_machine.build_dataset
python -m matchup_machine.build_pitcher_tendencies
python -m matchup_machine.build_batter_pa_projection
python -m matchup_machine.build_player_index
```

### 5. Retrain the model

Update the train/val/test cutoffs in `config.py` to reflect the new data range, then:
```bash
python -m matchup_machine.train_outcome_model
```

---

## Verifying the Model Works

```bash
python -c "
from matchup_machine.fantasy_inference import load_artifacts, find_player_id
from matchup_machine.fantasy_scoring import expected_hitter_points_per_pa

artifacts = load_artifacts()
model, feature_cols, pitcher_profiles, batter_profiles, player_index, pa_proj, matchups = artifacts

batter_id = int(find_player_id(player_index, 'Aaron Judge'))
pas = matchups[(matchups['batter'] == batter_id) & matchups['outcome_id'].notna()]
X = pas.reindex(columns=feature_cols, fill_value=0).fillna(0).astype(float)
probs = model.predict_proba(X).mean(axis=0)

from matchup_machine.build_dataset import OUTCOME_LABELS
outcome_probs = dict(zip(OUTCOME_LABELS, probs))
ev = expected_hitter_points_per_pa(outcome_probs)
print(f'Aaron Judge: {ev:.3f} expected pts/PA ({ev*4:.2f} pts over 4 PA)')
"
```

---

## Uploading Artifacts to HuggingFace Hub

The `fantasy_mlb_ai` dashboard downloads model artifacts from HuggingFace Hub when running on Streamlit Cloud. After generating the artifacts locally:

### 1. Create a HuggingFace account and a new model repository

Go to [huggingface.co](https://huggingface.co) → New Model → set visibility (private recommended)

### 2. Install the HF CLI and log in

```bash
pip install huggingface_hub
huggingface-cli login
```

### 3. Upload the required files

```bash
# From the workspace root
HF_REPO="your-username/baseball-xgb-model"

huggingface-cli upload $HF_REPO packages/matchup_machine/models/xgb_outcome_model.joblib xgb_outcome_model.joblib
huggingface-cli upload $HF_REPO packages/matchup_machine/models/outcome_feature_cols.json outcome_feature_cols.json
huggingface-cli upload $HF_REPO packages/matchup_machine/data/modeling/matchups.parquet matchups.parquet
huggingface-cli upload $HF_REPO packages/matchup_machine/data/pitcher_profiles/pitcher_profiles.parquet pitcher_profiles.parquet
huggingface-cli upload $HF_REPO packages/matchup_machine/data/player_index.csv player_index.csv
huggingface-cli upload $HF_REPO packages/matchup_machine/data/batter_pa_projection_2026.parquet batter_pa_projection_2026.parquet
```

> `matchups.parquet` is ~500 MB — the upload will take a few minutes and requires Git LFS enabled on the repo (HuggingFace enables this automatically).

### 4. Add your credentials to Streamlit secrets

```toml
HF_TOKEN = "hf_your_token_here"
HF_MODEL_REPO = "your-username/baseball-xgb-model"
```

---

## Configuration Reference

All paths and date ranges are in `config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `STATCAST_START` | `2023-04-01` | Earliest data to collect |
| `STATCAST_END` | `2025-09-28` | Latest data to collect |
| `TRAIN_END` | `2024-09-30` | Train/val boundary |
| `VAL_START` | `2025-04-01` | Val period start |
| `VAL_END` | `2025-07-31` | Val period end |
| `TEST_START` | `2025-08-01` | Test period start |

---

## Fantasy Scoring Weights

Defined in `fantasy_scoring.py`. Edit to match your league settings:

```python
HITTER_OUTCOME_POINTS = {
    "single":          1.0,   # 1 TB
    "double":          2.0,   # 2 TB
    "triple":          3.0,   # 3 TB
    "home_run":        6.0,   # 4 TB + R + RBI
    "walk":            1.0,   # BB
    "strikeout":      -1.0,   # K
    "ball_in_play_out": 0.0,
    "other":           1.0,   # HBP, sac, etc.
}
```
