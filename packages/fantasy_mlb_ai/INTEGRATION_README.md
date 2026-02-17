# MLB Fantasy Analytics System - Phase 1 Integration

**Integrated System**: Combines `matchup_machine` predictive models with `fantasy_mlb_ai` roster management.

## Overview

This integration connects two MLB analytics projects into a unified fantasy baseball management system:

- **matchup_machine**: XGBoost outcome prediction models (0.80+ AUC) trained on 2M+ Statcast pitches
- **fantasy_mlb_ai**: Real-time roster management with ESPN API integration

## What's New (Phase 1)

### 1. ML Projection Engine (`ml_projections.py`)
- Loads trained XGBoost models from `matchup_machine`
- Generates player-specific fantasy point projections
- Provides confidence scoring based on sample size
- Handles 2M+ historical plate appearances for predictions

**Key Features:**
- Automatic player name matching
- Outcome probability distributions (HR, K, walk, single, etc.)
- Expected fantasy points per plate appearance
- Confidence levels: high (300+ PAs), medium (150-300), low (<150)

### 2. Enhanced Recommendations (`recommend_actions_ml.py`)
- Integrates ML projections into daily lineup decisions
- Combines injury management with performance predictions
- Sorts recommendations by projected fantasy points
- Provides summary statistics for starting lineup vs bench

### 3. Integration Test (`test_ml_integration.py`)
- Demonstrates system functionality with sample roster
- Tests player lookup, projection generation, and batch processing
- Validates integration between both projects

## Installation

### Prerequisites
Both projects must be in the same parent directory:
```
Projects/
├── matchup_machine/
│   ├── models/
│   │   ├── xgb_outcome_model.joblib
│   │   └── outcome_feature_cols.json
│   ├── data/
│   └── src/
└── fantasy_mlb_ai/
    ├── ml_projections.py
    ├── recommend_actions_ml.py
    └── test_ml_integration.py
```

### Dependencies
```bash
# From matchup_machine
pip install xgboost pandas numpy scikit-learn pyarrow joblib

# From fantasy_mlb_ai  
pip install pandas requests espn-api python-dotenv
```

## Usage

### Quick Test
```bash
cd fantasy_mlb_ai
python test_ml_integration.py
```

**Expected Output:**
```
MLB FANTASY ANALYTICS SYSTEM - INTEGRATION TEST
Loading ML models...
✓ Loaded models successfully (2,111,738 historical PAs)

PLAYER PROJECTIONS (Per 4 PA)
           name  ml_projection  ml_confidence
    Aaron Judge           6.86           high
   Mookie Betts           6.68           high
      Juan Soto           6.37           high
...
```

### Daily Recommendations (During Season)
```bash
cd fantasy_mlb_ai
python recommend_actions_ml.py
```

This will:
1. Fetch today's MLB games and your ESPN roster
2. Load ML models and generate projections
3. Produce recommendations sorted by projected points
4. Save results to `data/recs/recommendations-ml-YYYY-MM-DD.csv`

### Standalone Projections
```python
from ml_projections import MLProjectionEngine

engine = MLProjectionEngine()

# Single player projection
proj = engine.get_batter_projection("Aaron Judge", default_pa=4)
print(f"Expected points: {proj['expected_points']}")
print(f"Confidence: {proj['confidence']}")
print(f"Outcome probabilities: {proj['outcome_probs']}")

# Batch roster projections
roster_df = pd.read_csv("data/my_roster.csv")
roster_with_ml = engine.get_roster_projections(roster_df, games_df, team_map)
```

## How It Works

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    fantasy_mlb_ai                           │
│  ┌────────────────────────────────────────────────────┐    │
│  │ fetch_daily_data.py                                 │    │
│  │ - ESPN API → roster data                            │    │
│  │ - MLB API → today's games & probable pitchers       │    │
│  └──────────────────┬──────────────────────────────────┘    │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐    │
│  │ ml_projections.py                                    │    │
│  │                                                       │    │
│  │  ┌───────────────────────────────────────────────┐  │    │
│  │  │ MLProjectionEngine                            │  │    │
│  │  │ - Load XGBoost models                         │  │    │
│  │  │ - Match players to historical data            │  │    │
│  │  │ - Generate outcome probabilities              │  │    │
│  │  │ - Calculate fantasy point projections         │  │    │
│  │  └──────────────┬────────────────────────────────┘  │    │
│  │                 │                                     │    │
│  │            Imports from matchup_machine              │    │
│  │                 │                                     │    │
│  └─────────────────┼─────────────────────────────────────┘    │
│                    │                                          │
└────────────────────┼──────────────────────────────────────────┘
                     │
┌────────────────────▼──────────────────────────────────────────┐
│                  matchup_machine                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ models/                                               │    │
│  │ - xgb_outcome_model.joblib (trained XGBoost)         │    │
│  │ - outcome_feature_cols.json                          │    │
│  │ - outcome_labels.json                                │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ data/                                                 │    │
│  │ - matchups.parquet (2M+ historical PAs)              │    │
│  │ - pitcher_profiles.parquet                           │    │
│  │ - player_index.csv                                   │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ src/                                                  │    │
│  │ - fantasy_inference.py (model loading & inference)   │    │
│  │ - fantasy_scoring.py (points calculation)            │    │
│  │ - build_dataset.py (outcome labels)                  │    │
│  └──────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
```

### Projection Pipeline

1. **Player Lookup**
   - Input: Player name (e.g., "Aaron Judge")
   - Fuzzy match against `player_index.csv`
   - Returns: `batter_id` from Statcast database

2. **Historical Analysis**
   - Filter `matchups.parquet` for player's PAs since 2024
   - Extract feature vectors (pitcher profile, context, handedness, etc.)
   - Handle missing values and dtype conversions

3. **ML Inference**
   - Run XGBoost multiclass model on historical PAs
   - Get probability distribution over 8 outcomes:
     - single, double, triple, home_run
     - walk, strikeout, ball_in_play_out, other
   - Average probabilities across all PAs

4. **Fantasy Scoring**
   - Map outcome probabilities to fantasy points:
     - HR: 6 pts (4 TB + 1 R + 1 RBI)
     - Triple: 3 pts, Double: 2 pts, Single: 1 pt
     - Walk: 1 pt, Strikeout: -1 pt
   - Calculate expected value: `EV = Σ(prob × points)`
   - Multiply by expected PAs (default: 4)

5. **Confidence Scoring**
   - High: 300+ historical PAs
   - Medium: 150-300 PAs
   - Low: <150 PAs
   - None: Player not found

## Sample Output

```
============================================================
DAILY RECOMMENDATIONS (ML-Enhanced)
============================================================
           name position lineupSlot  ml_projection ml_confidence                              action
    Aaron Judge       OF         OF           6.86          high    Starting (high confidence)
      Juan Soto       OF         OF           6.37          high    Starting (high confidence)
Freddie Freeman       1B         1B           6.36          high    Starting (high confidence)
   Mookie Betts       OF         BE           6.68          high    Consider starting (proj: 6.68 pts)
  Shohei Ohtani    OF/DH         OF           6.24          high    Starting (high confidence)

============================================================
PROJECTION SUMMARY
============================================================
Total projected points (active players): 32.5
Average projection per player: 6.50
Starting: 4 players
Benched: 1 players
Starting lineup total: 25.8 pts
Bench total: 6.7 pts
```

## Technical Details

### Dtype Handling
The integration includes a safe wrapper (`safe_estimate_batter_probs`) to handle pandas Int32 dtype issues when interfacing with XGBoost models. All Int types are converted to float64 before prediction.

### Feature Engineering
Projections use features from `matchup_machine`:
- Batter rolling stats (launch speed, launch angle, hit rate)
- Pitcher profiles (velocity, spin rate, location tendencies)
- Handedness matchups
- Pitch context (count, baserunners, inning)

### Error Handling
- Graceful degradation if ML models unavailable
- Player name fuzzy matching with disambiguation
- Missing data imputation using median values
- Confidence scoring based on sample size

## Future Enhancements (Phase 2+)

### Phase 2: Pitcher-Aware Projections
- Use today's probable pitchers for matchup-specific projections
- Adjust predictions based on pitcher quality
- Historical head-to-head performance

### Phase 3: Production Features
- Automated daily runs (scheduled tasks)
- Prediction accuracy tracking
- Web dashboard (Streamlit/Flask)
- Waiver wire rankings powered by ML
- Trade analyzer

## Testing

Run the test suite:
```bash
# Test ML engine standalone
python ml_projections.py

# Test full integration
python test_ml_integration.py
```

## Performance

- **Model Loading**: ~2-3 seconds (one-time)
- **Single Player Projection**: ~100-200ms
- **Full Roster (15-20 players)**: ~2-3 seconds
- **Historical Data Size**: 2.1M PAs, ~500MB parquet

## Troubleshooting

### "ML models not available"
- Ensure `matchup_machine` is in parent directory
- Check that trained models exist in `matchup_machine/models/`
- Verify data files exist in `matchup_machine/data/`

### "Player not found"
- Try full name: "Aaron Judge" not "Judge"
- Check spelling
- Player may not be in 2023-2025 Statcast database

### Import errors
- Install all dependencies from both projects
- Verify Python path includes matchup_machine/src

## License

Personal use only. MLB Stats API and ESPN Fantasy API terms apply.

---

**Built by**: Austen Haymond
**Version**: 1.0.0 (Phase 1)
**Date**: January 2026
