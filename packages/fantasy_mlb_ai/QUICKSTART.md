# Fantasy MLB AI - Quick Start Guide

Get started with the new features in under 5 minutes!

---

## 🚀 Setup (One-Time)

### 1. Install Dependencies

The package should already be installed in editable mode. If not:

```bash
cd packages/fantasy_mlb_ai
uv pip install -e .
```

### 2. Configure Automation (Optional)

Set up daily automation to run at 9 AM:

```powershell
cd packages/fantasy_mlb_ai/scripts
.\schedule_daily.ps1 -PythonPath "python" -Time "09:00"
```

To test immediately:
```powershell
Start-ScheduledTask -TaskName "FantasyBaseballDaily"
```

### 3. Configure Email Notifications (Optional)

Create or edit `.env` file in project root:

```bash
# Email notifications
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_TO=recipient@example.com

# ESPN API (for waiver wire)
ESPN_LEAGUE_ID=your-league-id
ESPN_SWID=your-swid-cookie
ESPN_S2=your-s2-cookie
```

---

## 📊 Daily Usage

### Option 1: Automated (Recommended)

Once configured, the system runs automatically every day at 9 AM:
1. Fetches your roster and today's games
2. Generates pitcher-aware projections
3. Tracks predictions for accuracy analysis
4. Collects yesterday's actual results
5. Generates accuracy report
6. Sends email notification (if configured)

Check your inbox or look in `data/recs/recommendations-YYYY-MM-DD.csv`

### Option 2: Manual Run

```bash
cd packages/fantasy_mlb_ai/scripts
python daily_workflow.py
```

Output saved to:
- **Recommendations**: `data/recs/recommendations-YYYY-MM-DD.csv`
- **Logs**: `logs/daily_workflow_YYYY-MM-DD.log`
- **Reports**: `data/reports/accuracy_report_YYYY-MM-DD.txt`

---

## 🎯 Using the New Features

### 1. Pitcher-Aware Projections

```python
from fantasy_mlb_ai import PitcherAwareEngine

engine = PitcherAwareEngine()

# Single matchup projection
proj = engine.get_matchup_projection(
    batter_name="Aaron Judge",
    pitcher_name="Gerrit Cole",
    default_pa=4
)

print(f"Expected points: {proj['expected_points']}")
print(f"Confidence: {proj['confidence']}")
print(f"Matchup type: {proj['matchup_type']}")
```

### 2. Prediction Tracking

```python
from fantasy_mlb_ai import PredictionTracker

tracker = PredictionTracker()

# Log a prediction
tracker.log_prediction(
    date_str="2026-04-02",
    player_name="Shohei Ohtani",
    projected=8.5,
    confidence="high",
    matchup_type="pitcher_profile",
    pitcher="Blake Snell"
)

# Later, update with actual result
tracker.update_actual(
    date_str="2026-04-02",
    player_name="Shohei Ohtani",
    actual_points=10.0,
    pa_actual=5
)

# Check accuracy
accuracy = tracker.calculate_accuracy(days=30)
print(f"30-day MAE: {accuracy['mae']:.2f}")
print(f"Sample size: {accuracy['sample_size']}")

tracker.close()
```

### 3. Accuracy Reports

```python
from fantasy_mlb_ai import AccuracyReporter

with AccuracyReporter() as reporter:
    # Daily report
    print(reporter.generate_daily_report("2026-04-01"))
    
    # Weekly trends
    print(reporter.generate_weekly_report(weeks=2))
    
    # Player-specific
    print(reporter.generate_player_report("Aaron Judge"))
    
    # Overall trend
    print(reporter.generate_trend_report(days=30))
```

### 4. Waiver Wire Analysis

```python
from fantasy_mlb_ai import WaiverWireAnalyzer
import pandas as pd

analyzer = WaiverWireAnalyzer(
    espn_league_id="YOUR_LEAGUE_ID",
    espn_swid="YOUR_SWID",
    espn_s2="YOUR_S2"
)

# Get available players
available = analyzer.fetch_available_players(
    position="OF",
    max_ownership=40.0,
    limit=50
)

# Project ROS value
with_projections = analyzer.project_ros_value(
    available,
    weeks_remaining=12
)

# Top adds
top_adds = analyzer.suggest_adds(with_projections, top_n=15)
print(top_adds)

# Add/drop pairs
roster = pd.read_csv("data/my_roster.csv")
pairs = analyzer.generate_add_drop_pairs(
    roster,
    with_projections,
    min_improvement=10.0
)

for pair in pairs:
    print(f"Drop {pair['drop_name']} ({pair['drop_projection']:.1f} pts)")
    print(f"  Add {pair['add_name']} ({pair['add_projection']:.1f} pts)")
    print(f"  Improvement: +{pair['improvement']:.1f} pts\n")
```

---

## 📁 Output Files

### Daily Recommendations
**Location**: `data/recs/recommendations-YYYY-MM-DD.csv`

Columns:
- `name`, `position`, `proTeam`
- `ml_projection` - Expected fantasy points
- `ml_confidence` - Confidence level (very_high, high, medium, low)
- `matchup_type` - Type of matchup (head_to_head, pitcher_profile, general)
- `opponent_pitcher` - Today's opposing pitcher
- `action` - Recommended action

### Accuracy Reports
**Location**: `data/reports/accuracy_report_YYYY-MM-DD.txt`

Contains:
- Overall MAE, RMSE, correlation
- Breakdown by confidence level
- Breakdown by matchup type
- Best and worst predictions

### Prediction Database
**Location**: `data/predictions/predictions.db`

SQLite database tracking all predictions and actuals.

Query example:
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/predictions/predictions.db')
df = pd.read_sql_query("SELECT * FROM predictions WHERE date >= '2026-04-01'", conn)
print(df)
```

---

## 🔧 Common Tasks

### Collect Yesterday's Actuals

```bash
python scripts/collect_actuals.py
```

Or for a specific date:
```bash
python scripts/collect_actuals.py 2026-04-01
```

### Run Tests

```bash
cd packages/fantasy_mlb_ai
pytest tests/ -v
```

### View Logs

```bash
# Today's workflow log
cat logs/daily_workflow_2026-04-02.log

# Or in PowerShell
Get-Content logs/daily_workflow_2026-04-02.log
```

### Disable Automation

```powershell
.\scripts\schedule_daily.ps1 -Remove
```

---

## 📊 Understanding Output

### Confidence Levels

| Level | Meaning | Typical Scenario |
|-------|---------|------------------|
| **very_high** | 95%+ reliable | H2H matchup with 20+ PAs |
| **high** | 85-95% reliable | H2H with 10+ PAs or strong pitcher profile |
| **medium** | 70-85% reliable | Standard pitcher profile (150+ PAs) |
| **low** | 50-70% reliable | Limited data (100-150 PAs) |
| **very_low** | <50% reliable | Very limited data (<100 PAs) |

### Matchup Types

| Type | Description | When Used |
|------|-------------|-----------|
| **head_to_head** | Batter vs this specific pitcher | 10+ historical PAs between them |
| **pitcher_profile** | Pitcher's tendencies + batter's approach | Pitcher in database, no H2H |
| **general** | Batter's overall performance | Pitcher not in database |
| **no_game** | Player's team not playing today | Off day |

---

## 🆘 Troubleshooting

### "ML models not available"
- Ensure matchup_machine models are trained: `cd packages/matchup_machine && python src/train_outcome_model.py`
- Check that `models/xgb_outcome_model.joblib` exists

### "No predictions with actuals"
- Run `collect_actuals.py` to fetch yesterday's results
- Predictions need 24 hours for actuals to be available

### Email notifications not working
- Check `.env` file has correct SMTP settings
- For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833)
- Test SMTP connection: `python -m smtplib -d smtp.gmail.com:587`

### Task Scheduler not running
- Open Task Scheduler (taskschd.msc)
- Find "FantasyBaseballDaily"
- Check "Last Run Result" - should be 0x0 (success)
- Verify Python path is correct

---

## 📈 Next Steps

1. **Let it run for a week** - Build up prediction history
2. **Review accuracy reports** - See which matchup types are most accurate
3. **Check waiver wire** - Find undervalued players
4. **Tune confidence thresholds** - Adjust based on your accuracy data

---

## 🎓 Learn More

- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Full Plan**: See `../../plan.md`
- **Architecture**: See `../../CLAUDE.md`

---

**Questions?** Check the logs in `logs/` or run with verbose output:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
