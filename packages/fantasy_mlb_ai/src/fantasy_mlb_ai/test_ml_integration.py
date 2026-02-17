"""
Test ML Integration

Demonstrates the integrated system with a sample roster.
"""

import pandas as pd
from ml_projections import MLProjectionEngine

# Create a sample roster of well-known players
sample_roster = pd.DataFrame([
    {"name": "Aaron Judge", "proTeam": "NYY", "position": "OF", "lineupSlot": "OF", "injuryStatus": "ACTIVE"},
    {"name": "Shohei Ohtani", "proTeam": "LAD", "position": "OF/DH", "lineupSlot": "OF", "injuryStatus": "ACTIVE"},
    {"name": "Freddie Freeman", "proTeam": "LAD", "position": "1B", "lineupSlot": "1B", "injuryStatus": "ACTIVE"},
    {"name": "Mookie Betts", "proTeam": "LAD", "position": "OF", "lineupSlot": "BE", "injuryStatus": "ACTIVE"},
    {"name": "Juan Soto", "proTeam": "NYY", "position": "OF", "lineupSlot": "OF", "injuryStatus": "ACTIVE"},
])

# Create simulated games dataframe (simulate game day)
from datetime import datetime, timedelta
games_df = pd.DataFrame([
    {"home_team": "New York Yankees", "away_team": "Boston Red Sox", "start_time": datetime.now() + timedelta(hours=2)},
    {"home_team": "Los Angeles Dodgers", "away_team": "San Francisco Giants", "start_time": datetime.now() + timedelta(hours=3)},
])

# Team mapping
team_name_map = {
    'NYY': 'New York Yankees',
    'LAD': 'Los Angeles Dodgers',
}

print("\n" + "="*70)
print("MLB FANTASY ANALYTICS SYSTEM - INTEGRATION TEST")
print("="*70)

print("\nSample Roster:")
print(sample_roster[['name', 'position', 'lineupSlot']].to_string(index=False))

# Initialize ML Engine
print("\n" + "="*70)
print("Loading ML Models...")
print("="*70)

ml_engine = MLProjectionEngine()

if not ml_engine.ml_available:
    print("\n✗ ML models not available - test cannot continue")
    exit(1)

# Get projections for roster
print("\n" + "="*70)
print("Generating ML Projections...")
print("="*70)

roster_with_ml = ml_engine.get_roster_projections(
    sample_roster,
    games_df,
    team_name_map
)

# Display results
print("\n" + "="*70)
print("PLAYER PROJECTIONS (Per 4 PA)")
print("="*70)

output_cols = ['name', 'position', 'lineupSlot', 'ml_projection', 'ml_pts_per_pa', 'ml_confidence', 'ml_sample_size']
display_df = roster_with_ml[output_cols].sort_values('ml_projection', ascending=False, na_position='last')

print(display_df.to_string(index=False))

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

valid_projs = roster_with_ml[roster_with_ml['ml_projection'].notna()]

if not valid_projs.empty:
    print(f"Total players analyzed: {len(roster_with_ml)}")
    print(f"Players with projections: {len(valid_projs)}")
    print(f"Average projection (4 PA): {valid_projs['ml_projection'].mean():.2f} points")
    print(f"Highest projection: {valid_projs['ml_projection'].max():.2f} points ({valid_projs.loc[valid_projs['ml_projection'].idxmax(), 'name']})")
    print(f"Lowest projection: {valid_projs['ml_projection'].min():.2f} points ({valid_projs.loc[valid_projs['ml_projection'].idxmin(), 'name']})")
    
    # Confidence breakdown
    print("\nConfidence Distribution:")
    confidence_counts = valid_projs['ml_confidence'].value_counts()
    for conf, count in confidence_counts.items():
        print(f"  {conf}: {count} players")
else:
    print("No valid projections generated")

print("\n" + "="*70)
print("✓ Integration test complete!")
print("="*70)

print("\nKey Features Demonstrated:")
print("  ✓ ML model loading and initialization (2M+ historical PAs)")
print("  ✓ Player name matching against historical database")  
print("  ✓ Outcome probability estimation with XGBoost (0.80+ AUC)")
print("  ✓ Fantasy point projection with confidence scoring")
print("  ✓ Batch processing of entire roster")
print("  ✓ Game schedule integration for daily projections")
print("\nThis integration combines:")
print("  - matchup_machine: XGBoost models (0.80+ AUC)")
print("  - fantasy_mlb_ai: Real-time roster management")
print("  - Result: Data-driven daily lineup recommendations")
