"""
Enhanced Recommendation Engine with ML Projections

Integrates matchup_machine's XGBoost models to provide data-driven
fantasy baseball lineup recommendations.
"""

import pandas as pd
import requests
from datetime import datetime, date, timezone, timedelta
import subprocess
import sys
import os

# Import ML projection engine
from ml_projections import MLProjectionEngine

# Run fetch_daily_data.py before starting recommendations
def run_data_fetch():
    script_path = os.path.join(os.path.dirname(__file__), "fetch_daily_data.py")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("fetch_daily_data.py failed:")
        print(result.stderr)
        sys.exit(1)
    else:
        print("Data fetch complete. Output:")
        print(result.stdout)

# Run the data fetch script
run_data_fetch()

today_str = date.today().strftime('%Y-%m-%d')
tomorrow_str = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')

# === Load Data ===
roster_df = pd.read_csv("data/my_roster.csv")
games_df = pd.read_csv("data/mlb_games_today.csv")

# Convert start_time to datetime for comparison
games_df['start_time'] = pd.to_datetime(games_df['start_time'])

# === Team Name Mapping ===
team_name_map = {
    'Tor': 'Toronto Blue Jays',
    'Phi': 'Philadelphia Phillies',
    'Ari': 'Arizona Diamondbacks',
    'NYY': 'New York Yankees',
    'SD': 'San Diego Padres',
    'Mil': 'Milwaukee Brewers',
    'LAA': 'Los Angeles Angels',
    'NYM': 'New York Mets',
    'Bos': 'Boston Red Sox',
    'Tex': 'Texas Rangers',
    'Atl': 'Atlanta Braves',
    'Bal': 'Baltimore Orioles',
    'ChC': 'Chicago Cubs',
    'ChW': 'Chicago White Sox',
    'Cin': 'Cincinnati Reds',
    'Cle': 'Cleveland Guardians',
    'Col': 'Colorado Rockies',
    'Det': 'Detroit Tigers',
    'Hou': 'Houston Astros',
    'KC': 'Kansas City Royals',
    'LAD': 'Los Angeles Dodgers',
    'Mia': 'Miami Marlins',
    'Min': 'Minnesota Twins',
    'Oak': 'Oakland Athletics',
    'Pit': 'Pittsburgh Pirates',
    'Sea': 'Seattle Mariners',
    'SF': 'San Francisco Giants',
    'StL': 'St. Louis Cardinals',
    'TB': 'Tampa Bay Rays',
    'Was': 'Washington Nationals'
}

# === Get Probable Starters from MLB Stats API ===
def get_probable_pitchers():
    today = date.today().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher"
    response = requests.get(url).json()
    probables = set()
    for date_info in response.get("dates", []):
        for game in date_info.get("games", []):
            for side in ["home", "away"]:
                team_info = game["teams"][side]
                if "probablePitcher" in team_info:
                    probables.add(team_info["probablePitcher"]["fullName"])
    return probables

probable_pitchers = get_probable_pitchers()

def get_teams_playing_on(date_str):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    response = requests.get(url).json()
    
    teams = set()
    for date_data in response.get("dates", []):
        for game in date_data.get("games", []):
            teams.add(game["teams"]["home"]["team"]["name"])
            teams.add(game["teams"]["away"]["team"]["name"])
    return teams

teams_today = get_teams_playing_on(today_str)
teams_tomorrow = get_teams_playing_on(tomorrow_str)

# === Initialize ML Engine ===
print("\n" + "="*60)
print("Initializing ML Projection Engine")
print("="*60)
ml_engine = MLProjectionEngine()

# === Get ML Projections for Roster ===
print("\n" + "="*60)
print("Generating ML Projections")
print("="*60)

roster_with_ml = ml_engine.get_roster_projections(
    roster_df, 
    games_df, 
    team_name_map
)

# === Build Recommendations ===
recommendations = []

for _, player in roster_with_ml.iterrows():
    name = player["name"]
    team = player["proTeam"]
    injury = player["injuryStatus"]
    position = player["position"]
    lineup_slot = player["lineupSlot"]
    
    ml_proj = player.get("ml_projection")
    ml_confidence = player.get("ml_confidence", "none")
    ml_pts_per_pa = player.get("ml_pts_per_pa")
    
    # Check if game's already started
    full_team_name = team_name_map.get(team, None)
    plays_today = full_team_name in teams_today if full_team_name else False
    plays_tomorrow = full_team_name in teams_tomorrow if full_team_name else False
    
    game_locked = False
    
    if plays_today:
        team_games = games_df[(games_df["home_team"] == full_team_name) | (games_df["away_team"] == full_team_name)]
        if not team_games.empty:
            earliest_game = team_games['start_time'].min()
            if earliest_game < datetime.now(timezone.utc):
                game_locked = True
    
    note = ""
    action = ""
    
    # Injury-based recommendations
    if injury != "ACTIVE":
        if lineup_slot == "IL":
            note = f"{injury}" 
            action = "(already on IL)"
        elif game_locked and lineup_slot != "BE":
            note = f"{injury} (game locked)"
            action = "Too late to sub"
        elif plays_today and lineup_slot != "BE":
            note = injury
            action = "Move to IL or bench"
        elif plays_today and lineup_slot == "BE":
            note = injury
            action = "All clear (bench)"
        elif plays_tomorrow:
            note = f"{injury}, team plays tomorrow"
            action = "Move to IL or bench (early)"
        else:
            note = f"{injury}, no games today or tomorrow"
            action = "Move to IL if eligible"
    
    # ML-based recommendations for active players
    elif injury == "ACTIVE" and plays_today and not game_locked:
        if ml_proj is not None:
            # High projection players should start
            if ml_proj >= 6.0 and lineup_slot == "BE":
                note = f"Strong matchup"
                action = f"Consider starting (proj: {ml_proj} pts, {ml_confidence} conf)"
            # Low projection players might sit
            elif ml_proj < 3.0 and lineup_slot not in ["BE", "IL"]:
                note = f"Weak matchup"
                action = f"Consider benching (proj: {ml_proj} pts, {ml_confidence} conf)"
            # Good lineup spot
            elif lineup_slot not in ["BE", "IL"]:
                note = f"Projected: {ml_proj} pts"
                action = f"Starting ({ml_confidence} confidence)"
        else:
            # No ML projection available
            if plays_today and lineup_slot not in ["BE", "IL"]:
                note = "Game today"
                action = "Starting (no ML data)"
    
    # Add to recommendations if there's something to report
    if note or action or (ml_proj is not None and plays_today):
        recommendations.append({
            "name": name,
            "proTeam": team,
            "position": position,
            "lineupSlot": lineup_slot,
            "injuryStatus": injury,
            "ml_projection": ml_proj,
            "ml_confidence": ml_confidence,
            "note": note,
            "action": action
        })

# === Output ===
rec_df = pd.DataFrame(recommendations)
rec_df.to_csv(f"data/recs/recommendations-ml-{date.today()}.csv", index=False)

# === Console Summary ===
print("\n" + "="*60)
print("DAILY RECOMMENDATIONS (ML-Enhanced)")
print("="*60)

if not rec_df.empty:
    # Sort by ML projection (descending) for players with games today
    rec_df_sorted = rec_df.sort_values('ml_projection', ascending=False, na_position='last')
    
    # Format output
    display_cols = ['name', 'position', 'lineupSlot', 'ml_projection', 'ml_confidence', 'action']
    available_cols = [col for col in display_cols if col in rec_df_sorted.columns]
    
    print(rec_df_sorted[available_cols].to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*60)
    print("PROJECTION SUMMARY")
    print("="*60)
    
    active_with_proj = rec_df[
        (rec_df['injuryStatus'] == 'ACTIVE') & 
        (rec_df['ml_projection'].notna())
    ]
    
    if not active_with_proj.empty:
        total_proj = active_with_proj['ml_projection'].sum()
        avg_proj = active_with_proj['ml_projection'].mean()
        
        starting = active_with_proj[active_with_proj['lineupSlot'].isin(['C', '1B', '2B', '3B', 'SS', 'OF', 'Util', 'P'])]
        benched = active_with_proj[active_with_proj['lineupSlot'] == 'BE']
        
        print(f"Total projected points (active players): {total_proj:.1f}")
        print(f"Average projection per player: {avg_proj:.2f}")
        print(f"Starting: {len(starting)} players")
        print(f"Benched: {len(benched)} players")
        
        if not starting.empty:
            print(f"Starting lineup total: {starting['ml_projection'].sum():.1f} pts")
        if not benched.empty:
            print(f"Bench total: {benched['ml_projection'].sum():.1f} pts")
else:
    print("All clear. No substitutions or actions needed today.")

print("\n" + "="*60)
