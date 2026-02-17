import pandas as pd
import requests
from datetime import datetime, date, timezone, timedelta
import subprocess
import sys
import os

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
        print(result.stdout)  # <-- Show all print() output from fetch_daily_data.py

# Run the data fetch script
run_data_fetch()

today_str = date.today().strftime('%Y-%m-%d')
tomorrow_str = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')

# === Load Data ===
roster_df = pd.read_csv("data/my_roster.csv")
games_df = pd.read_csv("data/mlb_games_today.csv")

# Convert start_time to datetime for comparison
games_df['start_time'] = pd.to_datetime(games_df['start_time'])

# === Build Set of Teams Playing Today ===
# Team abbreviation to full name map (ESPN-style abbreviations to MLB full names)
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
    'Tex': 'Texas Rangers'
    # Add others as needed
}
teams_playing = set(games_df['home_team']).union(set(games_df['away_team']))

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

# === Build Recommendation Logic ===
recommendations = []

for _, player in roster_df.iterrows():
    name = player["name"]
    team = player["proTeam"]
    injury = player["injuryStatus"]
    eligible = eval(player["eligibleSlots"])
    position = player["position"]
    lineup_slot = player["lineupSlot"]

    # Check if game's already started
    full_team_name = team_name_map.get(team, None)
    plays_today = full_team_name in teams_today if full_team_name else False
    plays_tomorrow = full_team_name in teams_tomorrow if full_team_name else False

    game_locked = False

    if plays_today:
        team_games = games_df[(games_df["home_team"] == full_team_name) | (games_df["away_team"] == full_team_name)]
        earliest_game = team_games['start_time'].min()
        if earliest_game < datetime.now(timezone.utc):
            game_locked = True


    note = ""
    action = ""

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

    if note or action:
        recommendations.append({
            "name": name,
            "proTeam": team,
            "injuryStatus": injury,
            "position": position,
            "lineupSlot": lineup_slot,
            "note": note,
            "action": action
        })

# === Output ===
rec_df = pd.DataFrame(recommendations)
rec_df.to_csv(f"data/recs/recommendations-{date.today()}.csv", index=False)

# === Console Summary ===
if not rec_df.empty:
    print("RECOMMENDATIONS:")
    print(rec_df.to_string(index=False))
else:
    print("All clear. No substitutions or injury actions needed today.")
