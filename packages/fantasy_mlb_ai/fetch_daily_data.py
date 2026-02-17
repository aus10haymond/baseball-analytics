import pandas as pd
from datetime import date
from bs4 import BeautifulSoup
from espn_api.baseball import League
from dotenv import load_dotenv
import requests
import os

# Load credentials from .env file
load_dotenv()
os.environ["GH_TOKEN"] = os.getenv("GH_TOKEN")

# ESPN Fantasy Credentials (put these in your .env)
LEAGUE_ID = os.getenv("ESPN_LEAGUE_ID")
SEASON_YEAR = int(os.getenv("ESPN_SEASON_YEAR"))
SWID = os.getenv("ESPN_SWID")
ESPN_S2 = os.getenv("ESPN_S2")

# === MLB GAMES ===
def fetch_today_games():
    today = date.today().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"

    print(f"Fetching MLB games for {today} from MLB Stats API...")

    response = requests.get(url)
    data = response.json()

    games = []
    for date_data in data.get("dates", []):
        for game in date_data.get("games", []):
            game_info = {
                "date": game["gameDate"],
                "status": game["status"]["detailedState"],
                "home_team": game["teams"]["home"]["team"]["name"],
                "away_team": game["teams"]["away"]["team"]["name"],
                "start_time": game.get("gameDate", "N/A")
            }
            games.append(game_info)

    if games:
        df = pd.DataFrame(games)
        df.to_csv("data/mlb_games_today.csv", index=False)
        print(f"Saved {len(df)} scheduled games for today.")
    else:
        print("No games found for today.")

# === FANTASY ROSTER ===
def fetch_fantasy_roster():
    league = League(league_id=LEAGUE_ID, year=SEASON_YEAR, swid=SWID, espn_s2=ESPN_S2)
    my_team = league.teams[4]
    players = my_team.roster
    roster_data = [{
        "name": p.name,
        "position": p.position,
        "injuryStatus": p.injuryStatus,
        "proTeam": p.proTeam,
        "eligibleSlots": p.eligibleSlots,
        "lineupSlot": p.lineupSlot,
    } for p in players]

    df = pd.DataFrame(roster_data)
    df.to_csv("data/my_roster.csv", index=False)
    print(f"Saved current roster with {len(df)} players")

# === MAIN ===
if __name__ == "__main__":
    fetch_today_games()
    fetch_fantasy_roster()
