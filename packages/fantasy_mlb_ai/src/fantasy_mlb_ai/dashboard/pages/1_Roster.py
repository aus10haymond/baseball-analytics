"""
Roster Page

Users manually enter their fantasy baseball roster here.
Each player needs: name, position, and MLB team.

Future: ESPN account connection will auto-populate this.
"""

import sys
from pathlib import Path

# Ensure dashboard utils are importable when run as a page
_DASHBOARD_DIR = Path(__file__).parent.parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import pandas as pd
import streamlit as st

from components.espn_connect import espn_connect_ui
from utils.player_lookup import get_player_info, get_player_names
from utils.session_state import (
    add_player,
    clear_roster,
    get_roster,
    init_session_state,
    remove_player,
    set_roster,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP", "UTIL", "DH"]

MLB_TEAMS = [
    "Arizona Diamondbacks", "Atlanta Braves", "Baltimore Orioles",
    "Boston Red Sox", "Chicago Cubs", "Chicago White Sox",
    "Cincinnati Reds", "Cleveland Guardians", "Colorado Rockies",
    "Detroit Tigers", "Houston Astros", "Kansas City Royals",
    "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins",
    "Milwaukee Brewers", "Minnesota Twins", "New York Mets",
    "New York Yankees", "Oakland Athletics", "Philadelphia Phillies",
    "Pittsburgh Pirates", "San Diego Padres", "San Francisco Giants",
    "Seattle Mariners", "St. Louis Cardinals", "Tampa Bay Rays",
    "Texas Rangers", "Toronto Blue Jays", "Washington Nationals",
]

SAMPLE_ROSTER = [
    {"name": "Aaron Judge", "position": "OF", "team": "New York Yankees"},
    {"name": "Freddie Freeman", "position": "1B", "team": "Los Angeles Dodgers"},
    {"name": "Jose Ramirez", "position": "3B", "team": "Cleveland Guardians"},
    {"name": "Bobby Witt Jr.", "position": "SS", "team": "Kansas City Royals"},
    {"name": "William Contreras", "position": "C", "team": "Milwaukee Brewers"},
    {"name": "Matt McLain", "position": "2B", "team": "Cincinnati Reds"},
    {"name": "Yordan Alvarez", "position": "OF", "team": "Houston Astros"},
    {"name": "Adolis Garcia", "position": "OF", "team": "Texas Rangers"},
    {"name": "Kyle Tucker", "position": "OF", "team": "Houston Astros"},
    {"name": "Shohei Ohtani", "position": "DH", "team": "Los Angeles Dodgers"},
    {"name": "Zack Wheeler", "position": "SP", "team": "Philadelphia Phillies"},
    {"name": "Gerrit Cole", "position": "SP", "team": "New York Yankees"},
    {"name": "Emmanuel Clase", "position": "RP", "team": "Cleveland Guardians"},
]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Roster — Fantasy MLB AI", page_icon="🏟️", layout="wide")
init_session_state()

st.title("🏟️ My Roster")
st.caption("Add your fantasy baseball players. Projections are generated on the Recommendations page.")

# ---------------------------------------------------------------------------
# ESPN connect banner
# ---------------------------------------------------------------------------

espn_connect_ui()

st.divider()

# ---------------------------------------------------------------------------
# Add player form
# ---------------------------------------------------------------------------

st.subheader("Add a Player")

# Initialise autofill state
if "roster_autofill_position" not in st.session_state:
    st.session_state.roster_autofill_position = POSITIONS[0]
if "roster_autofill_team" not in st.session_state:
    st.session_state.roster_autofill_team = MLB_TEAMS[0]

_all_player_names = get_player_names()


def _on_player_name_change() -> None:
    """Autofill position and team when a known player is selected."""
    selected = st.session_state.get("roster_player_name_select")
    if selected:
        info = get_player_info(selected)
        if info:
            st.session_state.roster_autofill_position = info["position"]
            st.session_state.roster_autofill_team = info["team"]


col1, col2, col3, col4 = st.columns([3, 1.5, 3, 1.5])

with col1:
    name_input = st.selectbox(
        "Player Name",
        options=[""] + _all_player_names,
        index=0,
        key="roster_player_name_select",
        on_change=_on_player_name_change,
        placeholder="Type to search…",
    )

_pos_index = POSITIONS.index(st.session_state.roster_autofill_position) if st.session_state.roster_autofill_position in POSITIONS else 0
_team_index = MLB_TEAMS.index(st.session_state.roster_autofill_team) if st.session_state.roster_autofill_team in MLB_TEAMS else 0

with col2:
    position_input = st.selectbox("Position", POSITIONS, index=_pos_index, key="roster_position_select")

with col3:
    team_input = st.selectbox("MLB Team", MLB_TEAMS, index=_team_index, key="roster_team_select")

with col4:
    st.write("")  # vertical spacing
    st.write("")
    add_clicked = st.button("Add Player", use_container_width=True, type="primary")

if add_clicked:
    chosen_name = name_input.strip() if name_input else ""
    if not chosen_name:
        st.error("Select a player name from the dropdown.")
    else:
        success = add_player(chosen_name, position_input, team_input)
        if success:
            st.success(f"Added {chosen_name} ({position_input})")
            # Reset autofill state for next entry
            st.session_state.roster_autofill_position = POSITIONS[0]
            st.session_state.roster_autofill_team = MLB_TEAMS[0]
            st.rerun()
        else:
            st.warning(f"{chosen_name} is already on your roster.")

# ---------------------------------------------------------------------------
# Current roster table
# ---------------------------------------------------------------------------

st.subheader("Current Roster")

roster = get_roster()

if not roster:
    st.info(
        "Your roster is empty. Add players using the form above, "
        "or load the sample roster to try it out.",
        icon="👆",
    )
else:
    # Group by position category for display
    batters = [p for p in roster if p["position"] not in ("SP", "RP")]
    pitchers = [p for p in roster if p["position"] in ("SP", "RP")]

    def _render_player_table(players: list, label: str):
        if not players:
            return
        st.markdown(f"**{label}**")
        for i, player in enumerate(players):
            col1, col2, col3, col4 = st.columns([3, 1.5, 3, 1])
            with col1:
                st.write(player["name"])
            with col2:
                st.write(player["position"])
            with col3:
                st.write(player["team"])
            with col4:
                if st.button("Remove", key=f"remove_{player['name']}_{i}", use_container_width=True):
                    remove_player(player["name"])
                    st.rerun()

    _render_player_table(batters, f"Batters ({len(batters)})")
    if batters and pitchers:
        st.divider()
    _render_player_table(pitchers, f"Pitchers ({len(pitchers)})")

    st.divider()
    st.caption(f"**{len(roster)} total players** on your roster.")

# ---------------------------------------------------------------------------
# Bulk actions
# ---------------------------------------------------------------------------

col_sample, col_clear, col_spacer = st.columns([2, 2, 4])

with col_sample:
    if st.button("Load Sample Roster", use_container_width=True):
        set_roster(SAMPLE_ROSTER)
        st.success("Sample roster loaded!")
        st.rerun()

with col_clear:
    if roster:
        if st.button("Clear Roster", use_container_width=True, type="secondary"):
            clear_roster()
            st.rerun()

# ---------------------------------------------------------------------------
# Paste / bulk import
# ---------------------------------------------------------------------------

with st.expander("Bulk Import (paste a list)"):
    st.markdown(
        "Paste one player per line in the format: `Name, Position, Team`\n\n"
        "Example:\n```\nAaron Judge, OF, New York Yankees\nGerrit Cole, SP, New York Yankees\n```"
    )
    bulk_text = st.text_area("Player list", height=150, placeholder="Aaron Judge, OF, New York Yankees")

    if st.button("Import Players"):
        added = 0
        errors = []
        for line in bulk_text.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                errors.append(f"Skipped (bad format): {line}")
                continue
            pname, ppos, pteam = parts
            if ppos not in POSITIONS:
                errors.append(f"Unknown position '{ppos}' for {pname} — use: {', '.join(POSITIONS)}")
                continue
            if add_player(pname, ppos, pteam):
                added += 1
            else:
                errors.append(f"Duplicate: {pname}")

        if added:
            st.success(f"Added {added} player(s).")
            st.rerun()
        for err in errors:
            st.warning(err)

# ---------------------------------------------------------------------------
# Navigation prompt
# ---------------------------------------------------------------------------

if roster:
    st.divider()
    st.success(
        f"Roster ready! Head to **Recommendations** to get today's projections "
        f"for your {len(roster)} players.",
        icon="📊",
    )
