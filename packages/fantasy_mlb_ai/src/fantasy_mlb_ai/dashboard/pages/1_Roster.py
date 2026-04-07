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

import random

import pandas as pd
import streamlit as st

from components.espn_connect import espn_connect_ui
from utils.player_lookup import PLAYER_DATA, get_player_info, get_player_names
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

def _generate_sample_roster() -> list[dict]:
    """
    Build a randomised 19-player fantasy roster following the recommended layout:
      1 C · 1 1B · 1 2B · 1 3B · 1 SS · 4 OF · 2 UTIL · 5 SP · 3 RP
    UTIL slots are filled from any unused hitter (DH or leftover position player).
    """
    # Group lookup names by position
    by_pos: dict[str, list[str]] = {}
    for name, info in PLAYER_DATA.items():
        by_pos.setdefault(info["position"], []).append(name)

    used: set[str] = set()
    roster: list[dict] = []

    def _pick(pos: str, slot: str) -> dict | None:
        """Pick a random unused player from *pos* and label them as *slot*."""
        candidates = [n for n in by_pos.get(pos, []) if n not in used]
        if not candidates:
            return None
        name = random.choice(candidates)
        used.add(name)
        return {"name": name, "position": slot, "team": PLAYER_DATA[name]["team"]}

    # Core lineup slots
    for pos in ("C", "1B", "2B", "3B", "SS"):
        player = _pick(pos, pos)
        if player:
            roster.append(player)

    # 4 outfielders
    for _ in range(4):
        player = _pick("OF", "OF")
        if player:
            roster.append(player)

    # 2 UTIL — draw from DH first, then any remaining position player
    util_positions = ["DH", "C", "1B", "2B", "3B", "SS", "OF"]
    util_filled = 0
    for pos in util_positions:
        if util_filled >= 2:
            break
        player = _pick(pos, "UTIL")
        if player:
            roster.append(player)
            util_filled += 1

    # 5 starting pitchers
    for _ in range(5):
        player = _pick("SP", "SP")
        if player:
            roster.append(player)

    # 3 relief pitchers
    for _ in range(3):
        player = _pick("RP", "RP")
        if player:
            roster.append(player)

    return roster

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

_all_player_names = get_player_names()

# Initialise widget keys so the selectboxes have a defined starting value
if "roster_position_select" not in st.session_state:
    st.session_state.roster_position_select = POSITIONS[0]
if "roster_team_select" not in st.session_state:
    st.session_state.roster_team_select = MLB_TEAMS[0]


def _on_player_name_change() -> None:
    """Autofill position and team by writing directly into the widget keys."""
    selected = st.session_state.get("roster_player_name_select")
    if selected:
        info = get_player_info(selected)
        if info:
            st.session_state.roster_position_select = info["position"]
            st.session_state.roster_team_select = info["team"]


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

with col2:
    position_input = st.selectbox("Position", POSITIONS, key="roster_position_select")

with col3:
    team_input = st.selectbox("MLB Team", MLB_TEAMS, key="roster_team_select")

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
            # Reset widgets for next entry
            st.session_state.roster_player_name_select = ""
            st.session_state.roster_position_select = POSITIONS[0]
            st.session_state.roster_team_select = MLB_TEAMS[0]
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
        set_roster(_generate_sample_roster())
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
