"""
ESPN Connect Component — Stub

Handles ESPN Fantasy account connection and roster import.
Currently shows a "coming soon" placeholder.

To wire up ESPN OAuth / cookie auth:
  1. Obtain user's ESPN SWID + espn_s2 cookies via OAuth or manual entry
  2. Store in st.session_state["espn_credentials"]
  3. Use espn-api to fetch roster: League(league_id, year, espn_s2, swid)

ESPN API docs: https://github.com/cwendt94/espn-api
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from utils.session_state import (
    get_espn_credentials,
    has_espn_connected,
    set_roster,
)


MLB_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP", "UTIL", "DH"]


def espn_connect_ui() -> None:
    """
    Render the ESPN connection panel.
    Shows connected status if credentials are present, otherwise a connect prompt.
    """
    if has_espn_connected():
        _connected_ui()
    else:
        _connect_prompt_ui()


def _connect_prompt_ui() -> None:
    with st.expander("Connect ESPN Fantasy Account", expanded=False):
        st.info(
            "Direct ESPN account connection is coming soon. "
            "For now, enter your roster manually below.",
            icon="📡",
        )
        # Placeholder form — will become real OAuth / cookie flow
        st.markdown("**Future flow:**")
        st.markdown(
            "1. Sign in with ESPN\n"
            "2. Select your league\n"
            "3. Roster auto-imports here"
        )
        st.button("Connect ESPN (coming soon)", disabled=True, key="espn_connect_btn")


def _connected_ui() -> None:
    creds = get_espn_credentials()
    with st.expander("ESPN Account Connected", expanded=False):
        st.success(f"Connected to league {creds.get('league_id', '—')}", icon="✅")
        if st.button("Sync Roster from ESPN", key="espn_sync_btn"):
            _import_espn_roster(creds)
        if st.button("Disconnect ESPN", key="espn_disconnect_btn"):
            st.session_state["espn_credentials"] = None
            st.rerun()


def _import_espn_roster(creds: dict) -> None:
    """
    Pull the user's current roster from ESPN and load it into session state.

    TODO (espn): Replace stub with real implementation:
        from espn_api.baseball import League
        league = League(
            league_id=creds["league_id"],
            year=datetime.now().year,
            espn_s2=creds["espn_s2"],
            swid=creds["swid"],
        )
        my_team = league.teams[creds["team_index"]]
        roster = [
            {"name": p.name, "position": p.eligibleSlots[0], "team": p.proTeam}
            for p in my_team.roster
        ]
        set_roster(roster)
    """
    st.warning("ESPN roster import not yet implemented.", icon="🚧")


def import_espn_roster(creds: Optional[dict] = None) -> bool:
    """
    Programmatic import (used from other pages).
    Returns True on success.
    """
    if creds is None:
        creds = get_espn_credentials()
    if creds is None:
        return False
    _import_espn_roster(creds)
    return False  # stub always fails until implemented
