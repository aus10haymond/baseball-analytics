"""
Session State Helpers

Centralises all st.session_state access so the rest of the app
never writes keys directly.  Each helper is written to be auth-aware:
when user accounts are added (Supabase), the helpers here are the
only things that need updating — pages stay the same.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Initialisation — call once from app.py
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Set default values for all session state keys."""
    defaults: Dict[str, Any] = {
        # Roster: list of dicts {name, position, team}
        "roster": [],
        # Projections: list of dicts produced by run_projections()
        "projections": None,
        # Whether projections are stale (roster changed)
        "projections_stale": False,
        # Chat history: list of {role: "user"|"assistant", content: str}
        "chat_history": [],
        # --- Future auth stub ---
        # Will hold a Supabase user object when auth is wired up.
        "user": None,
        # --- Future ESPN stub ---
        # Will hold ESPN OAuth tokens / credentials.
        "espn_credentials": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

def get_roster() -> List[Dict[str, str]]:
    return st.session_state.get("roster", [])


def set_roster(roster: List[Dict[str, str]]) -> None:
    st.session_state["roster"] = roster
    # Projections must be re-run after any roster change
    st.session_state["projections"] = None
    st.session_state["projections_stale"] = True


def add_player(name: str, position: str, team: str) -> bool:
    """
    Add a player to the roster.
    Returns False if a player with the same name already exists.
    """
    roster = get_roster()
    if any(p["name"].lower() == name.lower() for p in roster):
        return False
    roster.append({"name": name.strip(), "position": position, "team": team})
    set_roster(roster)
    return True


def remove_player(name: str) -> None:
    roster = [p for p in get_roster() if p["name"] != name]
    set_roster(roster)


def clear_roster() -> None:
    set_roster([])


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

def get_projections() -> Optional[List[Dict]]:
    return st.session_state.get("projections")


def set_projections(projections: List[Dict]) -> None:
    st.session_state["projections"] = projections
    st.session_state["projections_stale"] = False


def projections_are_stale() -> bool:
    return st.session_state.get("projections_stale", False)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def get_chat_history() -> List[Dict[str, str]]:
    return st.session_state.get("chat_history", [])


def add_chat_message(role: str, content: str) -> None:
    """Append a message to the chat history. role is 'user' or 'assistant'."""
    history = get_chat_history()
    history.append({"role": role, "content": content})
    st.session_state["chat_history"] = history


def clear_chat_history() -> None:
    st.session_state["chat_history"] = []


# ---------------------------------------------------------------------------
# Auth stubs — replace internals when Supabase is wired up
# ---------------------------------------------------------------------------

def get_current_user() -> Optional[Dict]:
    """
    Return the currently logged-in user, or None for anonymous.

    TODO (auth): Replace with Supabase session lookup:
        from supabase import create_client
        supabase = create_client(url, key)
        session = supabase.auth.get_session()
        return session.user if session else None
    """
    return st.session_state.get("user")


def is_authenticated() -> bool:
    return get_current_user() is not None


# ---------------------------------------------------------------------------
# ESPN stubs — replace internals when OAuth is wired up
# ---------------------------------------------------------------------------

def get_espn_credentials() -> Optional[Dict]:
    """
    Return stored ESPN credentials, or None.

    TODO (espn): Replace with OAuth token storage / Supabase lookup.
    """
    return st.session_state.get("espn_credentials")


def has_espn_connected() -> bool:
    return get_espn_credentials() is not None
