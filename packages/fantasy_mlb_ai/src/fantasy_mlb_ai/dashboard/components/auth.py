"""
Auth Component — Stub

Provides UI elements and guards for authentication.
Currently a no-op (anonymous access).

To wire up Supabase auth:
  1. pip install supabase
  2. Add SUPABASE_URL and SUPABASE_ANON_KEY to secrets
  3. Replace the TODO sections below
"""

from __future__ import annotations

import streamlit as st

from ..utils.session_state import get_current_user, is_authenticated


def login_ui() -> None:
    """
    Render a login form in the sidebar.

    TODO (auth): Replace with Supabase Auth UI:
        from supabase import create_client
        supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Sign in"):
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state["user"] = response.user
            st.rerun()
    """
    with st.sidebar:
        st.divider()
        st.caption("Account")
        st.info("User accounts coming soon. Your roster is stored in this browser session.", icon="🔒")


def user_badge() -> None:
    """
    Show user info in the sidebar when logged in.
    Currently shows anonymous indicator.
    """
    with st.sidebar:
        user = get_current_user()
        if user:
            st.caption(f"Signed in as **{user.get('email', 'User')}**")
        else:
            st.caption("Anonymous session")


def require_auth(message: str = "Please sign in to access this feature.") -> bool:
    """
    Gate a page behind authentication.
    Returns True if authenticated, False otherwise (and renders a prompt).

    Usage in a page:
        if not require_auth():
            st.stop()

    TODO (auth): When Supabase is wired, this will redirect to the login page.
    """
    # Stub: always allow access (no auth yet)
    return True
