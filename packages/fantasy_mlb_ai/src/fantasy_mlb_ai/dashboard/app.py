"""
Fantasy MLB AI Dashboard — Entry Point

Run locally:
    streamlit run packages/fantasy_mlb_ai/src/fantasy_mlb_ai/dashboard/app.py

Streamlit Cloud:
    Set the main file path to the line above in the deploy settings.
"""

import streamlit as st

from utils.session_state import init_session_state
from components.auth import login_ui, user_badge
from utils.model_loader import ml_available

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fantasy MLB AI",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

init_session_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚾ Fantasy MLB AI")
    st.caption("ML-powered lineup decisions")
    st.divider()

    # Navigation hint
    st.markdown(
        """
        **Pages**
        - 🏟️ **Roster** — Enter your players
        - 📊 **Recommendations** — Today's projections
        - 💬 **Explainer** — Ask the AI assistant
        """
    )

    # ML status indicator
    st.divider()
    st.caption("System Status")
    if ml_available():
        st.success("ML models loaded", icon="✅")
    else:
        st.warning("ML models not loaded", icon="⚠️")
        with st.expander("How to enable ML projections"):
            st.markdown(
                """
                1. Upload model artifacts to HuggingFace Hub
                2. Add `HF_TOKEN` and `HF_MODEL_REPO` to Streamlit secrets
                3. Restart the app

                Or run locally with the trained matchup_machine models.
                """
            )

    user_badge()
    login_ui()

# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------

st.title("⚾ Fantasy MLB AI")
st.subheader("ML-powered fantasy baseball projections and lineup advice")

st.markdown(
    """
    Welcome! This dashboard uses a trained XGBoost model on Statcast pitch-by-pitch
    data to project daily fantasy points for your roster — adjusted for today's
    probable pitcher matchups.

    **Get started in 3 steps:**
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "**Step 1 — Enter Your Roster**\n\n"
        "Go to the **Roster** page and add your players. "
        "Include name, position, and MLB team.",
        icon="🏟️",
    )

with col2:
    st.info(
        "**Step 2 — Get Recommendations**\n\n"
        "The **Recommendations** page projects today's fantasy points "
        "for each player, adjusted for their pitching matchup.",
        icon="📊",
    )

with col3:
    st.info(
        "**Step 3 — Ask Questions**\n\n"
        "The **Explainer** chatbot can answer questions like "
        '"Should I start Judge today?" or "Who has the best matchup?"',
        icon="💬",
    )

st.divider()

# Quick stats if roster exists
from utils.session_state import get_roster, get_projections

roster = get_roster()
projections = get_projections()

if roster:
    st.markdown(f"**Your roster:** {len(roster)} players loaded")
    if projections:
        import pandas as pd
        df = pd.DataFrame(projections)
        proj_col = "projected_points"
        if proj_col in df.columns:
            valid = df[df[proj_col].notna()]
            if not valid.empty:
                top = valid.nlargest(3, proj_col)
                st.markdown("**Top projected players today:**")
                for _, row in top.iterrows():
                    st.markdown(
                        f"- **{row['name']}** ({row['position']}) — "
                        f"{row[proj_col]:.2f} pts projected"
                    )
    else:
        st.markdown("Go to **Recommendations** to run today's projections.")
else:
    st.markdown("Go to **Roster** to add your players and get started.")

st.divider()
st.caption(
    "Projections are based on historical Statcast data (2020–2024) and today's "
    "MLB schedule. Not financial advice. Play responsibly."
)
