"""
Explainer Page — AI Chatbot

An LLM-powered assistant that answers questions about the user's
projections, helps with start/sit decisions, and explains what
drives the model's recommendations.

Backed by the LLMClient (OpenAI or Anthropic) with the user's
current roster + projections as system context.
"""

import sys
from pathlib import Path

_DASHBOARD_DIR = Path(__file__).parent.parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import streamlit as st

from utils.chat_explainer import get_explainer
from utils.session_state import (
    add_chat_message,
    clear_chat_history,
    get_chat_history,
    get_projections,
    get_roster,
    init_session_state,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Explainer — Fantasy MLB AI",
    page_icon="💬",
    layout="wide",
)
init_session_state()

# ---------------------------------------------------------------------------
# Sidebar context panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Roster Context")
    roster = get_roster()
    if roster:
        for p in roster:
            st.caption(f"• {p['name']} ({p['position']})")
    else:
        st.caption("No roster loaded")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        clear_chat_history()
        st.rerun()

    st.divider()
    st.caption(
        "The assistant has access to your current roster and today's "
        "projections. Ask anything about start/sit decisions, matchups, "
        "or player analysis."
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("💬 Explainer Assistant")
st.caption("Ask questions about your roster, projections, or player matchups.")

# ---------------------------------------------------------------------------
# Explainer availability check
# ---------------------------------------------------------------------------

explainer = get_explainer()

if not explainer.available:
    st.error(
        f"The AI assistant is not available: **{explainer.error_message}**\n\n"
        "Add `DM_LLM_API_KEY` and `DM_LLM_PROVIDER` (`huggingface`, `openai`, or `anthropic`) "
        "to your Streamlit secrets to enable the chatbot.",
        icon="🔑",
    )
    st.divider()
    st.markdown("**To configure:**")
    st.code(
        """
# In Streamlit Cloud → Settings → Secrets:
DM_LLM_PROVIDER = "huggingface"           # or "openai" / "anthropic"
DM_LLM_API_KEY  = "hf_..."               # your HuggingFace token
DM_LLM_MODEL    = "google/gemma-3-12b-it" # optional, defaults set per provider
        """,
        language="toml",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Projection context notice
# ---------------------------------------------------------------------------

projections = get_projections()
roster = get_roster()

if not roster:
    st.info(
        "Your roster is empty. Head to the **Roster** page to add players "
        "so the assistant has context for your questions.",
        icon="👈",
    )
elif not projections:
    st.info(
        "No projections loaded yet. Go to **Recommendations** and run projections "
        "so the assistant can reference today's numbers.",
        icon="📊",
    )
else:
    st.success(
        f"Assistant has context for {len(roster)} players and today's projections.",
        icon="✅",
    )

# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------

with st.expander("Suggested questions", expanded=not bool(get_chat_history())):
    suggestions = [
        "Who should I start today?",
        "Which player has the best pitcher matchup?",
        "Should I sit anyone with a tough matchup today?",
        "Who has the highest confidence projection?",
        "Explain the difference between head-to-head and general projections.",
        "Which batters are facing left-handed pitchers today?",
        "Who is projected to be the most valuable player today?",
        "What does 'very low confidence' mean for a projection?",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                add_chat_message("user", suggestion)
                with st.spinner("Thinking..."):
                    reply = explainer.chat(
                        user_message=suggestion,
                        history=get_chat_history()[:-1],  # exclude the message just added
                        roster=roster,
                        projections=projections,
                    )
                add_chat_message("assistant", reply)
                st.rerun()

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

st.divider()

history = get_chat_history()

if not history:
    st.markdown(
        "<div style='text-align:center;color:gray;padding:2rem 0'>"
        "No messages yet. Ask a question below or pick a suggestion above."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if user_input := st.chat_input("Ask about your roster, matchups, or projections..."):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    add_chat_message("user", user_input)

    # Generate and display assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = explainer.chat(
                user_message=user_input,
                history=history,  # history before this message
                roster=roster,
                projections=projections,
            )
        st.markdown(reply)
    add_chat_message("assistant", reply)
    st.rerun()
