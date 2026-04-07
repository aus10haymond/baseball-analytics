"""
Chat Explainer

A synchronous, Streamlit-friendly LLM chatbot for explaining
fantasy baseball projections to users.

Uses the LLMClient from diamond_mind's orchestrator directly —
no Redis, no async agent lifecycle required.

Supports OpenAI and Anthropic (configured via secrets/env vars).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# LLMClient import — try diamond_mind package, fallback to raw HTTP
# ---------------------------------------------------------------------------

_DIAMOND_MIND_SRC = None
for _candidate in [
    Path(__file__).parents[5] / "diamond_mind" / "src" / "diamond_mind",  # packages/diamond_mind/src/diamond_mind
    Path(__file__).parents[6] / "diamond_mind" / "src" / "diamond_mind",
    Path(__file__).parents[7] / "diamond_mind" / "src" / "diamond_mind",
]:
    if _candidate.exists():
        _DIAMOND_MIND_SRC = _candidate
        break

if _DIAMOND_MIND_SRC and str(_DIAMOND_MIND_SRC) not in sys.path:
    sys.path.insert(0, str(_DIAMOND_MIND_SRC))

try:
    from agents.orchestrator.llm_client import LLMClient, LLMConfigError, LLMError  # type: ignore
    _LLM_CLIENT_AVAILABLE = True
except ImportError:
    _LLM_CLIENT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Secret resolution
# ---------------------------------------------------------------------------

def _get_secret(key: str) -> Optional[str]:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key)


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """\
You are a friendly and knowledgeable fantasy baseball assistant powered by \
advanced ML projections. You help users make smart start/sit decisions, \
understand player matchups, and optimize their lineups.

Keep answers concise (2-4 sentences) unless the user asks for detail. \
Use plain language — avoid jargon. When referencing stats or projections, \
cite the numbers you see in the context below.

You have access to today's projections for the user's roster. Use them \
to give specific, data-driven advice."""


def _build_system_prompt(roster: List[Dict], projections: Optional[List[Dict]]) -> str:
    lines = [_BASE_SYSTEM_PROMPT, "\n\n--- USER'S ROSTER & TODAY'S PROJECTIONS ---\n"]

    if not roster:
        lines.append("(No roster entered yet)")
    elif not projections:
        lines.append("Players on roster (no projections run yet):")
        for p in roster:
            lines.append(f"  - {p['name']} ({p['position']}, {p['team']})")
    else:
        lines.append(
            f"{'Player':<25} {'Pos':<5} {'Team':<25} {'Proj Pts':>8} {'Confidence':<12} {'Matchup':<15} {'Pitcher'}"
        )
        lines.append("-" * 100)
        for proj in sorted(projections, key=lambda x: x.get("projected_points") or 0, reverse=True):
            pts = proj.get("projected_points")
            pts_str = f"{pts:.2f}" if pts is not None else "N/A"
            lines.append(
                f"{proj.get('name', ''):<25} "
                f"{proj.get('position', ''):<5} "
                f"{proj.get('team', ''):<25} "
                f"{pts_str:>8} "
                f"{proj.get('confidence', 'N/A'):<12} "
                f"{proj.get('matchup_type', 'N/A'):<15} "
                f"{proj.get('opponent_pitcher') or 'TBD'}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Async runner helper
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine from synchronous Streamlit code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Jupyter / nested loop environments
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# ChatExplainer
# ---------------------------------------------------------------------------

class ChatExplainer:
    """
    Synchronous LLM chatbot for the dashboard Explainer page.

    Usage::

        explainer = ChatExplainer()
        if not explainer.available:
            st.warning("Configure DM_LLM_API_KEY to enable the chatbot.")
        else:
            reply = explainer.chat(user_message, history, roster, projections)
    """

    def __init__(self):
        self._client: Optional[LLMClient] = None
        self._error: Optional[str] = None
        self._init_client()

    def _init_client(self) -> None:
        if not _LLM_CLIENT_AVAILABLE:
            self._error = (
                "diamond_mind package not found. "
                "Ensure it is installed in the same Python environment."
            )
            return

        provider = _get_secret("DM_LLM_PROVIDER") or "openai"
        api_key = _get_secret("DM_LLM_API_KEY")
        _default_models = {
            "anthropic": "claude-sonnet-4-6",
            "huggingface": "google/gemma-3-12b-it",
            "ollama": "gemma3:4b",
            "openai": "gpt-4o-mini",
        }
        model = _get_secret("DM_LLM_MODEL") or _default_models.get(provider, "gpt-4o-mini")

        if not api_key:
            self._error = (
                "No LLM API key found. "
                "Add `DM_LLM_API_KEY` to your Streamlit secrets or environment."
            )
            return

        try:
            self._client = LLMClient(
                provider=provider,
                model=model,
                api_key=api_key,
                temperature=0.5,
                max_tokens=600,
            )
        except LLMConfigError as exc:
            self._error = str(exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def error_message(self) -> Optional[str]:
        return self._error

    def chat(
        self,
        user_message: str,
        history: List[Dict[str, str]],
        roster: List[Dict],
        projections: Optional[List[Dict]],
    ) -> str:
        """
        Send a message and return the assistant's reply.

        Args:
            user_message: The user's latest question.
            history: Prior conversation turns [{role, content}, ...].
            roster: Current session roster.
            projections: Current session projections (may be None).

        Returns:
            Assistant reply string, or an error message.
        """
        if not self.available:
            return f"Chatbot unavailable: {self._error}"

        system_prompt = _build_system_prompt(roster, projections)

        # Build the prompt including conversation history
        history_text = ""
        for msg in history[-8:]:  # last 8 turns to stay within token budget
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role_label}: {msg['content']}\n"

        full_prompt = f"{history_text}User: {user_message}\nAssistant:"

        try:
            reply = _run_async(
                self._client.call(full_prompt, system_prompt=system_prompt)
            )
            return reply.strip()
        except Exception as exc:
            return f"Sorry, I encountered an error: {exc}"


# ---------------------------------------------------------------------------
# Module-level singleton (cached per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_explainer() -> ChatExplainer:
    """Return a cached ChatExplainer instance."""
    return ChatExplainer()
