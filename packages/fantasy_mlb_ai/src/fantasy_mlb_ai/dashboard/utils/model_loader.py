"""
Model Loader

Loads matchup_machine ML artifacts for the dashboard.
Priority order:
  1. Local file paths (dev / local run)
  2. Hugging Face Hub download (Streamlit Cloud / production)

All artifacts are cached with st.cache_resource so they are shared
across user sessions and only loaded once per server process.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import streamlit as st


# ---------------------------------------------------------------------------
# Local path resolution
# ---------------------------------------------------------------------------

def _find_local_matchup_machine() -> Optional[Path]:
    """
    Walk up from this file to find the matchup_machine source tree.
    Returns the src/ path if found, else None.
    """
    candidates = [
        # Monorepo layout: packages/matchup_machine/src
        Path(__file__).parents[6] / "matchup_machine" / "src",
        Path(__file__).parents[5] / "matchup_machine" / "src",
        # Env-var override
        Path(os.getenv("MATCHUP_MACHINE_PATH", "__none__")),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_local_model() -> Optional[Path]:
    """Return path to the local joblib model file if it exists."""
    candidates = [
        Path(__file__).parents[6] / "matchup_machine" / "models" / "xgb_outcome_model.joblib",
        Path(__file__).parents[5] / "matchup_machine" / "models" / "xgb_outcome_model.joblib",
        Path(os.getenv("XGB_MODEL_PATH", "__none__")),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# HuggingFace Hub download
# ---------------------------------------------------------------------------

def _download_from_hub(hf_repo: str, hf_token: str) -> Optional[Path]:
    """
    Download ML artifacts from a HuggingFace Hub repository.

    Expected repo layout:
        xgb_outcome_model.joblib
        matchups.parquet
        player_index.csv
        pitcher_profiles.parquet  (optional)
        batter_profiles.parquet   (optional)
        pa_projections.parquet    (optional)

    Returns the local cache directory, or None on failure.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        st.warning("huggingface_hub not installed. Run: pip install huggingface_hub")
        return None

    try:
        local_dir = snapshot_download(
            repo_id=hf_repo,
            token=hf_token,
            local_dir=Path("/tmp/baseball_artifacts"),
            ignore_patterns=["*.md", "*.txt"],
        )
        return Path(local_dir)
    except Exception as exc:
        st.warning(f"HuggingFace Hub download failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading ML models (this happens once)...")
def load_artifacts() -> Optional[Tuple[Any, ...]]:
    """
    Load all matchup_machine artifacts.

    Returns a tuple matching the return value of
    ``matchup_machine.fantasy_inference.load_artifacts()``, or None if
    artifacts are unavailable.

    Tuple: (model, feature_cols, pitcher_profiles, batter_profiles,
             player_index, pa_proj, matchups)
    """
    # Resolve matchup_machine source
    mm_src = _find_local_matchup_machine()
    if mm_src and str(mm_src) not in sys.path:
        sys.path.insert(0, str(mm_src))

    # Try importing load_artifacts from matchup_machine
    try:
        from fantasy_inference import load_artifacts as _load  # type: ignore
        artifacts = _load()
        return artifacts
    except ImportError:
        pass  # matchup_machine not on path — try HF Hub path below
    except Exception as exc:
        st.warning(f"Failed to load local artifacts: {exc}")
        return None

    # Try HF Hub
    hf_token = _get_secret("HF_TOKEN")
    hf_repo = _get_secret("HF_MODEL_REPO")

    if not hf_token or not hf_repo:
        return None  # No remote source configured

    artifact_dir = _download_from_hub(hf_repo, hf_token)
    if artifact_dir is None:
        return None

    # Point matchup_machine at the downloaded artifacts via env vars so
    # load_artifacts() picks them up.
    os.environ.setdefault("MM_MODEL_PATH", str(artifact_dir / "xgb_outcome_model.joblib"))
    os.environ.setdefault("MM_MATCHUPS_PATH", str(artifact_dir / "matchups.parquet"))
    os.environ.setdefault("MM_PLAYER_INDEX_PATH", str(artifact_dir / "player_index.csv"))

    if str(artifact_dir) not in sys.path:
        sys.path.insert(0, str(artifact_dir))

    try:
        from fantasy_inference import load_artifacts as _load  # type: ignore
        return _load()
    except Exception as exc:
        st.warning(f"Failed to load Hub artifacts: {exc}")
        return None


def ml_available() -> bool:
    """True if ML artifacts loaded successfully."""
    return load_artifacts() is not None


# ---------------------------------------------------------------------------
# Secret resolution helper
# ---------------------------------------------------------------------------

def _get_secret(key: str) -> Optional[str]:
    """Read a secret from st.secrets then env vars."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key)
