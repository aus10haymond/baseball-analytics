"""
Recommendations Page

Runs ML projections for the user's roster against today's probable pitchers
and displays start/sit recommendations with charts.
"""

import sys
from pathlib import Path

_DASHBOARD_DIR = Path(__file__).parent.parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from utils.model_loader import load_artifacts, ml_available
from utils.session_state import (
    get_projections,
    get_roster,
    init_session_state,
    set_projections,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Recommendations — Fantasy MLB AI",
    page_icon="📊",
    layout="wide",
)
init_session_state()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_ORDER = {"very_high": 5, "high": 4, "medium": 3, "low": 2, "very_low": 1, "none": 0}
CONFIDENCE_COLORS = {
    "very_high": "#1a7a1a",
    "high": "#2ecc71",
    "medium": "#f39c12",
    "low": "#e74c3c",
    "very_low": "#c0392b",
    "none": "#95a5a6",
}
MATCHUP_LABELS = {
    "head_to_head": "Head-to-Head",
    "pitcher_profile": "Pitcher Profile",
    "general": "General",
    "no_game": "No Game",
    "no_data": "No Data",
    "error": "Error",
}


# ---------------------------------------------------------------------------
# MLB API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_todays_matchups() -> Dict[str, Dict]:
    """
    Fetch today's probable pitchers from the free MLB Stats API.
    Returns {full_team_name: {opponent_pitcher, opponent_team, is_home}}.
    """
    today = date.today().strftime("%Y-%m-%d")
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={today}&hydrate=probablePitcher"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except Exception:
        return {}

    matchups: Dict[str, Dict] = {}
    for date_info in data.get("dates", []):
        for game in date_info.get("games", []):
            home = game["teams"]["home"]["team"]["name"]
            away = game["teams"]["away"]["team"]["name"]
            home_pitcher = game["teams"]["home"].get("probablePitcher", {}).get("fullName")
            away_pitcher = game["teams"]["away"].get("probablePitcher", {}).get("fullName")
            if away_pitcher:
                matchups[home] = {"opponent_pitcher": away_pitcher, "opponent_team": away, "is_home": True}
            if home_pitcher:
                matchups[away] = {"opponent_pitcher": home_pitcher, "opponent_team": home, "is_home": False}
    return matchups


# ---------------------------------------------------------------------------
# Projection runner
# ---------------------------------------------------------------------------

def _run_projections(roster: List[Dict], matchups: Dict[str, Dict]) -> List[Dict]:
    """
    Generate pitcher-aware ML projections for each roster player.
    Falls back to placeholder rows if ML artifacts aren't available.
    """
    artifacts = load_artifacts()

    results = []
    for player in roster:
        name = player["name"]
        team = player["team"]
        position = player["position"]

        # Skip pitchers for batting projections
        if position in ("SP", "RP"):
            results.append({
                "name": name,
                "position": position,
                "team": team,
                "projected_points": None,
                "confidence": "none",
                "matchup_type": "pitcher",
                "opponent_pitcher": None,
                "opponent_team": None,
                "is_home": None,
                "sample_size": None,
                "error": "Pitcher projections not yet supported",
            })
            continue

        # Find today's matchup for this player's team
        matchup = matchups.get(team, {})
        opponent_pitcher = matchup.get("opponent_pitcher")
        opponent_team = matchup.get("opponent_team")
        is_home = matchup.get("is_home")

        if not matchup:
            results.append({
                "name": name,
                "position": position,
                "team": team,
                "projected_points": None,
                "confidence": "none",
                "matchup_type": "no_game",
                "opponent_pitcher": None,
                "opponent_team": None,
                "is_home": None,
                "sample_size": None,
                "error": "No game today",
            })
            continue

        if artifacts is None:
            # ML not available — return placeholder
            results.append({
                "name": name,
                "position": position,
                "team": team,
                "projected_points": None,
                "confidence": "none",
                "matchup_type": "no_data",
                "opponent_pitcher": opponent_pitcher,
                "opponent_team": opponent_team,
                "is_home": is_home,
                "sample_size": None,
                "error": "ML models not loaded",
            })
            continue

        # Run pitcher-aware projection
        try:
            model, feature_cols, pitcher_profiles, batter_profiles, player_index, pa_proj, matchups_df = artifacts

            # Import projection helpers inline to avoid top-level import errors
            import sys as _sys
            from pathlib import Path as _Path
            _mm = _Path(__file__).parents[5] / "matchup_machine" / "src"
            if _mm.exists() and str(_mm) not in _sys.path:
                _sys.path.insert(0, str(_mm))

            from matchup_machine.fantasy_inference import find_player_id  # type: ignore
            from matchup_machine.fantasy_scoring import expected_hitter_points_per_pa  # type: ignore
            from matchup_machine.build_dataset import OUTCOME_LABELS  # type: ignore

            batter_id = int(find_player_id(player_index, name))
            pitcher_id: Optional[int] = None
            h2h_sample = 0

            if opponent_pitcher:
                try:
                    pitcher_id = int(find_player_id(player_index, opponent_pitcher))
                except ValueError:
                    pitcher_id = None

            batter_pas = matchups_df[
                (matchups_df["batter"] == batter_id)
                & (matchups_df["date"].dt.year >= 2024)
                & (matchups_df["outcome_id"].notna())
            ].copy()

            if batter_pas.empty:
                raise ValueError(f"No historical data for {name}")

            # Choose projection data based on available matchup info
            if pitcher_id is not None:
                h2h = batter_pas[batter_pas["pitcher"] == pitcher_id]
                if not h2h.empty:
                    projection_data = h2h
                    matchup_type = "head_to_head"
                    h2h_sample = len(h2h)
                else:
                    pp = pitcher_profiles[pitcher_profiles["pitcher"] == pitcher_id]
                    if not pp.empty:
                        projection_data = batter_pas.copy()
                        for col in [c for c in pp.columns if c != "pitcher"]:
                            if col in projection_data.columns:
                                projection_data[col] = pp[col].iloc[0]
                        matchup_type = "pitcher_profile"
                    else:
                        projection_data = batter_pas
                        matchup_type = "general"
            else:
                projection_data = batter_pas
                matchup_type = "general"

            X = projection_data.reindex(columns=feature_cols, fill_value=0)
            X_filled = X.copy()
            for col in X_filled.columns:
                if X_filled[col].isna().any():
                    if str(X_filled[col].dtype).startswith("Int"):
                        X_filled[col] = X_filled[col].astype("float64")
                    X_filled[col] = X_filled[col].fillna(X_filled[col].median())
                if str(X_filled[col].dtype).startswith("Int"):
                    X_filled[col] = X_filled[col].astype("float64")

            probs = model.predict_proba(X_filled)
            avg_probs = probs.mean(axis=0)
            outcome_probs = {label: float(p) for label, p in zip(OUTCOME_LABELS, avg_probs)}
            ev_per_pa = expected_hitter_points_per_pa(outcome_probs)
            expected_points = round(ev_per_pa * 4, 2)

            # --- Confidence via relative standard error ---
            # Compute expected fantasy points independently for each PA, then
            # measure how much the model's predictions vary.  RSE = SEM / mean
            # captures both sample size and prediction variance in one number.
            # Lower RSE = tighter distribution = higher confidence.
            import numpy as _np
            per_pa_ev = _np.array([
                expected_hitter_points_per_pa(
                    {label: float(p) for label, p in zip(OUTCOME_LABELS, row)}
                )
                for row in probs
            ])
            n = len(per_pa_ev)
            mean_ev = per_pa_ev.mean()
            std_ev = per_pa_ev.std() if n > 1 else 1.0
            sem = std_ev / _np.sqrt(n)
            # Avoid division by zero for players projected near 0 pts/PA
            rse = sem / abs(mean_ev) if abs(mean_ev) > 0.01 else 1.0

            # Matchup type scales RSE: specific data reduces effective uncertainty
            _matchup_adj = {
                "head_to_head": 0.55,
                "pitcher_profile": 0.80,
                "general": 1.20,
            }
            adj_rse = rse * _matchup_adj.get(matchup_type, 1.0)

            if adj_rse < 0.08:
                confidence = "very_high"   # ±8% relative error
            elif adj_rse < 0.15:
                confidence = "high"        # ±15%
            elif adj_rse < 0.28:
                confidence = "medium"      # ±28%
            elif adj_rse < 0.50:
                confidence = "low"         # ±50%
            else:
                confidence = "very_low"

            sample_size = n

            results.append({
                "name": name,
                "position": position,
                "team": team,
                "projected_points": expected_points,
                "confidence": confidence,
                "confidence_rse": round(float(adj_rse), 3),
                "matchup_type": matchup_type,
                "opponent_pitcher": opponent_pitcher,
                "opponent_team": opponent_team,
                "is_home": is_home,
                "sample_size": sample_size,
                "error": None,
            })

        except ValueError as exc:
            results.append({
                "name": name,
                "position": position,
                "team": team,
                "projected_points": None,
                "confidence": "none",
                "matchup_type": "error",
                "opponent_pitcher": opponent_pitcher,
                "opponent_team": opponent_team,
                "is_home": is_home,
                "sample_size": None,
                "error": str(exc),
            })
        except Exception as exc:
            results.append({
                "name": name,
                "position": position,
                "team": team,
                "projected_points": None,
                "confidence": "none",
                "matchup_type": "error",
                "opponent_pitcher": opponent_pitcher,
                "opponent_team": opponent_team,
                "is_home": is_home,
                "sample_size": None,
                "error": f"Projection failed: {exc}",
            })

    return results


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _confidence_badge(conf: str) -> str:
    color = CONFIDENCE_COLORS.get(conf, "#95a5a6")
    label = conf.replace("_", " ").title()
    return f'<span style="color:{color};font-weight:bold">{label}</span>'


def _render_summary_metrics(df: pd.DataFrame) -> None:
    batters = df[~df["position"].isin(["SP", "RP"])]
    playable = batters[batters["projected_points"].notna()]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Players with projections", len(playable))
    with col2:
        if not playable.empty:
            st.metric("Avg projected pts", f"{playable['projected_points'].mean():.2f}")
    with col3:
        if not playable.empty:
            st.metric("Top projection", f"{playable['projected_points'].max():.2f}")
    with col4:
        high_conf = playable[playable["confidence"].isin(["high", "very_high"])]
        st.metric("High-confidence picks", len(high_conf))


def _render_projection_chart(df: pd.DataFrame) -> None:
    playable = df[df["projected_points"].notna() & ~df["position"].isin(["SP", "RP"])]
    if playable.empty:
        return

    sorted_df = playable.sort_values("projected_points", ascending=True)
    colors = [CONFIDENCE_COLORS.get(c, "#95a5a6") for c in sorted_df["confidence"]]

    fig = go.Figure(go.Bar(
        x=sorted_df["projected_points"],
        y=sorted_df["name"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in sorted_df["projected_points"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Projected: %{x:.2f} pts<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Projected Fantasy Points Today",
        xaxis_title="Projected Points",
        yaxis_title="",
        height=max(300, len(sorted_df) * 32),
        margin=dict(l=20, r=60, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def _render_confidence_breakdown(df: pd.DataFrame) -> None:
    playable = df[df["projected_points"].notna()]
    if playable.empty:
        return
    conf_counts = playable["confidence"].value_counts().reset_index()
    conf_counts.columns = ["confidence", "count"]
    conf_counts["label"] = conf_counts["confidence"].str.replace("_", " ").str.title()
    conf_counts["color"] = conf_counts["confidence"].map(CONFIDENCE_COLORS)

    fig = px.pie(
        conf_counts,
        values="count",
        names="label",
        color="confidence",
        color_discrete_map={row["confidence"]: row["color"] for _, row in conf_counts.iterrows()},
        title="Confidence Breakdown",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")


def _render_recommendations_table(df: pd.DataFrame) -> None:
    batters = df[~df["position"].isin(["SP", "RP"])].copy()
    pitchers = df[df["position"].isin(["SP", "RP"])].copy()

    batters_ranked = batters.sort_values(
        by="projected_points", ascending=False, na_position="last"
    ).reset_index(drop=True)

    st.markdown("#### Batters")
    if batters_ranked.empty:
        st.info("No batters on roster.")
    else:
        for i, row in batters_ranked.iterrows():
            pts = f"{row['projected_points']:.2f}" if pd.notna(row["projected_points"]) else "—"
            conf = row["confidence"]
            conf_color = CONFIDENCE_COLORS.get(conf, "#95a5a6")
            matchup_label = MATCHUP_LABELS.get(row["matchup_type"], row["matchup_type"])
            pitcher_str = row["opponent_pitcher"] if pd.notna(row.get("opponent_pitcher")) else "—"
            home_away = "🏠" if row.get("is_home") else "✈️"
            error = row.get("error")

            with st.container():
                c1, c2, c3, c4, c5, c6 = st.columns([3, 1, 3, 2, 2, 2])
                with c1:
                    st.markdown(f"**{row['name']}**")
                with c2:
                    st.markdown(f"`{row['position']}`")
                with c3:
                    if pd.notna(row.get("opponent_pitcher")):
                        st.markdown(f"{home_away} vs **{pitcher_str}**")
                    elif row["matchup_type"] == "no_game":
                        st.markdown("*No game today*")
                    else:
                        st.markdown("—")
                with c4:
                    rse = row.get("confidence_rse")
                    rse_str = f" (RSE {rse:.2f})" if pd.notna(rse) else ""
                    st.markdown(
                        f'<span style="color:{conf_color};font-size:0.85em">'
                        f'{conf.replace("_"," ").title()}{rse_str}</span>',
                        unsafe_allow_html=True,
                    )
                with c5:
                    st.markdown(f"**{pts}** pts")
                with c6:
                    if isinstance(error, str) and row["matchup_type"] not in ("no_game", "pitcher"):
                        st.caption(f"⚠️ {error[:40]}")

        st.divider()

    st.markdown("#### Pitchers")
    if pitchers.empty:
        st.info("No pitchers on roster.")
    else:
        for _, row in pitchers.iterrows():
            c1, c2, c3 = st.columns([3, 1, 6])
            with c1:
                st.markdown(f"**{row['name']}**")
            with c2:
                st.markdown(f"`{row['position']}`")
            with c3:
                st.caption("Pitcher projections coming in a future update.")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("📊 Today's Recommendations")
st.caption(f"Projections for {date.today().strftime('%A, %B %-d, %Y')}")

roster = get_roster()

if not roster:
    st.warning("Your roster is empty. Go to the **Roster** page to add your players first.", icon="👈")
    st.stop()

# ---------------------------------------------------------------------------
# ML status banner
# ---------------------------------------------------------------------------

if not ml_available():
    st.error(
        "ML models are not loaded. Projections are unavailable. "
        "Upload your model artifacts to HuggingFace Hub and add "
        "`HF_TOKEN` + `HF_MODEL_REPO` to your Streamlit secrets.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Run / refresh projections
# ---------------------------------------------------------------------------

existing_projections = get_projections()
run_btn_label = "Refresh Projections" if existing_projections else "Run Projections"

col_btn, col_status = st.columns([2, 6])
with col_btn:
    run_clicked = st.button(run_btn_label, type="primary", use_container_width=True)

if run_clicked or existing_projections is None:
    with st.spinner("Fetching today's matchups and running projections..."):
        matchups = fetch_todays_matchups()
        projections = _run_projections(roster, matchups)
        set_projections(projections)
    st.success("Projections updated!")
    existing_projections = projections

if existing_projections is None:
    st.stop()

df = pd.DataFrame(existing_projections)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

st.divider()
_render_summary_metrics(df)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

st.divider()
chart_col, conf_col = st.columns([3, 1])

with chart_col:
    _render_projection_chart(df)

with conf_col:
    _render_confidence_breakdown(df)

# ---------------------------------------------------------------------------
# Recommendations table
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Player Breakdown")

# Filters
filter_col1, filter_col2, _ = st.columns([2, 2, 4])
with filter_col1:
    show_no_game = st.checkbox("Show players with no game today", value=False)
with filter_col2:
    min_conf = st.selectbox(
        "Minimum confidence",
        options=["all", "very_high", "high", "medium", "low"],
        format_func=lambda x: "All" if x == "all" else x.replace("_", " ").title(),
    )

filtered_df = df.copy()
if not show_no_game:
    filtered_df = filtered_df[filtered_df["matchup_type"] != "no_game"]
if min_conf != "all":
    min_val = CONFIDENCE_ORDER[min_conf]
    filtered_df = filtered_df[
        filtered_df["confidence"].map(CONFIDENCE_ORDER).fillna(0) >= min_val
    ]

_render_recommendations_table(filtered_df)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

st.divider()
csv = df.to_csv(index=False)
st.download_button(
    label="Download as CSV",
    data=csv,
    file_name=f"recommendations-{date.today()}.csv",
    mime="text/csv",
)

st.caption(
    "Confidence is measured by **relative standard error (RSE)** — how much the model's "
    "per-PA predictions vary, normalised by the projection size, then adjusted for matchup "
    "quality (H2H ×0.55 · pitcher profile ×0.80 · general ×1.20). "
    "**Very High** RSE < 0.08 · **High** < 0.15 · **Medium** < 0.28 · **Low** < 0.50 · "
    "**Very Low** ≥ 0.50. RSE scores are shown next to each tier label."
)
