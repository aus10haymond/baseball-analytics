"""
Diamond Mind Admin Page

Password-gated control panel for the Diamond Mind multi-agent system.
Lets the admin dispatch tasks to agents, monitor queue health, inspect
results, and review alerts — all without touching Redis CLI.

Access: requires ADMIN_PASSWORD set in Streamlit secrets.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_DASHBOARD_DIR = Path(__file__).parent.parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import streamlit as st

from utils.session_state import init_session_state
from utils.dm_redis import (
    ALERT_QUEUE,
    RESULT_QUEUE,
    TASK_QUEUE,
    clear_queue,
    get_client,
    get_heartbeats,
    get_queue_depths,
    get_recent_raw,
    get_result_by_task_id,
    push_task,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Diamond Mind Admin",
    page_icon="💎",
    layout="wide",
)
init_session_state()

# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------

_AUTH_KEY = "dm_admin_authenticated"

def _check_password(entered: str) -> bool:
    try:
        expected = st.secrets.get("ADMIN_PASSWORD", "")
    except Exception:
        import os
        expected = os.getenv("ADMIN_PASSWORD", "")
    return bool(expected) and entered == expected


if not st.session_state.get(_AUTH_KEY, False):
    st.title("💎 Diamond Mind Admin")
    st.markdown("This page is restricted. Enter the admin password to continue.")
    with st.form("admin_login"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if _check_password(pw):
                st.session_state[_AUTH_KEY] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()

# ---------------------------------------------------------------------------
# Authenticated — main UI
# ---------------------------------------------------------------------------

st.title("💎 Diamond Mind Admin")
st.caption("Multi-agent system control panel — admin only.")

with st.sidebar:
    if st.button("Sign out", use_container_width=True):
        st.session_state[_AUTH_KEY] = False
        st.rerun()

# Redis availability check
_redis_ok = get_client() is not None
if not _redis_ok:
    st.error("Cannot connect to Redis.", icon="🔴")
    st.markdown(
        """
        The Diamond Mind agents communicate over Redis, which needs to be running
        and reachable from Streamlit Cloud. The easiest free option is **Upstash**:

        **1. Create a free Redis database at [upstash.com](https://upstash.com)**
        - Sign up → Create Database → choose a region close to you
        - Copy the **Endpoint**, **Port**, and **Password** from the database details page

        **2. Add these to your Streamlit Cloud secrets**
        (Settings → Secrets):
        ```toml
        DM_REDIS_HOST     = "your-endpoint.upstash.io"
        DM_REDIS_PORT     = "6379"
        DM_REDIS_PASSWORD = "your-password"
        ```

        **3. Also set these same values in your local `.env`** so local dev works:
        ```
        DM_REDIS_HOST=your-endpoint.upstash.io
        DM_REDIS_PORT=6379
        DM_REDIS_PASSWORD=your-password
        ```

        Once the secrets are saved, Streamlit Cloud will restart the app automatically.
        """
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_status, tab_send, tab_results, tab_alerts = st.tabs(
    ["System Status", "Send Task", "Results", "Alerts"]
)

# ── Tab 1: System Status ──────────────────────────────────────────────────

with tab_status:
    st.subheader("Queue Depths")

    if st.button("Refresh", key="refresh_status"):
        st.rerun()

    depths = get_queue_depths()
    c1, c2, c3 = st.columns(3)
    c1.metric("Tasks pending", depths.get(TASK_QUEUE, "—"))
    c2.metric("Results stored", depths.get(RESULT_QUEUE, "—"))
    c3.metric("Alerts", depths.get(ALERT_QUEUE, "—"))

    st.divider()
    st.subheader("Agent Heartbeats")

    heartbeats = get_heartbeats()
    agents = ["orchestrator", "data_quality", "model_monitor", "feature_engineer", "explainer"]

    cols = st.columns(len(agents))
    now = datetime.now()
    for col, agent in zip(cols, agents):
        ts_str = heartbeats.get(agent)
        if ts_str:
            try:
                last = datetime.fromisoformat(ts_str)
                elapsed = (now - last).total_seconds()
                label = f"{elapsed:.0f}s ago"
                icon = "🟢" if elapsed < 120 else "🟡" if elapsed < 300 else "🔴"
                col.metric(agent, f"{icon} {label}")
            except ValueError:
                col.metric(agent, "⚠️ bad timestamp")
        else:
            col.metric(agent, "⚫ no heartbeat")

    st.caption(
        "Green = heartbeat within 2 min. Yellow = 2–5 min. Red = >5 min. "
        "Black = agent has never sent a heartbeat (not running)."
    )

    st.divider()
    st.subheader("Danger Zone")
    exp = st.expander("Clear queues")
    with exp:
        st.warning("This permanently deletes all pending items from the selected queue.", icon="⚠️")
        qcol1, qcol2, qcol3 = st.columns(3)
        if qcol1.button("Clear task queue", use_container_width=True):
            clear_queue(TASK_QUEUE)
            st.success("Task queue cleared.")
        if qcol2.button("Clear result queue", use_container_width=True):
            clear_queue(RESULT_QUEUE)
            st.success("Result queue cleared.")
        if qcol3.button("Clear alert queue", use_container_width=True):
            clear_queue(ALERT_QUEUE)
            st.success("Alert queue cleared.")

# ── Tab 2: Send Task ──────────────────────────────────────────────────────

# Task-type options and default parameter templates per agent
_TASK_DEFS: dict[str, dict[str, dict]] = {
    "orchestrator": {
        "route_task": {
            "task_description": "Check for data quality issues in the batter projection file",
            "context": {},
        },
        "system_health": {
            "stale_threshold_seconds": 180,
        },
        "resolve_conflict": {
            "conflict_description": "Two agents disagree on retraining",
            "recommendations": [
                {"agent_id": "model_monitor", "recommendation": "retrain now", "data": {}},
                {"agent_id": "data_quality", "recommendation": "wait for more data", "data": {}},
            ],
        },
        "retrain_model": {
            "model_name": "xgb_outcome_model",
            "reason": "manual trigger",
            "triggered_by": "admin",
        },
    },
    "data_quality": {
        "check_data_quality": {
            "data_source": "packages/matchup_machine/data/player_index.csv",
            "auto_fix": False,
            "threshold": 3.0,
            "contamination": 0.05,
        },
        "detect_anomalies": {
            "data_source": "packages/matchup_machine/data/player_index.csv",
            "threshold": 3.0,
            "contamination": 0.05,
        },
        "validate_schema": {
            "data_source": "packages/matchup_machine/data/player_index.csv",
        },
        "repair_data": {
            "data_source": "packages/matchup_machine/data/player_index.csv",
            "output_path": "packages/matchup_machine/data/player_index_repaired.csv",
        },
    },
    "model_monitor": {
        "check_drift": {
            "model_name": "xgb_outcome_model",
            "data_source": "packages/matchup_machine/data/player_index.csv",
        },
        "evaluate_performance": {
            "model_name": "xgb_outcome_model",
            "accuracy": 0.72,
            "precision": 0.68,
            "recall": 0.70,
            "f1_score": 0.69,
            "log_loss": 0.55,
        },
        "run_ab_test": {
            "test_name": "outcome_model_v2_vs_v1",
            "model_a": "xgb_outcome_model_v1",
            "model_b": "xgb_outcome_model_v2",
            "traffic_split": 0.5,
        },
        "register_model_version": {
            "model_name": "xgb_outcome_model",
            "version_id": "v2",
            "metrics": {"accuracy": 0.72},
            "model_path": "packages/matchup_machine/models/xgb_outcome_model.joblib",
        },
        "rollback_model": {
            "model_name": "xgb_outcome_model",
            "target_version": "v1",
        },
    },
    "feature_engineer": {
        "search_features": {
            "data_source": "packages/matchup_machine/data/player_index.csv",
            "target_column": "is_batter",
            "max_features": 20,
        },
        "evaluate_feature": {
            "feature_name": "rolling_avg_pa",
            "feature_definition": "rolling mean of plate appearances over last 30 days",
            "source_features": ["projected_pa"],
        },
        "generate_features": {
            "data_source": "packages/matchup_machine/data/player_index.csv",
            "existing_features": ["player_id", "is_batter", "is_pitcher"],
        },
    },
    "explainer": {
        "explain_prediction": {
            "model_path": "packages/matchup_machine/models/xgb_outcome_model.joblib",
            "player_id": 592450,
            "features": {"projected_pa": 650, "is_batter": 1},
        },
        "explain_batch": {
            "model_path": "packages/matchup_machine/models/xgb_outcome_model.joblib",
            "player_ids": [592450, 660271],
            "features_list": [
                {"projected_pa": 650, "is_batter": 1},
                {"projected_pa": 480, "is_batter": 1},
            ],
        },
        "get_cached": {
            "task_id": "enter-task-id-here",
        },
        "clear_cache": {
            "task_id": None,
        },
    },
}

with tab_send:
    st.subheader("Dispatch a Task")

    col_agent, col_type = st.columns([1, 1])

    with col_agent:
        selected_agent = st.selectbox(
            "Target Agent",
            options=list(_TASK_DEFS.keys()),
            key="send_agent",
        )

    task_types = list(_TASK_DEFS[selected_agent].keys())

    with col_type:
        selected_task_type = st.selectbox(
            "Task Type",
            options=task_types,
            key="send_task_type",
        )

    default_params = _TASK_DEFS[selected_agent][selected_task_type]
    default_json = json.dumps(default_params, indent=2)

    params_json = st.text_area(
        "Parameters (JSON)",
        value=default_json,
        height=220,
        key=f"send_params_{selected_agent}_{selected_task_type}",
        help="Edit the JSON parameters below. The defaults are pre-filled for the selected task type.",
    )

    priority = st.select_slider(
        "Priority",
        options=["low", "medium", "high", "critical"],
        value="medium",
    )

    send_col, _ = st.columns([1, 3])
    with send_col:
        send_clicked = st.button("Send Task", type="primary", use_container_width=True)

    if send_clicked:
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        else:
            task_id = push_task(
                agent_id=selected_agent,
                task_type=selected_task_type,
                parameters=params,
                priority=priority,
            )
            if task_id:
                st.success(f"Task dispatched. **Task ID:** `{task_id}`")
                st.info(
                    "Check the **Results** tab and enter this task ID to retrieve the result "
                    "after the agent processes it.",
                    icon="💡",
                )
            else:
                st.error("Failed to push task — is Redis running?")

# ── Tab 3: Results ────────────────────────────────────────────────────────

with tab_results:
    st.subheader("Look up by Task ID")

    lookup_col, btn_col = st.columns([3, 1])
    with lookup_col:
        lookup_id = st.text_input("Task ID", placeholder="paste task id here", label_visibility="collapsed")
    with btn_col:
        lookup_clicked = st.button("Look up", use_container_width=True)

    if lookup_clicked and lookup_id.strip():
        result = get_result_by_task_id(lookup_id.strip())
        if result:
            status = result.get("status", "unknown")
            color = {"completed": "🟢", "failed": "🔴", "running": "🟡"}.get(status, "⚫")
            st.markdown(f"**Status:** {color} `{status}`")
            st.markdown(f"**Agent:** `{result.get('agent_id')}`")
            if result.get("error_message"):
                st.error(result["error_message"])
            if result.get("metrics"):
                st.markdown("**Metrics:**")
                st.json(result["metrics"])
            if result.get("result_data"):
                with st.expander("Full result data", expanded=True):
                    st.json(result["result_data"])
        else:
            st.warning("No result found for that task ID yet. The agent may still be processing it.")

    st.divider()
    st.subheader("Recent Results")

    n_results = st.slider("Show last N results", 5, 50, 10, key="n_results")
    if st.button("Refresh results", key="refresh_results"):
        st.rerun()

    recent = get_recent_raw(RESULT_QUEUE, n_results)
    if not recent:
        st.info("No results in the queue yet.")
    else:
        for r in recent:
            status = r.get("status", "unknown")
            color = {"completed": "🟢", "failed": "🔴", "running": "🟡"}.get(status, "⚫")
            agent = r.get("agent_id", "unknown")
            task_id_short = r.get("task_id", "")[:8]
            completed_at = r.get("completed_at", "")[:19].replace("T", " ")
            label = f"{color} `{agent}` — `{task_id_short}…` — {completed_at}"
            with st.expander(label):
                if r.get("metrics"):
                    st.markdown("**Metrics:**")
                    st.json(r["metrics"])
                if r.get("error_message"):
                    st.error(r["error_message"])
                if r.get("result_data"):
                    st.json(r["result_data"])

# ── Tab 4: Alerts ─────────────────────────────────────────────────────────

_SEVERITY_ICON = {"info": "🔵", "warning": "🟡", "error": "🔴", "critical": "🚨"}

with tab_alerts:
    st.subheader("Recent Alerts")

    alert_col1, alert_col2 = st.columns([1, 1])
    with alert_col1:
        n_alerts = st.slider("Show last N alerts", 5, 100, 20, key="n_alerts")
    with alert_col2:
        severity_filter = st.multiselect(
            "Filter by severity",
            ["info", "warning", "error", "critical"],
            default=["warning", "error", "critical"],
        )

    if st.button("Refresh alerts", key="refresh_alerts"):
        st.rerun()

    alerts = get_recent_raw(ALERT_QUEUE, n_alerts)
    filtered = [a for a in alerts if a.get("severity") in severity_filter] if severity_filter else alerts

    if not filtered:
        st.info("No alerts matching the current filter.")
    else:
        for alert in filtered:
            severity = alert.get("severity", "info")
            icon = _SEVERITY_ICON.get(severity, "⚫")
            agent = alert.get("agent_id", "unknown")
            message = alert.get("message", "")
            ts = alert.get("created_at", "")[:19].replace("T", " ")
            requires_action = alert.get("requires_action", False)

            label = f"{icon} `{agent}` — {message[:80]}{'…' if len(message) > 80 else ''} — {ts}"
            with st.expander(label):
                st.markdown(f"**Severity:** `{severity}`")
                st.markdown(f"**Agent:** `{agent}`")
                st.markdown(f"**Message:** {message}")
                if requires_action:
                    st.warning("This alert requires action.", icon="⚠️")
                if alert.get("suggested_actions"):
                    st.markdown("**Suggested actions:**")
                    for action in alert["suggested_actions"]:
                        st.markdown(f"- {action}")
                if alert.get("details"):
                    with st.expander("Details"):
                        st.json(alert["details"])
