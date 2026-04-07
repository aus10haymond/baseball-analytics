"""
Diamond Mind Redis Helper (Sync)

Thin synchronous wrapper around redis.Redis for use in Streamlit pages,
which are not async.  Reads connection settings from st.secrets (Streamlit
Cloud) with fallbacks to environment variables / defaults for local dev.

Usage:
    from utils.dm_redis import get_client, push_task, get_recent_results, ...
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Optional

import redis
import streamlit as st


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def _redis_url() -> str:
    """Build a Redis URL from secrets → env vars → localhost defaults."""
    try:
        host = st.secrets.get("DM_REDIS_HOST", "localhost")
        port = int(st.secrets.get("DM_REDIS_PORT", 6379))
        db   = int(st.secrets.get("DM_REDIS_DB", 0))
        pw   = st.secrets.get("DM_REDIS_PASSWORD", None)
    except Exception:
        import os
        host = os.getenv("DM_REDIS_HOST", "localhost")
        port = int(os.getenv("DM_REDIS_PORT", 6379))
        db   = int(os.getenv("DM_REDIS_DB", 0))
        pw   = os.getenv("DM_REDIS_PASSWORD", None)

    if pw:
        return f"redis://:{pw}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


@st.cache_resource(show_spinner=False)
def get_client() -> Optional[redis.Redis]:
    """Return a cached sync Redis client, or None if Redis is unreachable."""
    try:
        client = redis.Redis.from_url(_redis_url(), decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Queue names (match diamond_mind defaults)
# ---------------------------------------------------------------------------

TASK_QUEUE    = "diamond_mind:tasks"
RESULT_QUEUE  = "diamond_mind:results"
ALERT_QUEUE   = "diamond_mind:alerts"


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def push_task(
    agent_id: str,
    task_type: str,
    parameters: dict[str, Any],
    priority: str = "medium",
    timeout_seconds: Optional[int] = None,
) -> Optional[str]:
    """
    Push an AgentTask JSON payload onto the task queue.
    Returns the generated task_id, or None on failure.
    """
    client = get_client()
    if client is None:
        return None

    task_id = str(uuid.uuid4())
    payload = {
        "task_id": task_id,
        "agent_id": agent_id,
        "task_type": task_type,
        "priority": priority,
        "parameters": parameters,
        "created_at": datetime.now().isoformat(),
        "timeout_seconds": timeout_seconds,
        "retry_count": 0,
        "max_retries": 3,
    }
    client.lpush(TASK_QUEUE, json.dumps(payload))
    return task_id


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def get_result_by_task_id(task_id: str) -> Optional[dict]:
    """Look up a stored result by task_id (stored in hash by the agent)."""
    client = get_client()
    if client is None:
        return None
    raw = client.hget(f"results:{task_id}", "data")
    if raw:
        return json.loads(raw)
    return None


def get_recent_raw(queue: str, n: int = 20) -> list[dict]:
    """Return the last *n* JSON items from a list queue (newest first)."""
    client = get_client()
    if client is None:
        return []
    items = client.lrange(queue, 0, n - 1)
    out = []
    for item in items:
        try:
            out.append(json.loads(item))
        except json.JSONDecodeError:
            pass
    return out


# ---------------------------------------------------------------------------
# Health / queue-depth helpers
# ---------------------------------------------------------------------------

def get_queue_depths() -> dict[str, int]:
    client = get_client()
    if client is None:
        return {q: -1 for q in (TASK_QUEUE, RESULT_QUEUE, ALERT_QUEUE)}
    return {
        TASK_QUEUE:   client.llen(TASK_QUEUE),
        RESULT_QUEUE: client.llen(RESULT_QUEUE),
        ALERT_QUEUE:  client.llen(ALERT_QUEUE),
    }


def get_heartbeats() -> dict[str, str]:
    """Return {agent_id: iso_timestamp} for all agents that have sent a heartbeat."""
    client = get_client()
    if client is None:
        return {}
    return client.hgetall("agent_heartbeats") or {}


def clear_queue(queue: str) -> bool:
    """Delete all items from a queue. Returns True on success."""
    client = get_client()
    if client is None:
        return False
    client.delete(queue)
    return True
