"""
Shared test fixtures for Diamond Mind tests.

Provides fake Redis connections, sample schemas, data fixtures,
and mock LLM responses used across the test suite.
"""

import sys
from pathlib import Path

# Ensure shared package is importable (fallback if pythonpath config doesn't apply)
_dm_src = str(Path(__file__).resolve().parent.parent / "src" / "diamond_mind")
if _dm_src not in sys.path:
    sys.path.insert(0, _dm_src)

import pytest
import numpy as np
import pandas as pd
import fakeredis.aioredis

from shared.schemas import (
    AgentType,
    TaskStatus,
    TaskPriority,
    AlertSeverity,
    ConfidenceLevel,
    AgentTask,
    AgentResult,
    AgentAlert,
    DataAnomalyReport,
    DataQualityMetrics,
)
from shared.messaging import MessageQueue, message_queue


# ---------------------------------------------------------------------------
# Redis fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def fake_mq():
    """Create a standalone MessageQueue with a fake Redis backend."""
    mq = MessageQueue()
    mq.redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield mq
    await mq.redis_client.flushall()
    await mq.redis_client.aclose()


@pytest.fixture
async def global_mq():
    """Patch the global message_queue singleton with fake Redis.

    Use this fixture for tests that exercise code relying on the
    module-level ``message_queue`` instance (e.g. BaseAgent).
    """
    message_queue.redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield message_queue
    await message_queue.redis_client.flushall()
    await message_queue.redis_client.aclose()
    message_queue.redis_client = None


# ---------------------------------------------------------------------------
# Sample schema fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_task():
    """A representative AgentTask for data-quality work."""
    return AgentTask(
        task_id="test_task_001",
        agent_id=AgentType.DATA_QUALITY,
        task_type="check_anomalies",
        priority=TaskPriority.HIGH,
        parameters={"data_source": "test_data", "threshold": 3.0},
    )


@pytest.fixture
def sample_result():
    """A completed AgentResult with metrics."""
    return AgentResult(
        task_id="test_task_001",
        agent_id=AgentType.DATA_QUALITY,
        status=TaskStatus.COMPLETED,
        result_data={"issues_found": 3, "auto_fixed": 2},
        metrics={"anomaly_score": 0.85},
        duration_seconds=12.5,
    )


@pytest.fixture
def sample_alert():
    """A warning-level AgentAlert from the model monitor."""
    return AgentAlert(
        alert_id="alert_001",
        agent_id=AgentType.MODEL_MONITOR,
        severity=AlertSeverity.WARNING,
        message="Model drift detected",
        details={"psi_score": 0.25},
        requires_action=True,
        suggested_actions=["Retrain model", "Review data pipeline"],
    )


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataframe():
    """Small baseball analytics DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "player_name": [f"Player_{i}" for i in range(20)],
            "batting_avg": np.random.uniform(0.200, 0.350, 20).round(3),
            "home_runs": np.random.randint(5, 45, 20),
            "rbi": np.random.randint(20, 120, 20),
            "ops": np.random.uniform(0.600, 1.100, 20).round(3),
            "war": np.random.uniform(-0.5, 8.0, 20).round(1),
        }
    )


@pytest.fixture
def sample_parquet(tmp_path, sample_dataframe):
    """Write sample DataFrame to a temporary parquet file and return path."""
    path = tmp_path / "sample_data.parquet"
    sample_dataframe.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_responses():
    """Canned LLM responses for Orchestrator / Explainer tests."""
    return {
        "task_routing": {
            "role": "assistant",
            "content": (
                '{"target_agent": "data_quality", '
                '"task_type": "check_anomalies", "priority": "high"}'
            ),
        },
        "explanation": {
            "role": "assistant",
            "content": (
                "The model predicted a high batting average because "
                "the player has historically performed well against left-handed pitchers."
            ),
        },
        "decision": {
            "role": "assistant",
            "content": (
                '{"action": "retrain", '
                '"reason": "Significant drift detected in pitch velocity features"}'
            ),
        },
    }
