"""
Integration tests: end-to-end agent communication.

Covers:
- Task publishing via the global message queue
- Orchestrator route_task → downstream task queued (keyword fallback routing)
- DataQualityAgent._execute_task → result published and retrievable by task_id
- ModelMonitorAgent._execute_task → result published
- Heartbeat round-trip through fakeredis
- Orchestrator system_health with stale/missing heartbeats
- Queue depth changes after publishing
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_dm_src = str(Path(__file__).resolve().parents[3] / "src" / "diamond_mind")
if _dm_src not in sys.path:
    sys.path.insert(0, _dm_src)

from shared.schemas import AgentType, TaskStatus, TaskPriority, AgentTask
from shared.config import settings
from shared.messaging import message_queue
from agents.orchestrator.agent import OrchestratorAgent
from agents.data_quality.agent import DataQualityAgent
from agents.model_monitor.agent import ModelMonitorAgent


# ── Helpers ────────────────────────────────────────────────────────────────


def _task(
    agent_id: AgentType,
    task_type: str,
    params: dict,
    task_id: str = "integ-001",
    priority: TaskPriority = TaskPriority.HIGH,
) -> AgentTask:
    return AgentTask(
        task_id=task_id,
        agent_id=agent_id,
        task_type=task_type,
        priority=priority,
        parameters=params,
    )


@pytest.fixture
def clean_parquet(tmp_path) -> Path:
    """Write a small baseball parquet file and return its path."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "batting_avg": np.random.uniform(0.220, 0.330, 30).round(3),
            "home_runs": np.random.randint(10, 35, 30),
            "rbi": np.random.randint(30, 100, 30),
            "ops": np.random.uniform(0.650, 0.950, 30).round(3),
        }
    )
    path = tmp_path / "clean.parquet"
    df.to_parquet(path, index=False)
    return path


# ── Task publishing / queue depth ─────────────────────────────────────────


async def test_publish_task_increases_queue_depth(global_mq):
    """Publishing a task should increase queue depth by 1."""
    before = await global_mq.get_queue_depth(settings.task_queue_name)
    task = _task(AgentType.DATA_QUALITY, "check_anomalies", {}, task_id="depth-001")
    await global_mq.publish_task(task)
    after = await global_mq.get_queue_depth(settings.task_queue_name)
    assert after == before + 1


async def test_consume_task_decreases_queue_depth(global_mq):
    """Consuming a task should decrease queue depth by 1."""
    task = _task(AgentType.DATA_QUALITY, "check_anomalies", {}, task_id="consume-001")
    await global_mq.publish_task(task)
    before = await global_mq.get_queue_depth(settings.task_queue_name)
    consumed = await global_mq.consume_task(timeout=1)
    after = await global_mq.get_queue_depth(settings.task_queue_name)
    assert consumed is not None
    assert consumed.task_id == "consume-001"
    assert after == before - 1


async def test_result_stored_and_retrievable(global_mq, sample_result):
    """Published results should be retrievable by task_id."""
    await global_mq.publish_result(sample_result)
    retrieved = await global_mq.get_result(sample_result.task_id)
    assert retrieved is not None
    assert retrieved.task_id == sample_result.task_id
    assert retrieved.status == TaskStatus.COMPLETED


# ── Orchestrator routing ───────────────────────────────────────────────────


async def test_orchestrator_route_task_queues_downstream(global_mq):
    """Orchestrator route_task should enqueue a task for the target agent."""
    orch = OrchestratorAgent()
    await orch.initialize()

    before = await global_mq.get_queue_depth(settings.task_queue_name)
    task = _task(
        AgentType.ORCHESTRATOR,
        "route_task",
        {"task_description": "check data quality anomalies in batting stats"},
        task_id="route-001",
    )
    result = await orch.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.result_data["routed_to"] == AgentType.DATA_QUALITY.value
    after = await global_mq.get_queue_depth(settings.task_queue_name)
    assert after == before + 1


async def test_orchestrator_route_task_drift_keywords(global_mq):
    """Keywords 'drift' and 'monitor' route to model_monitor."""
    orch = OrchestratorAgent()
    await orch.initialize()

    task = _task(
        AgentType.ORCHESTRATOR,
        "route_task",
        {"task_description": "detect model drift in pitcher prediction model"},
        task_id="route-002",
    )
    result = await orch.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.result_data["routed_to"] == AgentType.MODEL_MONITOR.value


async def test_orchestrator_route_task_feature_keywords(global_mq):
    """Keyword 'feature' routes to feature_engineer."""
    orch = OrchestratorAgent()
    await orch.initialize()

    task = _task(
        AgentType.ORCHESTRATOR,
        "route_task",
        {"task_description": "run feature engineering on batting data"},
        task_id="route-003",
    )
    result = await orch.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.result_data["routed_to"] == AgentType.FEATURE_ENGINEER.value


async def test_orchestrator_route_task_explanation_keywords(global_mq):
    """Keyword 'explain' routes to explainer."""
    orch = OrchestratorAgent()
    await orch.initialize()

    task = _task(
        AgentType.ORCHESTRATOR,
        "route_task",
        {"task_description": "explain this SHAP prediction for a batter"},
        task_id="route-004",
    )
    result = await orch.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.result_data["routed_to"] == AgentType.EXPLAINER.value


# ── DataQualityAgent result propagation ────────────────────────────────────


async def test_data_quality_execute_task_publishes_result(global_mq, clean_parquet):
    """DQ agent should publish a COMPLETED result after _execute_task."""
    agent = DataQualityAgent()
    await agent.initialize()

    task = _task(
        AgentType.DATA_QUALITY,
        "check_data_quality",
        {"data_source": str(clean_parquet)},
        task_id="dq-exec-001",
    )
    await agent._execute_task(task)

    result = await global_mq.get_result("dq-exec-001")
    assert result is not None
    assert result.status == TaskStatus.COMPLETED
    assert result.agent_id == AgentType.DATA_QUALITY


async def test_data_quality_result_propagates_metrics(global_mq, clean_parquet):
    """DQ result should include quality_metrics in result_data and metrics dict."""
    agent = DataQualityAgent()
    await agent.initialize()

    task = _task(
        AgentType.DATA_QUALITY,
        "check_data_quality",
        {"data_source": str(clean_parquet)},
        task_id="dq-metrics-001",
    )
    result = await agent.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert "quality_metrics" in result.result_data
    assert "repair_log" in result.result_data
    assert result.metrics["completeness_score"] >= 0.0


# ── ModelMonitorAgent result propagation ───────────────────────────────────


async def test_model_monitor_execute_task_publishes_result(global_mq):
    """MM trigger_retraining should publish a COMPLETED result and queue a task."""
    agent = ModelMonitorAgent()
    await agent.initialize()

    task = _task(
        AgentType.MODEL_MONITOR,
        "trigger_retraining",
        {"model_name": "batting_avg_model", "reason": "integration test"},
        task_id="mm-exec-001",
    )
    await agent._execute_task(task)

    result = await global_mq.get_result("mm-exec-001")
    assert result is not None
    assert result.status == TaskStatus.COMPLETED
    assert result.result_data["model_name"] == "batting_avg_model"


async def test_model_monitor_retraining_queues_orchestrator_task(global_mq):
    """trigger_retraining should push a retrain_model task for the orchestrator."""
    agent = ModelMonitorAgent()
    await agent.initialize()

    before = await global_mq.get_queue_depth(settings.task_queue_name)
    task = _task(
        AgentType.MODEL_MONITOR,
        "trigger_retraining",
        {"model_name": "era_model", "reason": "drift"},
        task_id="mm-queue-001",
    )
    result = await agent.handle_task(task)

    after = await global_mq.get_queue_depth(settings.task_queue_name)
    assert result.status == TaskStatus.COMPLETED
    assert result.result_data["retraining_task_id"] is not None
    assert after == before + 1


# ── Heartbeat round-trip ───────────────────────────────────────────────────


async def test_heartbeat_round_trip(global_mq):
    """update_agent_heartbeat → get_agent_heartbeat should return a recent timestamp."""
    from datetime import datetime, timedelta

    await global_mq.update_agent_heartbeat(AgentType.DATA_QUALITY.value)
    heartbeat = await global_mq.get_agent_heartbeat(AgentType.DATA_QUALITY.value)

    assert heartbeat is not None
    assert isinstance(heartbeat, datetime)
    assert abs((datetime.now() - heartbeat).total_seconds()) < 5


async def test_missing_heartbeat_returns_none(global_mq):
    """An agent that never sent a heartbeat should return None."""
    heartbeat = await global_mq.get_agent_heartbeat("ghost_agent")
    assert heartbeat is None


# ── Orchestrator system_health ─────────────────────────────────────────────


async def test_orchestrator_system_health_no_heartbeats(global_mq):
    """System health with no agent heartbeats should report unhealthy agents."""
    orch = OrchestratorAgent()
    orch.start_time = __import__("datetime").datetime.now()

    task = _task(
        AgentType.ORCHESTRATOR,
        "system_health",
        {"stale_threshold_seconds": 60},
        task_id="health-001",
    )
    result = await orch.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    status = result.result_data
    assert "all_agents_healthy" in status
    # No heartbeats registered → system is not fully healthy
    assert status["all_agents_healthy"] is False


async def test_orchestrator_system_health_with_fresh_heartbeats(global_mq):
    """Fresh heartbeats for all agents should yield a healthy system."""
    from datetime import datetime

    # Register heartbeats for all non-orchestrator agents
    for agent_type in AgentType:
        if agent_type != AgentType.ORCHESTRATOR:
            await global_mq.update_agent_heartbeat(agent_type.value)

    orch = OrchestratorAgent()
    orch.start_time = datetime.now()

    task = _task(
        AgentType.ORCHESTRATOR,
        "system_health",
        {"stale_threshold_seconds": 60},
        task_id="health-002",
    )
    result = await orch.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.result_data["all_agents_healthy"] is True


# ── Multi-agent queue sharing ──────────────────────────────────────────────


async def test_multiple_agents_share_single_queue(global_mq):
    """DQ and MM agents should both push to the same task queue."""
    dq = DataQualityAgent()
    mm = ModelMonitorAgent()
    await dq.initialize()
    await mm.initialize()

    before = await global_mq.get_queue_depth(settings.task_queue_name)

    # MM trigger_retraining pushes one task to the orchestrator
    mm_task = _task(
        AgentType.MODEL_MONITOR,
        "trigger_retraining",
        {"model_name": "shared_queue_model"},
        task_id="shared-mm-001",
    )
    await mm.handle_task(mm_task)

    # Orchestrator route_task pushes one task to a specialist
    orch = OrchestratorAgent()
    await orch.initialize()
    orch_task = _task(
        AgentType.ORCHESTRATOR,
        "route_task",
        {"task_description": "check anomalies"},
        task_id="shared-orch-001",
    )
    await orch.handle_task(orch_task)

    after = await global_mq.get_queue_depth(settings.task_queue_name)
    assert after == before + 2
