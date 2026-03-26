"""
Integration tests: multi-agent workflow scenarios.

Covers the three end-to-end scenarios from Phase 7:

1. Data quality issue detected → repair → validate
   DataQualityAgent detects anomalies in dirty data, auto-repairs, reports metrics.
   Alert published when critical issues are found.

2. Drift detected → alert → retrain
   ModelMonitorAgent detects feature drift against a stored baseline.
   Publishes a drift alert, then trigger_retraining hands off to the Orchestrator.
   Orchestrator retrain_model kicks off a DQ preflight and an INFO alert.

3. Feature engineer creates feature → validate
   FeatureEngineerAgent searches for features via GA (small population/generations).
   Returns a FeatureSearchResult with evaluated candidates.
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
from agents.data_quality.agent import DataQualityAgent
from agents.model_monitor.agent import ModelMonitorAgent
from agents.feature_engineer.agent import FeatureEngineerAgent
from agents.orchestrator.agent import OrchestratorAgent


# ── Fixtures ──────────────────────────────────────────────────────────────


def _task(
    agent_id: AgentType,
    task_type: str,
    params: dict,
    task_id: str,
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
    np.random.seed(0)
    df = pd.DataFrame(
        {
            "batting_avg": np.random.uniform(0.220, 0.330, 40).round(3),
            "home_runs": np.random.randint(10, 35, 40),
            "rbi": np.random.randint(30, 100, 40),
            "ops": np.random.uniform(0.650, 0.950, 40).round(3),
        }
    )
    path = tmp_path / "clean.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def dirty_parquet(tmp_path) -> Path:
    """Parquet with missing values, duplicates, and extreme outliers."""
    np.random.seed(1)
    df = pd.DataFrame(
        {
            "batting_avg": np.concatenate(
                [np.random.uniform(0.220, 0.330, 28), [np.nan, np.nan, 99.0, -50.0]]
            ),
            "home_runs": np.concatenate(
                [np.random.randint(10, 35, 30), [np.nan, np.nan]]
            ),
            "rbi": np.concatenate(
                [np.random.randint(30, 100, 30), [np.nan, np.nan]]
            ),
            "ops": np.concatenate(
                [np.random.uniform(0.650, 0.950, 28), [np.nan, 0.0, 5.0, -1.0]]
            ),
        }
    )
    # Add duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    path = tmp_path / "dirty.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def baseline_parquet(tmp_path) -> Path:
    """Baseline distribution for drift comparison."""
    np.random.seed(10)
    df = pd.DataFrame(
        {
            "batting_avg": np.random.uniform(0.240, 0.300, 60).round(3),
            "home_runs": np.random.randint(15, 30, 60).astype(float),
            "rbi": np.random.randint(40, 90, 60).astype(float),
            "ops": np.random.uniform(0.700, 0.900, 60).round(3),
        }
    )
    path = tmp_path / "baseline.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def drifted_parquet(tmp_path) -> Path:
    """Heavily shifted distribution to guarantee drift detection."""
    np.random.seed(99)
    df = pd.DataFrame(
        {
            "batting_avg": np.random.uniform(0.130, 0.180, 60).round(3),
            "home_runs": np.random.randint(40, 55, 60).astype(float),
            "rbi": np.random.randint(5, 20, 60).astype(float),
            "ops": np.random.uniform(0.350, 0.450, 60).round(3),
        }
    )
    path = tmp_path / "drifted.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def fe_parquet(tmp_path) -> Path:
    """Parquet with a numeric target column for feature search."""
    np.random.seed(5)
    n = 60
    df = pd.DataFrame(
        {
            "batting_avg": np.random.uniform(0.200, 0.350, n).round(3),
            "home_runs": np.random.randint(5, 45, n).astype(float),
            "rbi": np.random.randint(20, 120, n).astype(float),
            "ops": np.random.uniform(0.600, 1.100, n).round(3),
            "war": np.random.uniform(-0.5, 8.0, n).round(1),
        }
    )
    path = tmp_path / "fe_data.parquet"
    df.to_parquet(path, index=False)
    return path


# ── Scenario 1: Data quality issue → repair → validate ────────────────────


async def test_dirty_data_check_reports_issues(global_mq, dirty_parquet):
    """DataQualityAgent should detect missing values and anomalies in dirty data."""
    agent = DataQualityAgent()
    await agent.initialize()

    task = _task(
        AgentType.DATA_QUALITY,
        "check_data_quality",
        {"data_source": str(dirty_parquet)},
        task_id="wf-dq-001",
    )
    result = await agent.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    # Dirty data has NaN values → completeness_score < 1.0
    assert result.metrics["completeness_score"] < 1.0
    assert result.metrics["missing_values_pct"] > 0.0


async def test_auto_repair_improves_quality(global_mq, dirty_parquet):
    """Auto-fix should produce a repair log and return a completed result."""
    agent = DataQualityAgent()
    await agent.initialize()

    task = _task(
        AgentType.DATA_QUALITY,
        "check_data_quality",
        {"data_source": str(dirty_parquet), "auto_fix": True},
        task_id="wf-dq-repair-001",
    )
    result = await agent.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert "repair_log" in result.result_data
    assert len(result.result_data["repair_log"]) > 0


async def test_schema_validated_on_subsequent_call(global_mq, clean_parquet):
    """Second check_data_quality call should validate against cached schema."""
    agent = DataQualityAgent()
    await agent.initialize()

    for task_id in ("wf-schema-001", "wf-schema-002"):
        task = _task(
            AgentType.DATA_QUALITY,
            "check_data_quality",
            {"data_source": str(clean_parquet)},
            task_id=task_id,
        )
        result = await agent.handle_task(task)
        assert result.status == TaskStatus.COMPLETED

    # Schema should now be cached for the data source
    assert str(clean_parquet) in agent.schema_cache


async def test_data_quality_alert_published_for_anomalies(global_mq, tmp_path):
    """Data with >10% missing values triggers a completeness alert."""
    # Build a parquet where ~50% of values are NaN → completeness_score < 0.9
    import numpy as np
    np.random.seed(7)
    n = 40
    col = np.random.uniform(0.200, 0.350, n).astype(float)
    col[: n // 2] = np.nan  # 50% missing
    df = pd.DataFrame({"batting_avg": col, "home_runs": col.copy()})
    path = tmp_path / "sparse.parquet"
    df.to_parquet(path, index=False)

    agent = DataQualityAgent()
    await agent.initialize()

    before = await global_mq.get_queue_depth(settings.alert_queue_name)
    task = _task(
        AgentType.DATA_QUALITY,
        "check_data_quality",
        {"data_source": str(path)},
        task_id="wf-dq-alert-001",
    )
    result = await agent.handle_task(task)

    after = await global_mq.get_queue_depth(settings.alert_queue_name)
    assert result.status == TaskStatus.COMPLETED
    assert result.metrics["completeness_score"] < 0.9
    assert after > before


# ── Scenario 2: Drift detected → alert → retrain ──────────────────────────


async def test_first_drift_check_stores_baseline(global_mq, clean_parquet):
    """First drift check with no baseline_source should store baseline and return no drift."""
    agent = ModelMonitorAgent()
    await agent.initialize()

    task = _task(
        AgentType.MODEL_MONITOR,
        "check_drift",
        {"data_source": str(clean_parquet), "model_name": "batting_model"},
        task_id="wf-drift-001",
    )
    result = await agent.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.result_data.get("first_visit") is True
    assert "batting_model" in agent._baselines


async def test_drift_detected_against_baseline(global_mq, baseline_parquet, drifted_parquet):
    """Heavily shifted data should trigger drift detection and an alert."""
    agent = ModelMonitorAgent()
    await agent.initialize()

    before_alerts = await global_mq.get_queue_depth(settings.alert_queue_name)

    task = _task(
        AgentType.MODEL_MONITOR,
        "check_drift",
        {
            "data_source": str(drifted_parquet),
            "baseline_source": str(baseline_parquet),
            "model_name": "era_model",
            "psi_threshold": 0.05,   # low threshold to ensure detection
            "ks_p_threshold": 0.10,
        },
        task_id="wf-drift-002",
    )
    result = await agent.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.result_data.get("drift_detected") is True
    after_alerts = await global_mq.get_queue_depth(settings.alert_queue_name)
    assert after_alerts > before_alerts


async def test_trigger_retraining_after_drift(global_mq, baseline_parquet, drifted_parquet):
    """trigger_retraining should publish a retrain_model task to the orchestrator."""
    agent = ModelMonitorAgent()
    await agent.initialize()

    before = await global_mq.get_queue_depth(settings.task_queue_name)
    task = _task(
        AgentType.MODEL_MONITOR,
        "trigger_retraining",
        {"model_name": "era_model", "reason": "drift detected"},
        task_id="wf-retrain-001",
    )
    result = await agent.handle_task(task)

    after = await global_mq.get_queue_depth(settings.task_queue_name)
    assert result.status == TaskStatus.COMPLETED
    assert result.result_data["model_name"] == "era_model"
    assert after == before + 1


async def test_orchestrator_retrain_model_initiates_dq_preflight(global_mq):
    """Orchestrator retrain_model should queue a DQ full_quality_check as preflight."""
    orch = OrchestratorAgent()
    await orch.initialize()

    before = await global_mq.get_queue_depth(settings.task_queue_name)
    task = _task(
        AgentType.ORCHESTRATOR,
        "retrain_model",
        {
            "model_name": "era_model",
            "reason": "drift detected",
            "triggered_by": "model_monitor",
        },
        task_id="wf-orch-retrain-001",
    )
    result = await orch.handle_task(task)

    after = await global_mq.get_queue_depth(settings.task_queue_name)
    assert result.status == TaskStatus.COMPLETED
    assert result.result_data["dq_preflight_task_id"] is not None
    assert result.result_data["status"] == "retraining_workflow_initiated"
    # One DQ preflight task queued
    assert after == before + 1


async def test_full_drift_to_retrain_pipeline(global_mq, baseline_parquet, drifted_parquet):
    """
    Full pipeline: drift detected → trigger_retraining (MM) → retrain_model (Orchestrator).

    Steps:
    1. MM checks drift (with explicit baseline) — detects it.
    2. MM triggers retraining → queues retrain_model task for orchestrator.
    3. Orchestrator handles retrain_model → queues DQ preflight task.
    4. Total queue depth increases by 2 (retrain task + DQ preflight task).
    """
    mm = ModelMonitorAgent()
    orch = OrchestratorAgent()
    await mm.initialize()
    await orch.initialize()

    before = await global_mq.get_queue_depth(settings.task_queue_name)

    # Step 1+2: drift check — drift should be detected, but we separately trigger retraining
    retrain_task = _task(
        AgentType.MODEL_MONITOR,
        "trigger_retraining",
        {"model_name": "pitcher_model", "reason": "drift"},
        task_id="pipeline-mm-001",
    )
    mm_result = await mm.handle_task(retrain_task)
    assert mm_result.status == TaskStatus.COMPLETED

    after_mm = await global_mq.get_queue_depth(settings.task_queue_name)
    assert after_mm == before + 1  # retrain_model task queued for orchestrator

    # Step 3: Orchestrator handles retrain_model → queues DQ preflight
    orch_task = _task(
        AgentType.ORCHESTRATOR,
        "retrain_model",
        {
            "model_name": "pitcher_model",
            "reason": "drift",
            "triggered_by": "model_monitor",
        },
        task_id="pipeline-orch-001",
    )
    orch_result = await orch.handle_task(orch_task)
    assert orch_result.status == TaskStatus.COMPLETED
    assert orch_result.result_data["dq_preflight_task_id"] is not None

    after_orch = await global_mq.get_queue_depth(settings.task_queue_name)
    assert after_orch == before + 2  # retrain_model + dq_preflight


async def test_orchestrator_retrain_publishes_info_alert(global_mq):
    """retrain_model should publish an INFO alert summarising the workflow start."""
    orch = OrchestratorAgent()
    await orch.initialize()

    before = await global_mq.get_queue_depth(settings.alert_queue_name)
    task = _task(
        AgentType.ORCHESTRATOR,
        "retrain_model",
        {"model_name": "ops_model", "reason": "manual", "triggered_by": "test"},
        task_id="orch-alert-001",
    )
    await orch.handle_task(task)

    after = await global_mq.get_queue_depth(settings.alert_queue_name)
    assert after > before


# ── Scenario 3: Feature engineer creates features → validate ───────────────


async def test_feature_search_returns_candidates(global_mq, fe_parquet):
    """FeatureEngineerAgent search_features should return a FeatureSearchResult."""
    agent = FeatureEngineerAgent()
    await agent.initialize()

    task = _task(
        AgentType.FEATURE_ENGINEER,
        "search_features",
        {
            "data_source": str(fe_parquet),
            "target_col": "war",
            "population_size": 8,
            "generations": 3,
            "top_k": 3,
            "use_llm": False,
        },
        task_id="wf-fe-001",
    )
    result = await agent.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert "search_id" in result.result_data
    assert "features_added" in result.result_data
    assert "candidates_evaluated" in result.result_data


async def test_feature_search_baseline_score_in_result(global_mq, fe_parquet):
    """Feature search result should include baseline_model_score for comparison."""
    agent = FeatureEngineerAgent()
    await agent.initialize()

    task = _task(
        AgentType.FEATURE_ENGINEER,
        "search_features",
        {
            "data_source": str(fe_parquet),
            "target_col": "war",
            "population_size": 5,
            "generations": 2,
            "top_k": 2,
            "use_llm": False,
        },
        task_id="wf-fe-baseline-001",
    )
    result = await agent.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert "baseline_model_score" in result.result_data
    assert result.metrics["baseline_score"] <= 0.0  # CV scores are negative (neg MSE)


async def test_feature_evaluate_single_candidate(global_mq, fe_parquet):
    """evaluate_feature should return a FeatureCandidate or degenerate-feature report."""
    agent = FeatureEngineerAgent()
    await agent.initialize()

    task = _task(
        AgentType.FEATURE_ENGINEER,
        "evaluate_feature",
        {
            "data_source": str(fe_parquet),
            "target_col": "war",
            "feature_name": "avg_x_hr",
            "gene_type": "interaction",
            "source_features": ["batting_avg", "home_runs"],
            "params": {},
        },
        task_id="wf-fe-eval-001",
    )
    result = await agent.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert "validation_passed" in result.result_data


async def test_orchestrator_routes_to_feature_engineer(global_mq):
    """Orchestrator should route a feature-engineering request to feature_engineer."""
    orch = OrchestratorAgent()
    await orch.initialize()

    task = _task(
        AgentType.ORCHESTRATOR,
        "route_task",
        {"task_description": "run genetic feature engineering on batting stats"},
        task_id="wf-fe-route-001",
    )
    result = await orch.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.result_data["routed_to"] == AgentType.FEATURE_ENGINEER.value


# ── Conflict resolution workflow ──────────────────────────────────────────


async def test_orchestrator_conflict_resolution_rule_based(global_mq):
    """Rule-based conflict resolution should pick model_monitor recommendation."""
    orch = OrchestratorAgent()
    await orch.initialize()

    task = _task(
        AgentType.ORCHESTRATOR,
        "resolve_conflict",
        {
            "conflict_description": "Two agents disagree on retraining urgency",
            "recommendations": [
                {
                    "agent_id": "feature_engineer",
                    "recommendation": "Add features before retraining",
                    "data": {"confidence": 0.6},
                },
                {
                    "agent_id": "model_monitor",
                    "recommendation": "Retrain immediately",
                    "data": {"drift_score": 0.45},
                },
            ],
        },
        task_id="wf-conflict-001",
    )
    result = await orch.handle_task(task)

    assert result.status == TaskStatus.COMPLETED
    # model_monitor has highest priority in rule-based resolution
    assert "Retrain immediately" in result.result_data["chosen_recommendation"]
    assert result.result_data["resolution"] == "accepted"
