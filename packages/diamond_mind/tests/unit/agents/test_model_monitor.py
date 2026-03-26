"""
Unit tests for the ModelMonitorAgent and its helper modules.

Covers:
- drift_detection: calculate_psi, run_ks_test, detect_feature_drift
- ab_testing: VariantStats, ABTest
- ModelMonitorAgent: _load_data, _compute_performance_metrics, _check_degradation
- handle_task: check_drift, evaluate_performance, trigger_retraining,
               run_ab_test, record_ab_outcome, register_model_version, rollback_model
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the shared package is on the path
_dm_src = str(Path(__file__).resolve().parents[3] / "src" / "diamond_mind")
if _dm_src not in sys.path:
    sys.path.insert(0, _dm_src)

from shared.schemas import AgentType, TaskStatus, AgentTask, TaskPriority
from agents.model_monitor.agent import ModelMonitorAgent
from agents.model_monitor.drift_detection import calculate_psi, run_ks_test, detect_feature_drift
from agents.model_monitor.ab_testing import ABTest, VariantStats


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent():
    return ModelMonitorAgent()


@pytest.fixture
def baseline_df():
    np.random.seed(0)
    return pd.DataFrame(
        {
            "batting_avg": np.random.uniform(0.220, 0.330, 100).round(3),
            "home_runs": np.random.randint(10, 35, 100).astype(float),
            "ops": np.random.uniform(0.650, 0.950, 100).round(3),
        }
    )


@pytest.fixture
def drifted_df():
    """Distribution shifted by +0.1 on all features to force PSI/KS detection."""
    np.random.seed(1)
    return pd.DataFrame(
        {
            "batting_avg": np.random.uniform(0.320, 0.430, 100).round(3),
            "home_runs": np.random.randint(30, 55, 100).astype(float),
            "ops": np.random.uniform(0.750, 1.050, 100).round(3),
        }
    )


@pytest.fixture
def parquet_path(tmp_path, baseline_df):
    p = tmp_path / "baseline.parquet"
    baseline_df.to_parquet(p, index=False)
    return p


@pytest.fixture
def drifted_parquet_path(tmp_path, drifted_df):
    p = tmp_path / "drifted.parquet"
    drifted_df.to_parquet(p, index=False)
    return p


def make_task(task_type: str, parameters: dict | None = None) -> AgentTask:
    return AgentTask(
        task_id="test_mm_001",
        agent_id=AgentType.MODEL_MONITOR,
        task_type=task_type,
        priority=TaskPriority.HIGH,
        parameters=parameters or {},
    )


# ---------------------------------------------------------------------------
# calculate_psi
# ---------------------------------------------------------------------------


class TestCalculatePSI:
    def test_identical_distributions_near_zero(self):
        data = np.random.default_rng(0).uniform(0, 1, 200)
        psi = calculate_psi(data, data.copy())
        assert psi < 0.05

    def test_shifted_distribution_high_psi(self):
        expected = np.random.default_rng(0).normal(0, 1, 500)
        actual = np.random.default_rng(1).normal(3, 1, 500)
        psi = calculate_psi(expected, actual)
        assert psi > 0.2

    def test_returns_non_negative(self):
        rng = np.random.default_rng(42)
        psi = calculate_psi(rng.uniform(0, 1, 100), rng.uniform(0, 1, 100))
        assert psi >= 0.0

    def test_custom_bins(self):
        data = np.arange(100, dtype=float)
        psi = calculate_psi(data, data, bins=5)
        assert psi < 0.01


# ---------------------------------------------------------------------------
# run_ks_test
# ---------------------------------------------------------------------------


class TestRunKSTest:
    def test_same_distribution_high_p(self):
        data = np.random.default_rng(0).normal(0, 1, 300)
        ks_stat, p_value = run_ks_test(data, data.copy())
        assert p_value > 0.05

    def test_different_distributions_low_p(self):
        rng = np.random.default_rng(5)
        expected = rng.normal(0, 1, 200)
        actual = rng.normal(5, 1, 200)
        ks_stat, p_value = run_ks_test(expected, actual)
        assert p_value < 0.05
        assert ks_stat > 0.0

    def test_returns_two_floats(self):
        data = np.ones(50)
        result = run_ks_test(data, data + 1)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# detect_feature_drift
# ---------------------------------------------------------------------------


class TestDetectFeatureDrift:
    def test_no_drift_same_data(self, baseline_df):
        result = detect_feature_drift(baseline_df, baseline_df.copy())
        assert result.drift_detected is False
        assert result.drift_score == 0.0

    def test_drift_detected_on_shifted_data(self, baseline_df, drifted_df):
        result = detect_feature_drift(baseline_df, drifted_df)
        assert result.drift_detected is True
        assert len(result.affected_features) > 0
        assert result.drift_score > 0.0

    def test_psi_and_ks_scores_populated(self, baseline_df, drifted_df):
        result = detect_feature_drift(baseline_df, drifted_df)
        for col in baseline_df.select_dtypes(include=[np.number]).columns:
            assert col in result.psi_scores
            assert col in result.ks_statistics

    def test_recommendation_non_empty(self, baseline_df, drifted_df):
        result = detect_feature_drift(baseline_df, drifted_df)
        assert isinstance(result.recommendation, str)
        assert len(result.recommendation) > 0

    def test_handles_non_overlapping_columns(self, baseline_df):
        other = pd.DataFrame({"unrelated": np.arange(100, dtype=float)})
        result = detect_feature_drift(baseline_df, other)
        # No common numeric columns → no drift possible
        assert result.drift_detected is False


# ---------------------------------------------------------------------------
# VariantStats
# ---------------------------------------------------------------------------


class TestVariantStats:
    def test_initial_state(self):
        v = VariantStats(name="champion")
        assert v.n == 0
        assert v.mean_absolute_error is None
        assert v.accuracy is None

    def test_record_updates_counts(self):
        v = VariantStats(name="challenger")
        v.record(0.280, 0.300)
        v.record(0.260, 0.270)
        assert v.n == 2

    def test_mean_absolute_error(self):
        v = VariantStats(name="test")
        v.record(1.0, 2.0)  # error = 1.0
        v.record(3.0, 3.0)  # error = 0.0
        assert v.mean_absolute_error == pytest.approx(0.5)

    def test_accuracy_within_half(self):
        v = VariantStats(name="test")
        v.record(0.3, 0.3)    # error 0.0 → within 0.5
        v.record(0.3, 10.0)   # error 9.7 → outside 0.5
        assert v.accuracy == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ABTest
# ---------------------------------------------------------------------------


class TestABTest:
    def test_assign_returns_valid_variant(self):
        test = ABTest("t1", min_samples=5)
        variants = {test.assign() for _ in range(50)}
        assert variants <= {"champion", "challenger"}

    def test_deterministic_assignment_by_entity(self):
        test = ABTest("t2", min_samples=5, challenger_traffic_pct=0.2)
        v1 = test.assign(entity_id="player_1")
        v2 = test.assign(entity_id="player_1")
        assert v1 == v2

    def test_has_sufficient_data_false_initially(self):
        test = ABTest("t3", min_samples=10)
        assert test.has_sufficient_data() is False

    def test_has_sufficient_data_after_filling(self):
        test = ABTest("t4", min_samples=3)
        for _ in range(3):
            test.record("champion", 0.3, 0.3)
            test.record("challenger", 0.3, 0.3)
        assert test.has_sufficient_data() is True

    def test_test_significance_raises_without_data(self):
        test = ABTest("t5", min_samples=50)
        with pytest.raises(RuntimeError, match="Insufficient data"):
            test.test_significance()

    def test_challenger_wins_when_lower_error(self):
        test = ABTest("t6", min_samples=5, significance=0.05)
        rng = np.random.default_rng(0)
        # Champion: large errors
        for _ in range(40):
            test.record("champion", rng.normal(0.5, 0.3), 0.3)
        # Challenger: near-perfect predictions
        for _ in range(40):
            test.record("challenger", 0.3, 0.3)
        result = test.test_significance()
        assert result["challenger_wins"] is True

    def test_summary_structure(self):
        test = ABTest("t7", min_samples=10)
        s = test.summary()
        assert "champion" in s
        assert "challenger" in s
        assert "has_sufficient_data" in s

    def test_should_promote_false_without_data(self):
        test = ABTest("t8", min_samples=100)
        assert test.should_promote_challenger() is False


# ---------------------------------------------------------------------------
# ModelMonitorAgent._load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_loads_parquet(self, agent, parquet_path, baseline_df):
        df = agent._load_data(str(parquet_path))
        assert len(df) == len(baseline_df)

    def test_loads_csv(self, agent, tmp_path, baseline_df):
        csv_path = tmp_path / "data.csv"
        baseline_df.to_csv(csv_path, index=False)
        df = agent._load_data(str(csv_path))
        assert len(df) == len(baseline_df)

    def test_raises_on_missing_file(self, agent):
        with pytest.raises(FileNotFoundError):
            agent._load_data("/nonexistent/file.parquet")

    def test_raises_on_unsupported_format(self, agent, tmp_path):
        p = tmp_path / "data.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file format"):
            agent._load_data(str(p))


# ---------------------------------------------------------------------------
# ModelMonitorAgent._compute_performance_metrics
# ---------------------------------------------------------------------------


class TestComputePerformanceMetrics:
    def test_perfect_predictions(self, agent):
        preds = [0.3] * 20
        actuals = [0.3] * 20
        m = agent._compute_performance_metrics("model_a", "v1", preds, actuals, [10.0] * 20)
        assert m.accuracy == pytest.approx(1.0)
        assert m.prediction_count == 20
        assert m.avg_prediction_time_ms == pytest.approx(10.0)

    def test_all_wrong_predictions(self, agent):
        preds = [0.0] * 20
        actuals = [1.0] * 20
        m = agent._compute_performance_metrics("model_b", "v1", preds, actuals, [])
        assert m.accuracy == pytest.approx(0.0)

    def test_empty_predictions(self, agent):
        m = agent._compute_performance_metrics("model_c", "v1", [], [], [])
        assert m.accuracy == 0.0
        assert m.prediction_count == 0

    def test_model_name_and_version_stored(self, agent):
        m = agent._compute_performance_metrics("my_model", "v3", [0.3], [0.3], [])
        assert m.model_name == "my_model"
        assert m.model_version == "v3"


# ---------------------------------------------------------------------------
# ModelMonitorAgent._check_degradation
# ---------------------------------------------------------------------------


class TestCheckDegradation:
    def test_no_history_no_degradation(self, agent):
        m = agent._compute_performance_metrics("m", "v1", [0.3] * 10, [0.3] * 10, [])
        agent._performance_history["m"] = __import__("collections").deque([m])
        degraded, delta = agent._check_degradation("m", m)
        assert degraded is False

    def test_detects_degradation(self, agent):
        from collections import deque

        m_good = agent._compute_performance_metrics("m2", "v1", [0.3] * 10, [0.3] * 10, [])
        m_bad = agent._compute_performance_metrics("m2", "v2", [0.0] * 10, [1.0] * 10, [])
        agent._performance_history["m2"] = deque([m_good, m_bad])
        degraded, delta = agent._check_degradation("m2", m_bad)
        assert degraded is True
        assert delta < 0

    def test_improvement_not_flagged(self, agent):
        from collections import deque

        m_bad = agent._compute_performance_metrics("m3", "v1", [0.0] * 10, [1.0] * 10, [])
        m_good = agent._compute_performance_metrics("m3", "v2", [0.3] * 10, [0.3] * 10, [])
        agent._performance_history["m3"] = deque([m_bad, m_good])
        degraded, delta = agent._check_degradation("m3", m_good)
        assert degraded is False
        assert delta > 0


# ---------------------------------------------------------------------------
# handle_task — dispatch and result shape
# ---------------------------------------------------------------------------


class TestHandleTask:
    @pytest.mark.asyncio
    async def test_check_drift_first_visit(self, agent, global_mq, parquet_path):
        task = make_task("check_drift", {"data_source": str(parquet_path), "model_name": "mdl"})
        result = await agent._check_drift(task)
        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["first_visit"] is True

    @pytest.mark.asyncio
    async def test_check_drift_detects_drift(self, agent, global_mq, parquet_path, drifted_parquet_path):
        # Seed baseline
        await agent._check_drift(
            make_task("check_drift", {"data_source": str(parquet_path), "model_name": "mdl2"})
        )
        # Check drifted data
        result = await agent._check_drift(
            make_task(
                "check_drift",
                {"data_source": str(drifted_parquet_path), "model_name": "mdl2"},
            )
        )
        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["drift_detected"] is True

    @pytest.mark.asyncio
    async def test_evaluate_performance_task(self, agent, global_mq):
        task = make_task(
            "evaluate_performance",
            {
                "model_name": "batting_model",
                "model_version": "v1",
                "predictions": [0.3] * 20,
                "actuals": [0.3] * 20,
                "prediction_times_ms": [5.0] * 20,
            },
        )
        result = await agent._evaluate_performance(task)
        assert result.status == TaskStatus.COMPLETED
        assert "metrics" in result.result_data
        assert result.metrics["accuracy"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_run_ab_test_task(self, agent, global_mq):
        task = make_task("run_ab_test", {"test_id": "exp_1", "entity_id": "player_99"})
        result = await agent._run_ab_test(task)
        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["assigned_variant"] in ("champion", "challenger")
        assert "exp_1" in agent._active_ab_tests

    @pytest.mark.asyncio
    async def test_record_ab_outcome_task(self, agent, global_mq):
        # Create a test first
        agent._active_ab_tests["exp_2"] = ABTest("exp_2", min_samples=5)
        task = make_task(
            "record_ab_outcome",
            {"test_id": "exp_2", "variant": "champion", "prediction": 0.3, "actual": 0.3},
        )
        result = await agent._record_ab_outcome(task)
        assert result.status == TaskStatus.COMPLETED
        assert agent._active_ab_tests["exp_2"].champion.n == 1

    @pytest.mark.asyncio
    async def test_record_ab_outcome_unknown_test_raises(self, agent):
        task = make_task(
            "record_ab_outcome",
            {"test_id": "no_such_test", "variant": "champion", "prediction": 0.3, "actual": 0.3},
        )
        with pytest.raises(ValueError, match="No active A/B test"):
            await agent._record_ab_outcome(task)

    @pytest.mark.asyncio
    async def test_register_and_rollback_model(self, agent, global_mq):
        reg_task = make_task(
            "register_model_version",
            {"model_name": "bat_model", "version_id": "v1", "model_path": "/models/v1.pkl"},
        )
        await agent._register_model_version(reg_task)

        reg_task2 = make_task(
            "register_model_version",
            {"model_name": "bat_model", "version_id": "v2", "model_path": "/models/v2.pkl"},
        )
        reg_result = await agent._register_model_version(reg_task2)
        assert reg_result.result_data["total_versions"] == 2

        rollback_task = make_task("rollback_model", {"model_name": "bat_model"})
        rollback_result = await agent._rollback_model(rollback_task)
        assert rollback_result.status == TaskStatus.COMPLETED
        assert rollback_result.result_data["rolled_back_to"] == "v1"

    @pytest.mark.asyncio
    async def test_rollback_unknown_model_raises(self, agent):
        task = make_task("rollback_model", {"model_name": "ghost_model"})
        with pytest.raises(ValueError, match="No registered versions"):
            await agent._rollback_model(task)

    @pytest.mark.asyncio
    async def test_unknown_task_type_raises(self, agent):
        task = make_task("unknown_task")
        with pytest.raises(ValueError, match="Unknown task type"):
            await agent.handle_task(task)
