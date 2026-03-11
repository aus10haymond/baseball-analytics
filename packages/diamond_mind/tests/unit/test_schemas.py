"""Tests for shared/schemas.py – Pydantic model validation and serialization."""

import pytest
from pydantic import ValidationError

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
    DriftDetectionResult,
    ModelPerformanceMetrics,
    FeatureCandidate,
    FeatureSearchResult,
    PredictionExplanation,
    AgentHealthStatus,
    SystemStatus,
)


# ── Enums ──────────────────────────────────────────────────────────────────


class TestEnums:
    def test_agent_type_values(self):
        assert AgentType.ORCHESTRATOR == "orchestrator"
        assert AgentType.DATA_QUALITY == "data_quality"
        assert AgentType.MODEL_MONITOR == "model_monitor"
        assert AgentType.FEATURE_ENGINEER == "feature_engineer"
        assert AgentType.EXPLAINER == "explainer"
        assert len(AgentType) == 5

    def test_task_status_values(self):
        expected = {"pending", "running", "completed", "failed", "cancelled"}
        assert {s.value for s in TaskStatus} == expected

    def test_task_priority_ordering(self):
        values = [p.value for p in TaskPriority]
        assert "low" in values and "critical" in values

    def test_alert_severity_values(self):
        expected = {"info", "warning", "error", "critical"}
        assert {s.value for s in AlertSeverity} == expected

    def test_confidence_level_count(self):
        assert len(ConfidenceLevel) == 5


# ── AgentTask ──────────────────────────────────────────────────────────────


class TestAgentTask:
    def test_create_with_defaults(self):
        task = AgentTask(
            task_id="t1",
            agent_id=AgentType.DATA_QUALITY,
            task_type="check_anomalies",
        )
        assert task.priority == TaskPriority.MEDIUM
        assert task.parameters == {}
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert task.timeout_seconds is None
        assert task.created_at is not None

    def test_create_with_all_fields(self):
        task = AgentTask(
            task_id="t2",
            agent_id=AgentType.MODEL_MONITOR,
            task_type="check_drift",
            priority=TaskPriority.CRITICAL,
            parameters={"model": "xgboost_v1"},
            timeout_seconds=120,
            retry_count=1,
            max_retries=5,
        )
        assert task.priority == TaskPriority.CRITICAL
        assert task.parameters["model"] == "xgboost_v1"
        assert task.timeout_seconds == 120

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            AgentTask(task_type="check")  # missing task_id, agent_id

    def test_negative_retry_count_rejected(self):
        with pytest.raises(ValidationError):
            AgentTask(
                task_id="t",
                agent_id=AgentType.DATA_QUALITY,
                task_type="x",
                retry_count=-1,
            )

    def test_json_round_trip(self):
        task = AgentTask(
            task_id="rt1",
            agent_id=AgentType.EXPLAINER,
            task_type="explain_prediction",
            parameters={"prediction_id": "pred_001"},
        )
        restored = AgentTask.model_validate_json(task.model_dump_json())
        assert restored.task_id == task.task_id
        assert restored.agent_id == task.agent_id
        assert restored.parameters == task.parameters


# ── AgentResult ────────────────────────────────────────────────────────────


class TestAgentResult:
    def test_create_completed(self):
        result = AgentResult(
            task_id="t1",
            agent_id=AgentType.DATA_QUALITY,
            status=TaskStatus.COMPLETED,
            duration_seconds=5.5,
        )
        assert result.error_message is None
        assert result.result_data is None
        assert result.artifacts == []

    def test_create_failed(self):
        result = AgentResult(
            task_id="t2",
            agent_id=AgentType.MODEL_MONITOR,
            status=TaskStatus.FAILED,
            error_message="Connection timeout",
            duration_seconds=30.0,
        )
        assert result.error_message == "Connection timeout"

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            AgentResult(
                task_id="t",
                agent_id=AgentType.DATA_QUALITY,
                status=TaskStatus.COMPLETED,
                duration_seconds=-1.0,
            )

    def test_json_round_trip(self, sample_result):
        restored = AgentResult.model_validate_json(sample_result.model_dump_json())
        assert restored.task_id == sample_result.task_id
        assert restored.metrics == sample_result.metrics


# ── AgentAlert ─────────────────────────────────────────────────────────────


class TestAgentAlert:
    def test_create_with_defaults(self):
        alert = AgentAlert(
            alert_id="a1",
            agent_id=AgentType.ORCHESTRATOR,
            severity=AlertSeverity.INFO,
            message="All systems nominal",
        )
        assert alert.requires_action is False
        assert alert.suggested_actions == []
        assert alert.related_task_id is None

    def test_create_actionable_alert(self, sample_alert):
        assert sample_alert.requires_action is True
        assert len(sample_alert.suggested_actions) == 2


# ── Data Quality schemas ──────────────────────────────────────────────────


class TestDataQualitySchemas:
    def test_anomaly_report(self):
        report = DataAnomalyReport(
            anomaly_id="anom_001",
            anomaly_type="outlier",
            severity=AlertSeverity.WARNING,
            affected_columns=["batting_avg", "ops"],
            row_count=5,
            detection_method="isolation_forest",
            auto_fixable=True,
        )
        assert report.row_count == 5
        assert report.fix_applied is False

    def test_negative_row_count_rejected(self):
        with pytest.raises(ValidationError):
            DataAnomalyReport(
                anomaly_id="a",
                anomaly_type="outlier",
                severity=AlertSeverity.ERROR,
                row_count=-1,
                detection_method="z_score",
                auto_fixable=False,
            )

    def test_quality_metrics(self):
        m = DataQualityMetrics(
            total_records=1000,
            missing_values_pct=2.5,
            outlier_count=10,
            schema_valid=True,
            completeness_score=0.975,
            consistency_score=0.99,
        )
        assert m.total_records == 1000
        assert m.completeness_score == 0.975

    def test_quality_metrics_pct_out_of_range(self):
        with pytest.raises(ValidationError):
            DataQualityMetrics(
                total_records=100,
                missing_values_pct=150.0,  # > 100
                schema_valid=True,
                completeness_score=0.9,
                consistency_score=0.9,
            )

    def test_quality_metrics_score_out_of_range(self):
        with pytest.raises(ValidationError):
            DataQualityMetrics(
                total_records=100,
                missing_values_pct=1.0,
                schema_valid=True,
                completeness_score=1.5,  # > 1
                consistency_score=0.9,
            )


# ── Model Monitor schemas ─────────────────────────────────────────────────


class TestModelMonitorSchemas:
    def test_drift_result(self):
        result = DriftDetectionResult(
            drift_detected=True,
            drift_score=0.25,
            drift_type="data_drift",
            affected_features=["pitch_speed", "spin_rate"],
            psi_scores={"pitch_speed": 0.3, "spin_rate": 0.2},
            recommendation="Retrain model with recent data",
        )
        assert result.drift_detected is True
        assert len(result.affected_features) == 2

    def test_performance_metrics_valid(self):
        m = ModelPerformanceMetrics(
            model_name="xgboost_matchups",
            model_version="1.0.0",
            accuracy=0.82,
            auc=0.85,
            precision=0.80,
            recall=0.78,
            f1_score=0.79,
            prediction_count=5000,
            avg_prediction_time_ms=15.3,
        )
        assert m.auc == 0.85

    def test_performance_metrics_invalid_auc(self):
        with pytest.raises(ValidationError):
            ModelPerformanceMetrics(
                model_name="test",
                model_version="1.0",
                accuracy=0.8,
                auc=1.5,  # > 1
                prediction_count=100,
                avg_prediction_time_ms=10.0,
            )

    def test_performance_metrics_optional_none(self):
        m = ModelPerformanceMetrics(
            model_name="test",
            model_version="1.0",
            accuracy=0.8,
            prediction_count=100,
            avg_prediction_time_ms=10.0,
        )
        assert m.auc is None
        assert m.precision is None


# ── Feature Engineer schemas ───────────────────────────────────────────────


class TestFeatureEngineerSchemas:
    def test_feature_candidate(self):
        c = FeatureCandidate(
            feature_name="pitch_speed_sq",
            feature_definition="pitch_speed ** 2",
            feature_type="polynomial",
            source_features=["pitch_speed"],
            importance_score=0.15,
        )
        assert c.validation_passed is False

    def test_feature_search_result(self):
        r = FeatureSearchResult(
            search_id="search_001",
            generation=3,
            candidates_evaluated=50,
            best_model_score=0.85,
            baseline_model_score=0.80,
            improvement_pct=6.25,
            search_duration_seconds=120.5,
        )
        assert r.improvement_pct == 6.25


# ── Explainer schemas ─────────────────────────────────────────────────────


class TestExplainerSchemas:
    def test_prediction_explanation(self):
        e = PredictionExplanation(
            prediction_id="pred_001",
            player_name="Mike Trout",
            predicted_value=8.5,
            confidence=ConfidenceLevel.HIGH,
            top_features=[
                {"feature": "batting_avg", "shap_value": 2.1},
                {"feature": "ops", "shap_value": 1.8},
            ],
            narrative_explanation="Expected to perform well due to high BA.",
        )
        assert e.confidence == ConfidenceLevel.HIGH
        assert len(e.top_features) == 2


# ── Orchestrator schemas ──────────────────────────────────────────────────


class TestOrchestratorSchemas:
    def test_agent_health(self):
        h = AgentHealthStatus(
            agent_id=AgentType.DATA_QUALITY,
            is_healthy=True,
            uptime_seconds=3600.0,
            tasks_completed=50,
            tasks_failed=2,
            error_rate=0.038,
        )
        assert h.is_healthy is True

    def test_system_status(self):
        agent = AgentHealthStatus(
            agent_id=AgentType.DATA_QUALITY,
            is_healthy=True,
            uptime_seconds=3600.0,
        )
        status = SystemStatus(
            all_agents_healthy=True,
            agent_statuses=[agent],
            total_tasks_pending=5,
        )
        assert len(status.agent_statuses) == 1
        assert status.total_tasks_pending == 5
