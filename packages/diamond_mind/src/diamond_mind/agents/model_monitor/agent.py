"""
Model Monitor Agent

Monitors ML model performance, detects concept drift, triggers automatic
retraining, manages A/B tests, and maintains a versioned model registry.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from shared.base_agent import BaseAgent
from shared import AgentType, AgentTask, AgentResult, TaskStatus, AlertSeverity, TaskPriority
from shared.schemas import DriftDetectionResult, ModelPerformanceMetrics

from .drift_detection import detect_feature_drift, calculate_psi, run_ks_test
from .ab_testing import ABTest


# Maximum number of performance snapshots kept per model in history
_MAX_HISTORY = 100


class ModelMonitorAgent(BaseAgent):
    """Agent responsible for monitoring ML model performance and drift."""

    def __init__(self):
        super().__init__(AgentType.MODEL_MONITOR)

        # model_name -> deque of ModelPerformanceMetrics (most-recent last)
        self._performance_history: Dict[str, deque] = {}

        # model_name -> {version_id -> {"metrics": ..., "path": ...}}
        self._model_registry: Dict[str, Dict[str, Any]] = {}

        # test_id -> ABTest instance
        self._active_ab_tests: Dict[str, ABTest] = {}

        # model_name -> baseline DataFrame (stored in memory for drift checks)
        self._baselines: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        self.logger.info("Model Monitor Agent initialized")

    async def cleanup(self):
        self.logger.info("Model Monitor Agent cleaned up")

    # ------------------------------------------------------------------
    # Task dispatch
    # ------------------------------------------------------------------

    async def handle_task(self, task: AgentTask) -> AgentResult:
        handlers = {
            "check_drift": self._check_drift,
            "evaluate_performance": self._evaluate_performance,
            "trigger_retraining": self._trigger_retraining,
            "run_ab_test": self._run_ab_test,
            "record_ab_outcome": self._record_ab_outcome,
            "register_model_version": self._register_model_version,
            "rollback_model": self._rollback_model,
        }
        handler = handlers.get(task.task_type)
        if handler is None:
            raise ValueError(f"Unknown task type: {task.task_type}")
        return await handler(task)

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------

    async def _check_drift(self, task: AgentTask) -> AgentResult:
        """Detect feature drift between baseline and current data."""
        data_source = task.parameters.get("data_source", "")
        baseline_source = task.parameters.get("baseline_source", "")
        model_name = task.parameters.get("model_name", data_source)
        psi_threshold = task.parameters.get("psi_threshold", 0.2)
        ks_p_threshold = task.parameters.get("ks_p_threshold", 0.05)

        current_df = self._load_data(data_source)

        if baseline_source:
            baseline_df = self._load_data(baseline_source)
            self._baselines[model_name] = baseline_df
        elif model_name in self._baselines:
            baseline_df = self._baselines[model_name]
        else:
            # First visit — store as baseline; nothing to compare yet
            self._baselines[model_name] = current_df
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result_data={
                    "drift_detected": False,
                    "first_visit": True,
                    "message": "Baseline stored; no prior baseline to compare against.",
                },
                metrics={"drift_score": 0.0},
                duration_seconds=0.0,
            )

        drift_result = detect_feature_drift(
            baseline_df, current_df,
            psi_threshold=psi_threshold,
            ks_p_threshold=ks_p_threshold,
        )

        if drift_result.drift_detected:
            await self.publish_alert(
                severity=AlertSeverity.WARNING,
                message=f"Feature drift detected in {model_name}",
                details={
                    "drift_score": drift_result.drift_score,
                    "affected_features": drift_result.affected_features,
                    "recommendation": drift_result.recommendation,
                },
                requires_action=True,
                suggested_actions=["Trigger retraining", "Inspect upstream data pipeline"],
            )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data=drift_result.model_dump(mode="json"),
            metrics={
                "drift_score": drift_result.drift_score,
                "affected_feature_count": float(len(drift_result.affected_features)),
            },
            duration_seconds=0.0,
        )

    async def _evaluate_performance(self, task: AgentTask) -> AgentResult:
        """Compute and store model performance metrics; alert on degradation."""
        model_name = task.parameters.get("model_name", "")
        model_version = task.parameters.get("model_version", "unknown")
        predictions = task.parameters.get("predictions", [])
        actuals = task.parameters.get("actuals", [])
        prediction_times_ms = task.parameters.get("prediction_times_ms", [])

        metrics = self._compute_performance_metrics(
            model_name=model_name,
            model_version=model_version,
            predictions=predictions,
            actuals=actuals,
            prediction_times_ms=prediction_times_ms,
        )

        # Store in history
        if model_name not in self._performance_history:
            self._performance_history[model_name] = deque(maxlen=_MAX_HISTORY)
        self._performance_history[model_name].append(metrics)

        # Check for degradation vs. previous snapshot
        degraded, delta = self._check_degradation(model_name, metrics)
        if degraded:
            await self.publish_alert(
                severity=AlertSeverity.WARNING,
                message=f"Performance degradation detected for {model_name}",
                details={
                    "current_accuracy": metrics.accuracy,
                    "accuracy_delta": delta,
                    "model_version": model_version,
                },
                requires_action=True,
                suggested_actions=["Trigger retraining", "Check data pipeline"],
            )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "metrics": metrics.model_dump(mode="json"),
                "degradation_detected": degraded,
                "accuracy_delta": delta,
            },
            metrics={
                "accuracy": metrics.accuracy,
                "prediction_count": float(metrics.prediction_count),
                "avg_prediction_time_ms": metrics.avg_prediction_time_ms,
            },
            duration_seconds=0.0,
        )

    async def _trigger_retraining(self, task: AgentTask) -> AgentResult:
        """Publish a retraining task to the orchestrator."""
        model_name = task.parameters.get("model_name", "")
        reason = task.parameters.get("reason", "manual trigger")
        priority_str = task.parameters.get("priority", "medium").upper()
        priority = TaskPriority[priority_str] if priority_str in TaskPriority.__members__ else TaskPriority.MEDIUM

        retraining_task_id = await self.publish_task(
            target_agent=AgentType.ORCHESTRATOR,
            task_type="retrain_model",
            parameters={
                "model_name": model_name,
                "reason": reason,
                "triggered_by": self.agent_id.value,
            },
            priority=priority,
        )

        self.logger.info(f"Retraining triggered for {model_name} (task {retraining_task_id})")

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "model_name": model_name,
                "reason": reason,
                "retraining_task_id": retraining_task_id,
            },
            metrics={},
            duration_seconds=0.0,
        )

    async def _run_ab_test(self, task: AgentTask) -> AgentResult:
        """Create or retrieve an A/B test and assign the current entity."""
        test_id = task.parameters.get("test_id", "")
        entity_id = task.parameters.get("entity_id", None)
        min_samples = task.parameters.get("min_samples", 50)
        significance = task.parameters.get("significance", 0.05)
        challenger_traffic_pct = task.parameters.get("challenger_traffic_pct", 0.2)

        if test_id not in self._active_ab_tests:
            self._active_ab_tests[test_id] = ABTest(
                test_id=test_id,
                min_samples=min_samples,
                significance=significance,
                challenger_traffic_pct=challenger_traffic_pct,
            )

        test = self._active_ab_tests[test_id]
        variant = test.assign(entity_id=entity_id)
        summary = test.summary()

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={"test_id": test_id, "assigned_variant": variant, "summary": summary},
            metrics={
                "champion_n": float(test.champion.n),
                "challenger_n": float(test.challenger.n),
            },
            duration_seconds=0.0,
        )

    async def _record_ab_outcome(self, task: AgentTask) -> AgentResult:
        """Record a prediction/actual pair for an A/B test variant."""
        test_id = task.parameters.get("test_id", "")
        variant = task.parameters.get("variant", "champion")
        prediction = float(task.parameters.get("prediction", 0.0))
        actual = float(task.parameters.get("actual", 0.0))

        if test_id not in self._active_ab_tests:
            raise ValueError(f"No active A/B test with id: {test_id}")

        test = self._active_ab_tests[test_id]
        test.record(variant=variant, prediction=prediction, actual=actual)

        should_promote = None
        significance_result = None
        if test.has_sufficient_data():
            significance_result = test.test_significance()
            should_promote = significance_result["challenger_wins"]
            if should_promote:
                await self.publish_alert(
                    severity=AlertSeverity.INFO,
                    message=f"A/B test {test_id}: challenger is ready for promotion",
                    details=significance_result,
                    requires_action=True,
                    suggested_actions=["Promote challenger to champion", "Register new model version"],
                )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "test_id": test_id,
                "summary": test.summary(),
                "significance_result": significance_result,
                "should_promote": should_promote,
            },
            metrics={
                "champion_n": float(test.champion.n),
                "challenger_n": float(test.challenger.n),
            },
            duration_seconds=0.0,
        )

    async def _register_model_version(self, task: AgentTask) -> AgentResult:
        """Add a model version to the in-memory registry."""
        model_name = task.parameters.get("model_name", "")
        version_id = task.parameters.get("version_id", "")
        model_path = task.parameters.get("model_path", None)
        metadata = task.parameters.get("metadata", {})

        if model_name not in self._model_registry:
            self._model_registry[model_name] = {}

        self._model_registry[model_name][version_id] = {
            "version_id": version_id,
            "model_path": model_path,
            "metadata": metadata,
        }

        versions = list(self._model_registry[model_name].keys())
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "model_name": model_name,
                "version_id": version_id,
                "total_versions": len(versions),
                "versions": versions,
            },
            metrics={"total_versions": float(len(versions))},
            duration_seconds=0.0,
        )

    async def _rollback_model(self, task: AgentTask) -> AgentResult:
        """Retrieve a previous model version from the registry."""
        model_name = task.parameters.get("model_name", "")
        target_version = task.parameters.get("target_version", None)

        if model_name not in self._model_registry:
            raise ValueError(f"No registered versions for model: {model_name}")

        versions = self._model_registry[model_name]
        if not versions:
            raise ValueError(f"Registry for {model_name} is empty")

        if target_version is not None:
            if target_version not in versions:
                raise ValueError(
                    f"Version {target_version!r} not found for {model_name}. "
                    f"Available: {list(versions.keys())}"
                )
            version_info = versions[target_version]
        else:
            # Roll back to the second-to-last version if multiple exist
            version_list = list(versions.keys())
            if len(version_list) < 2:
                raise ValueError(
                    f"Cannot rollback {model_name}: only one version registered"
                )
            target_version = version_list[-2]
            version_info = versions[target_version]

        await self.publish_alert(
            severity=AlertSeverity.WARNING,
            message=f"Model rollback: {model_name} reverted to version {target_version}",
            details={"model_name": model_name, "rolled_back_to": target_version},
            requires_action=False,
        )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "model_name": model_name,
                "rolled_back_to": target_version,
                "version_info": version_info,
            },
            metrics={},
            duration_seconds=0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers — synchronous, pure functions for easy testing
    # ------------------------------------------------------------------

    def _load_data(self, data_source: str) -> pd.DataFrame:
        """Load a DataFrame from a parquet or CSV file."""
        path = Path(data_source)
        if not path.exists():
            raise FileNotFoundError(f"Data source not found: {data_source}")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported file format: {path.suffix!r}. Use .parquet or .csv.")

    def _compute_performance_metrics(
        self,
        model_name: str,
        model_version: str,
        predictions: List[float],
        actuals: List[float],
        prediction_times_ms: List[float],
    ) -> ModelPerformanceMetrics:
        """Compute ModelPerformanceMetrics from raw predictions and actuals."""
        preds = np.asarray(predictions, dtype=float)
        acts = np.asarray(actuals, dtype=float)

        if len(preds) == 0:
            accuracy = 0.0
        else:
            # Binary accuracy: within 0.5 of actual
            accuracy = float((np.abs(preds - acts) < 0.5).mean())

        avg_time = float(np.mean(prediction_times_ms)) if prediction_times_ms else 0.0

        return ModelPerformanceMetrics(
            model_name=model_name,
            model_version=model_version,
            accuracy=round(accuracy, 6),
            prediction_count=len(predictions),
            avg_prediction_time_ms=round(avg_time, 4),
        )

    def _check_degradation(
        self, model_name: str, current: ModelPerformanceMetrics, threshold: float = 0.05
    ):
        """
        Compare current accuracy to the previous snapshot.

        Returns (degraded: bool, delta: float).
        """
        history = self._performance_history.get(model_name)
        # Need at least 2 entries (previous + current just appended)
        if not history or len(history) < 2:
            return False, 0.0

        previous = list(history)[-2]  # second-to-last
        delta = current.accuracy - previous.accuracy
        return delta < -threshold, round(delta, 6)


if __name__ == "__main__":
    import asyncio
    from shared.messaging import init_messaging, shutdown_messaging

    async def main():
        await init_messaging()
        agent = ModelMonitorAgent()
        try:
            await agent.start()
        except KeyboardInterrupt:
            await agent.stop()
        finally:
            await shutdown_messaging()

    asyncio.run(main())
