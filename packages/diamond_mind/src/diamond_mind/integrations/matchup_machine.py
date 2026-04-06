"""
Integration between diamond_mind agents and matchup_machine.

Submits tasks to:
- DataQualityAgent  — validate matchups.parquet and pitcher profiles
- ModelMonitorAgent — track XGBoost drift and performance
- FeatureEngineerAgent — discover new Statcast features
- OrchestratorAgent — trigger retraining workflows
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys as _sys
from pathlib import Path as _Path

# The diamond_mind codebase uses bare `shared.*` imports with src/diamond_mind on sys.path.
_dm_src = str(_Path(__file__).resolve().parents[1])
if _dm_src not in _sys.path:
    _sys.path.insert(0, _dm_src)

from shared.config import settings  # noqa: E402
from shared.messaging import MessageQueue  # noqa: E402
from shared.schemas import AgentTask, AgentResult, AgentType, TaskPriority, TaskStatus  # noqa: E402

logger = logging.getLogger(__name__)

# Logical model name used in all monitor tasks
XGBOOST_MODEL_NAME = "xgb_outcome_model"


def _task_id() -> str:
    return str(uuid.uuid4())


class MatchupMachineIntegration:
    """
    Bridge between diamond_mind agents and the matchup_machine package.

    Exposes high-level methods that translate matchup_machine concepts
    (Statcast data, XGBoost models, pitcher profiles) into AgentTask
    messages and publish them to the shared Redis queue.

    Usage::

        async with MatchupMachineIntegration() as integration:
            task_id = await integration.validate_statcast_data()
            result  = await integration.wait_for_result(task_id)
    """

    def __init__(self, queue: Optional[MessageQueue] = None) -> None:
        self._settings = settings
        self._queue = queue
        self._owns_queue = queue is None

    async def __aenter__(self) -> "MatchupMachineIntegration":
        if self._owns_queue:
            self._queue = MessageQueue(self._settings)
            await self._queue.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._owns_queue and self._queue:
            await self._queue.disconnect()

    # ------------------------------------------------------------------
    # Data Quality
    # ------------------------------------------------------------------

    async def validate_statcast_data(
        self,
        *,
        auto_fix: bool = False,
        priority: TaskPriority = TaskPriority.MEDIUM,
    ) -> str:
        """
        Ask DataQualityAgent to validate matchups.parquet.

        Returns the task_id so the caller can poll for a result.
        """
        data_path = self._matchup_machine_path() / "data" / "matchups.parquet"
        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.DATA_QUALITY,
            task_type="check_data_quality",
            priority=priority,
            parameters={
                "data_source": str(data_path),
                "auto_fix": auto_fix,
                "threshold": self._settings.dq_anomaly_threshold,
                "contamination": 0.05,
            },
        )
        await self._queue.publish_task(task)
        logger.info("Submitted Statcast data quality check (task_id=%s)", task_id)
        return task_id

    async def validate_pitcher_profiles(
        self,
        *,
        priority: TaskPriority = TaskPriority.LOW,
    ) -> str:
        """Ask DataQualityAgent to validate pitcher_profiles.parquet."""
        data_path = (
            self._matchup_machine_path()
            / "data"
            / "pitcher_profiles"
            / "pitcher_profiles.parquet"
        )
        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.DATA_QUALITY,
            task_type="validate_schema",
            priority=priority,
            parameters={"data_source": str(data_path)},
        )
        await self._queue.publish_task(task)
        logger.info("Submitted pitcher profiles schema check (task_id=%s)", task_id)
        return task_id

    # ------------------------------------------------------------------
    # Model Monitoring
    # ------------------------------------------------------------------

    async def check_model_drift(
        self,
        *,
        baseline_source: Optional[str] = None,
        current_source: Optional[str] = None,
        psi_threshold: float = 0.2,
        ks_p_threshold: float = 0.05,
        priority: TaskPriority = TaskPriority.HIGH,
    ) -> str:
        """
        Ask ModelMonitorAgent to check for feature drift in Statcast data.

        If sources are not provided the integration falls back to the
        standard data paths inside matchup_machine.
        """
        mm_data = self._matchup_machine_path() / "data"
        baseline = baseline_source or str(mm_data / "modeling" / "train.parquet")
        current = current_source or str(mm_data / "matchups.parquet")

        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.MODEL_MONITOR,
            task_type="check_drift",
            priority=priority,
            parameters={
                "data_source": current,
                "baseline_source": baseline,
                "model_name": XGBOOST_MODEL_NAME,
                "psi_threshold": psi_threshold,
                "ks_p_threshold": ks_p_threshold,
            },
        )
        await self._queue.publish_task(task)
        logger.info("Submitted drift check for %s (task_id=%s)", XGBOOST_MODEL_NAME, task_id)
        return task_id

    async def evaluate_model_performance(
        self,
        predictions: List[float],
        actuals: List[float],
        *,
        model_version: str = "latest",
        prediction_times_ms: Optional[List[float]] = None,
        priority: TaskPriority = TaskPriority.HIGH,
    ) -> str:
        """
        Ask ModelMonitorAgent to evaluate XGBoost outcome model performance.

        Parameters
        ----------
        predictions:
            Predicted class probabilities or scores (one per sample).
        actuals:
            Ground-truth labels (one per sample).
        model_version:
            Version tag for the model being evaluated.
        prediction_times_ms:
            Optional per-prediction latency measurements.
        """
        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.MODEL_MONITOR,
            task_type="evaluate_performance",
            priority=priority,
            parameters={
                "model_name": XGBOOST_MODEL_NAME,
                "model_version": model_version,
                "predictions": predictions,
                "actuals": actuals,
                "prediction_times_ms": prediction_times_ms or [],
            },
        )
        await self._queue.publish_task(task)
        logger.info(
            "Submitted performance evaluation for %s v%s (task_id=%s)",
            XGBOOST_MODEL_NAME,
            model_version,
            task_id,
        )
        return task_id

    async def register_model_version(
        self,
        version_id: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.LOW,
    ) -> str:
        """Register a newly trained XGBoost model version with the monitor."""
        model_path = str(
            self._matchup_machine_path() / "models" / "xgb_outcome_model.joblib"
        )
        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.MODEL_MONITOR,
            task_type="register_model_version",
            priority=priority,
            parameters={
                "model_name": XGBOOST_MODEL_NAME,
                "version_id": version_id,
                "model_path": model_path,
                "metadata": metadata or {},
            },
        )
        await self._queue.publish_task(task)
        logger.info(
            "Registered %s version %s (task_id=%s)", XGBOOST_MODEL_NAME, version_id, task_id
        )
        return task_id

    async def trigger_retraining(
        self,
        reason: str,
        *,
        priority: TaskPriority = TaskPriority.HIGH,
    ) -> str:
        """Ask OrchestratorAgent to orchestrate an XGBoost retraining run."""
        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.ORCHESTRATOR,
            task_type="retrain_model",
            priority=priority,
            parameters={
                "model_name": XGBOOST_MODEL_NAME,
                "reason": reason,
                "triggered_by": "matchup_machine_integration",
            },
        )
        await self._queue.publish_task(task)
        logger.info(
            "Triggered retraining for %s — reason: %s (task_id=%s)",
            XGBOOST_MODEL_NAME,
            reason,
            task_id,
        )
        return task_id

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    async def request_feature_discovery(
        self,
        *,
        target_column: str = "outcome_id",
        max_features: int = 20,
        priority: TaskPriority = TaskPriority.LOW,
    ) -> str:
        """
        Ask FeatureEngineerAgent to search for new predictive Statcast features.

        The agent will run its genetic algorithm + LLM pipeline on the
        matchup dataset and return candidate features ranked by importance.
        """
        data_path = self._matchup_machine_path() / "data" / "matchups.parquet"
        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.FEATURE_ENGINEER,
            task_type="discover_features",
            priority=priority,
            parameters={
                "data_source": str(data_path),
                "target_column": target_column,
                "max_features": max_features,
                "domain": "statcast",
            },
        )
        await self._queue.publish_task(task)
        logger.info("Submitted Statcast feature discovery (task_id=%s)", task_id)
        return task_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def wait_for_result(
        self,
        task_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> Optional[AgentResult]:
        """
        Poll the result queue until the task completes or times out.

        Returns the AgentResult, or None if the timeout is reached.
        """
        elapsed = 0.0
        while elapsed < timeout:
            result = await self._queue.get_result(task_id)
            if result is not None:
                status = result.get("status")
                if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    return AgentResult(**result)
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        logger.warning("Timed out waiting for task_id=%s after %.0fs", task_id, timeout)
        return None

    def _matchup_machine_path(self) -> Path:
        if self._settings.matchup_machine_path:
            return Path(self._settings.matchup_machine_path)
        # packages/diamond_mind/src/diamond_mind/integrations/ → 4 levels up = packages/
        return Path(__file__).resolve().parents[4] / "matchup_machine"
