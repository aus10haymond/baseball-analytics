"""
Integration between diamond_mind agents and fantasy_mlb_ai.

Submits tasks to:
- DataQualityAgent  — validate ESPN roster data
- ModelMonitorAgent — track projection accuracy via PredictionTracker metrics
- OrchestratorAgent — coordinate the daily fantasy workflow
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

PROJECTION_MODEL_NAME = "fantasy_mlb_projection"


def _task_id() -> str:
    return str(uuid.uuid4())


class FantasyMLBAIIntegration:
    """
    Bridge between diamond_mind agents and the fantasy_mlb_ai package.

    Exposes high-level methods that translate fantasy_mlb_ai concepts
    (ESPN roster data, ML projections, prediction accuracy) into AgentTask
    messages and publish them to the shared Redis queue.

    Usage::

        async with FantasyMLBAIIntegration() as integration:
            task_id = await integration.sync_accuracy_to_model_monitor()
            result  = await integration.wait_for_result(task_id)
    """

    def __init__(self, queue: Optional[MessageQueue] = None) -> None:
        self._settings = settings
        self._queue = queue
        self._owns_queue = queue is None

    async def __aenter__(self) -> "FantasyMLBAIIntegration":
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

    async def validate_roster_data(
        self,
        roster_csv_path: str,
        *,
        auto_fix: bool = False,
        priority: TaskPriority = TaskPriority.MEDIUM,
    ) -> str:
        """
        Ask DataQualityAgent to validate an ESPN roster CSV file.

        Parameters
        ----------
        roster_csv_path:
            Path to the CSV produced by ``recommend_actions_ml.py``.
        """
        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.DATA_QUALITY,
            task_type="check_data_quality",
            priority=priority,
            parameters={
                "data_source": roster_csv_path,
                "auto_fix": auto_fix,
                "threshold": self._settings.dq_anomaly_threshold,
                "contamination": 0.05,
            },
        )
        await self._queue.publish_task(task)
        logger.info("Submitted roster data quality check (task_id=%s)", task_id)
        return task_id

    async def validate_recommendations_file(
        self,
        recommendations_path: str,
        *,
        priority: TaskPriority = TaskPriority.LOW,
    ) -> str:
        """Ask DataQualityAgent to validate a daily recommendations CSV."""
        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.DATA_QUALITY,
            task_type="validate_schema",
            priority=priority,
            parameters={"data_source": recommendations_path},
        )
        await self._queue.publish_task(task)
        logger.info(
            "Submitted recommendations schema check (task_id=%s)", task_id
        )
        return task_id

    # ------------------------------------------------------------------
    # Model Monitoring — projection accuracy
    # ------------------------------------------------------------------

    async def sync_accuracy_to_model_monitor(
        self,
        *,
        days: int = 30,
        model_version: str = "latest",
        priority: TaskPriority = TaskPriority.HIGH,
    ) -> str:
        """
        Read accuracy metrics from PredictionTracker and forward them to
        ModelMonitorAgent as an ``evaluate_performance`` task.

        This keeps diamond_mind's model registry up to date with how well
        the fantasy projection engine is actually performing.
        """
        predictions, actuals = self._load_recent_predictions(days=days)
        if not predictions:
            logger.warning(
                "No predictions with actuals found for the last %d days; skipping sync",
                days,
            )
            return ""

        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.MODEL_MONITOR,
            task_type="evaluate_performance",
            priority=priority,
            parameters={
                "model_name": PROJECTION_MODEL_NAME,
                "model_version": model_version,
                "predictions": predictions,
                "actuals": actuals,
                "prediction_times_ms": [],
            },
        )
        await self._queue.publish_task(task)
        logger.info(
            "Synced %d predictions to ModelMonitorAgent (task_id=%s)",
            len(predictions),
            task_id,
        )
        return task_id

    async def check_projection_drift(
        self,
        *,
        baseline_date: Optional[str] = None,
        current_date: Optional[str] = None,
        psi_threshold: float = 0.2,
        ks_p_threshold: float = 0.05,
        priority: TaskPriority = TaskPriority.MEDIUM,
    ) -> str:
        """
        Ask ModelMonitorAgent to compare projection distributions across dates.

        Writes temporary CSV snapshots of prediction data and submits a
        ``check_drift`` task so the monitor can compute PSI/KS statistics.
        """
        import tempfile
        import pandas as pd

        baseline_df, current_df = self._load_prediction_snapshots(
            baseline_date=baseline_date, current_date=current_date
        )
        if baseline_df is None or current_df is None:
            logger.warning("Could not load prediction snapshots; skipping drift check")
            return ""

        tmp = Path(tempfile.mkdtemp())
        baseline_path = str(tmp / "baseline_projections.csv")
        current_path = str(tmp / "current_projections.csv")
        baseline_df.to_csv(baseline_path, index=False)
        current_df.to_csv(current_path, index=False)

        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.MODEL_MONITOR,
            task_type="check_drift",
            priority=priority,
            parameters={
                "data_source": current_path,
                "baseline_source": baseline_path,
                "model_name": PROJECTION_MODEL_NAME,
                "psi_threshold": psi_threshold,
                "ks_p_threshold": ks_p_threshold,
            },
        )
        await self._queue.publish_task(task)
        logger.info(
            "Submitted projection drift check (task_id=%s, baseline=%s, current=%s)",
            task_id,
            baseline_date or "30d ago",
            current_date or "today",
        )
        return task_id

    async def register_projection_model(
        self,
        version_id: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.LOW,
    ) -> str:
        """Register the current projection engine version with ModelMonitorAgent."""
        mm_path = self._fantasy_mlb_path() / "models"
        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.MODEL_MONITOR,
            task_type="register_model_version",
            priority=priority,
            parameters={
                "model_name": PROJECTION_MODEL_NAME,
                "version_id": version_id,
                "model_path": str(mm_path),
                "metadata": metadata or {},
            },
        )
        await self._queue.publish_task(task)
        logger.info(
            "Registered %s version %s (task_id=%s)",
            PROJECTION_MODEL_NAME,
            version_id,
            task_id,
        )
        return task_id

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    async def trigger_daily_workflow(
        self,
        *,
        context: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.HIGH,
    ) -> str:
        """
        Ask OrchestratorAgent to coordinate the daily fantasy workflow.

        The orchestrator will route sub-tasks to the appropriate agents
        (data quality → projections → notification).
        """
        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.ORCHESTRATOR,
            task_type="route_task",
            priority=priority,
            parameters={
                "task_description": (
                    "Run the daily fantasy MLB workflow: validate roster data, "
                    "generate pitcher-aware projections, check prediction drift, "
                    "and report today's recommendations."
                ),
                "context": context or {"triggered_by": "fantasy_mlb_ai_integration"},
            },
        )
        await self._queue.publish_task(task)
        logger.info("Triggered daily fantasy workflow via orchestrator (task_id=%s)", task_id)
        return task_id

    async def request_projection_explanation(
        self,
        player_name: str,
        projection: Dict[str, Any],
        *,
        priority: TaskPriority = TaskPriority.LOW,
    ) -> str:
        """
        Ask ExplainerAgent to generate a SHAP-based narrative for a player's projection.

        Parameters
        ----------
        player_name:
            Display name of the player.
        projection:
            Dict produced by ``MLProjectionEngine.get_batter_projection()``.
        """
        task_id = _task_id()
        task = AgentTask(
            task_id=task_id,
            agent_id=AgentType.EXPLAINER,
            task_type="explain_prediction",
            priority=priority,
            parameters={
                "player_name": player_name,
                "projection": projection,
                "model_name": PROJECTION_MODEL_NAME,
            },
        )
        await self._queue.publish_task(task)
        logger.info(
            "Requested projection explanation for %s (task_id=%s)", player_name, task_id
        )
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

    def _load_recent_predictions(
        self, *, days: int
    ) -> tuple[List[float], List[float]]:
        """
        Load recent predictions and actuals from the PredictionTracker database.

        Returns (predictions, actuals) — two parallel lists of floats.
        """
        try:
            import sqlite3

            db_path = self._fantasy_mlb_path() / "data" / "predictions" / "predictions.db"
            if not db_path.exists():
                logger.warning("PredictionTracker DB not found at %s", db_path)
                return [], []

            conn = sqlite3.connect(str(db_path))
            rows = conn.execute(
                """
                SELECT projected_points, actual_points
                FROM predictions
                WHERE actual_points IS NOT NULL
                  AND date >= date('now', ? || ' days')
                """,
                (f"-{days}",),
            ).fetchall()
            conn.close()

            predictions = [r[0] for r in rows]
            actuals = [r[1] for r in rows]
            return predictions, actuals
        except Exception as exc:
            logger.error("Failed to load predictions from tracker: %s", exc)
            return [], []

    def _load_prediction_snapshots(
        self,
        *,
        baseline_date: Optional[str],
        current_date: Optional[str],
    ):
        """
        Return two DataFrames (baseline, current) for drift comparison.

        Each DataFrame contains numeric projection columns.
        Returns (None, None) if data cannot be loaded.
        """
        try:
            import sqlite3
            import pandas as pd

            db_path = self._fantasy_mlb_path() / "data" / "predictions" / "predictions.db"
            if not db_path.exists():
                return None, None

            conn = sqlite3.connect(str(db_path))

            baseline_clause = (
                f"AND date = '{baseline_date}'"
                if baseline_date
                else "AND date <= date('now', '-30 days')"
            )
            current_clause = (
                f"AND date = '{current_date}'"
                if current_date
                else "AND date >= date('now', '-7 days')"
            )

            cols = "projected_points, confidence"
            baseline_df = pd.read_sql_query(
                f"SELECT {cols} FROM predictions WHERE 1=1 {baseline_clause}", conn
            )
            current_df = pd.read_sql_query(
                f"SELECT {cols} FROM predictions WHERE 1=1 {current_clause}", conn
            )
            conn.close()

            if baseline_df.empty or current_df.empty:
                return None, None

            # Encode confidence as ordinal so drift detection can compare it
            conf_map = {"low": 0, "medium": 1, "high": 2}
            for df in (baseline_df, current_df):
                df["confidence_ord"] = df["confidence"].map(conf_map).fillna(1)

            return baseline_df.drop(columns=["confidence"]), current_df.drop(
                columns=["confidence"]
            )
        except Exception as exc:
            logger.error("Failed to load prediction snapshots: %s", exc)
            return None, None

    def _fantasy_mlb_path(self) -> Path:
        if self._settings.fantasy_mlb_path:
            return Path(self._settings.fantasy_mlb_path)
        # packages/diamond_mind/src/diamond_mind/integrations/ → 4 levels up = packages/
        return Path(__file__).resolve().parents[4] / "fantasy_mlb_ai"
