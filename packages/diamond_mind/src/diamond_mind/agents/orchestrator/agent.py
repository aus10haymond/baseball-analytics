"""
Orchestrator Agent

LLM-based coordinator that manages other agents, routes incoming tasks
to the appropriate specialist agent, monitors system health, and resolves
conflicts between competing agent recommendations.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from shared.base_agent import BaseAgent
from shared import AgentType, AgentTask, AgentResult, TaskStatus, AlertSeverity, TaskPriority
from shared.schemas import AgentHealthStatus, SystemStatus
from shared import settings, message_queue

from .llm_client import LLMClient, LLMError, LLMConfigError, extract_json


# ------------------------------------------------------------------
# System prompt describing agent capabilities (used for routing)
# ------------------------------------------------------------------

_ROUTING_SYSTEM_PROMPT = """\
You are the Orchestrator for a baseball analytics multi-agent ML system.
Your job is to route incoming requests to the correct specialist agent.

Available agents:
  - data_quality: Detects anomalies, validates schemas, repairs dirty data.
    Task types: check_anomalies, validate_schema, repair_data, full_quality_check
  - model_monitor: Tracks model performance, detects drift, runs A/B tests.
    Task types: check_drift, evaluate_performance, run_ab_test, register_model_version, rollback_model
  - feature_engineer: Generates and evaluates new features using genetic algorithms.
    Task types: search_features, evaluate_feature, generate_features
  - explainer: Produces human-readable explanations and SHAP analysis for predictions.
    Task types: explain_prediction, generate_report, batch_explain

Respond ONLY with a JSON object (no markdown, no commentary):
{
  "target_agent": "<agent_id>",
  "task_type": "<task_type>",
  "parameters": { ... },
  "reasoning": "<one sentence>"
}
"""

_CONFLICT_SYSTEM_PROMPT = """\
You are the Orchestrator for a baseball analytics multi-agent ML system.
Two specialist agents have produced conflicting recommendations.
Analyze the conflict and return a resolution.

Respond ONLY with a JSON object:
{
  "resolution": "<accepted | rejected | compromise>",
  "chosen_recommendation": "<description of what to do>",
  "reasoning": "<explanation>",
  "action_required": true | false
}
"""


class OrchestratorAgent(BaseAgent):
    """LLM-based coordinator for the Diamond Mind agent system."""

    def __init__(self):
        super().__init__(AgentType.ORCHESTRATOR)
        self._llm: Optional[LLMClient] = None
        self._llm_available: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        self.logger.info("Initializing Orchestrator Agent")
        try:
            self._llm = LLMClient.from_settings(settings)
            self._llm_available = True
            self.logger.info(
                f"LLM client ready: provider={settings.llm_provider} model={settings.llm_model}"
            )
        except LLMConfigError as exc:
            self._llm_available = False
            self.logger.warning(
                f"LLM unavailable (no API key configured): {exc}. "
                "Orchestrator will use rule-based fallbacks."
            )

    async def cleanup(self):
        self.logger.info("Orchestrator Agent cleaned up")

    # ------------------------------------------------------------------
    # Task dispatch
    # ------------------------------------------------------------------

    async def handle_task(self, task: AgentTask) -> AgentResult:
        handlers = {
            "route_task": self._route_task,
            "system_health": self._system_health,
            "resolve_conflict": self._resolve_conflict,
            "retrain_model": self._retrain_model,
        }
        handler = handlers.get(task.task_type)
        if handler is None:
            raise ValueError(f"Unknown task type: {task.task_type}")
        return await handler(task)

    # ------------------------------------------------------------------
    # 4.2  Task Routing
    # ------------------------------------------------------------------

    async def _route_task(self, task: AgentTask) -> AgentResult:
        """
        Use the LLM to determine which agent and task_type best handles the request,
        then publish the routed task.
        """
        task_description = task.parameters.get("task_description", "")
        context = task.parameters.get("context", {})

        routing = await self._call_routing_llm(task_description, context)

        target = AgentType(routing["target_agent"])
        routed_task_id = await self.publish_task(
            target_agent=target,
            task_type=routing["task_type"],
            parameters=routing.get("parameters", {}),
            priority=task.priority,
        )

        self.logger.info(
            f"Routed '{task_description[:60]}' → {target.value}/{routing['task_type']} "
            f"(task {routed_task_id})"
        )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "routed_to": target.value,
                "task_type": routing["task_type"],
                "routed_task_id": routed_task_id,
                "reasoning": routing.get("reasoning", ""),
            },
            metrics={},
            duration_seconds=0.0,
        )

    async def _call_routing_llm(
        self, task_description: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ask the LLM to determine how to route ``task_description``.

        Falls back to a keyword-based heuristic when the LLM is unavailable.
        """
        if self._llm_available and self._llm is not None:
            prompt = f"Task: {task_description}\nContext: {json.dumps(context)}"
            try:
                raw = await self._llm.call(prompt, system_prompt=_ROUTING_SYSTEM_PROMPT)
                return extract_json(raw)
            except (LLMError, LLMConfigError) as exc:
                self.logger.warning(f"LLM routing call failed ({exc}); using fallback.")

        return self._rule_based_routing(task_description)

    def _rule_based_routing(self, task_description: str) -> Dict[str, Any]:
        """
        Keyword-based fallback routing when the LLM is unavailable.

        Returns the same dict shape as the LLM response.
        """
        desc = task_description.lower()

        if any(kw in desc for kw in ("drift", "monitor", "ab test", "performance", "retrain")):
            return {
                "target_agent": "model_monitor",
                "task_type": "check_drift",
                "parameters": {},
                "reasoning": "Rule-based: drift/monitoring keyword detected.",
            }
        if any(kw in desc for kw in ("anomaly", "quality", "schema", "missing", "repair")):
            return {
                "target_agent": "data_quality",
                "task_type": "check_anomalies",
                "parameters": {},
                "reasoning": "Rule-based: data quality keyword detected.",
            }
        if any(kw in desc for kw in ("feature", "engineering", "genetic", "evolve")):
            return {
                "target_agent": "feature_engineer",
                "task_type": "search_features",
                "parameters": {},
                "reasoning": "Rule-based: feature engineering keyword detected.",
            }
        if any(kw in desc for kw in ("explain", "shap", "prediction", "narrative")):
            return {
                "target_agent": "explainer",
                "task_type": "explain_prediction",
                "parameters": {},
                "reasoning": "Rule-based: explanation keyword detected.",
            }

        # Default: data quality as a safe starting point
        return {
            "target_agent": "data_quality",
            "task_type": "full_quality_check",
            "parameters": {},
            "reasoning": "Rule-based: no keyword match; defaulting to data quality check.",
        }

    # ------------------------------------------------------------------
    # 4.3  System Health Monitoring
    # ------------------------------------------------------------------

    async def _system_health(self, task: AgentTask) -> AgentResult:
        """
        Check the heartbeat and queue state of every registered agent.

        Returns a ``SystemStatus`` with per-agent ``AgentHealthStatus`` entries
        and publishes an alert for any agent that has missed its heartbeat.
        """
        stale_threshold_seconds = task.parameters.get(
            "stale_threshold_seconds",
            settings.heartbeat_interval_seconds * 3,
        )

        agent_statuses: List[AgentHealthStatus] = []
        now = datetime.now()

        for agent_type in AgentType:
            if agent_type == AgentType.ORCHESTRATOR:
                # Self — always healthy while handling this task
                status = AgentHealthStatus(
                    agent_id=agent_type,
                    is_healthy=True,
                    uptime_seconds=self.get_uptime_seconds(),
                    tasks_completed=self.tasks_completed,
                    tasks_failed=self.tasks_failed,
                    avg_task_duration_seconds=0.0,
                    last_heartbeat=now,
                    error_rate=self.get_error_rate(),
                )
            else:
                heartbeat = await message_queue.get_agent_heartbeat(agent_type.value)
                if heartbeat is None:
                    is_healthy = False
                    last_hb = now - timedelta(seconds=stale_threshold_seconds + 1)
                else:
                    elapsed = (now - heartbeat).total_seconds()
                    is_healthy = elapsed <= stale_threshold_seconds
                    last_hb = heartbeat

                status = AgentHealthStatus(
                    agent_id=agent_type,
                    is_healthy=is_healthy,
                    uptime_seconds=0.0,
                    last_heartbeat=last_hb,
                )

            agent_statuses.append(status)

        all_healthy = all(s.is_healthy for s in agent_statuses)

        task_queue_depth = await message_queue.get_queue_depth(settings.task_queue_name)
        result_queue_depth = await message_queue.get_queue_depth(settings.result_queue_name)

        system_status = SystemStatus(
            all_agents_healthy=all_healthy,
            agent_statuses=agent_statuses,
            total_tasks_pending=task_queue_depth,
            total_tasks_running=0,
            message_queue_depth=task_queue_depth + result_queue_depth,
        )

        # Alert on unhealthy agents
        unhealthy = [s for s in agent_statuses if not s.is_healthy]
        if unhealthy:
            await self.publish_alert(
                severity=AlertSeverity.WARNING,
                message=f"{len(unhealthy)} agent(s) are unhealthy: "
                f"{', '.join(s.agent_id.value for s in unhealthy)}",
                details={
                    "unhealthy_agents": [s.agent_id.value for s in unhealthy],
                    "stale_threshold_seconds": stale_threshold_seconds,
                },
                requires_action=True,
                suggested_actions=["Check agent processes", "Restart unresponsive agents"],
            )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data=system_status.model_dump(mode="json"),
            metrics={
                "unhealthy_agent_count": float(len(unhealthy)),
                "task_queue_depth": float(task_queue_depth),
                "all_healthy": float(all_healthy),
            },
            duration_seconds=0.0,
        )

    # ------------------------------------------------------------------
    # 4.4  Conflict Resolution
    # ------------------------------------------------------------------

    async def _resolve_conflict(self, task: AgentTask) -> AgentResult:
        """
        Use the LLM to resolve conflicting recommendations from two agents.

        ``task.parameters`` must include:
          - ``conflict_description``: str — human-readable description of the conflict
          - ``recommendations``: list of dicts with keys agent_id, recommendation, data
        """
        conflict_description = task.parameters.get("conflict_description", "")
        recommendations: List[Dict[str, Any]] = task.parameters.get("recommendations", [])

        resolution = await self._call_conflict_llm(conflict_description, recommendations)

        if resolution.get("action_required", False):
            await self.publish_alert(
                severity=AlertSeverity.WARNING,
                message=f"Conflict resolved — action required: {resolution.get('chosen_recommendation', '')}",
                details=resolution,
                requires_action=True,
                suggested_actions=["Review conflict resolution and act accordingly"],
            )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data=resolution,
            metrics={},
            duration_seconds=0.0,
        )

    async def _call_conflict_llm(
        self, conflict_description: str, recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ask the LLM to resolve the conflict; fallback to first recommendation."""
        if self._llm_available and self._llm is not None:
            prompt = (
                f"Conflict: {conflict_description}\n"
                f"Recommendations:\n{json.dumps(recommendations, indent=2)}"
            )
            try:
                raw = await self._llm.call(prompt, system_prompt=_CONFLICT_SYSTEM_PROMPT)
                return extract_json(raw)
            except (LLMError, LLMConfigError) as exc:
                self.logger.warning(f"LLM conflict resolution failed ({exc}); using fallback.")

        return self._rule_based_conflict_resolution(recommendations)

    def _rule_based_conflict_resolution(
        self, recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fallback: accept the first recommendation from the highest-priority agent.
        Priority order: model_monitor > data_quality > feature_engineer > explainer
        """
        priority_order = [
            AgentType.MODEL_MONITOR.value,
            AgentType.DATA_QUALITY.value,
            AgentType.FEATURE_ENGINEER.value,
            AgentType.EXPLAINER.value,
        ]

        chosen = None
        for agent_id in priority_order:
            for rec in recommendations:
                if rec.get("agent_id") == agent_id:
                    chosen = rec
                    break
            if chosen:
                break

        if chosen is None and recommendations:
            chosen = recommendations[0]

        return {
            "resolution": "accepted",
            "chosen_recommendation": chosen.get("recommendation", "") if chosen else "",
            "reasoning": "Rule-based: selected highest-priority agent recommendation.",
            "action_required": False,
        }

    # ------------------------------------------------------------------
    # Retraining orchestration (receives tasks from ModelMonitorAgent)
    # ------------------------------------------------------------------

    async def _retrain_model(self, task: AgentTask) -> AgentResult:
        """
        Orchestrate a model retraining workflow:
          1. Trigger a data quality check first.
          2. Log the retraining request (actual training would be a separate service).
        """
        model_name = task.parameters.get("model_name", "")
        reason = task.parameters.get("reason", "unknown")
        triggered_by = task.parameters.get("triggered_by", "unknown")

        self.logger.info(
            f"Retraining requested for model={model_name!r} reason={reason!r} "
            f"triggered_by={triggered_by!r}"
        )

        # Step 1: Ensure data quality before retraining
        dq_task_id = await self.publish_task(
            target_agent=AgentType.DATA_QUALITY,
            task_type="full_quality_check",
            parameters={"model_name": model_name, "triggered_by": "retrain_workflow"},
            priority=TaskPriority.HIGH,
        )

        await self.publish_alert(
            severity=AlertSeverity.INFO,
            message=f"Retraining workflow started for {model_name}",
            details={
                "model_name": model_name,
                "reason": reason,
                "triggered_by": triggered_by,
                "dq_preflight_task_id": dq_task_id,
            },
            requires_action=False,
        )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "model_name": model_name,
                "reason": reason,
                "dq_preflight_task_id": dq_task_id,
                "status": "retraining_workflow_initiated",
            },
            metrics={},
            duration_seconds=0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers (sync, for easy unit-testing)
    # ------------------------------------------------------------------

    def build_routing_prompt(self, task_description: str, context: Dict[str, Any]) -> str:
        """Return the user-side prompt for routing (exposed for testing)."""
        return f"Task: {task_description}\nContext: {json.dumps(context)}"

    def build_conflict_prompt(
        self, conflict_description: str, recommendations: List[Dict[str, Any]]
    ) -> str:
        """Return the user-side prompt for conflict resolution (exposed for testing)."""
        return (
            f"Conflict: {conflict_description}\n"
            f"Recommendations:\n{json.dumps(recommendations, indent=2)}"
        )


if __name__ == "__main__":
    import asyncio
    from shared.messaging import init_messaging, shutdown_messaging

    async def main():
        await init_messaging()
        agent = OrchestratorAgent()
        try:
            await agent.start()
        except KeyboardInterrupt:
            await agent.stop()
        finally:
            await shutdown_messaging()

    asyncio.run(main())
