"""Tests for shared/base_agent.py – Base agent lifecycle and task execution."""

import asyncio
import pytest
from datetime import datetime

from shared.schemas import (
    AgentType,
    TaskStatus,
    TaskPriority,
    AlertSeverity,
    AgentTask,
    AgentResult,
)
from shared.base_agent import BaseAgent


# ── Concrete test agent ────────────────────────────────────────────────────


class ConcreteAgent(BaseAgent):
    """Minimal concrete agent used for testing."""

    def __init__(self, agent_id=AgentType.DATA_QUALITY):
        super().__init__(agent_id)
        self.initialized = False
        self.cleaned_up = False
        self.fail_on_task = False

    async def initialize(self):
        self.initialized = True

    async def cleanup(self):
        self.cleaned_up = True

    async def handle_task(self, task: AgentTask) -> AgentResult:
        if self.fail_on_task:
            raise ValueError("Simulated task failure")
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={"processed": True},
            duration_seconds=0.1,
        )


# ── Helper ─────────────────────────────────────────────────────────────────


def _make_task(task_id="t_001", priority=TaskPriority.MEDIUM):
    return AgentTask(
        task_id=task_id,
        agent_id=AgentType.DATA_QUALITY,
        task_type="check_anomalies",
        priority=priority,
    )


# ── Initialisation ─────────────────────────────────────────────────────────


class TestInit:
    def test_default_state(self):
        agent = ConcreteAgent()
        assert agent.agent_id == AgentType.DATA_QUALITY
        assert agent.is_running is False
        assert agent.tasks_completed == 0
        assert agent.tasks_failed == 0
        assert agent.start_time is None

    def test_different_agent_types(self):
        for agent_type in AgentType:
            agent = ConcreteAgent(agent_type)
            assert agent.agent_id == agent_type


# ── Metrics ────────────────────────────────────────────────────────────────


class TestMetrics:
    def test_error_rate_no_tasks(self):
        agent = ConcreteAgent()
        assert agent.get_error_rate() == 0.0

    def test_error_rate_with_tasks(self):
        agent = ConcreteAgent()
        agent.tasks_completed = 8
        agent.tasks_failed = 2
        assert agent.get_error_rate() == pytest.approx(0.2)

    def test_uptime_not_started(self):
        assert ConcreteAgent().get_uptime_seconds() == 0

    def test_uptime_after_start(self):
        agent = ConcreteAgent()
        agent.start_time = datetime.now()
        assert agent.get_uptime_seconds() >= 0


# ── Task execution ─────────────────────────────────────────────────────────


class TestTaskExecution:
    async def test_success_increments_completed(self, global_mq):
        agent = ConcreteAgent()
        await agent._execute_task(_make_task())

        assert agent.tasks_completed == 1
        assert agent.tasks_failed == 0

    async def test_failure_increments_failed(self, global_mq):
        agent = ConcreteAgent()
        agent.fail_on_task = True
        await agent._execute_task(_make_task())

        assert agent.tasks_completed == 0
        assert agent.tasks_failed == 1

    async def test_result_published_on_success(self, global_mq):
        agent = ConcreteAgent()
        await agent._execute_task(_make_task("res_ok"))

        result = await global_mq.get_result("res_ok")
        assert result is not None
        assert result.status == TaskStatus.COMPLETED

    async def test_failure_result_published(self, global_mq):
        agent = ConcreteAgent()
        agent.fail_on_task = True
        await agent._execute_task(_make_task("res_fail"))

        result = await global_mq.get_result("res_fail")
        assert result is not None
        assert result.status == TaskStatus.FAILED
        assert "Simulated task failure" in result.error_message

    async def test_critical_failure_publishes_alert(self, global_mq):
        agent = ConcreteAgent()
        agent.fail_on_task = True
        await agent._execute_task(_make_task("crit_001", TaskPriority.CRITICAL))

        from shared.config import settings

        depth = await global_mq.get_queue_depth(settings.alert_queue_name)
        assert depth == 1

    async def test_non_critical_failure_no_alert(self, global_mq):
        agent = ConcreteAgent()
        agent.fail_on_task = True
        await agent._execute_task(_make_task("lo_001", TaskPriority.LOW))

        from shared.config import settings

        depth = await global_mq.get_queue_depth(settings.alert_queue_name)
        assert depth == 0


# ── Lifecycle ──────────────────────────────────────────────────────────────


class TestLifecycle:
    async def test_stop_sets_flags(self, global_mq):
        agent = ConcreteAgent()
        agent.is_running = True
        await agent.stop()

        assert agent.is_running is False
        assert agent.cleaned_up is True


# ── Publishing helpers ─────────────────────────────────────────────────────


class TestPublishing:
    async def test_publish_task_to_another_agent(self, global_mq):
        agent = ConcreteAgent(AgentType.ORCHESTRATOR)
        task_id = await agent.publish_task(
            target_agent=AgentType.DATA_QUALITY,
            task_type="check_anomalies",
            parameters={"src": "test"},
            priority=TaskPriority.HIGH,
        )
        assert task_id is not None

        from shared.config import settings

        depth = await global_mq.get_queue_depth(settings.task_queue_name)
        assert depth == 1


# ── Task context manager ──────────────────────────────────────────────────


class TestTaskContext:
    async def test_context_completes(self, global_mq):
        agent = ConcreteAgent()
        async with agent.task_context("ctx_001"):
            await asyncio.sleep(0.01)
        # reaching here means no exception was raised
