"""Tests for shared/messaging.py – Redis-based messaging system."""

import pytest

from shared.schemas import (
    AgentType,
    TaskStatus,
    TaskPriority,
    AlertSeverity,
    AgentTask,
    AgentResult,
    AgentAlert,
)
from shared.config import settings


# ── Task operations ────────────────────────────────────────────────────────


class TestTaskOperations:
    async def test_publish_task(self, fake_mq, sample_task):
        assert await fake_mq.publish_task(sample_task) is True

        depth = await fake_mq.get_queue_depth(settings.task_queue_name)
        assert depth == 1

    async def test_consume_task(self, fake_mq, sample_task):
        await fake_mq.publish_task(sample_task)
        consumed = await fake_mq.consume_task(timeout=1)

        assert consumed is not None
        assert consumed.task_id == sample_task.task_id
        assert consumed.agent_id == sample_task.agent_id
        assert consumed.task_type == sample_task.task_type

    async def test_consume_empty_queue_returns_none(self, fake_mq):
        consumed = await fake_mq.consume_task(timeout=1)
        assert consumed is None

    async def test_publish_multiple_tasks(self, fake_mq):
        tasks = [
            AgentTask(
                task_id=f"task_{i}",
                agent_id=AgentType.DATA_QUALITY,
                task_type="check",
            )
            for i in range(5)
        ]
        for t in tasks:
            await fake_mq.publish_task(t)

        depth = await fake_mq.get_queue_depth(settings.task_queue_name)
        assert depth == 5

    async def test_fifo_ordering(self, fake_mq):
        """lpush + brpop should give FIFO behaviour."""
        for i in range(3):
            task = AgentTask(
                task_id=f"task_{i}",
                agent_id=AgentType.DATA_QUALITY,
                task_type="check",
            )
            await fake_mq.publish_task(task)

        first = await fake_mq.consume_task(timeout=1)
        assert first.task_id == "task_0"

    async def test_task_round_trip_preserves_parameters(self, fake_mq):
        task = AgentTask(
            task_id="param_test",
            agent_id=AgentType.MODEL_MONITOR,
            task_type="drift_check",
            parameters={"model": "xgb_v2", "threshold": 0.15},
        )
        await fake_mq.publish_task(task)
        consumed = await fake_mq.consume_task(timeout=1)

        assert consumed.parameters == task.parameters


# ── Result operations ──────────────────────────────────────────────────────


class TestResultOperations:
    async def test_publish_result(self, fake_mq, sample_result):
        assert await fake_mq.publish_result(sample_result) is True

    async def test_get_result_by_task_id(self, fake_mq, sample_result):
        await fake_mq.publish_result(sample_result)
        retrieved = await fake_mq.get_result(sample_result.task_id)

        assert retrieved is not None
        assert retrieved.task_id == sample_result.task_id
        assert retrieved.status == sample_result.status
        assert retrieved.metrics == sample_result.metrics

    async def test_get_nonexistent_result(self, fake_mq):
        assert await fake_mq.get_result("does_not_exist") is None


# ── Alert operations ───────────────────────────────────────────────────────


class TestAlertOperations:
    async def test_publish_alert(self, fake_mq, sample_alert):
        assert await fake_mq.publish_alert(sample_alert) is True

        depth = await fake_mq.get_queue_depth(settings.alert_queue_name)
        assert depth == 1


# ── Heartbeat operations ──────────────────────────────────────────────────


class TestHeartbeatOperations:
    async def test_update_and_get_heartbeat(self, fake_mq):
        await fake_mq.update_agent_heartbeat("data_quality")
        heartbeat = await fake_mq.get_agent_heartbeat("data_quality")
        assert heartbeat is not None

    async def test_get_nonexistent_heartbeat(self, fake_mq):
        assert await fake_mq.get_agent_heartbeat("nonexistent") is None


# ── Queue utilities ────────────────────────────────────────────────────────


class TestQueueUtilities:
    async def test_queue_depth_empty(self, fake_mq):
        assert await fake_mq.get_queue_depth(settings.task_queue_name) == 0

    async def test_clear_queue(self, fake_mq, sample_task):
        await fake_mq.publish_task(sample_task)
        await fake_mq.publish_task(sample_task)
        assert await fake_mq.get_queue_depth(settings.task_queue_name) == 2

        await fake_mq.clear_queue(settings.task_queue_name)
        assert await fake_mq.get_queue_depth(settings.task_queue_name) == 0

    async def test_publish_generic_message(self, fake_mq):
        success = await fake_mq.publish_message("test_channel", {"key": "value"})
        assert success is True
