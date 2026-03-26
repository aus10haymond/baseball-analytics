"""
Unit tests for the OrchestratorAgent and its LLM client helpers.

Covers:
- LLMClient: request building, response parsing, extract_json, retry logic
- OrchestratorAgent._rule_based_routing
- OrchestratorAgent._rule_based_conflict_resolution
- OrchestratorAgent.handle_task: route_task, system_health, resolve_conflict, retrain_model
- All LLM-dependent paths mocked — no real API calls made
"""

import sys
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_dm_src = str(Path(__file__).resolve().parents[3] / "src" / "diamond_mind")
if _dm_src not in sys.path:
    sys.path.insert(0, _dm_src)

from shared.schemas import AgentType, TaskStatus, AgentTask, TaskPriority
from agents.orchestrator.agent import OrchestratorAgent
from agents.orchestrator.llm_client import (
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    _build_openai_request,
    _build_anthropic_request,
    _parse_openai_response,
    _parse_anthropic_response,
    extract_json,
    _RateLimiter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_task(task_type: str, parameters: dict | None = None) -> AgentTask:
    return AgentTask(
        task_id="test_orch_001",
        agent_id=AgentType.ORCHESTRATOR,
        task_type=task_type,
        priority=TaskPriority.HIGH,
        parameters=parameters or {},
    )


def agent_with_mock_llm(response_text: str) -> OrchestratorAgent:
    """Return an OrchestratorAgent whose LLM client returns a fixed response."""
    agent = OrchestratorAgent()
    mock_llm = AsyncMock()
    mock_llm.call = AsyncMock(return_value=response_text)
    agent._llm = mock_llm
    agent._llm_available = True
    return agent


# ---------------------------------------------------------------------------
# LLMClient configuration
# ---------------------------------------------------------------------------


class TestLLMClientConfig:
    def test_raises_on_empty_api_key(self):
        with pytest.raises(LLMConfigError, match="No API key"):
            LLMClient(provider="openai", model="gpt-4", api_key="")

    def test_constructs_with_valid_key(self):
        client = LLMClient(provider="openai", model="gpt-4", api_key="sk-test")
        assert client.provider == "openai"
        assert client.model == "gpt-4"

    def test_unknown_provider_raises_on_dispatch(self):
        client = LLMClient(provider="unknown_llm", model="foo", api_key="key")
        with pytest.raises(LLMConfigError, match="Unknown LLM provider"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(client._dispatch("hello", ""))


# ---------------------------------------------------------------------------
# Request builders and response parsers
# ---------------------------------------------------------------------------


class TestOpenAIRequestBuilder:
    def test_includes_system_message(self):
        payload = _build_openai_request("gpt-4", "Be helpful", "Hi", 0.5, 100)
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "Be helpful"
        assert payload["messages"][1]["role"] == "user"

    def test_omits_system_when_empty(self):
        payload = _build_openai_request("gpt-4", "", "Hi", 0.5, 100)
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    def test_model_and_params(self):
        payload = _build_openai_request("gpt-4o", "sys", "user", 0.2, 256)
        assert payload["model"] == "gpt-4o"
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 256


class TestAnthropicRequestBuilder:
    def test_includes_system_field(self):
        payload = _build_anthropic_request("claude-3", "sys", "user", 0.7, 500)
        assert payload["system"] == "sys"
        assert payload["messages"][0]["role"] == "user"

    def test_omits_system_when_empty(self):
        payload = _build_anthropic_request("claude-3", "", "user", 0.7, 500)
        assert "system" not in payload


class TestResponseParsers:
    def test_parse_openai(self):
        data = {"choices": [{"message": {"content": "Hello world"}}]}
        assert _parse_openai_response(data) == "Hello world"

    def test_parse_anthropic(self):
        data = {"content": [{"text": "Hello world"}]}
        assert _parse_anthropic_response(data) == "Hello world"


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


class TestExtractJSON:
    def test_plain_json(self):
        result = extract_json('{"foo": "bar"}')
        assert result == {"foo": "bar"}

    def test_markdown_fenced(self):
        text = '```json\n{"target_agent": "data_quality"}\n```'
        result = extract_json(text)
        assert result["target_agent"] == "data_quality"

    def test_json_with_surrounding_text(self):
        text = 'Here is my response: {"key": 42} Done.'
        result = extract_json(text)
        assert result["key"] == 42

    def test_no_json_raises(self):
        with pytest.raises(LLMError, match="No JSON object found"):
            extract_json("This is plain text with no JSON.")

    def test_invalid_json_raises(self):
        with pytest.raises(LLMError, match="Invalid JSON"):
            extract_json("{broken json}")


# ---------------------------------------------------------------------------
# _RateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_does_not_raise(self):
        limiter = _RateLimiter(min_interval_seconds=0.0)
        await limiter.acquire()
        await limiter.acquire()

    @pytest.mark.asyncio
    async def test_enforces_interval(self):
        import time

        limiter = _RateLimiter(min_interval_seconds=0.05)
        await limiter.acquire()
        t0 = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - t0
        # Should have waited at least 0.04 s (allow small margin)
        assert elapsed >= 0.04


# ---------------------------------------------------------------------------
# LLMClient.call — retry logic via mocked _dispatch
# ---------------------------------------------------------------------------


class TestLLMClientRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        client = LLMClient(provider="openai", model="gpt-4", api_key="sk-test", max_retries=2)
        client._dispatch = AsyncMock(return_value="ok")
        result = await client.call("hello")
        assert result == "ok"
        assert client._dispatch.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        client = LLMClient(
            provider="openai", model="gpt-4", api_key="sk-test",
            max_retries=2, min_interval_seconds=0.0,
        )
        client._dispatch = AsyncMock(
            side_effect=[LLMError("transient"), LLMError("transient"), "ok"]
        )
        result = await client.call("hello")
        assert result == "ok"
        assert client._dispatch.call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries_and_raises(self):
        client = LLMClient(
            provider="openai", model="gpt-4", api_key="sk-test",
            max_retries=1, min_interval_seconds=0.0,
        )
        client._dispatch = AsyncMock(side_effect=LLMError("persistent"))
        with pytest.raises(LLMError, match="persistent"):
            await client.call("hello")
        assert client._dispatch.call_count == 2  # 1 attempt + 1 retry

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self):
        client = LLMClient(
            provider="openai", model="gpt-4", api_key="sk-test",
            max_retries=2, min_interval_seconds=0.0,
        )
        client._dispatch = AsyncMock(
            side_effect=[LLMRateLimitError("429"), "success"]
        )
        result = await client.call("hello")
        assert result == "success"

    @pytest.mark.asyncio
    async def test_rate_limit_exhausted_raises(self):
        client = LLMClient(
            provider="openai", model="gpt-4", api_key="sk-test",
            max_retries=0, min_interval_seconds=0.0,
        )
        client._dispatch = AsyncMock(side_effect=LLMRateLimitError("429"))
        with pytest.raises(LLMRateLimitError):
            await client.call("hello")


# ---------------------------------------------------------------------------
# OrchestratorAgent._rule_based_routing
# ---------------------------------------------------------------------------


class TestRuleBasedRouting:
    def setup_method(self):
        self.agent = OrchestratorAgent()

    def test_drift_keywords(self):
        result = self.agent._rule_based_routing("check drift in the batting model")
        assert result["target_agent"] == "model_monitor"

    def test_anomaly_keywords(self):
        result = self.agent._rule_based_routing("detect anomalies in statcast data")
        assert result["target_agent"] == "data_quality"

    def test_feature_keywords(self):
        result = self.agent._rule_based_routing("run genetic feature engineering search")
        assert result["target_agent"] == "feature_engineer"

    def test_explain_keywords(self):
        result = self.agent._rule_based_routing("explain this prediction with shap values")
        assert result["target_agent"] == "explainer"

    def test_default_fallback(self):
        result = self.agent._rule_based_routing("unrecognized request xyz")
        assert result["target_agent"] == "data_quality"

    def test_returns_required_keys(self):
        result = self.agent._rule_based_routing("check data quality")
        assert all(k in result for k in ("target_agent", "task_type", "parameters", "reasoning"))


# ---------------------------------------------------------------------------
# OrchestratorAgent._rule_based_conflict_resolution
# ---------------------------------------------------------------------------


class TestRuleBasedConflictResolution:
    def setup_method(self):
        self.agent = OrchestratorAgent()

    def test_prefers_model_monitor(self):
        recs = [
            {"agent_id": "data_quality", "recommendation": "do a"},
            {"agent_id": "model_monitor", "recommendation": "do b"},
        ]
        result = self.agent._rule_based_conflict_resolution(recs)
        assert result["chosen_recommendation"] == "do b"

    def test_falls_back_to_first_on_unknown_agents(self):
        recs = [{"agent_id": "unknown_agent", "recommendation": "do c"}]
        result = self.agent._rule_based_conflict_resolution(recs)
        assert result["chosen_recommendation"] == "do c"

    def test_empty_recommendations(self):
        result = self.agent._rule_based_conflict_resolution([])
        assert result["resolution"] == "accepted"
        assert result["chosen_recommendation"] == ""

    def test_returns_required_keys(self):
        result = self.agent._rule_based_conflict_resolution([])
        assert all(k in result for k in ("resolution", "chosen_recommendation", "reasoning", "action_required"))


# ---------------------------------------------------------------------------
# handle_task: route_task
# ---------------------------------------------------------------------------


class TestRouteTask:
    @pytest.mark.asyncio
    async def test_llm_path_publishes_task(self, global_mq):
        routing_json = json.dumps({
            "target_agent": "data_quality",
            "task_type": "check_anomalies",
            "parameters": {"data_source": "statcast"},
            "reasoning": "Data quality check requested.",
        })
        agent = agent_with_mock_llm(routing_json)

        task = make_task("route_task", {"task_description": "check anomalies in statcast data"})
        result = await agent._route_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["routed_to"] == "data_quality"
        assert result.result_data["task_type"] == "check_anomalies"
        assert "routed_task_id" in result.result_data

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_rules(self, global_mq):
        agent = OrchestratorAgent()
        mock_llm = AsyncMock()
        mock_llm.call = AsyncMock(side_effect=LLMError("API down"))
        agent._llm = mock_llm
        agent._llm_available = True

        task = make_task("route_task", {"task_description": "detect feature drift"})
        result = await agent._route_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["routed_to"] in ("model_monitor", "data_quality")

    @pytest.mark.asyncio
    async def test_llm_unavailable_uses_rules(self, global_mq):
        agent = OrchestratorAgent()
        agent._llm_available = False

        task = make_task("route_task", {"task_description": "explain prediction for player A"})
        result = await agent._route_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["routed_to"] == "explainer"


# ---------------------------------------------------------------------------
# handle_task: system_health
# ---------------------------------------------------------------------------


class TestSystemHealth:
    @pytest.mark.asyncio
    async def test_returns_system_status(self, global_mq):
        agent = OrchestratorAgent()
        task = make_task("system_health")
        result = await agent._system_health(task)

        assert result.status == TaskStatus.COMPLETED
        data = result.result_data
        assert "all_agents_healthy" in data
        assert "agent_statuses" in data
        assert isinstance(data["agent_statuses"], list)

    @pytest.mark.asyncio
    async def test_orchestrator_always_healthy(self, global_mq):
        agent = OrchestratorAgent()
        task = make_task("system_health")
        result = await agent._system_health(task)

        orch_entry = next(
            s for s in result.result_data["agent_statuses"]
            if s["agent_id"] == "orchestrator"
        )
        assert orch_entry["is_healthy"] is True

    @pytest.mark.asyncio
    async def test_stale_agents_flagged(self, global_mq):
        """Agents with no heartbeat should appear unhealthy."""
        agent = OrchestratorAgent()
        # No heartbeats registered → all non-orchestrator agents are stale
        task = make_task("system_health", {"stale_threshold_seconds": 1})
        result = await agent._system_health(task)

        statuses = {s["agent_id"]: s for s in result.result_data["agent_statuses"]}
        assert statuses["data_quality"]["is_healthy"] is False
        assert statuses["model_monitor"]["is_healthy"] is False

    @pytest.mark.asyncio
    async def test_all_healthy_when_heartbeats_fresh(self, global_mq):
        """Register fresh heartbeats for all agents; all should be healthy."""
        from shared.messaging import message_queue

        for agent_type in AgentType:
            await message_queue.update_agent_heartbeat(agent_type.value)

        agent = OrchestratorAgent()
        task = make_task("system_health", {"stale_threshold_seconds": 300})
        result = await agent._system_health(task)

        assert result.result_data["all_agents_healthy"] is True

    @pytest.mark.asyncio
    async def test_queue_depth_in_metrics(self, global_mq):
        agent = OrchestratorAgent()
        task = make_task("system_health")
        result = await agent._system_health(task)
        assert "task_queue_depth" in result.metrics


# ---------------------------------------------------------------------------
# handle_task: resolve_conflict
# ---------------------------------------------------------------------------


class TestResolveConflict:
    @pytest.mark.asyncio
    async def test_llm_resolution(self, global_mq):
        resolution_json = json.dumps({
            "resolution": "accepted",
            "chosen_recommendation": "retrain the model",
            "reasoning": "Model performance is significantly degraded.",
            "action_required": True,
        })
        agent = agent_with_mock_llm(resolution_json)

        task = make_task(
            "resolve_conflict",
            {
                "conflict_description": "Agent A says retrain; Agent B says continue.",
                "recommendations": [
                    {"agent_id": "model_monitor", "recommendation": "retrain"},
                    {"agent_id": "data_quality", "recommendation": "continue"},
                ],
            },
        )
        result = await agent._resolve_conflict(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["resolution"] == "accepted"
        assert "chosen_recommendation" in result.result_data

    @pytest.mark.asyncio
    async def test_fallback_when_llm_unavailable(self, global_mq):
        agent = OrchestratorAgent()
        agent._llm_available = False

        task = make_task(
            "resolve_conflict",
            {
                "conflict_description": "Conflict",
                "recommendations": [
                    {"agent_id": "data_quality", "recommendation": "option A"},
                    {"agent_id": "model_monitor", "recommendation": "option B"},
                ],
            },
        )
        result = await agent._resolve_conflict(task)
        # model_monitor has higher priority in rule-based fallback
        assert result.result_data["chosen_recommendation"] == "option B"

    @pytest.mark.asyncio
    async def test_empty_recommendations(self, global_mq):
        agent = OrchestratorAgent()
        agent._llm_available = False

        task = make_task("resolve_conflict", {"conflict_description": "No one agrees", "recommendations": []})
        result = await agent._resolve_conflict(task)
        assert result.status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# handle_task: retrain_model
# ---------------------------------------------------------------------------


class TestRetrainModel:
    @pytest.mark.asyncio
    async def test_publishes_dq_preflight(self, global_mq):
        agent = OrchestratorAgent()
        task = make_task(
            "retrain_model",
            {
                "model_name": "batting_model",
                "reason": "drift detected",
                "triggered_by": "model_monitor",
            },
        )
        result = await agent._retrain_model(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["model_name"] == "batting_model"
        assert "dq_preflight_task_id" in result.result_data
        assert result.result_data["status"] == "retraining_workflow_initiated"


# ---------------------------------------------------------------------------
# handle_task: unknown task type
# ---------------------------------------------------------------------------


class TestHandleTaskDispatch:
    @pytest.mark.asyncio
    async def test_unknown_task_raises(self):
        agent = OrchestratorAgent()
        task = make_task("nonexistent_task")
        with pytest.raises(ValueError, match="Unknown task type"):
            await agent.handle_task(task)

    @pytest.mark.asyncio
    async def test_known_tasks_dispatch(self, global_mq):
        agent = OrchestratorAgent()
        agent._llm_available = False
        # system_health should work with no LLM
        task = make_task("system_health")
        result = await agent.handle_task(task)
        assert result.status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Prompt builders (sync helpers)
# ---------------------------------------------------------------------------


class TestPromptBuilders:
    def test_routing_prompt_contains_description(self):
        agent = OrchestratorAgent()
        prompt = agent.build_routing_prompt("check drift", {"model": "batting"})
        assert "check drift" in prompt
        assert "batting" in prompt

    def test_conflict_prompt_contains_conflict(self):
        agent = OrchestratorAgent()
        recs = [{"agent_id": "model_monitor", "recommendation": "retrain"}]
        prompt = agent.build_conflict_prompt("agents disagree", recs)
        assert "agents disagree" in prompt
        assert "model_monitor" in prompt
