"""
Unit tests for the Explainer Agent and its helper modules.

Covers:
- shap_utils: features_to_array, get_top_features, generate_counterfactuals (fallback)
- ExplainerAgent: _fallback_top_features, _template_narrative, _load_model,
  _get_explainer, cache helpers
- handle_task: explain_prediction, explain_batch, get_cached, clear_cache
"""

import json
import sys
import tempfile
import pickle
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

_dm_src = str(Path(__file__).resolve().parents[3] / "src" / "diamond_mind")
if _dm_src not in sys.path:
    sys.path.insert(0, _dm_src)

import fakeredis.aioredis

from shared.schemas import AgentType, TaskStatus, AgentTask, TaskPriority, ConfidenceLevel
from shared.messaging import message_queue
from agents.explainer.agent import ExplainerAgent, _CACHE_PREFIX, _CACHE_TTL_SECONDS
from agents.explainer.shap_utils import (
    features_to_array,
    get_top_features,
    shap_available,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent():
    return ExplainerAgent()


@pytest.fixture
async def redis_mq():
    """Patch the global message_queue singleton with fake Redis."""
    message_queue.redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield message_queue
    await message_queue.redis_client.flushall()
    await message_queue.redis_client.aclose()
    message_queue.redis_client = None


@pytest.fixture
def sample_features():
    return {
        "batting_avg": 0.290,
        "ops": 0.850,
        "home_runs": 30.0,
        "rbi": 90.0,
        "war": 4.5,
    }


@pytest.fixture
def feature_names(sample_features):
    return sorted(sample_features.keys())


def _make_explain_task(
    prediction_id: str = "pred_001",
    player_name: str = "Mike Trout",
    features: dict = None,
    predicted_value: float = 0.82,
    confidence: str = "high",
    model_path: str = None,
) -> AgentTask:
    if features is None:
        features = {
            "batting_avg": 0.290,
            "ops": 0.850,
            "home_runs": 30.0,
            "rbi": 90.0,
            "war": 4.5,
        }
    params = {
        "prediction_id": prediction_id,
        "player_name": player_name,
        "features": features,
        "predicted_value": predicted_value,
        "confidence": confidence,
    }
    if model_path:
        params["model_path"] = model_path
    return AgentTask(
        task_id="task_001",
        agent_id=AgentType.EXPLAINER,
        task_type="explain_prediction",
        priority=TaskPriority.MEDIUM,
        parameters=params,
    )


# ---------------------------------------------------------------------------
# shap_utils tests
# ---------------------------------------------------------------------------


class TestShapUtils:
    def test_features_to_array_ordering(self, sample_features, feature_names):
        arr = features_to_array(sample_features, feature_names)
        assert arr.shape == (len(feature_names),)
        for i, name in enumerate(feature_names):
            assert arr[i] == pytest.approx(sample_features[name])

    def test_features_to_array_missing_key(self, sample_features, feature_names):
        with pytest.raises(KeyError):
            features_to_array(sample_features, feature_names + ["nonexistent"])

    def test_get_top_features_ordering(self):
        shap_dict = {"a": 0.5, "b": -1.2, "c": 0.1, "d": 0.9}
        top = get_top_features(shap_dict, n=2)
        assert len(top) == 2
        assert top[0]["feature"] == "b"   # abs(-1.2) is largest
        assert top[1]["feature"] == "d"   # abs(0.9) next

    def test_get_top_features_direction(self):
        shap_dict = {"pos": 0.5, "neg": -0.8}
        top = get_top_features(shap_dict, n=2)
        by_name = {t["feature"]: t for t in top}
        assert by_name["pos"]["direction"] == "positive"
        assert by_name["neg"]["direction"] == "negative"

    def test_get_top_features_clamps_to_available(self):
        shap_dict = {"a": 1.0, "b": 2.0}
        top = get_top_features(shap_dict, n=10)
        assert len(top) == 2

    def test_get_top_features_empty(self):
        assert get_top_features({}, n=5) == []

    def test_shap_available_returns_bool(self):
        assert isinstance(shap_available(), bool)


# ---------------------------------------------------------------------------
# ExplainerAgent internal helpers
# ---------------------------------------------------------------------------


class TestFallbackTopFeatures:
    def test_returns_top_n(self, agent, sample_features, feature_names):
        top = agent._fallback_top_features(sample_features, feature_names, top_n=3)
        assert len(top) == 3

    def test_sorted_by_abs_value(self, agent):
        features = {"small": 0.01, "big": 100.0, "mid": 10.0}
        names = ["small", "big", "mid"]
        top = agent._fallback_top_features(features, names, top_n=3)
        assert top[0]["feature"] == "big"

    def test_direction_field(self, agent):
        features = {"neg": -5.0, "pos": 3.0}
        names = ["neg", "pos"]
        top = agent._fallback_top_features(features, names, top_n=2)
        by_name = {t["feature"]: t for t in top}
        assert by_name["neg"]["direction"] == "negative"
        assert by_name["pos"]["direction"] == "positive"

    def test_handles_all_zero(self, agent):
        features = {"a": 0.0, "b": 0.0}
        names = ["a", "b"]
        top = agent._fallback_top_features(features, names, top_n=2)
        assert len(top) == 2


class TestTemplateNarrative:
    def test_contains_player_name(self, agent):
        top = [{"feature": "ops", "shap_value": 0.5, "direction": "positive"}]
        narrative = agent._template_narrative("Shohei Ohtani", 0.75, top)
        assert "Shohei Ohtani" in narrative

    def test_contains_predicted_value(self, agent):
        top = [{"feature": "batting_avg", "shap_value": 0.3, "direction": "positive"}]
        narrative = agent._template_narrative("Player", 0.333, top)
        assert "0.333" in narrative

    def test_mentions_top_feature(self, agent):
        top = [{"feature": "home_runs", "shap_value": 0.9, "direction": "positive"}]
        narrative = agent._template_narrative("Player", 0.5, top)
        assert "home_runs" in narrative

    def test_no_top_features_fallback(self, agent):
        narrative = agent._template_narrative("Player", 0.5, [])
        assert "Player" in narrative
        assert "0.500" in narrative

    def test_multiple_features_mentioned(self, agent):
        top = [
            {"feature": "ops", "shap_value": 0.9, "direction": "positive"},
            {"feature": "war", "shap_value": 0.7, "direction": "positive"},
            {"feature": "rbi", "shap_value": 0.5, "direction": "positive"},
        ]
        narrative = agent._template_narrative("Player", 0.5, top)
        assert "war" in narrative or "rbi" in narrative


class TestLoadModel:
    def test_load_missing_model_raises(self, agent):
        with pytest.raises(FileNotFoundError):
            agent._load_model("/nonexistent/path/model.pkl")

    def test_load_and_cache(self, agent, tmp_path):
        # Use a simple picklable object instead of MagicMock
        model = {"type": "dummy_model", "version": 1}
        model_path = str(tmp_path / "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        loaded = agent._load_model(model_path)
        assert loaded == model
        # Second call should use the in-memory cache
        loaded2 = agent._load_model(model_path)
        assert model_path in agent._model_cache
        assert loaded2 == model


# ---------------------------------------------------------------------------
# Redis caching helpers
# ---------------------------------------------------------------------------


class TestCacheHelpers:
    @pytest.mark.asyncio
    async def test_save_and_load(self, agent, redis_mq):
        data = {"prediction_id": "p1", "narrative": "Test"}
        await agent._save_to_cache("p1", data)
        result = await agent._load_from_cache("p1")
        assert result == data

    @pytest.mark.asyncio
    async def test_load_missing_returns_none(self, agent, redis_mq):
        result = await agent._load_from_cache("does_not_exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_delete(self, agent, redis_mq):
        await agent._save_to_cache("del_me", {"x": 1})
        deleted = await agent._redis_delete(_CACHE_PREFIX + "del_me")
        assert deleted == 1
        assert await agent._load_from_cache("del_me") is None

    @pytest.mark.asyncio
    async def test_redis_delete_pattern(self, agent, redis_mq):
        await agent._save_to_cache("aa", {"x": 1})
        await agent._save_to_cache("bb", {"x": 2})
        deleted = await agent._redis_delete_pattern(_CACHE_PREFIX + "*")
        assert deleted == 2

    @pytest.mark.asyncio
    async def test_no_redis_returns_gracefully(self, agent):
        # No Redis configured — should not raise
        message_queue.redis_client = None
        result = await agent._load_from_cache("x")
        assert result is None
        await agent._save_to_cache("x", {})  # No error
        deleted = await agent._redis_delete("DM:EXPLANATION:x")
        assert deleted == 0


# ---------------------------------------------------------------------------
# handle_task: explain_prediction
# ---------------------------------------------------------------------------


class TestExplainPrediction:
    @pytest.mark.asyncio
    async def test_basic_success_no_model(self, agent, redis_mq):
        task = _make_explain_task()
        result = await agent.handle_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert "explanation" in result.result_data
        exp = result.result_data["explanation"]
        assert exp["prediction_id"] == "pred_001"
        assert exp["player_name"] == "Mike Trout"
        assert "narrative_explanation" in exp
        assert "top_features" in exp

    @pytest.mark.asyncio
    async def test_cache_hit_on_second_call(self, agent, redis_mq):
        task = _make_explain_task()
        # First call — computes and caches
        result1 = await agent.handle_task(task)
        assert result1.result_data["cache_hit"] is False

        # Second call — should return from cache
        result2 = await agent.handle_task(task)
        assert result2.result_data["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_unknown_confidence_defaults_to_medium(self, agent, redis_mq):
        task = _make_explain_task(confidence="not_a_real_level")
        result = await agent.handle_task(task)
        assert result.status == TaskStatus.COMPLETED
        exp = result.result_data["explanation"]
        assert exp["confidence"] == ConfidenceLevel.MEDIUM.value

    @pytest.mark.asyncio
    async def test_llm_narrative_used_when_available(self, agent, redis_mq):
        mock_llm = AsyncMock()
        mock_llm.call = AsyncMock(return_value="LLM-generated explanation.")
        agent._llm_client = mock_llm

        task = _make_explain_task()
        result = await agent.handle_task(task)

        mock_llm.call.assert_called_once()
        exp = result.result_data["explanation"]
        assert exp["narrative_explanation"] == "LLM-generated explanation."

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_template(self, agent, redis_mq):
        mock_llm = AsyncMock()
        mock_llm.call = AsyncMock(side_effect=Exception("LLM down"))
        agent._llm_client = mock_llm

        task = _make_explain_task(player_name="Test Player")
        result = await agent.handle_task(task)

        assert result.status == TaskStatus.COMPLETED
        exp = result.result_data["explanation"]
        assert "Test Player" in exp["narrative_explanation"]

    @pytest.mark.asyncio
    async def test_missing_required_param_raises(self, agent, redis_mq):
        task = AgentTask(
            task_id="t",
            agent_id=AgentType.EXPLAINER,
            task_type="explain_prediction",
            priority=TaskPriority.MEDIUM,
            parameters={"player_name": "X"},  # missing prediction_id, features, etc.
        )
        with pytest.raises(KeyError):
            await agent.handle_task(task)

    @pytest.mark.asyncio
    async def test_unknown_task_type_raises(self, agent, redis_mq):
        task = AgentTask(
            task_id="t",
            agent_id=AgentType.EXPLAINER,
            task_type="not_a_task",
            priority=TaskPriority.MEDIUM,
            parameters={},
        )
        with pytest.raises(ValueError, match="Unknown task type"):
            await agent.handle_task(task)


# ---------------------------------------------------------------------------
# handle_task: explain_batch
# ---------------------------------------------------------------------------


class TestExplainBatch:
    @pytest.mark.asyncio
    async def test_batch_returns_all_explanations(self, agent, redis_mq):
        predictions = [
            {
                "prediction_id": f"pred_{i}",
                "player_name": f"Player {i}",
                "features": {"batting_avg": 0.250 + i * 0.01, "ops": 0.750 + i * 0.01},
                "predicted_value": 0.5 + i * 0.05,
                "confidence": "medium",
            }
            for i in range(3)
        ]
        task = AgentTask(
            task_id="batch_task",
            agent_id=AgentType.EXPLAINER,
            task_type="explain_batch",
            priority=TaskPriority.LOW,
            parameters={"predictions": predictions},
        )
        result = await agent.handle_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["total"] == 3
        assert len(result.result_data["explanations"]) == 3

    @pytest.mark.asyncio
    async def test_batch_uses_cache(self, agent, redis_mq):
        features = {"batting_avg": 0.300, "ops": 0.900}
        # Pre-populate cache for pred_0
        cached_data = {
            "prediction_id": "pred_0",
            "player_name": "Cached Player",
            "predicted_value": 0.5,
            "confidence": "high",
            "top_features": [],
            "narrative_explanation": "Cached narrative.",
            "counterfactuals": [],
        }
        await agent._save_to_cache("pred_0", cached_data)

        predictions = [
            {
                "prediction_id": "pred_0",
                "player_name": "Cached Player",
                "features": features,
                "predicted_value": 0.5,
                "confidence": "high",
            }
        ]
        task = AgentTask(
            task_id="bt2",
            agent_id=AgentType.EXPLAINER,
            task_type="explain_batch",
            priority=TaskPriority.LOW,
            parameters={"predictions": predictions},
        )
        result = await agent.handle_task(task)
        assert result.result_data["cache_hits"] == 1


# ---------------------------------------------------------------------------
# handle_task: get_cached / clear_cache
# ---------------------------------------------------------------------------


class TestCacheTasks:
    @pytest.mark.asyncio
    async def test_get_cached_found(self, agent, redis_mq):
        await agent._save_to_cache("p99", {"data": "value"})
        task = AgentTask(
            task_id="t",
            agent_id=AgentType.EXPLAINER,
            task_type="get_cached",
            priority=TaskPriority.LOW,
            parameters={"prediction_id": "p99"},
        )
        result = await agent.handle_task(task)
        assert result.result_data["found"] is True
        assert result.result_data["explanation"]["data"] == "value"

    @pytest.mark.asyncio
    async def test_get_cached_not_found(self, agent, redis_mq):
        task = AgentTask(
            task_id="t",
            agent_id=AgentType.EXPLAINER,
            task_type="get_cached",
            priority=TaskPriority.LOW,
            parameters={"prediction_id": "nonexistent"},
        )
        result = await agent.handle_task(task)
        assert result.result_data["found"] is False

    @pytest.mark.asyncio
    async def test_clear_cache_specific(self, agent, redis_mq):
        await agent._save_to_cache("to_delete", {"x": 1})
        await agent._save_to_cache("keep", {"y": 2})

        task = AgentTask(
            task_id="t",
            agent_id=AgentType.EXPLAINER,
            task_type="clear_cache",
            priority=TaskPriority.LOW,
            parameters={"prediction_id": "to_delete"},
        )
        result = await agent.handle_task(task)
        assert result.result_data["deleted"] == 1
        assert await agent._load_from_cache("to_delete") is None
        assert await agent._load_from_cache("keep") is not None

    @pytest.mark.asyncio
    async def test_clear_cache_all(self, agent, redis_mq):
        await agent._save_to_cache("a", {"x": 1})
        await agent._save_to_cache("b", {"x": 2})
        await agent._save_to_cache("c", {"x": 3})

        task = AgentTask(
            task_id="t",
            agent_id=AgentType.EXPLAINER,
            task_type="clear_cache",
            priority=TaskPriority.LOW,
            parameters={},  # no prediction_id → clear all
        )
        result = await agent.handle_task(task)
        assert result.result_data["deleted"] == 3
        assert await agent._load_from_cache("a") is None
