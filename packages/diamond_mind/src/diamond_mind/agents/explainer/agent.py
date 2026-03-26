"""
Explainer Agent

Generates human-readable explanations for model predictions using SHAP
and LLMs.  Supports explanation caching via Redis, counterfactual generation,
and batch explanations.

Supported task types
--------------------
explain_prediction  — explain a single prediction
explain_batch       — explain a list of predictions
get_cached          — retrieve a previously cached explanation
clear_cache         — remove one or all cached explanations
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from shared.base_agent import BaseAgent
from shared import AgentType, AgentTask, AgentResult, TaskStatus, AlertSeverity
from shared.schemas import ConfidenceLevel, PredictionExplanation
from shared import message_queue, settings

from agents.orchestrator.llm_client import LLMClient, LLMConfigError

from .shap_utils import (
    shap_available,
    create_tree_explainer,
    compute_shap_values,
    get_top_features,
    generate_counterfactuals,
    features_to_array,
)


# Redis key prefix and TTL for cached explanations
_CACHE_PREFIX = "DM:EXPLANATION:"
_CACHE_TTL_SECONDS = 3600  # 1 hour


class ExplainerAgent(BaseAgent):
    """
    Agent that explains model predictions via SHAP + LLM narratives.

    Lifecycle
    ---------
    - initialize()   : set up LLM client (if enabled) and log SHAP availability
    - cleanup()      : release any held resources
    - handle_task()  : dispatch to per-task handlers

    Task handlers
    -------------
    - _explain_prediction  : single-instance explanation
    - _explain_batch       : multi-instance explanation
    - _get_cached          : retrieve cached explanation from Redis
    - _clear_cache         : evict one or all cached explanations
    """

    def __init__(self):
        super().__init__(AgentType.EXPLAINER)

        # Lazily loaded models: path string → model object
        self._model_cache: Dict[str, Any] = {}

        # Lazily created SHAP explainers: model path → explainer
        self._explainer_cache: Dict[str, Any] = {}

        # LLM client (initialised in initialize() if enabled)
        self._llm_client: Optional[LLMClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        self.logger.info(
            "Explainer Agent initializing. "
            f"SHAP available: {shap_available()}, "
            f"LLM enabled: {settings.explainer_use_llm}"
        )

        if settings.explainer_use_llm:
            try:
                self._llm_client = LLMClient.from_settings(settings)
                self.logger.info(
                    f"LLM client ready: {settings.llm_provider}/{settings.llm_model}"
                )
            except LLMConfigError as exc:
                self.logger.warning(
                    f"LLM client not available ({exc}). "
                    "Narratives will use template-based fallback."
                )
                self._llm_client = None

        if not shap_available():
            self.logger.warning(
                "shap library not installed. SHAP-based explanations will be "
                "skipped; only feature-weight approximations will be used."
            )

    async def cleanup(self):
        self._model_cache.clear()
        self._explainer_cache.clear()
        self.logger.info("Explainer Agent cleaned up")

    # ------------------------------------------------------------------
    # Task dispatch
    # ------------------------------------------------------------------

    async def handle_task(self, task: AgentTask) -> AgentResult:
        handlers = {
            "explain_prediction": self._explain_prediction,
            "explain_batch": self._explain_batch,
            "get_cached": self._get_cached,
            "clear_cache": self._clear_cache,
        }
        handler = handlers.get(task.task_type)
        if handler is None:
            raise ValueError(f"Unknown task type: {task.task_type!r}")
        return await handler(task)

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------

    async def _explain_prediction(self, task: AgentTask) -> AgentResult:
        """Generate a full explanation for one prediction."""
        p = task.parameters

        prediction_id: str = p["prediction_id"]
        player_name: str = p["player_name"]
        features: Dict[str, float] = p["features"]
        feature_names: List[str] = p.get("feature_names") or sorted(features.keys())
        predicted_value: float = float(p["predicted_value"])
        confidence_str: str = p.get("confidence", "medium")
        model_path: Optional[str] = p.get("model_path")
        top_n: int = int(p.get("top_n_features", 5))

        # Check cache first
        cached = await self._load_from_cache(prediction_id)
        if cached is not None:
            self.logger.info(f"Cache hit for prediction {prediction_id}")
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result_data={"explanation": cached, "cache_hit": True},
                metrics={"cache_hit": 1.0},
                duration_seconds=0.0,
            )

        # Compute SHAP values
        top_features, counterfactuals = self._compute_shap_and_counterfactuals(
            model_path=model_path,
            features=features,
            feature_names=feature_names,
            predicted_value=predicted_value,
            top_n=top_n,
        )

        # Generate narrative
        narrative = await self._generate_narrative(
            player_name=player_name,
            predicted_value=predicted_value,
            top_features=top_features,
        )

        # Build PredictionExplanation
        try:
            confidence = ConfidenceLevel(confidence_str)
        except ValueError:
            confidence = ConfidenceLevel.MEDIUM

        explanation = PredictionExplanation(
            prediction_id=prediction_id,
            player_name=player_name,
            predicted_value=predicted_value,
            confidence=confidence,
            top_features=top_features,
            narrative_explanation=narrative,
            counterfactuals=counterfactuals,
        )

        explanation_dict = explanation.model_dump(mode="json")

        # Store in cache
        await self._save_to_cache(prediction_id, explanation_dict)

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={"explanation": explanation_dict, "cache_hit": False},
            metrics={
                "top_features_count": float(len(top_features)),
                "counterfactuals_count": float(len(counterfactuals)),
            },
            duration_seconds=0.0,
        )

    async def _explain_batch(self, task: AgentTask) -> AgentResult:
        """Explain a list of predictions."""
        p = task.parameters
        model_path: Optional[str] = p.get("model_path")
        predictions: List[Dict[str, Any]] = p["predictions"]
        top_n: int = int(p.get("top_n_features", 5))

        explanations = []
        cache_hits = 0

        for pred in predictions:
            prediction_id = pred["prediction_id"]

            cached = await self._load_from_cache(prediction_id)
            if cached is not None:
                explanations.append({"prediction_id": prediction_id, **cached, "cache_hit": True})
                cache_hits += 1
                continue

            features: Dict[str, float] = pred["features"]
            feature_names: List[str] = pred.get("feature_names") or sorted(features.keys())
            predicted_value = float(pred["predicted_value"])
            player_name: str = pred["player_name"]
            confidence_str: str = pred.get("confidence", "medium")

            top_features, counterfactuals = self._compute_shap_and_counterfactuals(
                model_path=model_path,
                features=features,
                feature_names=feature_names,
                predicted_value=predicted_value,
                top_n=top_n,
            )

            narrative = await self._generate_narrative(
                player_name=player_name,
                predicted_value=predicted_value,
                top_features=top_features,
            )

            try:
                confidence = ConfidenceLevel(confidence_str)
            except ValueError:
                confidence = ConfidenceLevel.MEDIUM

            explanation = PredictionExplanation(
                prediction_id=prediction_id,
                player_name=player_name,
                predicted_value=predicted_value,
                confidence=confidence,
                top_features=top_features,
                narrative_explanation=narrative,
                counterfactuals=counterfactuals,
            )
            explanation_dict = explanation.model_dump(mode="json")
            await self._save_to_cache(prediction_id, explanation_dict)
            explanations.append({**explanation_dict, "cache_hit": False})

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "explanations": explanations,
                "total": len(explanations),
                "cache_hits": cache_hits,
            },
            metrics={
                "total_explained": float(len(explanations)),
                "cache_hits": float(cache_hits),
            },
            duration_seconds=0.0,
        )

    async def _get_cached(self, task: AgentTask) -> AgentResult:
        """Retrieve a cached explanation without recomputing."""
        prediction_id: str = task.parameters["prediction_id"]
        cached = await self._load_from_cache(prediction_id)

        if cached is None:
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result_data={"found": False, "prediction_id": prediction_id},
                metrics={"cache_hit": 0.0},
                duration_seconds=0.0,
            )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={"found": True, "explanation": cached},
            metrics={"cache_hit": 1.0},
            duration_seconds=0.0,
        )

    async def _clear_cache(self, task: AgentTask) -> AgentResult:
        """Evict one or all cached explanations from Redis."""
        prediction_id: Optional[str] = task.parameters.get("prediction_id")

        if prediction_id:
            key = _CACHE_PREFIX + prediction_id
            deleted = await self._redis_delete(key)
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result_data={"deleted": deleted, "prediction_id": prediction_id},
                metrics={"deleted_count": float(deleted)},
                duration_seconds=0.0,
            )

        # Clear all explanation keys
        deleted = await self._redis_delete_pattern(_CACHE_PREFIX + "*")
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={"deleted": deleted, "prediction_id": None},
            metrics={"deleted_count": float(deleted)},
            duration_seconds=0.0,
        )

    # ------------------------------------------------------------------
    # SHAP / counterfactual helpers
    # ------------------------------------------------------------------

    def _compute_shap_and_counterfactuals(
        self,
        model_path: Optional[str],
        features: Dict[str, float],
        feature_names: List[str],
        predicted_value: float,
        top_n: int,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Compute top SHAP features and counterfactuals for one instance.

        Falls back to feature-weight approximation if shap is unavailable
        or no model path is provided.
        """
        instance = features_to_array(features, feature_names)

        if shap_available() and model_path:
            model = self._load_model(model_path)
            explainer = self._get_explainer(model_path, model)

            try:
                X = instance.reshape(1, -1)
                shap_dicts = compute_shap_values(explainer, X, feature_names)
                top_features = get_top_features(shap_dicts[0], n=top_n)

                # Predicted class index — use rounded value for tree models
                predicted_class = int(round(predicted_value))
                counterfactuals = generate_counterfactuals(
                    model=model,
                    instance=instance,
                    feature_names=feature_names,
                    predicted_class=predicted_class,
                )
                return top_features, counterfactuals
            except Exception as exc:
                self.logger.warning(
                    f"SHAP computation failed ({exc}); using fallback attribution."
                )

        # Fallback: use raw feature values as proxy importance
        top_features = self._fallback_top_features(features, feature_names, top_n)
        return top_features, []

    def _fallback_top_features(
        self,
        features: Dict[str, float],
        feature_names: List[str],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """
        Simple fallback when SHAP is unavailable.

        Ranks features by their absolute normalised value.
        """
        values = np.array([features[f] for f in feature_names], dtype=float)
        norm = np.abs(values)
        total = norm.sum() or 1.0
        importance = norm / total

        ranked = sorted(
            zip(feature_names, values, importance),
            key=lambda t: t[2],
            reverse=True,
        )[:top_n]

        return [
            {
                "feature": name,
                "shap_value": round(float(val), 6),
                "direction": "positive" if val >= 0 else "negative",
            }
            for name, val, _ in ranked
        ]

    # ------------------------------------------------------------------
    # Narrative generation
    # ------------------------------------------------------------------

    async def _generate_narrative(
        self,
        player_name: str,
        predicted_value: float,
        top_features: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a human-readable explanation.

        Uses LLM when available and enabled, otherwise falls back to a
        structured template.
        """
        if self._llm_client is not None:
            return await self._llm_narrative(player_name, predicted_value, top_features)
        return self._template_narrative(player_name, predicted_value, top_features)

    async def _llm_narrative(
        self,
        player_name: str,
        predicted_value: float,
        top_features: List[Dict[str, Any]],
    ) -> str:
        """Call the LLM to produce a narrative explanation."""
        feature_lines = "\n".join(
            f"- {f['feature']}: SHAP={f['shap_value']:.4f} ({f['direction']})"
            for f in top_features
        )
        prompt = (
            f"You are a baseball analytics assistant. Explain the following "
            f"model prediction in 2-3 concise sentences for a fantasy baseball user.\n\n"
            f"Player: {player_name}\n"
            f"Predicted value: {predicted_value:.3f}\n"
            f"Top contributing factors:\n{feature_lines}\n\n"
            f"Explanation:"
        )
        try:
            return await self._llm_client.call(prompt)
        except Exception as exc:
            self.logger.warning(f"LLM narrative failed ({exc}); using template.")
            return self._template_narrative(player_name, predicted_value, top_features)

    def _template_narrative(
        self,
        player_name: str,
        predicted_value: float,
        top_features: List[Dict[str, Any]],
    ) -> str:
        """Produce a deterministic template-based narrative."""
        if not top_features:
            return (
                f"The model predicted {predicted_value:.3f} for {player_name} "
                f"based on the available features."
            )

        top = top_features[0]
        direction_word = "positively" if top["direction"] == "positive" else "negatively"
        lines = [
            f"For {player_name}, the model predicted {predicted_value:.3f}.",
            f"The most influential factor was '{top['feature']}', "
            f"which {direction_word} contributed to this prediction.",
        ]

        if len(top_features) > 1:
            others = ", ".join(f"'{f['feature']}'" for f in top_features[1:3])
            lines.append(f"Other notable factors included {others}.")

        return " ".join(lines)

    # ------------------------------------------------------------------
    # Model / explainer caching
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> Any:
        """Load and cache a pickled model from disk."""
        if model_path not in self._model_cache:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            with open(path, "rb") as fh:
                self._model_cache[model_path] = pickle.load(fh)
            self.logger.info(f"Loaded model from {model_path}")
        return self._model_cache[model_path]

    def _get_explainer(self, model_path: str, model: Any) -> Any:
        """Return (or create and cache) a SHAP TreeExplainer for the model."""
        if model_path not in self._explainer_cache:
            self._explainer_cache[model_path] = create_tree_explainer(model)
            self.logger.info(f"Created SHAP explainer for {model_path}")
        return self._explainer_cache[model_path]

    # ------------------------------------------------------------------
    # Redis caching helpers
    # ------------------------------------------------------------------

    async def _load_from_cache(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Return the cached explanation dict, or None if not found."""
        redis = message_queue.redis_client
        if redis is None:
            return None
        key = _CACHE_PREFIX + prediction_id
        try:
            raw = await redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            self.logger.warning(f"Cache read error for {prediction_id}: {exc}")
            return None

    async def _save_to_cache(
        self, prediction_id: str, explanation: Dict[str, Any]
    ) -> None:
        """Persist an explanation dict in Redis with TTL."""
        redis = message_queue.redis_client
        if redis is None:
            return
        key = _CACHE_PREFIX + prediction_id
        try:
            await redis.setex(key, _CACHE_TTL_SECONDS, json.dumps(explanation))
        except Exception as exc:
            self.logger.warning(f"Cache write error for {prediction_id}: {exc}")

    async def _redis_delete(self, key: str) -> int:
        """Delete a single Redis key; returns 1 if deleted, 0 otherwise."""
        redis = message_queue.redis_client
        if redis is None:
            return 0
        try:
            return int(await redis.delete(key))
        except Exception as exc:
            self.logger.warning(f"Cache delete error for key {key}: {exc}")
            return 0

    async def _redis_delete_pattern(self, pattern: str) -> int:
        """Delete all Redis keys matching pattern; returns count deleted."""
        redis = message_queue.redis_client
        if redis is None:
            return 0
        try:
            keys = await redis.keys(pattern)
            if not keys:
                return 0
            return int(await redis.delete(*keys))
        except Exception as exc:
            self.logger.warning(f"Cache bulk delete error (pattern={pattern}): {exc}")
            return 0


if __name__ == "__main__":
    import asyncio
    from shared.messaging import init_messaging, shutdown_messaging

    async def main():
        await init_messaging()
        agent = ExplainerAgent()
        try:
            await agent.start()
        except KeyboardInterrupt:
            await agent.stop()
        finally:
            await shutdown_messaging()

    asyncio.run(main())
