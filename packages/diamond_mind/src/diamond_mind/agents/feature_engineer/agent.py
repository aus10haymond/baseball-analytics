"""
Feature Engineer Agent

Discovers and evaluates new features for baseball analytics models using:
  - Genetic Algorithm evolution (automated search)
  - Domain-specific baseball feature generators
  - LLM-based feature suggestions (via Orchestrator's LLMClient)
  - Cross-validated fitness evaluation and VIF multicollinearity checks
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shared.base_agent import BaseAgent
from shared import AgentType, AgentTask, AgentResult, TaskStatus, AlertSeverity, TaskPriority
from shared.schemas import FeatureCandidate, FeatureSearchResult
from shared import settings

from .feature_generators import generate_all_candidates, generate_baseball_features
from .genetic_algorithm import FeatureGA, FeatureGene

# Reuse the LLMClient from the orchestrator module — no need to duplicate
from agents.orchestrator.llm_client import LLMClient, LLMError, LLMConfigError, extract_json


_LLM_FEATURE_SYSTEM_PROMPT = """\
You are a baseball analytics expert and data scientist.
Given a list of existing features in a dataset, suggest 5–10 new derived
features that could improve predictive performance for player outcome models.

For each suggestion respond with a JSON array (no markdown):
[
  {
    "feature_name": "<snake_case_name>",
    "feature_definition": "<formula or description>",
    "feature_type": "<rolling_avg|interaction|polynomial|lag|baseball>",
    "source_features": ["<col1>", "<col2>"],
    "reasoning": "<one sentence why this might help>"
  },
  ...
]
"""


class FeatureEngineerAgent(BaseAgent):
    """Agent that discovers and evaluates new ML features for baseball models."""

    def __init__(self):
        super().__init__(AgentType.FEATURE_ENGINEER)
        self._llm: Optional[LLMClient] = None
        self._llm_available: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        self.logger.info("Initializing Feature Engineer Agent")
        try:
            self._llm = LLMClient.from_settings(settings)
            self._llm_available = True
            self.logger.info("LLM client ready for feature suggestions")
        except LLMConfigError as exc:
            self._llm_available = False
            self.logger.warning(f"LLM unavailable: {exc}. Suggestions will be skipped.")

    async def cleanup(self):
        self.logger.info("Feature Engineer Agent cleaned up")

    # ------------------------------------------------------------------
    # Task dispatch
    # ------------------------------------------------------------------

    async def handle_task(self, task: AgentTask) -> AgentResult:
        handlers = {
            "search_features": self._search_features,
            "evaluate_feature": self._evaluate_feature,
            "generate_features": self._generate_features,
            "get_llm_suggestions": self._get_llm_suggestions,
        }
        handler = handlers.get(task.task_type)
        if handler is None:
            raise ValueError(f"Unknown task type: {task.task_type}")
        return await handler(task)

    # ------------------------------------------------------------------
    # 5.5  Feature Search (GA + LLM + evaluation)
    # ------------------------------------------------------------------

    async def _search_features(self, task: AgentTask) -> AgentResult:
        """
        Full automated feature search:
          1. Load data.
          2. Run Genetic Algorithm.
          3. (Optionally) get LLM suggestions.
          4. Evaluate top candidates.
          5. Return FeatureSearchResult.
        """
        data_source = task.parameters.get("data_source", "")
        target_col = task.parameters.get("target_col", "")
        population_size = task.parameters.get("population_size", 30)
        generations = task.parameters.get("generations", 10)
        top_k = task.parameters.get("top_k", 5)
        use_llm = task.parameters.get("use_llm", True) and self._llm_available
        search_id = task.parameters.get("search_id", task.task_id)

        df = self._load_data(data_source)
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col!r} not found in data.")

        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y = df[target_col]
        available_cols = X.columns.tolist()

        # Baseline score (model with no new features)
        baseline_score = self._baseline_cv_score(X, y)

        # ── Genetic Algorithm ──────────────────────────────────────────
        ga = FeatureGA(
            population_size=population_size,
            generations=generations,
            seed=42,
        )
        ga_results = ga.evolve(X, y, available_cols)  # list of (gene, fitness)

        # ── LLM Suggestions ───────────────────────────────────────────
        llm_candidates: List[FeatureCandidate] = []
        if use_llm:
            llm_candidates = await self._fetch_llm_suggestions(
                existing_features=available_cols, df=df
            )

        # ── Evaluate top GA genes ─────────────────────────────────────
        evaluated: List[FeatureCandidate] = []
        for gene, fitness in ga_results[:top_k * 2]:
            candidate = self._evaluate_single_gene(gene, X, y, baseline_score, fitness)
            if candidate is not None:
                evaluated.append(candidate)
            if len(evaluated) >= top_k:
                break

        # Merge LLM candidates (pre-evaluated via feature definition)
        all_candidates = evaluated + llm_candidates

        # Sort by importance_score descending, take top_k
        all_candidates.sort(
            key=lambda c: c.importance_score if c.importance_score is not None else -1.0,
            reverse=True,
        )
        selected = all_candidates[:top_k]

        best_score = max((c.importance_score or 0.0) for c in selected) if selected else baseline_score
        improvement_pct = (
            ((best_score - baseline_score) / max(abs(baseline_score), 1e-6)) * 100
            if baseline_score != best_score
            else 0.0
        )

        search_result = FeatureSearchResult(
            search_id=search_id,
            generation=generations,
            candidates_evaluated=len(ga_results) + len(llm_candidates),
            features_added=selected,
            best_model_score=round(best_score, 6),
            baseline_model_score=round(baseline_score, 6),
            improvement_pct=round(improvement_pct, 4),
            search_duration_seconds=0.0,
        )

        if selected:
            await self.publish_alert(
                severity=AlertSeverity.INFO,
                message=f"Feature search complete: {len(selected)} candidates found",
                details={
                    "search_id": search_id,
                    "best_score": best_score,
                    "baseline_score": baseline_score,
                    "improvement_pct": improvement_pct,
                    "top_features": [c.feature_name for c in selected],
                },
                requires_action=False,
            )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data=search_result.model_dump(mode="json"),
            metrics={
                "candidates_evaluated": float(search_result.candidates_evaluated),
                "features_added": float(len(selected)),
                "improvement_pct": float(improvement_pct),
                "baseline_score": float(baseline_score),
                "best_score": float(best_score),
            },
            duration_seconds=0.0,
        )

    # ------------------------------------------------------------------
    # 5.3  Feature Evaluation
    # ------------------------------------------------------------------

    async def _evaluate_feature(self, task: AgentTask) -> AgentResult:
        """
        Evaluate a single user-specified feature definition.

        Expected parameters:
          data_source, target_col,
          feature_name, gene_type, source_features, params
        """
        data_source = task.parameters.get("data_source", "")
        target_col = task.parameters.get("target_col", "")
        feature_name = task.parameters.get("feature_name", "custom_feature")
        gene_type = task.parameters.get("gene_type", "rolling_avg")
        source_features = task.parameters.get("source_features", [])
        params = task.parameters.get("params", {})

        df = self._load_data(data_source)
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col!r} not found in data.")

        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y = df[target_col]

        gene = FeatureGene(
            gene_type=gene_type,
            source_features=source_features,
            params=params,
        )
        # Override auto-generated name
        gene.gene_id = feature_name

        baseline_score = self._baseline_cv_score(X, y)
        candidate = self._evaluate_single_gene(gene, X, y, baseline_score)
        if candidate is None:
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result_data={
                    "feature_name": feature_name,
                    "validation_passed": False,
                    "reason": "Feature produced degenerate values (all NaN or constant).",
                },
                metrics={},
                duration_seconds=0.0,
            )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data=candidate.model_dump(mode="json"),
            metrics={
                "importance_score": candidate.importance_score or 0.0,
                "performance_gain": candidate.performance_gain or 0.0,
                "multicollinearity_vif": candidate.multicollinearity_vif or 1.0,
            },
            duration_seconds=0.0,
        )

    # ------------------------------------------------------------------
    # 5.2  Feature Generation (bulk, no GA/LLM)
    # ------------------------------------------------------------------

    async def _generate_features(self, task: AgentTask) -> AgentResult:
        """
        Generate feature candidates from all built-in generators.
        Does not run the GA or call the LLM — pure generator pass.
        """
        data_source = task.parameters.get("data_source", "")
        max_per_type = task.parameters.get("max_per_type", 15)
        include_baseball = task.parameters.get("include_baseball", True)

        df = self._load_data(data_source)
        candidates = generate_all_candidates(
            df, max_per_type=max_per_type, include_baseball=include_baseball
        )

        names = [name for name, _ in candidates]
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "feature_names": names,
                "total_generated": len(names),
            },
            metrics={"total_generated": float(len(names))},
            duration_seconds=0.0,
        )

    # ------------------------------------------------------------------
    # 5.4  LLM-Based Suggestions
    # ------------------------------------------------------------------

    async def _get_llm_suggestions(self, task: AgentTask) -> AgentResult:
        """
        Ask the LLM for feature suggestions given the existing columns.
        """
        data_source = task.parameters.get("data_source", "")
        df = self._load_data(data_source)
        existing = df.select_dtypes(include=[np.number]).columns.tolist()

        candidates = await self._fetch_llm_suggestions(existing_features=existing, df=df)

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "suggestions": [c.model_dump(mode="json") for c in candidates],
                "total": len(candidates),
            },
            metrics={"total_suggestions": float(len(candidates))},
            duration_seconds=0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers — synchronous, pure logic (easy to test)
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

    def _baseline_cv_score(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 3
    ) -> float:
        """Cross-validated R² for a Ridge model on the original feature set."""
        X_clean, y_clean = self._drop_na_rows(X, y)
        if len(X_clean) < cv * 2 or X_clean.empty:
            return 0.0
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        try:
            scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring="r2")
            return float(np.mean(scores))
        except Exception:
            return 0.0

    def _drop_na_rows(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Drop rows with any NaN in X or y."""
        combined = X.copy()
        combined["__y__"] = y.values
        combined = combined.dropna()
        return combined.drop(columns=["__y__"]), combined["__y__"]

    def _compute_vif(self, X: pd.DataFrame, feature_col: str) -> float:
        """
        Compute the Variance Inflation Factor for ``feature_col`` against
        all other numeric columns in ``X`` using sklearn LinearRegression.

        VIF = 1 / (1 - R²).  Returns 1.0 if no other columns exist.
        """
        other_cols = [c for c in X.columns if c != feature_col]
        if not other_cols:
            return 1.0

        X_others = X[other_cols].select_dtypes(include=[np.number])
        y_feat = X[feature_col]

        combined = X_others.copy()
        combined["__y__"] = y_feat.values
        combined = combined.dropna()
        if len(combined) < 3:
            return 1.0

        X_fit = combined.drop(columns=["__y__"])
        y_fit = combined["__y__"]
        if X_fit.empty:
            return 1.0

        lr = LinearRegression()
        try:
            lr.fit(X_fit, y_fit)
            r2 = lr.score(X_fit, y_fit)
            return round(1.0 / max(1.0 - r2, 1e-6), 4)
        except Exception:
            return 1.0

    def _evaluate_single_gene(
        self,
        gene: FeatureGene,
        X: pd.DataFrame,
        y: pd.Series,
        baseline_score: float,
        pre_fitness: Optional[float] = None,
    ) -> Optional[FeatureCandidate]:
        """
        Compute fitness, VIF, and correlation metrics for one gene.

        Returns a ``FeatureCandidate`` or ``None`` if the gene is degenerate.
        """
        feature = gene.apply(X)
        if feature is None:
            return None

        fname = gene.to_feature_name()

        # Fitness (cross-validated score)
        if pre_fitness is not None:
            fitness = pre_fitness
        else:
            ga = FeatureGA(population_size=1, generations=1, cv_folds=3)
            fitness = ga.evaluate_fitness(gene, X, y)

        performance_gain = fitness - baseline_score

        # VIF
        X_aug = X.copy()
        X_aug[fname] = feature.values
        X_aug = X_aug.select_dtypes(include=[np.number]).dropna()
        vif = self._compute_vif(X_aug, fname) if fname in X_aug.columns else 1.0

        # Correlation with target
        feat_aligned = feature.dropna()
        y_aligned = y.loc[feat_aligned.index] if hasattr(y, "loc") else y
        try:
            corr = float(feat_aligned.corr(y_aligned))
        except Exception:
            corr = 0.0

        validation_passed = (
            performance_gain > 0
            and vif < 10.0
            and not np.isnan(corr)
        )

        return gene.to_feature_candidate(
            importance_score=round(fitness, 6),
            performance_gain=round(performance_gain, 6),
            correlation_with_target=round(corr, 6) if not np.isnan(corr) else None,
            multicollinearity_vif=vif,
            validation_passed=validation_passed,
        )

    async def _fetch_llm_suggestions(
        self, existing_features: List[str], df: pd.DataFrame
    ) -> List[FeatureCandidate]:
        """
        Call the LLM for baseball feature suggestions.

        Returns a (possibly empty) list of ``FeatureCandidate`` objects.
        """
        if not (self._llm_available and self._llm is not None):
            return []

        prompt = (
            f"Existing features: {', '.join(existing_features[:30])}\n"
            f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            "Suggest new features that would improve a baseball player performance model."
        )

        try:
            raw = await self._llm.call(prompt, system_prompt=_LLM_FEATURE_SYSTEM_PROMPT)
            return self._parse_llm_suggestions(raw, existing_features)
        except (LLMError, LLMConfigError) as exc:
            self.logger.warning(f"LLM feature suggestion failed: {exc}")
            return []

    def _parse_llm_suggestions(
        self, raw: str, existing_features: List[str]
    ) -> List[FeatureCandidate]:
        """
        Parse a JSON array of feature suggestions from an LLM response.

        Silently drops suggestions with missing required fields.
        """
        candidates: List[FeatureCandidate] = []

        # Extract JSON array from response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return candidates

        try:
            suggestions = json.loads(raw[start:end])
        except json.JSONDecodeError:
            return candidates

        for item in suggestions:
            if not isinstance(item, dict):
                continue
            feature_name = item.get("feature_name")
            feature_def = item.get("feature_definition")
            feature_type = item.get("feature_type", "baseball")
            sources = item.get("source_features", [])
            if not feature_name or not feature_def:
                continue

            # Validate that referenced columns exist
            valid_sources = [s for s in sources if s in existing_features]

            candidates.append(
                FeatureCandidate(
                    feature_name=feature_name,
                    feature_definition=feature_def,
                    feature_type=feature_type,
                    source_features=valid_sources,
                    validation_passed=False,  # Not yet empirically tested
                )
            )

        return candidates


if __name__ == "__main__":
    import asyncio
    from shared.messaging import init_messaging, shutdown_messaging

    async def main():
        await init_messaging()
        agent = FeatureEngineerAgent()
        try:
            await agent.start()
        except KeyboardInterrupt:
            await agent.stop()
        finally:
            await shutdown_messaging()

    asyncio.run(main())
