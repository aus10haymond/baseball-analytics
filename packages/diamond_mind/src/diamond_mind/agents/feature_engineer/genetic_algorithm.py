"""
Genetic Algorithm framework for automated feature discovery.

The GA evolves a population of ``FeatureGene`` individuals, each encoding
a single derived feature (rolling stat, interaction, polynomial, or lag).
Fitness is measured by 3-fold cross-validated R² when the feature is added
to a baseline Ridge regression.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from shared.schemas import FeatureCandidate


# ---------------------------------------------------------------------------
# Gene types and their parameter schemas
# ---------------------------------------------------------------------------

GENE_TYPES = ("rolling_avg", "interaction", "polynomial", "lag")

_ROLLING_WINDOWS = [3, 5, 7, 10, 14, 21]
_ROLLING_FUNCS = ["mean", "std", "max", "min"]
_INTERACTION_OPS = ["mul", "div", "add", "sub"]
_POLY_DEGREES = [2, 3]
_LAG_PERIODS = [1, 2, 3, 5, 7, 10]


# ---------------------------------------------------------------------------
# FeatureGene
# ---------------------------------------------------------------------------


@dataclass
class FeatureGene:
    """
    Represents a single derived feature (the 'gene' in the GA).

    Attributes:
        gene_type:       Category of transformation.
        source_features: Column name(s) the feature is derived from.
        params:          Transformation-specific hyperparameters.
        gene_id:         Unique identifier (auto-generated).
    """

    gene_type: str
    source_features: List[str]
    params: Dict[str, Any] = field(default_factory=dict)
    gene_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_feature_name(self) -> str:
        """Return a deterministic, human-readable feature name."""
        src = "_x_".join(self.source_features)
        if self.gene_type == "rolling_avg":
            w = self.params.get("window", "?")
            fn = self.params.get("func", "mean")
            return f"{src}_roll{w}_{fn}"
        if self.gene_type == "interaction":
            op = self.params.get("operation", "mul")
            return f"{src}_{op}"
        if self.gene_type == "polynomial":
            deg = self.params.get("degree", 2)
            return f"{src}_pow{deg}"
        if self.gene_type == "lag":
            lag = self.params.get("lag", 1)
            return f"{src}_lag{lag}"
        return f"{self.gene_type}_{src}"

    def apply(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Compute the feature from ``df``.

        Returns ``None`` if required columns are missing or the result is
        degenerate (all NaN or constant).
        """
        try:
            if self.gene_type == "rolling_avg":
                col = self.source_features[0]
                if col not in df.columns:
                    return None
                window = self.params.get("window", 5)
                func = self.params.get("func", "mean")
                roller = df[col].rolling(window=window, min_periods=1)
                result = getattr(roller, func)()
            elif self.gene_type == "interaction":
                if len(self.source_features) < 2:
                    return None
                c1, c2 = self.source_features[0], self.source_features[1]
                if c1 not in df.columns or c2 not in df.columns:
                    return None
                op = self.params.get("operation", "mul")
                s1, s2 = df[c1], df[c2]
                if op == "mul":
                    result = s1 * s2
                elif op == "div":
                    result = s1 / s2.replace(0, np.nan)
                elif op == "add":
                    result = s1 + s2
                elif op == "sub":
                    result = s1 - s2
                else:
                    return None
            elif self.gene_type == "polynomial":
                col = self.source_features[0]
                if col not in df.columns:
                    return None
                degree = self.params.get("degree", 2)
                result = df[col] ** degree
            elif self.gene_type == "lag":
                col = self.source_features[0]
                if col not in df.columns:
                    return None
                lag = self.params.get("lag", 1)
                result = df[col].shift(lag)
            else:
                return None

            result.name = self.to_feature_name()

            # Reject degenerate features
            valid = result.dropna()
            if len(valid) < 3 or valid.std() == 0:
                return None

            return result

        except Exception:
            return None

    def to_feature_candidate(
        self,
        importance_score: Optional[float] = None,
        performance_gain: Optional[float] = None,
        correlation_with_target: Optional[float] = None,
        multicollinearity_vif: Optional[float] = None,
        validation_passed: bool = False,
    ) -> FeatureCandidate:
        """Convert this gene to a ``FeatureCandidate`` schema object."""
        # FeatureCandidate.importance_score has ge=0; clamp negative R² values
        clamped_score = max(0.0, importance_score) if importance_score is not None else None

        return FeatureCandidate(
            feature_name=self.to_feature_name(),
            feature_definition=f"{self.gene_type}({', '.join(self.source_features)}, {self.params})",
            feature_type=self.gene_type,
            source_features=self.source_features,
            importance_score=clamped_score,
            performance_gain=performance_gain,
            correlation_with_target=correlation_with_target,
            multicollinearity_vif=multicollinearity_vif,
            validation_passed=validation_passed,
        )


# ---------------------------------------------------------------------------
# FeatureGA
# ---------------------------------------------------------------------------


class FeatureGA:
    """
    Genetic algorithm for evolving high-value feature transformations.

    Population members are ``FeatureGene`` instances.  Each generation:
      1. Evaluate fitness (cross-validated R² improvement).
      2. Select parents via tournament selection.
      3. Produce offspring via crossover + mutation.
      4. Replace the bottom half of the population with offspring.

    Args:
        population_size: Number of genes per generation.
        generations:     Number of evolution cycles.
        mutation_rate:   Probability of mutating each gene parameter.
        tournament_size: Candidates drawn per tournament selection.
        cv_folds:        Cross-validation folds for fitness evaluation.
        seed:            Random seed for reproducibility.
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        cv_folds: int = 3,
        seed: Optional[int] = None,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.cv_folds = cv_folds
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def initialize_population(
        self, available_columns: List[str], n: Optional[int] = None
    ) -> List[FeatureGene]:
        """
        Create ``n`` (default ``population_size``) random genes from
        ``available_columns``.
        """
        n = n or self.population_size
        if not available_columns:
            return []
        return [self._random_gene(available_columns) for _ in range(n)]

    def _random_gene(self, available_columns: List[str]) -> FeatureGene:
        """Sample a random ``FeatureGene`` using available column names."""
        gene_type = self._rng.choice(GENE_TYPES)

        if gene_type == "rolling_avg":
            col = self._rng.choice(available_columns)
            return FeatureGene(
                gene_type="rolling_avg",
                source_features=[col],
                params={
                    "window": int(self._rng.choice(_ROLLING_WINDOWS)),
                    "func": str(self._rng.choice(_ROLLING_FUNCS)),
                },
            )
        if gene_type == "interaction":
            if len(available_columns) < 2:
                # Fall back to rolling_avg if can't form a pair
                col = available_columns[0]
                return FeatureGene(
                    gene_type="rolling_avg",
                    source_features=[col],
                    params={"window": 5, "func": "mean"},
                )
            cols = list(self._rng.choice(available_columns, size=2, replace=False))
            return FeatureGene(
                gene_type="interaction",
                source_features=cols,
                params={"operation": str(self._rng.choice(_INTERACTION_OPS))},
            )
        if gene_type == "polynomial":
            col = self._rng.choice(available_columns)
            return FeatureGene(
                gene_type="polynomial",
                source_features=[col],
                params={"degree": int(self._rng.choice(_POLY_DEGREES))},
            )
        # lag
        col = self._rng.choice(available_columns)
        return FeatureGene(
            gene_type="lag",
            source_features=[col],
            params={"lag": int(self._rng.choice(_LAG_PERIODS))},
        )

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def evaluate_fitness(
        self,
        gene: FeatureGene,
        X: pd.DataFrame,
        y: pd.Series,
        baseline_score: float = 0.0,
    ) -> float:
        """
        Measure how much adding this gene's feature improves model performance.

        Returns the mean cross-validated R² of a Ridge regression trained on
        (X + new_feature).  Returns ``baseline_score`` on any failure.
        """
        if len(X) < self.cv_folds * 2:
            return baseline_score

        feature = gene.apply(X)
        if feature is None:
            return baseline_score

        # Align feature to X
        X_aug = X.copy()
        fname = gene.to_feature_name()
        X_aug[fname] = feature.values

        # Drop NaNs consistently
        combined = X_aug.copy()
        combined["__target__"] = y.values
        combined = combined.dropna()
        if len(combined) < self.cv_folds * 2:
            return baseline_score

        X_clean = combined.drop(columns=["__target__"])
        y_clean = combined["__target__"]

        # Only use numeric columns
        X_clean = X_clean.select_dtypes(include=[np.number])
        if X_clean.empty:
            return baseline_score

        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        try:
            scores = cross_val_score(
                model, X_clean, y_clean, cv=self.cv_folds, scoring="r2"
            )
            return float(np.mean(scores))
        except Exception:
            return baseline_score

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def tournament_select(
        self,
        population: List[FeatureGene],
        fitnesses: List[float],
    ) -> FeatureGene:
        """
        Tournament selection: draw ``tournament_size`` random candidates and
        return the one with the highest fitness.
        """
        if not population:
            raise ValueError("Population is empty")
        size = min(self.tournament_size, len(population))
        indices = self._rng.choice(len(population), size=size, replace=False)
        best_idx = indices[int(np.argmax([fitnesses[i] for i in indices]))]
        return population[best_idx]

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def crossover(self, parent1: FeatureGene, parent2: FeatureGene) -> FeatureGene:
        """
        Produce a child gene by mixing two parents.

        If both parents share the same ``gene_type``, parameters are blended.
        Otherwise the child inherits the type and source features of one
        randomly chosen parent but may swap individual parameter values.
        """
        # Choose which parent donates the type
        if self._rng.random() < 0.5 or parent1.gene_type != parent2.gene_type:
            dominant, recessive = (parent1, parent2) if self._rng.random() < 0.5 else (parent2, parent1)
        else:
            dominant, recessive = parent1, parent2

        child_type = dominant.gene_type
        child_sources = list(dominant.source_features)

        # Try to borrow one source feature from the recessive parent
        if recessive.source_features and self._rng.random() < 0.3:
            donor_src = recessive.source_features[0]
            if donor_src not in child_sources:
                # Swap last source
                child_sources[-1] = donor_src

        # Blend params: inherit each key from a randomly chosen parent
        child_params = {}
        all_keys = set(dominant.params) | set(recessive.params)
        for key in all_keys:
            if key in dominant.params and key in recessive.params:
                child_params[key] = (
                    dominant.params[key] if self._rng.random() < 0.5 else recessive.params[key]
                )
            else:
                child_params[key] = dominant.params.get(key, recessive.params.get(key))

        return FeatureGene(
            gene_type=child_type,
            source_features=child_sources,
            params=child_params,
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(self, gene: FeatureGene, available_columns: List[str]) -> FeatureGene:
        """
        Randomly perturb a gene's parameters and/or source features.

        Each mutable slot is independently mutated with probability
        ``mutation_rate``.
        """
        # Possibly swap one source feature
        new_sources = list(gene.source_features)
        if available_columns and self._rng.random() < self.mutation_rate:
            idx = int(self._rng.integers(0, len(new_sources)))
            new_sources[idx] = str(self._rng.choice(available_columns))

        new_params = dict(gene.params)

        if gene.gene_type == "rolling_avg":
            if self._rng.random() < self.mutation_rate:
                new_params["window"] = int(self._rng.choice(_ROLLING_WINDOWS))
            if self._rng.random() < self.mutation_rate:
                new_params["func"] = str(self._rng.choice(_ROLLING_FUNCS))

        elif gene.gene_type == "interaction":
            if self._rng.random() < self.mutation_rate:
                new_params["operation"] = str(self._rng.choice(_INTERACTION_OPS))
            # Possibly swap one source
            if len(available_columns) >= 2 and self._rng.random() < self.mutation_rate:
                new_sources = list(
                    self._rng.choice(available_columns, size=2, replace=False)
                )

        elif gene.gene_type == "polynomial":
            if self._rng.random() < self.mutation_rate:
                new_params["degree"] = int(self._rng.choice(_POLY_DEGREES))

        elif gene.gene_type == "lag":
            if self._rng.random() < self.mutation_rate:
                new_params["lag"] = int(self._rng.choice(_LAG_PERIODS))

        return FeatureGene(
            gene_type=gene.gene_type,
            source_features=new_sources,
            params=new_params,
        )

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------

    def evolve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        available_columns: Optional[List[str]] = None,
        progress_callback=None,
    ) -> List[Tuple[FeatureGene, float]]:
        """
        Run the full genetic algorithm.

        Args:
            X:                  Feature DataFrame (existing features).
            y:                  Target series.
            available_columns:  Columns to derive new features from.
                                Defaults to all numeric columns of X.
            progress_callback:  Optional callable(generation, best_fitness)
                                called after each generation.

        Returns:
            List of (gene, fitness) sorted descending by fitness.
        """
        if available_columns is None:
            available_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        if not available_columns:
            return []

        population = self.initialize_population(available_columns)
        fitnesses = [self.evaluate_fitness(g, X, y) for g in population]

        for gen in range(self.generations):
            # Elitism: keep top 20%
            elite_n = max(1, self.population_size // 5)
            sorted_idx = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
            elites = [population[i] for i in sorted_idx[:elite_n]]
            elite_fits = [fitnesses[i] for i in sorted_idx[:elite_n]]

            # Fill rest with offspring
            offspring: List[FeatureGene] = []
            while len(offspring) < self.population_size - elite_n:
                p1 = self.tournament_select(population, fitnesses)
                p2 = self.tournament_select(population, fitnesses)
                child = self.crossover(p1, p2)
                child = self.mutate(child, available_columns)
                offspring.append(child)

            offspring_fits = [self.evaluate_fitness(g, X, y) for g in offspring]

            population = elites + offspring
            fitnesses = elite_fits + offspring_fits

            if progress_callback:
                progress_callback(gen + 1, max(fitnesses))

        # Return sorted results
        paired = sorted(
            zip(population, fitnesses), key=lambda t: t[1], reverse=True
        )
        return [(g, f) for g, f in paired]
