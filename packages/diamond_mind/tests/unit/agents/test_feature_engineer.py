"""
Unit tests for the Feature Engineer Agent and its helper modules.

Covers:
- feature_generators: rolling_stat, interaction, polynomial, lag, baseball features,
  generate_all_candidates
- genetic_algorithm: FeatureGene (apply, to_feature_name, to_feature_candidate),
  FeatureGA (initialize_population, evaluate_fitness, tournament_select,
  crossover, mutate, evolve)
- FeatureEngineerAgent: _load_data, _baseline_cv_score, _compute_vif,
  _evaluate_single_gene, _parse_llm_suggestions
- handle_task: search_features, evaluate_feature, generate_features,
  get_llm_suggestions
"""

import sys
import json
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

_dm_src = str(Path(__file__).resolve().parents[3] / "src" / "diamond_mind")
if _dm_src not in sys.path:
    sys.path.insert(0, _dm_src)

from shared.schemas import AgentType, TaskStatus, AgentTask, TaskPriority
from agents.feature_engineer.agent import FeatureEngineerAgent
from agents.feature_engineer.feature_generators import (
    rolling_stat,
    interaction,
    polynomial,
    lag_feature,
    iso_power,
    babip,
    k_pct,
    bb_pct,
    contact_rate,
    obp_to_slg_ratio,
    generate_baseball_features,
    generate_all_candidates,
    generate_rolling_candidates,
    generate_interaction_candidates,
    generate_polynomial_candidates,
    generate_lag_candidates,
)
from agents.feature_engineer.genetic_algorithm import FeatureGA, FeatureGene, GENE_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def baseball_df():
    """Synthetic DataFrame with baseball-ish columns."""
    np.random.seed(42)
    n = 80
    return pd.DataFrame(
        {
            "batting_avg": np.random.uniform(0.220, 0.340, n).round(3),
            "home_runs": np.random.randint(5, 40, n).astype(float),
            "ops": np.random.uniform(0.650, 0.950, n).round(3),
            "rbi": np.random.randint(20, 120, n).astype(float),
            "strikeouts": np.random.randint(30, 180, n).astype(float),
            "walks": np.random.randint(15, 100, n).astype(float),
        }
    )


@pytest.fixture
def full_baseball_df():
    """DataFrame with all columns needed for baseball-domain features."""
    np.random.seed(7)
    n = 80
    ab = np.random.randint(300, 600, n).astype(float)
    h = (ab * np.random.uniform(0.22, 0.34, n)).astype(int).astype(float)
    hr = np.random.randint(5, 40, n).astype(float)
    k = np.random.randint(50, 180, n).astype(float)
    bb = np.random.randint(20, 100, n).astype(float)
    pa = ab + bb
    slg = np.random.uniform(0.35, 0.60, n).round(3)
    avg = (h / ab).round(3)
    obp = np.random.uniform(0.28, 0.42, n).round(3)
    return pd.DataFrame(
        {
            "batting_avg": avg,
            "slugging": slg,
            "obp": obp,
            "ops": obp + slg,
            "hits": h,
            "home_runs": hr,
            "at_bats": ab,
            "strikeouts": k,
            "walks": bb,
            "plate_appearances": pa,
        }
    )


@pytest.fixture
def parquet_path(tmp_path, baseball_df):
    p = tmp_path / "baseball.parquet"
    baseball_df.to_parquet(p, index=False)
    return p


@pytest.fixture
def parquet_with_target(tmp_path, baseball_df):
    df = baseball_df.copy()
    df["war"] = np.random.uniform(0, 8, len(df))
    p = tmp_path / "baseball_war.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture
def agent():
    return FeatureEngineerAgent()


def make_task(task_type: str, parameters: dict | None = None) -> AgentTask:
    return AgentTask(
        task_id="test_fe_001",
        agent_id=AgentType.FEATURE_ENGINEER,
        task_type=task_type,
        priority=TaskPriority.MEDIUM,
        parameters=parameters or {},
    )


# ---------------------------------------------------------------------------
# rolling_stat
# ---------------------------------------------------------------------------


class TestRollingStat:
    def test_mean_output_shape(self, baseball_df):
        result = rolling_stat(baseball_df["batting_avg"], window=5, func="mean")
        assert len(result) == len(baseball_df)

    def test_name_contains_window_and_func(self, baseball_df):
        result = rolling_stat(baseball_df["batting_avg"], window=7, func="std")
        assert "7" in result.name
        assert "std" in result.name

    def test_unknown_func_raises(self, baseball_df):
        with pytest.raises(ValueError, match="Unknown window func"):
            rolling_stat(baseball_df["batting_avg"], window=5, func="median")

    def test_min_periods_1_no_leading_nan(self, baseball_df):
        result = rolling_stat(baseball_df["batting_avg"], window=10, func="max")
        # With min_periods=1, first value should not be NaN
        assert not result.iloc[0:1].isnull().any()

    def test_all_funcs_produce_results(self, baseball_df):
        for fn in ["mean", "std", "max", "min", "sum"]:
            result = rolling_stat(baseball_df["batting_avg"], window=3, func=fn)
            assert len(result) == len(baseball_df)


# ---------------------------------------------------------------------------
# interaction
# ---------------------------------------------------------------------------


class TestInteraction:
    def test_mul(self, baseball_df):
        result = interaction(baseball_df["batting_avg"], baseball_df["home_runs"], "mul")
        expected = baseball_df["batting_avg"] * baseball_df["home_runs"]
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_div_by_zero_becomes_nan(self):
        s1 = pd.Series([1.0, 2.0, 3.0], name="a")
        s2 = pd.Series([0.0, 1.0, 2.0], name="b")
        result = interaction(s1, s2, "div")
        assert result.iloc[0] != result.iloc[0]  # NaN

    def test_name_contains_op(self, baseball_df):
        result = interaction(baseball_df["batting_avg"], baseball_df["ops"], "add")
        assert "add" in result.name

    def test_unknown_op_raises(self, baseball_df):
        with pytest.raises(ValueError, match="Unknown operation"):
            interaction(baseball_df["batting_avg"], baseball_df["ops"], "pow")

    def test_all_ops_return_series(self, baseball_df):
        for op in ["mul", "div", "add", "sub"]:
            result = interaction(baseball_df["batting_avg"], baseball_df["ops"], op)
            assert isinstance(result, pd.Series)
            assert len(result) == len(baseball_df)


# ---------------------------------------------------------------------------
# polynomial
# ---------------------------------------------------------------------------


class TestPolynomial:
    def test_squared(self, baseball_df):
        result = polynomial(baseball_df["batting_avg"], degree=2)
        expected = baseball_df["batting_avg"] ** 2
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_cubed(self, baseball_df):
        result = polynomial(baseball_df["batting_avg"], degree=3)
        assert result.iloc[0] == pytest.approx(baseball_df["batting_avg"].iloc[0] ** 3)

    def test_name_contains_degree(self, baseball_df):
        result = polynomial(baseball_df["batting_avg"], degree=3)
        assert "pow3" in result.name

    def test_degree_below_2_raises(self, baseball_df):
        with pytest.raises(ValueError, match="Degree must be >= 2"):
            polynomial(baseball_df["batting_avg"], degree=1)


# ---------------------------------------------------------------------------
# lag_feature
# ---------------------------------------------------------------------------


class TestLagFeature:
    def test_shift_by_one(self, baseball_df):
        result = lag_feature(baseball_df["batting_avg"], lag=1)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(baseball_df["batting_avg"].iloc[0])

    def test_name_contains_lag(self, baseball_df):
        result = lag_feature(baseball_df["batting_avg"], lag=3)
        assert "lag3" in result.name

    def test_lag_zero_raises(self, baseball_df):
        with pytest.raises(ValueError, match="Lag must be >= 1"):
            lag_feature(baseball_df["batting_avg"], lag=0)


# ---------------------------------------------------------------------------
# Baseball domain features
# ---------------------------------------------------------------------------


class TestBaseballFeatures:
    def test_iso_power(self):
        slg = pd.Series([0.500, 0.400])
        avg = pd.Series([0.300, 0.250])
        result = iso_power(slg, avg)
        assert result.iloc[0] == pytest.approx(0.200)
        assert result.name == "iso_power"

    def test_babip(self):
        h = pd.Series([150.0])
        hr = pd.Series([20.0])
        ab = pd.Series([500.0])
        k = pd.Series([100.0])
        result = babip(h, hr, ab, k)
        expected = (150 - 20) / (500 - 100 - 20)
        assert result.iloc[0] == pytest.approx(expected)

    def test_k_pct(self):
        k = pd.Series([100.0])
        pa = pd.Series([500.0])
        result = k_pct(k, pa)
        assert result.iloc[0] == pytest.approx(0.2)

    def test_bb_pct(self):
        bb = pd.Series([50.0])
        pa = pd.Series([500.0])
        result = bb_pct(bb, pa)
        assert result.iloc[0] == pytest.approx(0.1)

    def test_contact_rate(self):
        ab = pd.Series([500.0])
        k = pd.Series([100.0])
        result = contact_rate(ab, k)
        assert result.iloc[0] == pytest.approx(0.8)

    def test_obp_slg_ratio(self):
        obp = pd.Series([0.360])
        slg = pd.Series([0.480])
        result = obp_to_slg_ratio(obp, slg)
        assert result.iloc[0] == pytest.approx(0.75)

    def test_generate_baseball_features_full_df(self, full_baseball_df):
        result = generate_baseball_features(full_baseball_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(full_baseball_df)
        assert len(result.columns) > 0

    def test_generate_baseball_features_missing_cols(self, baseball_df):
        # baseball_df lacks slugging, obp, at_bats etc.
        result = generate_baseball_features(baseball_df)
        # Should still return a DataFrame, possibly with iso_approx only
        assert isinstance(result, pd.DataFrame)

    def test_generate_baseball_features_empty_df(self):
        empty = pd.DataFrame()
        result = generate_baseball_features(empty)
        assert result.empty


# ---------------------------------------------------------------------------
# Bulk candidate generation
# ---------------------------------------------------------------------------


class TestBulkGenerators:
    def test_rolling_candidates_count(self, baseball_df):
        cols = baseball_df.columns.tolist()
        candidates = generate_rolling_candidates(baseball_df, cols)
        # 6 cols × 3 windows × 3 funcs = 54
        assert len(candidates) == 54

    def test_interaction_candidates_returned(self, baseball_df):
        cols = baseball_df.columns.tolist()
        candidates = generate_interaction_candidates(baseball_df, cols)
        assert len(candidates) > 0

    def test_polynomial_candidates(self, baseball_df):
        cols = baseball_df.columns.tolist()
        candidates = generate_polynomial_candidates(baseball_df, cols)
        # 6 cols × 2 degrees = 12
        assert len(candidates) == 12

    def test_lag_candidates(self, baseball_df):
        cols = baseball_df.columns.tolist()
        candidates = generate_lag_candidates(baseball_df, cols)
        # 6 cols × 3 lags = 18
        assert len(candidates) == 18

    def test_generate_all_candidates_no_duplicates(self, baseball_df):
        candidates = generate_all_candidates(baseball_df)
        names = [n for n, _ in candidates]
        assert len(names) == len(set(names))

    def test_generate_all_candidates_returns_list_of_tuples(self, baseball_df):
        candidates = generate_all_candidates(baseball_df)
        assert isinstance(candidates, list)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in candidates)

    def test_generate_all_candidates_empty_df(self):
        empty = pd.DataFrame()
        candidates = generate_all_candidates(empty)
        assert candidates == []

    def test_max_per_type_respected(self, baseball_df):
        candidates = generate_all_candidates(baseball_df, max_per_type=5)
        # rolling + interaction + polynomial + lag ≤ 4 * 5 + baseball
        assert len(candidates) <= 4 * 5 + 15  # generous upper bound


# ---------------------------------------------------------------------------
# FeatureGene
# ---------------------------------------------------------------------------


class TestFeatureGene:
    def test_rolling_avg_name(self):
        gene = FeatureGene("rolling_avg", ["batting_avg"], {"window": 5, "func": "mean"})
        assert gene.to_feature_name() == "batting_avg_roll5_mean"

    def test_interaction_name(self):
        gene = FeatureGene("interaction", ["batting_avg", "ops"], {"operation": "mul"})
        assert "mul" in gene.to_feature_name()

    def test_polynomial_name(self):
        gene = FeatureGene("polynomial", ["home_runs"], {"degree": 2})
        assert "pow2" in gene.to_feature_name()

    def test_lag_name(self):
        gene = FeatureGene("lag", ["rbi"], {"lag": 3})
        assert "lag3" in gene.to_feature_name()

    def test_apply_rolling_avg(self, baseball_df):
        gene = FeatureGene("rolling_avg", ["batting_avg"], {"window": 3, "func": "mean"})
        result = gene.apply(baseball_df)
        assert result is not None
        assert len(result) == len(baseball_df)

    def test_apply_interaction(self, baseball_df):
        gene = FeatureGene("interaction", ["batting_avg", "ops"], {"operation": "mul"})
        result = gene.apply(baseball_df)
        assert result is not None

    def test_apply_polynomial(self, baseball_df):
        gene = FeatureGene("polynomial", ["home_runs"], {"degree": 2})
        result = gene.apply(baseball_df)
        assert result is not None
        assert result.iloc[0] == pytest.approx(baseball_df["home_runs"].iloc[0] ** 2)

    def test_apply_lag(self, baseball_df):
        gene = FeatureGene("lag", ["rbi"], {"lag": 1})
        result = gene.apply(baseball_df)
        assert result is not None
        assert pd.isna(result.iloc[0])

    def test_apply_missing_column_returns_none(self, baseball_df):
        gene = FeatureGene("rolling_avg", ["nonexistent_col"], {"window": 5, "func": "mean"})
        assert gene.apply(baseball_df) is None

    def test_apply_constant_series_returns_none(self):
        df = pd.DataFrame({"x": [1.0] * 20})
        gene = FeatureGene("polynomial", ["x"], {"degree": 2})
        # x ** 2 is also constant → should return None
        assert gene.apply(df) is None

    def test_to_feature_candidate(self):
        gene = FeatureGene("rolling_avg", ["batting_avg"], {"window": 5, "func": "mean"})
        candidate = gene.to_feature_candidate(importance_score=0.75, validation_passed=True)
        assert candidate.feature_name == gene.to_feature_name()
        assert candidate.importance_score == 0.75
        assert candidate.validation_passed is True


# ---------------------------------------------------------------------------
# FeatureGA
# ---------------------------------------------------------------------------


class TestFeatureGAInit:
    def test_initialize_population_size(self, baseball_df):
        ga = FeatureGA(population_size=10, seed=0)
        pop = ga.initialize_population(baseball_df.columns.tolist())
        assert len(pop) == 10

    def test_initialize_population_empty_cols(self):
        ga = FeatureGA(seed=0)
        pop = ga.initialize_population([])
        assert pop == []

    def test_all_gene_types_representable(self):
        ga = FeatureGA(population_size=100, seed=42)
        cols = ["a", "b", "c", "d"]
        df = pd.DataFrame({c: np.random.rand(30) for c in cols})
        pop = ga.initialize_population(df.columns.tolist())
        types = {g.gene_type for g in pop}
        # With 100 genes and 4 types, all should appear
        assert len(types) >= 2


class TestFeatureGAFitness:
    def test_returns_float(self, baseball_df):
        ga = FeatureGA(population_size=5, generations=1, cv_folds=3, seed=0)
        X = baseball_df.drop(columns=["rbi"])
        y = baseball_df["rbi"]
        gene = FeatureGene("rolling_avg", ["batting_avg"], {"window": 5, "func": "mean"})
        score = ga.evaluate_fitness(gene, X, y)
        assert isinstance(score, float)

    def test_degenerate_gene_returns_baseline(self, baseball_df):
        ga = FeatureGA(population_size=5, generations=1, seed=0)
        X = baseball_df.drop(columns=["rbi"])
        y = baseball_df["rbi"]
        # Gene that references a missing column
        gene = FeatureGene("rolling_avg", ["missing_col"], {"window": 5, "func": "mean"})
        score = ga.evaluate_fitness(gene, X, y, baseline_score=-999.0)
        assert score == -999.0

    def test_too_few_rows_returns_baseline(self):
        ga = FeatureGA(cv_folds=5, seed=0)
        tiny = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = pd.Series([0.0, 1.0, 2.0])
        gene = FeatureGene("polynomial", ["x"], {"degree": 2})
        score = ga.evaluate_fitness(gene, tiny, y, baseline_score=0.0)
        assert score == 0.0


class TestFeatureGATournament:
    def test_returns_a_gene(self, baseball_df):
        ga = FeatureGA(tournament_size=3, seed=0)
        pop = ga.initialize_population(baseball_df.columns.tolist(), n=10)
        fits = [float(i) for i in range(10)]
        winner = ga.tournament_select(pop, fits)
        assert isinstance(winner, FeatureGene)

    def test_highest_fitness_wins_large_tournament(self, baseball_df):
        ga = FeatureGA(tournament_size=10, seed=0)
        cols = baseball_df.columns.tolist()
        pop = ga.initialize_population(cols, n=10)
        # Make last gene have the highest fitness
        fits = [0.0] * 9 + [1.0]
        # With tournament_size = 10 (== pop size) best should always win
        winner = ga.tournament_select(pop, fits)
        assert winner is pop[9]

    def test_empty_population_raises(self):
        ga = FeatureGA(seed=0)
        with pytest.raises(ValueError, match="Population is empty"):
            ga.tournament_select([], [])


class TestFeatureGACrossover:
    def test_child_has_valid_gene_type(self, baseball_df):
        ga = FeatureGA(seed=0)
        cols = baseball_df.columns.tolist()
        p1 = ga._random_gene(cols)
        p2 = ga._random_gene(cols)
        child = ga.crossover(p1, p2)
        assert child.gene_type in GENE_TYPES

    def test_child_has_source_features(self, baseball_df):
        ga = FeatureGA(seed=1)
        cols = baseball_df.columns.tolist()
        p1 = ga._random_gene(cols)
        p2 = ga._random_gene(cols)
        child = ga.crossover(p1, p2)
        assert len(child.source_features) >= 1

    def test_child_params_non_empty(self, baseball_df):
        ga = FeatureGA(seed=2)
        cols = baseball_df.columns.tolist()
        p1 = ga._random_gene(cols)
        p2 = ga._random_gene(cols)
        child = ga.crossover(p1, p2)
        assert isinstance(child.params, dict)


class TestFeatureGAMutate:
    def test_returns_feature_gene(self, baseball_df):
        ga = FeatureGA(mutation_rate=1.0, seed=0)
        cols = baseball_df.columns.tolist()
        gene = ga._random_gene(cols)
        mutated = ga.mutate(gene, cols)
        assert isinstance(mutated, FeatureGene)

    def test_mutation_changes_gene(self, baseball_df):
        ga = FeatureGA(mutation_rate=1.0, seed=0)
        cols = baseball_df.columns.tolist()
        gene = FeatureGene("rolling_avg", ["batting_avg"], {"window": 5, "func": "mean"})
        results = set()
        for _ in range(20):
            m = ga.mutate(gene, cols)
            results.add((tuple(m.source_features), str(m.params)))
        # With mutation_rate=1.0, something should change
        assert len(results) > 1

    def test_zero_mutation_rate_preserves_gene(self, baseball_df):
        ga = FeatureGA(mutation_rate=0.0, seed=0)
        gene = FeatureGene("lag", ["rbi"], {"lag": 3})
        mutated = ga.mutate(gene, baseball_df.columns.tolist())
        assert mutated.gene_type == gene.gene_type
        assert mutated.params == gene.params


class TestFeatureGAEvolve:
    def test_evolve_returns_sorted_list(self, baseball_df):
        ga = FeatureGA(population_size=8, generations=2, seed=0)
        X = baseball_df.drop(columns=["rbi"])
        y = baseball_df["rbi"]
        results = ga.evolve(X, y)
        assert isinstance(results, list)
        fitnesses = [f for _, f in results]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_evolve_empty_columns(self, baseball_df):
        ga = FeatureGA(population_size=5, generations=1, seed=0)
        X = pd.DataFrame(index=baseball_df.index)
        y = baseball_df["rbi"]
        results = ga.evolve(X, y, available_columns=[])
        assert results == []

    def test_evolve_progress_callback(self, baseball_df):
        progress = []
        ga = FeatureGA(population_size=6, generations=3, seed=0)
        X = baseball_df.drop(columns=["rbi"])
        y = baseball_df["rbi"]
        ga.evolve(X, y, progress_callback=lambda gen, best: progress.append((gen, best)))
        assert len(progress) == 3
        assert progress[0][0] == 1
        assert progress[2][0] == 3


# ---------------------------------------------------------------------------
# FeatureEngineerAgent helpers
# ---------------------------------------------------------------------------


class TestAgentLoadData:
    def test_loads_parquet(self, agent, parquet_path, baseball_df):
        df = agent._load_data(str(parquet_path))
        assert len(df) == len(baseball_df)

    def test_loads_csv(self, agent, tmp_path, baseball_df):
        p = tmp_path / "data.csv"
        baseball_df.to_csv(p, index=False)
        df = agent._load_data(str(p))
        assert len(df) == len(baseball_df)

    def test_missing_file_raises(self, agent):
        with pytest.raises(FileNotFoundError):
            agent._load_data("/nonexistent/path.parquet")

    def test_unsupported_format_raises(self, agent, tmp_path):
        p = tmp_path / "data.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file format"):
            agent._load_data(str(p))


class TestBaselineCVScore:
    def test_returns_float(self, agent, baseball_df):
        X = baseball_df.drop(columns=["rbi"])
        y = baseball_df["rbi"]
        score = agent._baseline_cv_score(X, y)
        assert isinstance(score, float)

    def test_empty_dataframe_returns_zero(self, agent):
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        score = agent._baseline_cv_score(X, y)
        assert score == 0.0

    def test_too_few_rows_returns_zero(self, agent):
        X = pd.DataFrame({"x": [1.0, 2.0]})
        y = pd.Series([0.0, 1.0])
        score = agent._baseline_cv_score(X, y, cv=3)
        assert score == 0.0


class TestComputeVIF:
    def test_single_column_returns_one(self, agent, baseball_df):
        single = baseball_df[["batting_avg"]]
        vif = agent._compute_vif(single, "batting_avg")
        assert vif == 1.0

    def test_uncorrelated_features_low_vif(self, agent):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "a": rng.uniform(0, 1, 100),
                "b": rng.uniform(0, 1, 100),
                "c": rng.uniform(0, 1, 100),
            }
        )
        vif = agent._compute_vif(df, "a")
        assert vif < 5.0

    def test_highly_correlated_features_high_vif(self, agent):
        x = np.linspace(0, 1, 100)
        df = pd.DataFrame(
            {
                "a": x,
                "b": x + np.random.default_rng(1).normal(0, 0.001, 100),
            }
        )
        vif = agent._compute_vif(df, "a")
        assert vif > 10.0


class TestEvaluateSingleGene:
    def test_valid_gene_returns_candidate(self, agent, baseball_df):
        X = baseball_df.drop(columns=["rbi"])
        y = baseball_df["rbi"]
        gene = FeatureGene("rolling_avg", ["batting_avg"], {"window": 5, "func": "mean"})
        candidate = agent._evaluate_single_gene(gene, X, y, baseline_score=0.0)
        assert candidate is not None
        assert candidate.feature_name == gene.to_feature_name()
        assert candidate.importance_score is not None

    def test_degenerate_gene_returns_none(self, agent, baseball_df):
        X = baseball_df.drop(columns=["rbi"])
        y = baseball_df["rbi"]
        gene = FeatureGene("rolling_avg", ["nonexistent"], {"window": 5, "func": "mean"})
        assert agent._evaluate_single_gene(gene, X, y, baseline_score=0.0) is None

    def test_validation_passed_field_set(self, agent, baseball_df):
        X = baseball_df.drop(columns=["rbi"])
        y = baseball_df["rbi"]
        gene = FeatureGene("interaction", ["batting_avg", "ops"], {"operation": "mul"})
        candidate = agent._evaluate_single_gene(gene, X, y, baseline_score=-10.0)
        # With a very low baseline, gain should be positive → validation might pass
        assert isinstance(candidate.validation_passed, bool)


class TestParseLLMSuggestions:
    def test_parses_valid_json_array(self, agent):
        raw = json.dumps([
            {
                "feature_name": "iso_approx",
                "feature_definition": "ops * 0.6 - batting_avg",
                "feature_type": "baseball",
                "source_features": ["ops", "batting_avg"],
                "reasoning": "ISO approximation",
            }
        ])
        existing = ["ops", "batting_avg", "home_runs"]
        results = agent._parse_llm_suggestions(raw, existing)
        assert len(results) == 1
        assert results[0].feature_name == "iso_approx"

    def test_skips_items_missing_required_fields(self, agent):
        raw = json.dumps([{"feature_type": "baseball"}])
        results = agent._parse_llm_suggestions(raw, ["ops"])
        assert results == []

    def test_filters_invalid_source_columns(self, agent):
        raw = json.dumps([
            {
                "feature_name": "test_feat",
                "feature_definition": "x / y",
                "source_features": ["real_col", "nonexistent_col"],
            }
        ])
        results = agent._parse_llm_suggestions(raw, ["real_col"])
        assert results[0].source_features == ["real_col"]

    def test_handles_non_json_gracefully(self, agent):
        results = agent._parse_llm_suggestions("This is not JSON at all.", [])
        assert results == []

    def test_handles_markdown_fenced_array(self, agent):
        raw = "```json\n" + json.dumps([
            {
                "feature_name": "k_rate",
                "feature_definition": "k / pa",
                "feature_type": "baseball",
                "source_features": ["strikeouts"],
            }
        ]) + "\n```"
        results = agent._parse_llm_suggestions(raw, ["strikeouts"])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# handle_task: end-to-end
# ---------------------------------------------------------------------------


class TestHandleTask:
    @pytest.mark.asyncio
    async def test_generate_features(self, agent, global_mq, parquet_path):
        task = make_task("generate_features", {"data_source": str(parquet_path)})
        result = await agent._generate_features(task)
        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["total_generated"] > 0
        assert isinstance(result.result_data["feature_names"], list)

    @pytest.mark.asyncio
    async def test_evaluate_feature_valid(self, agent, global_mq, parquet_with_target):
        task = make_task(
            "evaluate_feature",
            {
                "data_source": str(parquet_with_target),
                "target_col": "war",
                "feature_name": "avg_roll5",
                "gene_type": "rolling_avg",
                "source_features": ["batting_avg"],
                "params": {"window": 5, "func": "mean"},
            },
        )
        result = await agent._evaluate_feature(task)
        assert result.status == TaskStatus.COMPLETED
        assert "importance_score" in result.metrics

    @pytest.mark.asyncio
    async def test_evaluate_feature_degenerate(self, agent, global_mq, parquet_with_target):
        task = make_task(
            "evaluate_feature",
            {
                "data_source": str(parquet_with_target),
                "target_col": "war",
                "feature_name": "bad_feat",
                "gene_type": "rolling_avg",
                "source_features": ["missing_column"],
                "params": {"window": 5, "func": "mean"},
            },
        )
        result = await agent._evaluate_feature(task)
        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["validation_passed"] is False

    @pytest.mark.asyncio
    async def test_search_features_runs_end_to_end(self, agent, global_mq, parquet_with_target):
        task = make_task(
            "search_features",
            {
                "data_source": str(parquet_with_target),
                "target_col": "war",
                "population_size": 6,
                "generations": 2,
                "top_k": 3,
                "use_llm": False,
            },
        )
        result = await agent._search_features(task)
        assert result.status == TaskStatus.COMPLETED
        assert "features_added" in result.result_data
        assert "baseline_model_score" in result.result_data
        assert result.metrics["candidates_evaluated"] > 0

    @pytest.mark.asyncio
    async def test_get_llm_suggestions_no_llm(self, agent, global_mq, parquet_path):
        agent._llm_available = False
        task = make_task("get_llm_suggestions", {"data_source": str(parquet_path)})
        result = await agent._get_llm_suggestions(task)
        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["total"] == 0

    @pytest.mark.asyncio
    async def test_get_llm_suggestions_with_mock_llm(self, agent, global_mq, parquet_path):
        suggestions = [
            {
                "feature_name": "avg_hr_ratio",
                "feature_definition": "batting_avg / home_runs",
                "feature_type": "interaction",
                "source_features": ["batting_avg", "home_runs"],
                "reasoning": "Ratio of avg to power",
            }
        ]
        mock_llm = AsyncMock()
        mock_llm.call = AsyncMock(return_value=json.dumps(suggestions))
        agent._llm = mock_llm
        agent._llm_available = True

        task = make_task("get_llm_suggestions", {"data_source": str(parquet_path)})
        result = await agent._get_llm_suggestions(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["total"] == 1
        assert result.result_data["suggestions"][0]["feature_name"] == "avg_hr_ratio"

    @pytest.mark.asyncio
    async def test_search_features_missing_target(self, agent, global_mq, parquet_path):
        task = make_task(
            "search_features",
            {
                "data_source": str(parquet_path),
                "target_col": "nonexistent_col",
                "use_llm": False,
            },
        )
        with pytest.raises(ValueError, match="Target column"):
            await agent._search_features(task)

    @pytest.mark.asyncio
    async def test_unknown_task_type_raises(self, agent):
        task = make_task("mystery_task")
        with pytest.raises(ValueError, match="Unknown task type"):
            await agent.handle_task(task)
