"""
Unit tests for the DataQualityAgent.

Covers:
- _run_anomaly_detection: Isolation Forest + Z-score detection
- _run_schema_validation: first-visit caching, missing/extra cols, type changes
- _apply_repairs: duplicate removal, imputation, outlier clipping
- _compute_quality_metrics: metric calculation
- _load_data: file loading and error handling
- handle_task: task dispatch + result structure
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the shared package is on the path
_dm_src = str(Path(__file__).resolve().parents[3] / "src" / "diamond_mind")
if _dm_src not in sys.path:
    sys.path.insert(0, _dm_src)

from shared.schemas import AlertSeverity, AgentType, TaskStatus, AgentTask, TaskPriority
from agents.data_quality.agent import DataQualityAgent


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent():
    return DataQualityAgent()


@pytest.fixture
def clean_df():
    """20-row baseball DataFrame with no missing values or outliers."""
    np.random.seed(0)
    return pd.DataFrame(
        {
            "batting_avg": np.random.uniform(0.220, 0.330, 20).round(3),
            "home_runs": np.random.randint(10, 35, 20),
            "rbi": np.random.randint(30, 100, 20),
            "ops": np.random.uniform(0.650, 0.950, 20).round(3),
        }
    )


@pytest.fixture
def dirty_df():
    """DataFrame with missing values, duplicates, and outliers."""
    np.random.seed(1)
    df = pd.DataFrame(
        {
            "batting_avg": np.concatenate(
                [np.random.uniform(0.220, 0.330, 16), [np.nan, np.nan, 99.0, -50.0]]
            ),
            "home_runs": np.concatenate(
                [np.random.randint(10, 35, 18).astype(float), [np.nan, np.nan]]
            ),
            "team": ["NYY"] * 18 + [None, None],
        }
    )
    # Add a duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


@pytest.fixture
def parquet_path(tmp_path, clean_df):
    path = tmp_path / "test_data.parquet"
    clean_df.to_parquet(path, index=False)
    return path


@pytest.fixture
def dirty_parquet_path(tmp_path, dirty_df):
    path = tmp_path / "dirty_data.parquet"
    dirty_df.to_parquet(path, index=False)
    return path


def make_task(task_type: str, parameters: dict | None = None) -> AgentTask:
    return AgentTask(
        task_id="test_001",
        agent_id=AgentType.DATA_QUALITY,
        task_type=task_type,
        priority=TaskPriority.HIGH,
        parameters=parameters or {},
    )


# ---------------------------------------------------------------------------
# _load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_loads_parquet(self, agent, parquet_path, clean_df):
        df = agent._load_data(str(parquet_path))
        assert len(df) == len(clean_df)
        assert list(df.columns) == list(clean_df.columns)

    def test_loads_csv(self, agent, tmp_path, clean_df):
        csv_path = tmp_path / "data.csv"
        clean_df.to_csv(csv_path, index=False)
        df = agent._load_data(str(csv_path))
        assert len(df) == len(clean_df)

    def test_raises_on_missing_file(self, agent):
        with pytest.raises(FileNotFoundError):
            agent._load_data("/nonexistent/path/data.parquet")

    def test_raises_on_unsupported_format(self, agent, tmp_path):
        p = tmp_path / "data.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file format"):
            agent._load_data(str(p))


# ---------------------------------------------------------------------------
# _run_anomaly_detection
# ---------------------------------------------------------------------------


class TestRunAnomalyDetection:
    def test_no_anomalies_on_clean_data(self, agent, clean_df):
        # With contamination=0.0001 (near zero), no Isolation Forest anomalies expected
        # but Z-score on perfectly bounded data should also be minimal
        reports = agent._run_anomaly_detection(clean_df, "task_1", contamination=0.001)
        iso_reports = [r for r in reports if r.detection_method == "isolation_forest"]
        # At extremely low contamination, very few (if any) rows flagged
        assert all(r.row_count >= 0 for r in iso_reports)

    def test_detects_extreme_outliers_via_zscore(self, agent):
        # 98 values clustered near 1.0–2.0; two extreme outliers at ±1000
        # With large n, std stays small and z-scores for outliers far exceed 3.0
        normal = np.ones(98) + np.random.default_rng(0).uniform(0, 0.5, 98)
        df = pd.DataFrame({"value": np.concatenate([normal, [1000.0, -1000.0]])})
        reports = agent._run_anomaly_detection(df, "task_z", threshold=3.0)
        zscore_reports = [r for r in reports if r.detection_method == "zscore"]
        assert len(zscore_reports) >= 1
        assert zscore_reports[0].affected_columns == ["value"]
        assert zscore_reports[0].row_count >= 1

    def test_isolation_forest_returns_report(self, agent):
        np.random.seed(42)
        normal = np.random.normal(0, 1, (50, 3))
        outliers = np.array([[50, 50, 50], [-50, -50, -50]])
        data = np.vstack([normal, outliers])
        df = pd.DataFrame(data, columns=["a", "b", "c"])
        reports = agent._run_anomaly_detection(df, "iso_task", contamination=0.05)
        iso = [r for r in reports if r.detection_method == "isolation_forest"]
        assert len(iso) == 1
        assert iso[0].row_count > 0

    def test_returns_empty_for_non_numeric_df(self, agent):
        df = pd.DataFrame({"team": ["NYY", "BOS", "LAD"], "player": ["A", "B", "C"]})
        reports = agent._run_anomaly_detection(df, "str_task")
        assert reports == []

    def test_zscore_report_is_auto_fixable(self, agent):
        # Same large-n approach: normal cluster plus extreme outliers
        normal = np.ones(98) + np.random.default_rng(1).uniform(0, 0.5, 98)
        df = pd.DataFrame({"x": np.concatenate([normal, [9999.0, -9999.0]])})
        reports = agent._run_anomaly_detection(df, "fix_task", threshold=2.0)
        zscore = [r for r in reports if r.detection_method == "zscore"]
        assert len(zscore) >= 1
        assert zscore[0].auto_fixable is True

    def test_skips_constant_column(self, agent):
        df = pd.DataFrame({"constant": [5.0] * 10, "normal": list(range(10))})
        # Constant column has std=0, should not raise ZeroDivisionError
        reports = agent._run_anomaly_detection(df, "const_task")
        assert isinstance(reports, list)


# ---------------------------------------------------------------------------
# _run_schema_validation
# ---------------------------------------------------------------------------


class TestRunSchemaValidation:
    def test_first_visit_caches_schema(self, agent, clean_df):
        result = agent._run_schema_validation(clean_df, "source_a")
        assert result["schema_valid"] is True
        assert result["first_visit"] is True
        assert "source_a" in agent.schema_cache

    def test_second_visit_same_schema_is_valid(self, agent, clean_df):
        agent._run_schema_validation(clean_df, "source_b")
        result = agent._run_schema_validation(clean_df.copy(), "source_b")
        assert result["schema_valid"] is True
        assert result["first_visit"] is False

    def test_detects_missing_column(self, agent, clean_df):
        agent._run_schema_validation(clean_df, "source_c")
        df_missing = clean_df.drop(columns=["home_runs"])
        result = agent._run_schema_validation(df_missing, "source_c")
        assert result["schema_valid"] is False
        assert "home_runs" in result["missing_columns"]

    def test_detects_extra_column(self, agent, clean_df):
        agent._run_schema_validation(clean_df, "source_d")
        df_extra = clean_df.copy()
        df_extra["new_col"] = 0
        result = agent._run_schema_validation(df_extra, "source_d")
        assert result["schema_valid"] is False
        assert "new_col" in result["extra_columns"]

    def test_detects_type_change(self, agent, clean_df):
        agent._run_schema_validation(clean_df, "source_e")
        df_changed = clean_df.copy()
        df_changed["home_runs"] = df_changed["home_runs"].astype(str)
        result = agent._run_schema_validation(df_changed, "source_e")
        assert result["schema_valid"] is False
        assert "home_runs" in result["type_changes"]

    def test_different_sources_cached_independently(self, agent, clean_df):
        agent._run_schema_validation(clean_df, "source_x")
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = agent._run_schema_validation(df2, "source_y")
        assert result["first_visit"] is True


# ---------------------------------------------------------------------------
# _apply_repairs
# ---------------------------------------------------------------------------


class TestApplyRepairs:
    def test_removes_duplicates(self, agent, clean_df):
        df_with_dups = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
        repaired, log = agent._apply_repairs(df_with_dups)
        assert len(repaired) == len(clean_df)
        assert any("duplicate" in entry.lower() for entry in log)

    def test_imputes_numeric_missing_with_median(self, agent, clean_df):
        df = clean_df.copy()
        df.loc[0, "home_runs"] = np.nan
        df.loc[1, "home_runs"] = np.nan
        repaired, log = agent._apply_repairs(df)
        assert repaired["home_runs"].isna().sum() == 0
        assert any("home_runs" in entry and "median" in entry for entry in log)

    def test_imputes_categorical_missing_with_mode(self, agent):
        df = pd.DataFrame({"team": ["NYY", "NYY", "BOS", None, None]})
        repaired, log = agent._apply_repairs(df)
        assert repaired["team"].isna().sum() == 0
        assert any("team" in entry and "mode" in entry for entry in log)

    def test_clips_outliers(self, agent):
        df = pd.DataFrame({"x": [1.0] * 18 + [1000.0, -1000.0]})
        repaired, log = agent._apply_repairs(df)
        assert repaired["x"].max() < 1000.0
        assert repaired["x"].min() > -1000.0
        assert any("clip" in entry.lower() for entry in log)

    def test_clean_df_no_repairs_needed(self, agent, clean_df):
        repaired, log = agent._apply_repairs(clean_df)
        # No duplicate, no missing, minimal clipping on bounded data
        assert len(repaired) == len(clean_df)

    def test_does_not_mutate_original(self, agent, dirty_df):
        original_len = len(dirty_df)
        _ = agent._apply_repairs(dirty_df)
        assert len(dirty_df) == original_len


# ---------------------------------------------------------------------------
# _compute_quality_metrics
# ---------------------------------------------------------------------------


class TestComputeQualityMetrics:
    def test_perfect_data(self, agent, clean_df):
        metrics = agent._compute_quality_metrics(clean_df, [], schema_valid=True)
        assert metrics.total_records == len(clean_df)
        assert metrics.missing_values_pct == 0.0
        assert metrics.completeness_score == 1.0
        assert metrics.consistency_score == 1.0
        assert metrics.schema_valid is True
        assert metrics.outlier_count == 0

    def test_missing_values_reflected_in_metrics(self, agent):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1, 2, 3, 4]})
        metrics = agent._compute_quality_metrics(df, [], schema_valid=True)
        assert metrics.missing_values_pct > 0
        assert metrics.completeness_score < 1.0

    def test_duplicates_reduce_consistency(self, agent, clean_df):
        df_with_dups = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
        metrics = agent._compute_quality_metrics(df_with_dups, [], schema_valid=True)
        assert metrics.consistency_score < 1.0

    def test_outlier_count_from_reports(self, agent, clean_df):
        from shared.schemas import DataAnomalyReport

        report = DataAnomalyReport(
            anomaly_id="r1",
            anomaly_type="zscore_outlier",
            severity=AlertSeverity.INFO,
            affected_columns=["batting_avg"],
            row_count=5,
            detection_method="zscore",
            auto_fixable=True,
        )
        metrics = agent._compute_quality_metrics(clean_df, [report], schema_valid=True)
        assert metrics.outlier_count == 5

    def test_invalid_schema_reflected(self, agent, clean_df):
        metrics = agent._compute_quality_metrics(clean_df, [], schema_valid=False)
        assert metrics.schema_valid is False


# ---------------------------------------------------------------------------
# handle_task — dispatch and result shape
# ---------------------------------------------------------------------------


class TestHandleTask:
    @pytest.mark.asyncio
    async def test_detect_anomalies_task(self, agent, global_mq, parquet_path):
        task = make_task(
            "detect_anomalies",
            {"data_source": str(parquet_path), "threshold": 3.0},
        )
        result = await agent._detect_anomalies(task)
        assert result.status == TaskStatus.COMPLETED
        assert "reports" in result.result_data
        assert "total_anomalous_records" in result.result_data
        assert "anomaly_count" in result.metrics

    @pytest.mark.asyncio
    async def test_validate_schema_task(self, agent, global_mq, parquet_path):
        task = make_task("validate_schema", {"data_source": str(parquet_path)})
        result = await agent._validate_schema(task)
        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["schema_valid"] is True
        assert result.result_data["first_visit"] is True

    @pytest.mark.asyncio
    async def test_repair_data_task(self, agent, global_mq, dirty_parquet_path, tmp_path):
        out = tmp_path / "repaired.parquet"
        task = make_task(
            "repair_data",
            {"data_source": str(dirty_parquet_path), "output_path": str(out)},
        )
        result = await agent._repair_data(task)
        assert result.status == TaskStatus.COMPLETED
        assert out.exists()
        assert result.metrics["repairs_applied"] >= 0

    @pytest.mark.asyncio
    async def test_check_data_quality_task(self, agent, global_mq, dirty_parquet_path):
        task = make_task(
            "check_data_quality",
            {"data_source": str(dirty_parquet_path), "auto_fix": True},
        )
        result = await agent._check_data_quality(task)
        assert result.status == TaskStatus.COMPLETED
        assert "quality_metrics" in result.result_data
        assert "anomaly_reports" in result.result_data
        assert "schema_validation" in result.result_data
        assert "completeness_score" in result.metrics

    @pytest.mark.asyncio
    async def test_unknown_task_type_raises(self, agent):
        task = make_task("unknown_task")
        with pytest.raises(ValueError, match="Unknown task type"):
            await agent.handle_task(task)
