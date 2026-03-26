"""
Data Quality Agent

Monitors data pipelines for anomalies, schema changes, and quality issues.
Automatically repairs common problems and alerts on critical issues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from shared.base_agent import BaseAgent
from shared import AgentType, AgentTask, AgentResult, TaskStatus, AlertSeverity
from shared.schemas import DataAnomalyReport, DataQualityMetrics


class DataQualityAgent(BaseAgent):
    """Agent responsible for monitoring and maintaining data quality."""

    def __init__(self):
        super().__init__(AgentType.DATA_QUALITY)
        # Keyed by data_source path string; stores {col: dtype_str}
        self.schema_cache: Dict[str, Dict[str, str]] = {}
        # Keyed by data_source path string; stores per-column stats for baseline
        self.baseline_stats: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """Initialize the Data Quality Agent."""
        self.logger.info("Data Quality Agent initialized")

    async def cleanup(self):
        """Clean up resources."""
        self.logger.info("Data Quality Agent cleaned up")

    # ------------------------------------------------------------------
    # Task dispatch
    # ------------------------------------------------------------------

    async def handle_task(self, task: AgentTask) -> AgentResult:
        """Route tasks to the appropriate handler."""
        handlers = {
            "check_data_quality": self._check_data_quality,
            "detect_anomalies": self._detect_anomalies,
            "validate_schema": self._validate_schema,
            "repair_data": self._repair_data,
        }
        handler = handlers.get(task.task_type)
        if handler is None:
            raise ValueError(f"Unknown task type: {task.task_type}")
        return await handler(task)

    # ------------------------------------------------------------------
    # Public task handlers
    # ------------------------------------------------------------------

    async def _check_data_quality(self, task: AgentTask) -> AgentResult:
        """Comprehensive data quality check: anomalies + schema + metrics."""
        data_source = task.parameters.get("data_source", "")
        auto_fix = task.parameters.get("auto_fix", False)
        threshold = task.parameters.get("threshold", 3.0)
        contamination = task.parameters.get("contamination", 0.05)

        df = self._load_data(data_source)

        anomaly_reports = self._run_anomaly_detection(
            df, task.task_id, threshold=threshold, contamination=contamination
        )
        schema_report = self._run_schema_validation(df, data_source)

        repair_log: List[str] = []
        if auto_fix:
            df, repair_log = self._apply_repairs(df)

        metrics = self._compute_quality_metrics(df, anomaly_reports, schema_report["schema_valid"])

        total_anomalies = sum(r.row_count for r in anomaly_reports)
        if not schema_report["schema_valid"] or metrics.completeness_score < 0.9:
            await self.publish_alert(
                severity=AlertSeverity.WARNING,
                message=f"Data quality issues detected in {data_source}",
                details={
                    "completeness_score": metrics.completeness_score,
                    "schema_valid": schema_report["schema_valid"],
                    "total_anomalies": total_anomalies,
                },
                requires_action=True,
                suggested_actions=["Run repair_data task", "Review data pipeline"],
            )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "quality_metrics": metrics.model_dump(mode="json"),
                "anomaly_reports": [r.model_dump(mode="json") for r in anomaly_reports],
                "schema_validation": schema_report,
                "repair_log": repair_log,
            },
            metrics={
                "completeness_score": metrics.completeness_score,
                "consistency_score": metrics.consistency_score,
                "total_anomalies": float(total_anomalies),
                "missing_values_pct": metrics.missing_values_pct,
            },
            duration_seconds=0.0,
        )

    async def _detect_anomalies(self, task: AgentTask) -> AgentResult:
        """Detect anomalies using Isolation Forest and Z-score methods."""
        data_source = task.parameters.get("data_source", "")
        threshold = task.parameters.get("threshold", 3.0)
        contamination = task.parameters.get("contamination", 0.05)

        df = self._load_data(data_source)
        reports = self._run_anomaly_detection(
            df, task.task_id, threshold=threshold, contamination=contamination
        )

        total_anomalous = sum(r.row_count for r in reports)
        if total_anomalous > 0:
            severity = AlertSeverity.ERROR if total_anomalous > 100 else AlertSeverity.WARNING
            await self.publish_alert(
                severity=severity,
                message=f"Detected {total_anomalous} anomalous records in {data_source}",
                details={"methods": [r.detection_method for r in reports]},
            )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "reports": [r.model_dump(mode="json") for r in reports],
                "total_anomalous_records": total_anomalous,
                "data_source": data_source,
            },
            metrics={"anomaly_count": float(total_anomalous)},
            duration_seconds=0.0,
        )

    async def _validate_schema(self, task: AgentTask) -> AgentResult:
        """Validate data schema against cached baseline; cache on first visit."""
        data_source = task.parameters.get("data_source", "")
        df = self._load_data(data_source)
        schema_report = self._run_schema_validation(df, data_source)

        if not schema_report["schema_valid"]:
            await self.publish_alert(
                severity=AlertSeverity.WARNING,
                message=f"Schema drift detected in {data_source}",
                details=schema_report,
                requires_action=True,
                suggested_actions=["Update schema cache", "Fix upstream data pipeline"],
            )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data=schema_report,
            metrics={"schema_valid": float(schema_report["schema_valid"])},
            duration_seconds=0.0,
        )

    async def _repair_data(self, task: AgentTask) -> AgentResult:
        """Repair common data quality issues: impute missing values, remove duplicates, clip outliers."""
        data_source = task.parameters.get("data_source", "")
        output_path = task.parameters.get("output_path", None)

        df = self._load_data(data_source)
        original_len = len(df)

        df_repaired, repair_log = self._apply_repairs(df)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df_repaired.to_parquet(output_path, index=False)

        rows_removed = original_len - len(df_repaired)
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.COMPLETED,
            result_data={
                "repair_log": repair_log,
                "original_row_count": original_len,
                "repaired_row_count": len(df_repaired),
                "rows_removed": rows_removed,
                "output_path": output_path,
            },
            metrics={
                "repairs_applied": float(len(repair_log)),
                "rows_removed": float(rows_removed),
            },
            duration_seconds=0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers — synchronous, pure functions for easy testing
    # ------------------------------------------------------------------

    def _load_data(self, data_source: str) -> pd.DataFrame:
        """Load a DataFrame from a parquet or CSV file path."""
        path = Path(data_source)
        if not path.exists():
            raise FileNotFoundError(f"Data source not found: {data_source}")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported file format: {path.suffix!r}. Use .parquet or .csv.")

    def _run_anomaly_detection(
        self,
        df: pd.DataFrame,
        task_id: str,
        threshold: float = 3.0,
        contamination: float = 0.05,
    ) -> List[DataAnomalyReport]:
        """Run Isolation Forest and Z-score anomaly detection on numeric columns."""
        reports: List[DataAnomalyReport] = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return reports

        X = df[numeric_cols].fillna(df[numeric_cols].median())

        # Isolation Forest (multivariate)
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(X)
        iso_count = int((preds == -1).sum())
        if iso_count > 0:
            reports.append(
                DataAnomalyReport(
                    anomaly_id=f"iso_forest_{task_id}",
                    anomaly_type="isolation_forest_outliers",
                    severity=AlertSeverity.WARNING,
                    affected_columns=numeric_cols,
                    row_count=iso_count,
                    detection_method="isolation_forest",
                    auto_fixable=False,
                )
            )

        # Z-score per column
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 2:
                continue
            std = col_data.std()
            if std == 0:
                continue
            z_scores = ((col_data - col_data.mean()) / std).abs()
            z_count = int((z_scores > threshold).sum())
            if z_count > 0:
                reports.append(
                    DataAnomalyReport(
                        anomaly_id=f"zscore_{col}_{task_id}",
                        anomaly_type="zscore_outlier",
                        severity=AlertSeverity.INFO,
                        affected_columns=[col],
                        row_count=z_count,
                        detection_method="zscore",
                        auto_fixable=True,
                        fix_description=f"Clip values beyond {threshold} std devs",
                    )
                )

        return reports

    def _run_schema_validation(
        self, df: pd.DataFrame, data_source: str
    ) -> Dict[str, Any]:
        """
        Compare df schema to cached baseline.  On first visit, cache and return valid.
        Returns a dict with schema_valid, missing_columns, extra_columns, type_changes.
        """
        current_schema = {col: str(dtype) for col, dtype in df.dtypes.items()}

        if data_source not in self.schema_cache:
            self.schema_cache[data_source] = current_schema
            return {
                "schema_valid": True,
                "missing_columns": [],
                "extra_columns": [],
                "type_changes": {},
                "first_visit": True,
            }

        expected = self.schema_cache[data_source]
        missing = sorted(set(expected) - set(current_schema))
        extra = sorted(set(current_schema) - set(expected))
        type_changes = {
            col: {"expected": expected[col], "actual": current_schema[col]}
            for col in set(expected) & set(current_schema)
            if expected[col] != current_schema[col]
        }

        schema_valid = not (missing or extra or type_changes)
        return {
            "schema_valid": schema_valid,
            "missing_columns": missing,
            "extra_columns": extra,
            "type_changes": type_changes,
            "first_visit": False,
        }

    def _apply_repairs(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply repairs in-place on a copy:
          1. Remove duplicate rows
          2. Impute missing numerics with column median
          3. Impute missing categoricals with column mode
          4. Clip numeric outliers at [1st, 99th] percentile

        Returns (repaired_df, repair_log).
        """
        df = df.copy()
        log: List[str] = []

        # 1. Duplicates
        n_dups = int(df.duplicated().sum())
        if n_dups:
            df = df.drop_duplicates()
            log.append(f"Removed {n_dups} duplicate rows")

        # 2 & 3. Missing value imputation
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                log.append(f"Imputed {n_missing} missing values in '{col}' with median ({fill_val:.4g})")
            else:
                mode_vals = df[col].mode()
                if len(mode_vals) > 0:
                    df[col] = df[col].fillna(mode_vals.iloc[0])
                    log.append(f"Imputed {n_missing} missing values in '{col}' with mode ('{mode_vals.iloc[0]}')")

        # 4. Outlier clipping
        for col in df.select_dtypes(include=[np.number]).columns:
            p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
            n_clipped = int(((df[col] < p1) | (df[col] > p99)).sum())
            if n_clipped > 0:
                df[col] = df[col].clip(lower=p1, upper=p99)
                log.append(f"Clipped {n_clipped} outliers in '{col}' to [{p1:.4g}, {p99:.4g}]")

        return df, log

    def _compute_quality_metrics(
        self,
        df: pd.DataFrame,
        anomaly_reports: List[DataAnomalyReport],
        schema_valid: bool,
    ) -> DataQualityMetrics:
        """Compute DataQualityMetrics from a DataFrame and prior analysis results."""
        total = len(df)
        total_cells = df.size or 1

        missing_pct = float(df.isna().sum().sum() / total_cells * 100)
        outlier_count = sum(r.row_count for r in anomaly_reports)

        # Completeness: fraction of non-null cells
        completeness = float(1 - df.isna().sum().sum() / total_cells)

        # Consistency: fraction of non-duplicate rows
        n_dups = int(df.duplicated().sum())
        consistency = float(1 - n_dups / max(total, 1))

        return DataQualityMetrics(
            total_records=total,
            missing_values_pct=round(missing_pct, 4),
            outlier_count=outlier_count,
            schema_valid=schema_valid,
            completeness_score=round(completeness, 4),
            consistency_score=round(consistency, 4),
        )


if __name__ == "__main__":
    import asyncio
    from shared.messaging import init_messaging, shutdown_messaging

    async def main():
        await init_messaging()
        agent = DataQualityAgent()
        try:
            await agent.start()
        except KeyboardInterrupt:
            await agent.stop()
        finally:
            await shutdown_messaging()

    asyncio.run(main())
