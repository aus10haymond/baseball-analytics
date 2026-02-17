"""
Pydantic schemas for agent communication and data validation.

All messages between agents use these strongly-typed schemas to ensure
data integrity and enable validation at boundaries.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================
# Enums
# ============================================

class AgentType(str, Enum):
    """Types of agents in the system."""
    ORCHESTRATOR = "orchestrator"
    DATA_QUALITY = "data_quality"
    MODEL_MONITOR = "model_monitor"
    FEATURE_ENGINEER = "feature_engineer"
    EXPLAINER = "explainer"


class TaskStatus(str, Enum):
    """Status of agent tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ConfidenceLevel(str, Enum):
    """Confidence levels for predictions and decisions."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ============================================
# Task Messages
# ============================================

class AgentTask(BaseModel):
    """Task to be executed by an agent."""
    task_id: str = Field(..., description="Unique task identifier")
    agent_id: AgentType = Field(..., description="Target agent")
    task_type: str = Field(..., description="Type of task to execute")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    timeout_seconds: Optional[int] = Field(default=None)
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_123",
                "agent_id": "data_quality",
                "task_type": "check_anomalies",
                "priority": "high",
                "parameters": {"data_source": "statcast"}
            }
        }


class AgentResult(BaseModel):
    """Result from an agent task execution."""
    task_id: str
    agent_id: AgentType
    status: TaskStatus
    result_data: Optional[Dict[str, Any]] = Field(default=None)
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list, description="Paths to generated artifacts")
    duration_seconds: float = Field(..., ge=0)
    completed_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = Field(default=None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_123",
                "agent_id": "data_quality",
                "status": "completed",
                "metrics": {"issues_found": 3, "auto_fixed": 2},
                "duration_seconds": 45.2
            }
        }


# ============================================
# Alert Messages
# ============================================

class AgentAlert(BaseModel):
    """Alert raised by an agent."""
    alert_id: str
    agent_id: AgentType
    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    requires_action: bool = Field(default=False)
    suggested_actions: List[str] = Field(default_factory=list)
    related_task_id: Optional[str] = Field(default=None)


# ============================================
# Data Quality Specific
# ============================================

class DataAnomalyReport(BaseModel):
    """Report of data quality issues detected."""
    anomaly_id: str
    anomaly_type: str = Field(..., description="Type of anomaly (e.g., 'missing_values', 'outlier', 'schema_change')")
    severity: AlertSeverity
    affected_columns: List[str] = Field(default_factory=list)
    row_count: int = Field(..., ge=0)
    detection_method: str
    auto_fixable: bool
    fix_applied: bool = Field(default=False)
    fix_description: Optional[str] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.now)


class DataQualityMetrics(BaseModel):
    """Metrics for data quality assessment."""
    total_records: int = Field(..., ge=0)
    missing_values_pct: float = Field(..., ge=0, le=100)
    outlier_count: int = Field(default=0, ge=0)
    schema_valid: bool
    completeness_score: float = Field(..., ge=0, le=1)
    consistency_score: float = Field(..., ge=0, le=1)
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================
# Model Monitor Specific
# ============================================

class DriftDetectionResult(BaseModel):
    """Result from concept drift detection."""
    drift_detected: bool
    drift_score: float = Field(..., ge=0)
    drift_type: str = Field(..., description="e.g., 'data_drift', 'concept_drift', 'prediction_drift'")
    affected_features: List[str] = Field(default_factory=list)
    psi_scores: Dict[str, float] = Field(default_factory=dict, description="Population Stability Index per feature")
    ks_statistics: Dict[str, float] = Field(default_factory=dict, description="KS statistics per feature")
    recommendation: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelPerformanceMetrics(BaseModel):
    """Model performance tracking."""
    model_name: str
    model_version: str
    accuracy: float = Field(..., ge=0, le=1)
    auc: Optional[float] = Field(default=None, ge=0, le=1)
    precision: Optional[float] = Field(default=None, ge=0, le=1)
    recall: Optional[float] = Field(default=None, ge=0, le=1)
    f1_score: Optional[float] = Field(default=None, ge=0, le=1)
    prediction_count: int = Field(..., ge=0)
    avg_prediction_time_ms: float = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @field_validator('auc', 'precision', 'recall', 'f1_score')
    @classmethod
    def validate_optional_metrics(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError('Metric must be between 0 and 1')
        return v


# ============================================
# Feature Engineer Specific
# ============================================

class FeatureCandidate(BaseModel):
    """Candidate feature for evaluation."""
    feature_name: str
    feature_definition: str = Field(..., description="Code or formula defining the feature")
    feature_type: str = Field(..., description="e.g., 'rolling_avg', 'interaction', 'polynomial'")
    source_features: List[str] = Field(default_factory=list)
    importance_score: Optional[float] = Field(default=None, ge=0)
    performance_gain: Optional[float] = Field(default=None)
    correlation_with_target: Optional[float] = Field(default=None, ge=-1, le=1)
    multicollinearity_vif: Optional[float] = Field(default=None)
    validation_passed: bool = Field(default=False)


class FeatureSearchResult(BaseModel):
    """Result from automated feature search."""
    search_id: str
    generation: int = Field(..., ge=1)
    candidates_evaluated: int = Field(..., ge=0)
    features_added: List[FeatureCandidate] = Field(default_factory=list)
    best_model_score: float
    baseline_model_score: float
    improvement_pct: float
    search_duration_seconds: float = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================
# Explainer Specific
# ============================================

class PredictionExplanation(BaseModel):
    """Explanation for a single prediction."""
    prediction_id: str
    player_name: str
    predicted_value: float
    confidence: ConfidenceLevel
    top_features: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top contributing features with SHAP values"
    )
    narrative_explanation: str = Field(..., description="Human-readable explanation")
    counterfactuals: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================
# Orchestrator Specific
# ============================================

class AgentHealthStatus(BaseModel):
    """Health status of an agent."""
    agent_id: AgentType
    is_healthy: bool
    uptime_seconds: float = Field(..., ge=0)
    tasks_completed: int = Field(default=0, ge=0)
    tasks_failed: int = Field(default=0, ge=0)
    avg_task_duration_seconds: float = Field(default=0, ge=0)
    last_heartbeat: datetime = Field(default_factory=datetime.now)
    error_rate: float = Field(default=0, ge=0, le=1)


class SystemStatus(BaseModel):
    """Overall system health status."""
    all_agents_healthy: bool
    agent_statuses: List[AgentHealthStatus]
    total_tasks_pending: int = Field(default=0, ge=0)
    total_tasks_running: int = Field(default=0, ge=0)
    message_queue_depth: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)
