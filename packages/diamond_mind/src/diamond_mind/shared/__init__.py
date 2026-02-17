"""Shared utilities for Diamond Mind agents."""

from shared.schemas import (
    AgentType,
    TaskStatus,
    TaskPriority,
    AlertSeverity,
    ConfidenceLevel,
    AgentTask,
    AgentResult,
    AgentAlert,
    DataAnomalyReport,
    DataQualityMetrics,
    DriftDetectionResult,
    ModelPerformanceMetrics,
    FeatureCandidate,
    FeatureSearchResult,
    PredictionExplanation,
    AgentHealthStatus,
    SystemStatus,
)

from shared.config import settings, ensure_directories
from shared.logging_utils import get_agent_logger, log_with_context
from shared.messaging import message_queue, init_messaging, shutdown_messaging

__all__ = [
    # Schemas
    "AgentType",
    "TaskStatus",
    "TaskPriority",
    "AlertSeverity",
    "ConfidenceLevel",
    "AgentTask",
    "AgentResult",
    "AgentAlert",
    "DataAnomalyReport",
    "DataQualityMetrics",
    "DriftDetectionResult",
    "ModelPerformanceMetrics",
    "FeatureCandidate",
    "FeatureSearchResult",
    "PredictionExplanation",
    "AgentHealthStatus",
    "SystemStatus",
    # Config
    "settings",
    "ensure_directories",
    # Logging
    "get_agent_logger",
    "log_with_context",
    # Messaging
    "message_queue",
    "init_messaging",
    "shutdown_messaging",
]
