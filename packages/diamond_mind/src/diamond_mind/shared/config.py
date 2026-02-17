"""
Configuration management for Diamond Mind agents.

Loads configuration from environment variables and provides
type-safe access to settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Project paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Root directory of the project"
    )
    
    # Sister project paths (for integration)
    fantasy_mlb_path: Optional[Path] = Field(
        default=None,
        description="Path to fantasy_mlb_ai project"
    )
    matchup_machine_path: Optional[Path] = Field(
        default=None,
        description="Path to matchup_machine project"
    )
    
    # Redis configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    
    # Task queue configuration
    task_queue_name: str = Field(default="diamond_mind:tasks", description="Redis queue for tasks")
    result_queue_name: str = Field(default="diamond_mind:results", description="Redis queue for results")
    alert_queue_name: str = Field(default="diamond_mind:alerts", description="Redis queue for alerts")
    
    # Agent configuration
    orchestrator_enabled: bool = Field(default=True)
    data_quality_enabled: bool = Field(default=True)
    model_monitor_enabled: bool = Field(default=True)
    feature_engineer_enabled: bool = Field(default=True)
    explainer_enabled: bool = Field(default=True)
    
    # Heartbeat and health checks
    heartbeat_interval_seconds: int = Field(default=60, description="Agent heartbeat interval")
    health_check_timeout_seconds: int = Field(default=10, description="Health check timeout")
    
    # Task execution
    default_task_timeout_seconds: int = Field(default=300, description="Default task timeout")
    max_concurrent_tasks: int = Field(default=5, description="Max concurrent tasks per agent")
    task_retry_delay_seconds: int = Field(default=5, description="Delay between retries")
    
    # Data quality agent settings
    dq_check_interval_hours: int = Field(default=1, description="Data quality check interval")
    dq_anomaly_threshold: float = Field(default=3.0, description="Anomaly detection threshold (std devs)")
    
    # Model monitor settings
    mm_check_interval_hours: int = Field(default=24, description="Model monitoring check interval")
    mm_drift_threshold: float = Field(default=0.15, description="PSI threshold for drift detection")
    mm_performance_threshold: float = Field(default=0.05, description="Performance degradation threshold")
    
    # Feature engineer settings
    fe_search_enabled: bool = Field(default=True, description="Enable automated feature search")
    fe_max_features_per_search: int = Field(default=50, description="Max features to evaluate per search")
    fe_search_interval_days: int = Field(default=7, description="Feature search interval")
    
    # Explainer settings
    explainer_shap_samples: int = Field(default=100, description="Samples for SHAP explanations")
    explainer_use_llm: bool = Field(default=True, description="Use LLM for narrative generation")
    
    # LLM configuration (for orchestrator and explainer)
    llm_provider: str = Field(default="openai", description="LLM provider (openai, anthropic, etc.)")
    llm_model: str = Field(default="gpt-4", description="LLM model name")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API key")
    llm_temperature: float = Field(default=0.7, description="LLM temperature")
    llm_max_tokens: int = Field(default=500, description="Max tokens for LLM responses")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    log_rotation: str = Field(default="1 day", description="Log rotation interval")
    log_retention: str = Field(default="30 days", description="Log retention period")
    
    # Database (for persistence)
    database_url: str = Field(
        default="sqlite:///diamond_mind.db",
        description="Database connection URL"
    )
    
    # Monitoring
    enable_prometheus: bool = Field(default=False, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    
    # Development/Debug
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")
    
    model_config = SettingsConfigDict(
        env_prefix="DM_",  # All env vars start with DM_
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    def get_sister_project_path(self, project_name: str) -> Path:
        """Get path to a sister project (fantasy_mlb_ai or matchup_machine)."""
        if project_name == "fantasy_mlb_ai":
            if self.fantasy_mlb_path:
                return self.fantasy_mlb_path
            # Default: sibling directory
            return self.project_root.parent / "fantasy_mlb_ai"
        elif project_name == "matchup_machine":
            if self.matchup_machine_path:
                return self.matchup_machine_path
            # Default: sibling directory
            return self.project_root.parent / "matchup_machine"
        else:
            raise ValueError(f"Unknown project: {project_name}")
    
    def get_logs_dir(self) -> Path:
        """Get logs directory path."""
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        return logs_dir
    
    def get_data_dir(self) -> Path:
        """Get data directory path."""
        data_dir = self.project_root / "data"
        data_dir.mkdir(exist_ok=True)
        return data_dir
    
    def get_models_dir(self) -> Path:
        """Get models directory path."""
        models_dir = self.project_root / "models"
        models_dir.mkdir(exist_ok=True)
        return models_dir


# Global settings instance
settings = Settings()


# Helper functions
def get_redis_url() -> str:
    """Get Redis connection URL."""
    auth = f":{settings.redis_password}@" if settings.redis_password else ""
    return f"redis://{auth}{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        settings.get_logs_dir(),
        settings.get_data_dir(),
        settings.get_models_dir(),
        settings.project_root / "reports",
        settings.project_root / "agent_reports",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
