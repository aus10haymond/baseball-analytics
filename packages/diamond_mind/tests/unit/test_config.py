"""Tests for shared/config.py – Configuration management."""

import pytest
from pathlib import Path

from shared.config import Settings, get_redis_url, ensure_directories


# ── Default Settings ───────────────────────────────────────────────────────


class TestSettingsDefaults:
    def test_redis_defaults(self):
        s = Settings()
        assert s.redis_host == "localhost"
        assert s.redis_port == 6379
        assert s.redis_db == 0
        assert s.redis_password is None

    def test_queue_name_defaults(self):
        s = Settings()
        assert s.task_queue_name == "diamond_mind:tasks"
        assert s.result_queue_name == "diamond_mind:results"
        assert s.alert_queue_name == "diamond_mind:alerts"

    def test_agent_flags_enabled(self):
        s = Settings()
        assert s.orchestrator_enabled is True
        assert s.data_quality_enabled is True
        assert s.model_monitor_enabled is True
        assert s.feature_engineer_enabled is True
        assert s.explainer_enabled is True

    def test_heartbeat_defaults(self):
        s = Settings()
        assert s.heartbeat_interval_seconds == 60
        assert s.health_check_timeout_seconds == 10

    def test_task_execution_defaults(self):
        s = Settings()
        assert s.default_task_timeout_seconds == 300
        assert s.max_concurrent_tasks == 5

    def test_llm_defaults(self):
        s = Settings()
        assert s.llm_provider == "openai"
        assert s.llm_model == "gpt-4"
        assert s.llm_api_key is None
        assert s.llm_temperature == 0.7
        assert s.llm_max_tokens == 500

    def test_log_defaults(self):
        s = Settings()
        assert s.log_level == "INFO"
        assert s.log_file is None

    def test_debug_defaults(self):
        s = Settings()
        assert s.debug_mode is False
        assert s.enable_profiling is False


# ── Environment Variable Override ──────────────────────────────────────────


class TestSettingsEnvOverride:
    def test_override_redis_host(self, monkeypatch):
        monkeypatch.setenv("DM_REDIS_HOST", "redis.example.com")
        s = Settings()
        assert s.redis_host == "redis.example.com"

    def test_override_redis_port(self, monkeypatch):
        monkeypatch.setenv("DM_REDIS_PORT", "6380")
        s = Settings()
        assert s.redis_port == 6380

    def test_override_log_level(self, monkeypatch):
        monkeypatch.setenv("DM_LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"

    def test_override_debug_mode(self, monkeypatch):
        monkeypatch.setenv("DM_DEBUG_MODE", "true")
        s = Settings()
        assert s.debug_mode is True

    def test_override_heartbeat_interval(self, monkeypatch):
        monkeypatch.setenv("DM_HEARTBEAT_INTERVAL_SECONDS", "30")
        s = Settings()
        assert s.heartbeat_interval_seconds == 30


# ── Redis URL ──────────────────────────────────────────────────────────────


class TestRedisUrl:
    def test_default_url(self, monkeypatch):
        from shared.config import settings

        monkeypatch.setattr(settings, "redis_host", "localhost")
        monkeypatch.setattr(settings, "redis_port", 6379)
        monkeypatch.setattr(settings, "redis_db", 0)
        monkeypatch.setattr(settings, "redis_password", None)

        url = get_redis_url()
        assert url == "redis://localhost:6379/0"

    def test_url_with_password(self, monkeypatch):
        from shared.config import settings

        monkeypatch.setattr(settings, "redis_host", "redis.prod")
        monkeypatch.setattr(settings, "redis_port", 6380)
        monkeypatch.setattr(settings, "redis_db", 2)
        monkeypatch.setattr(settings, "redis_password", "s3cret")

        url = get_redis_url()
        assert url == "redis://:s3cret@redis.prod:6380/2"


# ── Directory helpers ──────────────────────────────────────────────────────


class TestDirectoryHelpers:
    def test_get_sister_project_fantasy(self):
        s = Settings()
        path = s.get_sister_project_path("fantasy_mlb_ai")
        assert "fantasy_mlb_ai" in str(path)

    def test_get_sister_project_matchup(self):
        s = Settings()
        path = s.get_sister_project_path("matchup_machine")
        assert "matchup_machine" in str(path)

    def test_get_sister_project_invalid(self):
        s = Settings()
        with pytest.raises(ValueError, match="Unknown project"):
            s.get_sister_project_path("nonexistent")

    def test_get_logs_dir(self, tmp_path):
        s = Settings()
        s.project_root = tmp_path
        logs_dir = s.get_logs_dir()
        assert logs_dir.exists()
        assert logs_dir == tmp_path / "logs"

    def test_get_data_dir(self, tmp_path):
        s = Settings()
        s.project_root = tmp_path
        data_dir = s.get_data_dir()
        assert data_dir.exists()
        assert data_dir == tmp_path / "data"

    def test_get_models_dir(self, tmp_path):
        s = Settings()
        s.project_root = tmp_path
        models_dir = s.get_models_dir()
        assert models_dir.exists()
        assert models_dir == tmp_path / "models"

    def test_ensure_directories(self, tmp_path, monkeypatch):
        from shared.config import settings as global_settings

        monkeypatch.setattr(global_settings, "project_root", tmp_path)
        ensure_directories()

        for name in ("logs", "data", "models", "reports", "agent_reports"):
            assert (tmp_path / name).exists()
