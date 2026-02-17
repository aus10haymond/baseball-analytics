"""
Structured logging utilities for Diamond Mind agents.

Provides consistent logging across all agents with JSON formatting
for easy parsing and analysis.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import json

from shared.config import settings


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


def setup_logging(
    agent_name: str,
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    use_json: bool = True
) -> logging.Logger:
    """
    Set up logging for an agent.
    
    Args:
        agent_name: Name of the agent (used as logger name)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, uses settings.
        use_json: Whether to use JSON formatting
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(agent_name)
    
    # Set level
    level = log_level or settings.log_level
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    if use_json:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file or settings.log_file:
        file_path = log_file or settings.log_file
        if file_path:
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(logging.DEBUG)
            
            if use_json:
                file_formatter = JSONFormatter()
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    
    # Also log to agent-specific file
    agent_log_dir = settings.get_logs_dir() / agent_name
    agent_log_dir.mkdir(exist_ok=True)
    agent_log_file = agent_log_dir / f"{agent_name}.log"
    
    agent_file_handler = logging.FileHandler(agent_log_file)
    agent_file_handler.setLevel(logging.DEBUG)
    
    if use_json:
        agent_file_formatter = JSONFormatter()
    else:
        agent_file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
    agent_file_handler.setFormatter(agent_file_formatter)
    logger.addHandler(agent_file_handler)
    
    return logger


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    **context: Any
):
    """
    Log a message with additional context fields.
    
    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        **context: Additional context fields to include
    """
    log_record = logger.makeRecord(
        logger.name,
        getattr(logging, level.upper()),
        "",
        0,
        message,
        (),
        None
    )
    log_record.extra_fields = context
    logger.handle(log_record)


# Convenience functions
def get_agent_logger(agent_name: str) -> logging.Logger:
    """Get or create a logger for an agent."""
    return setup_logging(agent_name)
