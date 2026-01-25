"""
Structured logging configuration for orchestrator events (ATOM-109).

Provides JSON Lines formatted logging for easier parsing and analysis.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON Lines for structured logging."""

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        extra_fields: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {}

        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if self.include_level:
            log_data["level"] = record.levelname

        if self.include_logger:
            log_data["logger"] = record.name

        log_data["message"] = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields from the record
        for key in ["task_id", "agent_id", "event_type", "duration_ms", "status"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        # Add configured extra fields
        log_data.update(self.extra_fields)

        return json.dumps(log_data, default=str)


class OrchestratorLogger:
    """Structured logger for orchestrator events."""

    def __init__(
        self,
        name: str = "orchestrator",
        log_file: Optional[Path] = None,
        json_format: bool = True,
        level: int = logging.INFO,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.json_format = json_format

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatter
        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        # Console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_event(
        self,
        event_type: str,
        message: str,
        level: int = logging.INFO,
        **kwargs: Any,
    ) -> None:
        """Log an orchestrator event with structured data."""
        extra = {"event_type": event_type, **kwargs}
        self.logger.log(level, message, extra=extra)

    def task_created(self, task_id: str, description: str, priority: int) -> None:
        """Log task creation event."""
        self.log_event(
            "task_created",
            f"Task created: {task_id}",
            task_id=task_id,
            description=description[:100],  # Truncate for log
            priority=priority,
        )

    def task_claimed(self, task_id: str, agent_id: str) -> None:
        """Log task claim event."""
        self.log_event(
            "task_claimed",
            f"Task {task_id} claimed by {agent_id}",
            task_id=task_id,
            agent_id=agent_id,
        )

    def task_completed(
        self,
        task_id: str,
        agent_id: str,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Log task completion event."""
        self.log_event(
            "task_completed",
            f"Task {task_id} completed by {agent_id}",
            task_id=task_id,
            agent_id=agent_id,
            duration_ms=duration_ms,
            status="done",
        )

    def task_failed(
        self,
        task_id: str,
        agent_id: str,
        error: str,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Log task failure event."""
        self.log_event(
            "task_failed",
            f"Task {task_id} failed: {error}",
            level=logging.ERROR,
            task_id=task_id,
            agent_id=agent_id,
            error=error[:200],  # Truncate for log
            duration_ms=duration_ms,
            status="failed",
        )

    def agent_registered(self, agent_id: str, capabilities: list[str]) -> None:
        """Log agent registration event."""
        self.log_event(
            "agent_registered",
            f"Agent registered: {agent_id}",
            agent_id=agent_id,
            capabilities=capabilities,
        )

    def agent_heartbeat(self, agent_id: str) -> None:
        """Log agent heartbeat event."""
        self.log_event(
            "agent_heartbeat",
            f"Agent heartbeat: {agent_id}",
            level=logging.DEBUG,
            agent_id=agent_id,
        )

    def agent_deregistered(self, agent_id: str) -> None:
        """Log agent deregistration event."""
        self.log_event(
            "agent_deregistered",
            f"Agent deregistered: {agent_id}",
            agent_id=agent_id,
        )

    def orchestration_error(self, error: str, context: Optional[dict[str, Any]] = None) -> None:
        """Log orchestration error."""
        self.log_event(
            "orchestration_error",
            f"Orchestration error: {error}",
            level=logging.ERROR,
            error=error,
            **(context or {}),
        )


def configure_structured_logging(
    log_dir: Optional[Path] = None,
    json_format: bool = True,
    level: int = logging.INFO,
) -> OrchestratorLogger:
    """
    Configure structured logging for the orchestrator.

    Args:
        log_dir: Directory for log files. If None, logs only to stderr.
        json_format: If True, use JSON Lines format.
        level: Logging level.

    Returns:
        Configured OrchestratorLogger instance.
    """
    log_file = None
    if log_dir:
        log_file = log_dir / "orchestrator.jsonl" if json_format else log_dir / "orchestrator.log"

    return OrchestratorLogger(
        name="orchestrator",
        log_file=log_file,
        json_format=json_format,
        level=level,
    )
