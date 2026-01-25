"""
Structured logging configuration for orchestrator events (ATOM-109, OBS-004).

Provides JSON Lines formatted logging for easier parsing and analysis,
with correlation ID support for distributed tracing.

Features:
- JSON Lines format for structured logging
- Correlation ID generation and propagation
- Thread-local context for correlation IDs
- Integration with OpenTelemetry trace context
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar

# Thread-local and context-var storage for correlation IDs
_correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)
_task_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "task_id", default=None
)
_agent_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "agent_id", default=None
)

T = TypeVar("T")


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return f"corr-{uuid.uuid4().hex[:16]}"


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context."""
    return _correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> contextvars.Token:
    """Set the correlation ID in context."""
    return _correlation_id_var.set(correlation_id)


def clear_correlation_id(token: contextvars.Token) -> None:
    """Clear the correlation ID using the token from set_correlation_id."""
    _correlation_id_var.reset(token)


def get_task_id() -> Optional[str]:
    """Get the current task ID from context."""
    return _task_id_var.get()


def set_task_id(task_id: str) -> contextvars.Token:
    """Set the task ID in context."""
    return _task_id_var.set(task_id)


def get_agent_id() -> Optional[str]:
    """Get the current agent ID from context."""
    return _agent_id_var.get()


def set_agent_id(agent_id: str) -> contextvars.Token:
    """Set the agent ID in context."""
    return _agent_id_var.set(agent_id)


@contextmanager
def correlation_context(
    correlation_id: Optional[str] = None,
    task_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Iterator[str]:
    """
    Context manager for setting correlation context.

    Usage:
        with correlation_context(task_id="task-001", agent_id="worker-1") as corr_id:
            logger.info("Processing task")  # Will include correlation_id, task_id, agent_id
    """
    # Generate correlation ID if not provided
    corr_id = correlation_id or get_correlation_id() or generate_correlation_id()

    tokens = []
    tokens.append(_correlation_id_var.set(corr_id))

    if task_id is not None:
        tokens.append(_task_id_var.set(task_id))
    if agent_id is not None:
        tokens.append(_agent_id_var.set(agent_id))

    try:
        yield corr_id
    finally:
        # Reset in reverse order
        for token in reversed(tokens):
            try:
                # Determine which var to reset based on the token
                if token == tokens[0]:
                    _correlation_id_var.reset(token)
                elif len(tokens) > 1 and task_id is not None and token == tokens[1]:
                    _task_id_var.reset(token)
                elif len(tokens) > 2 and agent_id is not None:
                    _agent_id_var.reset(token)
            except ValueError:
                pass


class CorrelationIDFilter(logging.Filter):
    """Logging filter that adds correlation context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation context fields to the log record."""
        # Add correlation ID
        record.correlation_id = get_correlation_id() or "-"

        # Add task ID if available
        task_id = get_task_id()
        if task_id:
            record.task_id = task_id

        # Add agent ID if available
        agent_id = get_agent_id()
        if agent_id:
            record.agent_id = agent_id

        # Try to get trace context from OpenTelemetry if available
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                record.trace_id = format(span.get_span_context().trace_id, "032x")
                record.span_id = format(span.get_span_context().span_id, "016x")
        except ImportError:
            pass

        return True


class JSONFormatter(logging.Formatter):
    """Format log records as JSON Lines for structured logging."""

    # Fields to extract from log records
    CONTEXT_FIELDS = [
        "correlation_id",
        "task_id",
        "agent_id",
        "event_type",
        "duration_ms",
        "status",
        "trace_id",
        "span_id",
        "priority",
        "error",
        "capabilities",
        "description",
    ]

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_location: bool = False,
        extra_fields: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_location = include_location
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {}

        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if self.include_level:
            log_data["level"] = record.levelname

        if self.include_logger:
            log_data["logger"] = record.name

        # Add source location for debugging
        if self.include_location:
            log_data["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        log_data["message"] = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add context fields from the record
        for key in self.CONTEXT_FIELDS:
            value = getattr(record, key, None)
            if value is not None and value != "-":
                log_data[key] = value

        # Add configured extra fields
        log_data.update(self.extra_fields)

        return json.dumps(log_data, default=str)


class CorrelatedFormatter(logging.Formatter):
    """Standard formatter with correlation ID support."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
    ):
        if fmt is None:
            fmt = "%(asctime)s [%(correlation_id)s] %(name)s %(levelname)s - %(message)s"
        super().__init__(fmt, datefmt)


class OrchestratorLogger:
    """Structured logger for orchestrator events with correlation ID support."""

    def __init__(
        self,
        name: str = "orchestrator",
        log_file: Optional[Path] = None,
        json_format: bool = True,
        level: int = logging.INFO,
        include_location: bool = False,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.json_format = json_format

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Add correlation ID filter
        correlation_filter = CorrelationIDFilter()
        self.logger.addFilter(correlation_filter)

        # Create formatter
        if json_format:
            formatter = JSONFormatter(include_location=include_location)
        else:
            formatter = CorrelatedFormatter()

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

    def with_correlation(
        self,
        correlation_id: Optional[str] = None,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> "CorrelationContextLogger":
        """
        Create a context-bound logger with correlation IDs.

        Usage:
            with logger.with_correlation(task_id="task-001") as log:
                log.info("Processing")
        """
        return CorrelationContextLogger(
            self,
            correlation_id=correlation_id,
            task_id=task_id,
            agent_id=agent_id,
        )

    # -------------------------------------------------------------------------
    # Task events
    # -------------------------------------------------------------------------

    def task_created(
        self,
        task_id: str,
        description: str,
        priority: int,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log task creation event."""
        with correlation_context(correlation_id=correlation_id, task_id=task_id):
            self.log_event(
                "task_created",
                f"Task created: {task_id}",
                task_id=task_id,
                description=description[:100],
                priority=priority,
            )

    def task_claimed(
        self,
        task_id: str,
        agent_id: str,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log task claim event."""
        with correlation_context(
            correlation_id=correlation_id,
            task_id=task_id,
            agent_id=agent_id,
        ):
            self.log_event(
                "task_claimed",
                f"Task {task_id} claimed by {agent_id}",
                task_id=task_id,
                agent_id=agent_id,
            )

    def task_started(
        self,
        task_id: str,
        agent_id: str,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log task start event."""
        with correlation_context(
            correlation_id=correlation_id,
            task_id=task_id,
            agent_id=agent_id,
        ):
            self.log_event(
                "task_started",
                f"Task {task_id} started by {agent_id}",
                task_id=task_id,
                agent_id=agent_id,
                status="in_progress",
            )

    def task_completed(
        self,
        task_id: str,
        agent_id: str,
        duration_ms: Optional[float] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log task completion event."""
        with correlation_context(
            correlation_id=correlation_id,
            task_id=task_id,
            agent_id=agent_id,
        ):
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
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log task failure event."""
        with correlation_context(
            correlation_id=correlation_id,
            task_id=task_id,
            agent_id=agent_id,
        ):
            self.log_event(
                "task_failed",
                f"Task {task_id} failed: {error}",
                level=logging.ERROR,
                task_id=task_id,
                agent_id=agent_id,
                error=error[:200],
                duration_ms=duration_ms,
                status="failed",
            )

    # -------------------------------------------------------------------------
    # Agent events
    # -------------------------------------------------------------------------

    def agent_registered(
        self,
        agent_id: str,
        capabilities: list[str],
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log agent registration event."""
        with correlation_context(correlation_id=correlation_id, agent_id=agent_id):
            self.log_event(
                "agent_registered",
                f"Agent registered: {agent_id}",
                agent_id=agent_id,
                capabilities=capabilities,
            )

    def agent_heartbeat(
        self,
        agent_id: str,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log agent heartbeat event."""
        with correlation_context(correlation_id=correlation_id, agent_id=agent_id):
            self.log_event(
                "agent_heartbeat",
                f"Agent heartbeat: {agent_id}",
                level=logging.DEBUG,
                agent_id=agent_id,
            )

    def agent_deregistered(
        self,
        agent_id: str,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log agent deregistration event."""
        with correlation_context(correlation_id=correlation_id, agent_id=agent_id):
            self.log_event(
                "agent_deregistered",
                f"Agent deregistered: {agent_id}",
                agent_id=agent_id,
            )

    # -------------------------------------------------------------------------
    # Orchestration events
    # -------------------------------------------------------------------------

    def orchestration_started(
        self,
        goal: str,
        max_workers: int,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Log orchestration start event. Returns the correlation ID."""
        corr_id = correlation_id or generate_correlation_id()
        with correlation_context(correlation_id=corr_id):
            self.log_event(
                "orchestration_started",
                f"Orchestration started: {goal[:100]}",
                goal=goal[:200],
                max_workers=max_workers,
            )
        return corr_id

    def orchestration_completed(
        self,
        tasks_completed: int,
        tasks_failed: int,
        duration_seconds: float,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log orchestration completion event."""
        with correlation_context(correlation_id=correlation_id):
            self.log_event(
                "orchestration_completed",
                f"Orchestration completed: {tasks_completed} done, {tasks_failed} failed",
                tasks_completed=tasks_completed,
                tasks_failed=tasks_failed,
                duration_seconds=duration_seconds,
            )

    def orchestration_error(
        self,
        error: str,
        context: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log orchestration error."""
        with correlation_context(correlation_id=correlation_id):
            self.log_event(
                "orchestration_error",
                f"Orchestration error: {error}",
                level=logging.ERROR,
                error=error,
                **(context or {}),
            )


class CorrelationContextLogger:
    """Context manager for logging with correlation IDs."""

    def __init__(
        self,
        logger: OrchestratorLogger,
        correlation_id: Optional[str] = None,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        self._logger = logger
        self._correlation_id = correlation_id or generate_correlation_id()
        self._task_id = task_id
        self._agent_id = agent_id
        self._tokens: list[contextvars.Token] = []

    def __enter__(self) -> "CorrelationContextLogger":
        self._tokens.append(set_correlation_id(self._correlation_id))
        if self._task_id:
            self._tokens.append(set_task_id(self._task_id))
        if self._agent_id:
            self._tokens.append(set_agent_id(self._agent_id))
        return self

    def __exit__(self, *args: Any) -> None:
        for token in reversed(self._tokens):
            try:
                _correlation_id_var.reset(token)
            except ValueError:
                try:
                    _task_id_var.reset(token)
                except ValueError:
                    try:
                        _agent_id_var.reset(token)
                    except ValueError:
                        pass

    @property
    def correlation_id(self) -> str:
        return self._correlation_id

    def info(self, message: str, **kwargs: Any) -> None:
        self._logger.log_event("info", message, level=logging.INFO, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._logger.log_event("debug", message, level=logging.DEBUG, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._logger.log_event("warning", message, level=logging.WARNING, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._logger.log_event("error", message, level=logging.ERROR, **kwargs)


def configure_structured_logging(
    log_dir: Optional[Path] = None,
    json_format: bool = True,
    level: int = logging.INFO,
    include_location: bool = False,
) -> OrchestratorLogger:
    """
    Configure structured logging for the orchestrator.

    Args:
        log_dir: Directory for log files. If None, logs only to stderr.
        json_format: If True, use JSON Lines format.
        level: Logging level.
        include_location: If True, include source file location in logs.

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
        include_location=include_location,
    )


# Global logger instance
_logger_instance: Optional[OrchestratorLogger] = None


def get_logger() -> OrchestratorLogger:
    """Get or create the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = OrchestratorLogger()
    return _logger_instance


def initialize_logging(
    log_dir: Optional[Path] = None,
    json_format: bool = True,
    level: int = logging.INFO,
) -> OrchestratorLogger:
    """Initialize the global logger instance."""
    global _logger_instance
    _logger_instance = configure_structured_logging(
        log_dir=log_dir,
        json_format=json_format,
        level=level,
    )
    return _logger_instance
