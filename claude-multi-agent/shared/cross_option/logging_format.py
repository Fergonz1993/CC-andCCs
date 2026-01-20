"""
Common Logging Format (adv-cross-005)

Provides a standardized logging format that works across all coordination
options. Supports JSON structured logging, correlation IDs, and multiple
output targets.
"""

import json
import logging
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TextIO


class LogLevel(Enum):
    """Log levels for the common logging format."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def numeric(self) -> int:
        """Get numeric level for comparison."""
        return {
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }[self.value]

    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        """Create from string."""
        return cls(level.lower())


@dataclass
class LogEntry:
    """A single log entry with all metadata."""
    timestamp: datetime
    level: LogLevel
    message: str
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    correlation_id: Optional[str] = None
    option: Optional[str] = None  # 'A', 'B', or 'C'
    action: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    exception: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
        }

        if self.agent_id:
            d["agent_id"] = self.agent_id
        if self.task_id:
            d["task_id"] = self.task_id
        if self.correlation_id:
            d["correlation_id"] = self.correlation_id
        if self.option:
            d["option"] = self.option
        if self.action:
            d["action"] = self.action
        if self.details:
            d["details"] = self.details
        if self.source_file:
            d["source"] = {
                "file": self.source_file,
                "line": self.source_line,
            }
        if self.exception:
            d["exception"] = self.exception
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms

        return d

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def to_text(self, include_details: bool = True) -> str:
        """Convert to human-readable text format."""
        parts = [
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}]",
            f"[{self.level.value.upper()}]",
        ]

        if self.agent_id:
            parts.append(f"[{self.agent_id}]")
        if self.task_id:
            parts.append(f"[{self.task_id}]")
        if self.action:
            parts.append(f"{self.action}:")

        parts.append(self.message)

        if include_details and self.details:
            parts.append(f"| {json.dumps(self.details)}")

        if self.exception:
            parts.append(f"\n  Exception: {self.exception}")

        if self.duration_ms is not None:
            parts.append(f"({self.duration_ms:.2f}ms)")

        return " ".join(parts)


class JSONLogFormatter(logging.Formatter):
    """Formatter that outputs JSON log entries."""

    def __init__(
        self,
        option: Optional[str] = None,
        include_source: bool = False,
    ):
        super().__init__()
        self.option = option
        self.include_source = include_source

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created),
            level=LogLevel.from_string(record.levelname.lower()),
            message=record.getMessage(),
            option=self.option,
        )

        # Add extra fields if present
        if hasattr(record, "agent_id"):
            entry.agent_id = record.agent_id
        if hasattr(record, "task_id"):
            entry.task_id = record.task_id
        if hasattr(record, "correlation_id"):
            entry.correlation_id = record.correlation_id
        if hasattr(record, "action"):
            entry.action = record.action
        if hasattr(record, "details"):
            entry.details = record.details
        if hasattr(record, "duration_ms"):
            entry.duration_ms = record.duration_ms

        if self.include_source:
            entry.source_file = record.pathname
            entry.source_line = record.lineno

        if record.exc_info:
            entry.exception = self.formatException(record.exc_info)

        return entry.to_json()


class TextLogFormatter(logging.Formatter):
    """Formatter that outputs human-readable text."""

    def __init__(
        self,
        option: Optional[str] = None,
        include_details: bool = True,
        use_colors: bool = True,
    ):
        super().__init__()
        self.option = option
        self.include_details = include_details
        self.use_colors = use_colors

        self.colors = {
            LogLevel.DEBUG: "\033[90m",     # Gray
            LogLevel.INFO: "\033[0m",       # Default
            LogLevel.WARNING: "\033[33m",   # Yellow
            LogLevel.ERROR: "\033[31m",     # Red
            LogLevel.CRITICAL: "\033[91m",  # Bright Red
        }
        self.reset = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text."""
        level = LogLevel.from_string(record.levelname.lower())

        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created),
            level=level,
            message=record.getMessage(),
            option=self.option,
        )

        # Add extra fields if present
        if hasattr(record, "agent_id"):
            entry.agent_id = record.agent_id
        if hasattr(record, "task_id"):
            entry.task_id = record.task_id
        if hasattr(record, "action"):
            entry.action = record.action
        if hasattr(record, "details"):
            entry.details = record.details
        if hasattr(record, "duration_ms"):
            entry.duration_ms = record.duration_ms

        if record.exc_info:
            entry.exception = self.formatException(record.exc_info)

        text = entry.to_text(include_details=self.include_details)

        if self.use_colors:
            color = self.colors.get(level, "")
            return f"{color}{text}{self.reset}"

        return text


class CommonLogger:
    """
    Common logger that provides consistent logging across all options.

    Features:
    - Structured logging with JSON output
    - Correlation ID tracking
    - Multiple output targets (file, console, both)
    - Context-aware logging (agent, task, option)
    """

    _instance: Optional["CommonLogger"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        name: str = "coordination",
        level: LogLevel = LogLevel.INFO,
        option: Optional[str] = None,
        log_file: Optional[str] = None,
        json_output: bool = False,
        console_output: bool = True,
        include_source: bool = False,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            level: Minimum log level
            option: Coordination option ('A', 'B', or 'C')
            log_file: Path to log file (optional)
            json_output: Whether to output JSON format
            console_output: Whether to output to console
            include_source: Whether to include source file/line in logs
        """
        self.name = name
        self.level = level
        self.option = option
        self.json_output = json_output
        self.include_source = include_source

        # Thread-local storage for correlation ID and context
        self._local = threading.local()

        # Create underlying Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.numeric)
        self._logger.handlers = []  # Clear existing handlers

        # Add console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stderr)
            if json_output:
                console_handler.setFormatter(JSONLogFormatter(option, include_source))
            else:
                console_handler.setFormatter(TextLogFormatter(option, use_colors=True))
            self._logger.addHandler(console_handler)

        # Add file handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            # Always use JSON for file output
            file_handler.setFormatter(JSONLogFormatter(option, include_source))
            self._logger.addHandler(file_handler)

    @classmethod
    def get_instance(cls, **kwargs) -> "CommonLogger":
        """Get or create the singleton logger instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance."""
        with cls._lock:
            cls._instance = None

    def set_correlation_id(self, correlation_id: Optional[str] = None) -> str:
        """Set a correlation ID for the current thread."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]
        self._local.correlation_id = correlation_id
        return correlation_id

    def get_correlation_id(self) -> Optional[str]:
        """Get the current correlation ID."""
        return getattr(self._local, "correlation_id", None)

    def set_context(
        self,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """Set context for the current thread."""
        if agent_id is not None:
            self._local.agent_id = agent_id
        if task_id is not None:
            self._local.task_id = task_id

    def clear_context(self) -> None:
        """Clear context for the current thread."""
        self._local.agent_id = None
        self._local.task_id = None
        self._local.correlation_id = None

    def _log(
        self,
        level: LogLevel,
        message: str,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        exc_info: bool = False,
    ) -> None:
        """Internal logging method."""
        if level.numeric < self.level.numeric:
            return

        # Get context from thread-local storage
        agent_id = agent_id or getattr(self._local, "agent_id", None)
        task_id = task_id or getattr(self._local, "task_id", None)
        correlation_id = getattr(self._local, "correlation_id", None)

        # Create extra dict for the log record
        extra = {
            "agent_id": agent_id,
            "task_id": task_id,
            "correlation_id": correlation_id,
            "action": action,
            "details": details or {},
            "duration_ms": duration_ms,
        }

        # Log at the appropriate level
        log_method = getattr(self._logger, level.value)
        log_method(message, extra=extra, exc_info=exc_info)

    def debug(
        self,
        message: str,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Log a debug message."""
        self._log(LogLevel.DEBUG, message, action, details, **kwargs)

    def info(
        self,
        message: str,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Log an info message."""
        self._log(LogLevel.INFO, message, action, details, **kwargs)

    def warning(
        self,
        message: str,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Log a warning message."""
        self._log(LogLevel.WARNING, message, action, details, **kwargs)

    def error(
        self,
        message: str,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
        **kwargs,
    ) -> None:
        """Log an error message."""
        self._log(LogLevel.ERROR, message, action, details, exc_info=exc_info, **kwargs)

    def critical(
        self,
        message: str,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
        **kwargs,
    ) -> None:
        """Log a critical message."""
        self._log(LogLevel.CRITICAL, message, action, details, exc_info=exc_info, **kwargs)

    def log_task_event(
        self,
        task_id: str,
        event: str,
        details: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """Log a task-related event."""
        self.info(
            f"Task {event}",
            action=f"TASK_{event.upper()}",
            details=details,
            task_id=task_id,
            agent_id=agent_id,
        )

    def log_agent_event(
        self,
        agent_id: str,
        event: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an agent-related event."""
        self.info(
            f"Agent {event}",
            action=f"AGENT_{event.upper()}",
            details=details,
            agent_id=agent_id,
        )


def create_logger(
    name: str = "coordination",
    option: Optional[str] = None,
    log_dir: Optional[str] = None,
    level: Union[str, LogLevel] = LogLevel.INFO,
    json_output: bool = False,
) -> CommonLogger:
    """
    Create a configured logger instance.

    Args:
        name: Logger name
        option: Coordination option ('A', 'B', or 'C')
        log_dir: Directory for log files
        level: Minimum log level
        json_output: Whether to output JSON format

    Returns:
        Configured CommonLogger instance
    """
    if isinstance(level, str):
        level = LogLevel.from_string(level)

    log_file = None
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = str(log_path / f"{name}.log")

    return CommonLogger(
        name=name,
        level=level,
        option=option,
        log_file=log_file,
        json_output=json_output,
    )


# Convenience functions for quick logging

def get_logger(name: str = "coordination") -> CommonLogger:
    """Get or create the default logger."""
    return CommonLogger.get_instance(name=name)


def log_info(message: str, **kwargs) -> None:
    """Quick info logging."""
    get_logger().info(message, **kwargs)


def log_error(message: str, **kwargs) -> None:
    """Quick error logging."""
    get_logger().error(message, **kwargs)


def log_warning(message: str, **kwargs) -> None:
    """Quick warning logging."""
    get_logger().warning(message, **kwargs)


def log_debug(message: str, **kwargs) -> None:
    """Quick debug logging."""
    get_logger().debug(message, **kwargs)
