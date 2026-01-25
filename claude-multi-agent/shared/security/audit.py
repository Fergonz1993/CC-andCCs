"""
Audit Logging Module for Claude Multi-Agent Coordination System.

Provides comprehensive audit logging for all operations with:
- Tamper-evident checksum chains
- JSON Lines storage format
- Log rotation
- Export to JSON/CSV
- Context managers for audited operations

Compatible with Options A, B, and C of the multi-agent coordination system.
"""

import csv
import hashlib
import io
import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Union


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication events
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"
    AUTH_TOKEN_ISSUED = "auth.token_issued"
    AUTH_TOKEN_REVOKED = "auth.token_revoked"
    AUTH_KEY_CREATED = "auth.key_created"
    AUTH_KEY_REVOKED = "auth.key_revoked"

    # Task events
    TASK_CREATED = "task.created"
    TASK_CLAIMED = "task.claimed"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"
    TASK_DELETED = "task.deleted"
    TASK_UPDATED = "task.updated"

    # Agent events
    AGENT_REGISTERED = "agent.registered"
    AGENT_HEARTBEAT = "agent.heartbeat"
    AGENT_DEREGISTERED = "agent.deregistered"
    AGENT_ERROR = "agent.error"

    # Discovery events
    DISCOVERY_CREATED = "discovery.created"
    DISCOVERY_READ = "discovery.read"

    # Coordination events
    COORDINATION_INIT = "coordination.init"
    COORDINATION_START = "coordination.start"
    COORDINATION_STOP = "coordination.stop"

    # Security events
    SECURITY_ACCESS_DENIED = "security.access_denied"
    SECURITY_RATE_LIMITED = "security.rate_limited"
    SECURITY_CREDENTIAL_ACCESS = "security.credential_access"
    SECURITY_ENCRYPTION_ERROR = "security.encryption_error"
    SECURITY_KEY_ROTATED = "security.key_rotated"

    # Admin events
    ADMIN_CONFIG_CHANGED = "admin.config_changed"
    ADMIN_USER_MODIFIED = "admin.user_modified"

    # Data events
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"
    DATA_BACKUP = "data.backup"
    DATA_RESTORE = "data.restore"


class AuditLevel(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AuditEvent:
    """
    Represents an audit log event.

    Attributes:
        id: Unique identifier (UUID) for the event
        timestamp: ISO 8601 timestamp when the event occurred
        event_type: Type of audit event (from AuditEventType enum)
        level: Severity level (INFO, WARNING, ERROR, CRITICAL)
        agent_id: ID of the agent that performed the action
        action: Specific action performed
        resource_id: Optional ID of the affected resource
        resource_type: Optional type of the affected resource
        details: Additional event details as a dictionary
        checksum: SHA-256 checksum for tamper detection (includes previous checksum)
    """

    event_type: AuditEventType
    agent_id: Optional[str] = None
    action: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    level: AuditLevel = AuditLevel.INFO
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None

    # Additional metadata fields
    outcome: str = "success"  # success, failure, error
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    correlation_id: Optional[str] = None  # For tracing related events
    actor_role: Optional[str] = None  # Role of the agent (leader, worker, admin)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["level"] = self.level.value
        return d

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Create event from dictionary."""
        data = data.copy()
        if isinstance(data.get("event_type"), str):
            data["event_type"] = AuditEventType(data["event_type"])
        if isinstance(data.get("level"), str):
            data["level"] = AuditLevel(data["level"])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "AuditEvent":
        """Create event from JSON string."""
        return cls.from_dict(json.loads(json_str))


class AuditLogger:
    """
    Comprehensive audit logging system with tamper-evident checksum chains.

    Features:
    - JSON Lines format storage (one JSON object per line)
    - Checksum chain for tamper detection
    - Log rotation with configurable size limits
    - Export to JSON and CSV formats
    - Thread-safe operation
    - Multiple output destinations (file, callback, logger)
    - Event filtering and querying

    Usage:
        logger = AuditLogger(coordination_dir=".coordination")
        logger.log(AuditEventType.TASK_CREATED, agent_id="agent-1", action="create_task")

        # With context manager
        with logger.audited_operation(AuditEventType.TASK_CLAIMED, agent_id="worker-1"):
            # perform operation
            pass
    """

    DEFAULT_AUDIT_DIR = ".coordination/audit"
    DEFAULT_LOG_FILE = "audit.jsonl"

    def __init__(
        self,
        coordination_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        on_event: Optional[Callable[[AuditEvent], None]] = None,
        enable_console: bool = False,
        enable_tamper_detection: bool = True,
        max_file_size_mb: int = 10,
        max_backup_count: int = 5,
    ):
        """
        Initialize the audit logger.

        Args:
            coordination_dir: Base coordination directory (defaults to current dir)
            log_file: Full path to audit log file (overrides coordination_dir)
            logger: Python logger to use for audit events
            on_event: Callback function for each audit event
            enable_console: Whether to print events to console
            enable_tamper_detection: Whether to add checksum chain for integrity
            max_file_size_mb: Max log file size before rotation (in MB)
            max_backup_count: Number of backup files to keep during rotation
        """
        # Determine log file path
        if log_file:
            self.log_file = Path(log_file)
        elif coordination_dir:
            audit_dir = Path(coordination_dir) / "audit"
            self.log_file = audit_dir / self.DEFAULT_LOG_FILE
        else:
            self.log_file = Path(self.DEFAULT_AUDIT_DIR) / self.DEFAULT_LOG_FILE

        self.logger = logger
        self.on_event = on_event
        self.enable_console = enable_console
        self.enable_tamper_detection = enable_tamper_detection
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.max_backup_count = max_backup_count

        self._lock = threading.Lock()
        self._event_buffer: List[AuditEvent] = []
        self._buffer_size = 100
        self._last_checksum: Optional[str] = None

        # Initialize log directory and file
        self._initialize_log_file()

    def _initialize_log_file(self) -> None:
        """Initialize the log file and load the last checksum if present."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Load the last checksum from existing log for chain continuity
        if self.log_file.exists() and self.enable_tamper_detection:
            try:
                with open(self.log_file, "r") as f:
                    # Read last line to get the last checksum
                    last_line = None
                    for line in f:
                        if line.strip():
                            last_line = line
                    if last_line:
                        event_data = json.loads(last_line)
                        self._last_checksum = event_data.get("checksum")
            except (json.JSONDecodeError, IOError):
                pass

    def log(
        self,
        event_type: AuditEventType,
        agent_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        level: AuditLevel = AuditLevel.INFO,
        details: Optional[Dict[str, Any]] = None,
        outcome: str = "success",
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        actor_role: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of audit event
            agent_id: ID of the agent performing the action
            action: Specific action being performed
            resource_id: ID of the affected resource (optional)
            resource_type: Type of the affected resource (optional)
            level: Severity level (default INFO)
            details: Additional event details
            outcome: Result of the action (success, failure, error)
            ip_address: IP address of the actor
            correlation_id: ID for correlating related events
            actor_role: Role of the actor (leader, worker, admin)

        Returns:
            The created AuditEvent
        """
        event = AuditEvent(
            event_type=event_type,
            agent_id=agent_id,
            action=action,
            resource_id=resource_id,
            resource_type=resource_type,
            level=level,
            details=details or {},
            outcome=outcome,
            ip_address=ip_address,
            correlation_id=correlation_id,
            actor_role=actor_role,
        )

        self._process_event(event)
        return event

    def log_event(self, event: AuditEvent) -> None:
        """Log a pre-constructed audit event."""
        self._process_event(event)

    def _process_event(self, event: AuditEvent) -> None:
        """Process and store an audit event."""
        with self._lock:
            # Compute checksum chain if enabled
            if self.enable_tamper_detection:
                event.checksum = self._compute_checksum(event)
                self._last_checksum = event.checksum

            # Write to file
            self._write_to_file(event)

            # Send to Python logger
            if self.logger:
                self._log_to_logger(event)

            # Call callback
            if self.on_event:
                try:
                    self.on_event(event)
                except Exception:
                    pass  # Don't let callback errors affect logging

            # Console output
            if self.enable_console:
                self._log_to_console(event)

            # Buffer for query access
            self._event_buffer.append(event)
            if len(self._event_buffer) > self._buffer_size:
                self._event_buffer = self._event_buffer[-self._buffer_size :]

    def _compute_checksum(self, event: AuditEvent) -> str:
        """
        Compute checksum for tamper detection.

        The checksum includes the previous event's checksum to create a chain,
        making it impossible to modify past events without detection.
        """
        # Create a copy without the checksum field for hashing
        event_data = event.to_dict()
        event_data.pop("checksum", None)
        event_json = json.dumps(event_data, sort_keys=True, default=str)

        # Chain with previous checksum
        previous = self._last_checksum or "GENESIS"
        content = f"{previous}:{event_json}"

        return hashlib.sha256(content.encode()).hexdigest()

    def _write_to_file(self, event: AuditEvent) -> None:
        """Write event to log file in JSON Lines format."""
        # Check for rotation before writing
        if (
            self.log_file.exists()
            and self.log_file.stat().st_size > self.max_file_size
        ):
            self.rotate()

        # Write event as JSON line
        with open(self.log_file, "a") as f:
            f.write(event.to_json() + "\n")

    def _log_to_logger(self, event: AuditEvent) -> None:
        """Send event to Python logger."""
        if not self.logger:
            return

        # Map audit level to logging level
        level_map = {
            AuditLevel.DEBUG: logging.DEBUG,
            AuditLevel.INFO: logging.INFO,
            AuditLevel.WARNING: logging.WARNING,
            AuditLevel.ERROR: logging.ERROR,
            AuditLevel.CRITICAL: logging.CRITICAL,
        }
        level = level_map.get(event.level, logging.INFO)

        # Override based on outcome
        if event.outcome == "failure":
            level = max(level, logging.WARNING)
        elif event.outcome == "error":
            level = max(level, logging.ERROR)

        self.logger.log(
            level,
            f"[AUDIT] {event.event_type.value} | agent={event.agent_id} | "
            f"resource={event.resource_type}/{event.resource_id} | "
            f"outcome={event.outcome}",
            extra={"audit_event": event.to_dict()},
        )

    def _log_to_console(self, event: AuditEvent) -> None:
        """Print event to console."""
        outcome_symbol = {
            "success": "+",
            "failure": "-",
            "error": "!",
        }.get(event.outcome, "?")

        print(
            f"[{event.timestamp}] [{event.level.value}] [{outcome_symbol}] "
            f"{event.event_type.value} | {event.agent_id or 'system'} | "
            f"{event.resource_type or '-'}/{event.resource_id or '-'}"
        )

    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        agent_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        level: Optional[AuditLevel] = None,
        outcome: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        from_file: bool = False,
    ) -> List[AuditEvent]:
        """
        Query audit events with filtering.

        Args:
            event_type: Filter by event type
            agent_id: Filter by agent ID
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            level: Filter by minimum severity level
            outcome: Filter by outcome (success, failure, error)
            start_time: Filter events after this ISO timestamp
            end_time: Filter events before this ISO timestamp
            limit: Maximum number of events to return
            from_file: If True, read from log file instead of buffer

        Returns:
            List of matching AuditEvent objects
        """
        if from_file:
            events = list(self._read_events_from_file())
        else:
            events = self._event_buffer.copy()

        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        if resource_type:
            events = [e for e in events if e.resource_type == resource_type]
        if resource_id:
            events = [e for e in events if e.resource_id == resource_id]
        if level:
            level_order = list(AuditLevel)
            min_index = level_order.index(level)
            events = [e for e in events if level_order.index(e.level) >= min_index]
        if outcome:
            events = [e for e in events if e.outcome == outcome]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return events[-limit:]

    def _read_events_from_file(self) -> Iterator[AuditEvent]:
        """Read all events from the log file."""
        if not self.log_file.exists():
            return

        with open(self.log_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        yield AuditEvent.from_json(line)
                    except (json.JSONDecodeError, KeyError):
                        continue

    def verify_integrity(self) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Verify the integrity of the audit log by checking the checksum chain.

        Returns:
            Tuple of (is_valid, list_of_corruption_details)
            Each corruption detail contains: line_number, event_id, expected_checksum, actual_checksum
        """
        if not self.log_file.exists():
            return True, []

        corruptions: List[Dict[str, Any]] = []
        last_checksum = "GENESIS"

        with open(self.log_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    event_data = json.loads(line)
                    stored_checksum = event_data.get("checksum")

                    if stored_checksum:
                        # Compute expected checksum
                        event_data_copy = event_data.copy()
                        event_data_copy.pop("checksum", None)
                        event_json = json.dumps(
                            event_data_copy, sort_keys=True, default=str
                        )
                        content = f"{last_checksum}:{event_json}"
                        expected_checksum = hashlib.sha256(content.encode()).hexdigest()

                        if stored_checksum != expected_checksum:
                            corruptions.append(
                                {
                                    "line_number": line_num,
                                    "event_id": event_data.get("id"),
                                    "expected_checksum": expected_checksum,
                                    "actual_checksum": stored_checksum,
                                }
                            )

                        last_checksum = stored_checksum

                except json.JSONDecodeError:
                    corruptions.append(
                        {
                            "line_number": line_num,
                            "event_id": None,
                            "error": "Invalid JSON",
                        }
                    )

        return len(corruptions) == 0, corruptions

    def rotate(self) -> None:
        """
        Rotate log files.

        Moves current log to .1, shifts existing backups, and removes old backups
        beyond max_backup_count.
        """
        with self._lock:
            if not self.log_file.exists():
                return

            # Shift existing backups
            for i in range(self.max_backup_count - 1, 0, -1):
                old_path = Path(f"{self.log_file}.{i}")
                new_path = Path(f"{self.log_file}.{i + 1}")
                if old_path.exists():
                    if i + 1 > self.max_backup_count:
                        old_path.unlink()  # Remove files beyond max count
                    else:
                        old_path.rename(new_path)

            # Move current file to .1
            backup_path = Path(f"{self.log_file}.1")
            if self.log_file.exists():
                self.log_file.rename(backup_path)

            # Reset checksum chain for new file
            self._last_checksum = None

    def export(
        self,
        format: str = "json",
        output_path: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Union[str, int]:
        """
        Export audit events to JSON or CSV format.

        Args:
            format: Output format - "json" or "csv"
            output_path: Path to write output file. If None, returns string.
            start_time: Filter events after this ISO timestamp
            end_time: Filter events before this ISO timestamp

        Returns:
            If output_path is provided: number of events exported
            If output_path is None: exported data as string
        """
        events = self.get_events(
            start_time=start_time,
            end_time=end_time,
            limit=999999,  # Get all matching events
            from_file=True,
        )

        if format.lower() == "json":
            return self._export_json(events, output_path)
        elif format.lower() == "csv":
            return self._export_csv(events, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(
        self, events: List[AuditEvent], output_path: Optional[str]
    ) -> Union[str, int]:
        """Export events to JSON format."""
        data = {
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "event_count": len(events),
            "events": [e.to_dict() for e in events],
        }
        json_str = json.dumps(data, indent=2, default=str)

        if output_path:
            with open(output_path, "w") as f:
                f.write(json_str)
            return len(events)
        return json_str

    def _export_csv(
        self, events: List[AuditEvent], output_path: Optional[str]
    ) -> Union[str, int]:
        """Export events to CSV format."""
        if not events:
            return "" if output_path is None else 0

        # Define CSV columns
        fieldnames = [
            "id",
            "timestamp",
            "event_type",
            "level",
            "agent_id",
            "action",
            "resource_id",
            "resource_type",
            "outcome",
            "actor_role",
            "ip_address",
            "correlation_id",
            "details",
            "checksum",
        ]

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for event in events:
            row = event.to_dict()
            # Serialize details dict to JSON string for CSV
            row["details"] = json.dumps(row.get("details", {}))
            writer.writerow(row)

        csv_str = output.getvalue()

        if output_path:
            with open(output_path, "w", newline="") as f:
                f.write(csv_str)
            return len(events)
        return csv_str

    # =========================================================================
    # Convenience methods for common audit events
    # =========================================================================

    def log_auth_success(
        self,
        agent_id: str,
        actor_role: str,
        method: str = "api_key",
        ip_address: Optional[str] = None,
    ) -> AuditEvent:
        """Log successful authentication."""
        return self.log(
            event_type=AuditEventType.AUTH_LOGIN,
            agent_id=agent_id,
            actor_role=actor_role,
            action=f"authenticate:{method}",
            outcome="success",
            ip_address=ip_address,
        )

    def log_auth_failure(
        self,
        attempted_identity: Optional[str] = None,
        reason: str = "invalid_credentials",
        ip_address: Optional[str] = None,
    ) -> AuditEvent:
        """Log failed authentication."""
        return self.log(
            event_type=AuditEventType.AUTH_FAILED,
            agent_id=attempted_identity,
            action="authenticate",
            outcome="failure",
            level=AuditLevel.WARNING,
            details={"reason": reason},
            ip_address=ip_address,
        )

    def log_access_denied(
        self,
        agent_id: str,
        actor_role: str,
        permission: str,
        resource_type: str,
        resource_id: Optional[str] = None,
    ) -> AuditEvent:
        """Log access denied event."""
        return self.log(
            event_type=AuditEventType.SECURITY_ACCESS_DENIED,
            agent_id=agent_id,
            actor_role=actor_role,
            resource_type=resource_type,
            resource_id=resource_id,
            action=permission,
            outcome="failure",
            level=AuditLevel.WARNING,
            details={"required_permission": permission},
        )

    def log_task_event(
        self,
        event_type: AuditEventType,
        task_id: str,
        agent_id: str,
        actor_role: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log a task-related event."""
        return self.log(
            event_type=event_type,
            agent_id=agent_id,
            actor_role=actor_role,
            resource_type="task",
            resource_id=task_id,
            details=details or {},
        )

    def log_rate_limited(
        self,
        agent_id: str,
        limit_name: str,
        requests_made: int,
        limit: int,
        ip_address: Optional[str] = None,
    ) -> AuditEvent:
        """Log rate limit exceeded event."""
        return self.log(
            event_type=AuditEventType.SECURITY_RATE_LIMITED,
            agent_id=agent_id,
            action="rate_limit_exceeded",
            outcome="failure",
            level=AuditLevel.WARNING,
            details={
                "limit_name": limit_name,
                "requests_made": requests_made,
                "limit": limit,
            },
            ip_address=ip_address,
        )

    # =========================================================================
    # Context manager for audited operations
    # =========================================================================

    @contextmanager
    def audited_operation(
        self,
        event_type: AuditEventType,
        agent_id: str,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        actor_role: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager for auditing operations.

        Automatically logs the start and completion/failure of an operation.

        Usage:
            with audit_logger.audited_operation(
                AuditEventType.TASK_CLAIMED,
                agent_id="worker-1",
                resource_type="task",
                resource_id="task-123"
            ) as ctx:
                # perform operation
                ctx["details"]["files_modified"] = 5

        Args:
            event_type: Type of audit event
            agent_id: ID of the agent performing the action
            action: Specific action being performed
            resource_type: Type of resource being affected
            resource_id: ID of resource being affected
            actor_role: Role of the actor
            details: Initial details dict (will be extended)

        Yields:
            A context dict that can be modified to add details to the final event
        """
        context: Dict[str, Any] = {
            "details": details.copy() if details else {},
            "start_time": datetime.utcnow().isoformat() + "Z",
        }

        try:
            yield context
            # Success - log completion
            context["details"]["duration_ms"] = int(
                (
                    datetime.utcnow()
                    - datetime.fromisoformat(context["start_time"].rstrip("Z"))
                ).total_seconds()
                * 1000
            )
            self.log(
                event_type=event_type,
                agent_id=agent_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                actor_role=actor_role,
                details=context["details"],
                outcome="success",
            )
        except Exception as e:
            # Failure - log error
            context["details"]["duration_ms"] = int(
                (
                    datetime.utcnow()
                    - datetime.fromisoformat(context["start_time"].rstrip("Z"))
                ).total_seconds()
                * 1000
            )
            context["details"]["error"] = str(e)
            context["details"]["error_type"] = type(e).__name__
            self.log(
                event_type=event_type,
                agent_id=agent_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                actor_role=actor_role,
                details=context["details"],
                outcome="error",
                level=AuditLevel.ERROR,
            )
            raise


# =============================================================================
# Integration Helpers
# =============================================================================


def audit_task_operation(
    logger: AuditLogger,
    operation: str,
    task: Dict[str, Any],
    agent_id: str,
    actor_role: str = "worker",
    outcome: str = "success",
    extra_details: Optional[Dict[str, Any]] = None,
) -> AuditEvent:
    """
    Convenience function for auditing task operations.

    Maps common task operations to appropriate event types and logs them.

    Args:
        logger: AuditLogger instance
        operation: Operation name - "create", "claim", "start", "complete", "fail", "cancel"
        task: Task dictionary (must have 'id' field)
        agent_id: ID of the agent performing the operation
        actor_role: Role of the agent (default "worker")
        outcome: Result of operation (default "success")
        extra_details: Additional details to include

    Returns:
        The created AuditEvent
    """
    operation_map = {
        "create": AuditEventType.TASK_CREATED,
        "claim": AuditEventType.TASK_CLAIMED,
        "start": AuditEventType.TASK_STARTED,
        "complete": AuditEventType.TASK_COMPLETED,
        "fail": AuditEventType.TASK_FAILED,
        "cancel": AuditEventType.TASK_CANCELLED,
        "delete": AuditEventType.TASK_DELETED,
        "update": AuditEventType.TASK_UPDATED,
    }

    event_type = operation_map.get(operation.lower())
    if not event_type:
        raise ValueError(f"Unknown task operation: {operation}")

    # Determine level based on operation and outcome
    level = AuditLevel.INFO
    if operation.lower() == "fail" or outcome == "failure":
        level = AuditLevel.WARNING
    elif outcome == "error":
        level = AuditLevel.ERROR

    # Build details
    details = {
        "task_description": task.get("description", "")[:100],  # Truncate long descriptions
        "task_status": task.get("status"),
        "task_priority": task.get("priority"),
    }
    if extra_details:
        details.update(extra_details)

    return logger.log(
        event_type=event_type,
        agent_id=agent_id,
        actor_role=actor_role,
        action=operation,
        resource_type="task",
        resource_id=task.get("id"),
        level=level,
        outcome=outcome,
        details=details,
    )


# =============================================================================
# Module-level singleton for convenience
# =============================================================================

_default_logger: Optional[AuditLogger] = None


def get_audit_logger(
    coordination_dir: Optional[str] = None,
    **kwargs: Any,
) -> AuditLogger:
    """
    Get or create the default audit logger.

    Args:
        coordination_dir: Base coordination directory
        **kwargs: Additional arguments passed to AuditLogger constructor

    Returns:
        AuditLogger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger(coordination_dir=coordination_dir, **kwargs)
    return _default_logger


def reset_audit_logger() -> None:
    """Reset the default audit logger (useful for testing)."""
    global _default_logger
    _default_logger = None
