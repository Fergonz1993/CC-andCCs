"""
Audit Logging Module for Claude Multi-Agent Coordination System.

Provides comprehensive audit logging for all operations.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import hashlib
import threading


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


@dataclass
class AuditEvent:
    """Represents an audit log event."""
    event_type: AuditEventType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    actor: Optional[str] = None  # Who performed the action (agent_id, user_id)
    actor_role: Optional[str] = None
    resource_type: Optional[str] = None  # Type of resource affected
    resource_id: Optional[str] = None  # ID of the resource affected
    action: Optional[str] = None  # Specific action performed
    outcome: str = "success"  # success, failure, error
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    correlation_id: Optional[str] = None  # For tracing related events
    event_id: str = field(default_factory=lambda: hashlib.sha256(
        f"{datetime.now().isoformat()}{os.urandom(8).hex()}".encode()
    ).hexdigest()[:16])

    def to_dict(self) -> dict:
        d = asdict(self)
        d['event_type'] = self.event_type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "AuditEvent":
        data['event_type'] = AuditEventType(data['event_type'])
        return cls(**data)


class AuditLogger:
    """
    Comprehensive audit logging system.

    Features:
    - Structured audit logging
    - Multiple output destinations (file, callback, logger)
    - Event filtering
    - Tamper-evident logging (optional)
    - Log rotation support
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        on_event: Optional[Callable[[AuditEvent], None]] = None,
        enable_console: bool = False,
        enable_tamper_detection: bool = False,
        max_file_size_mb: int = 10,
        max_backup_count: int = 5,
    ):
        """
        Initialize the audit logger.

        Args:
            log_file: Path to audit log file
            logger: Python logger to use for audit events
            on_event: Callback function for each audit event
            enable_console: Whether to print events to console
            enable_tamper_detection: Whether to add integrity hashes
            max_file_size_mb: Max log file size before rotation
            max_backup_count: Number of backup files to keep
        """
        self.log_file = Path(log_file) if log_file else None
        self.logger = logger
        self.on_event = on_event
        self.enable_console = enable_console
        self.enable_tamper_detection = enable_tamper_detection
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.max_backup_count = max_backup_count

        self._lock = threading.Lock()
        self._event_buffer: List[AuditEvent] = []
        self._buffer_size = 100
        self._last_hash: Optional[str] = None

        # Initialize log file
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event_type: AuditEventType,
        actor: Optional[str] = None,
        actor_role: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Returns the created AuditEvent.
        """
        event = AuditEvent(
            event_type=event_type,
            actor=actor,
            actor_role=actor_role,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            details=details or {},
            ip_address=ip_address,
            correlation_id=correlation_id,
        )

        self._process_event(event)
        return event

    def log_event(self, event: AuditEvent) -> None:
        """Log a pre-constructed audit event."""
        self._process_event(event)

    def _process_event(self, event: AuditEvent) -> None:
        """Process and store an audit event."""
        with self._lock:
            # Add tamper detection if enabled
            if self.enable_tamper_detection:
                event.details['_integrity_hash'] = self._compute_integrity_hash(event)

            # Write to file
            if self.log_file:
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
                self._event_buffer = self._event_buffer[-self._buffer_size:]

    def _write_to_file(self, event: AuditEvent) -> None:
        """Write event to log file."""
        if not self.log_file:
            return

        # Check for rotation
        if self.log_file.exists() and self.log_file.stat().st_size > self.max_file_size:
            self._rotate_log_file()

        # Write event
        with open(self.log_file, 'a') as f:
            f.write(event.to_json() + '\n')

    def _rotate_log_file(self) -> None:
        """Rotate log files."""
        if not self.log_file:
            return

        # Shift existing backups
        for i in range(self.max_backup_count - 1, 0, -1):
            old_path = Path(f"{self.log_file}.{i}")
            new_path = Path(f"{self.log_file}.{i + 1}")
            if old_path.exists():
                old_path.rename(new_path)

        # Move current file to .1
        backup_path = Path(f"{self.log_file}.1")
        if self.log_file.exists():
            self.log_file.rename(backup_path)

    def _log_to_logger(self, event: AuditEvent) -> None:
        """Send event to Python logger."""
        if not self.logger:
            return

        level = logging.INFO
        if event.outcome == "failure":
            level = logging.WARNING
        elif event.outcome == "error":
            level = logging.ERROR
        elif event.event_type.value.startswith("security."):
            level = logging.WARNING

        self.logger.log(
            level,
            f"[AUDIT] {event.event_type.value} | actor={event.actor} | "
            f"resource={event.resource_type}/{event.resource_id} | "
            f"outcome={event.outcome}",
            extra={'audit_event': event.to_dict()},
        )

    def _log_to_console(self, event: AuditEvent) -> None:
        """Print event to console."""
        outcome_symbol = {
            'success': '+',
            'failure': '-',
            'error': '!',
        }.get(event.outcome, '?')

        print(
            f"[{event.timestamp}] [{outcome_symbol}] {event.event_type.value} "
            f"| {event.actor or 'system'} | {event.resource_type or '-'}/{event.resource_id or '-'}"
        )

    def _compute_integrity_hash(self, event: AuditEvent) -> str:
        """Compute integrity hash for tamper detection."""
        # Hash includes previous hash for chain integrity
        event_data = event.to_json()
        content = f"{self._last_hash or 'GENESIS'}:{event_data}"
        new_hash = hashlib.sha256(content.encode()).hexdigest()
        self._last_hash = new_hash
        return new_hash

    def query_events(
        self,
        event_type: Optional[AuditEventType] = None,
        actor: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        outcome: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """
        Query buffered audit events.

        For more comprehensive queries, use the log file directly.
        """
        events = self._event_buffer.copy()

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if actor:
            events = [e for e in events if e.actor == actor]
        if resource_type:
            events = [e for e in events if e.resource_type == resource_type]
        if resource_id:
            events = [e for e in events if e.resource_id == resource_id]
        if outcome:
            events = [e for e in events if e.outcome == outcome]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return events[-limit:]

    def export_events(
        self,
        output_path: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> int:
        """
        Export audit events to a file.

        Returns the number of events exported.
        """
        if not self.log_file or not self.log_file.exists():
            return 0

        count = 0
        with open(self.log_file, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                try:
                    event_data = json.loads(line)
                    timestamp = event_data.get('timestamp', '')

                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue

                    outfile.write(line)
                    count += 1
                except json.JSONDecodeError:
                    continue

        return count

    def verify_integrity(self) -> tuple[bool, List[int]]:
        """
        Verify the integrity of the audit log.

        Returns: (is_valid, list_of_corrupted_line_numbers)
        """
        if not self.log_file or not self.log_file.exists():
            return True, []

        corrupted_lines = []
        last_hash = 'GENESIS'

        with open(self.log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    event_data = json.loads(line)
                    stored_hash = event_data.get('details', {}).get('_integrity_hash')

                    if stored_hash and self.enable_tamper_detection:
                        # Remove hash to compute expected value
                        details = event_data.get('details', {}).copy()
                        del details['_integrity_hash']
                        event_data['details'] = details

                        event_json = json.dumps(event_data)
                        content = f"{last_hash}:{event_json}"
                        expected_hash = hashlib.sha256(content.encode()).hexdigest()

                        if stored_hash != expected_hash:
                            corrupted_lines.append(line_num)

                        last_hash = stored_hash

                except json.JSONDecodeError:
                    corrupted_lines.append(line_num)

        return len(corrupted_lines) == 0, corrupted_lines

    # Convenience methods for common audit events

    def log_auth_success(
        self,
        actor: str,
        actor_role: str,
        method: str = "api_key",
        ip_address: Optional[str] = None,
    ) -> AuditEvent:
        """Log successful authentication."""
        return self.log(
            event_type=AuditEventType.AUTH_LOGIN,
            actor=actor,
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
            actor=attempted_identity,
            action="authenticate",
            outcome="failure",
            details={'reason': reason},
            ip_address=ip_address,
        )

    def log_access_denied(
        self,
        actor: str,
        actor_role: str,
        permission: str,
        resource_type: str,
        resource_id: Optional[str] = None,
    ) -> AuditEvent:
        """Log access denied event."""
        return self.log(
            event_type=AuditEventType.SECURITY_ACCESS_DENIED,
            actor=actor,
            actor_role=actor_role,
            resource_type=resource_type,
            resource_id=resource_id,
            action=permission,
            outcome="failure",
            details={'required_permission': permission},
        )

    def log_task_event(
        self,
        event_type: AuditEventType,
        task_id: str,
        actor: str,
        actor_role: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log a task-related event."""
        return self.log(
            event_type=event_type,
            actor=actor,
            actor_role=actor_role,
            resource_type="task",
            resource_id=task_id,
            details=details or {},
        )

    def log_rate_limited(
        self,
        actor: str,
        limit_name: str,
        requests_made: int,
        limit: int,
        ip_address: Optional[str] = None,
    ) -> AuditEvent:
        """Log rate limit exceeded event."""
        return self.log(
            event_type=AuditEventType.SECURITY_RATE_LIMITED,
            actor=actor,
            action="rate_limit_exceeded",
            outcome="failure",
            details={
                'limit_name': limit_name,
                'requests_made': requests_made,
                'limit': limit,
            },
            ip_address=ip_address,
        )
