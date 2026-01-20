"""
Self-Healing Mechanisms (adv-rel-010)

Implements self-healing patterns that automatically detect and recover
from various failure scenarios without human intervention.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable, Type
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    """Types of recovery actions."""
    RESTART = "restart"           # Restart a component
    RESET = "reset"               # Reset state
    RECONNECT = "reconnect"       # Reconnect to external service
    FAILOVER = "failover"         # Switch to backup
    ROLLBACK = "rollback"         # Rollback to previous state
    SCALE_DOWN = "scale_down"     # Reduce load
    PURGE = "purge"               # Clear queue/cache
    CUSTOM = "custom"             # Custom recovery action


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0


@dataclass
class RecoveryEvent:
    """Record of a recovery action."""
    check_name: str
    action: RecoveryAction
    success: bool
    message: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class HealthCheck(ABC):
    """
    Abstract base class for health checks.

    Implement this to create custom health checks for different components.
    """

    def __init__(
        self,
        name: str,
        interval: float = 30.0,
        timeout: float = 10.0,
        failure_threshold: int = 3,
        success_threshold: int = 1,
    ):
        """
        Initialize health check.

        Args:
            name: Name of this health check
            interval: How often to run the check (seconds)
            timeout: Timeout for the check (seconds)
            failure_threshold: Consecutive failures before marking unhealthy
            success_threshold: Consecutive successes before marking healthy
        """
        self.name = name
        self.interval = interval
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold

        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._current_status = HealthStatus.UNKNOWN
        self._last_result: Optional[HealthCheckResult] = None

    @property
    def status(self) -> HealthStatus:
        return self._current_status

    @property
    def last_result(self) -> Optional[HealthCheckResult]:
        return self._last_result

    @abstractmethod
    def check(self) -> HealthCheckResult:
        """
        Perform the health check.

        Returns HealthCheckResult with status and details.
        """
        pass

    @abstractmethod
    def recover(self) -> RecoveryEvent:
        """
        Attempt to recover from an unhealthy state.

        Returns RecoveryEvent with result of the recovery attempt.
        """
        pass

    def run_check(self) -> HealthCheckResult:
        """Run the health check and update internal state."""
        start_time = time.time()

        try:
            result = self.check()
            result.duration_ms = (time.time() - start_time) * 1000
            self._last_result = result

            if result.status == HealthStatus.HEALTHY:
                self._consecutive_failures = 0
                self._consecutive_successes += 1

                if self._consecutive_successes >= self.success_threshold:
                    self._current_status = HealthStatus.HEALTHY

            elif result.status == HealthStatus.UNHEALTHY:
                self._consecutive_successes = 0
                self._consecutive_failures += 1

                if self._consecutive_failures >= self.failure_threshold:
                    self._current_status = HealthStatus.UNHEALTHY

            elif result.status == HealthStatus.DEGRADED:
                self._current_status = HealthStatus.DEGRADED

            return result

        except Exception as e:
            self._consecutive_failures += 1
            self._consecutive_successes = 0

            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check error: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._last_result = result

            if self._consecutive_failures >= self.failure_threshold:
                self._current_status = HealthStatus.UNHEALTHY

            return result


class WorkerHealthCheck(HealthCheck):
    """Health check for worker agents."""

    def __init__(
        self,
        worker_id: str,
        get_worker_status: Callable[[str], Dict[str, Any]],
        restart_worker: Callable[[str], bool],
        **kwargs,
    ):
        super().__init__(name=f"worker-{worker_id}", **kwargs)
        self.worker_id = worker_id
        self._get_worker_status = get_worker_status
        self._restart_worker = restart_worker

    def check(self) -> HealthCheckResult:
        try:
            status = self._get_worker_status(self.worker_id)

            is_running = status.get("is_running", False)
            current_task = status.get("current_task")
            last_activity = status.get("last_activity")

            if not is_running:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Worker is not running",
                    details=status,
                )

            # Check for stuck worker
            if last_activity:
                if isinstance(last_activity, str):
                    last_activity = datetime.fromisoformat(last_activity)
                age = datetime.now() - last_activity
                if age > timedelta(minutes=10) and current_task:
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.DEGRADED,
                        message=f"Worker may be stuck on task {current_task}",
                        details=status,
                    )

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Worker is healthy",
                details=status,
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check worker: {e}",
            )

    def recover(self) -> RecoveryEvent:
        event = RecoveryEvent(
            check_name=self.name,
            action=RecoveryAction.RESTART,
        )

        try:
            success = self._restart_worker(self.worker_id)
            event.success = success
            event.completed_at = datetime.now()
            event.message = "Worker restarted" if success else "Failed to restart worker"

            if success:
                self._consecutive_failures = 0
                self._current_status = HealthStatus.UNKNOWN

        except Exception as e:
            event.success = False
            event.error = str(e)
            event.completed_at = datetime.now()
            event.message = f"Recovery failed: {e}"

        return event


class TaskQueueHealthCheck(HealthCheck):
    """Health check for the task queue."""

    def __init__(
        self,
        get_queue_status: Callable[[], Dict[str, Any]],
        reset_stale_tasks: Callable[[], int],
        max_queue_depth: int = 1000,
        stale_threshold_minutes: int = 30,
        **kwargs,
    ):
        super().__init__(name="task-queue", **kwargs)
        self._get_queue_status = get_queue_status
        self._reset_stale_tasks = reset_stale_tasks
        self.max_queue_depth = max_queue_depth
        self.stale_threshold_minutes = stale_threshold_minutes

    def check(self) -> HealthCheckResult:
        try:
            status = self._get_queue_status()

            queue_depth = status.get("pending", 0) + status.get("available", 0)
            stale_count = status.get("stale_tasks", 0)
            failed_count = status.get("failed", 0)

            if queue_depth > self.max_queue_depth:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Queue depth {queue_depth} exceeds limit {self.max_queue_depth}",
                    details=status,
                )

            if stale_count > 0:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Found {stale_count} stale tasks",
                    details=status,
                )

            if failed_count > 10:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"High failure count: {failed_count}",
                    details=status,
                )

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Task queue is healthy",
                details=status,
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check queue: {e}",
            )

    def recover(self) -> RecoveryEvent:
        event = RecoveryEvent(
            check_name=self.name,
            action=RecoveryAction.RESET,
        )

        try:
            reset_count = self._reset_stale_tasks()
            event.success = True
            event.completed_at = datetime.now()
            event.message = f"Reset {reset_count} stale tasks"

            self._consecutive_failures = 0

        except Exception as e:
            event.success = False
            event.error = str(e)
            event.completed_at = datetime.now()
            event.message = f"Recovery failed: {e}"

        return event


class ConnectionHealthCheck(HealthCheck):
    """Health check for external connections."""

    def __init__(
        self,
        connection_name: str,
        test_connection: Callable[[], bool],
        reconnect: Callable[[], bool],
        **kwargs,
    ):
        super().__init__(name=f"connection-{connection_name}", **kwargs)
        self.connection_name = connection_name
        self._test_connection = test_connection
        self._reconnect = reconnect

    def check(self) -> HealthCheckResult:
        try:
            is_connected = self._test_connection()

            if is_connected:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Connected to {self.connection_name}",
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Not connected to {self.connection_name}",
                )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Connection check failed: {e}",
            )

    def recover(self) -> RecoveryEvent:
        event = RecoveryEvent(
            check_name=self.name,
            action=RecoveryAction.RECONNECT,
        )

        try:
            success = self._reconnect()
            event.success = success
            event.completed_at = datetime.now()
            event.message = "Reconnected" if success else "Reconnection failed"

            if success:
                self._consecutive_failures = 0
                self._current_status = HealthStatus.UNKNOWN

        except Exception as e:
            event.success = False
            event.error = str(e)
            event.completed_at = datetime.now()
            event.message = f"Recovery failed: {e}"

        return event


class SelfHealingManager:
    """
    Manages self-healing for the coordination system.

    Features:
    - Register multiple health checks
    - Automatic health monitoring
    - Automatic recovery attempts
    - Recovery cooldown to prevent thrashing
    - Event logging and history
    """

    def __init__(
        self,
        recovery_cooldown: float = 60.0,  # Seconds between recovery attempts
        max_recovery_attempts: int = 3,   # Max attempts before giving up
        on_unhealthy: Optional[Callable[[HealthCheckResult], None]] = None,
        on_recovery: Optional[Callable[[RecoveryEvent], None]] = None,
    ):
        """
        Initialize self-healing manager.

        Args:
            recovery_cooldown: Minimum time between recovery attempts
            max_recovery_attempts: Maximum recovery attempts per check
            on_unhealthy: Callback when a check becomes unhealthy
            on_recovery: Callback after a recovery attempt
        """
        self.recovery_cooldown = recovery_cooldown
        self.max_recovery_attempts = max_recovery_attempts
        self.on_unhealthy = on_unhealthy
        self.on_recovery = on_recovery

        self._checks: Dict[str, HealthCheck] = {}
        self._recovery_attempts: Dict[str, int] = {}
        self._last_recovery: Dict[str, datetime] = {}
        self._events: List[RecoveryEvent] = []
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

    def register_check(self, check: HealthCheck) -> None:
        """Register a health check."""
        with self._lock:
            self._checks[check.name] = check
            self._recovery_attempts[check.name] = 0
            logger.info(f"Registered health check: {check.name}")

    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)
            self._recovery_attempts.pop(name, None)
            self._last_recovery.pop(name, None)

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks and return results."""
        results = {}

        with self._lock:
            for name, check in self._checks.items():
                result = check.run_check()
                results[name] = result

                # Notify on unhealthy
                if result.status == HealthStatus.UNHEALTHY:
                    logger.warning(f"Health check failed: {name} - {result.message}")

                    if self.on_unhealthy:
                        try:
                            self.on_unhealthy(result)
                        except Exception as e:
                            logger.error(f"on_unhealthy callback error: {e}")

        return results

    def attempt_recovery(self, check_name: str) -> Optional[RecoveryEvent]:
        """
        Attempt recovery for a specific check.

        Returns the RecoveryEvent if recovery was attempted, None otherwise.
        """
        with self._lock:
            check = self._checks.get(check_name)
            if not check:
                return None

            # Check cooldown
            last = self._last_recovery.get(check_name)
            if last:
                elapsed = (datetime.now() - last).total_seconds()
                if elapsed < self.recovery_cooldown:
                    logger.debug(
                        f"Recovery for {check_name} in cooldown "
                        f"({self.recovery_cooldown - elapsed:.0f}s remaining)"
                    )
                    return None

            # Check max attempts
            attempts = self._recovery_attempts.get(check_name, 0)
            if attempts >= self.max_recovery_attempts:
                logger.warning(
                    f"Max recovery attempts ({self.max_recovery_attempts}) "
                    f"reached for {check_name}"
                )
                return None

            # Attempt recovery
            logger.info(f"Attempting recovery for {check_name} (attempt {attempts + 1})")

            event = check.recover()
            self._events.append(event)
            self._last_recovery[check_name] = datetime.now()

            if event.success:
                self._recovery_attempts[check_name] = 0
                logger.info(f"Recovery successful for {check_name}")
            else:
                self._recovery_attempts[check_name] = attempts + 1
                logger.warning(f"Recovery failed for {check_name}: {event.error}")

            if self.on_recovery:
                try:
                    self.on_recovery(event)
                except Exception as e:
                    logger.error(f"on_recovery callback error: {e}")

            return event

    def heal_all(self) -> List[RecoveryEvent]:
        """
        Check all and attempt recovery for unhealthy checks.

        Returns list of recovery events.
        """
        events = []

        # Run checks first
        results = self.run_all_checks()

        # Attempt recovery for unhealthy checks
        for name, result in results.items():
            if result.status == HealthStatus.UNHEALTHY:
                event = self.attempt_recovery(name)
                if event:
                    events.append(event)

        return events

    def start_monitoring(self, min_interval: float = 5.0) -> None:
        """
        Start automatic health monitoring and healing.

        Args:
            min_interval: Minimum interval between check cycles
        """
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(min_interval,),
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Started self-healing monitoring")

    def stop_monitoring(self) -> None:
        """Stop automatic monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10.0)
            self._monitor_thread = None
        logger.info("Stopped self-healing monitoring")

    def _monitor_loop(self, min_interval: float) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Run all checks at their intervals
                now = datetime.now()
                next_check_at: Optional[datetime] = None

                with self._lock:
                    for check in self._checks.values():
                        last_result = check.last_result
                        if last_result:
                            next_run = last_result.checked_at + timedelta(seconds=check.interval)
                            if next_run <= now:
                                result = check.run_check()

                                if result.status == HealthStatus.UNHEALTHY:
                                    self.attempt_recovery(check.name)

                                next_run = result.checked_at + timedelta(seconds=check.interval)

                            if next_check_at is None or next_run < next_check_at:
                                next_check_at = next_run
                        else:
                            # First run
                            check.run_check()

            except Exception as e:
                logger.error(f"Self-healing monitor error: {e}")

            # Sleep until next check or min_interval
            if next_check_at:
                sleep_time = max(min_interval, (next_check_at - datetime.now()).total_seconds())
            else:
                sleep_time = min_interval

            time.sleep(sleep_time)

    def reset_recovery_attempts(self, check_name: Optional[str] = None) -> None:
        """Reset recovery attempt counters."""
        with self._lock:
            if check_name:
                self._recovery_attempts[check_name] = 0
            else:
                for name in self._recovery_attempts:
                    self._recovery_attempts[name] = 0

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of all health checks."""
        with self._lock:
            checks = {}
            for name, check in self._checks.items():
                last = check.last_result
                checks[name] = {
                    "status": check.status.value,
                    "last_check": last.checked_at.isoformat() if last else None,
                    "message": last.message if last else None,
                    "recovery_attempts": self._recovery_attempts.get(name, 0),
                }

            healthy = sum(1 for c in self._checks.values() if c.status == HealthStatus.HEALTHY)
            total = len(self._checks)

            return {
                "overall_status": (
                    HealthStatus.HEALTHY.value if healthy == total
                    else HealthStatus.DEGRADED.value if healthy > 0
                    else HealthStatus.UNHEALTHY.value
                ),
                "healthy_checks": healthy,
                "total_checks": total,
                "checks": checks,
                "recent_events": [
                    {
                        "check": e.check_name,
                        "action": e.action.value,
                        "success": e.success,
                        "timestamp": e.started_at.isoformat(),
                    }
                    for e in self._events[-10:]  # Last 10 events
                ],
            }

    def get_event_history(self, limit: int = 100) -> List[RecoveryEvent]:
        """Get recovery event history."""
        return self._events[-limit:]
