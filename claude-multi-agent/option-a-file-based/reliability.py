"""
Reliability Integration for Option A (File-Based Coordination)

This module integrates the shared reliability patterns into the file-based
coordination system.
"""

import json
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Set

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from reliability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerRegistry
from reliability.retry import RetryWithJitter, RetryConfig, retry_with_jitter
from reliability.fallback import FallbackChain, CacheFallback, DefaultFallback, QueueFallback
from reliability.deadlock import DeadlockDetector, DeadlockRecovery, DeadlockInfo, DeadlockType
from reliability.consistency import DataConsistencyValidator, ConsistencyLevel, ValidationReport
from reliability.backup import BackupManager, BackupConfig, BackupType
from reliability.leader_election import LeaderElection, LeaderElectionConfig
from reliability.split_brain import SplitBrainPrevention, SplitBrainConfig
from reliability.degradation import GracefulDegradation, DegradationLevel, SystemHealth
from reliability.self_healing import SelfHealingManager, HealthCheck, HealthCheckResult, HealthStatus, RecoveryEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COORDINATION_DIR = Path(".coordination")
TASKS_FILE = COORDINATION_DIR / "tasks.json"


class FileCoordinationReliability:
    """
    Reliability wrapper for file-based coordination.

    Provides:
    - Circuit breaker for file operations
    - Retry with jitter for race conditions
    - Automatic backup on state changes
    - Deadlock detection for task dependencies
    - Data consistency validation
    - Leader election for multi-leader scenarios
    - Self-healing for stale workers
    """

    def __init__(
        self,
        coordination_dir: Optional[Path] = None,
        auto_backup: bool = True,
        auto_heal: bool = True,
    ):
        self.coordination_dir = coordination_dir or COORDINATION_DIR
        self.tasks_file = self.coordination_dir / "tasks.json"

        # Initialize components
        self._init_circuit_breakers()
        self._init_retry()
        self._init_fallback()
        self._init_backup(auto_backup)
        self._init_deadlock_detection()
        self._init_consistency_validator()
        self._init_leader_election()
        self._init_split_brain()
        self._init_degradation()
        self._init_self_healing(auto_heal)

    def _init_circuit_breakers(self) -> None:
        """Initialize circuit breakers for file operations."""
        self.file_breaker = CircuitBreaker(
            name="file-operations",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=30.0,
                window_seconds=60.0,
            ),
            on_open=lambda name: logger.warning(f"Circuit {name} opened - file operations disabled"),
            on_close=lambda name: logger.info(f"Circuit {name} closed - file operations restored"),
        )

        self.worker_registry = CircuitBreakerRegistry(
            default_config=CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=60.0,
            )
        )

    def _init_retry(self) -> None:
        """Initialize retry configuration."""
        self.retry_config = RetryConfig(
            max_attempts=3,
            initial_delay_seconds=0.5,
            max_delay_seconds=5.0,
            jitter_factor=0.5,
        )
        self.retry = RetryWithJitter(
            config=self.retry_config,
            on_retry=lambda attempt, error, delay: logger.warning(
                f"Retry attempt {attempt} after {delay:.2f}s: {error}"
            ),
        )

    def _init_fallback(self) -> None:
        """Initialize fallback strategies."""
        self.fallback_chain = FallbackChain()
        self.fallback_chain.add(CacheFallback(max_age_seconds=300))
        self.fallback_chain.add(QueueFallback())
        self.fallback_chain.add(DefaultFallback(default_value={"tasks": []}))

    def _init_backup(self, auto_backup: bool) -> None:
        """Initialize backup manager."""
        self.backup_manager = BackupManager(
            config=BackupConfig(
                backup_dir=str(self.coordination_dir / "backups"),
                max_backups=10,
                auto_backup_interval=300.0,
                compress=True,
            ),
            on_backup=lambda info: logger.info(f"Created backup: {info.id}"),
        )

        if auto_backup:
            self.backup_manager.start_auto_backup(self._get_current_state)

    def _init_deadlock_detection(self) -> None:
        """Initialize deadlock detection."""
        self.deadlock_detector = DeadlockDetector(
            stale_threshold_seconds=300.0,
            check_interval_seconds=30.0,
            on_deadlock=self._handle_deadlock,
        )
        self.deadlock_recovery = DeadlockRecovery(
            on_recovery=lambda dl, action: logger.info(f"Deadlock recovery: {action}"),
        )

    def _init_consistency_validator(self) -> None:
        """Initialize data consistency validator."""
        self.validator = DataConsistencyValidator(
            level=ConsistencyLevel.STANDARD,
            on_error=lambda error: logger.error(f"Consistency error: {error.message}"),
        )

    def _init_leader_election(self) -> None:
        """Initialize leader election."""
        import socket
        node_id = f"{socket.gethostname()}-{os.getpid()}"

        self.leader_election = LeaderElection(
            node_id=node_id,
            config=LeaderElectionConfig(
                heartbeat_interval=5.0,
                election_timeout=15.0,
                lock_file_path=str(self.coordination_dir / "leader.lock"),
            ),
            on_become_leader=lambda: logger.info("Became leader"),
            on_lose_leadership=lambda: logger.warning("Lost leadership"),
        )

    def _init_split_brain(self) -> None:
        """Initialize split-brain prevention."""
        import socket
        node_id = f"{socket.gethostname()}-{os.getpid()}"

        self.split_brain = SplitBrainPrevention(
            node_id=node_id,
            config=SplitBrainConfig(
                quorum_size=1,  # Single node by default
                fence_timeout=30.0,
                fence_file_path=str(self.coordination_dir / "fencing"),
            ),
            on_fenced=lambda node: logger.warning(f"Node fenced: {node}"),
            on_split_detected=lambda splits: logger.error(f"Split-brain detected: {splits}"),
        )

    def _init_degradation(self) -> None:
        """Initialize graceful degradation."""
        self.degradation = GracefulDegradation(
            on_degrade=lambda old, new: logger.warning(f"Degraded: {old.value} -> {new.value}"),
            on_recover=lambda old, new: logger.info(f"Recovered: {old.value} -> {new.value}"),
            auto_degrade=True,
        )

    def _init_self_healing(self, auto_heal: bool) -> None:
        """Initialize self-healing manager."""
        self.self_healing = SelfHealingManager(
            recovery_cooldown=60.0,
            max_recovery_attempts=3,
            on_unhealthy=lambda result: logger.warning(f"Unhealthy: {result.name}"),
            on_recovery=lambda event: logger.info(f"Recovery: {event.check_name} - {event.message}"),
        )

        # Register health checks
        self.self_healing.register_check(TaskQueueHealthCheck(self))

        if auto_heal:
            self.self_healing.start_monitoring()

    def _get_current_state(self) -> Dict[str, Any]:
        """Get current coordination state for backup."""
        if self.tasks_file.exists():
            return json.loads(self.tasks_file.read_text())
        return {"tasks": []}

    def _handle_deadlock(self, deadlock: DeadlockInfo) -> None:
        """Handle detected deadlock."""
        logger.warning(f"Deadlock detected: {deadlock.type.value}")

        # Attempt automatic recovery
        action = self.deadlock_recovery.recover(
            deadlock,
            reset_task=self.reset_task,
            cancel_task=self.cancel_task,
        )

        logger.info(f"Deadlock recovery action: {action}")

    def reset_task(self, task_id: str) -> None:
        """Reset a task to available state."""
        try:
            state = self._get_current_state()
            for task in state.get("tasks", []):
                if task["id"] == task_id:
                    task["status"] = "available"
                    task["claimed_by"] = None
                    task["claimed_at"] = None
                    logger.info(f"Reset task: {task_id}")
                    break
            self.tasks_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.error(f"Failed to reset task {task_id}: {e}")

    def cancel_task(self, task_id: str) -> None:
        """Cancel a task."""
        try:
            state = self._get_current_state()
            for task in state.get("tasks", []):
                if task["id"] == task_id:
                    task["status"] = "cancelled"
                    task["completed_at"] = datetime.now().isoformat()
                    logger.info(f"Cancelled task: {task_id}")
                    break
            self.tasks_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")

    # Public API for reliability-wrapped operations

    def load_tasks_safe(self) -> Dict[str, Any]:
        """Load tasks with reliability features."""
        if not self.file_breaker.can_execute():
            result = self.fallback_chain.execute(
                Exception("Circuit open"),
                {"cache_key": "tasks"}
            )
            return result.value or {"tasks": []}

        try:
            state = self.retry.execute_sync(self._get_current_state)
            self.file_breaker.record_success()

            # Cache for fallback
            self.fallback_chain._strategies[0].set_cache("tasks", state)

            return state

        except Exception as e:
            self.file_breaker.record_failure(e)
            result = self.fallback_chain.execute(e, {"cache_key": "tasks"})
            return result.value or {"tasks": []}

    def save_tasks_safe(self, state: Dict[str, Any]) -> bool:
        """Save tasks with reliability features."""
        if not self.file_breaker.can_execute():
            # Queue the operation for later
            self.fallback_chain.execute(
                Exception("Circuit open"),
                {"operation": "save", "state": state}
            )
            return False

        # Validate before saving
        report = self.validator.validate_state(state)
        if not report.is_valid:
            logger.error(f"Validation failed: {[e.message for e in report.errors]}")
            return False

        try:
            def save():
                self.tasks_file.write_text(json.dumps(state, indent=2))
                return True

            result = self.retry.execute_sync(save)
            self.file_breaker.record_success()

            # Update cache
            self.fallback_chain._strategies[0].set_cache("tasks", state)

            # Create backup on success
            self.backup_manager.create_backup(state, BackupType.FULL, "Post-save backup")

            return result

        except Exception as e:
            self.file_breaker.record_failure(e)
            return False

    def report_worker_activity(self, worker_id: str, success: bool = True) -> None:
        """Report worker activity to circuit breaker."""
        breaker = self.worker_registry.get_or_create(worker_id)
        if success:
            breaker.record_success()
        else:
            breaker.record_failure()

    def is_worker_healthy(self, worker_id: str) -> bool:
        """Check if a worker is healthy."""
        breaker = self.worker_registry.get(worker_id)
        if breaker:
            return not breaker.is_open
        return True

    def start(self) -> None:
        """Start reliability monitoring."""
        self.leader_election.start()
        self.split_brain.start_monitoring()
        self.deadlock_detector.start_monitoring()

        # Register tasks for deadlock detection
        state = self._get_current_state()
        for task in state.get("tasks", []):
            self.deadlock_detector.register_task(
                task["id"],
                dependencies=set(task.get("dependencies", [])),
                status=task.get("status", "available"),
            )

    def stop(self) -> None:
        """Stop reliability monitoring."""
        self.leader_election.stop()
        self.split_brain.stop_monitoring()
        self.deadlock_detector.stop_monitoring()
        self.self_healing.stop_monitoring()
        self.backup_manager.stop_auto_backup()

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        return {
            "leader": self.leader_election.get_status(),
            "split_brain": self.split_brain.get_cluster_status(),
            "degradation": self.degradation.get_status(),
            "self_healing": self.self_healing.get_health_summary(),
            "circuit_breakers": self.worker_registry.get_all_stats(),
            "backups": self.backup_manager.get_stats(),
        }


class TaskQueueHealthCheck(HealthCheck):
    """Health check for the task queue in file-based coordination."""

    def __init__(self, reliability: FileCoordinationReliability):
        super().__init__(
            name="task-queue",
            interval=30.0,
            timeout=10.0,
            failure_threshold=3,
        )
        self._reliability = reliability

    def check(self) -> HealthCheckResult:
        try:
            state = self._reliability._get_current_state()
            tasks = state.get("tasks", [])

            # Count by status
            status_counts = {}
            stale_count = 0
            now = datetime.now()

            for task in tasks:
                status = task.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

                # Check for stale in-progress tasks
                if status in ("claimed", "in_progress"):
                    claimed_at = task.get("claimed_at")
                    if claimed_at:
                        age = now - datetime.fromisoformat(claimed_at)
                        if age > timedelta(minutes=30):
                            stale_count += 1

            if stale_count > 0:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Found {stale_count} stale tasks",
                    details={"status_counts": status_counts, "stale_count": stale_count},
                )

            failed_count = status_counts.get("failed", 0)
            if failed_count > 10:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"High failure count: {failed_count}",
                    details={"status_counts": status_counts},
                )

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Task queue is healthy",
                details={"status_counts": status_counts},
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check task queue: {e}",
            )

    def recover(self) -> RecoveryEvent:
        """Recover by resetting stale tasks."""
        from reliability.self_healing import RecoveryAction, RecoveryEvent

        event = RecoveryEvent(
            check_name=self.name,
            action=RecoveryAction.RESET,
        )

        try:
            state = self._reliability._get_current_state()
            reset_count = 0
            now = datetime.now()

            for task in state.get("tasks", []):
                if task.get("status") in ("claimed", "in_progress"):
                    claimed_at = task.get("claimed_at")
                    if claimed_at:
                        age = now - datetime.fromisoformat(claimed_at)
                        if age > timedelta(minutes=30):
                            task["status"] = "available"
                            task["claimed_by"] = None
                            task["claimed_at"] = None
                            reset_count += 1

            if reset_count > 0:
                self._reliability.save_tasks_safe(state)

            event.success = True
            event.completed_at = datetime.now()
            event.message = f"Reset {reset_count} stale tasks"

        except Exception as e:
            event.success = False
            event.error = str(e)
            event.completed_at = datetime.now()
            event.message = f"Recovery failed: {e}"

        return event


# Convenience function to create a reliability-wrapped coordination
def create_reliable_coordination(
    coordination_dir: Optional[Path] = None,
    auto_backup: bool = True,
    auto_heal: bool = True,
) -> FileCoordinationReliability:
    """Create a reliability-wrapped file coordination instance."""
    return FileCoordinationReliability(
        coordination_dir=coordination_dir,
        auto_backup=auto_backup,
        auto_heal=auto_heal,
    )
