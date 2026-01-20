"""
Reliability Integration for Option C (Python Orchestrator)

This module integrates the shared reliability patterns into the Python
orchestrator for enhanced fault tolerance and self-healing.
"""

import asyncio
import logging
import os
import socket
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "shared"))

from reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
)
from reliability.retry import RetryWithJitter, RetryConfig
from reliability.fallback import FallbackChain, CacheFallback, DefaultFallback, QueueFallback
from reliability.deadlock import DeadlockDetector, DeadlockRecovery, DeadlockInfo
from reliability.consistency import DataConsistencyValidator, ConsistencyLevel
from reliability.backup import BackupManager, BackupConfig, BackupType
from reliability.leader_election import LeaderElection, LeaderElectionConfig
from reliability.split_brain import SplitBrainPrevention, SplitBrainConfig
from reliability.degradation import GracefulDegradation, DegradationLevel, SystemHealth, FeatureFlag
from reliability.self_healing import (
    SelfHealingManager,
    HealthCheck,
    HealthCheckResult,
    HealthStatus,
    RecoveryEvent,
    RecoveryAction,
    WorkerHealthCheck,
)

from .models import Task, TaskStatus, CoordinationState

logger = logging.getLogger(__name__)


class OrchestratorReliability:
    """
    Reliability layer for the Python orchestrator.

    Provides comprehensive reliability features:
    - Circuit breakers for agent communication
    - Retry with exponential backoff and jitter
    - Fallback strategies for failures
    - Deadlock detection in task dependencies
    - Data consistency validation
    - Automatic backup and restore
    - Leader election for HA
    - Split-brain prevention
    - Graceful degradation
    - Self-healing mechanisms
    """

    def __init__(
        self,
        state: CoordinationState,
        working_directory: Optional[Path] = None,
        node_id: Optional[str] = None,
    ):
        """
        Initialize reliability for the orchestrator.

        Args:
            state: The coordination state to protect
            working_directory: Working directory for coordination files
            node_id: Unique identifier for this orchestrator node
        """
        self.state = state
        self.working_directory = working_directory or Path(".")
        self.coordination_dir = self.working_directory / ".coordination"
        self.node_id = node_id or f"{socket.gethostname()}-{os.getpid()}"

        # Initialize all reliability components
        self._init_circuit_breakers()
        self._init_retry()
        self._init_fallback()
        self._init_deadlock_detection()
        self._init_consistency_validator()
        self._init_backup_manager()
        self._init_leader_election()
        self._init_split_brain()
        self._init_degradation()
        self._init_self_healing()

        logger.info(f"OrchestratorReliability initialized for node {self.node_id}")

    def _init_circuit_breakers(self) -> None:
        """Initialize circuit breakers for agent communication."""
        # Registry for per-worker circuit breakers
        self.worker_registry = CircuitBreakerRegistry(
            default_config=CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=30.0,
                window_seconds=60.0,
            )
        )

        # Main circuit breaker for orchestrator operations
        self.main_breaker = CircuitBreaker(
            name="orchestrator-main",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=60.0,
            ),
            on_open=self._on_circuit_open,
            on_close=self._on_circuit_close,
        )

    def _init_retry(self) -> None:
        """Initialize retry configuration."""
        self.retry = RetryWithJitter(
            config=RetryConfig(
                max_attempts=3,
                initial_delay_seconds=1.0,
                max_delay_seconds=30.0,
                jitter_factor=0.5,
            ),
            on_retry=self._on_retry,
        )

        # Aggressive retry for critical operations
        self.critical_retry = RetryWithJitter(
            config=RetryConfig(
                max_attempts=5,
                initial_delay_seconds=0.5,
                max_delay_seconds=10.0,
                jitter_factor=0.3,
            ),
        )

    def _init_fallback(self) -> None:
        """Initialize fallback chain."""
        self.fallback = FallbackChain()
        self.fallback.add(CacheFallback(max_age_seconds=300))
        self.fallback.add(QueueFallback())
        self.fallback.add(DefaultFallback(default_value=None))

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
            on_error=lambda e: logger.error(f"Consistency error: {e.message}"),
            on_warning=lambda w: logger.warning(f"Consistency warning: {w.message}"),
        )

    def _init_backup_manager(self) -> None:
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

    def _init_leader_election(self) -> None:
        """Initialize leader election."""
        self.leader_election = LeaderElection(
            node_id=self.node_id,
            config=LeaderElectionConfig(
                heartbeat_interval=5.0,
                election_timeout=15.0,
                lease_duration=30.0,
                lock_file_path=str(self.coordination_dir / "leader.lock"),
                state_file_path=str(self.coordination_dir / "leader-state.json"),
            ),
            on_become_leader=self._on_become_leader,
            on_lose_leadership=self._on_lose_leadership,
        )

    def _init_split_brain(self) -> None:
        """Initialize split-brain prevention."""
        self.split_brain = SplitBrainPrevention(
            node_id=self.node_id,
            config=SplitBrainConfig(
                quorum_size=1,  # Single node by default
                fence_timeout=30.0,
                quarantine_duration=60.0,
                heartbeat_interval=5.0,
                fence_file_path=str(self.coordination_dir / "fencing"),
            ),
            on_fenced=lambda node: logger.warning(f"Node fenced: {node}"),
            on_split_detected=self._on_split_brain,
        )

    def _init_degradation(self) -> None:
        """Initialize graceful degradation."""
        self.degradation = GracefulDegradation(
            on_degrade=self._on_degrade,
            on_recover=self._on_recover,
            auto_degrade=True,
            recovery_delay=60.0,
        )

    def _init_self_healing(self) -> None:
        """Initialize self-healing manager."""
        self.self_healing = SelfHealingManager(
            recovery_cooldown=60.0,
            max_recovery_attempts=3,
            on_unhealthy=lambda r: logger.warning(f"Unhealthy: {r.name} - {r.message}"),
            on_recovery=lambda e: logger.info(f"Recovery: {e.check_name} - {e.message}"),
        )

        # Register health checks
        self.self_healing.register_check(TaskQueueHealthCheck(self))
        self.self_healing.register_check(StateConsistencyHealthCheck(self))

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_circuit_open(self, name: str) -> None:
        """Handle circuit breaker opening."""
        logger.warning(f"Circuit breaker opened: {name}")
        self.degradation.degrade(f"Circuit {name} opened")

    def _on_circuit_close(self, name: str) -> None:
        """Handle circuit breaker closing."""
        logger.info(f"Circuit breaker closed: {name}")
        self.degradation.recover(f"Circuit {name} closed")

    def _on_retry(self, attempt: int, error: Exception, delay: float) -> None:
        """Handle retry event."""
        logger.warning(f"Retry attempt {attempt} after {delay:.2f}s: {error}")

    def _handle_deadlock(self, deadlock: DeadlockInfo) -> None:
        """Handle detected deadlock."""
        logger.error(f"Deadlock detected: {deadlock.type.value}")

        # Attempt recovery
        action = self.deadlock_recovery.recover(
            deadlock,
            reset_task=self._reset_task,
            cancel_task=self._cancel_task,
        )
        logger.info(f"Deadlock recovery action: {action}")

    def _on_become_leader(self) -> None:
        """Handle becoming leader."""
        logger.info(f"Node {self.node_id} became leader")

    def _on_lose_leadership(self) -> None:
        """Handle losing leadership."""
        logger.warning(f"Node {self.node_id} lost leadership")
        self.degradation.degrade("Lost leadership")

    def _on_split_brain(self, splits: List[List[str]]) -> None:
        """Handle split-brain detection."""
        logger.error(f"Split-brain detected: {splits}")
        self.degradation.degrade("Split-brain detected")

    def _on_degrade(self, old: DegradationLevel, new: DegradationLevel) -> None:
        """Handle degradation."""
        logger.warning(f"Degraded: {old.value} -> {new.value}")

    def _on_recover(self, old: DegradationLevel, new: DegradationLevel) -> None:
        """Handle recovery."""
        logger.info(f"Recovered: {old.value} -> {new.value}")

    # =========================================================================
    # Task Operations
    # =========================================================================

    def _reset_task(self, task_id: str) -> None:
        """Reset a task to available state."""
        task = self.state.get_task(task_id)
        if task:
            task.reset()
            logger.info(f"Reset task: {task_id}")

    def _cancel_task(self, task_id: str) -> None:
        """Cancel a task."""
        task = self.state.get_task(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            logger.info(f"Cancelled task: {task_id}")

    # =========================================================================
    # Public API
    # =========================================================================

    def start(self) -> None:
        """Start all reliability monitoring."""
        self.coordination_dir.mkdir(parents=True, exist_ok=True)

        # Start components
        self.leader_election.start()
        self.split_brain.start_monitoring()
        self.deadlock_detector.start_monitoring()
        self.self_healing.start_monitoring()
        self.backup_manager.start_auto_backup(self._get_state_for_backup)

        # Register tasks for deadlock detection
        for task in self.state.tasks:
            self.deadlock_detector.register_task(
                task.id,
                dependencies=set(task.dependencies),
                status=task.status.value,
            )

        logger.info("Reliability monitoring started")

    def stop(self) -> None:
        """Stop all reliability monitoring."""
        self.leader_election.stop()
        self.split_brain.stop_monitoring()
        self.deadlock_detector.stop_monitoring()
        self.self_healing.stop_monitoring()
        self.backup_manager.stop_auto_backup()

        logger.info("Reliability monitoring stopped")

    def _get_state_for_backup(self) -> Dict[str, Any]:
        """Get state dictionary for backup."""
        return self.state.model_dump()

    def get_worker_breaker(self, worker_id: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a worker."""
        return self.worker_registry.get_or_create(worker_id)

    def report_worker_activity(
        self,
        worker_id: str,
        success: bool = True,
        error: Optional[Exception] = None,
    ) -> None:
        """Report worker activity to update circuit breaker."""
        breaker = self.get_worker_breaker(worker_id)
        if success:
            breaker.record_success()
        else:
            breaker.record_failure(error)

        # Update split-brain heartbeat
        self.split_brain.heartbeat(worker_id)

    def is_worker_available(self, worker_id: str) -> bool:
        """Check if a worker is available (circuit not open)."""
        breaker = self.worker_registry.get(worker_id)
        if breaker:
            return not breaker.is_open
        return True

    def can_accept_task(self) -> bool:
        """Check if we can accept new tasks based on degradation level."""
        if not self.degradation.is_feature_enabled(FeatureFlag.TASK_CREATION):
            return False
        if self.split_brain.is_fenced:
            return False
        return self.main_breaker.can_execute()

    def can_claim_task(self) -> bool:
        """Check if we can claim tasks."""
        if not self.degradation.is_feature_enabled(FeatureFlag.TASK_CLAIMING):
            return False
        if self.split_brain.is_fenced:
            return False
        return True

    def report_health(
        self,
        error_rate: float = 0.0,
        queue_depth: int = 0,
        active_workers: int = 0,
        failed_tasks_recent: int = 0,
    ) -> None:
        """Report system health metrics for degradation decisions."""
        health = SystemHealth(
            error_rate=error_rate,
            queue_depth=queue_depth,
            active_workers=active_workers,
            failed_tasks_recent=failed_tasks_recent,
        )
        self.degradation.report_health(health)

    def validate_state(self) -> bool:
        """Validate current state consistency."""
        report = self.validator.validate_state(self.state.model_dump())
        return report.is_valid

    def create_backup(self, description: str = "") -> str:
        """Create a manual backup."""
        info = self.backup_manager.create_backup(
            self._get_state_for_backup(),
            BackupType.FULL,
            description,
        )
        return info.id

    def restore_backup(self, backup_id: str) -> bool:
        """Restore from a backup."""
        state_data, result = self.backup_manager.restore_backup(backup_id)
        if result.success and state_data:
            self.state = CoordinationState.model_validate(state_data)
            return True
        return False

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "node_id": self.node_id,
            "is_leader": self.leader_election.is_leader,
            "degradation_level": self.degradation.current_level.value,
            "can_accept_tasks": self.can_accept_task(),
            "can_claim_tasks": self.can_claim_task(),
            "circuit_breakers": {
                "main": self.main_breaker.to_dict(),
                "workers": self.worker_registry.get_all_stats(),
            },
            "split_brain": self.split_brain.get_cluster_status(),
            "self_healing": self.self_healing.get_health_summary(),
            "backups": self.backup_manager.get_stats(),
            "deadlocks_detected": len(self.deadlock_detector.get_detected_deadlocks()),
        }

    async def execute_with_reliability(
        self,
        operation: Callable[[], Any],
        operation_name: str = "operation",
    ) -> Any:
        """
        Execute an operation with full reliability features.

        Applies circuit breaker, retry with jitter, and fallback.
        """
        if not self.main_breaker.can_execute():
            result = self.fallback.execute(
                Exception("Circuit open"),
                {"operation": operation_name},
            )
            return result.value

        try:
            result = await self.retry.execute_async(operation)
            self.main_breaker.record_success()
            return result
        except Exception as e:
            self.main_breaker.record_failure(e)
            result = self.fallback.execute(e, {"operation": operation_name})
            return result.value


class TaskQueueHealthCheck(HealthCheck):
    """Health check for the task queue."""

    def __init__(self, reliability: OrchestratorReliability):
        super().__init__(
            name="task-queue",
            interval=30.0,
            timeout=10.0,
            failure_threshold=3,
        )
        self._reliability = reliability

    def check(self) -> HealthCheckResult:
        try:
            state = self._reliability.state
            tasks = state.tasks

            # Count status
            status_counts: Dict[str, int] = {}
            stale_count = 0
            now = datetime.now()

            for task in tasks:
                status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1

                # Check for stale tasks
                if task.status in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS):
                    if task.claimed_at:
                        age = now - task.claimed_at
                        if age > timedelta(minutes=30):
                            stale_count += 1

            if stale_count > 0:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Found {stale_count} stale tasks",
                    details={"status_counts": status_counts, "stale_count": stale_count},
                )

            failed_count = status_counts.get(TaskStatus.FAILED.value, 0)
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
        event = RecoveryEvent(
            check_name=self.name,
            action=RecoveryAction.RESET,
        )

        try:
            reset_count = 0
            now = datetime.now()

            for task in self._reliability.state.tasks:
                if task.status in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS):
                    if task.claimed_at:
                        age = now - task.claimed_at
                        if age > timedelta(minutes=30):
                            task.reset()
                            reset_count += 1

            event.success = True
            event.completed_at = datetime.now()
            event.message = f"Reset {reset_count} stale tasks"

        except Exception as e:
            event.success = False
            event.error = str(e)
            event.completed_at = datetime.now()
            event.message = f"Recovery failed: {e}"

        return event


class StateConsistencyHealthCheck(HealthCheck):
    """Health check for state consistency."""

    def __init__(self, reliability: OrchestratorReliability):
        super().__init__(
            name="state-consistency",
            interval=60.0,
            timeout=30.0,
            failure_threshold=2,
        )
        self._reliability = reliability

    def check(self) -> HealthCheckResult:
        try:
            report = self._reliability.validator.validate_state(
                self._reliability.state.model_dump()
            )

            if not report.is_valid:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Validation errors: {len(report.errors)}",
                    details={"errors": [e.message for e in report.errors]},
                )

            if report.has_warnings:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Validation warnings: {len(report.warnings)}",
                    details={"warnings": [w.message for w in report.warnings]},
                )

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="State is consistent",
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Consistency check failed: {e}",
            )

    def recover(self) -> RecoveryEvent:
        """Recover by restoring from latest backup."""
        event = RecoveryEvent(
            check_name=self.name,
            action=RecoveryAction.ROLLBACK,
        )

        try:
            latest = self._reliability.backup_manager.get_latest_backup()
            if latest:
                success = self._reliability.restore_backup(latest.id)
                event.success = success
                event.message = f"Restored from backup {latest.id}" if success else "Restore failed"
            else:
                event.success = False
                event.message = "No backup available for restore"

            event.completed_at = datetime.now()

        except Exception as e:
            event.success = False
            event.error = str(e)
            event.completed_at = datetime.now()
            event.message = f"Recovery failed: {e}"

        return event


def create_orchestrator_reliability(
    state: CoordinationState,
    working_directory: Optional[Path] = None,
) -> OrchestratorReliability:
    """Factory function to create orchestrator reliability layer."""
    return OrchestratorReliability(
        state=state,
        working_directory=working_directory,
    )
