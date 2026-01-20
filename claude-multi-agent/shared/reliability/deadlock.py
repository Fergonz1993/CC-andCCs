"""
Deadlock Detection and Recovery (adv-rel-004)

Implements deadlock detection for task dependencies and recovery mechanisms
to unblock stuck coordination scenarios.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Set, List, Any, Callable
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class DeadlockType(str, Enum):
    """Types of deadlocks that can be detected."""
    CIRCULAR_DEPENDENCY = "circular_dependency"  # A -> B -> C -> A
    RESOURCE_CONTENTION = "resource_contention"   # Multiple tasks waiting for same resource
    STALE_LOCK = "stale_lock"                     # Lock held too long
    ORPHANED_TASK = "orphaned_task"               # Task claimed but worker gone


@dataclass
class DeadlockInfo:
    """Information about a detected deadlock."""
    type: DeadlockType
    tasks_involved: List[str]
    resources_involved: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskNode:
    """Represents a task in the dependency graph."""
    task_id: str
    status: str
    claimed_by: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    claimed_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None


class DeadlockDetector:
    """
    Detects deadlocks in task coordination.

    Monitors:
    - Circular dependencies in task graph
    - Stale locks (tasks claimed but not progressing)
    - Resource contention
    - Orphaned tasks (claimed by dead workers)
    """

    def __init__(
        self,
        stale_threshold_seconds: float = 300.0,  # 5 minutes
        check_interval_seconds: float = 30.0,
        on_deadlock: Optional[Callable[[DeadlockInfo], None]] = None,
    ):
        """
        Initialize the deadlock detector.

        Args:
            stale_threshold_seconds: Time after which a claimed task is considered stale
            check_interval_seconds: How often to check for deadlocks
            on_deadlock: Callback when a deadlock is detected
        """
        self.stale_threshold = stale_threshold_seconds
        self.check_interval = check_interval_seconds
        self.on_deadlock = on_deadlock

        self._tasks: Dict[str, TaskNode] = {}
        self._active_workers: Dict[str, datetime] = {}
        self._resources: Dict[str, str] = {}  # resource -> holder task
        self._lock = threading.RLock()
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._detected_deadlocks: List[DeadlockInfo] = []

    def register_task(
        self,
        task_id: str,
        dependencies: Optional[Set[str]] = None,
        status: str = "available",
    ) -> None:
        """Register a task for monitoring."""
        with self._lock:
            node = TaskNode(
                task_id=task_id,
                status=status,
                dependencies=dependencies or set(),
            )
            self._tasks[task_id] = node

            # Update dependents for dependencies
            for dep_id in node.dependencies:
                if dep_id in self._tasks:
                    self._tasks[dep_id].dependents.add(task_id)

    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        claimed_by: Optional[str] = None,
    ) -> None:
        """Update task status."""
        with self._lock:
            if task_id not in self._tasks:
                return

            node = self._tasks[task_id]

            if status:
                node.status = status

            if claimed_by:
                node.claimed_by = claimed_by
                node.claimed_at = datetime.now()

            node.last_activity = datetime.now()

    def register_worker(self, worker_id: str) -> None:
        """Register a worker as active."""
        with self._lock:
            self._active_workers[worker_id] = datetime.now()

    def worker_heartbeat(self, worker_id: str) -> None:
        """Update worker's last heartbeat."""
        with self._lock:
            self._active_workers[worker_id] = datetime.now()

    def unregister_worker(self, worker_id: str) -> None:
        """Mark a worker as inactive."""
        with self._lock:
            self._active_workers.pop(worker_id, None)

    def acquire_resource(self, resource_id: str, task_id: str) -> bool:
        """Try to acquire a resource for a task."""
        with self._lock:
            if resource_id in self._resources:
                return False
            self._resources[resource_id] = task_id
            return True

    def release_resource(self, resource_id: str) -> None:
        """Release a resource."""
        with self._lock:
            self._resources.pop(resource_id, None)

    def detect_circular_dependencies(self) -> List[DeadlockInfo]:
        """
        Detect circular dependencies in the task graph.

        Uses DFS to find cycles in the dependency graph.
        """
        deadlocks = []

        with self._lock:
            # Build adjacency list
            visited: Set[str] = set()
            rec_stack: Set[str] = set()
            cycle_path: List[str] = []

            def dfs(task_id: str, path: List[str]) -> Optional[List[str]]:
                visited.add(task_id)
                rec_stack.add(task_id)
                path.append(task_id)

                node = self._tasks.get(task_id)
                if not node:
                    rec_stack.remove(task_id)
                    path.pop()
                    return None

                for dep_id in node.dependencies:
                    if dep_id not in visited:
                        result = dfs(dep_id, path)
                        if result:
                            return result
                    elif dep_id in rec_stack:
                        # Found cycle
                        cycle_start = path.index(dep_id)
                        return path[cycle_start:] + [dep_id]

                rec_stack.remove(task_id)
                path.pop()
                return None

            for task_id in self._tasks:
                if task_id not in visited:
                    cycle = dfs(task_id, [])
                    if cycle:
                        deadlock = DeadlockInfo(
                            type=DeadlockType.CIRCULAR_DEPENDENCY,
                            tasks_involved=cycle,
                            details={"cycle": " -> ".join(cycle)},
                        )
                        deadlocks.append(deadlock)

        return deadlocks

    def detect_stale_locks(self) -> List[DeadlockInfo]:
        """Detect tasks that have been claimed but not progressing."""
        deadlocks = []
        now = datetime.now()
        threshold = timedelta(seconds=self.stale_threshold)

        with self._lock:
            for task_id, node in self._tasks.items():
                if node.status in ("claimed", "in_progress") and node.claimed_at:
                    age = now - node.claimed_at
                    if age > threshold:
                        deadlock = DeadlockInfo(
                            type=DeadlockType.STALE_LOCK,
                            tasks_involved=[task_id],
                            details={
                                "claimed_by": node.claimed_by,
                                "claimed_at": node.claimed_at.isoformat(),
                                "age_seconds": age.total_seconds(),
                            },
                        )
                        deadlocks.append(deadlock)

        return deadlocks

    def detect_orphaned_tasks(self) -> List[DeadlockInfo]:
        """Detect tasks claimed by workers that are no longer active."""
        deadlocks = []
        now = datetime.now()
        worker_timeout = timedelta(seconds=60)  # 1 minute without heartbeat

        with self._lock:
            # Find inactive workers
            inactive_workers = set()
            for worker_id, last_seen in self._active_workers.items():
                if now - last_seen > worker_timeout:
                    inactive_workers.add(worker_id)

            # Find tasks claimed by inactive workers
            for task_id, node in self._tasks.items():
                if (
                    node.status in ("claimed", "in_progress")
                    and node.claimed_by
                    and node.claimed_by in inactive_workers
                ):
                    deadlock = DeadlockInfo(
                        type=DeadlockType.ORPHANED_TASK,
                        tasks_involved=[task_id],
                        details={
                            "claimed_by": node.claimed_by,
                            "worker_last_seen": self._active_workers.get(
                                node.claimed_by, datetime.min
                            ).isoformat(),
                        },
                    )
                    deadlocks.append(deadlock)

        return deadlocks

    def detect_resource_contention(self) -> List[DeadlockInfo]:
        """Detect resource contention issues."""
        deadlocks = []

        with self._lock:
            # Group tasks waiting for same resource
            waiting_for: Dict[str, List[str]] = defaultdict(list)

            for task_id, node in self._tasks.items():
                if node.status == "blocked":
                    # Check what resource it's waiting for
                    for dep_id in node.dependencies:
                        if dep_id in self._resources:
                            holder = self._resources[dep_id]
                            waiting_for[dep_id].append(task_id)

            # Report resources with multiple waiters
            for resource_id, waiters in waiting_for.items():
                if len(waiters) > 1:
                    deadlock = DeadlockInfo(
                        type=DeadlockType.RESOURCE_CONTENTION,
                        tasks_involved=waiters,
                        resources_involved=[resource_id],
                        details={
                            "holder": self._resources.get(resource_id),
                            "waiter_count": len(waiters),
                        },
                    )
                    deadlocks.append(deadlock)

        return deadlocks

    def run_detection(self) -> List[DeadlockInfo]:
        """Run all detection methods and return any deadlocks found."""
        all_deadlocks = []

        all_deadlocks.extend(self.detect_circular_dependencies())
        all_deadlocks.extend(self.detect_stale_locks())
        all_deadlocks.extend(self.detect_orphaned_tasks())
        all_deadlocks.extend(self.detect_resource_contention())

        for deadlock in all_deadlocks:
            logger.warning(f"Deadlock detected: {deadlock.type.value} - {deadlock.tasks_involved}")
            self._detected_deadlocks.append(deadlock)

            if self.on_deadlock:
                try:
                    self.on_deadlock(deadlock)
                except Exception as e:
                    logger.error(f"Deadlock callback error: {e}")

        return all_deadlocks

    def start_monitoring(self) -> None:
        """Start background deadlock monitoring."""
        if self._running:
            return

        self._running = True
        self._check_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._check_thread.start()
        logger.info("Deadlock detection started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
            self._check_thread = None
        logger.info("Deadlock detection stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self.run_detection()
            except Exception as e:
                logger.error(f"Deadlock detection error: {e}")

            time.sleep(self.check_interval)

    def get_detected_deadlocks(self) -> List[DeadlockInfo]:
        """Get all detected deadlocks."""
        return self._detected_deadlocks.copy()

    def clear_history(self) -> None:
        """Clear detected deadlock history."""
        self._detected_deadlocks.clear()


class DeadlockRecovery:
    """
    Recovery mechanisms for detected deadlocks.

    Provides strategies to recover from different types of deadlocks.
    """

    def __init__(
        self,
        on_recovery: Optional[Callable[[DeadlockInfo, str], None]] = None,
    ):
        """
        Initialize recovery mechanism.

        Args:
            on_recovery: Callback after recovery (deadlock, recovery_action)
        """
        self.on_recovery = on_recovery
        self._recovery_history: List[Dict[str, Any]] = []

    def recover(
        self,
        deadlock: DeadlockInfo,
        reset_task: Callable[[str], None],
        cancel_task: Optional[Callable[[str], None]] = None,
        release_resources: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Attempt to recover from a deadlock.

        Args:
            deadlock: The detected deadlock
            reset_task: Function to reset a task to available
            cancel_task: Function to cancel a task
            release_resources: Function to release all resources held by a task

        Returns:
            Description of recovery action taken
        """
        action = ""

        if deadlock.type == DeadlockType.CIRCULAR_DEPENDENCY:
            # Break the cycle by cancelling the lowest priority task
            # or the task with fewer dependents
            if cancel_task and deadlock.tasks_involved:
                victim = deadlock.tasks_involved[-1]  # Last in cycle
                cancel_task(victim)
                action = f"Cancelled task {victim} to break circular dependency"

        elif deadlock.type == DeadlockType.STALE_LOCK:
            # Reset stale tasks
            for task_id in deadlock.tasks_involved:
                reset_task(task_id)
                if release_resources:
                    release_resources(task_id)
            action = f"Reset {len(deadlock.tasks_involved)} stale task(s)"

        elif deadlock.type == DeadlockType.ORPHANED_TASK:
            # Reset orphaned tasks
            for task_id in deadlock.tasks_involved:
                reset_task(task_id)
                if release_resources:
                    release_resources(task_id)
            action = f"Reset {len(deadlock.tasks_involved)} orphaned task(s)"

        elif deadlock.type == DeadlockType.RESOURCE_CONTENTION:
            # Release contested resources and let tasks retry
            if release_resources:
                for resource in deadlock.resources_involved:
                    release_resources(resource)
            action = f"Released {len(deadlock.resources_involved)} contested resource(s)"

        # Record recovery
        self._recovery_history.append({
            "deadlock_type": deadlock.type.value,
            "tasks": deadlock.tasks_involved,
            "action": action,
            "timestamp": datetime.now().isoformat(),
        })

        logger.info(f"Deadlock recovery: {action}")

        if self.on_recovery:
            try:
                self.on_recovery(deadlock, action)
            except Exception as e:
                logger.error(f"Recovery callback error: {e}")

        return action

    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """Get recovery history."""
        return self._recovery_history.copy()
