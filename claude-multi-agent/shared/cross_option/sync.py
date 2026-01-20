"""
Cross-Option Task Synchronization (adv-cross-003)

Provides bidirectional synchronization between different coordination options.
Handles conflict resolution and maintains consistency across multiple
coordination systems running in parallel.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Callable
import json
import hashlib
import threading
import time

from .task_adapter import (
    UniversalTask,
    TaskStatus,
    AdapterFactory,
    TaskAdapter,
)


class SyncDirection(Enum):
    """Direction of synchronization."""
    SOURCE_TO_TARGET = "source_to_target"
    TARGET_TO_SOURCE = "target_to_source"
    BIDIRECTIONAL = "bidirectional"


class ConflictResolution(Enum):
    """Strategy for resolving conflicts."""
    SOURCE_WINS = "source_wins"
    TARGET_WINS = "target_wins"
    NEWEST_WINS = "newest_wins"
    MANUAL = "manual"
    MERGE = "merge"


@dataclass
class SyncConflict:
    """Represents a synchronization conflict."""
    task_id: str
    source_task: UniversalTask
    target_task: UniversalTask
    conflict_type: str  # 'status', 'result', 'both'
    resolved: bool = False
    resolution: Optional[str] = None
    resolved_task: Optional[UniversalTask] = None


@dataclass
class SyncResult:
    """Result of a synchronization operation."""
    success: bool
    direction: SyncDirection
    tasks_synced: int
    tasks_created: int
    tasks_updated: int
    conflicts_found: int
    conflicts_resolved: int
    conflicts: List[SyncConflict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    sync_time: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "direction": self.direction.value,
            "tasks_synced": self.tasks_synced,
            "tasks_created": self.tasks_created,
            "tasks_updated": self.tasks_updated,
            "conflicts_found": self.conflicts_found,
            "conflicts_resolved": self.conflicts_resolved,
            "errors": self.errors,
            "sync_time": self.sync_time.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


class TaskSynchronizer:
    """
    Synchronizes tasks between different coordination options.

    Supports:
    - Unidirectional sync (source -> target)
    - Bidirectional sync with conflict resolution
    - Continuous sync with configurable interval
    - Conflict detection and resolution
    """

    def __init__(
        self,
        source_option: str,
        target_option: str,
        conflict_resolution: ConflictResolution = ConflictResolution.NEWEST_WINS,
        conflict_handler: Optional[Callable[[SyncConflict], UniversalTask]] = None,
    ):
        """
        Initialize the synchronizer.

        Args:
            source_option: Source coordination option ('A', 'B', or 'C')
            target_option: Target coordination option ('A', 'B', or 'C')
            conflict_resolution: Strategy for resolving conflicts
            conflict_handler: Optional custom conflict handler function
        """
        self.source_option = source_option.upper()
        self.target_option = target_option.upper()
        self.conflict_resolution = conflict_resolution
        self.conflict_handler = conflict_handler

        self._source_adapter = AdapterFactory.get_adapter(source_option)
        self._target_adapter = AdapterFactory.get_adapter(target_option)

        self._sync_lock = threading.Lock()
        self._continuous_sync_thread: Optional[threading.Thread] = None
        self._stop_continuous_sync = threading.Event()

        # Track sync state
        self._last_sync_hashes: Dict[str, str] = {}
        self._sync_history: List[SyncResult] = []

    def sync(
        self,
        source_path: str,
        target_path: str,
        direction: SyncDirection = SyncDirection.SOURCE_TO_TARGET,
    ) -> SyncResult:
        """
        Perform a synchronization between source and target.

        Args:
            source_path: Path to source state
            target_path: Path to target state
            direction: Synchronization direction

        Returns:
            SyncResult with details of the sync operation
        """
        start_time = time.time()

        with self._sync_lock:
            result = SyncResult(
                success=False,
                direction=direction,
                tasks_synced=0,
                tasks_created=0,
                tasks_updated=0,
                conflicts_found=0,
                conflicts_resolved=0,
            )

            try:
                # Load tasks from both sides
                source_tasks = self._source_adapter.load_tasks(source_path)
                target_tasks = self._target_adapter.load_tasks(target_path)

                source_by_id = {t.id: t for t in source_tasks}
                target_by_id = {t.id: t for t in target_tasks}

                synced_tasks: List[UniversalTask] = []
                new_target_tasks: List[UniversalTask] = []

                if direction in (SyncDirection.SOURCE_TO_TARGET, SyncDirection.BIDIRECTIONAL):
                    # Sync source -> target
                    for task_id, source_task in source_by_id.items():
                        if task_id in target_by_id:
                            target_task = target_by_id[task_id]

                            # Check for conflicts
                            if self._has_conflict(source_task, target_task):
                                conflict = SyncConflict(
                                    task_id=task_id,
                                    source_task=source_task,
                                    target_task=target_task,
                                    conflict_type=self._get_conflict_type(source_task, target_task),
                                )
                                result.conflicts.append(conflict)
                                result.conflicts_found += 1

                                # Resolve conflict
                                resolved_task = self._resolve_conflict(conflict)
                                if resolved_task:
                                    synced_tasks.append(resolved_task)
                                    conflict.resolved = True
                                    conflict.resolved_task = resolved_task
                                    result.conflicts_resolved += 1
                                    result.tasks_updated += 1
                            else:
                                # No conflict, update if source is newer
                                if self._is_newer(source_task, target_task):
                                    synced_tasks.append(source_task)
                                    result.tasks_updated += 1
                                else:
                                    synced_tasks.append(target_task)
                        else:
                            # New task from source
                            synced_tasks.append(source_task)
                            result.tasks_created += 1

                    # Add tasks only in target
                    for task_id, target_task in target_by_id.items():
                        if task_id not in source_by_id:
                            synced_tasks.append(target_task)

                if direction == SyncDirection.TARGET_TO_SOURCE:
                    # Sync target -> source (reverse)
                    for task_id, target_task in target_by_id.items():
                        if task_id in source_by_id:
                            source_task = source_by_id[task_id]

                            if self._has_conflict(target_task, source_task):
                                conflict = SyncConflict(
                                    task_id=task_id,
                                    source_task=target_task,
                                    target_task=source_task,
                                    conflict_type=self._get_conflict_type(target_task, source_task),
                                )
                                result.conflicts.append(conflict)
                                result.conflicts_found += 1

                                resolved_task = self._resolve_conflict(conflict)
                                if resolved_task:
                                    synced_tasks.append(resolved_task)
                                    conflict.resolved = True
                                    result.conflicts_resolved += 1
                                    result.tasks_updated += 1
                            else:
                                if self._is_newer(target_task, source_task):
                                    synced_tasks.append(target_task)
                                    result.tasks_updated += 1
                                else:
                                    synced_tasks.append(source_task)
                        else:
                            synced_tasks.append(target_task)
                            result.tasks_created += 1

                    for task_id, source_task in source_by_id.items():
                        if task_id not in target_by_id:
                            synced_tasks.append(source_task)

                if direction == SyncDirection.BIDIRECTIONAL:
                    # For bidirectional, also sync target -> source
                    source_updated = list(synced_tasks)
                    self._source_adapter.save_tasks(source_updated, source_path)

                # Save to target
                self._target_adapter.save_tasks(synced_tasks, target_path)

                result.tasks_synced = len(synced_tasks)
                result.success = True

            except Exception as e:
                result.errors.append(str(e))

            result.duration_seconds = time.time() - start_time
            self._sync_history.append(result)

            return result

    def start_continuous_sync(
        self,
        source_path: str,
        target_path: str,
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
        interval_seconds: float = 5.0,
    ) -> None:
        """
        Start continuous synchronization in a background thread.

        Args:
            source_path: Path to source state
            target_path: Path to target state
            direction: Synchronization direction
            interval_seconds: Interval between syncs
        """
        if self._continuous_sync_thread and self._continuous_sync_thread.is_alive():
            raise RuntimeError("Continuous sync already running")

        self._stop_continuous_sync.clear()

        def sync_loop():
            while not self._stop_continuous_sync.is_set():
                try:
                    # Check if state has changed
                    if self._state_changed(source_path, target_path):
                        self.sync(source_path, target_path, direction)
                except Exception:
                    pass  # Log but continue
                self._stop_continuous_sync.wait(interval_seconds)

        self._continuous_sync_thread = threading.Thread(target=sync_loop, daemon=True)
        self._continuous_sync_thread.start()

    def stop_continuous_sync(self) -> None:
        """Stop continuous synchronization."""
        self._stop_continuous_sync.set()
        if self._continuous_sync_thread:
            self._continuous_sync_thread.join(timeout=5.0)
            self._continuous_sync_thread = None

    def get_sync_history(self) -> List[SyncResult]:
        """Get the history of sync operations."""
        return list(self._sync_history)

    def get_pending_conflicts(self) -> List[SyncConflict]:
        """Get unresolved conflicts from all sync operations."""
        conflicts = []
        for result in self._sync_history:
            for conflict in result.conflicts:
                if not conflict.resolved:
                    conflicts.append(conflict)
        return conflicts

    def resolve_conflict_manually(
        self,
        conflict: SyncConflict,
        resolution: str,
        resolved_task: UniversalTask,
    ) -> None:
        """
        Manually resolve a conflict.

        Args:
            conflict: The conflict to resolve
            resolution: Description of how it was resolved
            resolved_task: The resulting task after resolution
        """
        conflict.resolved = True
        conflict.resolution = resolution
        conflict.resolved_task = resolved_task

    def _has_conflict(self, task1: UniversalTask, task2: UniversalTask) -> bool:
        """Check if two tasks have a conflict."""
        # Different status is a conflict
        if task1.status != task2.status:
            # Unless one is clearly newer
            if task1.completed_at and task2.completed_at:
                return abs((task1.completed_at - task2.completed_at).total_seconds()) < 60
            return True

        # Both have results but they differ
        if task1.result and task2.result:
            if task1.result.output != task2.result.output:
                return True

        return False

    def _get_conflict_type(self, task1: UniversalTask, task2: UniversalTask) -> str:
        """Determine the type of conflict."""
        status_conflict = task1.status != task2.status
        result_conflict = (
            task1.result and task2.result and
            task1.result.output != task2.result.output
        )

        if status_conflict and result_conflict:
            return "both"
        elif status_conflict:
            return "status"
        else:
            return "result"

    def _is_newer(self, task1: UniversalTask, task2: UniversalTask) -> bool:
        """Check if task1 is newer than task2."""
        # Compare completed_at first
        if task1.completed_at and task2.completed_at:
            return task1.completed_at > task2.completed_at

        # Then claimed_at
        if task1.claimed_at and task2.claimed_at:
            return task1.claimed_at > task2.claimed_at

        # Then created_at
        if task1.created_at and task2.created_at:
            return task1.created_at > task2.created_at

        # Default: task1 is not newer
        return False

    def _resolve_conflict(self, conflict: SyncConflict) -> Optional[UniversalTask]:
        """Resolve a conflict based on the configured strategy."""
        if self.conflict_handler:
            return self.conflict_handler(conflict)

        if self.conflict_resolution == ConflictResolution.SOURCE_WINS:
            return conflict.source_task

        elif self.conflict_resolution == ConflictResolution.TARGET_WINS:
            return conflict.target_task

        elif self.conflict_resolution == ConflictResolution.NEWEST_WINS:
            if self._is_newer(conflict.source_task, conflict.target_task):
                return conflict.source_task
            else:
                return conflict.target_task

        elif self.conflict_resolution == ConflictResolution.MERGE:
            return self._merge_tasks(conflict.source_task, conflict.target_task)

        elif self.conflict_resolution == ConflictResolution.MANUAL:
            # Don't auto-resolve, return None
            return None

        return conflict.source_task

    def _merge_tasks(
        self,
        task1: UniversalTask,
        task2: UniversalTask,
    ) -> UniversalTask:
        """Merge two conflicting tasks."""
        # Use the more "advanced" status
        status_priority = {
            TaskStatus.AVAILABLE: 0,
            TaskStatus.CLAIMED: 1,
            TaskStatus.IN_PROGRESS: 2,
            TaskStatus.DONE: 3,
            TaskStatus.FAILED: 3,
            TaskStatus.CANCELLED: 4,
        }

        if status_priority.get(task1.status, 0) >= status_priority.get(task2.status, 0):
            merged = UniversalTask(
                id=task1.id,
                description=task1.description,
                status=task1.status,
                priority=task1.priority,
                claimed_by=task1.claimed_by or task2.claimed_by,
                dependencies=list(set(task1.dependencies + task2.dependencies)),
                context=task1.context or task2.context,
                result=task1.result or task2.result,
                created_at=min(task1.created_at, task2.created_at) if task1.created_at and task2.created_at else task1.created_at or task2.created_at,
                claimed_at=task1.claimed_at or task2.claimed_at,
                completed_at=task1.completed_at or task2.completed_at,
                tags=list(set(task1.tags + task2.tags)),
            )
        else:
            merged = UniversalTask(
                id=task2.id,
                description=task2.description,
                status=task2.status,
                priority=task2.priority,
                claimed_by=task2.claimed_by or task1.claimed_by,
                dependencies=list(set(task1.dependencies + task2.dependencies)),
                context=task2.context or task1.context,
                result=task2.result or task1.result,
                created_at=min(task1.created_at, task2.created_at) if task1.created_at and task2.created_at else task1.created_at or task2.created_at,
                claimed_at=task2.claimed_at or task1.claimed_at,
                completed_at=task2.completed_at or task1.completed_at,
                tags=list(set(task1.tags + task2.tags)),
            )

        return merged

    def _state_changed(self, source_path: str, target_path: str) -> bool:
        """Check if state has changed since last sync."""
        source_hash = self._compute_hash(source_path)
        target_hash = self._compute_hash(target_path)

        key = f"{source_path}:{target_path}"
        last_hash = self._last_sync_hashes.get(key, "")
        current_hash = f"{source_hash}:{target_hash}"

        if current_hash != last_hash:
            self._last_sync_hashes[key] = current_hash
            return True

        return False

    def _compute_hash(self, path: str) -> str:
        """Compute a hash of the state file."""
        try:
            p = Path(path)
            if p.is_file():
                content = p.read_bytes()
            elif p.is_dir():
                # Hash the tasks.json file for directories
                tasks_file = p / "tasks.json"
                if tasks_file.exists():
                    content = tasks_file.read_bytes()
                else:
                    return ""
            else:
                return ""
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""


def create_synchronizer(
    source_option: str,
    target_option: str,
    **kwargs,
) -> TaskSynchronizer:
    """
    Create a TaskSynchronizer with the given options.

    Args:
        source_option: Source coordination option
        target_option: Target coordination option
        **kwargs: Additional arguments for TaskSynchronizer

    Returns:
        Configured TaskSynchronizer instance
    """
    return TaskSynchronizer(source_option, target_option, **kwargs)
