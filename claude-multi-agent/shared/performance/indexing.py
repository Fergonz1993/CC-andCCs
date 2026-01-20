"""
Index structures for fast task queries.

Implements adv-perf-005: Index structures for fast task queries

Features:
- Multi-key indexing (by status, priority, agent)
- Composite indexes for complex queries
- Automatic index maintenance
- Query optimization
"""

import threading
from typing import TypeVar, Generic, Optional, Set, Dict, List, Any, Callable
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import heapq

T = TypeVar("T")


@dataclass
class IndexEntry:
    """An entry in an index."""
    task_id: str
    value: Any
    priority: int = 5


class IndexType(Enum):
    """Types of indexes supported."""
    HASH = "hash"           # O(1) equality lookups
    BTREE = "btree"         # Ordered lookups, range queries
    PRIORITY = "priority"   # Priority queue for task scheduling


class HashIndex(Generic[T]):
    """
    Hash-based index for O(1) equality lookups.

    Used for indexing by:
    - status (available, claimed, done, etc.)
    - claimed_by (agent ID)
    - tags
    """

    def __init__(self, key_extractor: Callable[[T], Any]):
        self._index: Dict[Any, Set[str]] = defaultdict(set)
        self._key_extractor = key_extractor
        self._lock = threading.RLock()

    def add(self, task_id: str, task: T) -> None:
        """Add a task to the index."""
        key = self._key_extractor(task)
        with self._lock:
            if isinstance(key, list):
                for k in key:
                    self._index[k].add(task_id)
            else:
                self._index[key].add(task_id)

    def remove(self, task_id: str, task: T) -> None:
        """Remove a task from the index."""
        key = self._key_extractor(task)
        with self._lock:
            if isinstance(key, list):
                for k in key:
                    self._index[k].discard(task_id)
            else:
                self._index[key].discard(task_id)

    def update(self, task_id: str, old_task: T, new_task: T) -> None:
        """Update index when task changes."""
        old_key = self._key_extractor(old_task)
        new_key = self._key_extractor(new_task)

        if old_key != new_key:
            with self._lock:
                self._index[old_key].discard(task_id)
                if isinstance(new_key, list):
                    for k in new_key:
                        self._index[k].add(task_id)
                else:
                    self._index[new_key].add(task_id)

    def get(self, key: Any) -> Set[str]:
        """Get all task IDs matching the key."""
        with self._lock:
            return self._index.get(key, set()).copy()

    def keys(self) -> List[Any]:
        """Get all indexed keys."""
        with self._lock:
            return list(self._index.keys())


class PriorityIndex:
    """
    Priority-based index for efficient task scheduling.

    Maintains a min-heap of tasks by priority for O(1) access
    to the highest priority available task.
    """

    def __init__(self):
        self._heap: List[tuple[int, str]] = []  # (priority, task_id)
        self._removed: Set[str] = set()  # Lazily removed items
        self._lock = threading.RLock()

    def add(self, task_id: str, priority: int) -> None:
        """Add a task to the priority queue."""
        with self._lock:
            if task_id in self._removed:
                self._removed.discard(task_id)
            heapq.heappush(self._heap, (priority, task_id))

    def remove(self, task_id: str) -> None:
        """Mark a task for lazy removal."""
        with self._lock:
            self._removed.add(task_id)

    def update_priority(self, task_id: str, new_priority: int) -> None:
        """Update a task's priority."""
        with self._lock:
            self._removed.add(task_id)
            heapq.heappush(self._heap, (new_priority, task_id))

    def peek(self) -> Optional[str]:
        """Get the highest priority task without removing it."""
        with self._lock:
            while self._heap and self._heap[0][1] in self._removed:
                heapq.heappop(self._heap)
            return self._heap[0][1] if self._heap else None

    def pop(self) -> Optional[str]:
        """Remove and return the highest priority task."""
        with self._lock:
            while self._heap:
                priority, task_id = heapq.heappop(self._heap)
                if task_id not in self._removed:
                    return task_id
                self._removed.discard(task_id)
            return None

    def get_top_n(self, n: int) -> List[tuple[int, str]]:
        """Get the top N highest priority tasks."""
        with self._lock:
            # Filter out removed items and get top N
            valid = [
                (p, tid) for p, tid in self._heap
                if tid not in self._removed
            ]
            return heapq.nsmallest(n, valid)


class CompositeIndex:
    """
    Composite index for complex queries.

    Supports queries like:
    - status=available AND dependencies_satisfied=True
    - priority<=3 AND claimed_by IS NULL
    """

    def __init__(self, index1: HashIndex, index2: HashIndex):
        self._index1 = index1
        self._index2 = index2
        self._lock = threading.RLock()

    def get_intersection(self, key1: Any, key2: Any) -> Set[str]:
        """Get tasks matching both conditions."""
        with self._lock:
            set1 = self._index1.get(key1)
            set2 = self._index2.get(key2)
            return set1 & set2

    def get_union(self, key1: Any, key2: Any) -> Set[str]:
        """Get tasks matching either condition."""
        with self._lock:
            set1 = self._index1.get(key1)
            set2 = self._index2.get(key2)
            return set1 | set2


class TaskIndex:
    """
    Comprehensive index manager for task queries.

    Provides fast access patterns:
    - O(1) lookup by ID
    - O(1) lookup by status
    - O(1) lookup by claimed_by
    - O(1) get highest priority available task
    - O(1) check if dependencies are satisfied

    Example:
        index = TaskIndex()
        index.add_task(task)

        # Fast queries
        available = index.get_by_status("available")
        highest_priority = index.get_highest_priority_available()
        worker_tasks = index.get_by_agent("worker-1")
    """

    def __init__(self):
        # Primary index by ID
        self._tasks: Dict[str, Any] = {}

        # Secondary indexes
        self._status_index: HashIndex = HashIndex(lambda t: t.get("status"))
        self._agent_index: HashIndex = HashIndex(lambda t: t.get("claimed_by"))
        self._tag_index: HashIndex = HashIndex(lambda t: t.get("tags", []))
        self._priority_index = PriorityIndex()

        # Dependency tracking
        self._depends_on: Dict[str, Set[str]] = defaultdict(set)  # task -> its dependencies
        self._blocked_by: Dict[str, Set[str]] = defaultdict(set)  # task -> tasks depending on it

        # Completed task IDs for fast dependency checking
        self._completed_ids: Set[str] = set()

        self._lock = threading.RLock()

    def add_task(self, task: Dict[str, Any]) -> None:
        """Add a task to all indexes."""
        task_id = task["id"]

        with self._lock:
            self._tasks[task_id] = task
            self._status_index.add(task_id, task)
            self._agent_index.add(task_id, task)
            self._tag_index.add(task_id, task)

            # Add to priority index if available
            if task.get("status") == "available":
                self._priority_index.add(task_id, task.get("priority", 5))

            # Track dependencies
            deps = task.get("dependencies", [])
            self._depends_on[task_id] = set(deps)
            for dep in deps:
                self._blocked_by[dep].add(task_id)

            # Track completion
            if task.get("status") == "done":
                self._completed_ids.add(task_id)

    def remove_task(self, task_id: str) -> None:
        """Remove a task from all indexes."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return

            self._status_index.remove(task_id, task)
            self._agent_index.remove(task_id, task)
            self._tag_index.remove(task_id, task)
            self._priority_index.remove(task_id)

            # Clean up dependency tracking
            for dep in self._depends_on[task_id]:
                self._blocked_by[dep].discard(task_id)
            del self._depends_on[task_id]

            if task_id in self._blocked_by:
                del self._blocked_by[task_id]

            self._completed_ids.discard(task_id)
            del self._tasks[task_id]

    def update_task(self, task_id: str, new_task: Dict[str, Any]) -> None:
        """Update a task in all indexes."""
        with self._lock:
            old_task = self._tasks.get(task_id)
            if not old_task:
                self.add_task(new_task)
                return

            # Update indexes
            self._status_index.update(task_id, old_task, new_task)
            self._agent_index.update(task_id, old_task, new_task)
            self._tag_index.update(task_id, old_task, new_task)

            # Update priority index
            old_status = old_task.get("status")
            new_status = new_task.get("status")

            if old_status == "available" and new_status != "available":
                self._priority_index.remove(task_id)
            elif old_status != "available" and new_status == "available":
                self._priority_index.add(task_id, new_task.get("priority", 5))
            elif new_status == "available":
                old_priority = old_task.get("priority", 5)
                new_priority = new_task.get("priority", 5)
                if old_priority != new_priority:
                    self._priority_index.update_priority(task_id, new_priority)

            # Track completion
            if new_status == "done":
                self._completed_ids.add(task_id)

            self._tasks[task_id] = new_task

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all tasks with a specific status."""
        with self._lock:
            task_ids = self._status_index.get(status)
            return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    def get_by_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all tasks claimed by an agent."""
        with self._lock:
            task_ids = self._agent_index.get(agent_id)
            return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    def get_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Get all tasks with a specific tag."""
        with self._lock:
            task_ids = self._tag_index.get(tag)
            return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    def get_available_with_deps_satisfied(self) -> List[Dict[str, Any]]:
        """Get available tasks whose dependencies are all completed."""
        with self._lock:
            available_ids = self._status_index.get("available")
            result = []

            for task_id in available_ids:
                deps = self._depends_on.get(task_id, set())
                if deps.issubset(self._completed_ids):
                    result.append(self._tasks[task_id])

            return result

    def get_highest_priority_available(self) -> Optional[Dict[str, Any]]:
        """Get the highest priority available task with satisfied dependencies."""
        with self._lock:
            # Get top candidates from priority index
            candidates = self._priority_index.get_top_n(20)

            for priority, task_id in candidates:
                if task_id not in self._tasks:
                    continue

                task = self._tasks[task_id]
                if task.get("status") != "available":
                    continue

                deps = self._depends_on.get(task_id, set())
                if deps.issubset(self._completed_ids):
                    return task

            return None

    def are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies of a task are completed."""
        with self._lock:
            deps = self._depends_on.get(task_id, set())
            return deps.issubset(self._completed_ids)

    def get_blocked_tasks(self, task_id: str) -> List[str]:
        """Get tasks that are waiting on this task."""
        with self._lock:
            return list(self._blocked_by.get(task_id, set()))

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self._lock:
            return {
                "total_tasks": len(self._tasks),
                "completed_tasks": len(self._completed_ids),
                "status_distribution": {
                    key: len(self._status_index.get(key))
                    for key in self._status_index.keys()
                },
            }


# Singleton instance
_task_index: Optional[TaskIndex] = None


def get_task_index() -> TaskIndex:
    """Get or create the global task index."""
    global _task_index
    if _task_index is None:
        _task_index = TaskIndex()
    return _task_index
