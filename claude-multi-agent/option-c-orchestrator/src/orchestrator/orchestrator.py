"""
Hybrid Orchestrator

- File-backed, synchronous mode when initialized with coordination_dir
- Async mode (delegate to async_orchestrator.Orchestrator) otherwise
"""

from __future__ import annotations

import fcntl
import json
import logging
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional

from .async_orchestrator import Orchestrator as AsyncOrchestrator
from .config import DEFAULT_HEARTBEAT_TIMEOUT
from .metrics import MetricsCollector
from .models import TaskStatus
from .planner import CycleDetectedError, TaskDAG

logger = logging.getLogger(__name__)


class OrchestrationError(Exception):
    """Raised when orchestration operations fail validation."""


class TaskSchemaError(Exception):
    """Raised when task data fails schema validation (ATOM-108)."""


# Valid task statuses per schema
VALID_TASK_STATUSES = {"available", "claimed", "in_progress", "done", "failed"}


class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


@dataclass
class Agent:
    id: str
    status: AgentStatus = AgentStatus.ACTIVE
    capabilities: list[str] = field(default_factory=list)
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())


class FileOrchestrator:
    """Synchronous, file-backed orchestrator for tests and simple workflows."""

    def __init__(self, coordination_dir: str, goal: str | None = None):
        self.coordination_dir = Path(coordination_dir)
        self.coordination_dir.mkdir(parents=True, exist_ok=True)

        self.tasks_file = self.coordination_dir / "tasks.json"
        self.agents_file = self.coordination_dir / "agents.json"
        self.discoveries_file = self.coordination_dir / "discoveries.json"
        self._lock_file = self.coordination_dir / ".lock"

        self._tasks: list[dict[str, Any]] = []
        self._agents: list[dict[str, Any]] = []
        self._discoveries: list[dict[str, Any]] = []

        # Metrics collector for queue monitoring
        self._metrics = MetricsCollector(metrics_dir=self.coordination_dir / "metrics")

        self._load_state()

        if goal:
            plan_file = self.coordination_dir / "master-plan.md"
            plan_file.write_text(f"# Master Plan\n\n{goal}\n", encoding="utf-8")

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        """
        Acquire an exclusive file lock for atomic operations.

        Uses fcntl.flock() for cross-process synchronization.
        This prevents TOCTOU race conditions in claim_task() and similar operations.
        """
        self._lock_file.touch(exist_ok=True)
        with open(self._lock_file, "r") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------

    def _read_json(self, path: Path, key: str) -> list[dict[str, Any]]:
        if not path.exists():
            logger.info("Missing %s; initializing empty %s", path.name, key)
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get(key, []) if isinstance(data, dict) else []
        except json.JSONDecodeError as e:
            # Create backup of corrupted file before returning empty
            backup_path = path.with_suffix(f".json.corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            try:
                shutil.copy2(path, backup_path)
                logger.warning(
                    "Invalid JSON in %s (backed up to %s): %s",
                    path.name, backup_path.name, e
                )
            except OSError as backup_error:
                logger.error("Failed to backup corrupted file %s: %s", path.name, backup_error)
            return []

    def _write_json(self, path: Path, key: str, items: list[dict[str, Any]]) -> None:
        """Write JSON with atomic write pattern to prevent corruption."""
        # Write to temp file first, then rename (atomic on POSIX)
        temp_path = path.with_suffix(".json.tmp")
        try:
            temp_path.write_text(json.dumps({key: items}, indent=2), encoding="utf-8")
            temp_path.replace(path)  # Atomic rename
        except OSError:
            # Fallback to direct write if rename fails
            if temp_path.exists():
                temp_path.unlink()
            path.write_text(json.dumps({key: items}, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        self._tasks = self._read_json(self.tasks_file, "tasks")
        self._agents = self._read_json(self.agents_file, "agents")
        self._discoveries = self._read_json(self.discoveries_file, "discoveries")

        # Validate no duplicate task IDs (ATOM-107)
        self._validate_no_duplicate_task_ids()

        # Validate no dependency cycles (ATOM-001)
        self._validate_no_dependency_cycles(self._tasks)

        # Ensure files exist
        if not self.tasks_file.exists():
            self._write_json(self.tasks_file, "tasks", self._tasks)
        if not self.agents_file.exists():
            self._write_json(self.agents_file, "agents", self._agents)
        if not self.discoveries_file.exists():
            self._write_json(self.discoveries_file, "discoveries", self._discoveries)

    def _validate_no_duplicate_task_ids(self) -> None:
        """Validate that no duplicate task IDs exist in the loaded tasks (ATOM-107)."""
        seen_ids: set[str] = set()
        duplicates: list[str] = []
        for task in self._tasks:
            task_id = task.get("id", "")
            if task_id in seen_ids:
                duplicates.append(task_id)
            seen_ids.add(task_id)

        if duplicates:
            logger.warning(
                "Duplicate task IDs detected in tasks.json: %s. "
                "Keeping first occurrence of each duplicate.",
                duplicates
            )
            # Deduplicate, keeping first occurrence
            unique_tasks: list[dict[str, Any]] = []
            seen: set[str] = set()
            for task in self._tasks:
                task_id = task.get("id", "")
                if task_id not in seen:
                    unique_tasks.append(task)
                    seen.add(task_id)
            self._tasks = unique_tasks
            self._save_tasks()

    def _validate_no_dependency_cycles(self, tasks: list[dict[str, Any]]) -> None:
        """Validate that the task dependency graph has no cycles (ATOM-001)."""
        dag = TaskDAG()
        for task in tasks:
            task_id = task.get("id")
            if not task_id:
                continue
            dependencies = task.get("dependencies", []) or []
            if not isinstance(dependencies, list):
                continue
            dag.add_task(task_id, dependencies=dependencies)

        try:
            dag.validate()
        except CycleDetectedError as exc:
            raise OrchestrationError(
                f"Dependency cycle detected: {' -> '.join(exc.cycle)}"
            ) from exc

    def _parse_iso_datetime(self, value: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None

    def _requeue_stale_tasks(self, heartbeat_timeout_seconds: int = DEFAULT_HEARTBEAT_TIMEOUT) -> None:
        """Requeue tasks claimed by stale agents (ATOM-002)."""
        now = datetime.now()
        stale_agent_ids: set[str] = set()

        for agent in self._agents:
            agent_id = agent.get("id", "")
            last_heartbeat = self._parse_iso_datetime(agent.get("last_heartbeat", ""))
            if last_heartbeat is None:
                agent["status"] = AgentStatus.INACTIVE.value
                if agent_id:
                    stale_agent_ids.add(agent_id)
                continue

            elapsed = (now - last_heartbeat).total_seconds()
            if elapsed > heartbeat_timeout_seconds:
                agent["status"] = AgentStatus.INACTIVE.value
                if agent_id:
                    stale_agent_ids.add(agent_id)

        if not stale_agent_ids:
            return

        for task in self._tasks:
            if task.get("assigned_to") in stale_agent_ids and task.get("status") in {
                TaskStatus.CLAIMED.value,
                TaskStatus.IN_PROGRESS.value,
            }:
                task["status"] = TaskStatus.AVAILABLE.value
                task["assigned_to"] = None
                task.pop("claimed_at", None)

        self._save_agents()
        self._save_tasks()

    def _validate_task_schema(self, task: dict[str, Any], task_index: int) -> list[str]:
        """
        Validate a single task against the schema (ATOM-108).

        Returns a list of validation errors (empty if valid).
        """
        errors: list[str] = []

        # Required fields
        required_fields = ["id", "description", "status", "priority"]
        for field_name in required_fields:
            if field_name not in task:
                errors.append(f"Task {task_index}: missing required field '{field_name}'")

        # Type validations
        if "id" in task and not isinstance(task["id"], str):
            errors.append(f"Task {task_index}: 'id' must be a string")

        if "description" in task and not isinstance(task["description"], str):
            errors.append(f"Task {task_index}: 'description' must be a string")

        if "status" in task:
            if not isinstance(task["status"], str):
                errors.append(f"Task {task_index}: 'status' must be a string")
            elif task["status"] not in VALID_TASK_STATUSES:
                errors.append(
                    f"Task {task_index}: 'status' must be one of {VALID_TASK_STATUSES}, "
                    f"got '{task['status']}'"
                )

        if "priority" in task:
            if not isinstance(task["priority"], int):
                errors.append(f"Task {task_index}: 'priority' must be an integer")
            elif not (1 <= task["priority"] <= 10):
                errors.append(f"Task {task_index}: 'priority' must be between 1 and 10")

        if "dependencies" in task:
            if not isinstance(task["dependencies"], list):
                errors.append(f"Task {task_index}: 'dependencies' must be an array")
            elif not all(isinstance(d, str) for d in task["dependencies"]):
                errors.append(f"Task {task_index}: all dependencies must be strings")

        return errors

    def validate_tasks_schema(self, strict: bool = False) -> tuple[bool, list[str]]:
        """
        Validate all tasks against the schema (ATOM-108).

        Args:
            strict: If True, raises TaskSchemaError on validation failure.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        all_errors: list[str] = []

        for i, task in enumerate(self._tasks):
            errors = self._validate_task_schema(task, i)
            all_errors.extend(errors)

        is_valid = len(all_errors) == 0

        if not is_valid:
            logger.warning("Task schema validation failed: %s", all_errors)
            if strict:
                raise TaskSchemaError(f"Schema validation failed: {all_errors}")

        return is_valid, all_errors

    def _save_tasks(self) -> None:
        self._write_json(self.tasks_file, "tasks", self._tasks)

    def _save_agents(self) -> None:
        self._write_json(self.agents_file, "agents", self._agents)

    def _save_discoveries(self) -> None:
        self._write_json(self.discoveries_file, "discoveries", self._discoveries)

    # ---------------------------------------------------------------------
    # Task management
    # ---------------------------------------------------------------------

    def add_task(self, description: str, priority: int = 5, dependencies: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Add a new task to the queue.

        RACE CONDITION FIX: Uses file locking to ensure atomic read-modify-write.
        Without the lock, concurrent add_task calls could lose tasks due to
        overwriting each other's changes.
        """
        if not description or not description.strip():
            raise ValueError("Task description cannot be empty")
        if priority < 1 or priority > 10:
            raise ValueError("Task priority must be between 1 and 10")

        with self._file_lock():
            self._load_state()  # Reload state to get latest tasks
            dependencies = dependencies or []
            existing_ids = {t["id"] for t in self._tasks}
            for dep in dependencies:
                if dep not in existing_ids:
                    raise OrchestrationError(f"Dependency not found: {dep}")

            task = {
                "id": f"task-{uuid.uuid4().hex[:8]}",
                "description": description,
                "status": TaskStatus.AVAILABLE.value,
                "priority": priority,
                "dependencies": dependencies,
                "assigned_to": None,
                "created_at": datetime.now().isoformat(),
            }
            self._validate_no_dependency_cycles(self._tasks + [task])
            self._tasks.append(task)
            self._save_tasks()
            self._metrics.record_task_created(task["id"])
            return task

    def add_task_with_id(
        self,
        task_id: str,
        description: str,
        priority: int = 5,
        dependencies: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Add a task with a specific ID. Raises OrchestrationError if ID already exists (ATOM-107).

        RACE CONDITION FIX: Uses file locking to ensure atomic read-modify-write.
        Without the lock, the duplicate ID check could pass for two concurrent calls,
        resulting in duplicate task IDs in the queue.
        """
        if not task_id or not task_id.strip():
            raise ValueError("Task ID cannot be empty")
        if not description or not description.strip():
            raise ValueError("Task description cannot be empty")
        if priority < 1 or priority > 10:
            raise ValueError("Task priority must be between 1 and 10")

        with self._file_lock():
            self._load_state()  # Reload state to get latest tasks

            # Check for duplicate ID (ATOM-107)
            existing_ids = {t["id"] for t in self._tasks}
            if task_id in existing_ids:
                raise OrchestrationError(f"Duplicate task ID: {task_id}")

            dependencies = dependencies or []
            for dep in dependencies:
                if dep not in existing_ids:
                    raise OrchestrationError(f"Dependency not found: {dep}")

            task = {
                "id": task_id,
                "description": description,
                "status": TaskStatus.AVAILABLE.value,
                "priority": priority,
                "dependencies": dependencies,
                "assigned_to": None,
                "created_at": datetime.now().isoformat(),
            }
            self._validate_no_dependency_cycles(self._tasks + [task])
            self._tasks.append(task)
            self._save_tasks()
            self._metrics.record_task_created(task["id"])
            return task

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        return next((t for t in self._tasks if t["id"] == task_id), None)

    def get_all_tasks(self) -> list[dict[str, Any]]:
        return list(self._tasks)

    def get_available_tasks(self) -> list[dict[str, Any]]:
        # Use file lock since _requeue_stale_tasks writes to disk
        with self._file_lock():
            self._load_state()
            self._requeue_stale_tasks()
            done_ids = {t["id"] for t in self._tasks if t["status"] == TaskStatus.DONE.value}
            available = [
                t for t in self._tasks
                if t["status"] == TaskStatus.AVAILABLE.value and all(dep in done_ids for dep in t.get("dependencies", []))
            ]
            return sorted(available, key=lambda t: t.get("priority", 5))

    def claim_task(self, task_id: str, agent_id: str) -> dict[str, Any]:
        """
        Atomically claim a task for an agent.

        Uses file locking to prevent TOCTOU race conditions where two agents
        could claim the same task simultaneously.
        """
        with self._file_lock():
            # Re-read state while holding lock to ensure consistency
            self._load_state()
            self._requeue_stale_tasks()
            task = self.get_task(task_id)
            if task is None:
                raise OrchestrationError("Task not found")

            if task["status"] != TaskStatus.AVAILABLE.value:
                if task["status"] == TaskStatus.CLAIMED.value and task.get("assigned_to") == agent_id:
                    return task
                raise OrchestrationError("Task not available")

            done_ids = {t["id"] for t in self._tasks if t["status"] == TaskStatus.DONE.value}
            for dep in task.get("dependencies", []):
                if dep not in done_ids:
                    raise OrchestrationError("Dependencies not satisfied")

            task["status"] = TaskStatus.CLAIMED.value
            task["assigned_to"] = agent_id
            task["claimed_at"] = datetime.now().isoformat()
            self._save_tasks()
            self._metrics.record_task_claimed(task_id)
            return task

    def complete_task(self, task_id: str, result: str) -> dict[str, Any]:
        """
        Complete a claimed task with a result.

        RACE CONDITION FIX: Uses file locking to ensure atomic read-modify-write.
        Without the lock, two processes could read stale state and overwrite
        each other's changes.
        """
        with self._file_lock():
            self._load_state()
            task = self.get_task(task_id)
            if task is None:
                raise OrchestrationError("Task not found")
            if task.get("assigned_to") is None or task["status"] != TaskStatus.CLAIMED.value:
                raise OrchestrationError("Task not assigned")

            task["status"] = TaskStatus.DONE.value
            task["result"] = result
            task["completed_at"] = datetime.now().isoformat()
            self._save_tasks()
            self._metrics.record_task_completed(task_id, success=True)
            return task

    def fail_task(self, task_id: str, error: str) -> dict[str, Any]:
        """
        Mark a claimed task as failed with an error message.

        RACE CONDITION FIX: Uses file locking to ensure atomic read-modify-write.
        Without the lock, two processes could read stale state and overwrite
        each other's changes.
        """
        with self._file_lock():
            self._load_state()
            task = self.get_task(task_id)
            if task is None:
                raise OrchestrationError("Task not found")
            if task.get("assigned_to") is None or task["status"] != TaskStatus.CLAIMED.value:
                raise OrchestrationError("Task not assigned")

            task["status"] = TaskStatus.FAILED.value
            task["error"] = error
            task["completed_at"] = datetime.now().isoformat()
            self._save_tasks()
            self._metrics.record_task_completed(task_id, success=False)
            return task

    # ---------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------

    def get_metrics(self, format: str = "json") -> str:
        """Get current queue metrics in the specified format (json or prometheus)."""
        metrics = self._metrics.get_queue_metrics(self._tasks, self._agents)
        return self._metrics.export_metrics(metrics, format)

    def save_metrics(self, path: Optional[Path] = None) -> Path:
        """Save metrics to a file. Returns the path where metrics were saved."""
        metrics = self._metrics.get_queue_metrics(self._tasks, self._agents)
        if path is None:
            path = self.coordination_dir / "metrics" / "latest.json"
        self._metrics.save_metrics(metrics, path)
        return path

    # ---------------------------------------------------------------------
    # Agent management
    # ---------------------------------------------------------------------

    def register_agent(self, agent_id: str, capabilities: Optional[list[str]] = None) -> dict[str, Any]:
        capabilities = capabilities or []
        existing = self.get_agent(agent_id)
        if existing:
            existing["status"] = AgentStatus.ACTIVE.value
            existing["capabilities"] = capabilities
            existing["last_heartbeat"] = datetime.now().isoformat()
            self._save_agents()
            return existing

        agent = {
            "id": agent_id,
            "status": AgentStatus.ACTIVE.value,
            "capabilities": capabilities,
            "last_heartbeat": datetime.now().isoformat(),
        }
        self._agents.append(agent)
        self._save_agents()
        return agent

    def get_agent(self, agent_id: str) -> Optional[dict[str, Any]]:
        return next((a for a in self._agents if a["id"] == agent_id), None)

    def get_all_agents(self) -> list[dict[str, Any]]:
        return list(self._agents)

    def agent_heartbeat(self, agent_id: str) -> None:
        agent = self.get_agent(agent_id)
        if not agent:
            raise OrchestrationError("Agent not found")
        agent["status"] = AgentStatus.ACTIVE.value
        agent["last_heartbeat"] = datetime.now().isoformat()
        self._save_agents()

    def deregister_agent(self, agent_id: str) -> None:
        agent = self.get_agent(agent_id)
        if not agent:
            return
        agent["status"] = AgentStatus.INACTIVE.value
        self._save_agents()

    # ---------------------------------------------------------------------
    # Discoveries
    # ---------------------------------------------------------------------

    def add_discovery(self, title: str, content: str, agent_id: str) -> dict[str, Any]:
        discovery = {
            "id": f"disc-{uuid.uuid4().hex[:8]}",
            "title": title,
            "content": content,
            "agent_id": agent_id,
            "created_at": datetime.now().isoformat(),
        }
        self._discoveries.append(discovery)
        self._save_discoveries()
        return discovery

    def get_discoveries(self) -> list[dict[str, Any]]:
        return list(self._discoveries)


class Orchestrator:
    """Hybrid orchestrator facade used by tests and runtime code."""

    def __init__(
        self,
        coordination_dir: Optional[str] = None,
        goal: Optional[str] = None,
        **kwargs: Any,
    ):
        if coordination_dir is not None:
            self._impl: Any = FileOrchestrator(coordination_dir=coordination_dir, goal=goal)
        else:
            self._impl = AsyncOrchestrator(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)
