"""
Hybrid Orchestrator

- File-backed, synchronous mode when initialized with coordination_dir
- Async mode (delegate to async_orchestrator.Orchestrator) otherwise
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .async_orchestrator import Orchestrator as AsyncOrchestrator
from .models import Task, TaskStatus, Agent as AgentModel


class OrchestrationError(Exception):
    """Raised when orchestration operations fail validation."""


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

        self._tasks: list[dict[str, Any]] = []
        self._agents: list[dict[str, Any]] = []
        self._discoveries: list[dict[str, Any]] = []

        self._load_state()

        if goal:
            plan_file = self.coordination_dir / "master-plan.md"
            plan_file.write_text(f"# Master Plan\n\n{goal}\n", encoding="utf-8")

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------

    def _read_json(self, path: Path, key: str) -> list[dict[str, Any]]:
        if not path.exists():
            print(f"[orchestrator] missing {path.name}; initializing empty {key}", file=sys.stderr)
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get(key, []) if isinstance(data, dict) else []
        except json.JSONDecodeError:
            print(f"[orchestrator] invalid JSON in {path.name}; initializing empty {key}", file=sys.stderr)
            return []

    def _write_json(self, path: Path, key: str, items: list[dict[str, Any]]) -> None:
        path.write_text(json.dumps({key: items}, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        self._tasks = self._read_json(self.tasks_file, "tasks")
        self._agents = self._read_json(self.agents_file, "agents")
        self._discoveries = self._read_json(self.discoveries_file, "discoveries")

        # Ensure files exist
        if not self.tasks_file.exists():
            self._write_json(self.tasks_file, "tasks", self._tasks)
        if not self.agents_file.exists():
            self._write_json(self.agents_file, "agents", self._agents)
        if not self.discoveries_file.exists():
            self._write_json(self.discoveries_file, "discoveries", self._discoveries)

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
        if not description or not description.strip():
            raise ValueError("Task description cannot be empty")
        if priority < 1 or priority > 10:
            raise ValueError("Task priority must be between 1 and 10")

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
        self._tasks.append(task)
        self._save_tasks()
        return task

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        return next((t for t in self._tasks if t["id"] == task_id), None)

    def get_all_tasks(self) -> list[dict[str, Any]]:
        return list(self._tasks)

    def get_available_tasks(self) -> list[dict[str, Any]]:
        self._load_state()
        done_ids = {t["id"] for t in self._tasks if t["status"] == TaskStatus.DONE.value}
        available = [
            t for t in self._tasks
            if t["status"] == TaskStatus.AVAILABLE.value and all(dep in done_ids for dep in t.get("dependencies", []))
        ]
        return sorted(available, key=lambda t: t.get("priority", 5))

    def claim_task(self, task_id: str, agent_id: str) -> dict[str, Any]:
        self._load_state()
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
        return task

    def complete_task(self, task_id: str, result: str) -> dict[str, Any]:
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
        return task

    def fail_task(self, task_id: str, error: str) -> dict[str, Any]:
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
        return task

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
