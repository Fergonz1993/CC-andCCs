"""
Universal Task Adapter Interface (adv-cross-001)

Provides a common interface for tasks across all three coordination options.
Each option has different internal representations, but this adapter allows
seamless conversion and manipulation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Union
import json


class TaskStatus(Enum):
    """Universal task status enum compatible with all options."""
    AVAILABLE = "available"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, value: str) -> "TaskStatus":
        """Convert string to TaskStatus, handling various formats."""
        normalized = value.lower().replace("-", "_")
        for status in cls:
            if status.value == normalized:
                return status
        raise ValueError(f"Unknown task status: {value}")


@dataclass
class TaskContext:
    """Context information for a task."""
    files: List[str] = field(default_factory=list)
    hints: str = ""
    parent_task: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class TaskResult:
    """Result of a completed task."""
    output: str = ""
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    error: Optional[str] = None
    notes: str = ""
    discoveries: List[str] = field(default_factory=list)
    subtasks_created: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class UniversalTask:
    """
    Universal task representation that works across all options.

    This is the canonical representation that all adapters convert to/from.
    """
    id: str
    description: str
    status: TaskStatus
    priority: int = 5
    claimed_by: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    context: Optional[TaskContext] = None
    result: Optional[TaskResult] = None
    created_at: Optional[datetime] = None
    claimed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    estimated_duration_minutes: Optional[int] = None
    actual_duration_minutes: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    source_option: Optional[str] = None  # 'A', 'B', or 'C'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        d = {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
        }

        if self.claimed_by:
            d["claimed_by"] = self.claimed_by
        if self.dependencies:
            d["dependencies"] = self.dependencies
        if self.context:
            d["context"] = self.context.to_dict()
        if self.result:
            d["result"] = self.result.to_dict()
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        if self.claimed_at:
            d["claimed_at"] = self.claimed_at.isoformat()
        if self.completed_at:
            d["completed_at"] = self.completed_at.isoformat()
        if self.tags:
            d["tags"] = self.tags
        if self.estimated_duration_minutes:
            d["estimated_duration_minutes"] = self.estimated_duration_minutes
        if self.actual_duration_minutes:
            d["actual_duration_minutes"] = self.actual_duration_minutes
        if self.retry_count:
            d["retry_count"] = self.retry_count
        if self.max_retries != 3:
            d["max_retries"] = self.max_retries
        if self.source_option:
            d["source_option"] = self.source_option

        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalTask":
        """Create from dictionary representation."""
        context = None
        if data.get("context"):
            context = TaskContext(**data["context"])

        result = None
        if data.get("result"):
            result = TaskResult(**data["result"])

        created_at = None
        if data.get("created_at"):
            if isinstance(data["created_at"], str):
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            else:
                created_at = data["created_at"]

        claimed_at = None
        if data.get("claimed_at"):
            if isinstance(data["claimed_at"], str):
                claimed_at = datetime.fromisoformat(data["claimed_at"].replace("Z", "+00:00"))
            else:
                claimed_at = data["claimed_at"]

        completed_at = None
        if data.get("completed_at"):
            if isinstance(data["completed_at"], str):
                completed_at = datetime.fromisoformat(data["completed_at"].replace("Z", "+00:00"))
            else:
                completed_at = data["completed_at"]

        return cls(
            id=data["id"],
            description=data["description"],
            status=TaskStatus.from_string(data["status"]),
            priority=data.get("priority", 5),
            claimed_by=data.get("claimed_by"),
            dependencies=data.get("dependencies", []),
            context=context,
            result=result,
            created_at=created_at,
            claimed_at=claimed_at,
            completed_at=completed_at,
            tags=data.get("tags", []),
            estimated_duration_minutes=data.get("estimated_duration_minutes"),
            actual_duration_minutes=data.get("actual_duration_minutes"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            source_option=data.get("source_option"),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "UniversalTask":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class TaskAdapter(ABC):
    """
    Abstract base class for task adapters.

    Each option implements this interface to convert between its
    native format and the UniversalTask format.
    """

    @property
    @abstractmethod
    def option_name(self) -> str:
        """Return the option name ('A', 'B', or 'C')."""
        pass

    @abstractmethod
    def to_universal(self, native_task: Any) -> UniversalTask:
        """Convert a native task to UniversalTask."""
        pass

    @abstractmethod
    def from_universal(self, universal_task: UniversalTask) -> Any:
        """Convert a UniversalTask to native format."""
        pass

    @abstractmethod
    def load_tasks(self, source: str) -> List[UniversalTask]:
        """Load all tasks from the source (file path or connection string)."""
        pass

    @abstractmethod
    def save_tasks(self, tasks: List[UniversalTask], destination: str) -> None:
        """Save all tasks to the destination."""
        pass


class OptionAAdapter(TaskAdapter):
    """
    Adapter for Option A (File-based coordination).

    Handles conversion between Option A's Task dataclass format
    and the UniversalTask format.
    """

    @property
    def option_name(self) -> str:
        return "A"

    def to_universal(self, native_task: Union[Dict[str, Any], Any]) -> UniversalTask:
        """Convert Option A task dict to UniversalTask."""
        # Handle both dict and dataclass formats
        if hasattr(native_task, "to_dict"):
            data = native_task.to_dict()
        elif isinstance(native_task, dict):
            data = native_task
        else:
            data = asdict(native_task)

        context = None
        if data.get("context"):
            ctx = data["context"]
            context = TaskContext(
                files=ctx.get("files", []),
                hints=ctx.get("hints", ""),
                parent_task=ctx.get("parent_task"),
            )

        result = None
        if data.get("result"):
            res = data["result"]
            result = TaskResult(
                output=res.get("output", ""),
                files_modified=res.get("files_modified", []),
                files_created=res.get("files_created", []),
                error=res.get("error"),
                notes=res.get("notes", ""),
            )

        return UniversalTask(
            id=data["id"],
            description=data["description"],
            status=TaskStatus.from_string(data["status"]),
            priority=data.get("priority", 5),
            claimed_by=data.get("claimed_by"),
            dependencies=data.get("dependencies", []),
            context=context,
            result=result,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            claimed_at=datetime.fromisoformat(data["claimed_at"]) if data.get("claimed_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            tags=data.get("tags", []),
            source_option="A",
        )

    def from_universal(self, universal_task: UniversalTask) -> Dict[str, Any]:
        """Convert UniversalTask to Option A format."""
        data = {
            "id": universal_task.id,
            "description": universal_task.description,
            "status": universal_task.status.value,
            "priority": universal_task.priority,
        }

        if universal_task.claimed_by:
            data["claimed_by"] = universal_task.claimed_by

        if universal_task.dependencies:
            data["dependencies"] = universal_task.dependencies

        if universal_task.context:
            data["context"] = {
                "files": universal_task.context.files,
                "hints": universal_task.context.hints,
            }
            if universal_task.context.parent_task:
                data["context"]["parent_task"] = universal_task.context.parent_task

        if universal_task.result:
            data["result"] = {
                "output": universal_task.result.output,
                "files_modified": universal_task.result.files_modified,
                "files_created": universal_task.result.files_created,
            }
            if universal_task.result.error:
                data["result"]["error"] = universal_task.result.error
            if universal_task.result.notes:
                data["result"]["notes"] = universal_task.result.notes

        if universal_task.created_at:
            data["created_at"] = universal_task.created_at.isoformat()
        if universal_task.claimed_at:
            data["claimed_at"] = universal_task.claimed_at.isoformat()
        if universal_task.completed_at:
            data["completed_at"] = universal_task.completed_at.isoformat()

        return data

    def load_tasks(self, source: str) -> List[UniversalTask]:
        """Load tasks from Option A's tasks.json file."""
        import json
        from pathlib import Path

        tasks_file = Path(source)
        if not tasks_file.exists():
            return []

        with open(tasks_file, "r") as f:
            data = json.load(f)

        tasks = []
        for task_data in data.get("tasks", []):
            tasks.append(self.to_universal(task_data))

        return tasks

    def save_tasks(self, tasks: List[UniversalTask], destination: str) -> None:
        """Save tasks to Option A's tasks.json format."""
        import json
        from pathlib import Path
        from datetime import datetime

        native_tasks = [self.from_universal(t) for t in tasks]

        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "tasks": native_tasks,
        }

        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, "w") as f:
            json.dump(data, f, indent=2)


class OptionBAdapter(TaskAdapter):
    """
    Adapter for Option B (MCP Server).

    Handles conversion between Option B's TypeScript-style task format
    and the UniversalTask format.
    """

    @property
    def option_name(self) -> str:
        return "B"

    def to_universal(self, native_task: Dict[str, Any]) -> UniversalTask:
        """Convert Option B task to UniversalTask."""
        context = None
        if native_task.get("context"):
            ctx = native_task["context"]
            context = TaskContext(
                files=ctx.get("files", []),
                hints=ctx.get("hints", ""),
                parent_task=ctx.get("parent_task"),
            )

        result = None
        if native_task.get("result"):
            res = native_task["result"]
            result = TaskResult(
                output=res.get("output", ""),
                files_modified=res.get("files_modified", []),
                files_created=res.get("files_created", []),
                error=res.get("error"),
            )

        return UniversalTask(
            id=native_task["id"],
            description=native_task["description"],
            status=TaskStatus.from_string(native_task["status"]),
            priority=native_task.get("priority", 5),
            claimed_by=native_task.get("claimed_by"),
            dependencies=native_task.get("dependencies", []),
            context=context,
            result=result,
            created_at=datetime.fromisoformat(native_task["created_at"].replace("Z", "+00:00")) if native_task.get("created_at") else None,
            claimed_at=datetime.fromisoformat(native_task["claimed_at"].replace("Z", "+00:00")) if native_task.get("claimed_at") else None,
            completed_at=datetime.fromisoformat(native_task["completed_at"].replace("Z", "+00:00")) if native_task.get("completed_at") else None,
            source_option="B",
        )

    def from_universal(self, universal_task: UniversalTask) -> Dict[str, Any]:
        """Convert UniversalTask to Option B format."""
        data = {
            "id": universal_task.id,
            "description": universal_task.description,
            "status": universal_task.status.value,
            "priority": universal_task.priority,
            "claimed_by": universal_task.claimed_by,
            "dependencies": universal_task.dependencies or [],
            "context": None,
            "result": None,
            "created_at": universal_task.created_at.isoformat() if universal_task.created_at else datetime.now().isoformat(),
            "claimed_at": universal_task.claimed_at.isoformat() if universal_task.claimed_at else None,
            "completed_at": universal_task.completed_at.isoformat() if universal_task.completed_at else None,
        }

        if universal_task.context:
            data["context"] = {
                "files": universal_task.context.files,
                "hints": universal_task.context.hints,
            }
            if universal_task.context.parent_task:
                data["context"]["parent_task"] = universal_task.context.parent_task

        if universal_task.result:
            data["result"] = {
                "output": universal_task.result.output,
                "files_modified": universal_task.result.files_modified,
                "files_created": universal_task.result.files_created,
            }
            if universal_task.result.error:
                data["result"]["error"] = universal_task.result.error

        return data

    def load_tasks(self, source: str) -> List[UniversalTask]:
        """Load tasks from Option B's mcp-state.json file."""
        import json
        from pathlib import Path

        state_file = Path(source)
        if not state_file.exists():
            return []

        with open(state_file, "r") as f:
            data = json.load(f)

        tasks = []
        for task_data in data.get("tasks", []):
            tasks.append(self.to_universal(task_data))

        return tasks

    def save_tasks(self, tasks: List[UniversalTask], destination: str) -> None:
        """Save tasks to Option B's state format."""
        import json
        from pathlib import Path

        # Load existing state or create new
        dest_path = Path(destination)
        if dest_path.exists():
            with open(dest_path, "r") as f:
                data = json.load(f)
        else:
            data = {
                "master_plan": "",
                "goal": "",
                "tasks": [],
                "agents": {},
                "discoveries": [],
                "created_at": datetime.now().isoformat(),
            }

        data["tasks"] = [self.from_universal(t) for t in tasks]
        data["last_activity"] = datetime.now().isoformat()

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w") as f:
            json.dump(data, f, indent=2)


class OptionCAdapter(TaskAdapter):
    """
    Adapter for Option C (Python Orchestrator).

    Handles conversion between Option C's Pydantic model format
    and the UniversalTask format.
    """

    @property
    def option_name(self) -> str:
        return "C"

    def to_universal(self, native_task: Any) -> UniversalTask:
        """Convert Option C task to UniversalTask."""
        # Handle both Pydantic model and dict formats
        if hasattr(native_task, "model_dump"):
            data = native_task.model_dump()
        elif hasattr(native_task, "dict"):
            data = native_task.dict()
        elif isinstance(native_task, dict):
            data = native_task
        else:
            raise TypeError(f"Unsupported native task type: {type(native_task)}")

        context = None
        if data.get("context"):
            ctx = data["context"]
            context = TaskContext(
                files=ctx.get("files", []),
                hints=ctx.get("hints", ""),
                parent_task=ctx.get("parent_task"),
            )

        result = None
        if data.get("result"):
            res = data["result"]
            result = TaskResult(
                output=res.get("output", ""),
                files_modified=res.get("files_modified", []),
                files_created=res.get("files_created", []),
                error=res.get("error"),
                discoveries=res.get("discoveries", []),
                subtasks_created=res.get("subtasks_created", []),
            )

        # Handle status which might be enum or string
        status = data.get("status", "available")
        if hasattr(status, "value"):
            status = status.value

        return UniversalTask(
            id=data["id"],
            description=data["description"],
            status=TaskStatus.from_string(status),
            priority=data.get("priority", 5),
            claimed_by=data.get("claimed_by"),
            dependencies=data.get("dependencies", []),
            context=context,
            result=result,
            created_at=data.get("created_at"),
            claimed_at=data.get("claimed_at"),
            completed_at=data.get("completed_at"),
            source_option="C",
        )

    def from_universal(self, universal_task: UniversalTask) -> Dict[str, Any]:
        """Convert UniversalTask to Option C format."""
        data = {
            "id": universal_task.id,
            "description": universal_task.description,
            "status": universal_task.status.value,
            "priority": universal_task.priority,
            "claimed_by": universal_task.claimed_by,
            "dependencies": universal_task.dependencies or [],
            "context": None,
            "result": None,
            "created_at": universal_task.created_at or datetime.now(),
            "claimed_at": universal_task.claimed_at,
            "completed_at": universal_task.completed_at,
        }

        if universal_task.context:
            data["context"] = {
                "files": universal_task.context.files,
                "hints": universal_task.context.hints,
            }
            if universal_task.context.parent_task:
                data["context"]["parent_task"] = universal_task.context.parent_task

        if universal_task.result:
            data["result"] = {
                "output": universal_task.result.output,
                "files_modified": universal_task.result.files_modified,
                "files_created": universal_task.result.files_created,
                "error": universal_task.result.error,
                "discoveries": universal_task.result.discoveries,
                "subtasks_created": universal_task.result.subtasks_created,
            }

        return data

    def load_tasks(self, source: str) -> List[UniversalTask]:
        """Load tasks from Option C's state file."""
        import json
        from pathlib import Path

        state_file = Path(source)
        if not state_file.exists():
            return []

        with open(state_file, "r") as f:
            data = json.load(f)

        tasks = []
        for task_data in data.get("tasks", []):
            tasks.append(self.to_universal(task_data))

        return tasks

    def save_tasks(self, tasks: List[UniversalTask], destination: str) -> None:
        """Save tasks to Option C's state format."""
        import json
        from pathlib import Path

        # Load existing state or create new
        dest_path = Path(destination)
        if dest_path.exists():
            with open(dest_path, "r") as f:
                data = json.load(f)
        else:
            data = {
                "goal": "",
                "master_plan": "",
                "tasks": [],
                "discoveries": [],
                "working_directory": ".",
                "max_parallel_workers": 3,
                "task_timeout_seconds": 600,
            }

        data["tasks"] = [self.from_universal(t) for t in tasks]
        data["last_activity"] = datetime.now().isoformat()

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w") as f:
            json.dump(data, f, indent=2, default=str)


class AdapterFactory:
    """Factory for creating task adapters."""

    _adapters: Dict[str, type] = {
        "A": OptionAAdapter,
        "B": OptionBAdapter,
        "C": OptionCAdapter,
    }

    @classmethod
    def get_adapter(cls, option: str) -> TaskAdapter:
        """Get an adapter for the specified option."""
        option = option.upper()
        if option not in cls._adapters:
            raise ValueError(f"Unknown option: {option}. Valid options: A, B, C")
        return cls._adapters[option]()

    @classmethod
    def register_adapter(cls, option: str, adapter_class: type) -> None:
        """Register a custom adapter for an option."""
        cls._adapters[option.upper()] = adapter_class

    @classmethod
    def convert_task(
        cls,
        task: Any,
        from_option: str,
        to_option: str,
    ) -> Any:
        """Convert a task from one option format to another."""
        source_adapter = cls.get_adapter(from_option)
        target_adapter = cls.get_adapter(to_option)

        universal = source_adapter.to_universal(task)
        return target_adapter.from_universal(universal)
