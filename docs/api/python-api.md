# Python API Reference

This document provides detailed API reference for the Option C Python Orchestrator.

## Installation

```bash
cd option-c-orchestrator
pip install -e .
```

## Core Classes

### Orchestrator

The main class for coordinating multiple Claude Code agents.

```python
from orchestrator import Orchestrator
```

#### Constructor

```python
Orchestrator(
    working_directory: str = ".",
    max_workers: int = 3,
    model: str = "claude-sonnet-4-20250514",
    task_timeout: int = 600,
    on_task_complete: Optional[Callable[[Task], None]] = None,
    on_discovery: Optional[Callable[[Discovery], None]] = None,
    verbose: bool = True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_directory` | `str` | `"."` | Base directory for the project |
| `max_workers` | `int` | `3` | Maximum number of worker agents |
| `model` | `str` | `"claude-sonnet-4-20250514"` | Claude model to use |
| `task_timeout` | `int` | `600` | Task timeout in seconds |
| `on_task_complete` | `Callable` | `None` | Callback when task completes |
| `on_discovery` | `Callable` | `None` | Callback when discovery is made |
| `verbose` | `bool` | `True` | Enable verbose output |

**Example:**

```python
orch = Orchestrator(
    working_directory="./my-project",
    max_workers=5,
    model="claude-sonnet-4-20250514",
    task_timeout=300,
    on_task_complete=lambda t: print(f"Completed: {t.id}"),
    verbose=True,
)
```

---

#### Methods

##### initialize

```python
async def initialize(self, goal: str, master_plan: str = "") -> None
```

Initialize the orchestration session.

**Parameters:**
- `goal`: The overall project goal
- `master_plan`: Optional high-level plan (leader will create one if empty)

**Example:**

```python
await orch.initialize(
    goal="Build user authentication system",
    master_plan="1. Design schema\n2. Implement endpoints\n3. Add tests"
)
```

---

##### start / stop

```python
async def start(self) -> None
async def stop(self) -> None
```

Start or stop the orchestrator and all agents.

---

##### add_task

```python
def add_task(
    self,
    description: str,
    priority: int = 5,
    dependencies: Optional[list[str]] = None,
    context_files: Optional[list[str]] = None,
    hints: str = "",
) -> Task
```

Add a new task to the queue.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `description` | `str` | required | What needs to be done |
| `priority` | `int` | `5` | Priority 1-10 (lower = higher) |
| `dependencies` | `list[str]` | `None` | Task IDs that must complete first |
| `context_files` | `list[str]` | `None` | Relevant file paths |
| `hints` | `str` | `""` | Hints for the worker |

**Returns:** `Task` object

**Example:**

```python
task = orch.add_task(
    description="Implement user login endpoint",
    priority=1,
    dependencies=["task-001"],
    context_files=["src/auth/login.ts"],
    hints="Use JWT for tokens"
)
print(f"Created: {task.id}")
```

---

##### add_tasks_batch

```python
def add_tasks_batch(self, tasks: list[dict[str, Any]]) -> list[Task]
```

Add multiple tasks at once.

**Example:**

```python
tasks = orch.add_tasks_batch([
    {"description": "Create User model", "priority": 1},
    {"description": "Create login endpoint", "priority": 2, "dependencies": ["task-1"]},
    {"description": "Write tests", "priority": 3},
])
```

---

##### claim_task

```python
async def claim_task(self, agent_id: str) -> Optional[Task]
```

Claim an available task for an agent.

**Returns:** `Task` or `None` if no tasks available

---

##### complete_task

```python
async def complete_task(self, task_id: str, result: TaskResult) -> bool
```

Mark a task as completed.

---

##### fail_task

```python
async def fail_task(self, task_id: str, error: str) -> bool
```

Mark a task as failed.

---

##### run_with_leader_planning

```python
async def run_with_leader_planning(self) -> dict[str, Any]
```

Full orchestration flow with automatic planning.

1. Leader analyzes codebase and creates tasks
2. Workers execute tasks in parallel
3. Results are aggregated

**Returns:**
```python
{
    "goal": str,
    "tasks_completed": int,
    "tasks_failed": int,
    "discoveries": int,
    "summary": str,
}
```

**Example:**

```python
result = await orch.run_with_leader_planning()
print(f"Completed {result['tasks_completed']} tasks")
print(f"Summary: {result['summary']}")
```

---

##### run_with_predefined_tasks

```python
async def run_with_predefined_tasks(self) -> dict[str, Any]
```

Execute predefined tasks without leader planning.

**Example:**

```python
orch.add_task("Task 1", priority=1)
orch.add_task("Task 2", priority=2)
result = await orch.run_with_predefined_tasks()
```

---

##### get_status

```python
def get_status(self) -> dict[str, Any]
```

Get current orchestration status.

**Returns:**
```python
{
    "goal": str,
    "progress": {
        "total_tasks": int,
        "percent_complete": int,
        "by_status": {"available": int, "done": int, ...}
    },
    "agents": [...],
    "discoveries": [...],
    "last_activity": str,
}
```

---

##### save_state / load_state

```python
def save_state(self, filepath: str) -> None
def load_state(self, filepath: str) -> None
```

Persist or restore orchestration state.

**Example:**

```python
# Save checkpoint
orch.save_state("checkpoint.json")

# Later: restore
orch.load_state("checkpoint.json")
```

---

### Task

Represents a task in the coordination system.

```python
from orchestrator.models import Task, TaskStatus, TaskResult, TaskContext
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique task identifier |
| `description` | `str` | What needs to be done |
| `status` | `TaskStatus` | Current status |
| `priority` | `int` | Priority (1 = highest) |
| `claimed_by` | `Optional[str]` | Agent that claimed the task |
| `dependencies` | `list[str]` | Task IDs that must complete first |
| `context` | `TaskContext` | Additional context |
| `result` | `Optional[TaskResult]` | Execution result |
| `created_at` | `datetime` | Creation timestamp |
| `claimed_at` | `Optional[datetime]` | Claim timestamp |
| `completed_at` | `Optional[datetime]` | Completion timestamp |

#### Methods

```python
def claim(self, agent_id: str) -> None
def start(self) -> None
def complete(self, result: TaskResult) -> None
def fail(self, error: str) -> None
```

---

### TaskStatus

Enum for task status values.

```python
class TaskStatus(str, Enum):
    AVAILABLE = "available"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
```

---

### TaskResult

Result of task execution.

```python
@dataclass
class TaskResult:
    output: str
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    error: Optional[str] = None
    discoveries: list[str] = field(default_factory=list)
    subtasks_created: list[str] = field(default_factory=list)
```

---

### TaskContext

Additional context for task execution.

```python
@dataclass
class TaskContext:
    files: list[str] = field(default_factory=list)
    hints: str = ""
    parent_task: Optional[str] = None
```

---

### Discovery

A shared finding between agents.

```python
@dataclass
class Discovery:
    agent_id: str
    content: str
    related_task: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
```

---

### CoordinationState

Complete state of the coordination system.

```python
@dataclass
class CoordinationState:
    goal: str = ""
    master_plan: str = ""
    working_directory: str = "."
    tasks: list[Task] = field(default_factory=list)
    discoveries: list[Discovery] = field(default_factory=list)
    max_parallel_workers: int = 3
    task_timeout_seconds: int = 600
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
```

#### Methods

```python
def add_task(self, task: Task) -> None
def get_task(self, task_id: str) -> Optional[Task]
def get_available_tasks(self) -> list[Task]
def get_progress(self) -> dict[str, Any]
```

---

## Complete Example

```python
import asyncio
from orchestrator import Orchestrator
from orchestrator.models import TaskResult

async def main():
    # Create orchestrator
    orch = Orchestrator(
        working_directory="./my-project",
        max_workers=3,
        verbose=True,
    )

    # Initialize with goal
    await orch.initialize("Build user authentication system")

    # Add tasks manually
    task1 = orch.add_task("Create User model", priority=1)
    task2 = orch.add_task(
        "Implement login endpoint",
        priority=2,
        dependencies=[task1.id],
        context_files=["src/models/user.ts"],
        hints="Use bcrypt for password hashing"
    )
    task3 = orch.add_task(
        "Write unit tests",
        priority=3,
        dependencies=[task1.id, task2.id]
    )

    # Run orchestration
    result = await orch.run_with_predefined_tasks()

    # Check results
    print(f"Goal: {result['goal']}")
    print(f"Tasks completed: {result['tasks_completed']}")
    print(f"Tasks failed: {result['tasks_failed']}")
    print(f"Discoveries: {result['discoveries']}")
    print(f"\nSummary:\n{result['summary']}")

    # Save state for later
    orch.save_state("orchestration-state.json")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Callback Examples

### Task Completion Callback

```python
def on_task_complete(task: Task):
    print(f"Task {task.id} completed!")
    if task.result:
        print(f"  Output: {task.result.output[:100]}...")
        print(f"  Files modified: {task.result.files_modified}")

orch = Orchestrator(on_task_complete=on_task_complete)
```

### Discovery Callback

```python
def on_discovery(discovery: Discovery):
    print(f"Discovery by {discovery.agent_id}:")
    print(f"  {discovery.content}")
    print(f"  Tags: {discovery.tags}")

orch = Orchestrator(on_discovery=on_discovery)
```

---

## Error Handling

```python
from orchestrator.exceptions import (
    OrchestrationError,
    TaskNotFoundError,
    AgentNotFoundError,
    TimeoutError,
)

try:
    result = await orch.run_with_leader_planning()
except TimeoutError as e:
    print(f"Task timed out: {e}")
except OrchestrationError as e:
    print(f"Orchestration failed: {e}")
```
