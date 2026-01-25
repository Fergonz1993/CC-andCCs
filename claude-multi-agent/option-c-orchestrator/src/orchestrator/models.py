"""
Data models for the Claude Multi-Agent Orchestrator.

Uses Pydantic for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid

from .config import DEFAULT_MODEL


class TaskStatus(str, Enum):
    """Task lifecycle states."""
    PENDING = "pending"           # Created but not yet available
    AVAILABLE = "available"       # Ready to be claimed
    CLAIMED = "claimed"           # Claimed by a worker
    IN_PROGRESS = "in_progress"   # Work has started
    DONE = "done"                 # Successfully completed
    FAILED = "failed"             # Failed to complete
    BLOCKED = "blocked"           # Waiting on dependencies


class AgentRole(str, Enum):
    """Agent roles in the coordination system."""
    LEADER = "leader"
    WORKER = "worker"
    SPECIALIST = "specialist"  # For domain-specific tasks


class TaskContext(BaseModel):
    """Additional context for a task."""
    files: List[str] = Field(default_factory=list, description="Relevant file paths")
    hints: str = Field(default="", description="Hints for the worker")
    parent_task: Optional[str] = Field(default=None, description="Parent task ID for subtasks")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class TaskResult(BaseModel):
    """Result of a completed task."""
    output: str = Field(description="Summary of what was done")
    files_modified: List[str] = Field(default_factory=list)
    files_created: List[str] = Field(default_factory=list)
    files_deleted: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None, description="Error message if failed")
    subtasks_created: List[str] = Field(default_factory=list, description="New task IDs")
    discoveries: List[str] = Field(default_factory=list, description="Important findings")


class Task(BaseModel):
    """A discrete unit of work in the coordination system."""
    id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    description: str = Field(description="What needs to be done")
    status: TaskStatus = Field(default=TaskStatus.AVAILABLE)
    priority: int = Field(default=5, ge=1, le=10, description="1=highest, 10=lowest")

    # Assignment
    claimed_by: Optional[str] = Field(default=None, description="Agent ID")

    # Dependencies
    dependencies: List[str] = Field(default_factory=list, description="Task IDs that must complete first")
    blocks: List[str] = Field(default_factory=list, description="Task IDs blocked by this task")

    # Context and results
    context: TaskContext = Field(default_factory=TaskContext)
    result: Optional[TaskResult] = Field(default=None)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    claimed_at: Optional[datetime] = Field(default=None)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    # Retry tracking
    attempts: int = Field(default=0)
    max_attempts: int = Field(default=3)

    def can_start(self, completed_task_ids: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_task_ids for dep in self.dependencies)

    def claim(self, agent_id: str) -> None:
        """Claim this task for an agent."""
        self.status = TaskStatus.CLAIMED
        self.claimed_by = agent_id
        self.claimed_at = datetime.now()
        self.attempts += 1

    def start(self) -> None:
        """Mark task as in progress."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def complete(self, result: TaskResult) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.DONE
        self.result = result
        self.completed_at = datetime.now()

    def fail(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.result = TaskResult(output="", error=error)
        self.completed_at = datetime.now()

    def reset(self, force: bool = False) -> bool:
        """
        Reset task for retry.

        Only allows reset from FAILED or IN_PROGRESS states (for timeouts).
        Returns True if reset was successful, False if rejected.

        Args:
            force: If True, allow reset from any non-DONE state (use with caution)
        """
        # Check if retry attempts remain
        if self.attempts >= self.max_attempts:
            return False

        # Validate current state - only reset from appropriate states
        # CLAIMED is included because a worker may die after claiming
        valid_reset_states = {TaskStatus.FAILED, TaskStatus.IN_PROGRESS, TaskStatus.CLAIMED}
        if force:
            valid_reset_states.add(TaskStatus.BLOCKED)

        if self.status not in valid_reset_states:
            return False

        # Cannot reset a completed task
        if self.status == TaskStatus.DONE:
            return False

        # Perform atomic-ish reset (all fields updated together)
        self.status = TaskStatus.AVAILABLE
        self.claimed_by = None
        self.claimed_at = None
        self.started_at = None
        self.completed_at = None
        self.result = None
        return True


class AgentMetrics(BaseModel):
    """Performance metrics for an agent."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_time_working: float = 0.0  # seconds
    avg_task_time: float = 0.0
    discoveries_made: int = 0


class Agent(BaseModel):
    """Represents a Claude Code agent instance."""
    id: str = Field(description="Unique identifier (e.g., 'worker-1')")
    role: AgentRole = Field(description="Agent's role")

    # Status
    is_active: bool = Field(default=False)
    current_task: Optional[str] = Field(default=None, description="Current task ID")

    # Communication
    pid: Optional[int] = Field(default=None, description="Process ID")

    # Tracking
    last_heartbeat: datetime = Field(default_factory=datetime.now)
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)

    # Configuration
    working_directory: str = Field(default=".")
    model: str = Field(default=DEFAULT_MODEL)
    max_concurrent_tasks: int = Field(default=1)


class Discovery(BaseModel):
    """An important finding shared between agents."""
    id: str = Field(default_factory=lambda: f"disc-{uuid.uuid4().hex[:8]}")
    agent_id: str
    content: str
    tags: List[str] = Field(default_factory=list)
    related_task: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)


class CoordinationState(BaseModel):
    """Complete state of the coordination system."""
    # Project info
    goal: str = Field(default="", description="Overall project goal")
    master_plan: str = Field(default="", description="High-level plan")
    working_directory: str = Field(default=".")

    # Tasks and agents
    tasks: List[Task] = Field(default_factory=list)
    agents: Dict[str, Agent] = Field(default_factory=dict)
    discoveries: List[Discovery] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)

    # Configuration
    max_parallel_workers: int = Field(default=3)
    task_timeout_seconds: int = Field(default=600)  # 10 minutes

    def get_available_tasks(self) -> List[Task]:
        """Get tasks that are ready to be claimed."""
        done_ids = {t.id for t in self.tasks if t.status == TaskStatus.DONE}
        return [
            t for t in self.tasks
            if t.status == TaskStatus.AVAILABLE and t.can_start(done_ids)
        ]

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def add_task(self, task: Task) -> None:
        """Add a new task."""
        self.tasks.append(task)
        self.last_activity = datetime.now()

    def get_progress(self) -> Dict[str, Any]:
        """Get progress summary."""
        by_status = {}
        for task in self.tasks:
            by_status[task.status.value] = by_status.get(task.status.value, 0) + 1

        total = len(self.tasks)
        done = by_status.get("done", 0)

        return {
            "total_tasks": total,
            "by_status": by_status,
            "percent_complete": round(done / total * 100, 1) if total > 0 else 0,
            "active_agents": sum(1 for a in self.agents.values() if a.is_active),
        }
