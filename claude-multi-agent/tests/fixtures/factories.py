"""
Test data factories for generating realistic test data.

Provides factory functions for creating tasks, agents, and other entities.

Usage:
    from tests.fixtures.factories import TaskFactory, AgentFactory

    task = TaskFactory.create(description="Custom task")
    tasks = TaskFactory.create_batch(10)
    agent = AgentFactory.create_worker()
"""

import uuid
import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


# ============================================================================
# Task Factory
# ============================================================================

@dataclass
class TaskData:
    """Task data structure for testing."""
    id: str
    description: str
    status: str
    priority: int
    claimed_by: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    context: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    claimed_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        d = {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "created_at": self.created_at,
        }
        if self.claimed_by:
            d["claimed_by"] = self.claimed_by
        if self.context:
            d["context"] = self.context
        if self.result:
            d["result"] = self.result
        if self.claimed_at:
            d["claimed_at"] = self.claimed_at
        if self.completed_at:
            d["completed_at"] = self.completed_at
        return d


class TaskFactory:
    """Factory for creating task test data."""

    _counter = 0

    @classmethod
    def _generate_id(cls) -> str:
        """Generate a unique task ID."""
        cls._counter += 1
        return f"task-{datetime.now().strftime('%Y%m%d%H%M%S')}-{cls._counter:04d}"

    @classmethod
    def _random_description(cls) -> str:
        """Generate a random task description."""
        verbs = ["Implement", "Fix", "Refactor", "Add", "Remove", "Update", "Test", "Document"]
        nouns = ["user model", "API endpoint", "database schema", "authentication",
                 "caching layer", "logging system", "error handling", "unit tests"]
        return f"{random.choice(verbs)} {random.choice(nouns)}"

    @classmethod
    def create(
        cls,
        id: Optional[str] = None,
        description: Optional[str] = None,
        status: str = "available",
        priority: int = 5,
        claimed_by: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        context_files: Optional[List[str]] = None,
        hints: str = "",
        **kwargs
    ) -> TaskData:
        """Create a single task with optional overrides."""
        return TaskData(
            id=id or cls._generate_id(),
            description=description or cls._random_description(),
            status=status,
            priority=priority,
            claimed_by=claimed_by,
            dependencies=dependencies or [],
            context={"files": context_files or [], "hints": hints} if context_files or hints else None,
            **kwargs
        )

    @classmethod
    def create_batch(
        cls,
        count: int,
        status: str = "available",
        priority_range: tuple = (1, 10),
        **kwargs
    ) -> List[TaskData]:
        """Create multiple tasks."""
        return [
            cls.create(
                status=status,
                priority=random.randint(*priority_range),
                **kwargs
            )
            for _ in range(count)
        ]

    @classmethod
    def create_with_dependencies(
        cls,
        chain_length: int = 3,
        **kwargs
    ) -> List[TaskData]:
        """Create a chain of dependent tasks."""
        tasks = []
        prev_id = None

        for i in range(chain_length):
            task = cls.create(
                description=f"Chain task {i+1}/{chain_length}",
                dependencies=[prev_id] if prev_id else [],
                priority=i + 1,
                **kwargs
            )
            tasks.append(task)
            prev_id = task.id

        return tasks

    @classmethod
    def create_workflow(
        cls,
        structure: str = "linear"
    ) -> List[TaskData]:
        """Create tasks representing a workflow.

        Structures:
        - linear: A -> B -> C
        - parallel: A, B, C (no dependencies)
        - diamond: A -> B, A -> C -> D, B -> D
        - tree: A -> B, A -> C, B -> D, B -> E
        """
        if structure == "linear":
            return cls.create_with_dependencies(3)

        elif structure == "parallel":
            return cls.create_batch(3)

        elif structure == "diamond":
            a = cls.create(description="Diamond: A", priority=1)
            b = cls.create(description="Diamond: B", priority=2, dependencies=[a.id])
            c = cls.create(description="Diamond: C", priority=2, dependencies=[a.id])
            d = cls.create(description="Diamond: D", priority=3, dependencies=[b.id, c.id])
            return [a, b, c, d]

        elif structure == "tree":
            a = cls.create(description="Tree: A", priority=1)
            b = cls.create(description="Tree: B", priority=2, dependencies=[a.id])
            c = cls.create(description="Tree: C", priority=2, dependencies=[a.id])
            d = cls.create(description="Tree: D", priority=3, dependencies=[b.id])
            e = cls.create(description="Tree: E", priority=3, dependencies=[b.id])
            return [a, b, c, d, e]

        else:
            raise ValueError(f"Unknown structure: {structure}")


# ============================================================================
# Agent Factory
# ============================================================================

@dataclass
class AgentData:
    """Agent data structure for testing."""
    id: str
    role: str
    capabilities: str = ""
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    current_task: Optional[str] = None
    tasks_completed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "role": self.role,
            "capabilities": self.capabilities,
            "registered_at": self.registered_at,
            "last_seen": self.last_seen,
            "current_task": self.current_task,
            "tasks_completed": self.tasks_completed,
        }


class AgentFactory:
    """Factory for creating agent test data."""

    _counter = 0

    @classmethod
    def _generate_id(cls, prefix: str = "agent") -> str:
        """Generate a unique agent ID."""
        cls._counter += 1
        return f"{prefix}-{cls._counter}"

    @classmethod
    def create_worker(
        cls,
        id: Optional[str] = None,
        capabilities: Optional[str] = None,
        **kwargs
    ) -> AgentData:
        """Create a worker agent."""
        default_capabilities = random.choice([
            "python,testing",
            "javascript,frontend",
            "typescript,backend",
            "rust,systems",
            "general",
        ])
        return AgentData(
            id=id or cls._generate_id("worker"),
            role="worker",
            capabilities=capabilities or default_capabilities,
            **kwargs
        )

    @classmethod
    def create_leader(
        cls,
        id: Optional[str] = None,
        **kwargs
    ) -> AgentData:
        """Create a leader agent."""
        return AgentData(
            id=id or cls._generate_id("leader"),
            role="leader",
            capabilities="planning,coordination",
            **kwargs
        )

    @classmethod
    def create_pool(
        cls,
        num_workers: int = 3,
        include_leader: bool = True
    ) -> List[AgentData]:
        """Create a pool of agents."""
        agents = []

        if include_leader:
            agents.append(cls.create_leader())

        for _ in range(num_workers):
            agents.append(cls.create_worker())

        return agents


# ============================================================================
# Discovery Factory
# ============================================================================

@dataclass
class DiscoveryData:
    """Discovery data structure for testing."""
    id: str
    agent_id: str
    content: str
    tags: List[str] = field(default_factory=list)
    related_task: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content,
            "tags": self.tags,
            "created_at": self.created_at,
        }
        if self.related_task:
            d["related_task"] = self.related_task
        return d


class DiscoveryFactory:
    """Factory for creating discovery test data."""

    _counter = 0

    @classmethod
    def _generate_id(cls) -> str:
        """Generate a unique discovery ID."""
        cls._counter += 1
        return f"disc-{cls._counter:04d}"

    @classmethod
    def _random_content(cls) -> str:
        """Generate random discovery content."""
        patterns = [
            "Found potential bug in {}",
            "Performance issue identified in {}",
            "Code duplication in {} and {}",
            "Missing error handling in {}",
            "Security concern: {}",
            "Architecture suggestion: {}",
        ]
        components = ["auth module", "API layer", "database queries", "cache logic",
                      "error handling", "input validation", "logging system"]

        pattern = random.choice(patterns)
        return pattern.format(*random.sample(components, pattern.count("{}")))

    @classmethod
    def create(
        cls,
        id: Optional[str] = None,
        agent_id: str = "worker-1",
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        related_task: Optional[str] = None,
        **kwargs
    ) -> DiscoveryData:
        """Create a single discovery."""
        default_tags = random.sample(["bug", "performance", "security", "architecture",
                                       "code-quality", "documentation"], k=random.randint(1, 3))
        return DiscoveryData(
            id=id or cls._generate_id(),
            agent_id=agent_id,
            content=content or cls._random_content(),
            tags=tags or default_tags,
            related_task=related_task,
            **kwargs
        )

    @classmethod
    def create_batch(cls, count: int, **kwargs) -> List[DiscoveryData]:
        """Create multiple discoveries."""
        return [cls.create(**kwargs) for _ in range(count)]


# ============================================================================
# Coordination State Factory
# ============================================================================

class CoordinationStateFactory:
    """Factory for creating complete coordination state test data."""

    @classmethod
    def create_empty(cls) -> Dict[str, Any]:
        """Create empty coordination state."""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "tasks": [],
        }

    @classmethod
    def create_with_tasks(
        cls,
        num_tasks: int = 10,
        status_distribution: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Create state with tasks in various statuses."""
        if status_distribution is None:
            status_distribution = {
                "available": 0.4,
                "claimed": 0.1,
                "in_progress": 0.2,
                "done": 0.2,
                "failed": 0.1,
            }

        tasks = []
        for i in range(num_tasks):
            # Determine status based on distribution
            rand = random.random()
            cumulative = 0
            status = "available"
            for s, prob in status_distribution.items():
                cumulative += prob
                if rand <= cumulative:
                    status = s
                    break

            task = TaskFactory.create(status=status)

            # Add appropriate fields for non-available statuses
            if status in ("claimed", "in_progress", "done", "failed"):
                task.claimed_by = f"worker-{random.randint(1, 5)}"
                task.claimed_at = datetime.now().isoformat()

            if status in ("done", "failed"):
                task.completed_at = datetime.now().isoformat()
                if status == "done":
                    task.result = {"output": "Completed successfully"}
                else:
                    task.result = {"error": "Task failed"}

            tasks.append(task.to_dict())

        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "tasks": tasks,
        }

    @classmethod
    def create_realistic_project(
        cls,
        project_type: str = "api"
    ) -> Dict[str, Any]:
        """Create realistic project state with typical task structure."""
        if project_type == "api":
            task_descriptions = [
                ("Set up project structure", 1, []),
                ("Configure database connection", 2, [0]),
                ("Implement User model", 2, [0]),
                ("Implement Auth model", 2, [0]),
                ("Create user registration endpoint", 3, [2]),
                ("Create login endpoint", 3, [2, 3]),
                ("Add JWT authentication", 3, [3]),
                ("Implement API rate limiting", 4, [6]),
                ("Add input validation", 4, [4, 5]),
                ("Write unit tests", 5, [4, 5, 6]),
                ("Write integration tests", 5, [9]),
                ("Add API documentation", 5, [4, 5]),
                ("Set up CI/CD", 6, [10]),
            ]
        else:
            # Default simple project
            task_descriptions = [
                ("Initialize project", 1, []),
                ("Implement core feature", 2, [0]),
                ("Add tests", 3, [1]),
                ("Documentation", 4, [1]),
            ]

        tasks = []
        for i, (desc, priority, deps) in enumerate(task_descriptions):
            task = TaskFactory.create(
                description=desc,
                priority=priority,
                dependencies=[tasks[d].id for d in deps] if deps else []
            )
            tasks.append(task)

        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "tasks": [t.to_dict() for t in tasks],
        }
