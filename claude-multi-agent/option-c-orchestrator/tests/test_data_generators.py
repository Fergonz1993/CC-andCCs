"""
Test data generators for orchestrator testing.

Feature: adv-test-015 - Test data generators

This module provides factories and generators for creating realistic
test data with configurable complexity levels.

Usage:
    from tests.test_data_generators import TaskFactory, ScenarioGenerator

    # Generate a single task
    task = TaskFactory.create()

    # Generate a complex scenario
    scenario = ScenarioGenerator.generate_workflow_scenario(complexity="high")
"""

import random
import string
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from orchestrator.models import (
    Task,
    TaskStatus,
    TaskResult,
    TaskContext,
    Discovery,
    CoordinationState,
    AgentRole,
    Agent,
    AgentMetrics,
)


class ComplexityLevel(str, Enum):
    """Complexity levels for generated test data."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


# =============================================================================
# Faker-like utilities (no external dependencies)
# =============================================================================

class FakeDataProvider:
    """Provides fake data for test generation."""

    TECH_WORDS = [
        "api", "database", "service", "component", "module", "interface",
        "handler", "controller", "model", "view", "router", "middleware",
        "utility", "helper", "config", "settings", "auth", "cache",
    ]

    VERBS = [
        "create", "implement", "update", "fix", "refactor", "add",
        "remove", "optimize", "test", "validate", "integrate", "configure",
    ]

    FILE_EXTENSIONS = [".py", ".ts", ".js", ".json", ".yaml", ".md"]

    TAGS = [
        "backend", "frontend", "api", "database", "testing", "security",
        "performance", "documentation", "refactoring", "bug-fix",
    ]

    @classmethod
    def task_description(cls) -> str:
        """Generate a realistic task description."""
        verb = random.choice(cls.VERBS)
        tech = random.choice(cls.TECH_WORDS)
        return f"{verb.capitalize()} {tech} functionality"

    @classmethod
    def file_path(cls, base_dir: str = "src") -> str:
        """Generate a realistic file path."""
        tech = random.choice(cls.TECH_WORDS)
        ext = random.choice(cls.FILE_EXTENSIONS)
        return f"{base_dir}/{tech}{ext}"

    @classmethod
    def discovery_content(cls) -> str:
        """Generate realistic discovery content."""
        templates = [
            "Found pattern X in the codebase that could be reused",
            "API endpoint requires authentication token",
            "Database schema needs migration for new field",
            "Performance bottleneck identified in query",
            "Security consideration: input validation needed",
            "Documentation is outdated for this component",
        ]
        return random.choice(templates)

    @classmethod
    def error_message(cls) -> str:
        """Generate realistic error message."""
        errors = [
            "Module not found: dependency missing",
            "Type error: expected string, got int",
            "Connection refused: database unavailable",
            "Permission denied: insufficient access",
            "Timeout: operation exceeded time limit",
            "Validation failed: required field missing",
        ]
        return random.choice(errors)

    @classmethod
    def agent_id(cls, role: str = "worker") -> str:
        """Generate agent ID."""
        return f"{role}-{random.randint(1, 999):03d}"

    @classmethod
    def tags(cls, count: int = 2) -> List[str]:
        """Generate random tags."""
        return random.sample(cls.TAGS, min(count, len(cls.TAGS)))


# =============================================================================
# Task Factory
# =============================================================================

class TaskFactory:
    """
    Factory for creating Task instances with various configurations.

    Supports:
    - Default task creation
    - Task creation with specific states
    - Batch task creation
    - Task sequences with dependencies
    """

    _sequence = 0

    @classmethod
    def _next_sequence(cls) -> int:
        cls._sequence += 1
        return cls._sequence

    @classmethod
    def create(
        self,
        description: Optional[str] = None,
        status: TaskStatus = TaskStatus.AVAILABLE,
        priority: int = 5,
        dependencies: Optional[List[str]] = None,
        context_files: Optional[List[str]] = None,
        hints: str = "",
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Task:
        """
        Create a single Task with the given parameters.

        Args:
            description: Task description (auto-generated if None)
            status: Initial status
            priority: Task priority (1-10)
            dependencies: List of dependency task IDs
            context_files: Relevant file paths
            hints: Hints for the worker
            tags: Tags for categorization
            **kwargs: Additional Task model fields

        Returns:
            Configured Task instance
        """
        if description is None:
            description = FakeDataProvider.task_description()

        if context_files is None:
            context_files = [FakeDataProvider.file_path() for _ in range(random.randint(0, 3))]

        if tags is None:
            tags = FakeDataProvider.tags(random.randint(1, 3))

        task = Task(
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            context=TaskContext(
                files=context_files,
                hints=hints,
                tags=tags,
            ),
            **kwargs
        )

        # Apply status changes
        if status != TaskStatus.AVAILABLE:
            TaskFactory._apply_status(task, status)

        return task

    @classmethod
    def _apply_status(cls, task: Task, status: TaskStatus) -> None:
        """Apply a specific status to a task."""
        if status in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS, TaskStatus.DONE, TaskStatus.FAILED):
            task.claim(FakeDataProvider.agent_id())

        if status in (TaskStatus.IN_PROGRESS, TaskStatus.DONE, TaskStatus.FAILED):
            task.start()

        if status == TaskStatus.DONE:
            task.complete(TaskResultFactory.create())

        if status == TaskStatus.FAILED:
            task.fail(FakeDataProvider.error_message())

    @classmethod
    def create_batch(
        cls,
        count: int,
        status_distribution: Optional[Dict[TaskStatus, float]] = None,
        **kwargs
    ) -> List[Task]:
        """
        Create multiple tasks with optional status distribution.

        Args:
            count: Number of tasks to create
            status_distribution: Dict mapping status to probability
            **kwargs: Common parameters for all tasks

        Returns:
            List of Task instances
        """
        if status_distribution is None:
            status_distribution = {TaskStatus.AVAILABLE: 1.0}

        statuses = list(status_distribution.keys())
        weights = list(status_distribution.values())

        tasks = []
        for _ in range(count):
            status = random.choices(statuses, weights=weights)[0]
            task = cls.create(status=status, **kwargs)
            tasks.append(task)

        return tasks

    @classmethod
    def create_dependency_chain(
        cls,
        length: int,
        base_priority: int = 1
    ) -> List[Task]:
        """
        Create a chain of tasks with sequential dependencies.

        Args:
            length: Number of tasks in the chain
            base_priority: Priority of the first task

        Returns:
            List of tasks where each depends on the previous
        """
        tasks = []
        prev_id = None

        for i in range(length):
            deps = [prev_id] if prev_id else []
            task = cls.create(
                description=f"Chain task {i + 1} of {length}",
                priority=base_priority + i,
                dependencies=deps
            )
            tasks.append(task)
            prev_id = task.id

        return tasks

    @classmethod
    def create_parallel_group(
        cls,
        count: int,
        parent_id: Optional[str] = None
    ) -> List[Task]:
        """
        Create tasks that can run in parallel.

        Args:
            count: Number of parallel tasks
            parent_id: Optional parent task ID for all tasks

        Returns:
            List of independent tasks
        """
        dependencies = [parent_id] if parent_id else []

        return [
            cls.create(
                description=f"Parallel task {i + 1}",
                dependencies=dependencies,
                context=TaskContext(parent_task=parent_id) if parent_id else TaskContext()
            )
            for i in range(count)
        ]


# =============================================================================
# Task Result Factory
# =============================================================================

class TaskResultFactory:
    """Factory for creating TaskResult instances."""

    @classmethod
    def create(
        cls,
        output: Optional[str] = None,
        files_modified: Optional[List[str]] = None,
        files_created: Optional[List[str]] = None,
        discoveries: Optional[List[str]] = None,
        error: Optional[str] = None,
        **kwargs
    ) -> TaskResult:
        """Create a TaskResult with realistic data."""
        if output is None:
            output = f"Completed task successfully. Modified {random.randint(1, 5)} files."

        if files_modified is None:
            files_modified = [
                FakeDataProvider.file_path()
                for _ in range(random.randint(1, 3))
            ]

        if files_created is None:
            files_created = [
                FakeDataProvider.file_path()
                for _ in range(random.randint(0, 2))
            ]

        if discoveries is None:
            discoveries = [
                FakeDataProvider.discovery_content()
                for _ in range(random.randint(0, 2))
            ]

        return TaskResult(
            output=output,
            files_modified=files_modified,
            files_created=files_created,
            discoveries=discoveries,
            error=error,
            **kwargs
        )

    @classmethod
    def create_error(cls, error: Optional[str] = None) -> TaskResult:
        """Create an error TaskResult."""
        return TaskResult(
            output="",
            error=error or FakeDataProvider.error_message()
        )


# =============================================================================
# Discovery Factory
# =============================================================================

class DiscoveryFactory:
    """Factory for creating Discovery instances."""

    @classmethod
    def create(
        cls,
        agent_id: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        related_task: Optional[str] = None,
        **kwargs
    ) -> Discovery:
        """Create a Discovery with realistic data."""
        return Discovery(
            agent_id=agent_id or FakeDataProvider.agent_id(),
            content=content or FakeDataProvider.discovery_content(),
            tags=tags or FakeDataProvider.tags(2),
            related_task=related_task,
            **kwargs
        )

    @classmethod
    def create_batch(cls, count: int, **kwargs) -> List[Discovery]:
        """Create multiple discoveries."""
        return [cls.create(**kwargs) for _ in range(count)]


# =============================================================================
# Agent Factory
# =============================================================================

class AgentFactory:
    """Factory for creating Agent instances."""

    @classmethod
    def create_worker(
        cls,
        agent_id: Optional[str] = None,
        is_active: bool = True,
        current_task: Optional[str] = None,
        **kwargs
    ) -> Agent:
        """Create a worker agent."""
        return Agent(
            id=agent_id or FakeDataProvider.agent_id("worker"),
            role=AgentRole.WORKER,
            is_active=is_active,
            current_task=current_task,
            **kwargs
        )

    @classmethod
    def create_leader(cls, **kwargs) -> Agent:
        """Create a leader agent."""
        return Agent(
            id="leader",
            role=AgentRole.LEADER,
            is_active=True,
            **kwargs
        )

    @classmethod
    def create_pool(cls, worker_count: int = 3) -> List[Agent]:
        """Create a pool of agents with leader and workers."""
        agents = [cls.create_leader()]
        agents.extend([
            cls.create_worker(agent_id=f"worker-{i + 1}")
            for i in range(worker_count)
        ])
        return agents


# =============================================================================
# Coordination State Factory
# =============================================================================

class CoordinationStateFactory:
    """Factory for creating CoordinationState instances."""

    @classmethod
    def create(
        cls,
        goal: str = "Complete the project",
        task_count: int = 10,
        complexity: ComplexityLevel = ComplexityLevel.MEDIUM,
        **kwargs
    ) -> CoordinationState:
        """
        Create a CoordinationState with realistic data.

        Args:
            goal: Project goal
            task_count: Number of tasks to generate
            complexity: Complexity level affects interconnections

        Returns:
            Configured CoordinationState
        """
        state = CoordinationState(
            goal=goal,
            master_plan=f"Plan for: {goal}",
            **kwargs
        )

        # Generate tasks based on complexity
        if complexity == ComplexityLevel.MINIMAL:
            tasks = TaskFactory.create_batch(task_count)
        elif complexity == ComplexityLevel.LOW:
            tasks = TaskFactory.create_batch(
                task_count,
                status_distribution={
                    TaskStatus.AVAILABLE: 0.7,
                    TaskStatus.IN_PROGRESS: 0.2,
                    TaskStatus.DONE: 0.1,
                }
            )
        elif complexity == ComplexityLevel.MEDIUM:
            # Mix of chains and parallel tasks
            chain = TaskFactory.create_dependency_chain(task_count // 3)
            parallel = TaskFactory.create_parallel_group(
                task_count // 3,
                parent_id=chain[-1].id if chain else None
            )
            remaining = TaskFactory.create_batch(task_count - len(chain) - len(parallel))
            tasks = chain + parallel + remaining
        elif complexity in (ComplexityLevel.HIGH, ComplexityLevel.EXTREME):
            # Complex dependency graph
            tasks = cls._generate_complex_tasks(task_count, complexity)
        else:
            tasks = TaskFactory.create_batch(task_count)

        for task in tasks:
            state.add_task(task)

        # Add discoveries
        discovery_count = task_count // 5 if complexity != ComplexityLevel.MINIMAL else 0
        state.discoveries.extend(DiscoveryFactory.create_batch(discovery_count))

        return state

    @classmethod
    def _generate_complex_tasks(
        cls,
        count: int,
        complexity: ComplexityLevel
    ) -> List[Task]:
        """Generate tasks with complex dependency structure."""
        tasks = []

        # Create layers of tasks
        layer_count = 4 if complexity == ComplexityLevel.HIGH else 6
        tasks_per_layer = count // layer_count

        prev_layer_ids: List[str] = []

        for layer in range(layer_count):
            layer_tasks = []
            for i in range(tasks_per_layer):
                # Each task depends on some tasks from the previous layer
                deps = []
                if prev_layer_ids:
                    dep_count = min(len(prev_layer_ids), random.randint(1, 3))
                    deps = random.sample(prev_layer_ids, dep_count)

                task = TaskFactory.create(
                    description=f"Layer {layer + 1} Task {i + 1}",
                    priority=layer + 1,
                    dependencies=deps
                )
                layer_tasks.append(task)

            tasks.extend(layer_tasks)
            prev_layer_ids = [t.id for t in layer_tasks]

        return tasks


# =============================================================================
# Scenario Generator
# =============================================================================

@dataclass
class TestScenario:
    """A complete test scenario with all required data."""
    name: str
    state: CoordinationState
    expected_outcomes: Dict[str, Any]
    setup_actions: List[Callable] = field(default_factory=list)
    verification_actions: List[Callable] = field(default_factory=list)


class ScenarioGenerator:
    """
    Generates complete test scenarios for various use cases.

    Each scenario includes:
    - Initial state
    - Expected outcomes
    - Setup and verification actions
    """

    @classmethod
    def generate_simple_workflow(cls) -> TestScenario:
        """Generate a simple linear workflow scenario."""
        state = CoordinationStateFactory.create(
            goal="Simple workflow test",
            task_count=5,
            complexity=ComplexityLevel.LOW
        )

        return TestScenario(
            name="simple_workflow",
            state=state,
            expected_outcomes={
                "total_tasks": 5,
                "can_complete": True,
            }
        )

    @classmethod
    def generate_dependency_resolution(cls) -> TestScenario:
        """Generate a scenario testing dependency resolution."""
        state = CoordinationState(goal="Test dependency resolution")

        # Create tasks with dependencies
        task_a = TaskFactory.create(description="Task A - No deps")
        task_b = TaskFactory.create(
            description="Task B - Depends on A",
            dependencies=[task_a.id]
        )
        task_c = TaskFactory.create(
            description="Task C - Depends on A and B",
            dependencies=[task_a.id, task_b.id]
        )

        state.add_task(task_a)
        state.add_task(task_b)
        state.add_task(task_c)

        return TestScenario(
            name="dependency_resolution",
            state=state,
            expected_outcomes={
                "initially_available": 1,  # Only A
                "after_a_complete": 1,  # Only B
                "after_b_complete": 1,  # Only C
            }
        )

    @classmethod
    def generate_parallel_execution(cls) -> TestScenario:
        """Generate a scenario testing parallel task execution."""
        state = CoordinationState(goal="Test parallel execution")

        # Create independent tasks
        for i in range(5):
            task = TaskFactory.create(
                description=f"Independent task {i + 1}",
                priority=i + 1
            )
            state.add_task(task)

        return TestScenario(
            name="parallel_execution",
            state=state,
            expected_outcomes={
                "all_available": True,
                "can_run_parallel": 5,
            }
        )

    @classmethod
    def generate_failure_recovery(cls) -> TestScenario:
        """Generate a scenario testing failure and recovery."""
        state = CoordinationState(goal="Test failure recovery")

        # Create task that will fail
        task = TaskFactory.create(
            description="Task that may fail",
            max_attempts=3
        )
        state.add_task(task)

        return TestScenario(
            name="failure_recovery",
            state=state,
            expected_outcomes={
                "max_retries": 3,
                "can_recover": True,
            }
        )

    @classmethod
    def generate_complex_dag(cls) -> TestScenario:
        """Generate a complex DAG workflow scenario."""
        state = CoordinationStateFactory.create(
            goal="Complex DAG workflow",
            task_count=20,
            complexity=ComplexityLevel.HIGH
        )

        return TestScenario(
            name="complex_dag",
            state=state,
            expected_outcomes={
                "has_dependencies": True,
                "is_acyclic": True,
            }
        )

    @classmethod
    def generate_discovery_sharing(cls) -> TestScenario:
        """Generate a scenario testing discovery sharing."""
        state = CoordinationState(goal="Test discovery sharing")

        # Add tasks that will create discoveries
        for i in range(3):
            task = TaskFactory.create(
                description=f"Research task {i + 1}",
                tags=FakeDataProvider.tags(2)
            )
            state.add_task(task)

        # Add initial discoveries
        state.discoveries.extend(DiscoveryFactory.create_batch(3))

        return TestScenario(
            name="discovery_sharing",
            state=state,
            expected_outcomes={
                "initial_discoveries": 3,
                "can_add_more": True,
            }
        )


# =============================================================================
# Pytest Fixtures
# =============================================================================

import pytest


@pytest.fixture
def task_factory():
    """Provide TaskFactory for tests."""
    return TaskFactory


@pytest.fixture
def result_factory():
    """Provide TaskResultFactory for tests."""
    return TaskResultFactory


@pytest.fixture
def discovery_factory():
    """Provide DiscoveryFactory for tests."""
    return DiscoveryFactory


@pytest.fixture
def agent_factory():
    """Provide AgentFactory for tests."""
    return AgentFactory


@pytest.fixture
def state_factory():
    """Provide CoordinationStateFactory for tests."""
    return CoordinationStateFactory


@pytest.fixture
def scenario_generator():
    """Provide ScenarioGenerator for tests."""
    return ScenarioGenerator


@pytest.fixture
def simple_scenario():
    """Provide a simple workflow scenario."""
    return ScenarioGenerator.generate_simple_workflow()


@pytest.fixture
def complex_scenario():
    """Provide a complex DAG scenario."""
    return ScenarioGenerator.generate_complex_dag()


# =============================================================================
# Test the generators themselves
# =============================================================================

class TestDataGenerators:
    """Tests for the data generators."""

    def test_task_factory_create(self):
        """Generator test: TaskFactory creates valid tasks."""
        task = TaskFactory.create()

        assert task.id.startswith("task-")
        assert len(task.description) > 0
        assert 1 <= task.priority <= 10

    def test_task_factory_batch(self):
        """Generator test: TaskFactory creates batches correctly."""
        tasks = TaskFactory.create_batch(10)

        assert len(tasks) == 10
        assert len(set(t.id for t in tasks)) == 10  # All unique IDs

    def test_task_factory_chain(self):
        """Generator test: TaskFactory creates dependency chains."""
        chain = TaskFactory.create_dependency_chain(5)

        assert len(chain) == 5
        for i in range(1, len(chain)):
            assert chain[i - 1].id in chain[i].dependencies

    def test_result_factory_create(self):
        """Generator test: TaskResultFactory creates valid results."""
        result = TaskResultFactory.create()

        assert len(result.output) > 0
        assert isinstance(result.files_modified, list)

    def test_discovery_factory_create(self):
        """Generator test: DiscoveryFactory creates valid discoveries."""
        discovery = DiscoveryFactory.create()

        assert discovery.id.startswith("disc-")
        assert len(discovery.content) > 0

    def test_state_factory_complexity_levels(self):
        """Generator test: CoordinationStateFactory respects complexity."""
        minimal = CoordinationStateFactory.create(
            task_count=5,
            complexity=ComplexityLevel.MINIMAL
        )
        high = CoordinationStateFactory.create(
            task_count=20,
            complexity=ComplexityLevel.HIGH
        )

        assert len(minimal.tasks) == 5
        assert len(high.tasks) == 20
        # High complexity should have more interconnections
        high_deps = sum(len(t.dependencies) for t in high.tasks)
        assert high_deps > 0

    def test_scenario_generator_simple(self):
        """Generator test: ScenarioGenerator creates valid scenarios."""
        scenario = ScenarioGenerator.generate_simple_workflow()

        assert scenario.name == "simple_workflow"
        assert scenario.state is not None
        assert len(scenario.state.tasks) > 0

    def test_scenario_generator_dependency(self):
        """Generator test: Dependency scenario has correct structure."""
        scenario = ScenarioGenerator.generate_dependency_resolution()

        assert len(scenario.state.tasks) == 3
        available = scenario.state.get_available_tasks()
        assert len(available) == 1  # Only task A
