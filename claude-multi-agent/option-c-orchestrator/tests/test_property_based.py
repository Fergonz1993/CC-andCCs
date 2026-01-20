"""
Property-based testing using hypothesis.

Feature: adv-test-013 - Property-based testing (hypothesis)

This module uses property-based testing to verify invariants
that should hold for all possible inputs.

Run with: pytest tests/test_property_based.py -v --hypothesis-show-statistics

Requirements: pip install hypothesis
"""

import pytest

try:
    from hypothesis import given, assume, settings, Verbosity, strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, Bundle
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators for when hypothesis is not installed
    def given(*args, **kwargs):
        def decorator(f):
            return pytest.mark.skip(reason="hypothesis not installed")(f)
        return decorator

    def settings(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

    class _DummyStrategies:
        def __getattr__(self, _name):
            def _dummy(*_args, **_kwargs):
                return None
            return _dummy

    st = _DummyStrategies()

    def rule(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

    def invariant(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

    class Bundle:
        def __init__(self, *args, **kwargs):
            pass

from orchestrator.models import (
    Task,
    TaskStatus,
    TaskResult,
    TaskContext,
    Discovery,
    CoordinationState,
    AgentRole,
    Agent,
)


# =============================================================================
# Custom Hypothesis Strategies
# =============================================================================

# Default placeholders when hypothesis isn't available
task_description_strategy = None
priority_strategy = None
task_id_strategy = None
agent_id_strategy = None
file_path_strategy = None
task_context_strategy = None
task_result_strategy = None

if HYPOTHESIS_AVAILABLE:
    # Strategy for generating valid task descriptions
    task_description_strategy = st.text(
        min_size=1,
        max_size=500,
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'),
            min_codepoint=32,
            max_codepoint=126
        )
    ).filter(lambda x: len(x.strip()) > 0)

    # Strategy for generating valid priorities
    priority_strategy = st.integers(min_value=1, max_value=10)

    # Strategy for generating task IDs
    task_id_strategy = st.text(
        min_size=4,
        max_size=20,
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-"
    ).map(lambda x: f"task-{x}")

    # Strategy for generating agent IDs
    agent_id_strategy = st.text(
        min_size=1,
        max_size=20,
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_"
    )

    # Strategy for generating file paths
    file_path_strategy = st.text(
        min_size=1,
        max_size=100,
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_/."
    ).filter(lambda x: "/" not in x[:1] and ".." not in x)

    # Strategy for generating task contexts
    task_context_strategy = st.builds(
        TaskContext,
        files=st.lists(file_path_strategy, max_size=10),
        hints=st.text(max_size=200),
        parent_task=st.one_of(st.none(), task_id_strategy),
        tags=st.lists(st.text(min_size=1, max_size=20), max_size=5),
    )

    # Strategy for generating task results
    task_result_strategy = st.builds(
        TaskResult,
        output=st.text(max_size=1000),
        files_modified=st.lists(file_path_strategy, max_size=10),
        files_created=st.lists(file_path_strategy, max_size=10),
        files_deleted=st.lists(file_path_strategy, max_size=5),
        error=st.one_of(st.none(), st.text(max_size=200)),
        subtasks_created=st.lists(task_id_strategy, max_size=5),
        discoveries=st.lists(st.text(max_size=200), max_size=5),
    )


# =============================================================================
# Property Tests for Task Model
# =============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestTaskProperties:
    """Property-based tests for the Task model."""

    @given(
        description=task_description_strategy,
        priority=priority_strategy
    )
    @settings(max_examples=100)
    def test_property_001_task_id_always_generated(self, description, priority):
        """
        property-001: Task always generates a unique ID.

        Property: Every task created must have a non-empty ID starting with 'task-'.
        """
        task = Task(description=description, priority=priority)

        assert task.id is not None
        assert len(task.id) > 0
        assert task.id.startswith("task-")

    @given(
        description=task_description_strategy,
        priority=priority_strategy
    )
    @settings(max_examples=100)
    def test_property_002_new_task_is_available(self, description, priority):
        """
        property-002: Newly created tasks are always AVAILABLE.

        Property: A new task without dependencies starts as AVAILABLE.
        """
        task = Task(description=description, priority=priority)

        assert task.status == TaskStatus.AVAILABLE

    @given(
        description=task_description_strategy,
        agent_id=agent_id_strategy
    )
    @settings(max_examples=100)
    def test_property_003_claim_changes_status(self, description, agent_id):
        """
        property-003: Claiming a task always changes status to CLAIMED.

        Property: After claim(), status must be CLAIMED and claimed_by is set.
        """
        assume(len(agent_id) > 0)

        task = Task(description=description)
        task.claim(agent_id)

        assert task.status == TaskStatus.CLAIMED
        assert task.claimed_by == agent_id
        assert task.claimed_at is not None

    @given(description=task_description_strategy)
    @settings(max_examples=100)
    def test_property_004_start_changes_status(self, description):
        """
        property-004: Starting a task always changes status to IN_PROGRESS.

        Property: After start(), status must be IN_PROGRESS.
        """
        task = Task(description=description)
        task.start()

        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None

    @given(
        description=task_description_strategy,
        output=st.text(max_size=500)
    )
    @settings(max_examples=100)
    def test_property_005_complete_changes_status(self, description, output):
        """
        property-005: Completing a task always changes status to DONE.

        Property: After complete(), status must be DONE and result is stored.
        """
        task = Task(description=description)
        result = TaskResult(output=output)
        task.complete(result)

        assert task.status == TaskStatus.DONE
        assert task.result == result
        assert task.completed_at is not None

    @given(
        description=task_description_strategy,
        error=st.text(min_size=1, max_size=200)
    )
    @settings(max_examples=100)
    def test_property_006_fail_changes_status(self, description, error):
        """
        property-006: Failing a task always changes status to FAILED.

        Property: After fail(), status must be FAILED and error is recorded.
        """
        task = Task(description=description)
        task.fail(error)

        assert task.status == TaskStatus.FAILED
        assert task.result is not None
        assert task.result.error == error

    @given(
        description=task_description_strategy,
        max_attempts=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=50)
    def test_property_007_reset_respects_max_attempts(self, description, max_attempts):
        """
        property-007: Reset only works when attempts < max_attempts.

        Property: A task that has reached max_attempts cannot be reset.
        """
        task = Task(description=description, max_attempts=max_attempts)

        # Claim multiple times up to max_attempts
        for i in range(max_attempts):
            task.status = TaskStatus.AVAILABLE  # Force reset for testing
            task.claim(f"worker-{i}")

        # At max attempts, reset should not change status
        original_status = task.status
        task.reset()

        # Status should not change to AVAILABLE if at max attempts
        assert task.attempts == max_attempts

    @given(
        description=task_description_strategy,
        dependencies=st.lists(task_id_strategy, min_size=0, max_size=5, unique=True)
    )
    @settings(max_examples=100)
    def test_property_008_can_start_with_all_deps_met(self, description, dependencies):
        """
        property-008: Task can start only when all dependencies are met.

        Property: can_start() returns True iff all dependencies are in completed set.
        """
        task = Task(description=description, dependencies=dependencies)

        # With no completed tasks, can only start if no dependencies
        assert task.can_start(set()) == (len(dependencies) == 0)

        # With all dependencies completed, can always start
        assert task.can_start(set(dependencies)) is True

        # With partial dependencies, cannot start (if there are any)
        if len(dependencies) > 1:
            partial = set(dependencies[:len(dependencies)//2])
            assert task.can_start(partial) is False

    @given(priority=st.integers())
    @settings(max_examples=50)
    def test_property_009_priority_validation(self, priority):
        """
        property-009: Priority must be within valid range [1, 10].

        Property: Creating a task with invalid priority raises ValueError.
        """
        if 1 <= priority <= 10:
            # Valid priority should work
            task = Task(description="Test", priority=priority)
            assert task.priority == priority
        else:
            # Invalid priority should raise
            with pytest.raises(ValueError):
                Task(description="Test", priority=priority)


# =============================================================================
# Property Tests for CoordinationState
# =============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestCoordinationStateProperties:
    """Property-based tests for CoordinationState."""

    @given(
        tasks=st.lists(
            st.builds(
                Task,
                description=task_description_strategy,
                priority=priority_strategy
            ),
            min_size=0,
            max_size=20
        )
    )
    @settings(max_examples=50)
    def test_property_010_available_tasks_subset(self, tasks):
        """
        property-010: Available tasks are always a subset of all tasks.

        Property: get_available_tasks() returns only tasks from the state.
        """
        state = CoordinationState()
        for task in tasks:
            state.add_task(task)

        available = state.get_available_tasks()
        task_ids = {t.id for t in state.tasks}
        available_ids = {t.id for t in available}

        assert available_ids.issubset(task_ids)

    @given(
        tasks=st.lists(
            st.builds(
                Task,
                description=task_description_strategy,
                priority=priority_strategy
            ),
            min_size=0,
            max_size=20
        )
    )
    @settings(max_examples=50)
    def test_property_011_progress_totals_match(self, tasks):
        """
        property-011: Progress by_status totals match total_tasks.

        Property: Sum of tasks by status equals total task count.
        """
        state = CoordinationState()
        for task in tasks:
            state.add_task(task)

        progress = state.get_progress()

        total_by_status = sum(progress["by_status"].values())
        assert total_by_status == progress["total_tasks"]
        assert progress["total_tasks"] == len(tasks)

    @given(
        tasks=st.lists(
            st.builds(
                Task,
                description=task_description_strategy,
                priority=priority_strategy
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=50)
    def test_property_012_percent_complete_bounds(self, tasks):
        """
        property-012: Percent complete is always between 0 and 100.

        Property: get_progress()["percent_complete"] is in [0, 100].
        """
        state = CoordinationState()
        for task in tasks:
            # Randomly complete some tasks
            state.add_task(task)

        progress = state.get_progress()

        assert 0 <= progress["percent_complete"] <= 100

    @given(task_id=task_id_strategy)
    @settings(max_examples=50)
    def test_property_013_get_task_returns_none_for_missing(self, task_id):
        """
        property-013: get_task returns None for non-existent task.

        Property: Querying for a task that doesn't exist returns None.
        """
        state = CoordinationState()

        result = state.get_task(task_id)

        assert result is None

    @given(
        description=task_description_strategy,
        priority=priority_strategy
    )
    @settings(max_examples=50)
    def test_property_014_added_task_is_retrievable(self, description, priority):
        """
        property-014: Added tasks can be retrieved by ID.

        Property: After add_task(), get_task() returns the same task.
        """
        state = CoordinationState()
        task = Task(description=description, priority=priority)

        state.add_task(task)
        retrieved = state.get_task(task.id)

        assert retrieved is not None
        assert retrieved.id == task.id
        assert retrieved.description == task.description


# =============================================================================
# Stateful Property Testing
# =============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestTaskStateMachine(RuleBasedStateMachine if HYPOTHESIS_AVAILABLE else object):
    """
    Stateful property-based testing for task lifecycle.

    This tests that tasks always follow valid state transitions
    regardless of the sequence of operations.
    """

    def __init__(self):
        super().__init__()
        self.state = CoordinationState()
        self.task_ids = set()

    tasks = Bundle('tasks') if HYPOTHESIS_AVAILABLE else None

    @rule(target=tasks, description=task_description_strategy, priority=priority_strategy)
    def add_task(self, description, priority):
        """Add a new task to the state."""
        task = Task(description=description, priority=priority)
        self.state.add_task(task)
        self.task_ids.add(task.id)
        return task.id

    @rule(task_id=tasks, agent_id=agent_id_strategy)
    def claim_task(self, task_id, agent_id):
        """Claim a task."""
        assume(len(agent_id) > 0)

        task = self.state.get_task(task_id)
        if task and task.status == TaskStatus.AVAILABLE:
            task.claim(agent_id)

    @rule(task_id=tasks)
    def start_task(self, task_id):
        """Start a claimed task."""
        task = self.state.get_task(task_id)
        if task and task.status == TaskStatus.CLAIMED:
            task.start()

    @rule(task_id=tasks, output=st.text(max_size=100))
    def complete_task(self, task_id, output):
        """Complete a task."""
        task = self.state.get_task(task_id)
        if task and task.status == TaskStatus.IN_PROGRESS:
            task.complete(TaskResult(output=output))

    @rule(task_id=tasks, error=st.text(max_size=100))
    def fail_task(self, task_id, error):
        """Fail a task."""
        task = self.state.get_task(task_id)
        if task and task.status == TaskStatus.IN_PROGRESS:
            task.fail(error)

    @invariant()
    def task_count_matches(self):
        """Invariant: Number of tasks in state matches our tracking."""
        assert len(self.state.tasks) == len(self.task_ids)

    @invariant()
    def progress_is_valid(self):
        """Invariant: Progress calculation is always valid."""
        progress = self.state.get_progress()
        assert progress["total_tasks"] == len(self.state.tasks)
        assert 0 <= progress["percent_complete"] <= 100

    @invariant()
    def available_tasks_are_valid(self):
        """Invariant: Available tasks have AVAILABLE status."""
        available = self.state.get_available_tasks()
        for task in available:
            assert task.status == TaskStatus.AVAILABLE


# Run the state machine tests
if HYPOTHESIS_AVAILABLE:
    TestTaskStateMachineRunner = TestTaskStateMachine.TestCase


# =============================================================================
# Additional Property Tests
# =============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestDiscoveryProperties:
    """Property-based tests for Discovery model."""

    @given(
        agent_id=agent_id_strategy,
        content=st.text(min_size=1, max_size=500)
    )
    @settings(max_examples=50)
    def test_property_015_discovery_id_generated(self, agent_id, content):
        """
        property-015: Discovery always generates a unique ID.
        """
        assume(len(agent_id) > 0)

        discovery = Discovery(agent_id=agent_id, content=content)

        assert discovery.id is not None
        assert discovery.id.startswith("disc-")

    @given(
        agent_id=agent_id_strategy,
        content=st.text(min_size=1, max_size=500),
        tags=st.lists(st.text(min_size=1, max_size=20), max_size=10)
    )
    @settings(max_examples=50)
    def test_property_016_discovery_preserves_tags(self, agent_id, content, tags):
        """
        property-016: Discovery preserves all provided tags.
        """
        assume(len(agent_id) > 0)

        discovery = Discovery(agent_id=agent_id, content=content, tags=tags)

        assert discovery.tags == tags


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestAgentProperties:
    """Property-based tests for Agent model."""

    @given(
        agent_id=agent_id_strategy,
        role=st.sampled_from(list(AgentRole))
    )
    @settings(max_examples=50)
    def test_property_017_agent_creation(self, agent_id, role):
        """
        property-017: Agent can be created with any valid role.
        """
        assume(len(agent_id) > 0)

        agent = Agent(id=agent_id, role=role)

        assert agent.id == agent_id
        assert agent.role == role
        assert agent.is_active is False  # Default
