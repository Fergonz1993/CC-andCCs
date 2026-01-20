"""
Tests for orchestrator models.

Run with: pytest tests/test_models.py -v
"""

import pytest
from datetime import datetime
from orchestrator.models import (
    Task,
    TaskStatus,
    TaskResult,
    TaskContext,
    Agent,
    AgentRole,
    AgentMetrics,
    Discovery,
    CoordinationState,
)


class TestTask:
    """Tests for the Task model."""

    def test_task_generates_id_with_prefix(self):
        """opt-c-model-001: Task model generates ID with task- prefix."""
        task = Task(description="Test task")
        assert task.id.startswith("task-")

    def test_task_id_has_hex_suffix(self):
        """opt-c-model-002: Task model ID has 8 hex chars after prefix."""
        task = Task(description="Test task")
        # ID format: task-{8 hex chars}
        parts = task.id.split("-")
        assert len(parts) == 2
        assert len(parts[1]) == 8

    def test_task_default_status_is_available(self):
        """opt-c-model-003: Task default status is AVAILABLE."""
        task = Task(description="Test task")
        assert task.status == TaskStatus.AVAILABLE

    def test_task_default_priority_is_5(self):
        """opt-c-model-004: Task default priority is 5."""
        task = Task(description="Test task")
        assert task.priority == 5

    def test_task_priority_validates_range(self):
        """opt-c-model-005: Task priority validates range 1-10."""
        # Valid priorities
        Task(description="Test", priority=1)
        Task(description="Test", priority=10)

        # Invalid priorities
        with pytest.raises(ValueError):
            Task(description="Test", priority=0)
        with pytest.raises(ValueError):
            Task(description="Test", priority=11)

    def test_task_claim_sets_status(self):
        """opt-c-model-006: Task.claim() sets status to CLAIMED."""
        task = Task(description="Test")
        task.claim("agent-1")
        assert task.status == TaskStatus.CLAIMED

    def test_task_claim_sets_claimed_by(self):
        """opt-c-model-007: Task.claim() sets claimed_by."""
        task = Task(description="Test")
        task.claim("agent-1")
        assert task.claimed_by == "agent-1"

    def test_task_claim_sets_timestamp(self):
        """opt-c-model-008: Task.claim() sets claimed_at timestamp."""
        task = Task(description="Test")
        task.claim("agent-1")
        assert task.claimed_at is not None
        assert isinstance(task.claimed_at, datetime)

    def test_task_claim_increments_attempts(self):
        """opt-c-model-009: Task.claim() increments attempts."""
        task = Task(description="Test")
        assert task.attempts == 0
        task.claim("agent-1")
        assert task.attempts == 1

    def test_task_start_sets_status(self):
        """opt-c-model-010: Task.start() sets status to IN_PROGRESS."""
        task = Task(description="Test")
        task.start()
        assert task.status == TaskStatus.IN_PROGRESS

    def test_task_start_sets_timestamp(self):
        """opt-c-model-011: Task.start() sets started_at timestamp."""
        task = Task(description="Test")
        task.start()
        assert task.started_at is not None

    def test_task_complete_sets_status(self):
        """opt-c-model-012: Task.complete() sets status to DONE."""
        task = Task(description="Test")
        result = TaskResult(output="Completed successfully")
        task.complete(result)
        assert task.status == TaskStatus.DONE

    def test_task_complete_sets_timestamp(self):
        """opt-c-model-013: Task.complete() sets completed_at timestamp."""
        task = Task(description="Test")
        result = TaskResult(output="Done")
        task.complete(result)
        assert task.completed_at is not None

    def test_task_complete_stores_result(self):
        """opt-c-model-014: Task.complete() stores result."""
        task = Task(description="Test")
        result = TaskResult(output="Done", files_modified=["test.py"])
        task.complete(result)
        assert task.result == result
        assert task.result.files_modified == ["test.py"]

    def test_task_fail_sets_status(self):
        """opt-c-model-015: Task.fail() sets status to FAILED."""
        task = Task(description="Test")
        task.fail("Something went wrong")
        assert task.status == TaskStatus.FAILED

    def test_task_fail_creates_result_with_error(self):
        """opt-c-model-016: Task.fail() creates TaskResult with error."""
        task = Task(description="Test")
        task.fail("Error message")
        assert task.result is not None
        assert task.result.error == "Error message"

    def test_task_reset_sets_available(self):
        """opt-c-model-017: Task.reset() sets status back to AVAILABLE."""
        task = Task(description="Test")
        task.claim("agent-1")
        task.reset()
        assert task.status == TaskStatus.AVAILABLE

    def test_task_reset_respects_max_attempts(self):
        """opt-c-model-018: Task.reset() only works if attempts < max_attempts."""
        task = Task(description="Test", max_attempts=1)
        task.claim("agent-1")  # attempts = 1
        task.reset()  # Should not reset because attempts == max_attempts
        assert task.status == TaskStatus.CLAIMED

    def test_task_can_start_no_dependencies(self):
        """opt-c-model-019: Task.can_start() returns True when no dependencies."""
        task = Task(description="Test")
        assert task.can_start(set()) is True

    def test_task_can_start_unmet_dependencies(self):
        """opt-c-model-020: Task.can_start() returns False when dependencies unmet."""
        task = Task(description="Test", dependencies=["task-1"])
        assert task.can_start(set()) is False

    def test_task_can_start_met_dependencies(self):
        """opt-c-model-021: Task.can_start() returns True when dependencies met."""
        task = Task(description="Test", dependencies=["task-1"])
        assert task.can_start({"task-1"}) is True


class TestEnums:
    """Tests for enum values."""

    def test_task_status_values(self):
        """opt-c-model-022: TaskStatus enum has all required values."""
        expected = {"pending", "available", "claimed", "in_progress", "done", "failed", "blocked"}
        actual = {s.value for s in TaskStatus}
        assert expected == actual

    def test_agent_role_values(self):
        """opt-c-model-023: AgentRole enum has LEADER, WORKER, SPECIALIST."""
        expected = {"leader", "worker", "specialist"}
        actual = {r.value for r in AgentRole}
        assert expected == actual


class TestTaskContext:
    """Tests for TaskContext model."""

    def test_task_context_has_all_fields(self):
        """opt-c-model-024: TaskContext has required fields."""
        ctx = TaskContext()
        assert hasattr(ctx, "files")
        assert hasattr(ctx, "hints")
        assert hasattr(ctx, "parent_task")
        assert hasattr(ctx, "tags")
        assert hasattr(ctx, "metadata")


class TestTaskResult:
    """Tests for TaskResult model."""

    def test_task_result_has_all_fields(self):
        """opt-c-model-025: TaskResult has required fields."""
        result = TaskResult(output="Done")
        assert hasattr(result, "output")
        assert hasattr(result, "files_modified")
        assert hasattr(result, "files_created")
        assert hasattr(result, "files_deleted")
        assert hasattr(result, "error")
        assert hasattr(result, "subtasks_created")
        assert hasattr(result, "discoveries")


class TestAgent:
    """Tests for Agent model."""

    def test_agent_has_all_fields(self):
        """opt-c-model-026: Agent model has all required fields."""
        agent = Agent(id="test", role=AgentRole.WORKER)
        assert hasattr(agent, "id")
        assert hasattr(agent, "role")
        assert hasattr(agent, "is_active")
        assert hasattr(agent, "current_task")
        assert hasattr(agent, "pid")
        assert hasattr(agent, "last_heartbeat")
        assert hasattr(agent, "metrics")
        assert hasattr(agent, "working_directory")
        assert hasattr(agent, "model")
        assert hasattr(agent, "max_concurrent_tasks")


class TestAgentMetrics:
    """Tests for AgentMetrics model."""

    def test_agent_metrics_has_all_fields(self):
        """opt-c-model-027: AgentMetrics has required fields."""
        metrics = AgentMetrics()
        assert hasattr(metrics, "tasks_completed")
        assert hasattr(metrics, "tasks_failed")
        assert hasattr(metrics, "total_time_working")
        assert hasattr(metrics, "avg_task_time")
        assert hasattr(metrics, "discoveries_made")


class TestDiscovery:
    """Tests for Discovery model."""

    def test_discovery_has_all_fields(self):
        """opt-c-model-028: Discovery model has required fields."""
        disc = Discovery(agent_id="agent-1", content="Found something")
        assert hasattr(disc, "id")
        assert hasattr(disc, "agent_id")
        assert hasattr(disc, "content")
        assert hasattr(disc, "tags")
        assert hasattr(disc, "related_task")
        assert hasattr(disc, "created_at")


class TestCoordinationState:
    """Tests for CoordinationState model."""

    def test_get_available_tasks(self):
        """opt-c-model-029: CoordinationState.get_available_tasks() filters correctly."""
        state = CoordinationState()

        # Add tasks with different statuses
        available_task = Task(description="Available")
        claimed_task = Task(description="Claimed")
        claimed_task.claim("agent-1")
        done_task = Task(description="Done")
        done_task.complete(TaskResult(output="Done"))

        state.tasks = [available_task, claimed_task, done_task]

        available = state.get_available_tasks()
        assert len(available) == 1
        assert available[0].description == "Available"

    def test_get_progress(self):
        """opt-c-model-030: CoordinationState.get_progress() returns correct stats."""
        state = CoordinationState()

        t1 = Task(description="T1")
        t2 = Task(description="T2")
        t2.complete(TaskResult(output="Done"))

        state.tasks = [t1, t2]

        progress = state.get_progress()
        assert progress["total_tasks"] == 2
        assert progress["by_status"]["available"] == 1
        assert progress["by_status"]["done"] == 1
        assert progress["percent_complete"] == 50.0
