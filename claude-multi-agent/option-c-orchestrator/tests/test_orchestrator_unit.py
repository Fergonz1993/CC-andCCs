"""
Unit tests for Option C orchestrator.py (test-003)

Tests the core orchestrator functionality including:
- Task management
- Agent coordination
- State persistence
- Error handling
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from orchestrator.orchestrator import (
    Orchestrator,
    TaskStatus,
    AgentStatus,
    OrchestrationError,
    TaskSchemaError,
)
from orchestrator.models import Task, Agent


class TestOrchestratorInitialization:
    """Test orchestrator initialization and setup."""

    def test_orchestrator_creates_coordination_dir(self):
        """Test that orchestrator creates coordination directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_dir = Path(tmpdir) / ".coordination"
            orchestrator = Orchestrator(coordination_dir=str(coord_dir))

            assert coord_dir.exists()
            assert (coord_dir / "tasks.json").exists()

    def test_orchestrator_loads_existing_state(self):
        """Test that orchestrator loads existing tasks and agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_dir = Path(tmpdir) / ".coordination"
            coord_dir.mkdir()

            # Create existing tasks file
            tasks_data = {
                "tasks": [
                    {
                        "id": "task-1",
                        "description": "Test task",
                        "status": "available",
                        "priority": 1,
                        "created_at": datetime.now().isoformat(),
                    }
                ]
            }
            (coord_dir / "tasks.json").write_text(json.dumps(tasks_data))

            orchestrator = Orchestrator(coordination_dir=str(coord_dir))
            tasks = orchestrator.get_all_tasks()

            assert len(tasks) == 1
            assert tasks[0]["id"] == "task-1"

    def test_orchestrator_with_custom_goal(self):
        """Test orchestrator initialization with custom goal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = Orchestrator(
                coordination_dir=tmpdir,
                goal="Build a web application"
            )

            plan_file = Path(tmpdir) / "master-plan.md"
            assert plan_file.exists()
            content = plan_file.read_text()
            assert "Build a web application" in content

    def test_hybrid_orchestrator_mode_selection(self):
        """Test hybrid selection: coordination_dir -> file mode, working_directory -> async mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_orch = Orchestrator(coordination_dir=tmpdir)
            assert not hasattr(file_orch._impl, "state")
            assert (Path(tmpdir) / "tasks.json").exists()

            async_orch = Orchestrator(working_directory=tmpdir, verbose=False)
            assert hasattr(async_orch._impl, "state")


class TestTaskManagement:
    """Test task creation, retrieval, and updates."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Orchestrator(coordination_dir=tmpdir)
            yield orch

    def test_add_task(self, orchestrator):
        """Test adding a new task."""
        task = orchestrator.add_task(
            description="Implement login feature",
            priority=1,
            dependencies=[]
        )

        assert task["id"] is not None
        assert task["description"] == "Implement login feature"
        assert task["priority"] == 1
        assert task["status"] == "available"

    def test_add_task_with_dependencies(self, orchestrator):
        """Test adding a task with dependencies."""
        task1 = orchestrator.add_task("Setup database", priority=1)
        task2 = orchestrator.add_task(
            "Create user table",
            priority=2,
            dependencies=[task1["id"]]
        )

        assert task2["dependencies"] == [task1["id"]]

    def test_get_task(self, orchestrator):
        """Test retrieving a specific task."""
        task = orchestrator.add_task("Test task", priority=1)
        retrieved = orchestrator.get_task(task["id"])

        assert retrieved is not None
        assert retrieved["id"] == task["id"]
        assert retrieved["description"] == "Test task"

    def test_get_nonexistent_task(self, orchestrator):
        """Test retrieving a task that doesn't exist."""
        task = orchestrator.get_task("nonexistent-id")
        assert task is None

    def test_get_all_tasks(self, orchestrator):
        """Test retrieving all tasks."""
        orchestrator.add_task("Task 1", priority=1)
        orchestrator.add_task("Task 2", priority=2)
        orchestrator.add_task("Task 3", priority=3)

        tasks = orchestrator.get_all_tasks()
        assert len(tasks) == 3

    def test_get_available_tasks(self, orchestrator):
        """Test retrieving only available tasks."""
        task1 = orchestrator.add_task("Task 1", priority=1)
        task2 = orchestrator.add_task("Task 2", priority=2)

        # Claim one task
        orchestrator.claim_task(task1["id"], "agent-1")

        available = orchestrator.get_available_tasks()
        assert len(available) == 1
        assert available[0]["id"] == task2["id"]

    def test_update_task_status(self, orchestrator):
        """Test updating task status."""
        task = orchestrator.add_task("Test task", priority=1)
        orchestrator.claim_task(task["id"], "agent-1")

        updated = orchestrator.get_task(task["id"])
        assert updated["status"] == "claimed"
        assert updated["assigned_to"] == "agent-1"

    def test_complete_task(self, orchestrator):
        """Test marking a task as complete."""
        task = orchestrator.add_task("Test task", priority=1)
        orchestrator.claim_task(task["id"], "agent-1")
        orchestrator.complete_task(task["id"], result="Task completed successfully")

        completed = orchestrator.get_task(task["id"])
        assert completed["status"] == "done"
        assert completed["result"] == "Task completed successfully"

    def test_fail_task(self, orchestrator):
        """Test marking a task as failed."""
        task = orchestrator.add_task("Test task", priority=1)
        orchestrator.claim_task(task["id"], "agent-1")
        orchestrator.fail_task(task["id"], error="Task failed due to error")

        failed = orchestrator.get_task(task["id"])
        assert failed["status"] == "failed"
        assert "error" in failed


class TestAgentManagement:
    """Test agent registration and management."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Orchestrator(coordination_dir=tmpdir)
            yield orch

    def test_register_agent(self, orchestrator):
        """Test registering a new agent."""
        agent = orchestrator.register_agent("agent-1", capabilities=["python", "testing"])

        assert agent["id"] == "agent-1"
        assert agent["status"] == "active"
        assert agent["capabilities"] == ["python", "testing"]

    def test_get_agent(self, orchestrator):
        """Test retrieving an agent."""
        orchestrator.register_agent("agent-1")
        agent = orchestrator.get_agent("agent-1")

        assert agent is not None
        assert agent["id"] == "agent-1"

    def test_get_all_agents(self, orchestrator):
        """Test retrieving all agents."""
        orchestrator.register_agent("agent-1")
        orchestrator.register_agent("agent-2")
        orchestrator.register_agent("agent-3")

        agents = orchestrator.get_all_agents()
        assert len(agents) == 3

    def test_agent_heartbeat(self, orchestrator):
        """Test agent heartbeat updates."""
        orchestrator.register_agent("agent-1")

        # Send heartbeat
        orchestrator.agent_heartbeat("agent-1")

        agent = orchestrator.get_agent("agent-1")
        assert "last_heartbeat" in agent

    def test_deregister_agent(self, orchestrator):
        """Test deregistering an agent."""
        orchestrator.register_agent("agent-1")
        orchestrator.deregister_agent("agent-1")

        agent = orchestrator.get_agent("agent-1")
        assert agent is None or agent["status"] == "inactive"


class TestTaskClaiming:
    """Test task claiming and assignment."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Orchestrator(coordination_dir=tmpdir)
            orch.register_agent("agent-1")
            yield orch

    def test_claim_available_task(self, orchestrator):
        """Test claiming an available task."""
        task = orchestrator.add_task("Test task", priority=1)
        claimed = orchestrator.claim_task(task["id"], "agent-1")

        assert claimed["status"] == "claimed"
        assert claimed["assigned_to"] == "agent-1"

    def test_claim_already_claimed_task(self, orchestrator):
        """Test claiming a task that's already claimed."""
        task = orchestrator.add_task("Test task", priority=1)
        orchestrator.claim_task(task["id"], "agent-1")

        # Try to claim with different agent
        with pytest.raises(OrchestrationError):
            orchestrator.claim_task(task["id"], "agent-2")

    def test_claim_nonexistent_task(self, orchestrator):
        """Test claiming a task that doesn't exist."""
        with pytest.raises(OrchestrationError):
            orchestrator.claim_task("nonexistent-id", "agent-1")

    def test_claim_task_with_unmet_dependencies(self, orchestrator):
        """Test claiming a task with unmet dependencies."""
        task1 = orchestrator.add_task("Task 1", priority=1)
        task2 = orchestrator.add_task("Task 2", priority=2, dependencies=[task1["id"]])

        # Try to claim task2 before task1 is complete
        with pytest.raises(OrchestrationError):
            orchestrator.claim_task(task2["id"], "agent-1")

    def test_claim_task_with_met_dependencies(self, orchestrator):
        """Test claiming a task with met dependencies."""
        task1 = orchestrator.add_task("Task 1", priority=1)
        task2 = orchestrator.add_task("Task 2", priority=2, dependencies=[task1["id"]])

        # Complete task1
        orchestrator.claim_task(task1["id"], "agent-1")
        orchestrator.complete_task(task1["id"], result="Done")

        # Now claim task2
        claimed = orchestrator.claim_task(task2["id"], "agent-1")
        assert claimed["status"] == "claimed"

    def test_claim_done_task_raises(self, orchestrator):
        """Test claiming a task that is already done."""
        task = orchestrator.add_task("Test task", priority=1)
        orchestrator.claim_task(task["id"], "agent-1")
        orchestrator.complete_task(task["id"], result="Done")

        with pytest.raises(OrchestrationError):
            orchestrator.claim_task(task["id"], "agent-1")

    def test_claim_failed_task_raises(self, orchestrator):
        """Test claiming a task that has failed."""
        task = orchestrator.add_task("Test task", priority=1)
        orchestrator.claim_task(task["id"], "agent-1")
        orchestrator.fail_task(task["id"], error="Failed")

        with pytest.raises(OrchestrationError):
            orchestrator.claim_task(task["id"], "agent-1")


class TestStatePersistence:
    """Test state saving and loading."""

    def test_tasks_persisted_to_disk(self):
        """Test that tasks are saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = Orchestrator(coordination_dir=tmpdir)
            orchestrator.add_task("Test task", priority=1)

            tasks_file = Path(tmpdir) / "tasks.json"
            assert tasks_file.exists()

            data = json.loads(tasks_file.read_text())
            assert len(data["tasks"]) == 1

    def test_state_reloaded_after_restart(self):
        """Test that state is reloaded after orchestrator restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create orchestrator and add task
            orch1 = Orchestrator(coordination_dir=tmpdir)
            task = orch1.add_task("Test task", priority=1)
            task_id = task["id"]

            # Create new orchestrator instance
            orch2 = Orchestrator(coordination_dir=tmpdir)
            loaded_task = orch2.get_task(task_id)

            assert loaded_task is not None
            assert loaded_task["description"] == "Test task"

    def test_agents_persisted_to_disk(self):
        """Test that agents are saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = Orchestrator(coordination_dir=tmpdir)
            orchestrator.register_agent("agent-1", capabilities=["python"])

            agents_file = Path(tmpdir) / "agents.json"
            assert agents_file.exists()

            data = json.loads(agents_file.read_text())
            assert any(agent["id"] == "agent-1" for agent in data.get("agents", []))

    def test_discoveries_persisted_to_disk(self):
        """Test that discoveries are saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = Orchestrator(coordination_dir=tmpdir)
            orchestrator.add_discovery("Finding 1", "Content 1", "agent-1")

            discoveries_file = Path(tmpdir) / "discoveries.json"
            assert discoveries_file.exists()

            data = json.loads(discoveries_file.read_text())
            assert any(d["title"] == "Finding 1" for d in data.get("discoveries", []))


class TestErrorHandling:
    """Test error handling and validation."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Orchestrator(coordination_dir=tmpdir)
            yield orch

    def test_add_task_with_invalid_priority(self, orchestrator):
        """Test adding a task with invalid priority."""
        with pytest.raises(ValueError):
            orchestrator.add_task("Test task", priority=0)

        with pytest.raises(ValueError):
            orchestrator.add_task("Test task", priority=11)

    def test_add_task_with_empty_description(self, orchestrator):
        """Test adding a task with empty description."""
        with pytest.raises(ValueError):
            orchestrator.add_task("", priority=1)

    def test_add_task_with_nonexistent_dependencies(self, orchestrator):
        """Test adding a task with nonexistent dependencies."""
        with pytest.raises(OrchestrationError):
            orchestrator.add_task(
                "Test task",
                priority=1,
                dependencies=["nonexistent-id"]
            )

    def test_complete_unassigned_task(self, orchestrator):
        """Test completing a task that hasn't been claimed."""
        task = orchestrator.add_task("Test task", priority=1)

        with pytest.raises(OrchestrationError):
            orchestrator.complete_task(task["id"], result="Done")


class TestConcurrency:
    """Test concurrent access and race conditions."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Orchestrator(coordination_dir=tmpdir)
            yield orch

    def test_multiple_agents_claiming_same_task(self, orchestrator):
        """Test that only one agent can claim a task."""
        orchestrator.register_agent("agent-1")
        orchestrator.register_agent("agent-2")

        task = orchestrator.add_task("Test task", priority=1)

        # First agent claims successfully
        orchestrator.claim_task(task["id"], "agent-1")

        # Second agent fails to claim
        with pytest.raises(OrchestrationError):
            orchestrator.claim_task(task["id"], "agent-2")

    def test_claim_respects_disk_state(self):
        """Test that stale instances reload disk state before claiming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch1 = Orchestrator(coordination_dir=tmpdir)
            task = orch1.add_task("Test task", priority=1)

            orch2 = Orchestrator(coordination_dir=tmpdir)

            orch1.claim_task(task["id"], "agent-1")

            with pytest.raises(OrchestrationError):
                orch2.claim_task(task["id"], "agent-2")


class TestDiscoveries:
    """Test discovery management."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Orchestrator(coordination_dir=tmpdir)
            yield orch

    def test_add_discovery(self, orchestrator):
        """Test adding a discovery."""
        discovery = orchestrator.add_discovery(
            title="Important Finding",
            content="This is an important discovery about the system",
            agent_id="agent-1"
        )

        assert discovery["title"] == "Important Finding"
        assert discovery["agent_id"] == "agent-1"

    def test_get_discoveries(self, orchestrator):
        """Test retrieving discoveries."""
        orchestrator.add_discovery("Finding 1", "Content 1", "agent-1")
        orchestrator.add_discovery("Finding 2", "Content 2", "agent-2")

        discoveries = orchestrator.get_discoveries()
        assert len(discoveries) >= 2


class TestTaskPrioritization:
    """Test task prioritization logic."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Orchestrator(coordination_dir=tmpdir)
            yield orch

    def test_tasks_sorted_by_priority(self, orchestrator):
        """Test that tasks are sorted by priority."""
        orchestrator.add_task("Low priority", priority=5)
        orchestrator.add_task("High priority", priority=1)
        orchestrator.add_task("Medium priority", priority=3)

        available = orchestrator.get_available_tasks()

        # Should be sorted by priority (1, 3, 5)
        assert available[0]["priority"] == 1
        assert available[1]["priority"] == 3
        assert available[2]["priority"] == 5


class TestRetryBackoffPolicy:
    """Test retry and backoff policies for task claiming (ATOM-105)."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Orchestrator(coordination_dir=tmpdir)
            yield orch

    def test_claim_already_claimed_task_by_same_agent_succeeds(self, orchestrator):
        """Test that an agent can re-claim a task it already owns (idempotent claim)."""
        task = orchestrator.add_task("Test task", priority=1)
        orchestrator.register_agent("agent-1")

        # First claim
        claimed = orchestrator.claim_task(task["id"], "agent-1")
        assert claimed["status"] == "claimed"

        # Same agent re-claiming should succeed (idempotent)
        reclaimed = orchestrator.claim_task(task["id"], "agent-1")
        assert reclaimed["status"] == "claimed"
        assert reclaimed["assigned_to"] == "agent-1"

    def test_claim_task_with_retry_after_failure(self, orchestrator):
        """Test that a failed task can be retried."""
        task = orchestrator.add_task("Test task", priority=1)
        orchestrator.register_agent("agent-1")
        orchestrator.register_agent("agent-2")

        # First agent claims and fails
        orchestrator.claim_task(task["id"], "agent-1")
        orchestrator.fail_task(task["id"], "Connection timeout")

        # Task should be in failed state, not reclaimable without reset
        failed_task = orchestrator.get_task(task["id"])
        assert failed_task["status"] == "failed"

    def test_claim_respects_dependency_satisfaction(self, orchestrator):
        """Test that claim checks dependency satisfaction before allowing claim."""
        task1 = orchestrator.add_task("First task", priority=1)
        task2 = orchestrator.add_task(
            "Dependent task",
            priority=2,
            dependencies=[task1["id"]]
        )

        orchestrator.register_agent("agent-1")

        # Can't claim task2 because task1 is not done
        with pytest.raises(OrchestrationError, match="Dependencies not satisfied"):
            orchestrator.claim_task(task2["id"], "agent-1")

        # Complete task1
        orchestrator.claim_task(task1["id"], "agent-1")
        orchestrator.complete_task(task1["id"], "Done")

        # Now task2 should be claimable
        claimed = orchestrator.claim_task(task2["id"], "agent-1")
        assert claimed["status"] == "claimed"

    def test_concurrent_claim_attempts_atomic(self):
        """Test that concurrent claim attempts are handled atomically via file lock."""
        import threading
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = Orchestrator(coordination_dir=tmpdir)
            task = orchestrator.add_task("Contested task", priority=1)

            results = {"agent-1": None, "agent-2": None}
            errors = {"agent-1": None, "agent-2": None}

            def try_claim(agent_id: str):
                try:
                    # Small delay to encourage race condition
                    time.sleep(0.01)
                    orch = Orchestrator(coordination_dir=tmpdir)
                    result = orch.claim_task(task["id"], agent_id)
                    results[agent_id] = result
                except OrchestrationError as e:
                    errors[agent_id] = str(e)

            t1 = threading.Thread(target=try_claim, args=("agent-1",))
            t2 = threading.Thread(target=try_claim, args=("agent-2",))

            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # Exactly one should succeed, one should fail
            successes = sum(1 for r in results.values() if r is not None)
            failures = sum(1 for e in errors.values() if e is not None)

            assert successes == 1, f"Expected 1 success, got {successes}"
            assert failures == 1, f"Expected 1 failure, got {failures}"

    def test_backoff_on_empty_queue(self, orchestrator):
        """Test behavior when claiming from empty queue (no tasks available)."""
        available = orchestrator.get_available_tasks()
        assert len(available) == 0

        # Attempting to claim non-existent task should raise OrchestrationError
        with pytest.raises(OrchestrationError, match="Task not found"):
            orchestrator.claim_task("nonexistent-task", "agent-1")


class TestTaskSchemaValidation:
    """Test task schema validation (ATOM-108)."""

    def test_validate_valid_tasks(self):
        """Test that valid tasks pass validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = Orchestrator(coordination_dir=tmpdir)
            orchestrator.add_task("Valid task", priority=5)

            is_valid, errors = orchestrator.validate_tasks_schema()
            assert is_valid
            assert len(errors) == 0

    def test_validate_malformed_tasks_from_file(self):
        """Test that malformed tasks are detected during validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_dir = Path(tmpdir)
            coord_dir.mkdir(exist_ok=True)

            # Create tasks file with malformed entries
            tasks_data = {
                "tasks": [
                    # Missing required fields
                    {"id": "task-1"},
                    # Invalid status
                    {"id": "task-2", "description": "Test", "status": "invalid", "priority": 5},
                    # Invalid priority
                    {"id": "task-3", "description": "Test", "status": "available", "priority": 100},
                    # Valid task for comparison
                    {"id": "task-4", "description": "Valid", "status": "available", "priority": 1},
                ]
            }
            (coord_dir / "tasks.json").write_text(json.dumps(tasks_data))

            orchestrator = Orchestrator(coordination_dir=str(coord_dir))

            is_valid, errors = orchestrator.validate_tasks_schema()
            assert not is_valid
            assert len(errors) >= 3  # At least 3 errors from the malformed tasks

    def test_validate_strict_mode_raises(self):
        """Test that strict mode raises TaskSchemaError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_dir = Path(tmpdir)
            coord_dir.mkdir(exist_ok=True)

            # Create tasks file with missing required field
            tasks_data = {
                "tasks": [{"id": "task-1"}]  # Missing description, status, priority
            }
            (coord_dir / "tasks.json").write_text(json.dumps(tasks_data))

            orchestrator = Orchestrator(coordination_dir=str(coord_dir))

            with pytest.raises(TaskSchemaError, match="Schema validation failed"):
                orchestrator.validate_tasks_schema(strict=True)

    def test_validate_dependencies_array_type(self):
        """Test that dependencies must be an array of strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_dir = Path(tmpdir)
            coord_dir.mkdir(exist_ok=True)

            # Create tasks file with invalid dependencies
            tasks_data = {
                "tasks": [
                    {
                        "id": "task-1",
                        "description": "Test",
                        "status": "available",
                        "priority": 1,
                        "dependencies": "not-an-array",  # Should be array
                    }
                ]
            }
            (coord_dir / "tasks.json").write_text(json.dumps(tasks_data))

            orchestrator = Orchestrator(coordination_dir=str(coord_dir))

            is_valid, errors = orchestrator.validate_tasks_schema()
            assert not is_valid
            assert any("dependencies" in e and "array" in e for e in errors)


class TestDuplicateTaskIdValidation:
    """Test duplicate task ID validation (ATOM-107)."""

    def test_add_task_with_id_rejects_duplicate(self):
        """Test that add_task_with_id rejects duplicate IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = Orchestrator(coordination_dir=tmpdir)

            # Add first task with specific ID
            orchestrator.add_task_with_id("task-custom-001", "First task", priority=1)

            # Attempt to add duplicate should fail
            with pytest.raises(OrchestrationError, match="Duplicate task ID"):
                orchestrator.add_task_with_id("task-custom-001", "Duplicate task", priority=2)

    def test_add_task_with_id_succeeds_with_unique_id(self):
        """Test that add_task_with_id works with unique IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = Orchestrator(coordination_dir=tmpdir)

            task1 = orchestrator.add_task_with_id("task-001", "First task", priority=1)
            task2 = orchestrator.add_task_with_id("task-002", "Second task", priority=2)

            assert task1["id"] == "task-001"
            assert task2["id"] == "task-002"
            assert len(orchestrator.get_all_tasks()) == 2

    def test_load_deduplicates_tasks_from_file(self):
        """Test that loading a file with duplicates deduplicates them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_dir = Path(tmpdir)
            coord_dir.mkdir(exist_ok=True)

            # Create tasks file with duplicates
            tasks_data = {
                "tasks": [
                    {"id": "task-1", "description": "First", "status": "available", "priority": 1},
                    {"id": "task-1", "description": "Duplicate", "status": "available", "priority": 2},
                    {"id": "task-2", "description": "Second", "status": "available", "priority": 3},
                ]
            }
            (coord_dir / "tasks.json").write_text(json.dumps(tasks_data))

            orchestrator = Orchestrator(coordination_dir=str(coord_dir))

            # Should have deduplicated
            tasks = orchestrator.get_all_tasks()
            assert len(tasks) == 2

            # First occurrence kept
            task1 = orchestrator.get_task("task-1")
            assert task1["description"] == "First"

    def test_add_task_with_id_validates_empty_id(self):
        """Test that add_task_with_id rejects empty IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = Orchestrator(coordination_dir=tmpdir)

            with pytest.raises(ValueError, match="Task ID cannot be empty"):
                orchestrator.add_task_with_id("", "Some task", priority=1)


class TestMetricsIntegration:
    """Test metrics collection in orchestrator (ATOM-104)."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Orchestrator(coordination_dir=tmpdir)
            yield orch

    def test_get_metrics_json(self, orchestrator):
        """Test retrieving metrics in JSON format."""
        orchestrator.add_task("Task 1", priority=1)
        orchestrator.add_task("Task 2", priority=2)

        metrics = orchestrator.get_metrics(format="json")
        data = json.loads(metrics)

        assert "queue_size" in data
        assert data["queue_size"]["total"] == 2
        assert data["queue_size"]["available"] == 2

    def test_get_metrics_prometheus(self, orchestrator):
        """Test retrieving metrics in Prometheus format."""
        orchestrator.add_task("Task 1", priority=1)

        metrics = orchestrator.get_metrics(format="prometheus")

        assert "orchestrator_queue_total 1" in metrics
        assert "orchestrator_queue_available 1" in metrics

    def test_save_metrics_to_file(self, orchestrator):
        """Test saving metrics to a file."""
        orchestrator.add_task("Task 1", priority=1)

        path = orchestrator.save_metrics()
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["queue_size"]["total"] == 1


class TestAgentCapabilityMatching:
    """Test agent capability matching for orchestrator routing (ATOM-111)."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = Orchestrator(coordination_dir=tmpdir)
            yield orch

    def test_register_agent_with_capabilities(self, orchestrator):
        """Test that agents can be registered with capabilities."""
        agent = orchestrator.register_agent(
            "agent-python",
            capabilities=["python", "testing", "debugging"]
        )

        assert agent["id"] == "agent-python"
        assert set(agent["capabilities"]) == {"python", "testing", "debugging"}

    def test_update_agent_capabilities(self, orchestrator):
        """Test that agent capabilities can be updated."""
        # Initial registration
        orchestrator.register_agent("agent-1", capabilities=["python"])

        # Update capabilities
        updated = orchestrator.register_agent(
            "agent-1",
            capabilities=["python", "typescript", "rust"]
        )

        assert set(updated["capabilities"]) == {"python", "typescript", "rust"}

    def test_get_agents_by_capability(self, orchestrator):
        """Test filtering agents by capability."""
        orchestrator.register_agent("agent-py", capabilities=["python", "testing"])
        orchestrator.register_agent("agent-ts", capabilities=["typescript", "testing"])
        orchestrator.register_agent("agent-rust", capabilities=["rust"])

        all_agents = orchestrator.get_all_agents()

        # Filter by capability
        python_agents = [a for a in all_agents if "python" in a.get("capabilities", [])]
        testing_agents = [a for a in all_agents if "testing" in a.get("capabilities", [])]

        assert len(python_agents) == 1
        assert python_agents[0]["id"] == "agent-py"

        assert len(testing_agents) == 2
        assert {a["id"] for a in testing_agents} == {"agent-py", "agent-ts"}

    def test_empty_capabilities_default(self, orchestrator):
        """Test that agents default to empty capabilities."""
        agent = orchestrator.register_agent("agent-generic")

        assert agent["capabilities"] == []

    def test_agent_capabilities_persisted(self):
        """Test that agent capabilities are persisted across orchestrator instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First instance
            orch1 = Orchestrator(coordination_dir=tmpdir)
            orch1.register_agent("persistent-agent", capabilities=["python", "web"])

            # Second instance (same directory)
            orch2 = Orchestrator(coordination_dir=tmpdir)
            agent = orch2.get_agent("persistent-agent")

            assert agent is not None
            assert set(agent["capabilities"]) == {"python", "web"}


class TestSpecializationRouter:
    """Test the SpecializationRouter for capability-based task routing (ATOM-111)."""

    def test_register_worker_specialization(self):
        """Test worker registration with SpecializationRouter."""
        from orchestrator.advanced import SpecializationRouter

        router = SpecializationRouter()
        router.register_worker(
            "worker-py",
            skills={"python", "testing"},
            preferred_task_types={"code", "test"},
            max_concurrent=2,
        )

        # Worker should be findable by skill
        # get_workers_for_task returns List[Tuple[str, float]]
        workers = router.get_workers_for_task(required_skills={"python"})
        worker_ids = [w[0] for w in workers]
        assert "worker-py" in worker_ids

    def test_skill_based_routing(self):
        """Test routing tasks to workers based on skills."""
        from orchestrator.advanced import SpecializationRouter

        router = SpecializationRouter()
        router.register_worker("worker-frontend", skills={"typescript", "react"})
        router.register_worker("worker-backend", skills={"python", "django"})
        router.register_worker("worker-devops", skills={"docker", "kubernetes"})

        # Find workers for Python tasks
        py_workers = router.get_workers_for_task(required_skills={"python"})
        py_worker_ids = [w[0] for w in py_workers]
        assert "worker-backend" in py_worker_ids
        assert "worker-frontend" not in py_worker_ids

        # Find workers for React tasks
        react_workers = router.get_workers_for_task(required_skills={"react"})
        react_worker_ids = [w[0] for w in react_workers]
        assert "worker-frontend" in react_worker_ids

    def test_no_matching_workers(self):
        """Test behavior when no workers match required skills."""
        from orchestrator.advanced import SpecializationRouter

        router = SpecializationRouter()
        router.register_worker("worker-1", skills={"python"})

        # Look for Rust workers when none exist
        workers = router.get_workers_for_task(required_skills={"rust"})
        assert len(workers) == 0

    def test_multiple_skill_requirement(self):
        """Test finding workers that match multiple skill requirements."""
        from orchestrator.advanced import SpecializationRouter

        router = SpecializationRouter()
        router.register_worker("worker-fullstack", skills={"python", "typescript", "react"})
        router.register_worker("worker-py-only", skills={"python"})

        # Require both Python and TypeScript
        workers = router.get_workers_for_task(required_skills={"python", "typescript"})
        worker_ids = [w[0] for w in workers]

        # Only fullstack worker has both
        assert "worker-fullstack" in worker_ids
        # py-only doesn't have typescript
        assert "worker-py-only" not in worker_ids

    def test_select_best_worker(self):
        """Test selecting the best worker based on priority boost."""
        from orchestrator.advanced import SpecializationRouter

        router = SpecializationRouter()
        router.register_worker(
            "worker-specialist",
            skills={"python"},
            priority_boost=2.0,  # Preferred for Python
        )
        router.register_worker(
            "worker-generalist",
            skills={"python", "typescript"},
            priority_boost=1.0,
        )

        # get_workers_for_task returns workers sorted by score descending
        workers = router.get_workers_for_task(required_skills={"python"})
        assert len(workers) > 0
        best = workers[0][0]  # First element is best, [0] gets worker_id

        # Specialist should be preferred due to higher priority boost
        assert best == "worker-specialist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
