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
    Task,
    TaskStatus,
    Agent,
    AgentStatus,
    OrchestrationError,
)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
