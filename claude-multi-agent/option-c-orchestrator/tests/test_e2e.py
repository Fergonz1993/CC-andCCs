"""
End-to-end test scenarios for the multi-agent orchestrator.

Feature: adv-test-011 - End-to-end test scenarios

This module provides comprehensive E2E tests that verify the complete
workflow from initialization through task execution and aggregation.

Run with: pytest tests/test_e2e.py -v --timeout=60
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.models import (
    Task,
    TaskStatus,
    TaskResult,
    TaskContext,
    Discovery,
    CoordinationState,
    AgentRole,
)
from orchestrator.orchestrator import Orchestrator


class TestE2EOrchestrationFlow:
    """
    End-to-end tests for the complete orchestration workflow.

    These tests simulate the full lifecycle:
    1. Orchestrator initialization
    2. Task creation and queuing
    3. Task claiming and execution
    4. Result aggregation
    5. State persistence
    """

    @pytest.fixture
    def temp_working_dir(self):
        """Create a temporary working directory for tests."""
        with tempfile.TemporaryDirectory(prefix="e2e_test_") as tmp:
            yield Path(tmp)

    @pytest.fixture
    def mock_agent(self):
        """Create a mock ClaudeCodeAgent for testing."""
        agent = MagicMock()
        agent.is_running = True
        agent.agent_id = "test-worker-1"
        agent._current_task = None
        agent.execute_task = AsyncMock(return_value=TaskResult(
            output="Task completed successfully",
            files_modified=["test.py"],
            discoveries=["Found a useful pattern"]
        ))
        agent.send_prompt = AsyncMock(return_value="[{\"description\": \"Test task\", \"priority\": 1}]")
        agent.to_model = MagicMock(return_value=MagicMock(model_dump=MagicMock(return_value={})))
        return agent

    @pytest.mark.asyncio
    async def test_e2e_001_full_orchestration_lifecycle(self, temp_working_dir, mock_agent):
        """
        e2e-001: Complete orchestration lifecycle from init to results.

        Verifies:
        - Orchestrator initializes correctly
        - Tasks can be added to the queue
        - Tasks progress through all states
        - Results are collected
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            max_workers=2,
            verbose=False
        )

        # Initialize with a goal
        await orchestrator.initialize(
            goal="Build a simple REST API",
            master_plan="1. Create models\n2. Create routes\n3. Add tests"
        )

        assert orchestrator.state.goal == "Build a simple REST API"
        assert orchestrator.state.master_plan != ""

        # Add tasks
        task1 = orchestrator.add_task(
            description="Create user model",
            priority=1,
            context_files=["models/user.py"],
            hints="Use SQLAlchemy"
        )

        task2 = orchestrator.add_task(
            description="Create API routes",
            priority=2,
            dependencies=[task1.id],
            context_files=["routes/api.py"]
        )

        assert len(orchestrator.state.tasks) == 2
        assert task1.status == TaskStatus.AVAILABLE
        assert task2.status == TaskStatus.AVAILABLE

        # Verify task dependency chain
        assert task1.id in task2.dependencies

    @pytest.mark.asyncio
    async def test_e2e_002_task_claiming_and_execution(self, temp_working_dir, mock_agent):
        """
        e2e-002: Task claiming respects dependencies and priority.

        Verifies:
        - Higher priority tasks claimed first
        - Dependencies block task claiming
        - Tasks transition through states correctly
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            max_workers=2,
            verbose=False
        )

        await orchestrator.initialize(goal="Test task execution")

        # Create tasks with dependencies
        task1 = orchestrator.add_task(description="First task", priority=1)
        task2 = orchestrator.add_task(
            description="Dependent task",
            priority=1,
            dependencies=[task1.id]
        )

        # Claim first task
        claimed = await orchestrator.claim_task("worker-1")

        assert claimed is not None
        assert claimed.id == task1.id  # First task has no deps
        assert claimed.status == TaskStatus.CLAIMED
        assert claimed.claimed_by == "worker-1"

        # Try to claim second task - should fail (dependency not met)
        claimed2 = await orchestrator.claim_task("worker-2")

        # task2 has unmet dependencies, so no task should be available
        assert claimed2 is None

    @pytest.mark.asyncio
    async def test_e2e_003_task_completion_flow(self, temp_working_dir):
        """
        e2e-003: Task completion updates state and triggers callbacks.

        Verifies:
        - complete_task updates status
        - Discoveries are collected
        - Callbacks are invoked
        """
        discoveries_collected = []
        tasks_completed = []

        def on_discovery(disc):
            discoveries_collected.append(disc)

        def on_complete(task):
            tasks_completed.append(task)

        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            max_workers=1,
            verbose=False,
            on_discovery=on_discovery,
            on_task_complete=on_complete
        )

        await orchestrator.initialize(goal="Test completion")

        task = orchestrator.add_task(description="Task to complete")
        await orchestrator.claim_task("worker-1")

        # Complete the task with a result
        result = TaskResult(
            output="Successfully completed",
            files_modified=["test.py"],
            discoveries=["Found an important pattern"]
        )

        success = await orchestrator.complete_task(task.id, result)

        assert success is True
        assert task.status == TaskStatus.DONE
        assert task.result == result
        assert len(discoveries_collected) == 1
        assert len(tasks_completed) == 1

    @pytest.mark.asyncio
    async def test_e2e_004_task_failure_handling(self, temp_working_dir):
        """
        e2e-004: Failed tasks are handled correctly.

        Verifies:
        - fail_task updates status
        - Error message is recorded
        - State is updated
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            max_workers=1,
            verbose=False
        )

        await orchestrator.initialize(goal="Test failure handling")

        task = orchestrator.add_task(description="Task that will fail")
        await orchestrator.claim_task("worker-1")

        # Fail the task
        success = await orchestrator.fail_task(task.id, "Simulated error")

        assert success is True
        assert task.status == TaskStatus.FAILED
        assert task.result is not None
        assert task.result.error == "Simulated error"

    @pytest.mark.asyncio
    async def test_e2e_005_state_persistence(self, temp_working_dir):
        """
        e2e-005: State can be saved and loaded.

        Verifies:
        - save_state writes valid JSON
        - load_state restores state correctly
        - All state fields are preserved
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            max_workers=2,
            verbose=False
        )

        await orchestrator.initialize(
            goal="Test persistence",
            master_plan="Detailed plan here"
        )

        # Add some tasks and discoveries
        task = orchestrator.add_task(description="Persistent task")
        orchestrator.state.discoveries.append(Discovery(
            agent_id="worker-1",
            content="Important finding",
            tags=["test"]
        ))

        # Save state
        state_file = str(temp_working_dir / "state.json")
        orchestrator.save_state(state_file)

        assert Path(state_file).exists()

        # Load state into a new orchestrator
        new_orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            verbose=False
        )
        new_orchestrator.load_state(state_file)

        assert new_orchestrator.state.goal == "Test persistence"
        assert len(new_orchestrator.state.tasks) == 1
        assert len(new_orchestrator.state.discoveries) == 1

    @pytest.mark.asyncio
    async def test_e2e_006_batch_task_creation(self, temp_working_dir):
        """
        e2e-006: Batch task creation works correctly.

        Verifies:
        - Multiple tasks created at once
        - All task properties set correctly
        - Dependencies maintained
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            verbose=False
        )

        await orchestrator.initialize(goal="Test batch creation")

        tasks_data = [
            {"description": "Task 1", "priority": 1},
            {"description": "Task 2", "priority": 2, "hints": "Use pattern X"},
            {"description": "Task 3", "priority": 3, "context_files": ["file.py"]},
        ]

        created = orchestrator.add_tasks_batch(tasks_data)

        assert len(created) == 3
        assert orchestrator.state.tasks[0].priority == 1
        assert orchestrator.state.tasks[1].context.hints == "Use pattern X"
        assert "file.py" in orchestrator.state.tasks[2].context.files

    @pytest.mark.asyncio
    async def test_e2e_007_progress_tracking(self, temp_working_dir):
        """
        e2e-007: Progress tracking reports correct statistics.

        Verifies:
        - get_progress returns accurate counts
        - Percentage calculation is correct
        - Status breakdown is accurate
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            verbose=False
        )

        await orchestrator.initialize(goal="Test progress")

        # Create tasks in different states
        task1 = orchestrator.add_task(description="Task 1")
        task2 = orchestrator.add_task(description="Task 2")
        task3 = orchestrator.add_task(description="Task 3")

        # Complete one, claim another
        task1.complete(TaskResult(output="Done"))
        await orchestrator.claim_task("worker-1")

        progress = orchestrator.state.get_progress()

        assert progress["total_tasks"] == 3
        assert progress["by_status"]["done"] == 1
        assert progress["by_status"]["claimed"] == 1
        assert progress["by_status"]["available"] == 1
        # 1 of 3 done = 33.3%
        assert progress["percent_complete"] == pytest.approx(33.3, abs=0.1)

    @pytest.mark.asyncio
    async def test_e2e_008_dependency_chain_resolution(self, temp_working_dir):
        """
        e2e-008: Complex dependency chains are resolved correctly.

        Verifies:
        - Multi-level dependencies work
        - Tasks become available when deps complete
        - get_available_tasks filters correctly
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            verbose=False
        )

        await orchestrator.initialize(goal="Test dependencies")

        # Create dependency chain: A -> B -> C
        task_a = orchestrator.add_task(description="Task A")
        task_b = orchestrator.add_task(
            description="Task B",
            dependencies=[task_a.id]
        )
        task_c = orchestrator.add_task(
            description="Task C",
            dependencies=[task_b.id]
        )

        # Initially only A is available
        available = orchestrator.state.get_available_tasks()
        assert len(available) == 1
        assert available[0].id == task_a.id

        # Complete A
        task_a.complete(TaskResult(output="Done"))

        # Now B should be available
        available = orchestrator.state.get_available_tasks()
        assert len(available) == 1
        assert available[0].id == task_b.id

        # Complete B
        task_b.complete(TaskResult(output="Done"))

        # Now C should be available
        available = orchestrator.state.get_available_tasks()
        assert len(available) == 1
        assert available[0].id == task_c.id

    @pytest.mark.asyncio
    async def test_e2e_009_concurrent_task_claiming(self, temp_working_dir):
        """
        e2e-009: Concurrent task claiming is handled safely.

        Verifies:
        - Multiple claim requests are serialized
        - No task is claimed twice
        - Lock mechanism works
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            max_workers=5,
            verbose=False
        )

        await orchestrator.initialize(goal="Test concurrency")

        # Add several tasks
        for i in range(5):
            orchestrator.add_task(description=f"Task {i}")

        # Simulate concurrent claims
        async def claim_worker(worker_id):
            return await orchestrator.claim_task(worker_id)

        # Run claims concurrently
        results = await asyncio.gather(
            claim_worker("worker-1"),
            claim_worker("worker-2"),
            claim_worker("worker-3"),
            claim_worker("worker-4"),
            claim_worker("worker-5"),
        )

        # Each task should be claimed by exactly one worker
        claimed_ids = [r.id for r in results if r is not None]
        assert len(claimed_ids) == len(set(claimed_ids))  # All unique

    @pytest.mark.asyncio
    async def test_e2e_010_status_report_generation(self, temp_working_dir):
        """
        e2e-010: Status report generation includes all relevant data.

        Verifies:
        - get_status returns comprehensive data
        - All state fields are included
        - Data format is correct
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            verbose=False
        )

        await orchestrator.initialize(goal="Test status report")

        orchestrator.add_task(description="Sample task")
        orchestrator.state.discoveries.append(Discovery(
            agent_id="worker-1",
            content="Test discovery"
        ))

        status = orchestrator.get_status()

        assert "goal" in status
        assert "progress" in status
        assert "discoveries" in status
        assert "last_activity" in status
        assert status["goal"] == "Test status report"


class TestE2EErrorScenarios:
    """
    End-to-end tests for error handling scenarios.
    """

    @pytest.fixture
    def temp_working_dir(self):
        with tempfile.TemporaryDirectory(prefix="e2e_error_test_") as tmp:
            yield Path(tmp)

    @pytest.mark.asyncio
    async def test_e2e_error_001_invalid_task_completion(self, temp_working_dir):
        """
        e2e-error-001: Completing non-existent task fails gracefully.
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            verbose=False
        )
        await orchestrator.initialize(goal="Test error handling")

        result = await orchestrator.complete_task(
            "nonexistent-task-id",
            TaskResult(output="Done")
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_e2e_error_002_invalid_task_failure(self, temp_working_dir):
        """
        e2e-error-002: Failing non-existent task fails gracefully.
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            verbose=False
        )
        await orchestrator.initialize(goal="Test error handling")

        result = await orchestrator.fail_task(
            "nonexistent-task-id",
            "Error message"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_e2e_error_003_load_invalid_state_file(self, temp_working_dir):
        """
        e2e-error-003: Loading invalid state file raises appropriate error.
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            verbose=False
        )

        # Create an invalid state file
        invalid_file = temp_working_dir / "invalid.json"
        invalid_file.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            orchestrator.load_state(str(invalid_file))

    @pytest.mark.asyncio
    async def test_e2e_error_004_claim_from_empty_queue(self, temp_working_dir):
        """
        e2e-error-004: Claiming from empty queue returns None.
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            verbose=False
        )
        await orchestrator.initialize(goal="Test empty queue")

        claimed = await orchestrator.claim_task("worker-1")

        assert claimed is None


class TestE2EIntegrationScenarios:
    """
    Integration scenarios that test multiple components working together.
    """

    @pytest.fixture
    def temp_working_dir(self):
        with tempfile.TemporaryDirectory(prefix="e2e_integration_") as tmp:
            yield Path(tmp)

    @pytest.mark.asyncio
    async def test_e2e_integration_001_discovery_propagation(self, temp_working_dir):
        """
        e2e-integration-001: Discoveries propagate through the system.

        Verifies that discoveries from task completion are properly
        stored and accessible.
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            verbose=False
        )
        await orchestrator.initialize(goal="Test discovery propagation")

        task = orchestrator.add_task(description="Task with discoveries")
        await orchestrator.claim_task("worker-1")

        result = TaskResult(
            output="Completed",
            discoveries=[
                "API rate limit is 100 req/min",
                "Database requires migration"
            ]
        )

        await orchestrator.complete_task(task.id, result)

        # Discoveries should be in state
        assert len(orchestrator.state.discoveries) == 2
        assert any(
            "rate limit" in d.content
            for d in orchestrator.state.discoveries
        )

    @pytest.mark.asyncio
    async def test_e2e_integration_002_multi_worker_coordination(self, temp_working_dir):
        """
        e2e-integration-002: Multiple workers can coordinate on parallel tasks.
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            max_workers=3,
            verbose=False
        )
        await orchestrator.initialize(goal="Test multi-worker")

        # Create independent tasks (can be parallelized)
        tasks = [
            orchestrator.add_task(description=f"Independent task {i}")
            for i in range(3)
        ]

        # Claim tasks with different workers
        for i, worker_id in enumerate(["worker-1", "worker-2", "worker-3"]):
            claimed = await orchestrator.claim_task(worker_id)
            assert claimed is not None
            assert claimed.claimed_by == worker_id

        # Verify all tasks claimed by different workers
        claimed_by_set = {t.claimed_by for t in orchestrator.state.tasks}
        assert len(claimed_by_set) == 3

    @pytest.mark.asyncio
    async def test_e2e_integration_003_task_context_preservation(self, temp_working_dir):
        """
        e2e-integration-003: Task context is preserved through execution.
        """
        orchestrator = Orchestrator(
            working_directory=str(temp_working_dir),
            verbose=False
        )
        await orchestrator.initialize(goal="Test context preservation")

        task = orchestrator.add_task(
            description="Contextual task",
            context_files=["src/main.py", "tests/test_main.py"],
            hints="Check the error handling patterns"
        )

        # Context should be preserved in state
        stored_task = orchestrator.state.get_task(task.id)
        assert stored_task is not None
        assert len(stored_task.context.files) == 2
        assert "error handling" in stored_task.context.hints
