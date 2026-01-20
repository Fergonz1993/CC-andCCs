"""
Snapshot testing for task states.

Feature: adv-test-012 - Snapshot testing for task states

This module provides snapshot testing capabilities for verifying
that task state serialization remains consistent over time.

Run with: pytest tests/test_snapshots.py -v
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import hashlib

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


class SnapshotManager:
    """
    Manages snapshot files for testing.

    Snapshots are stored as JSON files and can be used to verify
    that serialization behavior remains consistent.
    """

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._current_snapshots: Dict[str, Any] = {}

    def get_snapshot_path(self, name: str) -> Path:
        """Get the path for a named snapshot."""
        safe_name = name.replace(" ", "_").replace("/", "_")
        return self.snapshot_dir / f"{safe_name}.snapshot.json"

    def save_snapshot(self, name: str, data: Any) -> None:
        """Save a snapshot to disk."""
        path = self.get_snapshot_path(name)

        # Normalize datetime fields for consistent comparison
        normalized = self._normalize_data(data)

        with open(path, 'w') as f:
            json.dump(normalized, f, indent=2, sort_keys=True, default=str)

        self._current_snapshots[name] = normalized

    def load_snapshot(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a snapshot from disk."""
        path = self.get_snapshot_path(name)

        if not path.exists():
            return None

        with open(path, 'r') as f:
            return json.load(f)

    def assert_match(self, name: str, data: Any, update: bool = False) -> None:
        """
        Assert that data matches the stored snapshot.

        Args:
            name: Snapshot name
            data: Data to compare
            update: If True, update the snapshot instead of comparing
        """
        normalized = self._normalize_data(data)

        if update:
            self.save_snapshot(name, normalized)
            return

        stored = self.load_snapshot(name)

        if stored is None:
            # First run - save the snapshot
            self.save_snapshot(name, normalized)
            return

        # Compare normalized data
        assert normalized == stored, (
            f"Snapshot mismatch for '{name}'.\n"
            f"Expected:\n{json.dumps(stored, indent=2)}\n\n"
            f"Got:\n{json.dumps(normalized, indent=2)}"
        )

    def _normalize_data(self, data: Any) -> Any:
        """
        Normalize data for consistent snapshot comparison.

        - Converts datetime objects to a stable placeholder
        - Removes volatile fields like IDs
        - Sorts lists for consistent ordering
        """
        if isinstance(data, dict):
            result = {}
            for key, value in sorted(data.items()):
                # Skip volatile fields
                if key in ('id', 'created_at', 'claimed_at', 'started_at',
                           'completed_at', 'last_heartbeat', 'last_activity'):
                    result[key] = "<NORMALIZED>"
                else:
                    result[key] = self._normalize_data(value)
            return result
        elif isinstance(data, (list, tuple)):
            return [self._normalize_data(item) for item in data]
        elif isinstance(data, datetime):
            return "<DATETIME>"
        elif hasattr(data, 'model_dump'):
            return self._normalize_data(data.model_dump())
        else:
            return data

    def get_snapshot_hash(self, name: str) -> str:
        """Get a hash of a snapshot for quick comparison."""
        stored = self.load_snapshot(name)
        if stored is None:
            return ""
        content = json.dumps(stored, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()


@pytest.fixture
def snapshot_manager(tmp_path):
    """Create a snapshot manager for tests."""
    return SnapshotManager(tmp_path / "snapshots")


class TestTaskStateSnapshots:
    """
    Snapshot tests for Task model states.
    """

    def test_snapshot_001_new_task_state(self, snapshot_manager):
        """
        snapshot-001: New task has expected default state.

        Verifies the structure of a newly created task.
        """
        task = Task(description="Test task")

        snapshot_manager.assert_match("new_task_state", task.model_dump())

    def test_snapshot_002_claimed_task_state(self, snapshot_manager):
        """
        snapshot-002: Claimed task state structure is correct.
        """
        task = Task(description="Task to be claimed")
        task.claim("worker-1")

        snapshot_manager.assert_match("claimed_task_state", task.model_dump())

    def test_snapshot_003_in_progress_task_state(self, snapshot_manager):
        """
        snapshot-003: In-progress task state structure is correct.
        """
        task = Task(description="Task in progress")
        task.claim("worker-1")
        task.start()

        snapshot_manager.assert_match("in_progress_task_state", task.model_dump())

    def test_snapshot_004_completed_task_state(self, snapshot_manager):
        """
        snapshot-004: Completed task state structure is correct.
        """
        task = Task(description="Task to complete")
        task.claim("worker-1")
        task.start()
        task.complete(TaskResult(
            output="Successfully completed",
            files_modified=["src/main.py"],
            discoveries=["Found pattern X"]
        ))

        snapshot_manager.assert_match("completed_task_state", task.model_dump())

    def test_snapshot_005_failed_task_state(self, snapshot_manager):
        """
        snapshot-005: Failed task state structure is correct.
        """
        task = Task(description="Task that will fail")
        task.claim("worker-1")
        task.start()
        task.fail("Simulated failure")

        snapshot_manager.assert_match("failed_task_state", task.model_dump())

    def test_snapshot_006_task_with_dependencies(self, snapshot_manager):
        """
        snapshot-006: Task with dependencies has correct structure.
        """
        task = Task(
            description="Dependent task",
            dependencies=["task-001", "task-002"],
            priority=2
        )

        snapshot_manager.assert_match("dependent_task_state", task.model_dump())

    def test_snapshot_007_task_with_full_context(self, snapshot_manager):
        """
        snapshot-007: Task with full context has correct structure.
        """
        task = Task(
            description="Contextual task",
            priority=1,
            context=TaskContext(
                files=["src/main.py", "tests/test_main.py"],
                hints="Check error handling",
                parent_task="task-parent",
                tags=["backend", "api"],
                metadata={"team": "platform", "sprint": 5}
            )
        )

        snapshot_manager.assert_match("contextual_task_state", task.model_dump())

    def test_snapshot_008_reset_task_state(self, snapshot_manager):
        """
        snapshot-008: Reset task returns to available state.
        """
        task = Task(description="Task to reset", max_attempts=3)
        task.claim("worker-1")
        task.start()
        task.fail("First attempt failed")
        task.reset()

        snapshot_manager.assert_match("reset_task_state", task.model_dump())


class TestCoordinationStateSnapshots:
    """
    Snapshot tests for CoordinationState.
    """

    def test_snapshot_009_empty_coordination_state(self, snapshot_manager):
        """
        snapshot-009: Empty coordination state has correct structure.
        """
        state = CoordinationState()

        snapshot_manager.assert_match("empty_coordination_state", state.model_dump())

    def test_snapshot_010_coordination_state_with_tasks(self, snapshot_manager):
        """
        snapshot-010: Coordination state with tasks has correct structure.
        """
        state = CoordinationState(
            goal="Build a REST API",
            master_plan="1. Create models\n2. Add routes",
            working_directory="/project"
        )

        state.add_task(Task(description="Task 1"))
        state.add_task(Task(description="Task 2", priority=2))

        snapshot_manager.assert_match("coordination_state_with_tasks", state.model_dump())

    def test_snapshot_011_coordination_state_with_discoveries(self, snapshot_manager):
        """
        snapshot-011: Coordination state with discoveries has correct structure.
        """
        state = CoordinationState(goal="Test project")
        state.discoveries.append(Discovery(
            agent_id="worker-1",
            content="Found important pattern",
            tags=["architecture"],
            related_task="task-001"
        ))

        snapshot_manager.assert_match("coordination_state_with_discoveries", state.model_dump())

    def test_snapshot_012_full_coordination_state(self, snapshot_manager):
        """
        snapshot-012: Full coordination state has correct structure.
        """
        state = CoordinationState(
            goal="Complete project",
            master_plan="Detailed plan",
            working_directory="/project",
            max_parallel_workers=5,
            task_timeout_seconds=300
        )

        # Add tasks in various states
        task1 = Task(description="Completed task")
        task1.complete(TaskResult(output="Done"))
        state.add_task(task1)

        task2 = Task(description="In progress task")
        task2.claim("worker-1")
        task2.start()
        state.add_task(task2)

        task3 = Task(description="Available task")
        state.add_task(task3)

        # Add discoveries
        state.discoveries.append(Discovery(
            agent_id="worker-1",
            content="Important finding"
        ))

        snapshot_manager.assert_match("full_coordination_state", state.model_dump())


class TestAgentStateSnapshots:
    """
    Snapshot tests for Agent and AgentMetrics models.
    """

    def test_snapshot_013_agent_default_state(self, snapshot_manager):
        """
        snapshot-013: Agent default state has correct structure.
        """
        agent = Agent(id="worker-1", role=AgentRole.WORKER)

        snapshot_manager.assert_match("agent_default_state", agent.model_dump())

    def test_snapshot_014_agent_with_metrics(self, snapshot_manager):
        """
        snapshot-014: Agent with metrics has correct structure.
        """
        agent = Agent(
            id="worker-1",
            role=AgentRole.WORKER,
            is_active=True,
            current_task="task-001",
            pid=12345,
            working_directory="/project",
            model="claude-sonnet-4-20250514",
            metrics=AgentMetrics(
                tasks_completed=10,
                tasks_failed=2,
                total_time_working=3600.0,
                avg_task_time=300.0,
                discoveries_made=5
            )
        )

        snapshot_manager.assert_match("agent_with_metrics", agent.model_dump())

    def test_snapshot_015_leader_agent_state(self, snapshot_manager):
        """
        snapshot-015: Leader agent state has correct structure.
        """
        agent = Agent(
            id="leader",
            role=AgentRole.LEADER,
            is_active=True,
            working_directory="/project"
        )

        snapshot_manager.assert_match("leader_agent_state", agent.model_dump())


class TestTaskResultSnapshots:
    """
    Snapshot tests for TaskResult model.
    """

    def test_snapshot_016_minimal_task_result(self, snapshot_manager):
        """
        snapshot-016: Minimal task result has correct structure.
        """
        result = TaskResult(output="Task completed")

        snapshot_manager.assert_match("minimal_task_result", result.model_dump())

    def test_snapshot_017_full_task_result(self, snapshot_manager):
        """
        snapshot-017: Full task result has correct structure.
        """
        result = TaskResult(
            output="Completed comprehensive changes",
            files_modified=["src/main.py", "src/utils.py"],
            files_created=["src/new_module.py"],
            files_deleted=["src/deprecated.py"],
            subtasks_created=["task-sub-001", "task-sub-002"],
            discoveries=["Pattern X found", "Performance issue identified"]
        )

        snapshot_manager.assert_match("full_task_result", result.model_dump())

    def test_snapshot_018_error_task_result(self, snapshot_manager):
        """
        snapshot-018: Error task result has correct structure.
        """
        result = TaskResult(
            output="",
            error="Failed due to missing dependency"
        )

        snapshot_manager.assert_match("error_task_result", result.model_dump())


class TestDiscoverySnapshots:
    """
    Snapshot tests for Discovery model.
    """

    def test_snapshot_019_minimal_discovery(self, snapshot_manager):
        """
        snapshot-019: Minimal discovery has correct structure.
        """
        discovery = Discovery(
            agent_id="worker-1",
            content="Found an important pattern"
        )

        snapshot_manager.assert_match("minimal_discovery", discovery.model_dump())

    def test_snapshot_020_full_discovery(self, snapshot_manager):
        """
        snapshot-020: Full discovery has correct structure.
        """
        discovery = Discovery(
            agent_id="worker-1",
            content="Database schema requires migration",
            tags=["database", "migration", "critical"],
            related_task="task-001"
        )

        snapshot_manager.assert_match("full_discovery", discovery.model_dump())


class TestProgressSnapshots:
    """
    Snapshot tests for progress tracking.
    """

    def test_snapshot_021_empty_progress(self, snapshot_manager):
        """
        snapshot-021: Empty progress has correct structure.
        """
        state = CoordinationState()
        progress = state.get_progress()

        snapshot_manager.assert_match("empty_progress", progress)

    def test_snapshot_022_partial_progress(self, snapshot_manager):
        """
        snapshot-022: Partial progress has correct structure.
        """
        state = CoordinationState()

        # Add tasks in various states
        for i in range(3):
            task = Task(description=f"Task {i}")
            if i == 0:
                task.complete(TaskResult(output="Done"))
            elif i == 1:
                task.claim("worker-1")
            state.add_task(task)

        progress = state.get_progress()

        snapshot_manager.assert_match("partial_progress", progress)

    def test_snapshot_023_complete_progress(self, snapshot_manager):
        """
        snapshot-023: Complete progress has correct structure.
        """
        state = CoordinationState()

        # All tasks completed
        for i in range(5):
            task = Task(description=f"Task {i}")
            task.complete(TaskResult(output=f"Completed {i}"))
            state.add_task(task)

        progress = state.get_progress()

        snapshot_manager.assert_match("complete_progress", progress)
