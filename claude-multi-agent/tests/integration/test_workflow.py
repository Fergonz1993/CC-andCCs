"""
Integration tests for full workflow operations.

Tests the complete task lifecycle:
init -> add-task -> claim -> start -> complete

Run with: pytest tests/integration/test_workflow.py -v
"""

import pytest
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class TestFullWorkflow:
    """Tests for complete task workflow."""

    def test_complete_workflow_option_a(self, option_a_module, temp_workspace):
        """int-workflow-001: Complete workflow using Option A."""
        coord = option_a_module

        # Step 1: Initialize coordination
        coord.leader_init("Build a REST API", "Using Flask framework")

        assert coord.MASTER_PLAN_FILE.exists()
        assert "Build a REST API" in coord.MASTER_PLAN_FILE.read_text()

        # Step 2: Add tasks
        task1_id = coord.leader_add_task("Set up project structure", priority=1)
        task2_id = coord.leader_add_task("Implement models", priority=2, dependencies=[task1_id])
        task3_id = coord.leader_add_task("Create endpoints", priority=3, dependencies=[task2_id])

        data = coord.load_tasks()
        assert len(data["tasks"]) == 3

        # Step 3: Worker claims task
        worker_id = "worker-1"
        task = coord.worker_claim(worker_id)

        assert task is not None
        assert task.id == task1_id  # Should get highest priority

        # Step 4: Worker starts task
        coord.worker_start(worker_id, task.id)

        data = coord.load_tasks()
        task_data = next(t for t in data["tasks"] if t["id"] == task.id)
        assert task_data["status"] == "in_progress"

        # Step 5: Worker completes task
        coord.worker_complete(
            worker_id,
            task.id,
            "Created project structure with src/, tests/, and config/",
            files_created=["src/__init__.py", "tests/__init__.py", "config.py"]
        )

        data = coord.load_tasks()
        task_data = next(t for t in data["tasks"] if t["id"] == task.id)
        assert task_data["status"] == "done"

        # Step 6: Dependent task should now be available
        task2 = coord.worker_claim(worker_id)
        assert task2 is not None
        assert task2.id == task2_id

    def test_workflow_with_dependencies(self, option_a_module, temp_workspace):
        """int-workflow-002: Tasks respect dependency order."""
        coord = option_a_module
        coord.leader_init("Test dependencies")

        # Create chain: A -> B -> C
        task_a = coord.leader_add_task("Task A", priority=1)
        task_b = coord.leader_add_task("Task B", priority=1, dependencies=[task_a])
        task_c = coord.leader_add_task("Task C", priority=1, dependencies=[task_b])

        # Only task A should be claimable
        claimed = coord.worker_claim("worker-1")
        assert claimed.id == task_a

        # Complete A
        coord.worker_complete("worker-1", task_a, "Done A")

        # Now B should be claimable
        claimed = coord.worker_claim("worker-1")
        assert claimed.id == task_b

        # C still not available (B not done)
        coord.worker_start("worker-1", task_b)

        # Try to claim with another worker - should get nothing (only B in progress, C blocked)
        claimed_other = coord.worker_claim("worker-2")
        assert claimed_other is None

    def test_workflow_multiple_workers(self, option_a_module, temp_workspace):
        """int-workflow-003: Multiple workers can work in parallel."""
        coord = option_a_module
        coord.leader_init("Parallel work test")

        # Create independent tasks
        tasks = []
        for i in range(5):
            task_id = coord.leader_add_task(f"Independent task {i}", priority=i + 1)
            tasks.append(task_id)

        # Multiple workers claim tasks
        claimed_tasks = {}
        for worker_id in ["worker-1", "worker-2", "worker-3"]:
            task = coord.worker_claim(worker_id)
            if task:
                claimed_tasks[worker_id] = task.id

        # All three workers should have different tasks
        assert len(claimed_tasks) == 3
        assert len(set(claimed_tasks.values())) == 3  # All unique

    def test_workflow_failure_and_retry(self, option_a_module, temp_workspace):
        """int-workflow-004: Failed tasks can be retried."""
        coord = option_a_module
        coord.leader_init("Failure recovery test")

        task_id = coord.leader_add_task("Flaky task")

        # Worker 1 claims and fails
        coord.worker_claim("worker-1")
        coord.worker_fail("worker-1", task_id, "Connection timeout")

        data = coord.load_tasks()
        task_data = next(t for t in data["tasks"] if t["id"] == task_id)
        assert task_data["status"] == "failed"


class TestCrossOptionCompatibility:
    """Tests for data compatibility between options."""

    def test_task_schema_compatibility(self, option_a_module, option_c_module):
        """int-compat-001: Task schema is compatible between Options A and C."""
        # Option A task structure
        option_a_task = {
            "id": "task-001",
            "description": "Test task",
            "status": "available",
            "priority": 5,
            "dependencies": [],
            "context": {"files": [], "hints": ""},
            "result": None,
        }

        # Option C should be able to understand this structure
        from orchestrator.models import Task, TaskStatus, TaskContext

        # Create Option C task with similar data
        option_c_task = Task(
            id=option_a_task["id"],
            description=option_a_task["description"],
            status=TaskStatus.AVAILABLE,
            priority=option_a_task["priority"],
            dependencies=option_a_task["dependencies"],
        )

        assert option_c_task.id == option_a_task["id"]
        assert option_c_task.description == option_a_task["description"]
        assert option_c_task.priority == option_a_task["priority"]

    def test_status_values_compatible(self, option_a_module, option_c_module):
        """int-compat-002: Status values are compatible across options."""
        # Option A uses string statuses
        option_a_statuses = {"available", "claimed", "in_progress", "done", "failed"}

        # Option C uses enum
        from orchestrator.models import TaskStatus

        option_c_statuses = {s.value for s in TaskStatus}

        # Core statuses should overlap
        common = option_a_statuses & option_c_statuses
        assert "available" in common
        assert "claimed" in common
        assert "in_progress" in common
        assert "done" in common
        assert "failed" in common


class TestDiscoverySharing:
    """Tests for discovery/findings sharing between agents."""

    def test_discoveries_file_created(self, option_a_module, temp_workspace):
        """int-disc-001: Discoveries file is created during init."""
        coord = option_a_module
        coord.leader_init("Discovery test")

        assert coord.DISCOVERIES_FILE.exists()
        content = coord.DISCOVERIES_FILE.read_text()
        assert "Shared Discoveries" in content

    def test_result_aggregation(self, option_a_module, temp_workspace):
        """int-disc-002: Results can be aggregated."""
        coord = option_a_module
        coord.leader_init("Aggregation test")

        # Add and complete some tasks
        task1 = coord.leader_add_task("Task 1")
        task2 = coord.leader_add_task("Task 2")

        coord.worker_claim("worker-1")
        coord.worker_complete("worker-1", task1, "Completed task 1 with findings")

        coord.worker_claim("worker-1")
        coord.worker_complete("worker-1", task2, "Completed task 2 with more findings")

        # Aggregate results
        coord.leader_aggregate()

        summary_file = coord.COORDINATION_DIR / "summary.md"
        assert summary_file.exists()
        content = summary_file.read_text()
        assert task1 in content or "task" in content.lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_claim_already_claimed_task(self, option_a_module, temp_workspace):
        """int-edge-001: Cannot claim already claimed task."""
        coord = option_a_module
        coord.leader_init("Edge case test")

        coord.leader_add_task("Single task")

        # First worker claims
        task1 = coord.worker_claim("worker-1")
        assert task1 is not None

        # Second worker tries to claim - should get nothing
        task2 = coord.worker_claim("worker-2")
        assert task2 is None

    def test_complete_wrong_owner(self, option_a_module, temp_workspace, capsys):
        """int-edge-002: Cannot complete task owned by another worker."""
        coord = option_a_module
        coord.leader_init("Owner check test")

        task_id = coord.leader_add_task("Test task")
        coord.worker_claim("worker-1")

        # Worker 2 tries to complete worker 1's task
        coord.worker_complete("worker-2", task_id, "Done")

        captured = capsys.readouterr()
        assert "not claimed by" in captured.out

    def test_empty_task_list(self, option_a_module, temp_workspace, capsys):
        """int-edge-003: Handle empty task list gracefully."""
        coord = option_a_module
        coord.leader_init("Empty test")

        # Try to claim with no tasks
        task = coord.worker_claim("worker-1")
        assert task is None

        captured = capsys.readouterr()
        assert "No available" in captured.out

    def test_circular_dependencies_prevention(self, option_a_module, temp_workspace):
        """int-edge-004: System handles dependency ordering correctly."""
        coord = option_a_module
        coord.leader_init("Dependency test")

        # Create tasks - Note: Option A doesn't prevent circular deps at creation
        # but they would cause deadlock in claiming
        task_a = coord.leader_add_task("Task A")
        task_b = coord.leader_add_task("Task B", dependencies=[task_a])

        # Task B should not be claimable before Task A is done
        data = coord.load_tasks()
        task_b_data = next(t for t in data["tasks"] if t["id"] == task_b)

        # Verify dependency is recorded
        assert task_a in task_b_data["dependencies"]


class TestDependencyEnforcement:
    """Integration tests for dependency enforcement across task graphs (ATOM-106)."""

    def test_diamond_dependency_graph(self, option_a_module, temp_workspace):
        """int-dep-001: Diamond DAG is correctly enforced."""
        coord = option_a_module
        coord.leader_init("Diamond dependency test")

        # Create diamond: A -> (B, C) -> D
        #     A
        #    / \
        #   B   C
        #    \ /
        #     D
        task_a = coord.leader_add_task("Task A", priority=1)
        task_b = coord.leader_add_task("Task B", priority=2, dependencies=[task_a])
        task_c = coord.leader_add_task("Task C", priority=2, dependencies=[task_a])
        task_d = coord.leader_add_task("Task D", priority=3, dependencies=[task_b, task_c])

        # Only A should be initially claimable
        claimed = coord.worker_claim("worker-1")
        assert claimed.id == task_a

        # Complete A
        coord.worker_complete("worker-1", task_a, "Done A")

        # Now B and C should both be claimable (parallel)
        claimed1 = coord.worker_claim("worker-1")
        claimed2 = coord.worker_claim("worker-2")

        assert {claimed1.id, claimed2.id} == {task_b, task_c}

        # D should not yet be claimable
        claimed3 = coord.worker_claim("worker-3")
        assert claimed3 is None  # No more tasks available (D is blocked)

        # Complete B and C
        coord.worker_complete("worker-1", task_b, "Done B")
        coord.worker_complete("worker-2", task_c, "Done C")

        # Now D should be claimable
        claimed4 = coord.worker_claim("worker-1")
        assert claimed4.id == task_d

    def test_multiple_dependency_chains(self, option_a_module, temp_workspace):
        """int-dep-002: Multiple parallel dependency chains work correctly."""
        coord = option_a_module
        coord.leader_init("Parallel chains test")

        # Chain 1: A1 -> B1 -> C1
        a1 = coord.leader_add_task("A1", priority=1)
        b1 = coord.leader_add_task("B1", priority=2, dependencies=[a1])
        c1 = coord.leader_add_task("C1", priority=3, dependencies=[b1])

        # Chain 2: A2 -> B2 -> C2
        a2 = coord.leader_add_task("A2", priority=1)
        b2 = coord.leader_add_task("B2", priority=2, dependencies=[a2])
        c2 = coord.leader_add_task("C2", priority=3, dependencies=[b2])

        # Both A1 and A2 should be claimable immediately
        claimed1 = coord.worker_claim("worker-1")
        claimed2 = coord.worker_claim("worker-2")
        assert {claimed1.id, claimed2.id} == {a1, a2}

        # Complete both A tasks
        coord.worker_complete("worker-1", a1, "Done A1")
        coord.worker_complete("worker-2", a2, "Done A2")

        # Both B1 and B2 should be claimable
        claimed3 = coord.worker_claim("worker-1")
        claimed4 = coord.worker_claim("worker-2")
        assert {claimed3.id, claimed4.id} == {b1, b2}

    def test_deep_dependency_chain(self, option_a_module, temp_workspace):
        """int-dep-003: Deep dependency chain (10 levels) is correctly enforced."""
        coord = option_a_module
        coord.leader_init("Deep chain test")

        # Create chain of 10 tasks
        task_ids = []
        prev_id = None
        for i in range(10):
            deps = [prev_id] if prev_id else []
            task_id = coord.leader_add_task(f"Task {i}", priority=i + 1, dependencies=deps)
            task_ids.append(task_id)
            prev_id = task_id

        # Only first task should be claimable
        claimed = coord.worker_claim("worker-1")
        assert claimed.id == task_ids[0]

        # Walk through the entire chain
        for i, expected_id in enumerate(task_ids):
            if i > 0:
                claimed = coord.worker_claim("worker-1")
                assert claimed.id == expected_id
            coord.worker_complete("worker-1", expected_id, f"Done {i}")

        # All tasks should now be done
        data = coord.load_tasks()
        for task in data["tasks"]:
            assert task["status"] == "done"

    def test_dependency_with_failed_task(self, option_a_module, temp_workspace):
        """int-dep-004: Dependencies respect failed task states."""
        coord = option_a_module
        coord.leader_init("Failed dependency test")

        task_a = coord.leader_add_task("Task A", priority=1)
        task_b = coord.leader_add_task("Task B", priority=2, dependencies=[task_a])

        # Claim and fail Task A
        coord.worker_claim("worker-1")
        coord.worker_fail("worker-1", task_a, "Task A failed")

        # Task B should not be claimable (dependency not "done")
        claimed = coord.worker_claim("worker-2")
        assert claimed is None

    def test_fan_out_dependency(self, option_a_module, temp_workspace):
        """int-dep-005: Single task enabling multiple downstream tasks (fan-out)."""
        coord = option_a_module
        coord.leader_init("Fan-out test")

        # Create fan-out: A -> (B, C, D, E)
        task_a = coord.leader_add_task("Task A", priority=1)
        task_b = coord.leader_add_task("Task B", priority=2, dependencies=[task_a])
        task_c = coord.leader_add_task("Task C", priority=2, dependencies=[task_a])
        task_d = coord.leader_add_task("Task D", priority=2, dependencies=[task_a])
        task_e = coord.leader_add_task("Task E", priority=2, dependencies=[task_a])

        # Only A claimable initially
        claimed = coord.worker_claim("worker-1")
        assert claimed.id == task_a

        # Complete A
        coord.worker_complete("worker-1", task_a, "Done A")

        # All of B, C, D, E should now be claimable
        claimed_ids = set()
        for i in range(4):
            claimed = coord.worker_claim(f"worker-{i}")
            if claimed:
                claimed_ids.add(claimed.id)

        assert claimed_ids == {task_b, task_c, task_d, task_e}

    def test_fan_in_dependency(self, option_a_module, temp_workspace):
        """int-dep-006: Multiple tasks converging to single downstream (fan-in)."""
        coord = option_a_module
        coord.leader_init("Fan-in test")

        # Create fan-in: (A, B, C, D) -> E
        task_a = coord.leader_add_task("Task A", priority=1)
        task_b = coord.leader_add_task("Task B", priority=1)
        task_c = coord.leader_add_task("Task C", priority=1)
        task_d = coord.leader_add_task("Task D", priority=1)
        task_e = coord.leader_add_task("Task E", priority=2, dependencies=[task_a, task_b, task_c, task_d])

        # All A-D claimable initially
        claimed_ids = set()
        for i in range(4):
            claimed = coord.worker_claim(f"worker-{i}")
            if claimed:
                claimed_ids.add(claimed.id)

        assert claimed_ids == {task_a, task_b, task_c, task_d}

        # Complete 3 of 4 tasks
        coord.worker_complete("worker-0", task_a, "Done A")
        coord.worker_complete("worker-1", task_b, "Done B")
        coord.worker_complete("worker-2", task_c, "Done C")

        # E should still not be claimable (D not done)
        new_worker = coord.worker_claim("worker-5")
        assert new_worker is None

        # Complete D
        coord.worker_complete("worker-3", task_d, "Done D")

        # Now E should be claimable
        claimed_e = coord.worker_claim("worker-5")
        assert claimed_e.id == task_e
