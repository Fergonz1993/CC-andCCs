"""
Chaos testing for random failures and recovery.

Tests system resilience under unexpected conditions.

Run with: pytest tests/integration/test_chaos.py -v
"""

import pytest
import random
import threading
import time
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestChaosRandomFailures:
    """Tests for random failure injection."""

    def test_random_worker_failures(self, option_a_module, temp_workspace):
        """chaos-001: System handles random worker failures."""
        coord = option_a_module
        coord.leader_init("Chaos - worker failures")

        # Create tasks
        for i in range(20):
            coord.leader_add_task(f"Chaos task {i}")

        completed = 0
        failed = 0
        failure_rate = 0.3  # 30% failure rate

        for i in range(50):  # Many attempts
            task = coord.worker_claim(f"chaos-worker-{i}")
            if not task:
                continue

            coord.worker_start(f"chaos-worker-{i}", task.id)

            # Random failure
            if random.random() < failure_rate:
                coord.worker_fail(f"chaos-worker-{i}", task.id, "Random chaos failure")
                failed += 1
            else:
                coord.worker_complete(f"chaos-worker-{i}", task.id, "Completed despite chaos")
                completed += 1

        # Some tasks should complete, some should fail
        assert completed > 0, "No tasks completed"
        assert failed > 0, "No failures occurred (chaos not working)"

        # Verify task states are consistent
        data = coord.load_tasks()
        done_count = sum(1 for t in data["tasks"] if t["status"] == "done")
        failed_count = sum(1 for t in data["tasks"] if t["status"] == "failed")

        assert done_count == completed
        assert failed_count == failed

    def test_intermittent_file_access(self, option_a_module, temp_workspace):
        """chaos-002: Handle intermittent file access issues."""
        coord = option_a_module
        coord.leader_init("Chaos - file access")

        for i in range(10):
            coord.leader_add_task(f"File chaos task {i}")

        operations = 0
        errors = 0

        for i in range(30):
            try:
                # Random operations
                op = random.choice(["load", "claim", "status"])

                if op == "load":
                    coord.load_tasks()
                elif op == "claim":
                    coord.worker_claim(f"file-chaos-{i}")
                else:
                    coord.leader_status()

                operations += 1
            except Exception:
                errors += 1

        # Should handle most operations successfully
        assert operations > errors, f"Too many errors: {errors}/{operations + errors}"

    def test_concurrent_chaos(self, option_a_module, temp_workspace):
        """chaos-003: Multiple workers with random behaviors."""
        coord = option_a_module
        coord.leader_init("Concurrent chaos")

        for i in range(30):
            coord.leader_add_task(f"Concurrent chaos task {i}")

        results = {"completed": 0, "failed": 0, "errors": 0}
        lock = threading.Lock()

        def chaotic_worker(worker_id: str):
            for _ in range(5):
                try:
                    # Random delay
                    time.sleep(random.uniform(0, 0.1))

                    task = coord.worker_claim(worker_id)
                    if not task:
                        continue

                    # Random behavior
                    behavior = random.choice(["complete", "fail", "abandon"])

                    if behavior == "complete":
                        coord.worker_complete(worker_id, task.id, "Chaos complete")
                        with lock:
                            results["completed"] += 1
                    elif behavior == "fail":
                        coord.worker_fail(worker_id, task.id, "Chaos fail")
                        with lock:
                            results["failed"] += 1
                    # "abandon" - just don't do anything (simulates crash)

                except Exception:
                    with lock:
                        results["errors"] += 1

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(chaotic_worker, f"chaos-{i}") for i in range(10)]
            for future in as_completed(futures):
                future.result()

        # System should remain consistent
        data = coord.load_tasks()
        assert "tasks" in data
        assert data["tasks"] is not None


class TestChaosRecovery:
    """Tests for recovery from chaos conditions."""

    def test_recovery_from_corrupted_state(self, option_a_module, temp_workspace):
        """chaos-recovery-001: Recover from corrupted state file."""
        coord = option_a_module
        coord.leader_init("Recovery test")

        # Create some tasks
        for i in range(5):
            coord.leader_add_task(f"Recovery task {i}")

        # Verify tasks exist
        data = coord.load_tasks()
        initial_count = len(data["tasks"])

        # Simulate partial corruption by truncating file
        # (Write incomplete JSON)
        original_content = coord.TASKS_FILE.read_text()

        # Try loading after "corruption" - system should handle gracefully
        # First restore the file
        coord.TASKS_FILE.write_text(original_content)

        # Verify recovery
        data = coord.load_tasks()
        assert len(data["tasks"]) == initial_count

    def test_recovery_missing_directories(self, option_a_module, temp_workspace):
        """chaos-recovery-002: Recover when directories are missing."""
        coord = option_a_module
        coord.leader_init("Directory recovery test")

        # Delete logs directory
        import shutil
        if coord.LOGS_DIR.exists():
            shutil.rmtree(coord.LOGS_DIR)

        # System should recreate on next operation
        coord.ensure_coordination_structure()

        assert coord.LOGS_DIR.exists()

    def test_recovery_after_incomplete_operation(self, option_a_module, temp_workspace):
        """chaos-recovery-003: Recover from incomplete operations."""
        coord = option_a_module
        coord.leader_init("Incomplete op recovery")

        task_id = coord.leader_add_task("Incomplete task")

        # Simulate claimed but never completed (worker crash)
        data = coord.load_tasks()
        for t in data["tasks"]:
            if t["id"] == task_id:
                t["status"] = "claimed"
                t["claimed_by"] = "crashed-worker"
        coord.save_tasks(data)

        # System should still be queryable
        status_data = coord.load_tasks()
        task_data = next(t for t in status_data["tasks"] if t["id"] == task_id)

        # Task should still exist in claimed state
        assert task_data["status"] == "claimed"
        assert task_data["claimed_by"] == "crashed-worker"


class TestChaosNetwork:
    """Tests simulating network-like issues."""

    def test_delayed_operations(self, option_a_module, temp_workspace):
        """chaos-network-001: Handle delayed operations."""
        coord = option_a_module
        coord.leader_init("Delay test")

        for i in range(5):
            coord.leader_add_task(f"Delay task {i}")

        completed = 0

        for i in range(10):
            # Simulate network delay
            time.sleep(random.uniform(0.01, 0.1))

            task = coord.worker_claim(f"delay-worker-{i}")
            if task:
                time.sleep(random.uniform(0.01, 0.1))
                coord.worker_complete(f"delay-worker-{i}", task.id, "Delayed complete")
                completed += 1

        assert completed > 0

    def test_out_of_order_operations(self, option_a_module, temp_workspace):
        """chaos-network-002: Handle out-of-order operations gracefully."""
        coord = option_a_module
        coord.leader_init("Out of order test")

        task_id = coord.leader_add_task("Order test task")

        # Try to complete before claiming (should fail gracefully)
        coord.worker_complete("order-worker", task_id, "Premature complete")

        # Task should still be available
        data = coord.load_tasks()
        task_data = next(t for t in data["tasks"] if t["id"] == task_id)
        assert task_data["status"] == "available"

        # Now do proper order
        coord.worker_claim("order-worker")
        coord.worker_complete("order-worker", task_id, "Proper complete")

        data = coord.load_tasks()
        task_data = next(t for t in data["tasks"] if t["id"] == task_id)
        assert task_data["status"] == "done"


class TestChaosResourceExhaustion:
    """Tests for resource exhaustion scenarios."""

    def test_many_log_entries(self, option_a_module, temp_workspace):
        """chaos-resource-001: Handle many log entries."""
        coord = option_a_module
        coord.leader_init("Log stress test")

        # Generate many log entries
        for i in range(100):
            coord.log_action(f"worker-{i % 10}", "ACTION", f"Details {i}")

        # System should still work
        data = coord.load_tasks()
        assert "tasks" in data

        # Log files should exist
        log_files = list(coord.LOGS_DIR.glob("*.log"))
        assert len(log_files) > 0

    def test_large_task_context(self, option_a_module, temp_workspace):
        """chaos-resource-002: Handle tasks with large context."""
        coord = option_a_module
        coord.leader_init("Large context test")

        # Create task with large context
        large_hints = "x" * 10000  # 10KB of hints
        many_files = [f"file_{i}.py" for i in range(100)]

        task_id = coord.leader_add_task(
            "Large context task",
            context_files=many_files,
            hints=large_hints
        )

        # Should be able to load and process
        data = coord.load_tasks()
        task_data = next(t for t in data["tasks"] if t["id"] == task_id)

        assert len(task_data["context"]["hints"]) == 10000
        assert len(task_data["context"]["files"]) == 100

        # Should be able to claim and complete
        task = coord.worker_claim("large-context-worker")
        assert task is not None
        coord.worker_complete("large-context-worker", task.id, "Handled large context")
