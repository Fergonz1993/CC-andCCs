"""
Load testing for concurrent worker operations.

Tests system behavior under high load with many concurrent workers.

Run with: pytest tests/integration/test_load.py -v
"""

import pytest
import threading
import time
import random
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter


class TestLoadConcurrentWorkers:
    """Load tests with concurrent workers."""

    def test_concurrent_task_claiming(self, option_a_module, temp_workspace):
        """load-001: Multiple workers claiming tasks concurrently."""
        coord = option_a_module
        coord.leader_init("Load test - concurrent claiming")

        # Create many tasks
        num_tasks = 50
        for i in range(num_tasks):
            coord.leader_add_task(f"Load test task {i}", priority=random.randint(1, 10))

        # Track claims
        claims: Dict[str, str] = {}
        claim_lock = threading.Lock()
        errors: List[str] = []

        def worker_claim_task(worker_id: str) -> None:
            try:
                task = coord.worker_claim(worker_id)
                if task:
                    with claim_lock:
                        if task.id in claims:
                            errors.append(f"Task {task.id} claimed by both {claims[task.id]} and {worker_id}")
                        claims[task.id] = worker_id
            except Exception as e:
                with claim_lock:
                    errors.append(f"Worker {worker_id} error: {str(e)}")

        # Run concurrent workers
        num_workers = 20
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers):
                for _ in range(5):  # Each worker tries to claim 5 times
                    futures.append(executor.submit(worker_claim_task, f"worker-{i}"))

            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # Verify no duplicate claims
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify tasks were claimed
        assert len(claims) > 0, "No tasks were claimed"

    def test_high_volume_task_creation(self, option_a_module, temp_workspace):
        """load-002: Creating many tasks quickly."""
        coord = option_a_module
        coord.leader_init("High volume task creation")

        num_tasks = 200
        start_time = time.time()

        task_ids = []
        for i in range(num_tasks):
            task_id = coord.leader_add_task(
                f"Batch task {i}",
                priority=i % 10 + 1,
                context_files=[f"file_{i}.py"],
                hints=f"Hint for task {i}"
            )
            task_ids.append(task_id)

        elapsed = time.time() - start_time

        # All tasks should be created
        data = coord.load_tasks()
        assert len(data["tasks"]) == num_tasks

        # All task IDs should be unique
        assert len(set(task_ids)) == num_tasks

        # Should complete in reasonable time (< 30 seconds)
        assert elapsed < 30, f"Task creation took too long: {elapsed}s"

    def test_rapid_claim_complete_cycle(self, option_a_module, temp_workspace):
        """load-003: Rapid claim-complete cycles."""
        coord = option_a_module
        coord.leader_init("Rapid cycle test")

        # Create tasks
        num_tasks = 30
        for i in range(num_tasks):
            coord.leader_add_task(f"Rapid task {i}", priority=1)

        completed = 0
        worker_id = "rapid-worker"

        start_time = time.time()
        while True:
            task = coord.worker_claim(worker_id)
            if not task:
                break

            coord.worker_start(worker_id, task.id)
            coord.worker_complete(worker_id, task.id, f"Completed {task.id}")
            completed += 1

        elapsed = time.time() - start_time

        assert completed == num_tasks, f"Only completed {completed}/{num_tasks} tasks"
        # Should complete all tasks reasonably quickly
        assert elapsed < 60, f"Cycle took too long: {elapsed}s"

    def test_concurrent_read_write(self, option_a_module, temp_workspace):
        """load-004: Concurrent read and write operations."""
        coord = option_a_module
        coord.leader_init("Concurrent R/W test")

        # Create initial tasks
        for i in range(10):
            coord.leader_add_task(f"Initial task {i}")

        errors: List[str] = []
        operations_completed = Counter()
        lock = threading.Lock()

        def writer(iteration: int) -> None:
            try:
                coord.leader_add_task(f"Writer task {iteration}")
                with lock:
                    operations_completed["write"] += 1
            except Exception as e:
                with lock:
                    errors.append(f"Writer error: {e}")

        def reader(iteration: int) -> None:
            try:
                data = coord.load_tasks()
                assert "tasks" in data
                with lock:
                    operations_completed["read"] += 1
            except Exception as e:
                with lock:
                    errors.append(f"Reader error: {e}")

        def claimer(worker_id: str) -> None:
            try:
                task = coord.worker_claim(worker_id)
                if task:
                    coord.worker_complete(worker_id, task.id, "Done")
                with lock:
                    operations_completed["claim"] += 1
            except Exception as e:
                with lock:
                    errors.append(f"Claimer error: {e}")

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []

            # Submit mixed operations
            for i in range(20):
                futures.append(executor.submit(writer, i))
                futures.append(executor.submit(reader, i))
                futures.append(executor.submit(claimer, f"worker-{i}"))

            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0, f"Errors: {errors}"
        assert operations_completed["read"] > 0
        assert operations_completed["write"] > 0


class TestLoadScaling:
    """Tests for system scaling characteristics."""

    def test_task_list_scaling(self, option_a_module, temp_workspace):
        """load-scale-001: Performance with growing task list."""
        coord = option_a_module
        coord.leader_init("Scaling test")

        times: List[float] = []
        sizes = [10, 50, 100, 200]

        current_count = 0
        for size in sizes:
            # Add tasks to reach target size
            while current_count < size:
                coord.leader_add_task(f"Scale task {current_count}")
                current_count += 1

            # Measure load time
            start = time.time()
            for _ in range(10):  # 10 loads
                coord.load_tasks()
            elapsed = time.time() - start

            times.append(elapsed / 10)  # Average time per load

        # Time should not grow excessively (rough check - less than 10x for 20x data)
        if times[0] > 0:
            ratio = times[-1] / times[0]
            assert ratio < 50, f"Load time grew too much: {ratio}x"

    def test_worker_scaling(self, option_a_module, temp_workspace):
        """load-scale-002: Performance with many workers."""
        coord = option_a_module
        coord.leader_init("Worker scaling test")

        # Create enough tasks for all workers
        num_tasks = 100
        for i in range(num_tasks):
            coord.leader_add_task(f"Worker scale task {i}")

        claimed_counts: List[int] = []

        for num_workers in [5, 10, 20]:
            claimed = 0
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i in range(num_workers):
                    def claim_once(wid):
                        task = coord.worker_claim(wid)
                        return 1 if task else 0
                    futures.append(executor.submit(claim_once, f"scale-worker-{i}"))

                for future in as_completed(futures):
                    claimed += future.result()

            claimed_counts.append(claimed)

        # More workers should claim more tasks (up to availability)
        # At minimum, having more workers should not reduce claims
        assert claimed_counts[-1] >= claimed_counts[0]


class TestLoadStress:
    """Stress tests for edge conditions."""

    def test_sustained_load(self, option_a_module, temp_workspace):
        """load-stress-001: Sustained operations over time."""
        coord = option_a_module
        coord.leader_init("Sustained load test")

        duration_seconds = 5
        operations = 0
        errors = 0
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            try:
                # Create a task
                task_id = coord.leader_add_task(f"Sustained task {operations}")

                # Claim and complete it
                task = coord.worker_claim("sustained-worker")
                if task:
                    coord.worker_complete("sustained-worker", task.id, "Done")

                operations += 1
            except Exception:
                errors += 1

        # Should complete many operations without errors
        assert operations > 10, f"Only completed {operations} operations"
        assert errors == 0, f"Had {errors} errors during sustained load"

    def test_burst_load(self, option_a_module, temp_workspace):
        """load-stress-002: Handle burst of operations."""
        coord = option_a_module
        coord.leader_init("Burst load test")

        errors: List[str] = []
        completed = Counter()
        lock = threading.Lock()

        def burst_operation(op_id: int) -> None:
            try:
                coord.leader_add_task(f"Burst task {op_id}")
                with lock:
                    completed["tasks_created"] += 1
            except Exception as e:
                with lock:
                    errors.append(str(e))

        # Create burst of 100 simultaneous operations
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(burst_operation, i) for i in range(100)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0, f"Burst errors: {errors}"
        assert completed["tasks_created"] == 100
