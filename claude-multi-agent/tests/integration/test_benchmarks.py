"""
Performance benchmarks for the Claude Multi-Agent Coordination System.

Measures execution times for critical operations.

Run with: pytest tests/integration/test_benchmarks.py -v --benchmark-enable

Note: Requires pytest-benchmark (pip install pytest-benchmark)
"""

import pytest
import time
import statistics
from typing import List, Dict, Any, Callable
from pathlib import Path

# Try to import pytest-benchmark
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


class BenchmarkResult:
    """Simple benchmark result container."""

    def __init__(self, name: str, times: List[float]):
        self.name = name
        self.times = times
        self.mean = statistics.mean(times) if times else 0
        self.median = statistics.median(times) if times else 0
        self.stdev = statistics.stdev(times) if len(times) > 1 else 0
        self.min = min(times) if times else 0
        self.max = max(times) if times else 0

    def __str__(self) -> str:
        return (
            f"{self.name}: mean={self.mean*1000:.2f}ms, "
            f"median={self.median*1000:.2f}ms, "
            f"stdev={self.stdev*1000:.2f}ms, "
            f"min={self.min*1000:.2f}ms, max={self.max*1000:.2f}ms"
        )


def simple_benchmark(func: Callable, iterations: int = 100) -> BenchmarkResult:
    """Run a simple benchmark."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return BenchmarkResult(func.__name__, times)


class TestBenchmarkTaskOperations:
    """Benchmarks for task operations."""

    def test_benchmark_task_creation(self, option_a_module, temp_workspace):
        """bench-task-001: Benchmark task creation speed."""
        coord = option_a_module
        coord.leader_init("Benchmark - task creation")

        def create_task():
            coord.leader_add_task("Benchmark task", priority=5)

        result = simple_benchmark(create_task, iterations=100)

        print(f"\n{result}")

        # Assert reasonable performance (< 50ms per task)
        assert result.mean < 0.05, f"Task creation too slow: {result.mean*1000:.2f}ms"

    def test_benchmark_task_loading(self, option_a_module, temp_workspace):
        """bench-task-002: Benchmark task loading speed."""
        coord = option_a_module
        coord.leader_init("Benchmark - task loading")

        # Create some tasks first
        for i in range(50):
            coord.leader_add_task(f"Benchmark task {i}")

        def load_tasks():
            coord.load_tasks()

        result = simple_benchmark(load_tasks, iterations=100)

        print(f"\n{result}")

        # Assert reasonable performance (< 20ms per load)
        assert result.mean < 0.02, f"Task loading too slow: {result.mean*1000:.2f}ms"

    def test_benchmark_task_claiming(self, option_a_module, temp_workspace):
        """bench-task-003: Benchmark task claiming speed."""
        coord = option_a_module
        coord.leader_init("Benchmark - task claiming")

        # Create many tasks
        for i in range(200):
            coord.leader_add_task(f"Claim benchmark task {i}")

        claim_times = []
        worker_id = 0

        while True:
            start = time.perf_counter()
            task = coord.worker_claim(f"bench-worker-{worker_id}")
            elapsed = time.perf_counter() - start

            if not task:
                break

            claim_times.append(elapsed)
            worker_id += 1

        result = BenchmarkResult("claim_task", claim_times)
        print(f"\n{result}")

        # Assert reasonable performance (< 50ms per claim)
        assert result.mean < 0.05, f"Task claiming too slow: {result.mean*1000:.2f}ms"

    def test_benchmark_task_completion(self, option_a_module, temp_workspace):
        """bench-task-004: Benchmark task completion speed."""
        coord = option_a_module
        coord.leader_init("Benchmark - task completion")

        # Create and claim tasks
        task_ids = []
        for i in range(100):
            task_id = coord.leader_add_task(f"Complete benchmark task {i}")
            task_ids.append(task_id)

        complete_times = []

        for i, task_id in enumerate(task_ids):
            worker_id = f"complete-worker-{i}"
            coord.worker_claim(worker_id)

            start = time.perf_counter()
            coord.worker_complete(worker_id, task_id, "Benchmark complete")
            elapsed = time.perf_counter() - start

            complete_times.append(elapsed)

        result = BenchmarkResult("complete_task", complete_times)
        print(f"\n{result}")

        # Assert reasonable performance (< 50ms per completion)
        assert result.mean < 0.05, f"Task completion too slow: {result.mean*1000:.2f}ms"


class TestBenchmarkScaling:
    """Benchmarks for scaling characteristics."""

    def test_benchmark_scaling_task_count(self, option_a_module, temp_workspace):
        """bench-scale-001: Measure how operations scale with task count."""
        coord = option_a_module
        coord.leader_init("Benchmark - scaling")

        results: Dict[int, Dict[str, float]] = {}

        for task_count in [10, 50, 100, 200]:
            # Reset state
            data = {"version": "1.0", "tasks": [], "created_at": "", "last_updated": ""}
            coord.save_tasks(data)

            # Add tasks
            for i in range(task_count):
                coord.leader_add_task(f"Scale task {i}")

            # Measure load time
            load_times = []
            for _ in range(20):
                start = time.perf_counter()
                coord.load_tasks()
                load_times.append(time.perf_counter() - start)

            # Measure claim time
            claim_times = []
            for i in range(min(20, task_count)):
                start = time.perf_counter()
                coord.worker_claim(f"scale-worker-{i}")
                claim_times.append(time.perf_counter() - start)

            results[task_count] = {
                "load_mean": statistics.mean(load_times),
                "claim_mean": statistics.mean(claim_times),
            }

            print(f"\nTask count {task_count}:")
            print(f"  Load: {results[task_count]['load_mean']*1000:.2f}ms")
            print(f"  Claim: {results[task_count]['claim_mean']*1000:.2f}ms")

        # Check that scaling is reasonable (not exponential)
        # Load time at 200 tasks should be < 10x load time at 10 tasks
        ratio = results[200]["load_mean"] / results[10]["load_mean"]
        assert ratio < 20, f"Load time scaling too steep: {ratio:.1f}x"

    def test_benchmark_concurrent_operations(self, option_a_module, temp_workspace):
        """bench-scale-002: Measure concurrent operation throughput."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        coord = option_a_module
        coord.leader_init("Benchmark - concurrent")

        # Create tasks
        for i in range(100):
            coord.leader_add_task(f"Concurrent task {i}")

        operations_completed = {"count": 0}
        lock = threading.Lock()

        duration = 3  # seconds
        start_time = time.time()

        def worker_loop(worker_id: int):
            while time.time() - start_time < duration:
                task = coord.worker_claim(f"concurrent-{worker_id}")
                if task:
                    coord.worker_complete(f"concurrent-{worker_id}", task.id, "Done")
                    with lock:
                        operations_completed["count"] += 1
                else:
                    time.sleep(0.01)  # Brief sleep if no tasks

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_loop, i) for i in range(10)]
            for f in futures:
                f.result()

        throughput = operations_completed["count"] / duration
        print(f"\nConcurrent throughput: {throughput:.1f} ops/sec")

        # Should achieve reasonable throughput
        assert throughput > 1, f"Throughput too low: {throughput:.1f} ops/sec"


class TestBenchmarkMemory:
    """Benchmarks for memory usage patterns."""

    def test_benchmark_large_task_list(self, option_a_module, temp_workspace):
        """bench-mem-001: Handle large task lists efficiently."""
        import sys

        coord = option_a_module
        coord.leader_init("Benchmark - memory")

        # Create many tasks
        task_count = 500
        for i in range(task_count):
            coord.leader_add_task(
                f"Memory test task {i}",
                priority=i % 10 + 1,
                context_files=[f"file_{i}.py"],
                hints=f"Hints for task {i}"
            )

        # Load and measure
        start = time.perf_counter()
        data = coord.load_tasks()
        load_time = time.perf_counter() - start

        # Check data integrity
        assert len(data["tasks"]) == task_count

        # Estimate memory (rough)
        import json
        json_str = json.dumps(data)
        size_kb = len(json_str) / 1024

        print(f"\nLarge task list ({task_count} tasks):")
        print(f"  Load time: {load_time*1000:.2f}ms")
        print(f"  JSON size: {size_kb:.1f}KB")

        # Assert reasonable size (< 1MB for 500 tasks)
        assert size_kb < 1024, f"State too large: {size_kb:.1f}KB"

    def test_benchmark_repeated_operations(self, option_a_module, temp_workspace):
        """bench-mem-002: Repeated operations don't leak memory."""
        coord = option_a_module
        coord.leader_init("Benchmark - repeated ops")

        # Perform many operations
        iterations = 100

        start = time.perf_counter()

        for i in range(iterations):
            task_id = coord.leader_add_task(f"Repeated task {i}")
            task = coord.worker_claim(f"repeat-worker-{i}")
            if task:
                coord.worker_complete(f"repeat-worker-{i}", task.id, "Done")

            # Periodically load state
            if i % 10 == 0:
                coord.load_tasks()

        elapsed = time.perf_counter() - start

        # Final state check
        data = coord.load_tasks()
        done_count = sum(1 for t in data["tasks"] if t["status"] == "done")

        print(f"\nRepeated operations ({iterations} iterations):")
        print(f"  Total time: {elapsed*1000:.2f}ms")
        print(f"  Completed tasks: {done_count}")

        # All tasks should be done
        assert done_count == iterations


class TestBenchmarkFileIO:
    """Benchmarks for file I/O operations."""

    def test_benchmark_file_locking(self, option_a_module, temp_workspace):
        """bench-io-001: File locking overhead."""
        coord = option_a_module
        coord.leader_init("Benchmark - file locking")

        # Create test file
        test_file = temp_workspace / "lock_test.json"
        test_file.write_text('{"test": true}')

        # Measure lock acquisition
        lock_times = []
        for _ in range(100):
            start = time.perf_counter()
            with coord.file_lock(test_file):
                pass  # Just acquire and release
            lock_times.append(time.perf_counter() - start)

        result = BenchmarkResult("file_lock", lock_times)
        print(f"\n{result}")

        # Lock overhead should be minimal (< 10ms)
        assert result.mean < 0.01, f"Lock overhead too high: {result.mean*1000:.2f}ms"

    def test_benchmark_json_serialization(self, option_a_module, temp_workspace):
        """bench-io-002: JSON serialization/deserialization speed."""
        import json

        coord = option_a_module
        coord.leader_init("Benchmark - JSON")

        # Create sample data
        for i in range(100):
            coord.leader_add_task(f"JSON task {i}")

        data = coord.load_tasks()

        # Serialize benchmark
        serialize_times = []
        for _ in range(100):
            start = time.perf_counter()
            json_str = json.dumps(data)
            serialize_times.append(time.perf_counter() - start)

        # Deserialize benchmark
        deserialize_times = []
        for _ in range(100):
            start = time.perf_counter()
            json.loads(json_str)
            deserialize_times.append(time.perf_counter() - start)

        serialize_result = BenchmarkResult("json_serialize", serialize_times)
        deserialize_result = BenchmarkResult("json_deserialize", deserialize_times)

        print(f"\n{serialize_result}")
        print(f"{deserialize_result}")

        # Both should be fast (< 5ms)
        assert serialize_result.mean < 0.005
        assert deserialize_result.mean < 0.005
