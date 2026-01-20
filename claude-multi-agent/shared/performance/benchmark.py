#!/usr/bin/env python3
"""
Benchmark suite for performance features.

Measures the effectiveness of all performance optimizations:
- adv-perf-001: In-memory caching for task lookups
- adv-perf-002: Connection pooling for file operations
- adv-perf-003: Lazy loading of task results
- adv-perf-004: Batch write operations
- adv-perf-005: Index structures for fast task queries
- adv-perf-006: Compression for large task data
- adv-perf-007: Async I/O optimization
- adv-perf-008: Memory-mapped file access
- adv-perf-009: Query result caching with TTL
- adv-perf-010: Profiling hooks for bottleneck detection
"""

import asyncio
import json
import os
import random
import string
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any

# Import performance modules
from .cache import TaskCache, TTLCache, QueryCache
from .indexing import TaskIndex
from .compression import compress_data, decompress_data, CompressedStorage
from .file_pool import FilePool
from .batch import BatchWriter
from .lazy import LazyValue, LazyLoader
from .mmap_storage import MMapStorage
from .profiler import Profiler, profile, measure


def generate_task(task_id: str, num_deps: int = 0) -> Dict[str, Any]:
    """Generate a random task for benchmarking."""
    return {
        "id": task_id,
        "description": f"Task {task_id}: " + "".join(
            random.choices(string.ascii_letters + " ", k=100)
        ),
        "status": random.choice(["available", "claimed", "in_progress", "done", "failed"]),
        "priority": random.randint(1, 10),
        "claimed_by": f"worker-{random.randint(1, 5)}" if random.random() > 0.5 else None,
        "dependencies": [f"task-{i}" for i in range(num_deps)],
        "context": {
            "files": [f"file{i}.py" for i in range(random.randint(0, 5))],
            "hints": "Some helpful hints here",
        },
        "result": {
            "output": "".join(random.choices(string.ascii_letters, k=500)) if random.random() > 0.7 else None,
        } if random.random() > 0.5 else None,
        "created_at": "2024-01-01T00:00:00Z",
        "claimed_at": None,
        "completed_at": None,
    }


def generate_tasks(n: int) -> List[Dict[str, Any]]:
    """Generate N random tasks."""
    return [generate_task(f"task-{i}", num_deps=min(i, 3)) for i in range(n)]


class BenchmarkResult:
    """Result of a benchmark run."""

    def __init__(self, name: str):
        self.name = name
        self.durations: List[float] = []
        self.operations = 0

    def record(self, duration: float, operations: int = 1):
        self.durations.append(duration)
        self.operations += operations

    @property
    def total_time(self) -> float:
        return sum(self.durations)

    @property
    def avg_time(self) -> float:
        return self.total_time / len(self.durations) if self.durations else 0

    @property
    def ops_per_second(self) -> float:
        return self.operations / self.total_time if self.total_time > 0 else 0

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Total time: {self.total_time:.4f}s\n"
            f"  Avg time:   {self.avg_time:.6f}s\n"
            f"  Operations: {self.operations}\n"
            f"  Ops/sec:    {self.ops_per_second:.2f}"
        )


def benchmark_cache():
    """Benchmark the TaskCache implementation."""
    print("\n" + "=" * 60)
    print("Benchmarking: In-memory Cache (adv-perf-001)")
    print("=" * 60)

    cache = TaskCache[str, Dict](max_size=10000)
    tasks = generate_tasks(10000)

    # Benchmark cache writes
    result_write = BenchmarkResult("Cache Write")
    start = time.perf_counter()
    for task in tasks:
        cache.set(task["id"], task)
    result_write.record(time.perf_counter() - start, len(tasks))
    print(result_write)

    # Benchmark cache reads (hits)
    result_read = BenchmarkResult("Cache Read (hits)")
    start = time.perf_counter()
    for task in tasks:
        cache.get(task["id"])
    result_read.record(time.perf_counter() - start, len(tasks))
    print(result_read)

    # Benchmark cache reads (misses)
    result_miss = BenchmarkResult("Cache Read (misses)")
    start = time.perf_counter()
    for i in range(10000):
        cache.get(f"nonexistent-{i}")
    result_miss.record(time.perf_counter() - start, 10000)
    print(result_miss)

    print(f"\nCache Stats: {cache.stats}")


def benchmark_index():
    """Benchmark the TaskIndex implementation."""
    print("\n" + "=" * 60)
    print("Benchmarking: Task Index (adv-perf-005)")
    print("=" * 60)

    index = TaskIndex()
    tasks = generate_tasks(10000)

    # Half are done for dependency checking
    for i, task in enumerate(tasks):
        if i < 5000:
            task["status"] = "done"
        else:
            task["status"] = "available"

    # Benchmark index adds
    result_add = BenchmarkResult("Index Add")
    start = time.perf_counter()
    for task in tasks:
        index.add_task(task)
    result_add.record(time.perf_counter() - start, len(tasks))
    print(result_add)

    # Benchmark status lookups
    result_status = BenchmarkResult("Status Lookup")
    start = time.perf_counter()
    for _ in range(1000):
        index.get_by_status("available")
    result_status.record(time.perf_counter() - start, 1000)
    print(result_status)

    # Benchmark available with deps satisfied
    result_deps = BenchmarkResult("Available with Deps")
    start = time.perf_counter()
    for _ in range(1000):
        index.get_available_with_deps_satisfied()
    result_deps.record(time.perf_counter() - start, 1000)
    print(result_deps)

    # Benchmark highest priority
    result_priority = BenchmarkResult("Highest Priority")
    start = time.perf_counter()
    for _ in range(1000):
        index.get_highest_priority_available()
    result_priority.record(time.perf_counter() - start, 1000)
    print(result_priority)

    print(f"\nIndex Stats: {index.get_stats()}")


def benchmark_compression():
    """Benchmark the compression implementation."""
    print("\n" + "=" * 60)
    print("Benchmarking: Compression (adv-perf-006)")
    print("=" * 60)

    # Generate large data
    large_data = {
        "tasks": generate_tasks(1000),
        "metadata": {
            "description": "A" * 10000,
            "logs": ["Log entry " * 100 for _ in range(100)],
        },
    }

    original_size = len(json.dumps(large_data))
    print(f"Original data size: {original_size:,} bytes")

    # Benchmark compression
    result_compress = BenchmarkResult("Compress")
    start = time.perf_counter()
    for _ in range(100):
        compressed = compress_data(large_data)
    result_compress.record(time.perf_counter() - start, 100)
    print(result_compress)

    compressed = compress_data(large_data)
    if hasattr(compressed, "compressed_size"):
        print(f"Compressed size: {compressed.compressed_size:,} bytes")
        print(f"Compression ratio: {compressed.compression_ratio:.2%}")

    # Benchmark decompression
    result_decompress = BenchmarkResult("Decompress")
    start = time.perf_counter()
    for _ in range(100):
        decompress_data(compressed)
    result_decompress.record(time.perf_counter() - start, 100)
    print(result_decompress)


def benchmark_batch_writer():
    """Benchmark the batch writer implementation."""
    print("\n" + "=" * 60)
    print("Benchmarking: Batch Writer (adv-perf-004)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Without batching
        result_no_batch = BenchmarkResult("Without Batching")
        start = time.perf_counter()
        for i in range(1000):
            filepath = Path(tmpdir) / f"file_{i}.json"
            with open(filepath, "w") as f:
                json.dump({"id": i, "data": "test"}, f)
        result_no_batch.record(time.perf_counter() - start, 1000)
        print(result_no_batch)

        # With batching
        result_batch = BenchmarkResult("With Batching")
        writer = BatchWriter(max_batch_size=100, max_delay=0.1)
        start = time.perf_counter()
        for i in range(1000):
            filepath = str(Path(tmpdir) / f"batch_file_{i}.json")
            writer.write(filepath, {"id": i, "data": "test"})
        writer.flush()
        result_batch.record(time.perf_counter() - start, 1000)
        print(result_batch)

        print(f"\nWriter Stats: {writer.stats}")


def benchmark_lazy_loading():
    """Benchmark the lazy loading implementation."""
    print("\n" + "=" * 60)
    print("Benchmarking: Lazy Loading (adv-perf-003)")
    print("=" * 60)

    # Simulate expensive computation
    computation_time = 0.001  # 1ms

    def expensive_computation():
        time.sleep(computation_time)
        return {"result": "computed"}

    # Without lazy loading
    result_eager = BenchmarkResult("Eager Loading")
    start = time.perf_counter()
    values = []
    for _ in range(100):
        values.append(expensive_computation())
    # Only access 10
    for i in range(10):
        _ = values[i]
    result_eager.record(time.perf_counter() - start, 100)
    print(result_eager)

    # With lazy loading
    result_lazy = BenchmarkResult("Lazy Loading")
    start = time.perf_counter()
    lazy_values = [LazyValue(expensive_computation) for _ in range(100)]
    # Only access 10
    for i in range(10):
        _ = lazy_values[i].get()
    result_lazy.record(time.perf_counter() - start, 10)  # Only 10 were actually loaded
    print(result_lazy)

    print(f"Lazy loading saved: {(100 - 10) * computation_time:.3f}s of unnecessary computation")


def benchmark_query_cache():
    """Benchmark the query cache with TTL."""
    print("\n" + "=" * 60)
    print("Benchmarking: Query Cache with TTL (adv-perf-009)")
    print("=" * 60)

    cache = QueryCache(default_ttl=1.0, max_size=1000)

    # Simulate expensive query
    query_time = 0.001  # 1ms

    def expensive_query():
        time.sleep(query_time)
        return {"result": "data"}

    # Without caching
    result_no_cache = BenchmarkResult("Without Caching")
    start = time.perf_counter()
    for _ in range(100):
        expensive_query()
    result_no_cache.record(time.perf_counter() - start, 100)
    print(result_no_cache)

    # With caching
    result_cache = BenchmarkResult("With Caching")
    cache.set("query_key", expensive_query())
    start = time.perf_counter()
    for _ in range(100):
        cached = cache.get("query_key")
        if cached is None:
            cache.set("query_key", expensive_query())
    result_cache.record(time.perf_counter() - start, 100)
    print(result_cache)

    print(f"\nCache Stats: hit_rate={cache.stats.hit_rate:.2%}")


def benchmark_profiler():
    """Benchmark the profiler overhead."""
    print("\n" + "=" * 60)
    print("Benchmarking: Profiler Overhead (adv-perf-010)")
    print("=" * 60)

    profiler = Profiler(enabled=True)

    def simple_function(x):
        return x * 2

    # Without profiling
    result_no_profile = BenchmarkResult("Without Profiling")
    start = time.perf_counter()
    for i in range(100000):
        simple_function(i)
    result_no_profile.record(time.perf_counter() - start, 100000)
    print(result_no_profile)

    # With profiling
    @profiler.profile()
    def profiled_function(x):
        return x * 2

    result_profile = BenchmarkResult("With Profiling")
    start = time.perf_counter()
    for i in range(100000):
        profiled_function(i)
    result_profile.record(time.perf_counter() - start, 100000)
    print(result_profile)

    overhead = (result_profile.avg_time - result_no_profile.avg_time) / result_no_profile.avg_time * 100
    print(f"\nProfiler overhead: {overhead:.2f}%")


def benchmark_mmap_storage():
    """Benchmark memory-mapped storage."""
    print("\n" + "=" * 60)
    print("Benchmarking: Memory-Mapped Storage (adv-perf-008)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        mmap_path = Path(tmpdir) / "tasks.mmap"
        json_path = Path(tmpdir) / "tasks.json"

        tasks = generate_tasks(1000)

        # Regular JSON storage
        result_json_write = BenchmarkResult("JSON Write")
        start = time.perf_counter()
        for _ in range(10):
            with open(json_path, "w") as f:
                json.dump({"tasks": tasks}, f)
        result_json_write.record(time.perf_counter() - start, 10)
        print(result_json_write)

        result_json_read = BenchmarkResult("JSON Read")
        start = time.perf_counter()
        for _ in range(10):
            with open(json_path, "r") as f:
                json.load(f)
        result_json_read.record(time.perf_counter() - start, 10)
        print(result_json_read)

        # Memory-mapped storage
        storage = MMapStorage(str(mmap_path))
        storage.open()

        result_mmap_write = BenchmarkResult("MMap Write")
        start = time.perf_counter()
        for task in tasks:
            storage.store_task(task["id"], task)
        result_mmap_write.record(time.perf_counter() - start, len(tasks))
        print(result_mmap_write)

        result_mmap_read = BenchmarkResult("MMap Read")
        start = time.perf_counter()
        for task in tasks:
            storage.load_task(task["id"])
        result_mmap_read.record(time.perf_counter() - start, len(tasks))
        print(result_mmap_read)

        storage.close()

        print(f"\nMMap Stats: {storage.get_stats() if hasattr(storage, 'get_stats') else 'N/A'}")


def benchmark_file_pool():
    """Benchmark the file pool."""
    print("\n" + "=" * 60)
    print("Benchmarking: File Pool (adv-perf-002)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("Initial content\n")

        # Without pooling
        result_no_pool = BenchmarkResult("Without Pooling")
        start = time.perf_counter()
        for i in range(1000):
            with open(test_file, "r") as f:
                _ = f.read()
        result_no_pool.record(time.perf_counter() - start, 1000)
        print(result_no_pool)

        # With pooling
        pool = FilePool(max_handles=50)
        result_pool = BenchmarkResult("With Pooling")
        start = time.perf_counter()
        for i in range(1000):
            with pool.get(str(test_file), "r") as f:
                _ = f.read()
        result_pool.record(time.perf_counter() - start, 1000)
        print(result_pool)

        pool.close_all()
        print(f"\nPool Stats: {pool.get_stats()}")


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "=" * 80)
    print("PERFORMANCE FEATURE BENCHMARK SUITE")
    print("=" * 80)

    benchmarks = [
        ("Cache (adv-perf-001)", benchmark_cache),
        ("Index (adv-perf-005)", benchmark_index),
        ("Compression (adv-perf-006)", benchmark_compression),
        ("Batch Writer (adv-perf-004)", benchmark_batch_writer),
        ("Lazy Loading (adv-perf-003)", benchmark_lazy_loading),
        ("Query Cache TTL (adv-perf-009)", benchmark_query_cache),
        ("Profiler (adv-perf-010)", benchmark_profiler),
        ("MMap Storage (adv-perf-008)", benchmark_mmap_storage),
        ("File Pool (adv-perf-002)", benchmark_file_pool),
    ]

    for name, benchmark_fn in benchmarks:
        try:
            benchmark_fn()
        except Exception as e:
            print(f"\nError in {name}: {e}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_all_benchmarks()
