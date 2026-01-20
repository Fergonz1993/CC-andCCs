"""
Performance optimization module for Claude Multi-Agent Coordination System.

This module provides:
- In-memory caching for task lookups (adv-perf-001)
- Connection pooling for file operations (adv-perf-002)
- Lazy loading of task results (adv-perf-003)
- Batch write operations (adv-perf-004)
- Index structures for fast task queries (adv-perf-005)
- Compression for large task data (adv-perf-006)
- Async I/O optimization (adv-perf-007)
- Memory-mapped file access (adv-perf-008)
- Query result caching with TTL (adv-perf-009)
- Profiling hooks for bottleneck detection (adv-perf-010)
"""

from .cache import TaskCache, TTLCache, QueryCache
from .indexing import TaskIndex
from .compression import compress_data, decompress_data, CompressedStorage
from .file_pool import FilePool, AsyncFilePool
from .batch import BatchWriter, BatchOperation
from .lazy import LazyTaskResult, LazyLoader
from .mmap_storage import MMapStorage
from .profiler import Profiler, profile, get_profiler
from .async_io import AsyncFileOps, AsyncTaskQueue, AsyncTaskProcessor, AsyncTaskFileOps

__all__ = [
    # Cache (adv-perf-001, adv-perf-009)
    "TaskCache",
    "TTLCache",
    "QueryCache",
    # Indexing (adv-perf-005)
    "TaskIndex",
    # Compression (adv-perf-006)
    "compress_data",
    "decompress_data",
    "CompressedStorage",
    # File Pool (adv-perf-002)
    "FilePool",
    "AsyncFilePool",
    # Batch Operations (adv-perf-004)
    "BatchWriter",
    "BatchOperation",
    # Lazy Loading (adv-perf-003)
    "LazyTaskResult",
    "LazyLoader",
    # Memory-mapped (adv-perf-008)
    "MMapStorage",
    # Profiler (adv-perf-010)
    "Profiler",
    "profile",
    "get_profiler",
    # Async I/O (adv-perf-007)
    "AsyncFileOps",
    "AsyncTaskQueue",
    "AsyncTaskProcessor",
    "AsyncTaskFileOps",
]
