"""
Batch write operations.

Implements adv-perf-004: Batch write operations

Features:
- Batch multiple writes into single I/O operations
- Configurable flush triggers (count, size, time)
- Thread-safe batching
- Write coalescing for same file
"""

import asyncio
import json
import threading
import time
from typing import (
    Any, Callable, Dict, List, Optional, TypeVar, Generic,
    Union, Literal
)
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from collections import defaultdict

T = TypeVar("T")


class OperationType(Enum):
    """Types of batch operations."""
    WRITE = "write"
    APPEND = "append"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class BatchOperation:
    """A single operation in a batch."""
    operation_type: OperationType
    path: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "BatchOperation") -> bool:
        """Enable sorting by timestamp."""
        return self.timestamp < other.timestamp


class BatchWriter:
    """
    Batches write operations for efficiency.

    Collects writes and flushes them in batches, reducing
    I/O overhead for frequent small writes.

    Flush triggers:
    - Time: Flush after max_delay seconds
    - Count: Flush after max_batch_size operations
    - Size: Flush after max_bytes accumulated
    - Manual: Call flush() explicitly

    Example:
        writer = BatchWriter(max_batch_size=100, max_delay=1.0)

        writer.write("tasks.json", task_data)
        writer.write("logs/agent.log", log_entry)

        # Operations are batched and flushed automatically
        # Or flush manually:
        writer.flush()
    """

    def __init__(
        self,
        max_batch_size: int = 100,
        max_delay: float = 1.0,
        max_bytes: int = 1024 * 1024,  # 1MB
        write_callback: Optional[Callable[[str, bytes], None]] = None
    ):
        self.max_batch_size = max_batch_size
        self.max_delay = max_delay
        self.max_bytes = max_bytes
        self.write_callback = write_callback or self._default_write

        self._queue: List[BatchOperation] = []
        self._lock = threading.RLock()
        self._bytes_queued = 0
        self._first_queued_at: Optional[float] = None
        self._flush_timer: Optional[threading.Timer] = None

        # Statistics
        self.stats = {
            "operations_batched": 0,
            "batches_flushed": 0,
            "bytes_written": 0,
            "time_saved_estimate": 0.0,
        }

    def _default_write(self, path: str, data: bytes) -> None:
        """Default write implementation."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def _estimate_size(self, data: Any) -> int:
        """Estimate the size of data in bytes."""
        if isinstance(data, bytes):
            return len(data)
        elif isinstance(data, str):
            return len(data.encode("utf-8"))
        else:
            return len(json.dumps(data, default=str).encode("utf-8"))

    def _schedule_flush(self) -> None:
        """Schedule a delayed flush."""
        if self._flush_timer is not None:
            return

        self._flush_timer = threading.Timer(self.max_delay, self.flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _cancel_flush_timer(self) -> None:
        """Cancel any pending flush timer."""
        if self._flush_timer:
            self._flush_timer.cancel()
            self._flush_timer = None

    def write(
        self,
        path: str,
        data: Any,
        operation_type: OperationType = OperationType.WRITE
    ) -> None:
        """
        Add a write operation to the batch.

        Args:
            path: File path to write to
            data: Data to write (str, bytes, or JSON-serializable)
            operation_type: Type of operation (write, append, update)
        """
        with self._lock:
            operation = BatchOperation(
                operation_type=operation_type,
                path=path,
                data=data,
            )

            self._queue.append(operation)
            self._bytes_queued += self._estimate_size(data)
            self.stats["operations_batched"] += 1

            if self._first_queued_at is None:
                self._first_queued_at = time.time()
                self._schedule_flush()

            # Check flush triggers
            if self._should_flush():
                self.flush()

    def append(self, path: str, data: Any) -> None:
        """Add an append operation to the batch."""
        self.write(path, data, OperationType.APPEND)

    def _should_flush(self) -> bool:
        """Check if we should flush based on triggers."""
        if len(self._queue) >= self.max_batch_size:
            return True
        if self._bytes_queued >= self.max_bytes:
            return True
        if self._first_queued_at and (time.time() - self._first_queued_at) >= self.max_delay:
            return True
        return False

    def flush(self) -> int:
        """
        Flush all pending operations.

        Returns:
            Number of operations flushed
        """
        with self._lock:
            self._cancel_flush_timer()

            if not self._queue:
                return 0

            # Group operations by path for coalescing
            operations_by_path: Dict[str, List[BatchOperation]] = defaultdict(list)
            for op in self._queue:
                operations_by_path[op.path].append(op)

            count = len(self._queue)
            bytes_written = 0

            for path, ops in operations_by_path.items():
                try:
                    data = self._coalesce_operations(ops)
                    if data is not None:
                        data_bytes = data if isinstance(data, bytes) else data.encode("utf-8")
                        self.write_callback(path, data_bytes)
                        bytes_written += len(data_bytes)
                except Exception as e:
                    # Log error but continue with other paths
                    print(f"Error writing to {path}: {e}")

            # Update stats
            self.stats["batches_flushed"] += 1
            self.stats["bytes_written"] += bytes_written

            # Estimate time saved (assume ~1ms per operation)
            self.stats["time_saved_estimate"] += (count - len(operations_by_path)) * 0.001

            # Clear queue
            self._queue.clear()
            self._bytes_queued = 0
            self._first_queued_at = None

            return count

    def _coalesce_operations(
        self,
        operations: List[BatchOperation]
    ) -> Optional[Union[str, bytes]]:
        """
        Coalesce multiple operations on the same file.

        For writes, only the last write wins.
        For appends, all appends are combined.
        """
        if not operations:
            return None

        # Sort by timestamp
        operations.sort()

        # Check if any are write operations (not append)
        writes = [op for op in operations if op.operation_type == OperationType.WRITE]
        appends = [op for op in operations if op.operation_type == OperationType.APPEND]

        if writes:
            # Last write wins, then append anything after
            base_data = writes[-1].data
            last_write_time = writes[-1].timestamp

            # Get appends after the last write
            appends_after = [
                op for op in appends
                if op.timestamp > last_write_time
            ]

            if appends_after:
                # Combine with appends
                if isinstance(base_data, bytes):
                    result = base_data
                    for op in appends_after:
                        if isinstance(op.data, bytes):
                            result += op.data
                        else:
                            result += str(op.data).encode("utf-8")
                    return result
                else:
                    result = str(base_data)
                    for op in appends_after:
                        result += str(op.data)
                    return result

            # Just return the base data
            if isinstance(base_data, bytes):
                return base_data
            return json.dumps(base_data, indent=2, default=str)

        elif appends:
            # Combine all appends
            result = ""
            for op in appends:
                if isinstance(op.data, str):
                    result += op.data
                elif isinstance(op.data, bytes):
                    result += op.data.decode("utf-8", errors="replace")
                else:
                    result += json.dumps(op.data, default=str) + "\n"
            return result

        return None

    def close(self) -> None:
        """Flush remaining operations and clean up."""
        self.flush()
        self._cancel_flush_timer()

    def __enter__(self) -> "BatchWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncBatchWriter:
    """
    Async version of BatchWriter for asyncio-based applications.

    Example:
        async with AsyncBatchWriter() as writer:
            await writer.write("tasks.json", data)
    """

    def __init__(
        self,
        max_batch_size: int = 100,
        max_delay: float = 1.0,
        max_bytes: int = 1024 * 1024
    ):
        self.max_batch_size = max_batch_size
        self.max_delay = max_delay
        self.max_bytes = max_bytes

        self._queue: List[BatchOperation] = []
        self._lock = asyncio.Lock()
        self._bytes_queued = 0
        self._first_queued_at: Optional[float] = None
        self._flush_task: Optional[asyncio.Task] = None

    async def _schedule_flush(self) -> None:
        """Schedule a delayed flush."""
        if self._flush_task is not None:
            return

        async def delayed_flush():
            await asyncio.sleep(self.max_delay)
            await self.flush()

        self._flush_task = asyncio.create_task(delayed_flush())

    async def write(
        self,
        path: str,
        data: Any,
        operation_type: OperationType = OperationType.WRITE
    ) -> None:
        """Add a write operation to the batch."""
        async with self._lock:
            operation = BatchOperation(
                operation_type=operation_type,
                path=path,
                data=data,
            )

            self._queue.append(operation)

            if isinstance(data, bytes):
                self._bytes_queued += len(data)
            elif isinstance(data, str):
                self._bytes_queued += len(data.encode("utf-8"))
            else:
                self._bytes_queued += len(json.dumps(data, default=str).encode("utf-8"))

            if self._first_queued_at is None:
                self._first_queued_at = time.time()
                await self._schedule_flush()

            if self._should_flush():
                await self.flush()

    def _should_flush(self) -> bool:
        """Check if we should flush."""
        if len(self._queue) >= self.max_batch_size:
            return True
        if self._bytes_queued >= self.max_bytes:
            return True
        if self._first_queued_at and (time.time() - self._first_queued_at) >= self.max_delay:
            return True
        return False

    async def flush(self) -> int:
        """Flush all pending operations."""
        async with self._lock:
            if self._flush_task:
                self._flush_task.cancel()
                self._flush_task = None

            if not self._queue:
                return 0

            # Group by path
            operations_by_path: Dict[str, List[BatchOperation]] = defaultdict(list)
            for op in self._queue:
                operations_by_path[op.path].append(op)

            count = len(self._queue)

            # Write each file
            import aiofiles

            for path, ops in operations_by_path.items():
                data = self._coalesce_operations(ops)
                if data is not None:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(path, "w") as f:
                        if isinstance(data, bytes):
                            await f.write(data.decode("utf-8"))
                        else:
                            await f.write(data)

            self._queue.clear()
            self._bytes_queued = 0
            self._first_queued_at = None

            return count

    def _coalesce_operations(
        self,
        operations: List[BatchOperation]
    ) -> Optional[str]:
        """Coalesce operations (same as sync version)."""
        if not operations:
            return None

        operations.sort()

        writes = [op for op in operations if op.operation_type == OperationType.WRITE]

        if writes:
            data = writes[-1].data
            if isinstance(data, (dict, list)):
                return json.dumps(data, indent=2, default=str)
            return str(data)

        # Combine appends
        result = ""
        for op in operations:
            if isinstance(op.data, str):
                result += op.data
            else:
                result += json.dumps(op.data, default=str) + "\n"
        return result

    async def close(self) -> None:
        """Flush and clean up."""
        await self.flush()

    async def __aenter__(self) -> "AsyncBatchWriter":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class TaskBatchWriter(BatchWriter):
    """
    Specialized batch writer for task operations.

    Handles common patterns like:
    - Batching task status updates
    - Coalescing multiple updates to the same task
    - Maintaining tasks.json consistency
    """

    def __init__(
        self,
        tasks_file: str = "tasks.json",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tasks_file = tasks_file
        self._task_updates: Dict[str, Dict[str, Any]] = {}

    def update_task(self, task_id: str, updates: Dict[str, Any]) -> None:
        """
        Batch a task update.

        Multiple updates to the same task are coalesced.
        """
        with self._lock:
            if task_id not in self._task_updates:
                self._task_updates[task_id] = {}
            self._task_updates[task_id].update(updates)

            if self._first_queued_at is None:
                self._first_queued_at = time.time()
                self._schedule_flush()

    def flush_tasks(self) -> int:
        """Flush pending task updates."""
        with self._lock:
            if not self._task_updates:
                return 0

            # Read current tasks
            try:
                with open(self.tasks_file, "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                data = {"tasks": []}

            # Apply updates
            task_map = {t["id"]: t for t in data.get("tasks", [])}
            for task_id, updates in self._task_updates.items():
                if task_id in task_map:
                    task_map[task_id].update(updates)

            data["tasks"] = list(task_map.values())
            data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Write back
            with open(self.tasks_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            count = len(self._task_updates)
            self._task_updates.clear()
            return count

    def flush(self) -> int:
        """Flush both regular operations and task updates."""
        task_count = self.flush_tasks()
        op_count = super().flush()
        return task_count + op_count
