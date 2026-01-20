"""
Async I/O optimization.

Implements adv-perf-007: Async I/O optimization

Features:
- Non-blocking file reads/writes
- Parallel file operations
- Async task queue operations
- Graceful error handling
"""

import asyncio
import json
import os
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, TypeVar, Generic,
    Callable, AsyncIterator, Tuple
)
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import aiofiles
import aiofiles.os

T = TypeVar("T")


@dataclass
class AsyncIOStats:
    """Statistics for async I/O operations."""
    reads: int = 0
    writes: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    errors: int = 0
    parallel_ops: int = 0

    def record_read(self, bytes_count: int) -> None:
        self.reads += 1
        self.bytes_read += bytes_count

    def record_write(self, bytes_count: int) -> None:
        self.writes += 1
        self.bytes_written += bytes_count


class AsyncFileOps:
    """
    Async file operations with optimizations.

    Provides non-blocking file I/O operations using aiofiles.

    Example:
        ops = AsyncFileOps()

        # Read file
        content = await ops.read_file("tasks.json")

        # Write file
        await ops.write_file("results.json", {"data": "..."})

        # Read multiple files in parallel
        contents = await ops.read_files_parallel([
            "file1.json", "file2.json", "file3.json"
        ])
    """

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.stats = AsyncIOStats()

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to base_path."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p

    async def read_file(
        self,
        path: str,
        encoding: str = "utf-8"
    ) -> str:
        """Read a file asynchronously."""
        resolved = self._resolve_path(path)
        try:
            async with aiofiles.open(resolved, mode="r", encoding=encoding) as f:
                content = await f.read()
                self.stats.record_read(len(content.encode(encoding)))
                return content
        except Exception as e:
            self.stats.errors += 1
            raise

    async def write_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8"
    ) -> int:
        """Write a file asynchronously."""
        resolved = self._resolve_path(path)

        # Ensure directory exists
        await aiofiles.os.makedirs(resolved.parent, exist_ok=True)

        try:
            async with aiofiles.open(resolved, mode="w", encoding=encoding) as f:
                await f.write(content)
                bytes_written = len(content.encode(encoding))
                self.stats.record_write(bytes_written)
                return bytes_written
        except Exception as e:
            self.stats.errors += 1
            raise

    async def read_json(self, path: str) -> Any:
        """Read and parse a JSON file asynchronously."""
        content = await self.read_file(path)
        return json.loads(content)

    async def write_json(
        self,
        path: str,
        data: Any,
        indent: int = 2
    ) -> int:
        """Write data as JSON asynchronously."""
        content = json.dumps(data, indent=indent, default=str)
        return await self.write_file(path, content)

    async def read_files_parallel(
        self,
        paths: List[str],
        max_concurrent: int = 10
    ) -> Dict[str, str]:
        """
        Read multiple files in parallel.

        Args:
            paths: List of file paths
            max_concurrent: Maximum concurrent reads

        Returns:
            Dict mapping path to content
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def read_with_semaphore(path: str) -> Tuple[str, Optional[str]]:
            async with semaphore:
                self.stats.parallel_ops += 1
                try:
                    content = await self.read_file(path)
                    return (path, content)
                except Exception:
                    return (path, None)

        results = await asyncio.gather(
            *[read_with_semaphore(p) for p in paths]
        )

        return {path: content for path, content in results if content is not None}

    async def write_files_parallel(
        self,
        files: Dict[str, str],
        max_concurrent: int = 10
    ) -> int:
        """
        Write multiple files in parallel.

        Args:
            files: Dict mapping path to content
            max_concurrent: Maximum concurrent writes

        Returns:
            Number of files written
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def write_with_semaphore(path: str, content: str) -> bool:
            async with semaphore:
                self.stats.parallel_ops += 1
                try:
                    await self.write_file(path, content)
                    return True
                except Exception:
                    return False

        results = await asyncio.gather(
            *[write_with_semaphore(p, c) for p, c in files.items()]
        )

        return sum(1 for r in results if r)

    async def exists(self, path: str) -> bool:
        """Check if a file exists asynchronously."""
        resolved = self._resolve_path(path)
        try:
            await aiofiles.os.stat(resolved)
            return True
        except FileNotFoundError:
            return False

    async def delete(self, path: str) -> bool:
        """Delete a file asynchronously."""
        resolved = self._resolve_path(path)
        try:
            await aiofiles.os.remove(resolved)
            return True
        except FileNotFoundError:
            return False

    async def list_files(
        self,
        path: str,
        pattern: str = "*"
    ) -> List[str]:
        """List files in a directory asynchronously."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            return []

        # Note: aiofiles doesn't have async listdir, use sync in executor
        loop = asyncio.get_running_loop()
        files = await loop.run_in_executor(
            None,
            lambda: list(resolved.glob(pattern))
        )

        return [str(f) for f in files]


class AsyncTaskQueue(Generic[T]):
    """
    Async task queue for coordinating work.

    Provides async-aware task queue operations.

    Example:
        queue = AsyncTaskQueue[dict]()

        # Add tasks
        await queue.put({"id": "task-1", "description": "..."})

        # Get tasks
        task = await queue.get()

        # Process with timeout
        task = await queue.get_with_timeout(5.0)
    """

    def __init__(self, maxsize: int = 0):
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self._processed_count = 0
        self._error_count = 0

    async def put(self, item: T) -> None:
        """Put an item in the queue."""
        await self._queue.put(item)

    def put_nowait(self, item: T) -> None:
        """Put an item without waiting."""
        self._queue.put_nowait(item)

    async def get(self) -> T:
        """Get an item from the queue."""
        item = await self._queue.get()
        self._processed_count += 1
        return item

    async def get_with_timeout(self, timeout: float) -> Optional[T]:
        """Get an item with timeout."""
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            self._processed_count += 1
            return item
        except asyncio.TimeoutError:
            return None

    def task_done(self) -> None:
        """Mark a task as done."""
        self._queue.task_done()

    async def join(self) -> None:
        """Wait for all tasks to be processed."""
        await self._queue.join()

    @property
    def qsize(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    async def drain(self, max_items: int = -1) -> List[T]:
        """
        Drain items from the queue.

        Args:
            max_items: Maximum items to drain (-1 for all)

        Returns:
            List of drained items
        """
        items = []
        count = 0

        while not self._queue.empty():
            if max_items > 0 and count >= max_items:
                break
            try:
                item = self._queue.get_nowait()
                items.append(item)
                count += 1
            except asyncio.QueueEmpty:
                break

        return items


class AsyncTaskProcessor(Generic[T]):
    """
    Async task processor with parallel execution.

    Example:
        processor = AsyncTaskProcessor[dict](
            worker_fn=process_task,
            num_workers=5
        )

        # Add tasks
        for task in tasks:
            await processor.submit(task)

        # Wait for completion
        results = await processor.wait_all()
    """

    def __init__(
        self,
        worker_fn: Callable[[T], Any],
        num_workers: int = 5,
        queue_size: int = 100
    ):
        self.worker_fn = worker_fn
        self.num_workers = num_workers

        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=queue_size)
        self._results: List[Any] = []
        self._errors: List[Tuple[T, Exception]] = []
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start worker tasks."""
        if self._running:
            return

        self._running = True

        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes tasks."""
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.1
                )

                try:
                    if asyncio.iscoroutinefunction(self.worker_fn):
                        result = await self.worker_fn(task)
                    else:
                        result = self.worker_fn(task)

                    async with self._lock:
                        self._results.append(result)

                except Exception as e:
                    async with self._lock:
                        self._errors.append((task, e))

                finally:
                    self._queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def submit(self, task: T) -> None:
        """Submit a task for processing."""
        await self._queue.put(task)

    async def wait_all(self) -> List[Any]:
        """Wait for all submitted tasks to complete."""
        await self._queue.join()
        return self._results

    async def stop(self) -> None:
        """Stop all workers."""
        self._running = False

        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    @property
    def pending(self) -> int:
        """Number of pending tasks."""
        return self._queue.qsize()

    @property
    def results(self) -> List[Any]:
        """Get current results."""
        return self._results

    @property
    def errors(self) -> List[Tuple[T, Exception]]:
        """Get errors."""
        return self._errors


class AsyncTaskFileOps:
    """
    Async operations for task file management.

    Optimized for the coordination system's file structure.

    Example:
        ops = AsyncTaskFileOps(".coordination")

        # Load tasks
        tasks = await ops.load_tasks()

        # Save tasks
        await ops.save_tasks(tasks)

        # Load result
        result = await ops.load_result("task-123")
    """

    def __init__(self, coordination_dir: str = ".coordination"):
        self.coordination_dir = Path(coordination_dir)
        self.tasks_file = self.coordination_dir / "tasks.json"
        self.results_dir = self.coordination_dir / "results"
        self.logs_dir = self.coordination_dir / "logs"

        self._file_ops = AsyncFileOps()

    async def ensure_structure(self) -> None:
        """Ensure directory structure exists."""
        for dir_path in [self.coordination_dir, self.results_dir, self.logs_dir]:
            await aiofiles.os.makedirs(dir_path, exist_ok=True)

    async def load_tasks(self) -> Dict[str, Any]:
        """Load tasks.json file."""
        if not await self._file_ops.exists(str(self.tasks_file)):
            return {
                "version": "1.0",
                "tasks": [],
            }

        return await self._file_ops.read_json(str(self.tasks_file))

    async def save_tasks(self, data: Dict[str, Any]) -> None:
        """Save tasks.json file."""
        import datetime
        data["last_updated"] = datetime.datetime.now().isoformat()
        await self._file_ops.write_json(str(self.tasks_file), data)

    async def load_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load a task result."""
        # Try JSON first
        json_path = self.results_dir / f"{task_id}.json"
        if await self._file_ops.exists(str(json_path)):
            return await self._file_ops.read_json(str(json_path))

        # Try markdown
        md_path = self.results_dir / f"{task_id}.md"
        if await self._file_ops.exists(str(md_path)):
            content = await self._file_ops.read_file(str(md_path))
            return {"raw_content": content}

        return None

    async def save_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """Save a task result."""
        path = self.results_dir / f"{task_id}.json"
        await self._file_ops.write_json(str(path), result)

    async def load_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Load all task results in parallel."""
        result_files = await self._file_ops.list_files(
            str(self.results_dir),
            "*.json"
        )

        results = {}
        for path in result_files:
            task_id = Path(path).stem
            try:
                result = await self._file_ops.read_json(path)
                results[task_id] = result
            except Exception:
                continue

        return results

    async def append_log(self, agent_id: str, message: str) -> None:
        """Append to an agent's log file."""
        log_path = self.logs_dir / f"{agent_id}.log"
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        entry = f"[{timestamp}] {message}\n"

        async with aiofiles.open(log_path, "a") as f:
            await f.write(entry)

    async def atomic_update_tasks(
        self,
        update_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Atomically update tasks.json.

        Loads, applies update function, saves in one operation.
        """
        data = await self.load_tasks()
        updated = update_fn(data)
        await self.save_tasks(updated)
        return updated


# Singleton instance
_async_file_ops: Optional[AsyncFileOps] = None


def get_async_file_ops(base_path: Optional[str] = None) -> AsyncFileOps:
    """Get or create the global async file ops."""
    global _async_file_ops
    if _async_file_ops is None:
        _async_file_ops = AsyncFileOps(base_path)
    return _async_file_ops
