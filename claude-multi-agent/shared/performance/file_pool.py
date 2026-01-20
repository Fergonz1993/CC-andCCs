"""
Connection pooling for file operations.

Implements adv-perf-002: Connection pooling for file operations

Features:
- Reusable file handles for frequent operations
- Thread-safe file access
- Automatic handle cleanup
- Support for both sync and async I/O
"""

import asyncio
import os
import threading
import time
from typing import Optional, Dict, Any, BinaryIO, TextIO, Union
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import OrderedDict
import aiofiles
import fcntl


@dataclass
class FileHandle:
    """A pooled file handle with metadata."""
    path: str
    mode: str
    handle: Union[BinaryIO, TextIO]
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    is_locked: bool = False

    def touch(self) -> None:
        """Update last used time."""
        self.last_used = time.time()
        self.use_count += 1


class FilePool:
    """
    Synchronous file handle pool for efficient file operations.

    Maintains a pool of open file handles to reduce the overhead
    of repeatedly opening and closing files.

    Example:
        pool = FilePool(max_handles=50)

        with pool.get("tasks.json", "r") as f:
            data = json.load(f)

        with pool.get("tasks.json", "w") as f:
            json.dump(data, f)
    """

    def __init__(
        self,
        max_handles: int = 50,
        max_idle_time: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0  # 1 minute
    ):
        self.max_handles = max_handles
        self.max_idle_time = max_idle_time
        self.cleanup_interval = cleanup_interval

        self._handles: OrderedDict[str, FileHandle] = OrderedDict()
        self._lock = threading.RLock()
        self._handle_locks: Dict[str, threading.RLock] = {}
        self._last_cleanup = time.time()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def _make_key(self, path: str, mode: str) -> str:
        """Create a unique key for the handle."""
        return f"{os.path.abspath(path)}:{mode}"

    def _maybe_cleanup(self) -> None:
        """Periodically clean up idle handles."""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        self._last_cleanup = now
        expired = []

        for key, handle in self._handles.items():
            if not handle.is_locked and now - handle.last_used > self.max_idle_time:
                expired.append(key)

        for key in expired:
            self._close_handle(key)

    def _close_handle(self, key: str) -> None:
        """Close and remove a handle."""
        if key in self._handles:
            handle = self._handles[key]
            try:
                handle.handle.close()
            except Exception:
                pass
            del self._handles[key]
            self.stats["evictions"] += 1

    def _evict_lru(self) -> None:
        """Evict the least recently used handle."""
        # Find oldest unlocked handle
        for key, handle in list(self._handles.items()):
            if not handle.is_locked:
                self._close_handle(key)
                return

    @contextmanager
    def get(self, path: str, mode: str = "r"):
        """
        Get a file handle from the pool.

        Args:
            path: Path to the file
            mode: File mode (r, w, a, rb, wb, etc.)

        Yields:
            File handle
        """
        key = self._make_key(path, mode)

        with self._lock:
            self._maybe_cleanup()

            # Check if we have this handle
            if key in self._handles:
                handle = self._handles[key]
                handle.is_locked = True
                handle.touch()
                self._handles.move_to_end(key)
                self.stats["hits"] += 1
            else:
                # Need to open new handle
                self.stats["misses"] += 1

                # Evict if at capacity
                if len(self._handles) >= self.max_handles:
                    self._evict_lru()

                # Open new handle
                file_handle = open(path, mode)
                handle = FileHandle(
                    path=path,
                    mode=mode,
                    handle=file_handle,
                )
                handle.is_locked = True
                self._handles[key] = handle

            # Create lock for this handle if not exists
            if key not in self._handle_locks:
                self._handle_locks[key] = threading.RLock()

        # Use handle with its lock
        with self._handle_locks[key]:
            try:
                # For read modes, seek to beginning
                if "r" in mode and "w" not in mode:
                    handle.handle.seek(0)
                yield handle.handle
            finally:
                with self._lock:
                    handle.is_locked = False

                    # For write modes, flush immediately
                    if "w" in mode or "a" in mode:
                        handle.handle.flush()
                        # Keep write handles open for a short time
                        # but close them sooner than read handles
                        if handle.use_count > 5:
                            self._close_handle(key)

    def close_all(self) -> None:
        """Close all pooled handles."""
        with self._lock:
            for key in list(self._handles.keys()):
                self._close_handle(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                **self.stats,
                "open_handles": len(self._handles),
                "hit_rate": self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"]),
            }


class AsyncFilePool:
    """
    Asynchronous file handle pool for async/await file operations.

    Uses aiofiles for non-blocking file I/O.

    Example:
        pool = AsyncFilePool()

        async with pool.get("tasks.json", "r") as f:
            content = await f.read()
    """

    def __init__(
        self,
        max_handles: int = 50,
        max_idle_time: float = 300.0
    ):
        self.max_handles = max_handles
        self.max_idle_time = max_idle_time

        self._handles: OrderedDict[str, Any] = OrderedDict()
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()

        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def _make_key(self, path: str, mode: str) -> str:
        """Create a unique key for the handle."""
        return f"{os.path.abspath(path)}:{mode}"

    async def _maybe_cleanup(self) -> None:
        """Periodically clean up idle handles."""
        now = time.time()
        if now - self._last_cleanup < 60.0:
            return

        self._last_cleanup = now
        expired = []

        for key, (handle, metadata) in self._handles.items():
            if now - metadata["last_used"] > self.max_idle_time:
                expired.append(key)

        for key in expired:
            await self._close_handle(key)

    async def _close_handle(self, key: str) -> None:
        """Close and remove a handle."""
        if key in self._handles:
            handle, _ = self._handles[key]
            try:
                await handle.close()
            except Exception:
                pass
            del self._handles[key]
            self.stats["evictions"] += 1

    @contextmanager
    async def get(self, path: str, mode: str = "r"):
        """
        Get an async file handle from the pool.

        Note: Async context managers use `async with`

        Args:
            path: Path to the file
            mode: File mode

        Yields:
            Async file handle
        """
        key = self._make_key(path, mode)

        async with self._lock:
            await self._maybe_cleanup()

            if key in self._handles:
                handle, metadata = self._handles[key]
                metadata["last_used"] = time.time()
                metadata["use_count"] += 1
                self._handles.move_to_end(key)
                self.stats["hits"] += 1
            else:
                self.stats["misses"] += 1

                if len(self._handles) >= self.max_handles:
                    # Evict oldest
                    oldest_key = next(iter(self._handles))
                    await self._close_handle(oldest_key)

                handle = await aiofiles.open(path, mode)
                metadata = {
                    "created_at": time.time(),
                    "last_used": time.time(),
                    "use_count": 1,
                }
                self._handles[key] = (handle, metadata)

        try:
            if "r" in mode and "w" not in mode:
                await handle.seek(0)
            yield handle
        finally:
            if "w" in mode or "a" in mode:
                await handle.flush()

    async def close_all(self) -> None:
        """Close all pooled handles."""
        async with self._lock:
            for key in list(self._handles.keys()):
                await self._close_handle(key)


class LockedFileOperation:
    """
    Context manager for file operations with proper locking.

    Handles both read and write locks for safe concurrent access.

    Example:
        with LockedFileOperation("tasks.json") as op:
            data = op.read_json()
            data["tasks"].append(new_task)
            op.write_json(data)
    """

    def __init__(self, path: str, exclusive: bool = True):
        self.path = Path(path)
        self.exclusive = exclusive
        self._lock_file = None
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def __enter__(self) -> "LockedFileOperation":
        # Create lock file
        self._lock_path.touch(exist_ok=True)
        self._lock_file = open(self._lock_path, "r+")

        # Acquire lock
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        fcntl.flock(self._lock_file.fileno(), lock_type)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._lock_file:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()

    def read_text(self) -> str:
        """Read file as text."""
        return self.path.read_text() if self.path.exists() else ""

    def write_text(self, content: str) -> None:
        """Write text to file."""
        self.path.write_text(content)

    def read_json(self) -> Any:
        """Read file as JSON."""
        import json
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text())

    def write_json(self, data: Any, indent: int = 2) -> None:
        """Write data as JSON."""
        import json
        self.path.write_text(json.dumps(data, indent=indent, default=str))


# Singleton instances
_file_pool: Optional[FilePool] = None
_async_file_pool: Optional[AsyncFilePool] = None


def get_file_pool() -> FilePool:
    """Get or create the global file pool."""
    global _file_pool
    if _file_pool is None:
        _file_pool = FilePool()
    return _file_pool


def get_async_file_pool() -> AsyncFilePool:
    """Get or create the global async file pool."""
    global _async_file_pool
    if _async_file_pool is None:
        _async_file_pool = AsyncFilePool()
    return _async_file_pool
