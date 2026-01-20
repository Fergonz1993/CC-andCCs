"""
Memory-mapped file access for efficient large file operations.

Implements adv-perf-008: Memory-mapped file access

Features:
- Memory-mapped access for large files
- Efficient random access without loading entire file
- Automatic handling of file growth
- Thread-safe operations
"""

import mmap
import os
import threading
import json
from typing import Optional, Any, Dict, Iterator, Tuple
from pathlib import Path
from dataclasses import dataclass
import struct


@dataclass
class MMapConfig:
    """Configuration for memory-mapped storage."""
    # Initial file size for new files
    initial_size: int = 1024 * 1024  # 1MB

    # Growth factor when file needs to expand
    growth_factor: float = 1.5

    # Maximum file size
    max_size: int = 1024 * 1024 * 1024  # 1GB

    # Access mode: read-only or read-write
    readonly: bool = False


class MMapStorage:
    """
    Memory-mapped file storage for efficient large file access.

    Memory mapping allows the operating system to handle paging,
    enabling efficient access to large files without loading
    them entirely into memory.

    Example:
        storage = MMapStorage("large_tasks.dat")

        # Write data
        storage.write_at(0, b"Task data here")

        # Read data
        data = storage.read_at(0, 14)

        # Use as task storage
        storage.store_task("task-1", {"status": "done", ...})
        task = storage.load_task("task-1")
    """

    # File header format:
    # - 4 bytes: magic number (MMTS)
    # - 4 bytes: version
    # - 8 bytes: data length
    # - 8 bytes: index offset
    HEADER_SIZE = 24
    MAGIC = b"MMTS"
    VERSION = 1

    def __init__(
        self,
        path: str,
        config: Optional[MMapConfig] = None
    ):
        self.path = Path(path)
        self.config = config or MMapConfig()

        self._file = None
        self._mmap: Optional[mmap.mmap] = None
        self._lock = threading.RLock()
        self._index: Dict[str, Tuple[int, int]] = {}  # key -> (offset, length)
        self._data_end = self.HEADER_SIZE

    def open(self) -> None:
        """Open the memory-mapped file."""
        with self._lock:
            if self._mmap is not None:
                return

            # Create file if it doesn't exist
            if not self.path.exists():
                self._create_new_file()
            else:
                self._open_existing_file()

    def _create_new_file(self) -> None:
        """Create a new memory-mapped file."""
        # Create file with initial size
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "wb") as f:
            # Write header
            f.write(self.MAGIC)
            f.write(struct.pack("<I", self.VERSION))
            f.write(struct.pack("<Q", 0))  # data length
            f.write(struct.pack("<Q", 0))  # index offset

            # Extend to initial size
            f.seek(self.config.initial_size - 1)
            f.write(b"\x00")

        # Open for mapping
        access = mmap.ACCESS_READ if self.config.readonly else mmap.ACCESS_WRITE
        self._file = open(self.path, "r+b")
        self._mmap = mmap.mmap(
            self._file.fileno(),
            0,
            access=access
        )

        self._data_end = self.HEADER_SIZE
        self._index = {}

    def _open_existing_file(self) -> None:
        """Open an existing memory-mapped file."""
        access = mmap.ACCESS_READ if self.config.readonly else mmap.ACCESS_WRITE
        mode = "rb" if self.config.readonly else "r+b"

        self._file = open(self.path, mode)
        self._mmap = mmap.mmap(
            self._file.fileno(),
            0,
            access=access
        )

        # Read header
        magic = self._mmap[:4]
        if magic != self.MAGIC:
            raise ValueError(f"Invalid file format: {self.path}")

        version = struct.unpack("<I", self._mmap[4:8])[0]
        if version > self.VERSION:
            raise ValueError(f"Unsupported version: {version}")

        data_length = struct.unpack("<Q", self._mmap[8:16])[0]
        index_offset = struct.unpack("<Q", self._mmap[16:24])[0]

        self._data_end = self.HEADER_SIZE + data_length

        # Load index if present
        if index_offset > 0:
            self._load_index(index_offset)

    def _load_index(self, offset: int) -> None:
        """Load the index from file."""
        if self._mmap is None:
            return

        # Read index size
        index_size = struct.unpack("<I", self._mmap[offset:offset+4])[0]

        # Read index data
        index_data = self._mmap[offset+4:offset+4+index_size]
        self._index = json.loads(index_data.decode("utf-8"))

        # Convert list values to tuples
        self._index = {k: tuple(v) for k, v in self._index.items()}

    def _save_index(self) -> None:
        """Save the index to file."""
        if self._mmap is None or self.config.readonly:
            return

        # Serialize index
        index_data = json.dumps(self._index).encode("utf-8")
        index_size = len(index_data)

        # Ensure we have space
        index_offset = self._data_end
        total_needed = index_offset + 4 + index_size

        if total_needed > len(self._mmap):
            self._grow_file(total_needed)

        # Write index
        self._mmap[index_offset:index_offset+4] = struct.pack("<I", index_size)
        self._mmap[index_offset+4:index_offset+4+index_size] = index_data

        # Update header
        data_length = self._data_end - self.HEADER_SIZE
        self._mmap[8:16] = struct.pack("<Q", data_length)
        self._mmap[16:24] = struct.pack("<Q", index_offset)

    def _grow_file(self, min_size: int) -> None:
        """Grow the file to accommodate more data."""
        if self._mmap is None or self._file is None:
            return

        current_size = len(self._mmap)
        new_size = int(current_size * self.config.growth_factor)
        new_size = max(new_size, min_size)
        new_size = min(new_size, self.config.max_size)

        if new_size <= current_size:
            raise RuntimeError(f"Cannot grow file beyond {self.config.max_size} bytes")

        # Close current mapping
        self._mmap.close()

        # Extend file
        self._file.seek(new_size - 1)
        self._file.write(b"\x00")
        self._file.flush()

        # Reopen mapping
        self._mmap = mmap.mmap(
            self._file.fileno(),
            0,
            access=mmap.ACCESS_WRITE
        )

    def close(self) -> None:
        """Close the memory-mapped file."""
        with self._lock:
            if self._mmap:
                if not self.config.readonly:
                    self._save_index()
                self._mmap.close()
                self._mmap = None

            if self._file:
                self._file.close()
                self._file = None

    def __enter__(self) -> "MMapStorage":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def read_at(self, offset: int, length: int) -> bytes:
        """Read bytes at a specific offset."""
        with self._lock:
            if self._mmap is None:
                raise RuntimeError("Storage not open")
            return bytes(self._mmap[offset:offset+length])

    def write_at(self, offset: int, data: bytes) -> int:
        """Write bytes at a specific offset."""
        with self._lock:
            if self._mmap is None or self.config.readonly:
                raise RuntimeError("Storage not open or read-only")

            # Ensure we have space
            end = offset + len(data)
            if end > len(self._mmap):
                self._grow_file(end)

            self._mmap[offset:end] = data
            return len(data)

    def store_task(self, task_id: str, data: Dict[str, Any]) -> None:
        """Store a task in the memory-mapped file."""
        with self._lock:
            if self._mmap is None or self.config.readonly:
                raise RuntimeError("Storage not open or read-only")

            # Serialize task
            task_data = json.dumps(data, default=str).encode("utf-8")
            length = len(task_data)

            # Find or allocate space
            if task_id in self._index:
                # Check if existing space is sufficient
                old_offset, old_length = self._index[task_id]
                if length <= old_length:
                    # Reuse existing space
                    self._mmap[old_offset:old_offset+length] = task_data
                    # Zero out rest
                    if length < old_length:
                        self._mmap[old_offset+length:old_offset+old_length] = b"\x00" * (old_length - length)
                    self._index[task_id] = (old_offset, length)
                    return

            # Allocate new space at end
            offset = self._data_end

            # Ensure we have space (including index)
            total_needed = offset + length + 1024  # Extra for index
            if total_needed > len(self._mmap):
                self._grow_file(total_needed)

            # Write data
            self._mmap[offset:offset+length] = task_data
            self._index[task_id] = (offset, length)
            self._data_end = offset + length

    def load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load a task from the memory-mapped file."""
        with self._lock:
            if self._mmap is None:
                raise RuntimeError("Storage not open")

            if task_id not in self._index:
                return None

            offset, length = self._index[task_id]
            data = self._mmap[offset:offset+length]

            return json.loads(data.decode("utf-8"))

    def delete_task(self, task_id: str) -> bool:
        """Delete a task (marks space as free, doesn't reclaim)."""
        with self._lock:
            if task_id in self._index:
                # Just remove from index, space becomes dead
                del self._index[task_id]
                return True
            return False

    def list_tasks(self) -> Iterator[str]:
        """List all task IDs."""
        with self._lock:
            for task_id in self._index.keys():
                yield task_id

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            file_size = len(self._mmap) if self._mmap else 0
            used_space = sum(length for _, length in self._index.values())

            return {
                "file_size": file_size,
                "used_space": used_space,
                "task_count": len(self._index),
                "fragmentation": 1 - (used_space / max(1, self._data_end - self.HEADER_SIZE)),
            }


class MMapTaskStore:
    """
    High-level task store using memory-mapped storage.

    Provides a dict-like interface for storing tasks.

    Example:
        store = MMapTaskStore(".coordination/tasks.mmap")

        store["task-1"] = {"status": "done", "output": "..."}
        task = store["task-1"]
        del store["task-2"]

        for task_id in store:
            print(task_id)
    """

    def __init__(self, path: str, config: Optional[MMapConfig] = None):
        self._storage = MMapStorage(path, config)
        self._storage.open()

    def __getitem__(self, key: str) -> Dict[str, Any]:
        result = self._storage.load_task(key)
        if result is None:
            raise KeyError(key)
        return result

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        self._storage.store_task(key, value)

    def __delitem__(self, key: str) -> None:
        if not self._storage.delete_task(key):
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._storage._index

    def __iter__(self) -> Iterator[str]:
        return self._storage.list_tasks()

    def __len__(self) -> int:
        return len(self._storage._index)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a task with default value."""
        result = self._storage.load_task(key)
        return result if result is not None else default

    def items(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Iterate over all tasks."""
        for task_id in self._storage.list_tasks():
            task = self._storage.load_task(task_id)
            if task:
                yield task_id, task

    def close(self) -> None:
        """Close the store."""
        self._storage.close()

    def __enter__(self) -> "MMapTaskStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
