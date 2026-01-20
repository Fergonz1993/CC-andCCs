"""
Lazy loading of task results.

Implements adv-perf-003: Lazy loading of task results

Features:
- Deferred loading of large task results
- Memory-efficient result handling
- On-demand loading with caching
- Placeholder objects that load on access
"""

import json
import threading
from typing import TypeVar, Generic, Optional, Callable, Any, Dict
from dataclasses import dataclass, field
from pathlib import Path
from functools import cached_property
import weakref

T = TypeVar("T")


class LazyLoadError(Exception):
    """Error during lazy loading."""
    pass


class LazyValue(Generic[T]):
    """
    A lazily-loaded value that is only computed/loaded when accessed.

    Example:
        lazy_result = LazyValue(lambda: expensive_computation())
        # Nothing computed yet

        value = lazy_result.get()  # Now it's computed
        value = lazy_result.get()  # Returns cached value
    """

    def __init__(
        self,
        loader: Callable[[], T],
        on_load: Optional[Callable[[T], None]] = None
    ):
        self._loader = loader
        self._on_load = on_load
        self._value: Optional[T] = None
        self._loaded = False
        self._error: Optional[Exception] = None
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        """Check if value has been loaded."""
        return self._loaded

    def get(self) -> T:
        """Get the value, loading it if necessary."""
        with self._lock:
            if self._error:
                raise LazyLoadError(f"Previous load failed: {self._error}")

            if not self._loaded:
                try:
                    self._value = self._loader()
                    self._loaded = True

                    if self._on_load:
                        self._on_load(self._value)
                except Exception as e:
                    self._error = e
                    raise LazyLoadError(f"Failed to load value: {e}") from e

            return self._value  # type: ignore

    def reset(self) -> None:
        """Reset the lazy value so it will be reloaded."""
        with self._lock:
            self._value = None
            self._loaded = False
            self._error = None

    def preload(self) -> None:
        """Preload the value in the background."""
        self.get()


@dataclass
class LazyTaskResult:
    """
    A lazily-loaded task result.

    Task results can be large (containing full file contents, logs, etc.)
    This class provides a placeholder that only loads the full result
    when explicitly accessed.

    Example:
        result = LazyTaskResult(
            task_id="task-123",
            results_dir=Path(".coordination/results")
        )

        # Metadata is available immediately
        print(result.task_id)

        # Full output is loaded on demand
        print(result.output)  # Loads from file
    """

    task_id: str
    results_dir: Path
    summary: str = ""  # Brief summary available without loading
    status: str = "done"  # Can check status without loading full result

    _loaded_data: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def result_path(self) -> Path:
        """Path to the result file."""
        return self.results_dir / f"{self.task_id}.md"

    @property
    def result_json_path(self) -> Path:
        """Path to the JSON result file."""
        return self.results_dir / f"{self.task_id}.json"

    def _ensure_loaded(self) -> Dict[str, Any]:
        """Ensure result data is loaded."""
        with self._lock:
            if self._loaded_data is not None:
                return self._loaded_data

            # Try JSON first
            if self.result_json_path.exists():
                with open(self.result_json_path, "r") as f:
                    self._loaded_data = json.load(f)
                return self._loaded_data

            # Fall back to markdown
            if self.result_path.exists():
                content = self.result_path.read_text()
                self._loaded_data = self._parse_markdown_result(content)
                return self._loaded_data

            return {}

    def _parse_markdown_result(self, content: str) -> Dict[str, Any]:
        """Parse markdown result file into structured data."""
        result: Dict[str, Any] = {"raw_content": content}

        # Extract sections
        current_section = None
        section_content = []

        for line in content.split("\n"):
            if line.startswith("## "):
                if current_section and section_content:
                    result[current_section] = "\n".join(section_content).strip()
                current_section = line[3:].strip().lower().replace(" ", "_")
                section_content = []
            elif current_section:
                section_content.append(line)

        if current_section and section_content:
            result[current_section] = "\n".join(section_content).strip()

        return result

    @property
    def output(self) -> str:
        """Get the full output (lazy loaded)."""
        data = self._ensure_loaded()
        return data.get("output", data.get("raw_content", ""))

    @property
    def files_modified(self) -> list:
        """Get list of modified files."""
        data = self._ensure_loaded()
        files = data.get("files_modified", "")
        if isinstance(files, str):
            return [f.strip("- ") for f in files.split("\n") if f.strip()]
        return files

    @property
    def files_created(self) -> list:
        """Get list of created files."""
        data = self._ensure_loaded()
        files = data.get("files_created", "")
        if isinstance(files, str):
            return [f.strip("- ") for f in files.split("\n") if f.strip()]
        return files

    @property
    def error(self) -> Optional[str]:
        """Get error message if any."""
        data = self._ensure_loaded()
        return data.get("error") or data.get("failure_reason")

    def to_dict(self, include_output: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Args:
            include_output: If False, excludes large output field
        """
        result = {
            "task_id": self.task_id,
            "status": self.status,
            "summary": self.summary,
        }

        if include_output:
            data = self._ensure_loaded()
            result.update(data)

        return result


class LazyLoader(Generic[T]):
    """
    Factory for creating lazy-loaded objects.

    Manages a registry of lazy loaders and provides batch operations.

    Example:
        loader = LazyLoader[TaskResult]()

        loader.register("task-1", lambda: load_result("task-1"))
        loader.register("task-2", lambda: load_result("task-2"))

        # Load specific item
        result1 = loader.get("task-1")

        # Preload all in background
        loader.preload_all()
    """

    def __init__(self):
        self._registry: Dict[str, LazyValue[T]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        key: str,
        loader: Callable[[], T],
        on_load: Optional[Callable[[T], None]] = None
    ) -> None:
        """Register a lazy loader for a key."""
        with self._lock:
            self._registry[key] = LazyValue(loader, on_load)

    def get(self, key: str) -> Optional[T]:
        """Get a value by key, loading if necessary."""
        with self._lock:
            lazy = self._registry.get(key)

        if lazy is None:
            return None

        return lazy.get()

    def is_loaded(self, key: str) -> bool:
        """Check if a value is already loaded."""
        with self._lock:
            lazy = self._registry.get(key)
            return lazy.is_loaded if lazy else False

    def preload(self, key: str) -> None:
        """Preload a specific value."""
        with self._lock:
            lazy = self._registry.get(key)

        if lazy:
            lazy.preload()

    def preload_all(self, parallel: bool = True) -> None:
        """Preload all registered values."""
        with self._lock:
            items = list(self._registry.items())

        if parallel:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(lazy.preload)
                    for _, lazy in items
                    if not lazy.is_loaded
                ]
                concurrent.futures.wait(futures)
        else:
            for _, lazy in items:
                if not lazy.is_loaded:
                    lazy.preload()

    def keys(self) -> list:
        """Get all registered keys."""
        with self._lock:
            return list(self._registry.keys())

    def reset(self, key: str) -> None:
        """Reset a lazy value."""
        with self._lock:
            lazy = self._registry.get(key)
            if lazy:
                lazy.reset()

    def remove(self, key: str) -> None:
        """Remove a lazy value."""
        with self._lock:
            self._registry.pop(key, None)

    def clear(self) -> None:
        """Clear all registered loaders."""
        with self._lock:
            self._registry.clear()


class TaskResultLoader(LazyLoader[Dict[str, Any]]):
    """
    Specialized lazy loader for task results.

    Example:
        loader = TaskResultLoader(Path(".coordination/results"))

        # Register results directory
        loader.scan_results()

        # Get specific result (lazy loaded)
        result = loader.get("task-123")

        # Get summary without loading full results
        summaries = loader.get_summaries()
    """

    def __init__(self, results_dir: Path):
        super().__init__()
        self.results_dir = results_dir
        self._summaries: Dict[str, str] = {}

    def scan_results(self) -> int:
        """
        Scan results directory and register lazy loaders.

        Returns number of results found.
        """
        if not self.results_dir.exists():
            return 0

        count = 0
        for result_file in self.results_dir.glob("*.json"):
            task_id = result_file.stem
            self.register(
                task_id,
                lambda p=result_file: self._load_json(p)
            )
            count += 1

        for result_file in self.results_dir.glob("*.md"):
            task_id = result_file.stem
            if task_id not in self._registry:  # Prefer JSON
                self.register(
                    task_id,
                    lambda p=result_file: self._load_markdown(p)
                )
                count += 1

        return count

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load a JSON result file."""
        with open(path, "r") as f:
            return json.load(f)

    def _load_markdown(self, path: Path) -> Dict[str, Any]:
        """Load and parse a markdown result file."""
        content = path.read_text()
        result = LazyTaskResult(
            task_id=path.stem,
            results_dir=self.results_dir
        )
        return result._parse_markdown_result(content)

    def get_summaries(self) -> Dict[str, str]:
        """
        Get brief summaries without loading full results.

        Reads only first few lines of each file.
        """
        if not self.results_dir.exists():
            return {}

        summaries = {}

        for result_file in self.results_dir.glob("*"):
            if result_file.suffix in (".json", ".md"):
                task_id = result_file.stem
                with open(result_file, "r") as f:
                    # Read first 5 lines for summary
                    lines = []
                    for i, line in enumerate(f):
                        if i >= 5:
                            break
                        lines.append(line)

                    summary = "".join(lines)[:200]
                    summaries[task_id] = summary

        return summaries
