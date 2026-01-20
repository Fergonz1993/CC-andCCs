"""
Profiling hooks for bottleneck detection.

Implements adv-perf-010: Profiling hooks for bottleneck detection

Features:
- Function-level profiling with decorators
- Async-aware profiling
- Flame graph generation
- Statistical aggregation
- Low-overhead sampling
"""

import asyncio
import cProfile
import functools
import io
import pstats
import threading
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, TypeVar, Tuple,
    Union, ParamSpec
)
import json

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class ProfileSample:
    """A single profiling sample."""
    function_name: str
    module: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error: Optional[str] = None
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.module}.{self.function_name}"


@dataclass
class ProfileStats:
    """Aggregated statistics for a profiled function."""
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    error_count: int = 0

    # Time percentiles
    p50_time: float = 0.0
    p95_time: float = 0.0
    p99_time: float = 0.0

    _samples: List[float] = field(default_factory=list, repr=False)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.call_count if self.call_count > 0 else 0.0

    def add_sample(self, duration: float, success: bool = True) -> None:
        """Add a sample to the statistics."""
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)

        if not success:
            self.error_count += 1

        self._samples.append(duration)

        # Keep samples bounded
        if len(self._samples) > 10000:
            self._samples = self._samples[-5000:]

        # Recalculate percentiles periodically
        if self.call_count % 100 == 0:
            self._calculate_percentiles()

    def _calculate_percentiles(self) -> None:
        """Calculate time percentiles."""
        if not self._samples:
            return

        sorted_samples = sorted(self._samples)
        n = len(sorted_samples)

        self.p50_time = sorted_samples[int(n * 0.50)]
        self.p95_time = sorted_samples[int(n * 0.95)]
        self.p99_time = sorted_samples[min(int(n * 0.99), n - 1)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "p50_time": self.p50_time,
            "p95_time": self.p95_time,
            "p99_time": self.p99_time,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
        }


class Profiler:
    """
    Application profiler for performance monitoring.

    Provides hooks to profile function execution and identify bottlenecks.

    Example:
        profiler = Profiler()

        @profiler.profile()
        def slow_function():
            time.sleep(1)

        slow_function()

        # Get statistics
        stats = profiler.get_stats()
        for name, stat in stats.items():
            print(f"{name}: avg={stat.avg_time:.3f}s, count={stat.call_count}")
    """

    def __init__(self, enabled: bool = True, sample_rate: float = 1.0):
        self.enabled = enabled
        self.sample_rate = sample_rate  # 1.0 = profile everything

        self._stats: Dict[str, ProfileStats] = {}
        self._samples: List[ProfileSample] = []
        self._lock = threading.RLock()
        self._context_stack = threading.local()

        # cProfile for detailed profiling
        self._cprofiler: Optional[cProfile.Profile] = None

    def _should_sample(self) -> bool:
        """Determine if this call should be sampled."""
        import random
        return self.enabled and random.random() < self.sample_rate

    def _get_context_stack(self) -> List[str]:
        """Get the current profiling context stack."""
        if not hasattr(self._context_stack, "stack"):
            self._context_stack.stack = []
        return self._context_stack.stack

    def profile(
        self,
        name: Optional[str] = None,
        track_args: bool = False
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """
        Decorator to profile a function.

        Args:
            name: Custom name for the function
            track_args: Whether to include arguments in metadata
        """
        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            func_name = name or func.__name__
            module = func.__module__

            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                if not self._should_sample():
                    return func(*args, **kwargs)

                stack = self._get_context_stack()
                parent = stack[-1] if stack else None
                qualified = f"{module}.{func_name}"
                stack.append(qualified)

                start_time = time.perf_counter()
                success = True
                error = None

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time

                    stack.pop()

                    metadata = {}
                    if track_args:
                        metadata["args"] = repr(args)[:100]
                        metadata["kwargs"] = repr(kwargs)[:100]

                    sample = ProfileSample(
                        function_name=func_name,
                        module=module,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        success=success,
                        error=error,
                        parent=parent,
                        metadata=metadata,
                    )

                    self._record_sample(sample)

            return wrapper
        return decorator

    def profile_async(
        self,
        name: Optional[str] = None,
        track_args: bool = False
    ) -> Callable:
        """
        Decorator to profile an async function.
        """
        def decorator(func: Callable) -> Callable:
            func_name = name or func.__name__
            module = func.__module__

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if not self._should_sample():
                    return await func(*args, **kwargs)

                stack = self._get_context_stack()
                parent = stack[-1] if stack else None
                qualified = f"{module}.{func_name}"
                stack.append(qualified)

                start_time = time.perf_counter()
                success = True
                error = None

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time

                    stack.pop()

                    sample = ProfileSample(
                        function_name=func_name,
                        module=module,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        success=success,
                        error=error,
                        parent=parent,
                    )

                    self._record_sample(sample)

            return wrapper
        return decorator

    def _record_sample(self, sample: ProfileSample) -> None:
        """Record a profiling sample."""
        with self._lock:
            # Add to samples list
            self._samples.append(sample)

            # Keep samples bounded
            if len(self._samples) > 100000:
                self._samples = self._samples[-50000:]

            # Update stats
            qualified = sample.qualified_name
            if qualified not in self._stats:
                self._stats[qualified] = ProfileStats(function_name=qualified)

            self._stats[qualified].add_sample(sample.duration, sample.success)

    @contextmanager
    def measure(self, name: str):
        """
        Context manager for measuring a code block.

        Example:
            with profiler.measure("expensive_operation"):
                # code to measure
                pass
        """
        if not self._should_sample():
            yield
            return

        stack = self._get_context_stack()
        parent = stack[-1] if stack else None
        stack.append(name)

        start_time = time.perf_counter()
        success = True
        error = None

        try:
            yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time

            stack.pop()

            sample = ProfileSample(
                function_name=name,
                module="__context__",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=success,
                error=error,
                parent=parent,
            )

            self._record_sample(sample)

    def get_stats(self) -> Dict[str, ProfileStats]:
        """Get all profiling statistics."""
        with self._lock:
            return dict(self._stats)

    def get_top_functions(self, n: int = 10, by: str = "total_time") -> List[ProfileStats]:
        """Get the top N functions by a metric."""
        with self._lock:
            stats_list = list(self._stats.values())

            if by == "total_time":
                stats_list.sort(key=lambda s: s.total_time, reverse=True)
            elif by == "avg_time":
                stats_list.sort(key=lambda s: s.avg_time, reverse=True)
            elif by == "call_count":
                stats_list.sort(key=lambda s: s.call_count, reverse=True)
            elif by == "max_time":
                stats_list.sort(key=lambda s: s.max_time, reverse=True)
            elif by == "error_rate":
                stats_list.sort(key=lambda s: s.error_rate, reverse=True)

            return stats_list[:n]

    def get_recent_samples(self, n: int = 100) -> List[ProfileSample]:
        """Get the most recent N samples."""
        with self._lock:
            return self._samples[-n:]

    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._stats.clear()
            self._samples.clear()

    def start_cprofiler(self) -> None:
        """Start detailed cProfile profiling."""
        if self._cprofiler is None:
            self._cprofiler = cProfile.Profile()
        self._cprofiler.enable()

    def stop_cprofiler(self) -> str:
        """Stop cProfile and return statistics."""
        if self._cprofiler is None:
            return ""

        self._cprofiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(self._cprofiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(50)

        return stream.getvalue()

    def generate_flame_graph_data(self) -> List[Dict[str, Any]]:
        """
        Generate data for flame graph visualization.

        Returns data in format compatible with common flame graph tools.
        """
        with self._lock:
            # Build call tree from samples
            calls: Dict[Tuple[Optional[str], str], int] = defaultdict(int)

            for sample in self._samples:
                key = (sample.parent, sample.qualified_name)
                calls[key] += 1

            # Convert to flame graph format
            flame_data = []
            for (parent, name), count in calls.items():
                entry = {
                    "name": name,
                    "value": count,
                }
                if parent:
                    entry["parent"] = parent
                flame_data.append(entry)

            return flame_data

    def report(self) -> str:
        """Generate a text report of profiling results."""
        lines = ["=" * 60, "PROFILING REPORT", "=" * 60, ""]

        # Top functions by total time
        lines.append("Top 10 Functions by Total Time:")
        lines.append("-" * 40)

        for stat in self.get_top_functions(10, by="total_time"):
            lines.append(
                f"  {stat.function_name[:40]:40s} "
                f"total={stat.total_time:8.3f}s "
                f"avg={stat.avg_time:8.3f}s "
                f"calls={stat.call_count}"
            )

        lines.append("")

        # Top functions by average time
        lines.append("Top 10 Functions by Average Time:")
        lines.append("-" * 40)

        for stat in self.get_top_functions(10, by="avg_time"):
            lines.append(
                f"  {stat.function_name[:40]:40s} "
                f"avg={stat.avg_time:8.3f}s "
                f"p95={stat.p95_time:8.3f}s "
                f"calls={stat.call_count}"
            )

        lines.append("")

        # Functions with errors
        error_stats = [s for s in self._stats.values() if s.error_count > 0]
        if error_stats:
            lines.append("Functions with Errors:")
            lines.append("-" * 40)

            error_stats.sort(key=lambda s: s.error_rate, reverse=True)
            for stat in error_stats[:10]:
                lines.append(
                    f"  {stat.function_name[:40]:40s} "
                    f"errors={stat.error_count} "
                    f"rate={stat.error_rate:.1%}"
                )

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def export_json(self) -> str:
        """Export profiling data as JSON."""
        with self._lock:
            data = {
                "stats": {name: stat.to_dict() for name, stat in self._stats.items()},
                "samples": [
                    {
                        "function_name": s.function_name,
                        "module": s.module,
                        "duration": s.duration,
                        "success": s.success,
                        "parent": s.parent,
                    }
                    for s in self._samples[-1000:]  # Last 1000 samples
                ],
            }
            return json.dumps(data, indent=2)


# Global profiler instance
_profiler: Optional[Profiler] = None


def get_profiler(enabled: bool = True) -> Profiler:
    """Get or create the global profiler."""
    global _profiler
    if _profiler is None:
        _profiler = Profiler(enabled=enabled)
    return _profiler


def profile(
    name: Optional[str] = None,
    track_args: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to profile a function using the global profiler.

    Example:
        @profile()
        def my_function():
            pass

        @profile("custom_name", track_args=True)
        def another_function(x):
            pass
    """
    return get_profiler().profile(name, track_args)


def profile_async(
    name: Optional[str] = None,
    track_args: bool = False
) -> Callable:
    """
    Decorator to profile an async function using the global profiler.
    """
    return get_profiler().profile_async(name, track_args)


@contextmanager
def measure(name: str):
    """Context manager for measuring a code block."""
    with get_profiler().measure(name):
        yield
