"""
In-memory caching for task lookups and query results.

Implements:
- adv-perf-001: In-memory caching for task lookups
- adv-perf-009: Query result caching with TTL

Features:
- LRU eviction policy
- TTL-based expiration
- Thread-safe operations
- Cache statistics for monitoring
"""

import time
import threading
from typing import TypeVar, Generic, Optional, Dict, Any, Callable, Hashable
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


@dataclass
class CacheStats:
    """Statistics for cache monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata."""
    value: T
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # Time-to-live in seconds

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class TaskCache(Generic[K, T]):
    """
    High-performance in-memory cache for task lookups.

    Features:
    - LRU eviction policy
    - Thread-safe operations
    - Configurable max size
    - Access statistics

    Example:
        cache = TaskCache[str, Task](max_size=1000)
        cache.set("task-123", task)
        task = cache.get("task-123")
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[K, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def get(self, key: K) -> Optional[T]:
        """Get a value from cache."""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self.stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self.stats.hits += 1

            return entry.value

    def set(self, key: K, value: T) -> None:
        """Set a value in cache."""
        with self._lock:
            # Check if we need to evict
            if key not in self._cache and len(self._cache) >= self.max_size:
                # Evict least recently used (first item)
                self._cache.popitem(last=False)
                self.stats.evictions += 1

            self._cache[key] = CacheEntry(value=value)
            self._cache.move_to_end(key)

    def delete(self, key: K) -> bool:
        """Delete a value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def contains(self, key: K) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self._cache

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def get_or_set(self, key: K, factory: Callable[[], T]) -> T:
        """Get from cache or compute and cache the value."""
        with self._lock:
            value = self.get(key)
            if value is not None:
                return value

            value = factory()
            self.set(key, value)
            return value

    def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all keys starting with prefix."""
        with self._lock:
            keys_to_remove = [
                k for k in self._cache.keys()
                if isinstance(k, str) and k.startswith(prefix)
            ]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)


class TTLCache(TaskCache[K, T]):
    """
    Cache with time-to-live expiration.

    Extends TaskCache with automatic expiration of entries.

    Example:
        cache = TTLCache[str, dict](max_size=1000, default_ttl=300)  # 5 min TTL
        cache.set("query-results", results)  # Expires after 5 minutes
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        super().__init__(max_size)
        self.default_ttl = default_ttl
        self._cleanup_interval = 60.0  # Clean up expired entries every minute
        self._last_cleanup = time.time()

    def get(self, key: K) -> Optional[T]:
        """Get a value, returning None if expired."""
        with self._lock:
            self._maybe_cleanup()

            entry = self._cache.get(key)

            if entry is None:
                self.stats.misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self.stats.misses += 1
                self.stats.expirations += 1
                return None

            self._cache.move_to_end(key)
            entry.touch()
            self.stats.hits += 1

            return entry.value

    def set(self, key: K, value: T, ttl: Optional[float] = None) -> None:
        """Set a value with optional custom TTL."""
        with self._lock:
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self.stats.evictions += 1

            self._cache[key] = CacheEntry(
                value=value,
                ttl=ttl if ttl is not None else self.default_ttl
            )
            self._cache.move_to_end(key)

    def _maybe_cleanup(self) -> None:
        """Periodically clean up expired entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        expired_keys = [
            k for k, v in self._cache.items()
            if v.is_expired()
        ]

        for key in expired_keys:
            del self._cache[key]
            self.stats.expirations += 1


class QueryCache(TTLCache[str, Any]):
    """
    Specialized cache for query results.

    Provides caching for expensive queries like:
    - get_available_tasks()
    - get_tasks_by_status()
    - aggregation queries

    Example:
        cache = QueryCache(default_ttl=10.0)  # 10 second TTL

        @cache.cached("available_tasks")
        def get_available_tasks():
            return expensive_computation()
    """

    def __init__(self, default_ttl: float = 10.0, max_size: int = 500):
        super().__init__(max_size=max_size, default_ttl=default_ttl)
        self._invalidation_hooks: Dict[str, list[Callable[[], None]]] = {}

    def cached(self, cache_key: str, ttl: Optional[float] = None):
        """Decorator for caching function results."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                # Build cache key from function name and arguments
                key = f"{cache_key}:{hash((args, tuple(sorted(kwargs.items()))))}"

                result = self.get(key)
                if result is not None:
                    return result

                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                return result

            return wrapper
        return decorator

    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        count = self.invalidate_by_prefix(pattern)

        # Call any registered invalidation hooks
        for key, hooks in self._invalidation_hooks.items():
            if key.startswith(pattern):
                for hook in hooks:
                    hook()

        return count

    def on_invalidate(self, pattern: str, callback: Callable[[], None]) -> None:
        """Register a callback for when a pattern is invalidated."""
        if pattern not in self._invalidation_hooks:
            self._invalidation_hooks[pattern] = []
        self._invalidation_hooks[pattern].append(callback)


# Singleton instances for global use
_task_cache: Optional[TaskCache] = None
_query_cache: Optional[QueryCache] = None


def get_task_cache(max_size: int = 1000) -> TaskCache:
    """Get or create the global task cache."""
    global _task_cache
    if _task_cache is None:
        _task_cache = TaskCache(max_size=max_size)
    return _task_cache


def get_query_cache(default_ttl: float = 10.0) -> QueryCache:
    """Get or create the global query cache."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(default_ttl=default_ttl)
    return _query_cache
