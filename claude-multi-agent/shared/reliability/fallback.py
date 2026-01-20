"""
Fallback Strategies for Failures (adv-rel-003)

Implements fallback patterns for graceful degradation when primary
operations fail. Supports multiple fallback strategies.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, List, Dict, Generic, TypeVar
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class FallbackType(str, Enum):
    """Types of fallback strategies."""
    CACHE = "cache"           # Return cached value
    DEFAULT = "default"       # Return default value
    ALTERNATIVE = "alternative"  # Call alternative function
    QUEUE = "queue"           # Queue for later processing
    DEGRADE = "degrade"       # Return degraded response
    FAIL_SILENT = "fail_silent"  # Fail silently (return None)


@dataclass
class FallbackResult(Generic[T]):
    """Result of a fallback operation."""
    value: Optional[T] = None
    fallback_used: bool = False
    fallback_type: Optional[FallbackType] = None
    original_error: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.now)


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Execute the fallback strategy."""
        pass

    def can_handle(self, error: Exception) -> bool:
        """Check if this strategy can handle the given error."""
        return True


class CacheFallback(FallbackStrategy):
    """Return cached value on failure."""

    def __init__(
        self,
        name: str = "cache",
        cache_getter: Optional[Callable[[str], Any]] = None,
        max_age_seconds: float = 3600.0,
    ):
        super().__init__(name)
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._cache_getter = cache_getter
        self._max_age_seconds = max_age_seconds

    def set_cache(self, key: str, value: Any) -> None:
        """Set a cached value."""
        self._cache[key] = (value, datetime.now())

    def get_cache(self, key: str) -> Optional[Any]:
        """Get a cached value if not expired."""
        if key not in self._cache:
            if self._cache_getter:
                return self._cache_getter(key)
            return None

        value, timestamp = self._cache[key]
        age = (datetime.now() - timestamp).total_seconds()

        if age > self._max_age_seconds:
            del self._cache[key]
            return None

        return value

    def execute(self, error: Exception, context: Dict[str, Any]) -> Any:
        key = context.get("cache_key", "default")
        cached = self.get_cache(key)

        if cached is not None:
            logger.info(f"CacheFallback: returning cached value for key '{key}'")
            return cached

        logger.warning(f"CacheFallback: no cached value for key '{key}'")
        return None


class DefaultFallback(FallbackStrategy):
    """Return a default value on failure."""

    def __init__(self, name: str = "default", default_value: Any = None):
        super().__init__(name)
        self.default_value = default_value

    def execute(self, error: Exception, context: Dict[str, Any]) -> Any:
        logger.info(f"DefaultFallback: returning default value")
        return self.default_value


class AlternativeFallback(FallbackStrategy):
    """Call an alternative function on failure."""

    def __init__(
        self,
        name: str = "alternative",
        alternative_func: Optional[Callable[..., Any]] = None,
    ):
        super().__init__(name)
        self._alternative_func = alternative_func

    def set_alternative(self, func: Callable[..., Any]) -> None:
        """Set the alternative function."""
        self._alternative_func = func

    def execute(self, error: Exception, context: Dict[str, Any]) -> Any:
        if self._alternative_func is None:
            logger.warning("AlternativeFallback: no alternative function set")
            return None

        try:
            args = context.get("args", ())
            kwargs = context.get("kwargs", {})
            result = self._alternative_func(*args, **kwargs)
            logger.info("AlternativeFallback: alternative function succeeded")
            return result
        except Exception as e:
            logger.error(f"AlternativeFallback: alternative function failed: {e}")
            raise


class QueueFallback(FallbackStrategy):
    """Queue the operation for later processing."""

    def __init__(
        self,
        name: str = "queue",
        queue_func: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        super().__init__(name)
        self._queue: List[Dict[str, Any]] = []
        self._queue_func = queue_func

    def execute(self, error: Exception, context: Dict[str, Any]) -> Any:
        entry = {
            "context": context,
            "error": str(error),
            "timestamp": datetime.now().isoformat(),
        }

        if self._queue_func:
            try:
                self._queue_func(entry)
            except Exception as e:
                logger.error(f"QueueFallback: queue function failed: {e}")

        self._queue.append(entry)
        logger.info(f"QueueFallback: queued operation for later (queue size: {len(self._queue)})")
        return {"queued": True, "queue_size": len(self._queue)}

    def get_queued(self) -> List[Dict[str, Any]]:
        """Get all queued operations."""
        return self._queue.copy()

    def clear_queue(self) -> None:
        """Clear the queue."""
        self._queue.clear()


class DegradeFallback(FallbackStrategy):
    """Return a degraded response on failure."""

    def __init__(
        self,
        name: str = "degrade",
        degrade_func: Optional[Callable[[Exception, Dict[str, Any]], Any]] = None,
    ):
        super().__init__(name)
        self._degrade_func = degrade_func

    def execute(self, error: Exception, context: Dict[str, Any]) -> Any:
        if self._degrade_func:
            try:
                result = self._degrade_func(error, context)
                logger.info("DegradeFallback: returned degraded response")
                return result
            except Exception as e:
                logger.error(f"DegradeFallback: degrade function failed: {e}")
                return None

        # Default degraded response
        return {
            "degraded": True,
            "message": "Service temporarily unavailable",
            "error_type": type(error).__name__,
        }


class FailSilentFallback(FallbackStrategy):
    """Fail silently and return None."""

    def __init__(self, name: str = "fail_silent"):
        super().__init__(name)

    def execute(self, error: Exception, context: Dict[str, Any]) -> Any:
        logger.debug(f"FailSilentFallback: suppressing error: {error}")
        return None


class FallbackChain:
    """
    Chain of fallback strategies.

    Tries each strategy in order until one succeeds.

    Usage:
        chain = FallbackChain()
        chain.add(CacheFallback())
        chain.add(DefaultFallback(default_value="N/A"))

        result = chain.execute(error, context)
    """

    def __init__(self, strategies: Optional[List[FallbackStrategy]] = None):
        self._strategies = strategies or []
        self._stats = {
            "total_executions": 0,
            "fallbacks_used": {},
        }

    def add(self, strategy: FallbackStrategy) -> "FallbackChain":
        """Add a fallback strategy to the chain."""
        self._strategies.append(strategy)
        return self

    def clear(self) -> None:
        """Clear all strategies."""
        self._strategies.clear()

    def execute(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> FallbackResult:
        """
        Execute fallback strategies in order.

        Args:
            error: The original error that triggered the fallback
            context: Context information for the fallback

        Returns:
            FallbackResult with the value and metadata
        """
        context = context or {}
        self._stats["total_executions"] += 1

        for strategy in self._strategies:
            if not strategy.can_handle(error):
                continue

            try:
                value = strategy.execute(error, context)

                # Track usage
                self._stats["fallbacks_used"][strategy.name] = (
                    self._stats["fallbacks_used"].get(strategy.name, 0) + 1
                )

                return FallbackResult(
                    value=value,
                    fallback_used=True,
                    fallback_type=FallbackType(strategy.name) if strategy.name in FallbackType.__members__.values() else None,
                    original_error=error,
                )

            except Exception as e:
                logger.warning(
                    f"FallbackChain: strategy '{strategy.name}' failed: {e}, trying next..."
                )
                continue

        # All fallbacks failed
        logger.error(f"FallbackChain: all {len(self._strategies)} strategies failed")
        return FallbackResult(
            value=None,
            fallback_used=False,
            original_error=error,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get fallback chain statistics."""
        return self._stats.copy()


def with_fallback(
    fallback_value: Any = None,
    fallback_func: Optional[Callable[..., Any]] = None,
    cache_key: Optional[str] = None,
    silent: bool = False,
) -> Callable:
    """
    Decorator to add fallback behavior to a function.

    Usage:
        @with_fallback(fallback_value="default")
        async def my_function():
            ...

        @with_fallback(fallback_func=alternative_function)
        def another_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        chain = FallbackChain()

        if cache_key:
            chain.add(CacheFallback())

        if fallback_func:
            alt = AlternativeFallback()
            alt.set_alternative(fallback_func)
            chain.add(alt)

        if fallback_value is not None:
            chain.add(DefaultFallback(default_value=fallback_value))

        if silent:
            chain.add(FailSilentFallback())

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "args": args,
                    "kwargs": kwargs,
                    "cache_key": cache_key or func.__name__,
                }
                result = chain.execute(e, context)
                return result.value

        return wrapper

    return decorator
