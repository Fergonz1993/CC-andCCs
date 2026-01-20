"""
Circuit Breaker Pattern Implementation (adv-rel-001)

Implements the circuit breaker pattern to prevent cascading failures when
external services or workers fail repeatedly. The circuit breaker monitors
failure rates and opens the circuit when the threshold is exceeded.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Circuit is open, requests are rejected immediately
- HALF_OPEN: Testing if the service has recovered
"""

import time
import threading
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Rejecting all calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 3  # Successes in half-open before closing
    timeout_seconds: float = 30.0  # Time before attempting recovery
    window_seconds: float = 60.0  # Rolling window for failure tracking
    half_open_max_calls: int = 3  # Max concurrent calls in half-open state


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker instance."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None


class CircuitBreaker:
    """
    Circuit breaker for external calls.

    Tracks failure rates for workers, external services, or any callable.
    Opens the circuit when failures exceed threshold, preventing further
    calls until the service recovers.

    Usage:
        breaker = CircuitBreaker("worker-1", config=CircuitBreakerConfig(failure_threshold=3))

        # Use as decorator
        @breaker
        async def call_external_service():
            ...

        # Or use explicitly
        if breaker.can_execute():
            try:
                result = do_something()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = None,
        on_open: Optional[Callable[[str], None]] = None,
        on_close: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize a circuit breaker.

        Args:
            name: Identifier for this circuit breaker (e.g., "worker-1", "api-service")
            config: Configuration options
            on_state_change: Callback when state changes (name, old_state, new_state)
            on_open: Callback when circuit opens
            on_close: Callback when circuit closes
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change
        self.on_open = on_open
        self.on_close = on_close

        # State
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()

        # Tracking (using deque for sliding window)
        self._failure_times: deque[float] = deque()
        self._last_failure_time: Optional[float] = None
        self._last_state_change_time: float = time.time()

        # Half-open state tracking
        self._half_open_successes: int = 0
        self._half_open_failures: int = 0
        self._half_open_calls: int = 0

        # Statistics
        self._stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return self._stats

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    def can_execute(self) -> bool:
        """
        Check if a call can be executed.

        Returns True if circuit is closed or half-open with capacity.
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:  # OPEN
                return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.last_success_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                self._half_open_calls = max(0, self._half_open_calls - 1)

                # Check if we can close the circuit
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

            logger.debug(f"CircuitBreaker {self.name}: recorded success (state={self._state.value})")

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed call."""
        with self._lock:
            current_time = time.time()
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.last_failure_time = datetime.now()
            self._last_failure_time = current_time

            # Add to sliding window
            self._failure_times.append(current_time)
            self._cleanup_old_failures()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_failures += 1
                self._half_open_calls = max(0, self._half_open_calls - 1)
                # Any failure in half-open reopens the circuit
                self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if len(self._failure_times) >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

            logger.warning(
                f"CircuitBreaker {self.name}: recorded failure "
                f"(state={self._state.value}, failures={len(self._failure_times)}, error={error})"
            )

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_times.clear()
            self._reset_half_open_counters()
            logger.info(f"CircuitBreaker {self.name}: manually reset")

    def _check_state_transition(self) -> None:
        """Check if the circuit should transition states based on timeout."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_state_change_time
            if elapsed >= self.config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        self._last_state_change_time = time.time()
        self._stats.state_changes += 1
        self._stats.last_state_change = datetime.now()

        if new_state == CircuitState.HALF_OPEN:
            self._reset_half_open_counters()
        elif new_state == CircuitState.CLOSED:
            self._failure_times.clear()

        logger.info(f"CircuitBreaker {self.name}: {old_state.value} -> {new_state.value}")

        # Callbacks
        if self.on_state_change:
            try:
                self.on_state_change(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"CircuitBreaker callback error: {e}")

        if new_state == CircuitState.OPEN and self.on_open:
            try:
                self.on_open(self.name)
            except Exception as e:
                logger.error(f"CircuitBreaker on_open callback error: {e}")

        if new_state == CircuitState.CLOSED and self.on_close:
            try:
                self.on_close(self.name)
            except Exception as e:
                logger.error(f"CircuitBreaker on_close callback error: {e}")

    def _reset_half_open_counters(self) -> None:
        """Reset half-open state counters."""
        self._half_open_successes = 0
        self._half_open_failures = 0
        self._half_open_calls = 0

    def _cleanup_old_failures(self) -> None:
        """Remove failures outside the sliding window."""
        cutoff = time.time() - self.config.window_seconds
        while self._failure_times and self._failure_times[0] < cutoff:
            self._failure_times.popleft()

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator for sync functions."""
        def wrapper(*args, **kwargs) -> Any:
            if not self.can_execute():
                self._stats.rejected_calls += 1
                raise CircuitOpenError(
                    f"Circuit breaker {self.name} is open, call rejected"
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper

    def to_dict(self) -> Dict[str, Any]:
        """Serialize circuit breaker state to dictionary."""
        return {
            "name": self.name,
            "state": self._state.value,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "window_seconds": self.config.window_seconds,
            },
            "stats": {
                "total_calls": self._stats.total_calls,
                "successful_calls": self._stats.successful_calls,
                "failed_calls": self._stats.failed_calls,
                "rejected_calls": self._stats.rejected_calls,
                "state_changes": self._stats.state_changes,
            },
            "current_failures": len(self._failure_times),
            "half_open_successes": self._half_open_successes,
        }


class CircuitOpenError(Exception):
    """Raised when a call is rejected due to open circuit."""
    pass


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized management of circuit breakers for different
    workers, services, or components.
    """

    def __init__(self, default_config: Optional[CircuitBreakerConfig] = None):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        self._default_config = default_config or CircuitBreakerConfig()

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        **kwargs
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create a new one."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    config=config or self._default_config,
                    **kwargs
                )
            return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def remove(self, name: str) -> None:
        """Remove a circuit breaker."""
        with self._lock:
            self._breakers.pop(name, None)

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.to_dict() for name, breaker in self._breakers.items()}

    def get_open_circuits(self) -> list[str]:
        """Get names of all open circuits."""
        return [name for name, breaker in self._breakers.items() if breaker.is_open]
