"""
Rate Limiting Module for Claude Multi-Agent Coordination System.

Provides rate limiting per identity to prevent abuse.
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
import json
from pathlib import Path


class RateLimitExceeded(Exception):
    """Raised when a rate limit is exceeded."""
    def __init__(
        self,
        message: str,
        identity: str,
        limit_name: str,
        retry_after_seconds: float,
    ):
        self.identity = identity
        self.limit_name = limit_name
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit."""
    name: str
    max_requests: int
    window_seconds: float
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    burst_size: Optional[int] = None  # For token bucket, allows burst
    penalty_seconds: float = 0  # Extra wait time after limit hit
    apply_to_roles: Optional[List[str]] = None  # If None, applies to all


@dataclass
class RateLimitState:
    """Internal state for tracking rate limits."""
    tokens: float = 0
    last_update: float = field(default_factory=time.time)
    request_times: List[float] = field(default_factory=list)
    window_count: int = 0
    window_start: float = field(default_factory=time.time)
    penalty_until: float = 0


class RateLimiter:
    """
    Rate limiter with multiple algorithms and per-identity tracking.

    Features:
    - Token bucket algorithm (default, allows bursting)
    - Sliding window (smooth limiting)
    - Fixed window (simple, time-based)
    - Leaky bucket (constant rate)
    - Per-identity limits
    - Multiple limit configurations
    - Penalty periods for abusers
    - Persistence support
    """

    def __init__(
        self,
        default_config: Optional[RateLimitConfig] = None,
        persistence_path: Optional[str] = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            default_config: Default rate limit configuration
            persistence_path: Path to persist rate limit state
        """
        self._configs: Dict[str, RateLimitConfig] = {}
        self._state: Dict[str, Dict[str, RateLimitState]] = {}  # identity -> {limit_name -> state}
        self._lock = threading.Lock()
        self.persistence_path = Path(persistence_path) if persistence_path else None

        if default_config:
            self.add_config(default_config)

        if self.persistence_path and self.persistence_path.exists():
            self._load_state()

    def add_config(self, config: RateLimitConfig) -> None:
        """Add a rate limit configuration."""
        self._configs[config.name] = config

    def remove_config(self, name: str) -> bool:
        """Remove a rate limit configuration."""
        if name in self._configs:
            del self._configs[name]
            return True
        return False

    def check_rate_limit(
        self,
        identity: str,
        limit_name: Optional[str] = None,
        role: Optional[str] = None,
        cost: int = 1,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a request is allowed under rate limits.

        Args:
            identity: The identity making the request (agent_id, user_id, etc.)
            limit_name: Specific limit to check (None checks all applicable)
            role: The role of the identity (for role-based limits)
            cost: The cost of this request (default 1)

        Returns:
            (is_allowed, info_dict with remaining tokens, retry_after, etc.)
        """
        with self._lock:
            configs_to_check = self._get_applicable_configs(limit_name, role)

            results = {}
            all_allowed = True
            min_retry_after = 0

            for config in configs_to_check:
                state = self._get_or_create_state(identity, config.name, config)
                allowed, info = self._check_single_limit(config, state, cost)

                results[config.name] = info
                if not allowed:
                    all_allowed = False
                    if info.get('retry_after', 0) > min_retry_after:
                        min_retry_after = info['retry_after']

            return all_allowed, {
                'allowed': all_allowed,
                'limits': results,
                'retry_after': min_retry_after,
            }

    def consume(
        self,
        identity: str,
        limit_name: Optional[str] = None,
        role: Optional[str] = None,
        cost: int = 1,
        raise_on_limit: bool = True,
    ) -> Dict[str, Any]:
        """
        Consume rate limit tokens for a request.

        Args:
            identity: The identity making the request
            limit_name: Specific limit to consume from
            role: The role of the identity
            cost: The cost of this request
            raise_on_limit: Whether to raise RateLimitExceeded if limit hit

        Returns:
            Info dict with remaining tokens, etc.

        Raises:
            RateLimitExceeded: If limit exceeded and raise_on_limit is True
        """
        allowed, info = self.check_rate_limit(identity, limit_name, role, cost)

        if not allowed:
            if raise_on_limit:
                # Find which limit was exceeded
                exceeded_limit = None
                for limit_name, limit_info in info.get('limits', {}).items():
                    if not limit_info.get('allowed', True):
                        exceeded_limit = limit_name
                        break

                raise RateLimitExceeded(
                    f"Rate limit exceeded for {identity}",
                    identity=identity,
                    limit_name=exceeded_limit or "unknown",
                    retry_after_seconds=info.get('retry_after', 0),
                )
            return info

        # Actually consume the tokens
        with self._lock:
            configs_to_check = self._get_applicable_configs(limit_name, role)

            for config in configs_to_check:
                state = self._get_or_create_state(identity, config.name, config)
                self._consume_single_limit(config, state, cost)

            self._save_state()

        return info

    def get_status(self, identity: str) -> Dict[str, Any]:
        """Get current rate limit status for an identity."""
        with self._lock:
            if identity not in self._state:
                return {'identity': identity, 'limits': {}}

            limits = {}
            for limit_name, state in self._state[identity].items():
                config = self._configs.get(limit_name)
                if not config:
                    continue

                limits[limit_name] = self._calculate_status(config, state)

            return {'identity': identity, 'limits': limits}

    def reset(self, identity: str, limit_name: Optional[str] = None) -> None:
        """Reset rate limits for an identity."""
        with self._lock:
            if identity in self._state:
                if limit_name:
                    if limit_name in self._state[identity]:
                        del self._state[identity][limit_name]
                else:
                    del self._state[identity]
            self._save_state()

    def set_penalty(self, identity: str, penalty_seconds: float) -> None:
        """Set a penalty period for an identity."""
        with self._lock:
            penalty_until = time.time() + penalty_seconds
            if identity not in self._state:
                self._state[identity] = {}
            for limit_name in self._configs:
                if limit_name not in self._state[identity]:
                    self._state[identity][limit_name] = RateLimitState()
                self._state[identity][limit_name].penalty_until = penalty_until
            self._save_state()

    def _get_applicable_configs(
        self,
        limit_name: Optional[str],
        role: Optional[str],
    ) -> List[RateLimitConfig]:
        """Get configs that apply to this request."""
        if limit_name:
            config = self._configs.get(limit_name)
            return [config] if config else []

        configs = []
        for config in self._configs.values():
            if config.apply_to_roles is None or (role and role in config.apply_to_roles):
                configs.append(config)
        return configs

    def _get_or_create_state(
        self,
        identity: str,
        limit_name: str,
        config: RateLimitConfig,
    ) -> RateLimitState:
        """Get or create state for an identity/limit combination."""
        if identity not in self._state:
            self._state[identity] = {}

        if limit_name not in self._state[identity]:
            burst = config.burst_size or config.max_requests
            self._state[identity][limit_name] = RateLimitState(
                tokens=burst,
                last_update=time.time(),
                window_start=time.time(),
            )

        return self._state[identity][limit_name]

    def _check_single_limit(
        self,
        config: RateLimitConfig,
        state: RateLimitState,
        cost: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check a single rate limit."""
        now = time.time()

        # Check penalty period
        if state.penalty_until > now:
            return False, {
                'allowed': False,
                'retry_after': state.penalty_until - now,
                'reason': 'penalty',
            }

        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return self._check_token_bucket(config, state, cost, now)
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return self._check_sliding_window(config, state, cost, now)
        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return self._check_fixed_window(config, state, cost, now)
        elif config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            return self._check_leaky_bucket(config, state, cost, now)

        return True, {'allowed': True}

    def _check_token_bucket(
        self,
        config: RateLimitConfig,
        state: RateLimitState,
        cost: int,
        now: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket algorithm check."""
        # Refill tokens based on time passed
        time_passed = now - state.last_update
        refill_rate = config.max_requests / config.window_seconds
        tokens_to_add = time_passed * refill_rate

        burst = config.burst_size or config.max_requests
        new_tokens = min(burst, state.tokens + tokens_to_add)

        if new_tokens >= cost:
            return True, {
                'allowed': True,
                'remaining': new_tokens - cost,
                'limit': burst,
                'reset_in': config.window_seconds,
            }

        # Calculate retry after
        tokens_needed = cost - new_tokens
        retry_after = tokens_needed / refill_rate + config.penalty_seconds

        return False, {
            'allowed': False,
            'remaining': 0,
            'limit': burst,
            'retry_after': retry_after,
        }

    def _check_sliding_window(
        self,
        config: RateLimitConfig,
        state: RateLimitState,
        cost: int,
        now: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window algorithm check."""
        # Remove old requests outside the window
        window_start = now - config.window_seconds
        state.request_times = [t for t in state.request_times if t > window_start]

        current_count = len(state.request_times)

        if current_count + cost <= config.max_requests:
            return True, {
                'allowed': True,
                'remaining': config.max_requests - current_count - cost,
                'limit': config.max_requests,
                'window_seconds': config.window_seconds,
            }

        # Calculate retry after (when oldest request exits the window)
        if state.request_times:
            oldest = state.request_times[0]
            retry_after = (oldest + config.window_seconds) - now + config.penalty_seconds
        else:
            retry_after = config.penalty_seconds

        return False, {
            'allowed': False,
            'remaining': 0,
            'limit': config.max_requests,
            'retry_after': max(0, retry_after),
        }

    def _check_fixed_window(
        self,
        config: RateLimitConfig,
        state: RateLimitState,
        cost: int,
        now: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window algorithm check."""
        # Check if window has reset
        if now - state.window_start >= config.window_seconds:
            state.window_start = now
            state.window_count = 0

        if state.window_count + cost <= config.max_requests:
            return True, {
                'allowed': True,
                'remaining': config.max_requests - state.window_count - cost,
                'limit': config.max_requests,
                'reset_at': state.window_start + config.window_seconds,
            }

        retry_after = (state.window_start + config.window_seconds) - now + config.penalty_seconds

        return False, {
            'allowed': False,
            'remaining': 0,
            'limit': config.max_requests,
            'retry_after': max(0, retry_after),
        }

    def _check_leaky_bucket(
        self,
        config: RateLimitConfig,
        state: RateLimitState,
        cost: int,
        now: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Leaky bucket algorithm check."""
        # Similar to token bucket but with fixed drain rate
        time_passed = now - state.last_update
        drain_rate = 1 / (config.window_seconds / config.max_requests)
        drained = time_passed * drain_rate

        burst = config.burst_size or config.max_requests
        current_level = max(0, state.tokens - drained)

        if current_level + cost <= burst:
            return True, {
                'allowed': True,
                'current_level': current_level + cost,
                'capacity': burst,
            }

        # Queue is full
        overflow = (current_level + cost) - burst
        retry_after = overflow / drain_rate + config.penalty_seconds

        return False, {
            'allowed': False,
            'current_level': current_level,
            'capacity': burst,
            'retry_after': retry_after,
        }

    def _consume_single_limit(
        self,
        config: RateLimitConfig,
        state: RateLimitState,
        cost: int,
    ) -> None:
        """Actually consume tokens/increment counters."""
        now = time.time()

        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            time_passed = now - state.last_update
            refill_rate = config.max_requests / config.window_seconds
            tokens_to_add = time_passed * refill_rate
            burst = config.burst_size or config.max_requests
            state.tokens = min(burst, state.tokens + tokens_to_add) - cost
            state.last_update = now

        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            for _ in range(cost):
                state.request_times.append(now)

        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            if now - state.window_start >= config.window_seconds:
                state.window_start = now
                state.window_count = 0
            state.window_count += cost

        elif config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            time_passed = now - state.last_update
            drain_rate = 1 / (config.window_seconds / config.max_requests)
            drained = time_passed * drain_rate
            state.tokens = max(0, state.tokens - drained) + cost
            state.last_update = now

    def _calculate_status(
        self,
        config: RateLimitConfig,
        state: RateLimitState,
    ) -> Dict[str, Any]:
        """Calculate current status for a limit."""
        now = time.time()
        allowed, info = self._check_single_limit(config, state, 0)

        return {
            'name': config.name,
            'algorithm': config.algorithm.value,
            'max_requests': config.max_requests,
            'window_seconds': config.window_seconds,
            **info,
        }

    def _load_state(self) -> None:
        """Load persisted state."""
        if not self.persistence_path:
            return
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)

            for identity, limits in data.get('state', {}).items():
                self._state[identity] = {}
                for limit_name, state_data in limits.items():
                    self._state[identity][limit_name] = RateLimitState(
                        tokens=state_data.get('tokens', 0),
                        last_update=state_data.get('last_update', time.time()),
                        request_times=state_data.get('request_times', []),
                        window_count=state_data.get('window_count', 0),
                        window_start=state_data.get('window_start', time.time()),
                        penalty_until=state_data.get('penalty_until', 0),
                    )
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    def _save_state(self) -> None:
        """Persist state to file."""
        if not self.persistence_path:
            return

        state_data = {}
        for identity, limits in self._state.items():
            state_data[identity] = {}
            for limit_name, state in limits.items():
                state_data[identity][limit_name] = {
                    'tokens': state.tokens,
                    'last_update': state.last_update,
                    'request_times': state.request_times[-100:],  # Keep last 100
                    'window_count': state.window_count,
                    'window_start': state.window_start,
                    'penalty_until': state.penalty_until,
                }

        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persistence_path, 'w') as f:
            json.dump({'state': state_data}, f)


# Convenience function for creating common rate limits
def create_default_rate_limits() -> List[RateLimitConfig]:
    """Create a set of sensible default rate limits."""
    return [
        # General API rate limit
        RateLimitConfig(
            name="api_general",
            max_requests=100,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_size=20,
        ),
        # Task creation limit
        RateLimitConfig(
            name="task_create",
            max_requests=30,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            apply_to_roles=["leader", "admin"],
        ),
        # Task claim limit (for workers)
        RateLimitConfig(
            name="task_claim",
            max_requests=10,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            apply_to_roles=["worker"],
        ),
        # Discovery creation limit
        RateLimitConfig(
            name="discovery_create",
            max_requests=20,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        ),
        # Authentication attempts limit
        RateLimitConfig(
            name="auth_attempts",
            max_requests=5,
            window_seconds=300,  # 5 minutes
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            penalty_seconds=60,  # 1 minute penalty after hitting limit
        ),
    ]
