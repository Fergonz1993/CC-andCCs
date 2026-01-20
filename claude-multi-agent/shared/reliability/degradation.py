"""
Graceful Degradation Modes (adv-rel-009)

Implements graceful degradation patterns to maintain partial functionality
when the system is under stress or experiencing failures.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable, Set
from enum import Enum

logger = logging.getLogger(__name__)


class DegradationLevel(str, Enum):
    """Levels of degradation."""
    NORMAL = "normal"           # Full functionality
    LIGHT = "light"             # Minor feature restrictions
    MODERATE = "moderate"       # Some features disabled
    HEAVY = "heavy"             # Most non-essential features disabled
    CRITICAL = "critical"       # Minimal functionality only
    EMERGENCY = "emergency"     # Emergency mode, minimal operations


class FeatureFlag(str, Enum):
    """Features that can be enabled/disabled during degradation."""
    TASK_CREATION = "task_creation"
    TASK_CLAIMING = "task_claiming"
    PARALLEL_EXECUTION = "parallel_execution"
    DISCOVERIES = "discoveries"
    METRICS_COLLECTION = "metrics_collection"
    AUTO_BACKUP = "auto_backup"
    RESULT_AGGREGATION = "result_aggregation"
    WORKER_SCALING = "worker_scaling"
    LOGGING_VERBOSE = "logging_verbose"


@dataclass
class DegradationPolicy:
    """Policy for degradation at each level."""
    level: DegradationLevel
    max_workers: int
    max_tasks_per_minute: int
    enabled_features: Set[FeatureFlag]
    description: str


@dataclass
class SystemHealth:
    """Current system health metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    queue_depth: int = 0
    active_workers: int = 0
    failed_tasks_recent: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


# Default degradation policies
DEFAULT_POLICIES = {
    DegradationLevel.NORMAL: DegradationPolicy(
        level=DegradationLevel.NORMAL,
        max_workers=10,
        max_tasks_per_minute=100,
        enabled_features=set(FeatureFlag),
        description="Full functionality",
    ),
    DegradationLevel.LIGHT: DegradationPolicy(
        level=DegradationLevel.LIGHT,
        max_workers=8,
        max_tasks_per_minute=80,
        enabled_features=set(FeatureFlag) - {FeatureFlag.LOGGING_VERBOSE},
        description="Reduced logging verbosity",
    ),
    DegradationLevel.MODERATE: DegradationPolicy(
        level=DegradationLevel.MODERATE,
        max_workers=5,
        max_tasks_per_minute=50,
        enabled_features={
            FeatureFlag.TASK_CREATION,
            FeatureFlag.TASK_CLAIMING,
            FeatureFlag.PARALLEL_EXECUTION,
            FeatureFlag.DISCOVERIES,
            FeatureFlag.AUTO_BACKUP,
        },
        description="Metrics and scaling disabled",
    ),
    DegradationLevel.HEAVY: DegradationPolicy(
        level=DegradationLevel.HEAVY,
        max_workers=3,
        max_tasks_per_minute=20,
        enabled_features={
            FeatureFlag.TASK_CREATION,
            FeatureFlag.TASK_CLAIMING,
            FeatureFlag.AUTO_BACKUP,
        },
        description="Limited to core task operations",
    ),
    DegradationLevel.CRITICAL: DegradationPolicy(
        level=DegradationLevel.CRITICAL,
        max_workers=1,
        max_tasks_per_minute=5,
        enabled_features={
            FeatureFlag.TASK_CREATION,
            FeatureFlag.TASK_CLAIMING,
        },
        description="Single worker, minimal operations",
    ),
    DegradationLevel.EMERGENCY: DegradationPolicy(
        level=DegradationLevel.EMERGENCY,
        max_workers=0,
        max_tasks_per_minute=0,
        enabled_features=set(),
        description="Emergency mode - no new operations",
    ),
}


class GracefulDegradation:
    """
    Manages graceful degradation of the coordination system.

    Features:
    - Automatic degradation based on system health
    - Manual degradation level control
    - Feature flags for selective disabling
    - Gradual recovery when health improves
    """

    def __init__(
        self,
        policies: Optional[Dict[DegradationLevel, DegradationPolicy]] = None,
        on_degrade: Optional[Callable[[DegradationLevel, DegradationLevel], None]] = None,
        on_recover: Optional[Callable[[DegradationLevel, DegradationLevel], None]] = None,
        auto_degrade: bool = True,
        recovery_delay: float = 60.0,  # Delay before attempting recovery
    ):
        """
        Initialize graceful degradation.

        Args:
            policies: Degradation policies for each level
            on_degrade: Callback when system degrades (old_level, new_level)
            on_recover: Callback when system recovers (old_level, new_level)
            auto_degrade: Whether to automatically degrade based on health
            recovery_delay: Minimum time before attempting recovery
        """
        self.policies = policies or DEFAULT_POLICIES
        self.on_degrade = on_degrade
        self.on_recover = on_recover
        self.auto_degrade = auto_degrade
        self.recovery_delay = recovery_delay

        self._current_level = DegradationLevel.NORMAL
        self._last_level_change = datetime.now()
        self._health_history: List[SystemHealth] = []
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Thresholds for automatic degradation
        self._thresholds = {
            "error_rate_light": 0.05,      # 5% error rate
            "error_rate_moderate": 0.10,   # 10% error rate
            "error_rate_heavy": 0.20,      # 20% error rate
            "error_rate_critical": 0.40,   # 40% error rate
            "queue_depth_light": 50,
            "queue_depth_moderate": 100,
            "queue_depth_heavy": 200,
            "failed_tasks_moderate": 5,
            "failed_tasks_heavy": 10,
        }

    @property
    def current_level(self) -> DegradationLevel:
        """Get current degradation level."""
        return self._current_level

    @property
    def current_policy(self) -> DegradationPolicy:
        """Get current degradation policy."""
        return self.policies[self._current_level]

    def is_feature_enabled(self, feature: FeatureFlag) -> bool:
        """Check if a feature is enabled at current degradation level."""
        return feature in self.current_policy.enabled_features

    def get_max_workers(self) -> int:
        """Get maximum allowed workers at current level."""
        return self.current_policy.max_workers

    def get_rate_limit(self) -> int:
        """Get task rate limit at current level."""
        return self.current_policy.max_tasks_per_minute

    def set_level(self, level: DegradationLevel, reason: str = "") -> None:
        """
        Manually set degradation level.

        Args:
            level: New degradation level
            reason: Reason for the change
        """
        with self._lock:
            if level == self._current_level:
                return

            old_level = self._current_level
            self._current_level = level
            self._last_level_change = datetime.now()

            is_degrading = list(DegradationLevel).index(level) > list(DegradationLevel).index(old_level)

            logger.warning(
                f"Degradation level changed: {old_level.value} -> {level.value}"
                f" ({reason or 'manual change'})"
            )

            if is_degrading and self.on_degrade:
                try:
                    self.on_degrade(old_level, level)
                except Exception as e:
                    logger.error(f"on_degrade callback error: {e}")

            if not is_degrading and self.on_recover:
                try:
                    self.on_recover(old_level, level)
                except Exception as e:
                    logger.error(f"on_recover callback error: {e}")

    def degrade(self, reason: str = "") -> DegradationLevel:
        """
        Degrade to the next level.

        Returns the new level.
        """
        with self._lock:
            levels = list(DegradationLevel)
            current_idx = levels.index(self._current_level)

            if current_idx < len(levels) - 1:
                new_level = levels[current_idx + 1]
                self.set_level(new_level, reason)
                return new_level

            return self._current_level

    def recover(self, reason: str = "") -> DegradationLevel:
        """
        Recover to the previous level.

        Returns the new level.
        """
        with self._lock:
            # Check if enough time has passed since last change
            time_since_change = datetime.now() - self._last_level_change
            if time_since_change.total_seconds() < self.recovery_delay:
                logger.debug(
                    f"Recovery delayed: {self.recovery_delay - time_since_change.total_seconds():.0f}s remaining"
                )
                return self._current_level

            levels = list(DegradationLevel)
            current_idx = levels.index(self._current_level)

            if current_idx > 0:
                new_level = levels[current_idx - 1]
                self.set_level(new_level, reason)
                return new_level

            return self._current_level

    def report_health(self, health: SystemHealth) -> None:
        """
        Report current system health.

        Used for automatic degradation decisions.
        """
        with self._lock:
            self._health_history.append(health)

            # Keep only recent history
            cutoff = datetime.now() - timedelta(minutes=5)
            self._health_history = [
                h for h in self._health_history
                if h.timestamp > cutoff
            ]

            if self.auto_degrade:
                self._evaluate_health(health)

    def _evaluate_health(self, health: SystemHealth) -> None:
        """Evaluate health and adjust degradation level."""
        # Calculate average error rate from history
        if self._health_history:
            avg_error_rate = sum(h.error_rate for h in self._health_history) / len(self._health_history)
        else:
            avg_error_rate = health.error_rate

        # Determine appropriate level based on health
        if avg_error_rate >= self._thresholds["error_rate_critical"]:
            target = DegradationLevel.CRITICAL
            reason = f"High error rate: {avg_error_rate:.1%}"
        elif avg_error_rate >= self._thresholds["error_rate_heavy"]:
            target = DegradationLevel.HEAVY
            reason = f"Elevated error rate: {avg_error_rate:.1%}"
        elif avg_error_rate >= self._thresholds["error_rate_moderate"]:
            target = DegradationLevel.MODERATE
            reason = f"Moderate error rate: {avg_error_rate:.1%}"
        elif avg_error_rate >= self._thresholds["error_rate_light"]:
            target = DegradationLevel.LIGHT
            reason = f"Light error rate: {avg_error_rate:.1%}"
        elif health.queue_depth >= self._thresholds["queue_depth_heavy"]:
            target = DegradationLevel.HEAVY
            reason = f"High queue depth: {health.queue_depth}"
        elif health.queue_depth >= self._thresholds["queue_depth_moderate"]:
            target = DegradationLevel.MODERATE
            reason = f"Elevated queue depth: {health.queue_depth}"
        elif health.queue_depth >= self._thresholds["queue_depth_light"]:
            target = DegradationLevel.LIGHT
            reason = f"Light queue buildup: {health.queue_depth}"
        elif health.failed_tasks_recent >= self._thresholds["failed_tasks_heavy"]:
            target = DegradationLevel.HEAVY
            reason = f"Many recent failures: {health.failed_tasks_recent}"
        elif health.failed_tasks_recent >= self._thresholds["failed_tasks_moderate"]:
            target = DegradationLevel.MODERATE
            reason = f"Recent failures: {health.failed_tasks_recent}"
        else:
            target = DegradationLevel.NORMAL
            reason = "Health metrics normal"

        # Apply change if needed
        levels = list(DegradationLevel)
        current_idx = levels.index(self._current_level)
        target_idx = levels.index(target)

        if target_idx > current_idx:
            # Need to degrade
            self.degrade(reason)
        elif target_idx < current_idx:
            # Can potentially recover
            self.recover(reason)

    def start_monitoring(
        self,
        get_health: Callable[[], SystemHealth],
        interval: float = 10.0,
    ) -> None:
        """
        Start automatic health monitoring.

        Args:
            get_health: Function to get current system health
            interval: Monitoring interval in seconds
        """
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(get_health, interval),
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Started degradation monitoring")

    def stop_monitoring(self) -> None:
        """Stop automatic monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("Stopped degradation monitoring")

    def _monitor_loop(
        self,
        get_health: Callable[[], SystemHealth],
        interval: float,
    ) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                health = get_health()
                self.report_health(health)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

            time.sleep(interval)

    def get_status(self) -> Dict[str, Any]:
        """Get degradation status."""
        policy = self.current_policy
        return {
            "current_level": self._current_level.value,
            "description": policy.description,
            "max_workers": policy.max_workers,
            "max_tasks_per_minute": policy.max_tasks_per_minute,
            "enabled_features": [f.value for f in policy.enabled_features],
            "last_level_change": self._last_level_change.isoformat(),
            "health_history_size": len(self._health_history),
            "auto_degrade": self.auto_degrade,
        }

    def emergency_stop(self) -> None:
        """Immediately go to emergency mode."""
        self.set_level(DegradationLevel.EMERGENCY, "Emergency stop triggered")

    def reset_to_normal(self) -> None:
        """Reset to normal operation (use with caution)."""
        self.set_level(DegradationLevel.NORMAL, "Manual reset to normal")
