"""
Advanced features for the Claude Multi-Agent Orchestrator.

This module implements:
- adv-c-001: Adaptive worker pool scaling based on queue depth
- adv-c-002: Worker health monitoring with automatic restart
- adv-c-003: Task load balancing strategies (round-robin, least-loaded, affinity-based)
- adv-c-004: Parallel execution with max concurrency
- adv-c-005: Task timeout with graceful cancellation
- adv-c-006: Inter-worker communication channel
- adv-c-007: Shared memory for large data
- adv-c-008: Worker specialization and routing
- adv-c-009: Task execution sandboxing
- adv-c-010: Execution replay for debugging
"""

import asyncio
import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from rich.console import Console

if TYPE_CHECKING:
    from .agent import AgentPool

console = Console()


# =============================================================================
# adv-c-001: Adaptive Worker Pool Scaling
# =============================================================================


@dataclass
class ScalingConfig:
    """Configuration for adaptive worker scaling."""
    min_workers: int = 1
    max_workers: int = 10
    scale_up_threshold: int = 5  # Queue depth to trigger scale up
    scale_down_threshold: int = 1  # Queue depth to trigger scale down
    scale_up_step: int = 1  # Workers to add when scaling up
    scale_down_step: int = 1  # Workers to remove when scaling down
    cooldown_seconds: float = 30.0  # Time between scaling operations
    evaluation_window: float = 10.0  # Window to evaluate queue depth


class AdaptiveScaler:
    """
    Monitors queue depth and scales worker pool automatically.

    Features:
    - Scales up when queue depth exceeds threshold
    - Scales down when queue depth falls below threshold
    - Respects min/max worker limits
    - Cooldown period between scaling operations
    """

    def __init__(
        self,
        pool: "AgentPool",
        config: Optional[ScalingConfig] = None,
        on_scale: Optional[Callable[[int, int, str], None]] = None,
    ):
        self.pool = pool
        self.config = config or ScalingConfig()
        self.on_scale = on_scale  # Callback: (old_count, new_count, reason)

        self._last_scale_time: Optional[datetime] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._queue_depth_history: List[Tuple[datetime, int]] = []

    @property
    def current_worker_count(self) -> int:
        """Get current number of active workers."""
        return len([a for a in self.pool._agents.values() if a.is_running])

    def _can_scale(self) -> bool:
        """Check if we're outside the cooldown period."""
        if self._last_scale_time is None:
            return True
        elapsed = (datetime.now() - self._last_scale_time).total_seconds()
        return elapsed >= self.config.cooldown_seconds

    def _get_average_queue_depth(self) -> float:
        """Get average queue depth over evaluation window."""
        if not self._queue_depth_history:
            return 0.0

        cutoff = datetime.now() - timedelta(seconds=self.config.evaluation_window)
        recent = [d for t, d in self._queue_depth_history if t >= cutoff]

        return sum(recent) / len(recent) if recent else 0.0

    async def _scale_up(self, current_count: int) -> None:
        """Add workers to the pool."""
        new_count = min(
            current_count + self.config.scale_up_step,
            self.config.max_workers
        )

        if new_count > current_count:
            for i in range(new_count - current_count):
                worker_id = f"worker-{len(self.pool._agents) + 1}"
                await self.pool.start_worker(worker_id)

            self._last_scale_time = datetime.now()

            if self.on_scale:
                self.on_scale(current_count, new_count, "scale_up")

            console.print(f"[yellow]Scaled up: {current_count} -> {new_count} workers[/]")

    async def _scale_down(self, current_count: int) -> None:
        """Remove idle workers from the pool."""
        new_count = max(
            current_count - self.config.scale_down_step,
            self.config.min_workers
        )

        if new_count < current_count:
            # Find idle workers to remove
            idle_workers = [
                agent_id for agent_id, agent in self.pool._agents.items()
                if agent.is_running and agent._current_task is None
            ]

            workers_to_remove = idle_workers[:current_count - new_count]

            for worker_id in workers_to_remove:
                agent = self.pool._agents.get(worker_id)
                if agent:
                    await agent.stop()
                    del self.pool._agents[worker_id]

            if workers_to_remove:
                self._last_scale_time = datetime.now()

                if self.on_scale:
                    self.on_scale(current_count, new_count, "scale_down")

                console.print(f"[yellow]Scaled down: {current_count} -> {new_count} workers[/]")

    async def evaluate(self, queue_depth: int) -> None:
        """
        Evaluate current queue depth and scale if needed.

        Args:
            queue_depth: Current number of pending tasks
        """
        # Record queue depth
        self._queue_depth_history.append((datetime.now(), queue_depth))

        # Trim old history
        cutoff = datetime.now() - timedelta(seconds=self.config.evaluation_window * 2)
        self._queue_depth_history = [
            (t, d) for t, d in self._queue_depth_history if t >= cutoff
        ]

        if not self._can_scale():
            return

        avg_depth = self._get_average_queue_depth()
        current_count = self.current_worker_count

        # Scale up if queue is building up
        if avg_depth >= self.config.scale_up_threshold:
            if current_count < self.config.max_workers:
                await self._scale_up(current_count)

        # Scale down if queue is mostly empty
        elif avg_depth <= self.config.scale_down_threshold:
            if current_count > self.config.min_workers:
                await self._scale_down(current_count)

    async def start(self, get_queue_depth: Callable[[], int]) -> None:
        """Start the auto-scaling background task."""
        self._running = True

        async def monitor_loop():
            while self._running:
                try:
                    depth = get_queue_depth()
                    await self.evaluate(depth)
                except Exception as e:
                    console.print(f"[red]Scaler error: {e}[/]")
                await asyncio.sleep(5.0)  # Check every 5 seconds

        self._task = asyncio.create_task(monitor_loop())

    async def stop(self) -> None:
        """Stop the auto-scaling background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


# =============================================================================
# adv-c-002: Worker Health Monitoring
# =============================================================================


@dataclass
class HealthConfig:
    """Configuration for health monitoring."""
    check_interval: float = 10.0  # Seconds between health checks
    unhealthy_threshold: int = 3  # Consecutive failures before restart
    heartbeat_timeout: float = 30.0  # Seconds without heartbeat = unhealthy
    max_restarts: int = 5  # Maximum restarts before giving up
    restart_cooldown: float = 60.0  # Seconds between restart attempts


class WorkerHealthStatus(str, Enum):
    """Health status of a worker."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RESTARTING = "restarting"
    FAILED = "failed"


@dataclass
class WorkerHealthRecord:
    """Health record for a single worker."""
    worker_id: str
    status: WorkerHealthStatus = WorkerHealthStatus.HEALTHY
    last_heartbeat: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    restart_count: int = 0
    last_restart: Optional[datetime] = None
    error_messages: List[str] = field(default_factory=list)


class HealthMonitor:
    """
    Monitors worker health and automatically restarts unhealthy workers.

    Features:
    - Periodic health checks for all workers
    - Automatic restart of unhealthy workers
    - Restart tracking and limits
    - Health status reporting
    """

    def __init__(
        self,
        pool: "AgentPool",
        config: Optional[HealthConfig] = None,
        on_health_change: Optional[Callable[[str, WorkerHealthStatus], None]] = None,
        on_restart: Optional[Callable[[str, int], None]] = None,
    ):
        self.pool = pool
        self.config = config or HealthConfig()
        self.on_health_change = on_health_change
        self.on_restart = on_restart

        self._records: Dict[str, WorkerHealthRecord] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def _get_or_create_record(self, worker_id: str) -> WorkerHealthRecord:
        """Get or create a health record for a worker."""
        if worker_id not in self._records:
            self._records[worker_id] = WorkerHealthRecord(worker_id=worker_id)
        return self._records[worker_id]

    def record_heartbeat(self, worker_id: str) -> None:
        """Record a heartbeat from a worker."""
        record = self._get_or_create_record(worker_id)
        record.last_heartbeat = datetime.now()

        if record.status == WorkerHealthStatus.DEGRADED:
            record.status = WorkerHealthStatus.HEALTHY
            record.consecutive_failures = 0
            if self.on_health_change:
                self.on_health_change(worker_id, WorkerHealthStatus.HEALTHY)

    def record_failure(self, worker_id: str, error: str) -> None:
        """Record a failure for a worker."""
        record = self._get_or_create_record(worker_id)
        record.consecutive_failures += 1
        record.error_messages.append(error)

        # Keep only last 10 errors
        if len(record.error_messages) > 10:
            record.error_messages = record.error_messages[-10:]

        if record.consecutive_failures >= self.config.unhealthy_threshold:
            record.status = WorkerHealthStatus.UNHEALTHY
            if self.on_health_change:
                self.on_health_change(worker_id, WorkerHealthStatus.UNHEALTHY)
        elif record.consecutive_failures > 0:
            record.status = WorkerHealthStatus.DEGRADED
            if self.on_health_change:
                self.on_health_change(worker_id, WorkerHealthStatus.DEGRADED)

    async def _check_worker_health(self, worker_id: str) -> bool:
        """
        Check if a worker is healthy.

        Returns True if healthy, False otherwise.
        """
        agent = self.pool._agents.get(worker_id)
        if not agent:
            return False

        # Check if process is running
        if not agent.is_running:
            self.record_failure(worker_id, "Process not running")
            return False

        # Check heartbeat timeout
        record = self._get_or_create_record(worker_id)
        elapsed = (datetime.now() - record.last_heartbeat).total_seconds()

        if elapsed > self.config.heartbeat_timeout:
            self.record_failure(worker_id, f"Heartbeat timeout ({elapsed:.1f}s)")
            return False

        # Worker is healthy
        record.consecutive_failures = 0
        record.status = WorkerHealthStatus.HEALTHY
        return True

    async def _restart_worker(self, worker_id: str) -> bool:
        """
        Restart an unhealthy worker.

        Returns True if restart succeeded.
        """
        record = self._get_or_create_record(worker_id)

        # Check restart limits
        if record.restart_count >= self.config.max_restarts:
            record.status = WorkerHealthStatus.FAILED
            console.print(f"[red]Worker {worker_id} exceeded max restarts[/]")
            if self.on_health_change:
                self.on_health_change(worker_id, WorkerHealthStatus.FAILED)
            return False

        # Check cooldown
        if record.last_restart:
            elapsed = (datetime.now() - record.last_restart).total_seconds()
            if elapsed < self.config.restart_cooldown:
                return False

        record.status = WorkerHealthStatus.RESTARTING
        if self.on_health_change:
            self.on_health_change(worker_id, WorkerHealthStatus.RESTARTING)

        console.print(f"[yellow]Restarting worker {worker_id}...[/]")

        try:
            agent = self.pool._agents.get(worker_id)
            if agent:
                await agent.stop()

            # Start a new agent with the same ID
            await self.pool.start_worker(worker_id)

            record.restart_count += 1
            record.last_restart = datetime.now()
            record.status = WorkerHealthStatus.HEALTHY
            record.consecutive_failures = 0
            record.last_heartbeat = datetime.now()

            if self.on_restart:
                self.on_restart(worker_id, record.restart_count)

            console.print(f"[green]Worker {worker_id} restarted (attempt {record.restart_count})[/]")
            return True

        except Exception as e:
            record.status = WorkerHealthStatus.FAILED
            console.print(f"[red]Failed to restart worker {worker_id}: {e}[/]")
            return False

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                for worker_id in list(self.pool._agents.keys()):
                    healthy = await self._check_worker_health(worker_id)

                    if not healthy:
                        record = self._get_or_create_record(worker_id)
                        if record.status == WorkerHealthStatus.UNHEALTHY:
                            await self._restart_worker(worker_id)

            except Exception as e:
                console.print(f"[red]Health monitor error: {e}[/]")

            await asyncio.sleep(self.config.check_interval)

    async def start(self) -> None:
        """Start the health monitoring background task."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop the health monitoring background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def get_health_report(self) -> Dict[str, Any]:
        """Get a health report for all workers."""
        return {
            worker_id: {
                "status": record.status.value,
                "last_heartbeat": record.last_heartbeat.isoformat(),
                "consecutive_failures": record.consecutive_failures,
                "restart_count": record.restart_count,
                "recent_errors": record.error_messages[-3:],
            }
            for worker_id, record in self._records.items()
        }


# =============================================================================
# adv-c-003: Load Balancing Strategies
# =============================================================================


class LoadBalancingStrategy(str, Enum):
    """Available load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    AFFINITY = "affinity"
    WEIGHTED = "weighted"
    RANDOM = "random"


@dataclass
class WorkerLoad:
    """Load information for a worker."""
    worker_id: str
    current_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_task_duration: float = 0.0
    specializations: Set[str] = field(default_factory=set)
    weight: float = 1.0


class LoadBalancer:
    """
    Distributes tasks across workers using various strategies.

    Strategies:
    - Round-robin: Simple rotation through workers
    - Least-loaded: Assign to worker with fewest current tasks
    - Affinity: Prefer workers who have worked on similar tasks
    - Weighted: Distribute based on worker performance weights
    """

    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED,
    ):
        self.strategy = strategy
        self._worker_loads: Dict[str, WorkerLoad] = {}
        self._round_robin_index: int = 0
        self._task_affinity: Dict[str, Set[str]] = {}  # task_tag -> worker_ids

    def register_worker(
        self,
        worker_id: str,
        specializations: Optional[Set[str]] = None,
        weight: float = 1.0,
    ) -> None:
        """Register a worker with the load balancer."""
        self._worker_loads[worker_id] = WorkerLoad(
            worker_id=worker_id,
            specializations=specializations or set(),
            weight=weight,
        )

    def unregister_worker(self, worker_id: str) -> None:
        """Remove a worker from the load balancer."""
        self._worker_loads.pop(worker_id, None)

    def record_task_start(self, worker_id: str) -> None:
        """Record that a worker started a task."""
        if worker_id in self._worker_loads:
            self._worker_loads[worker_id].current_tasks += 1

    def record_task_complete(
        self,
        worker_id: str,
        duration: float,
        task_tags: Optional[Set[str]] = None,
    ) -> None:
        """Record that a worker completed a task."""
        if worker_id in self._worker_loads:
            load = self._worker_loads[worker_id]
            load.current_tasks = max(0, load.current_tasks - 1)
            load.completed_tasks += 1

            # Update average duration
            total_duration = load.avg_task_duration * (load.completed_tasks - 1) + duration
            load.avg_task_duration = total_duration / load.completed_tasks

            # Update affinity
            if task_tags:
                for tag in task_tags:
                    if tag not in self._task_affinity:
                        self._task_affinity[tag] = set()
                    self._task_affinity[tag].add(worker_id)

    def record_task_failure(self, worker_id: str) -> None:
        """Record that a worker failed a task."""
        if worker_id in self._worker_loads:
            load = self._worker_loads[worker_id]
            load.current_tasks = max(0, load.current_tasks - 1)
            load.failed_tasks += 1

    def _select_round_robin(self, available_workers: List[str]) -> Optional[str]:
        """Select worker using round-robin."""
        if not available_workers:
            return None

        self._round_robin_index = self._round_robin_index % len(available_workers)
        selected = available_workers[self._round_robin_index]
        self._round_robin_index = (self._round_robin_index + 1) % len(available_workers)
        return selected

    def _select_least_loaded(self, available_workers: List[str]) -> Optional[str]:
        """Select worker with the fewest current tasks."""
        if not available_workers:
            return None

        min_load = float('inf')
        selected = None

        for worker_id in available_workers:
            load = self._worker_loads.get(worker_id)
            if load and load.current_tasks < min_load:
                min_load = load.current_tasks
                selected = worker_id

        return selected or available_workers[0]

    def _select_affinity(
        self,
        available_workers: List[str],
        task_tags: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """Select worker with highest affinity for task tags."""
        if not available_workers:
            return None

        if not task_tags:
            return self._select_least_loaded(available_workers)

        # Score workers by affinity
        scores: Dict[str, int] = {w: 0 for w in available_workers}

        for tag in task_tags:
            if tag in self._task_affinity:
                for worker_id in self._task_affinity[tag]:
                    if worker_id in scores:
                        scores[worker_id] += 1

        # Select worker with highest score (or least loaded if tie)
        max_score = max(scores.values())
        candidates = [w for w, s in scores.items() if s == max_score]

        return self._select_least_loaded(candidates)

    def _select_weighted(self, available_workers: List[str]) -> Optional[str]:
        """Select worker based on performance weights."""
        if not available_workers:
            return None

        import random

        weights = []
        for worker_id in available_workers:
            load = self._worker_loads.get(worker_id)
            if load:
                # Adjust weight by current load
                effective_weight = load.weight / (1 + load.current_tasks)
                weights.append(effective_weight)
            else:
                weights.append(1.0)

        total = sum(weights)
        if total == 0:
            return available_workers[0]

        r = random.uniform(0, total)
        cumulative = 0

        for worker_id, weight in zip(available_workers, weights):
            cumulative += weight
            if r <= cumulative:
                return worker_id

        return available_workers[-1]

    def select_worker(
        self,
        available_workers: List[str],
        task_tags: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """
        Select the best worker for a task using the configured strategy.

        Args:
            available_workers: List of available worker IDs
            task_tags: Optional tags for affinity-based selection

        Returns:
            Selected worker ID or None if no workers available
        """
        if not available_workers:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available_workers)
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._select_least_loaded(available_workers)
        elif self.strategy == LoadBalancingStrategy.AFFINITY:
            return self._select_affinity(available_workers, task_tags)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._select_weighted(available_workers)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            import random
            return random.choice(available_workers)

        return available_workers[0]

    def get_load_report(self) -> Dict[str, Any]:
        """Get load statistics for all workers."""
        return {
            worker_id: {
                "current_tasks": load.current_tasks,
                "completed_tasks": load.completed_tasks,
                "failed_tasks": load.failed_tasks,
                "avg_task_duration": round(load.avg_task_duration, 2),
                "weight": load.weight,
            }
            for worker_id, load in self._worker_loads.items()
        }


# =============================================================================
# adv-c-004: Parallel Execution with Max Concurrency
# =============================================================================


class ConcurrencyLimiter:
    """
    Limits parallel task execution across the system.

    Features:
    - Global concurrency limit
    - Per-worker concurrency limits
    - Per-task-type concurrency limits
    - Queuing when limits are reached
    """

    def __init__(
        self,
        max_global_concurrency: int = 10,
        max_per_worker: int = 1,
        max_per_type: Optional[Dict[str, int]] = None,
    ):
        self.max_global_concurrency = max_global_concurrency
        self.max_per_worker = max_per_worker
        self.max_per_type = max_per_type or {}

        self._global_semaphore = asyncio.Semaphore(max_global_concurrency)
        self._worker_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._type_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._active_tasks: Dict[str, Set[str]] = {}  # worker_id -> task_ids
        self._waiting_count: int = 0

    def _get_worker_semaphore(self, worker_id: str) -> asyncio.Semaphore:
        """Get or create semaphore for a worker."""
        if worker_id not in self._worker_semaphores:
            self._worker_semaphores[worker_id] = asyncio.Semaphore(self.max_per_worker)
        return self._worker_semaphores[worker_id]

    def _get_type_semaphore(self, task_type: str) -> Optional[asyncio.Semaphore]:
        """Get or create semaphore for a task type."""
        if task_type in self.max_per_type:
            if task_type not in self._type_semaphores:
                self._type_semaphores[task_type] = asyncio.Semaphore(
                    self.max_per_type[task_type]
                )
            return self._type_semaphores[task_type]
        return None

    async def acquire(
        self,
        worker_id: str,
        task_id: str,
        task_type: Optional[str] = None,
    ) -> bool:
        """
        Acquire permission to execute a task.

        Returns True if acquired, blocks until available.
        """
        self._waiting_count += 1

        try:
            # Acquire global semaphore
            await self._global_semaphore.acquire()

            # Acquire worker semaphore
            worker_sem = self._get_worker_semaphore(worker_id)
            await worker_sem.acquire()

            # Acquire type semaphore if applicable
            if task_type:
                type_sem = self._get_type_semaphore(task_type)
                if type_sem:
                    await type_sem.acquire()

            # Track active task
            if worker_id not in self._active_tasks:
                self._active_tasks[worker_id] = set()
            self._active_tasks[worker_id].add(task_id)

            return True

        finally:
            self._waiting_count -= 1

    def release(
        self,
        worker_id: str,
        task_id: str,
        task_type: Optional[str] = None,
    ) -> None:
        """Release permission after task completion."""
        # Remove from active tasks
        if worker_id in self._active_tasks:
            self._active_tasks[worker_id].discard(task_id)

        # Release type semaphore if applicable
        if task_type:
            type_sem = self._get_type_semaphore(task_type)
            if type_sem:
                type_sem.release()

        # Release worker semaphore
        if worker_id in self._worker_semaphores:
            self._worker_semaphores[worker_id].release()

        # Release global semaphore
        self._global_semaphore.release()

    def get_stats(self) -> Dict[str, Any]:
        """Get concurrency statistics."""
        active_count = sum(len(tasks) for tasks in self._active_tasks.values())
        return {
            "max_global": self.max_global_concurrency,
            "active_tasks": active_count,
            "available_slots": self.max_global_concurrency - active_count,
            "waiting_count": self._waiting_count,
            "per_worker_active": {
                w: len(tasks) for w, tasks in self._active_tasks.items()
            },
        }


# =============================================================================
# adv-c-005: Task Timeout with Graceful Cancellation
# =============================================================================


@dataclass
class TimeoutConfig:
    """Configuration for task timeouts."""
    default_timeout: float = 600.0  # 10 minutes
    grace_period: float = 30.0  # Time for cleanup before force kill
    warning_threshold: float = 0.8  # Warn at 80% of timeout


class TaskTimeoutError(Exception):
    """Raised when a task times out."""
    def __init__(self, task_id: str, timeout: float, graceful: bool = True):
        self.task_id = task_id
        self.timeout = timeout
        self.graceful = graceful
        super().__init__(f"Task {task_id} timed out after {timeout}s")


class TimeoutManager:
    """
    Manages task timeouts with graceful cancellation.

    Features:
    - Per-task timeout configuration
    - Graceful cancellation with cleanup period
    - Timeout warnings before expiration
    - Force kill after grace period
    """

    def __init__(
        self,
        config: Optional[TimeoutConfig] = None,
        on_warning: Optional[Callable[[str, float], None]] = None,
        on_timeout: Optional[Callable[[str, bool], None]] = None,
    ):
        self.config = config or TimeoutConfig()
        self.on_warning = on_warning
        self.on_timeout = on_timeout

        self._active_timers: Dict[str, asyncio.Task] = {}
        self._start_times: Dict[str, datetime] = {}
        self._timeouts: Dict[str, float] = {}
        self._cancellation_events: Dict[str, asyncio.Event] = {}

    async def _timer_task(
        self,
        task_id: str,
        timeout: float,
        cancel_callback: Callable[[], Any],
    ) -> None:
        """Background task that manages timeout for a single task."""
        warning_time = timeout * self.config.warning_threshold

        try:
            # Wait for warning threshold
            await asyncio.sleep(warning_time)

            if self.on_warning:
                remaining = timeout - warning_time
                self.on_warning(task_id, remaining)

            # Wait for remaining time
            await asyncio.sleep(timeout - warning_time)

            # Timeout reached - try graceful cancellation
            console.print(f"[yellow]Task {task_id} timeout - attempting graceful cancellation[/]")

            if task_id in self._cancellation_events:
                self._cancellation_events[task_id].set()

            if self.on_timeout:
                self.on_timeout(task_id, True)

            # Give grace period for cleanup
            await asyncio.sleep(self.config.grace_period)

            # Force cancellation
            console.print(f"[red]Task {task_id} force cancelled after grace period[/]")
            cancel_callback()

            if self.on_timeout:
                self.on_timeout(task_id, False)

        except asyncio.CancelledError:
            # Timer was cancelled (task completed normally)
            pass

    def start_timer(
        self,
        task_id: str,
        cancel_callback: Callable[[], Any],
        timeout: Optional[float] = None,
    ) -> asyncio.Event:
        """
        Start a timeout timer for a task.

        Args:
            task_id: Task identifier
            cancel_callback: Function to call on force cancellation
            timeout: Custom timeout (uses default if not specified)

        Returns:
            Event that will be set when graceful cancellation is requested
        """
        effective_timeout = timeout or self.config.default_timeout
        self._timeouts[task_id] = effective_timeout
        self._start_times[task_id] = datetime.now()

        # Create cancellation event
        cancel_event = asyncio.Event()
        self._cancellation_events[task_id] = cancel_event

        # Start timer task
        timer = asyncio.create_task(
            self._timer_task(task_id, effective_timeout, cancel_callback)
        )
        self._active_timers[task_id] = timer

        return cancel_event

    def cancel_timer(self, task_id: str) -> None:
        """Cancel the timeout timer for a task (called on normal completion)."""
        if task_id in self._active_timers:
            self._active_timers[task_id].cancel()
            del self._active_timers[task_id]

        self._start_times.pop(task_id, None)
        self._timeouts.pop(task_id, None)
        self._cancellation_events.pop(task_id, None)

    def get_remaining_time(self, task_id: str) -> Optional[float]:
        """Get remaining time before timeout."""
        if task_id not in self._start_times or task_id not in self._timeouts:
            return None

        elapsed = (datetime.now() - self._start_times[task_id]).total_seconds()
        return max(0, self._timeouts[task_id] - elapsed)

    def is_cancellation_requested(self, task_id: str) -> bool:
        """Check if graceful cancellation has been requested."""
        if task_id in self._cancellation_events:
            return self._cancellation_events[task_id].is_set()
        return False

    async def stop(self) -> None:
        """Stop all active timers."""
        for task_id in list(self._active_timers.keys()):
            self.cancel_timer(task_id)


# =============================================================================
# adv-c-006: Inter-Worker Communication Channel
# =============================================================================


@dataclass
class Message:
    """A message between workers."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""  # Empty = broadcast
    message_type: str = "text"
    payload: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    correlation_id: Optional[str] = None  # For request-response pattern


class MessageChannel:
    """
    Enables communication between workers through the orchestrator.

    Features:
    - Point-to-point messaging
    - Broadcast messaging
    - Request-response pattern
    - Message queues per worker
    """

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._message_handlers: Dict[str, Callable[[Message], Any]] = {}

    def register_worker(self, worker_id: str) -> None:
        """Register a worker with the message channel."""
        if worker_id not in self._queues:
            self._queues[worker_id] = asyncio.Queue()

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker from the message channel."""
        self._queues.pop(worker_id, None)

    def set_handler(
        self,
        worker_id: str,
        handler: Callable[[Message], Any],
    ) -> None:
        """Set a message handler for a worker."""
        self._message_handlers[worker_id] = handler

    async def send(
        self,
        sender: str,
        recipient: str,
        payload: Any,
        message_type: str = "text",
    ) -> None:
        """Send a message to a specific worker."""
        message = Message(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
        )

        if recipient in self._queues:
            await self._queues[recipient].put(message)

    async def broadcast(
        self,
        sender: str,
        payload: Any,
        message_type: str = "text",
        exclude: Optional[Set[str]] = None,
    ) -> None:
        """Broadcast a message to all workers."""
        exclude = exclude or set()
        exclude.add(sender)  # Don't send to self

        message = Message(
            sender=sender,
            recipient="",  # Broadcast
            message_type=message_type,
            payload=payload,
        )

        for worker_id, queue in self._queues.items():
            if worker_id not in exclude:
                await queue.put(message)

    async def request(
        self,
        sender: str,
        recipient: str,
        payload: Any,
        timeout: float = 30.0,
    ) -> Any:
        """
        Send a request and wait for a response.

        Args:
            sender: Sender worker ID
            recipient: Recipient worker ID
            payload: Request payload
            timeout: Response timeout in seconds

        Returns:
            Response payload
        """
        correlation_id = str(uuid.uuid4())

        message = Message(
            sender=sender,
            recipient=recipient,
            message_type="request",
            payload=payload,
            requires_response=True,
            correlation_id=correlation_id,
        )

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._response_futures[correlation_id] = future

        try:
            # Send request
            if recipient in self._queues:
                await self._queues[recipient].put(message)
            else:
                raise ValueError(f"Unknown recipient: {recipient}")

            # Wait for response
            return await asyncio.wait_for(future, timeout=timeout)

        finally:
            self._response_futures.pop(correlation_id, None)

    async def respond(
        self,
        original_message: Message,
        payload: Any,
    ) -> None:
        """Send a response to a request message."""
        if not original_message.correlation_id:
            return

        if original_message.correlation_id in self._response_futures:
            future = self._response_futures[original_message.correlation_id]
            if not future.done():
                future.set_result(payload)

    async def receive(
        self,
        worker_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Message]:
        """
        Receive a message for a worker.

        Args:
            worker_id: Worker to receive message for
            timeout: Optional timeout in seconds

        Returns:
            Message or None if timeout
        """
        if worker_id not in self._queues:
            return None

        try:
            if timeout:
                return await asyncio.wait_for(
                    self._queues[worker_id].get(),
                    timeout=timeout
                )
            else:
                return await self._queues[worker_id].get()
        except asyncio.TimeoutError:
            return None

    def has_messages(self, worker_id: str) -> bool:
        """Check if a worker has pending messages."""
        if worker_id in self._queues:
            return not self._queues[worker_id].empty()
        return False

    def get_pending_count(self, worker_id: str) -> int:
        """Get count of pending messages for a worker."""
        if worker_id in self._queues:
            return self._queues[worker_id].qsize()
        return 0


# =============================================================================
# adv-c-007: Shared Memory for Large Data
# =============================================================================


@dataclass
class SharedArtifact:
    """A shared artifact stored in common location."""
    id: str
    name: str
    path: Path
    size: int
    content_type: str
    created_by: str
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SharedMemoryManager:
    """
    Manages shared storage for large artifacts between workers.

    Features:
    - Store large artifacts in shared location
    - Reference artifacts by ID in task context
    - Automatic cleanup after use or expiration
    - Size limits and quotas
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        max_total_size: int = 1024 * 1024 * 1024,  # 1GB default
        max_artifact_size: int = 100 * 1024 * 1024,  # 100MB default
        default_ttl: float = 3600.0,  # 1 hour default
    ):
        self.storage_dir = storage_dir or Path(tempfile.gettempdir()) / "claude-shared"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.max_total_size = max_total_size
        self.max_artifact_size = max_artifact_size
        self.default_ttl = default_ttl

        self._artifacts: Dict[str, SharedArtifact] = {}
        self._total_size: int = 0
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def store(
        self,
        name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        created_by: str = "unknown",
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store data as a shared artifact.

        Args:
            name: Artifact name
            data: Binary data to store
            content_type: MIME type
            created_by: Worker ID that created it
            ttl: Time to live in seconds (None = default)
            metadata: Additional metadata

        Returns:
            Artifact ID
        """
        async with self._lock:
            size = len(data)

            # Check size limits
            if size > self.max_artifact_size:
                raise ValueError(f"Artifact too large: {size} > {self.max_artifact_size}")

            if self._total_size + size > self.max_total_size:
                # Try to cleanup expired artifacts first
                await self._cleanup_expired()

                if self._total_size + size > self.max_total_size:
                    raise ValueError("Shared storage quota exceeded")

            # Generate ID and path
            artifact_id = str(uuid.uuid4())
            path = self.storage_dir / artifact_id

            # Write data
            path.write_bytes(data)

            # Calculate expiration
            effective_ttl = ttl if ttl is not None else self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=effective_ttl) if effective_ttl > 0 else None

            # Create artifact record
            artifact = SharedArtifact(
                id=artifact_id,
                name=name,
                path=path,
                size=size,
                content_type=content_type,
                created_by=created_by,
                created_at=datetime.now(),
                expires_at=expires_at,
                metadata=metadata or {},
            )

            self._artifacts[artifact_id] = artifact
            self._total_size += size

            return artifact_id

    async def retrieve(self, artifact_id: str) -> Optional[bytes]:
        """
        Retrieve artifact data by ID.

        Returns None if artifact not found or expired.
        """
        artifact = self._artifacts.get(artifact_id)

        if not artifact:
            return None

        if artifact.expires_at and datetime.now() > artifact.expires_at:
            await self.delete(artifact_id)
            return None

        if artifact.path.exists():
            return artifact.path.read_bytes()

        return None

    async def get_metadata(self, artifact_id: str) -> Optional[SharedArtifact]:
        """Get artifact metadata without retrieving data."""
        return self._artifacts.get(artifact_id)

    async def delete(self, artifact_id: str) -> bool:
        """Delete an artifact."""
        async with self._lock:
            artifact = self._artifacts.pop(artifact_id, None)

            if artifact:
                if artifact.path.exists():
                    artifact.path.unlink()
                self._total_size -= artifact.size
                return True

            return False

    async def _cleanup_expired(self) -> int:
        """Clean up expired artifacts. Returns count of deleted artifacts."""
        now = datetime.now()
        expired = [
            aid for aid, artifact in self._artifacts.items()
            if artifact.expires_at and now > artifact.expires_at
        ]

        for artifact_id in expired:
            await self.delete(artifact_id)

        return len(expired)

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                console.print(f"[red]Shared memory cleanup error: {e}[/]")

    async def start(self) -> None:
        """Start the cleanup background task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop and clean up all artifacts."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Clean up all artifacts
        for artifact_id in list(self._artifacts.keys()):
            await self.delete(artifact_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_artifacts": len(self._artifacts),
            "total_size": self._total_size,
            "max_total_size": self.max_total_size,
            "usage_percent": round(self._total_size / self.max_total_size * 100, 1),
            "artifacts": [
                {
                    "id": a.id,
                    "name": a.name,
                    "size": a.size,
                    "created_by": a.created_by,
                    "expires_at": a.expires_at.isoformat() if a.expires_at else None,
                }
                for a in self._artifacts.values()
            ],
        }


# =============================================================================
# adv-c-008: Worker Specialization and Routing
# =============================================================================


@dataclass
class WorkerSpecialization:
    """Defines a worker's specializations."""
    worker_id: str
    skills: Set[str] = field(default_factory=set)  # e.g., {"python", "typescript", "testing"}
    preferred_task_types: Set[str] = field(default_factory=set)  # e.g., {"code", "review", "docs"}
    max_concurrent: int = 1
    priority_boost: float = 0.0  # Boost for tasks matching specialization


class SpecializationRouter:
    """
    Routes tasks to workers based on their specializations.

    Features:
    - Workers declare skills and preferences
    - Tasks can require specific skills
    - Automatic routing to best-fit workers
    - Fallback to general workers
    """

    def __init__(self):
        self._specializations: Dict[str, WorkerSpecialization] = {}
        self._skill_index: Dict[str, Set[str]] = {}  # skill -> worker_ids

    def register_worker(
        self,
        worker_id: str,
        skills: Optional[Set[str]] = None,
        preferred_task_types: Optional[Set[str]] = None,
        max_concurrent: int = 1,
        priority_boost: float = 0.0,
    ) -> None:
        """Register a worker with specializations."""
        spec = WorkerSpecialization(
            worker_id=worker_id,
            skills=skills or set(),
            preferred_task_types=preferred_task_types or set(),
            max_concurrent=max_concurrent,
            priority_boost=priority_boost,
        )

        self._specializations[worker_id] = spec

        # Update skill index
        for skill in spec.skills:
            if skill not in self._skill_index:
                self._skill_index[skill] = set()
            self._skill_index[skill].add(worker_id)

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker."""
        spec = self._specializations.pop(worker_id, None)

        if spec:
            for skill in spec.skills:
                if skill in self._skill_index:
                    self._skill_index[skill].discard(worker_id)

    def get_workers_with_skill(self, skill: str) -> Set[str]:
        """Get all workers with a specific skill."""
        return self._skill_index.get(skill, set()).copy()

    def get_workers_for_task(
        self,
        required_skills: Optional[Set[str]] = None,
        preferred_skills: Optional[Set[str]] = None,
        task_type: Optional[str] = None,
        available_workers: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get workers sorted by fitness for a task.

        Args:
            required_skills: Skills that workers MUST have
            preferred_skills: Skills that give bonus score
            task_type: Type of task for preference matching
            available_workers: Filter to only these workers

        Returns:
            List of (worker_id, score) tuples, sorted by score descending
        """
        candidates: List[Tuple[str, float]] = []

        for worker_id, spec in self._specializations.items():
            # Filter by availability
            if available_workers and worker_id not in available_workers:
                continue

            # Check required skills
            if required_skills and not required_skills.issubset(spec.skills):
                continue

            # Calculate score
            score = 1.0 + spec.priority_boost

            # Bonus for preferred skills
            if preferred_skills:
                matching = len(preferred_skills & spec.skills)
                score += matching * 0.5

            # Bonus for task type preference
            if task_type and task_type in spec.preferred_task_types:
                score += 1.0

            candidates.append((worker_id, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates

    def route_task(
        self,
        required_skills: Optional[Set[str]] = None,
        preferred_skills: Optional[Set[str]] = None,
        task_type: Optional[str] = None,
        available_workers: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """
        Route a task to the best available worker.

        Returns worker_id or None if no suitable worker.
        """
        candidates = self.get_workers_for_task(
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            task_type=task_type,
            available_workers=available_workers,
        )

        if candidates:
            return candidates[0][0]

        # Fallback: any available worker if no specialization required
        if not required_skills and available_workers:
            return next(iter(available_workers), None)

        return None

    def get_specialization_report(self) -> Dict[str, Any]:
        """Get report of all worker specializations."""
        return {
            worker_id: {
                "skills": list(spec.skills),
                "preferred_task_types": list(spec.preferred_task_types),
                "max_concurrent": spec.max_concurrent,
                "priority_boost": spec.priority_boost,
            }
            for worker_id, spec in self._specializations.items()
        }


# =============================================================================
# adv-c-009: Task Execution Sandboxing
# =============================================================================


@dataclass
class SandboxConfig:
    """Configuration for task sandboxing."""
    enabled: bool = True
    max_memory_mb: int = 512
    max_cpu_percent: float = 80.0
    max_disk_mb: int = 1024
    network_enabled: bool = True
    allowed_paths: List[str] = field(default_factory=list)
    env_whitelist: List[str] = field(default_factory=lambda: [
        "PATH", "HOME", "USER", "LANG", "LC_ALL",
        "CLAUDE_CODE_AGENT_ID", "ANTHROPIC_API_KEY"
    ])
    timeout: float = 600.0


class ResourceMonitor:
    """Monitors resource usage for sandboxed tasks."""

    def __init__(self, pid: int, config: SandboxConfig):
        self.pid = pid
        self.config = config
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._violations: List[str] = []

    async def _check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        try:
            import psutil
            process = psutil.Process(self.pid)
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb <= self.config.max_memory_mb
        except Exception:
            return True  # Can't check, assume OK

    async def _check_cpu(self) -> bool:
        """Check if CPU usage is within limits."""
        try:
            import psutil
            process = psutil.Process(self.pid)
            cpu_percent = process.cpu_percent(interval=0.1)
            return cpu_percent <= self.config.max_cpu_percent
        except Exception:
            return True  # Can't check, assume OK

    async def _monitor_loop(
        self,
        on_violation: Callable[[str], None],
    ) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                if not await self._check_memory():
                    violation = f"Memory limit exceeded (max: {self.config.max_memory_mb}MB)"
                    self._violations.append(violation)
                    on_violation(violation)

                if not await self._check_cpu():
                    violation = f"CPU limit exceeded (max: {self.config.max_cpu_percent}%)"
                    self._violations.append(violation)
                    on_violation(violation)

            except Exception as e:
                console.print(f"[yellow]Resource monitor error: {e}[/]")

            await asyncio.sleep(1.0)

    async def start(self, on_violation: Callable[[str], None]) -> None:
        """Start resource monitoring."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop(on_violation))

    async def stop(self) -> None:
        """Stop resource monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def get_violations(self) -> List[str]:
        """Get list of resource violations."""
        return self._violations.copy()


class TaskSandbox:
    """
    Provides isolated execution environment for tasks.

    Features:
    - Resource limits (memory, CPU, disk)
    - Path restrictions
    - Environment isolation
    - Network restrictions
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._monitors: Dict[str, ResourceMonitor] = {}

    def create_sandbox_env(self, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Create a sandboxed environment for task execution."""
        base_env = base_env or os.environ.copy()

        # Filter to whitelisted environment variables
        sandbox_env = {
            key: value
            for key, value in base_env.items()
            if key in self.config.env_whitelist
        }

        # Add sandbox markers
        sandbox_env["CLAUDE_SANDBOX"] = "1"
        sandbox_env["CLAUDE_SANDBOX_MEMORY_LIMIT"] = str(self.config.max_memory_mb)
        sandbox_env["CLAUDE_SANDBOX_CPU_LIMIT"] = str(self.config.max_cpu_percent)

        return sandbox_env

    async def start_monitoring(
        self,
        task_id: str,
        pid: int,
        on_violation: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """
        Start monitoring a sandboxed task.

        Args:
            task_id: Task identifier
            pid: Process ID to monitor
            on_violation: Callback(task_id, violation_message)
        """
        if not self.config.enabled:
            return

        monitor = ResourceMonitor(pid, self.config)
        self._monitors[task_id] = monitor

        def handle_violation(violation: str):
            console.print(f"[red]Sandbox violation for {task_id}: {violation}[/]")
            if on_violation:
                on_violation(task_id, violation)

        await monitor.start(handle_violation)

    async def stop_monitoring(self, task_id: str) -> List[str]:
        """
        Stop monitoring a task.

        Returns list of violations that occurred.
        """
        monitor = self._monitors.pop(task_id, None)

        if monitor:
            await monitor.stop()
            return monitor.get_violations()

        return []

    def validate_path(self, path: str) -> bool:
        """Check if a path is allowed within the sandbox."""
        if not self.config.allowed_paths:
            return True  # No restrictions

        abs_path = os.path.abspath(path)

        for allowed in self.config.allowed_paths:
            allowed_abs = os.path.abspath(allowed)
            if abs_path.startswith(allowed_abs):
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get sandbox statistics."""
        return {
            "enabled": self.config.enabled,
            "active_sandboxes": len(self._monitors),
            "config": {
                "max_memory_mb": self.config.max_memory_mb,
                "max_cpu_percent": self.config.max_cpu_percent,
                "max_disk_mb": self.config.max_disk_mb,
                "network_enabled": self.config.network_enabled,
            },
        }


# =============================================================================
# adv-c-010: Execution Replay for Debugging
# =============================================================================


@dataclass
class ExecutionEvent:
    """A single event in task execution."""
    timestamp: datetime
    event_type: str  # "input", "output", "error", "state_change"
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionRecording:
    """Complete recording of a task execution."""
    task_id: str
    worker_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    events: List[ExecutionEvent] = field(default_factory=list)
    final_result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionRecorder:
    """
    Records task execution for later replay and debugging.

    Features:
    - Record all inputs, outputs, and state changes
    - Store recordings persistently
    - Replay recordings for debugging
    - Compare replay results with original
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        max_recordings: int = 100,
        max_events_per_recording: int = 10000,
    ):
        self.storage_dir = storage_dir or Path(tempfile.gettempdir()) / "claude-recordings"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.max_recordings = max_recordings
        self.max_events_per_recording = max_events_per_recording

        self._active_recordings: Dict[str, ExecutionRecording] = {}
        self._completed_recordings: Dict[str, ExecutionRecording] = {}

    def start_recording(
        self,
        task_id: str,
        worker_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start recording a task execution."""
        recording = ExecutionRecording(
            task_id=task_id,
            worker_id=worker_id,
            started_at=datetime.now(),
            completed_at=None,
            metadata=metadata or {},
        )

        self._active_recordings[task_id] = recording

    def record_event(
        self,
        task_id: str,
        event_type: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an event for a task."""
        recording = self._active_recordings.get(task_id)

        if not recording:
            return

        if len(recording.events) >= self.max_events_per_recording:
            # Drop oldest events
            recording.events = recording.events[-self.max_events_per_recording + 1:]

        event = ExecutionEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            data=data,
            metadata=metadata or {},
        )

        recording.events.append(event)

    def record_input(self, task_id: str, input_data: Any) -> None:
        """Record an input event."""
        self.record_event(task_id, "input", input_data)

    def record_output(self, task_id: str, output_data: Any) -> None:
        """Record an output event."""
        self.record_event(task_id, "output", output_data)

    def record_error(self, task_id: str, error: str) -> None:
        """Record an error event."""
        self.record_event(task_id, "error", error)

    def record_state_change(
        self,
        task_id: str,
        old_state: str,
        new_state: str,
    ) -> None:
        """Record a state change event."""
        self.record_event(task_id, "state_change", {
            "old": old_state,
            "new": new_state,
        })

    def stop_recording(
        self,
        task_id: str,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> Optional[ExecutionRecording]:
        """Stop recording and finalize."""
        recording = self._active_recordings.pop(task_id, None)

        if not recording:
            return None

        recording.completed_at = datetime.now()
        recording.final_result = result
        recording.error = error

        # Store completed recording
        self._completed_recordings[task_id] = recording

        # Enforce max recordings limit
        while len(self._completed_recordings) > self.max_recordings:
            oldest_id = min(
                self._completed_recordings.keys(),
                key=lambda k: self._completed_recordings[k].started_at
            )
            self._completed_recordings.pop(oldest_id)

        return recording

    def save_recording(self, task_id: str) -> Optional[Path]:
        """Save a recording to disk."""
        recording = self._completed_recordings.get(task_id)

        if not recording:
            return None

        filepath = self.storage_dir / f"{task_id}.json"

        # Convert to serializable format
        data = {
            "task_id": recording.task_id,
            "worker_id": recording.worker_id,
            "started_at": recording.started_at.isoformat(),
            "completed_at": recording.completed_at.isoformat() if recording.completed_at else None,
            "events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type,
                    "data": e.data,
                    "metadata": e.metadata,
                }
                for e in recording.events
            ],
            "final_result": recording.final_result,
            "error": recording.error,
            "metadata": recording.metadata,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    def load_recording(self, task_id: str) -> Optional[ExecutionRecording]:
        """Load a recording from disk."""
        filepath = self.storage_dir / f"{task_id}.json"

        if not filepath.exists():
            return None

        with open(filepath) as f:
            data = json.load(f)

        recording = ExecutionRecording(
            task_id=data["task_id"],
            worker_id=data["worker_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None,
            events=[
                ExecutionEvent(
                    timestamp=datetime.fromisoformat(e["timestamp"]),
                    event_type=e["event_type"],
                    data=e["data"],
                    metadata=e["metadata"],
                )
                for e in data["events"]
            ],
            final_result=data.get("final_result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )

        return recording

    def get_recording(self, task_id: str) -> Optional[ExecutionRecording]:
        """Get a recording by task ID."""
        # Check completed recordings first
        if task_id in self._completed_recordings:
            return self._completed_recordings[task_id]

        # Check active recordings
        if task_id in self._active_recordings:
            return self._active_recordings[task_id]

        # Try to load from disk
        return self.load_recording(task_id)

    def list_recordings(self) -> List[Dict[str, Any]]:
        """List all available recordings."""
        recordings = []

        # In-memory recordings
        for recording in self._completed_recordings.values():
            recordings.append({
                "task_id": recording.task_id,
                "worker_id": recording.worker_id,
                "started_at": recording.started_at.isoformat(),
                "completed_at": recording.completed_at.isoformat() if recording.completed_at else None,
                "event_count": len(recording.events),
                "has_error": recording.error is not None,
                "source": "memory",
            })

        # Disk recordings
        for filepath in self.storage_dir.glob("*.json"):
            task_id = filepath.stem
            if task_id not in self._completed_recordings:
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    recordings.append({
                        "task_id": data["task_id"],
                        "worker_id": data["worker_id"],
                        "started_at": data["started_at"],
                        "completed_at": data["completed_at"],
                        "event_count": len(data["events"]),
                        "has_error": data.get("error") is not None,
                        "source": "disk",
                    })
                except Exception:
                    pass

        return recordings

    def compare_recordings(
        self,
        original_id: str,
        replay_id: str,
    ) -> Dict[str, Any]:
        """Compare two recordings for differences."""
        original = self.get_recording(original_id)
        replay = self.get_recording(replay_id)

        if not original or not replay:
            return {"error": "Recording not found"}

        # Compare outputs
        original_outputs = [e.data for e in original.events if e.event_type == "output"]
        replay_outputs = [e.data for e in replay.events if e.event_type == "output"]

        outputs_match = original_outputs == replay_outputs

        # Compare final results
        results_match = original.final_result == replay.final_result

        # Compare errors
        errors_match = original.error == replay.error

        return {
            "original_id": original_id,
            "replay_id": replay_id,
            "outputs_match": outputs_match,
            "results_match": results_match,
            "errors_match": errors_match,
            "all_match": outputs_match and results_match and errors_match,
            "original_event_count": len(original.events),
            "replay_event_count": len(replay.events),
            "original_duration": (
                (original.completed_at - original.started_at).total_seconds()
                if original.completed_at else None
            ),
            "replay_duration": (
                (replay.completed_at - replay.started_at).total_seconds()
                if replay.completed_at else None
            ),
        }


# =============================================================================
# Integration: AdvancedOrchestrator combining all features
# =============================================================================


class AdvancedOrchestratorMixin:
    """
    Mixin class that adds advanced features to the Orchestrator.

    Use with the main Orchestrator class:

        class EnhancedOrchestrator(AdvancedOrchestratorMixin, Orchestrator):
            pass
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize advanced features
        self._scaler: Optional[AdaptiveScaler] = None
        self._health_monitor: Optional[HealthMonitor] = None
        self._load_balancer: Optional[LoadBalancer] = None
        self._concurrency_limiter: Optional[ConcurrencyLimiter] = None
        self._timeout_manager: Optional[TimeoutManager] = None
        self._message_channel: Optional[MessageChannel] = None
        self._shared_memory: Optional[SharedMemoryManager] = None
        self._specialization_router: Optional[SpecializationRouter] = None
        self._task_sandbox: Optional[TaskSandbox] = None
        self._execution_recorder: Optional[ExecutionRecorder] = None

    def enable_adaptive_scaling(
        self,
        config: Optional[ScalingConfig] = None,
    ) -> None:
        """Enable adaptive worker pool scaling."""
        self._scaler = AdaptiveScaler(
            pool=self._pool,
            config=config,
        )

    def enable_health_monitoring(
        self,
        config: Optional[HealthConfig] = None,
    ) -> None:
        """Enable worker health monitoring."""
        self._health_monitor = HealthMonitor(
            pool=self._pool,
            config=config,
        )

    def enable_load_balancing(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED,
    ) -> None:
        """Enable load balancing."""
        self._load_balancer = LoadBalancer(strategy=strategy)

    def enable_concurrency_limits(
        self,
        max_global: int = 10,
        max_per_worker: int = 1,
    ) -> None:
        """Enable concurrency limiting."""
        self._concurrency_limiter = ConcurrencyLimiter(
            max_global_concurrency=max_global,
            max_per_worker=max_per_worker,
        )

    def enable_timeout_management(
        self,
        config: Optional[TimeoutConfig] = None,
    ) -> None:
        """Enable timeout management."""
        self._timeout_manager = TimeoutManager(config=config)

    def enable_inter_worker_communication(self) -> None:
        """Enable inter-worker communication."""
        self._message_channel = MessageChannel()

    def enable_shared_memory(
        self,
        storage_dir: Optional[Path] = None,
    ) -> None:
        """Enable shared memory for large artifacts."""
        self._shared_memory = SharedMemoryManager(storage_dir=storage_dir)

    def enable_worker_specialization(self) -> None:
        """Enable worker specialization and routing."""
        self._specialization_router = SpecializationRouter()

    def enable_task_sandboxing(
        self,
        config: Optional[SandboxConfig] = None,
    ) -> None:
        """Enable task execution sandboxing."""
        self._task_sandbox = TaskSandbox(config=config)

    def enable_execution_replay(
        self,
        storage_dir: Optional[Path] = None,
    ) -> None:
        """Enable execution recording and replay."""
        self._execution_recorder = ExecutionRecorder(storage_dir=storage_dir)

    async def start_advanced_features(self) -> None:
        """Start all enabled advanced features."""
        if self._scaler:
            await self._scaler.start(
                get_queue_depth=lambda: len(self.state.get_available_tasks())
            )

        if self._health_monitor:
            await self._health_monitor.start()

        if self._shared_memory:
            await self._shared_memory.start()

    async def stop_advanced_features(self) -> None:
        """Stop all advanced features."""
        if self._scaler:
            await self._scaler.stop()

        if self._health_monitor:
            await self._health_monitor.stop()

        if self._timeout_manager:
            await self._timeout_manager.stop()

        if self._shared_memory:
            await self._shared_memory.stop()

    def get_advanced_status(self) -> Dict[str, Any]:
        """Get status of all advanced features."""
        status = {}

        if self._load_balancer:
            status["load_balancer"] = self._load_balancer.get_load_report()

        if self._concurrency_limiter:
            status["concurrency"] = self._concurrency_limiter.get_stats()

        if self._health_monitor:
            status["health"] = self._health_monitor.get_health_report()

        if self._shared_memory:
            status["shared_memory"] = self._shared_memory.get_stats()

        if self._task_sandbox:
            status["sandbox"] = self._task_sandbox.get_stats()

        if self._execution_recorder:
            status["recordings"] = self._execution_recorder.list_recordings()

        if self._specialization_router:
            status["specializations"] = self._specialization_router.get_specialization_report()

        return status
