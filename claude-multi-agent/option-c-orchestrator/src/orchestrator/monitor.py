"""
Task Monitor - Real-time monitoring and health tracking.

This module provides monitoring capabilities including:
- Real-time status dashboard data (adv-c-mon-001)
- Worker health checks (adv-c-mon-002)
- Anomaly detection alerts (adv-c-mon-003)
- Resource usage tracking (adv-c-mon-004)
- Execution timeline visualization (adv-c-mon-005)
"""

from __future__ import annotations

import asyncio
import json
import statistics
import psutil
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any, Callable, Awaitable

from pydantic import BaseModel, Field

from .models import TaskStatus


# =============================================================================
# Core Monitoring Data Structures
# =============================================================================


class AlertSeverity(str, Enum):
    """Severity levels for monitoring alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of monitoring alerts."""

    WORKER_UNHEALTHY = "worker_unhealthy"
    WORKER_TIMEOUT = "worker_timeout"
    TASK_SLOW = "task_slow"
    TASK_STUCK = "task_stuck"
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    QUEUE_BACKLOG = "queue_backlog"
    ANOMALY_DETECTED = "anomaly_detected"
    WORKER_CRASH = "worker_crash"


@dataclass
class Alert:
    """A monitoring alert."""

    id: str
    type: AlertType
    severity: AlertSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def acknowledge(self) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True

    def resolve(self) -> None:
        """Mark the alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()


@dataclass
class WorkerHealthStatus:
    """Health status for a single worker."""

    worker_id: str
    is_healthy: bool = True
    is_responsive: bool = True
    last_heartbeat: datetime = field(default_factory=datetime.now)
    last_task_completed: Optional[datetime] = None
    consecutive_failures: int = 0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    uptime_seconds: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_task: Optional[str] = None
    current_task_duration: float = 0.0


@dataclass
class TaskExecutionMetrics:
    """Metrics for a single task execution."""

    task_id: str
    worker_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    status: TaskStatus = TaskStatus.IN_PROGRESS
    cpu_usage: list[float] = field(default_factory=list)
    memory_usage: list[float] = field(default_factory=list)


@dataclass
class TimelineEvent:
    """An event in the execution timeline."""

    timestamp: datetime
    event_type: str
    task_id: Optional[str] = None
    worker_id: Optional[str] = None
    details: str = ""
    duration: float = 0.0


# =============================================================================
# Real-Time Status Dashboard (adv-c-mon-001)
# =============================================================================


class StatusDashboard:
    """
    Provides real-time status dashboard data.

    Implements adv-c-mon-001.
    """

    def __init__(self):
        self.start_time = datetime.now()
        self._task_metrics: dict[str, TaskExecutionMetrics] = {}
        self._worker_health: dict[str, WorkerHealthStatus] = {}
        self._alerts: list[Alert] = []
        self._event_history: deque[dict[str, Any]] = deque(maxlen=1000)
        self._update_callbacks: list[Callable[[dict[str, Any]], Awaitable[None]]] = []

    def register_update_callback(
        self, callback: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register a callback for dashboard updates."""
        self._update_callbacks.append(callback)

    async def _notify_update(self, update: dict[str, Any]) -> None:
        """Notify all callbacks of an update."""
        for callback in self._update_callbacks:
            try:
                await callback(update)
            except Exception:
                pass  # Don't let callback errors affect monitoring

    def record_task_start(self, task_id: str, worker_id: str) -> None:
        """Record a task starting execution."""
        self._task_metrics[task_id] = TaskExecutionMetrics(
            task_id=task_id,
            worker_id=worker_id,
            start_time=datetime.now(),
        )

        if worker_id in self._worker_health:
            self._worker_health[worker_id].current_task = task_id

        self._event_history.append(
            {
                "type": "task_start",
                "task_id": task_id,
                "worker_id": worker_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def record_task_complete(self, task_id: str, success: bool = True) -> None:
        """Record a task completing."""
        metrics = self._task_metrics.get(task_id)
        if metrics:
            metrics.end_time = datetime.now()
            metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()
            metrics.status = TaskStatus.DONE if success else TaskStatus.FAILED

            worker_health = self._worker_health.get(metrics.worker_id)
            if worker_health:
                worker_health.current_task = None
                worker_health.current_task_duration = 0.0
                worker_health.last_task_completed = datetime.now()
                if success:
                    worker_health.tasks_completed += 1
                    worker_health.consecutive_failures = 0
                else:
                    worker_health.tasks_failed += 1
                    worker_health.consecutive_failures += 1

        self._event_history.append(
            {
                "type": "task_complete",
                "task_id": task_id,
                "success": success,
                "duration": metrics.duration_seconds if metrics else 0,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def register_worker(self, worker_id: str) -> None:
        """Register a new worker for monitoring."""
        self._worker_health[worker_id] = WorkerHealthStatus(worker_id=worker_id)

    def update_worker_heartbeat(self, worker_id: str) -> None:
        """Update worker heartbeat timestamp."""
        if worker_id in self._worker_health:
            self._worker_health[worker_id].last_heartbeat = datetime.now()
            self._worker_health[worker_id].is_responsive = True

    def get_dashboard_data(self) -> dict[str, Any]:
        """
        Get complete dashboard data for display.

        Returns a structured dict suitable for UI rendering.
        """
        now = datetime.now()
        uptime = (now - self.start_time).total_seconds()

        # Task summary
        completed_tasks = [m for m in self._task_metrics.values() if m.status == TaskStatus.DONE]
        failed_tasks = [m for m in self._task_metrics.values() if m.status == TaskStatus.FAILED]
        in_progress = [
            m for m in self._task_metrics.values() if m.status == TaskStatus.IN_PROGRESS
        ]

        # Calculate throughput
        tasks_per_minute = (
            len(completed_tasks) / (uptime / 60) if uptime > 60 else len(completed_tasks)
        )

        # Average task duration
        durations = [m.duration_seconds for m in completed_tasks if m.duration_seconds > 0]
        avg_duration = statistics.mean(durations) if durations else 0

        # Worker summary
        active_workers = sum(1 for w in self._worker_health.values() if w.is_healthy)
        total_workers = len(self._worker_health)

        # Active alerts
        active_alerts = [a for a in self._alerts if not a.resolved]

        return {
            "timestamp": now.isoformat(),
            "uptime_seconds": uptime,
            "summary": {
                "tasks_completed": len(completed_tasks),
                "tasks_failed": len(failed_tasks),
                "tasks_in_progress": len(in_progress),
                "tasks_per_minute": round(tasks_per_minute, 2),
                "avg_task_duration": round(avg_duration, 2),
            },
            "workers": {
                "active": active_workers,
                "total": total_workers,
                "details": [
                    {
                        "id": w.worker_id,
                        "healthy": w.is_healthy,
                        "responsive": w.is_responsive,
                        "current_task": w.current_task,
                        "tasks_completed": w.tasks_completed,
                        "tasks_failed": w.tasks_failed,
                        "cpu_percent": w.cpu_percent,
                        "memory_mb": w.memory_mb,
                    }
                    for w in self._worker_health.values()
                ],
            },
            "alerts": {
                "active_count": len(active_alerts),
                "by_severity": {
                    severity.value: len(
                        [a for a in active_alerts if a.severity == severity]
                    )
                    for severity in AlertSeverity
                },
                "recent": [
                    {
                        "id": a.id,
                        "type": a.type.value,
                        "severity": a.severity.value,
                        "message": a.message,
                        "timestamp": a.timestamp.isoformat(),
                    }
                    for a in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[
                        :10
                    ]
                ],
            },
            "recent_events": list(self._event_history)[-20:],
        }

    def get_task_metrics(self, task_id: str) -> Optional[dict[str, Any]]:
        """Get detailed metrics for a specific task."""
        metrics = self._task_metrics.get(task_id)
        if not metrics:
            return None

        return {
            "task_id": metrics.task_id,
            "worker_id": metrics.worker_id,
            "start_time": metrics.start_time.isoformat(),
            "end_time": metrics.end_time.isoformat() if metrics.end_time else None,
            "duration_seconds": metrics.duration_seconds,
            "status": metrics.status.value,
            "cpu_usage": {
                "samples": len(metrics.cpu_usage),
                "avg": statistics.mean(metrics.cpu_usage) if metrics.cpu_usage else 0,
                "max": max(metrics.cpu_usage) if metrics.cpu_usage else 0,
            },
            "memory_usage": {
                "samples": len(metrics.memory_usage),
                "avg": statistics.mean(metrics.memory_usage) if metrics.memory_usage else 0,
                "max": max(metrics.memory_usage) if metrics.memory_usage else 0,
            },
        }

    def export_metrics(self) -> str:
        """Export all metrics as JSON."""
        return json.dumps(
            {
                "exported_at": datetime.now().isoformat(),
                "dashboard": self.get_dashboard_data(),
                "task_metrics": {
                    task_id: self.get_task_metrics(task_id)
                    for task_id in self._task_metrics
                },
            },
            indent=2,
        )


# =============================================================================
# Worker Health Checks (adv-c-mon-002)
# =============================================================================


class HealthCheckConfig(BaseModel):
    """Configuration for health checks."""

    heartbeat_interval: float = Field(default=5.0, description="Seconds between heartbeats")
    heartbeat_timeout: float = Field(
        default=30.0, description="Seconds before worker considered unresponsive"
    )
    max_consecutive_failures: int = Field(
        default=3, description="Failures before marking unhealthy"
    )
    cpu_threshold: float = Field(default=90.0, description="CPU % threshold for warning")
    memory_threshold: float = Field(default=90.0, description="Memory % threshold for warning")
    task_timeout: float = Field(default=600.0, description="Seconds before task considered stuck")


class WorkerHealthChecker:
    """
    Monitors worker health through periodic checks.

    Implements adv-c-mon-002.
    """

    def __init__(
        self,
        dashboard: StatusDashboard,
        config: Optional[HealthCheckConfig] = None,
    ):
        self.dashboard = dashboard
        self.config = config or HealthCheckConfig()
        self._running = False
        self._check_task: Optional[asyncio.Task[None]] = None
        self._worker_pids: dict[str, int] = {}
        self._on_unhealthy: list[Callable[[str, WorkerHealthStatus], None]] = []
        self._on_recovery: list[Callable[[str, WorkerHealthStatus], None]] = []

    def register_worker_pid(self, worker_id: str, pid: int) -> None:
        """Register a worker's process ID for monitoring."""
        self._worker_pids[worker_id] = pid
        self.dashboard.register_worker(worker_id)

    def on_unhealthy(self, callback: Callable[[str, WorkerHealthStatus], None]) -> None:
        """Register callback for when worker becomes unhealthy."""
        self._on_unhealthy.append(callback)

    def on_recovery(self, callback: Callable[[str, WorkerHealthStatus], None]) -> None:
        """Register callback for when worker recovers."""
        self._on_recovery.append(callback)

    async def start(self) -> None:
        """Start the health check loop."""
        self._running = True
        self._check_task = asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        """Stop the health check loop."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

    async def _health_check_loop(self) -> None:
        """Main health check loop."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1)  # Brief pause on error

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all workers."""
        now = datetime.now()

        for worker_id, health in self.dashboard._worker_health.items():
            was_healthy = health.is_healthy
            pid = self._worker_pids.get(worker_id)

            # Check process health
            if pid:
                try:
                    proc = psutil.Process(pid)
                    health.cpu_percent = proc.cpu_percent(interval=0.1)
                    health.memory_mb = proc.memory_info().rss / (1024 * 1024)
                    health.is_responsive = proc.is_running()
                except psutil.NoSuchProcess:
                    health.is_responsive = False
                    health.is_healthy = False
                except Exception:
                    pass

            # Check heartbeat timeout
            heartbeat_age = (now - health.last_heartbeat).total_seconds()
            if heartbeat_age > self.config.heartbeat_timeout:
                health.is_responsive = False

            # Check consecutive failures
            if health.consecutive_failures >= self.config.max_consecutive_failures:
                health.is_healthy = False

            # Check if task is stuck
            if health.current_task:
                metrics = self.dashboard._task_metrics.get(health.current_task)
                if metrics:
                    task_duration = (now - metrics.start_time).total_seconds()
                    health.current_task_duration = task_duration
                    if task_duration > self.config.task_timeout:
                        health.is_healthy = False

            # Determine final health status
            health.is_healthy = health.is_responsive and (
                health.consecutive_failures < self.config.max_consecutive_failures
            )

            # Trigger callbacks
            if was_healthy and not health.is_healthy:
                for callback in self._on_unhealthy:
                    callback(worker_id, health)

            if not was_healthy and health.is_healthy:
                for callback in self._on_recovery:
                    callback(worker_id, health)

    def check_worker(self, worker_id: str) -> WorkerHealthStatus:
        """Get current health status for a worker."""
        return self.dashboard._worker_health.get(
            worker_id, WorkerHealthStatus(worker_id=worker_id, is_healthy=False)
        )

    def get_all_health(self) -> dict[str, WorkerHealthStatus]:
        """Get health status for all workers."""
        return dict(self.dashboard._worker_health)

    def get_unhealthy_workers(self) -> list[str]:
        """Get list of unhealthy worker IDs."""
        return [
            worker_id
            for worker_id, health in self.dashboard._worker_health.items()
            if not health.is_healthy
        ]


# =============================================================================
# Anomaly Detection (adv-c-mon-003)
# =============================================================================


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""

    SLOW_TASK = "slow_task"
    HIGH_FAILURE_RATE = "high_failure_rate"
    RESOURCE_SPIKE = "resource_spike"
    UNUSUAL_PATTERN = "unusual_pattern"
    STUCK_QUEUE = "stuck_queue"


@dataclass
class Anomaly:
    """A detected anomaly."""

    type: AnomalyType
    severity: AlertSeverity
    description: str
    affected_entity: str  # task_id or worker_id
    detected_at: datetime = field(default_factory=datetime.now)
    metrics: dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """
    Detects anomalies in task execution and worker behavior.

    Implements adv-c-mon-003.
    """

    def __init__(self, dashboard: StatusDashboard):
        self.dashboard = dashboard
        self._task_duration_history: dict[str, list[float]] = defaultdict(list)
        self._worker_failure_rates: dict[str, deque[bool]] = defaultdict(
            lambda: deque(maxlen=20)
        )
        self._cpu_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=60))
        self._memory_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=60))
        self._anomaly_callbacks: list[Callable[[Anomaly], None]] = []
        self._detected_anomalies: list[Anomaly] = []
        self._baseline_task_duration: Optional[float] = None

    def on_anomaly(self, callback: Callable[[Anomaly], None]) -> None:
        """Register callback for detected anomalies."""
        self._anomaly_callbacks.append(callback)

    def _notify_anomaly(self, anomaly: Anomaly) -> None:
        """Notify callbacks of detected anomaly."""
        self._detected_anomalies.append(anomaly)

        # Create an alert
        alert_id = f"alert-{len(self.dashboard._alerts)}"
        alert = Alert(
            id=alert_id,
            type=AlertType.ANOMALY_DETECTED,
            severity=anomaly.severity,
            message=anomaly.description,
            details={"anomaly_type": anomaly.type.value, **anomaly.metrics},
        )
        self.dashboard._alerts.append(alert)

        for callback in self._anomaly_callbacks:
            try:
                callback(anomaly)
            except Exception:
                pass

    def record_task_duration(self, task_id: str, duration: float) -> None:
        """Record task duration for anomaly detection."""
        # Group by task prefix (e.g., "task-type-1" -> "task-type")
        task_type = "-".join(task_id.split("-")[:-1]) if "-" in task_id else task_id
        self._task_duration_history[task_type].append(duration)

        # Update baseline
        all_durations = [
            d for durations in self._task_duration_history.values() for d in durations
        ]
        if len(all_durations) >= 5:
            self._baseline_task_duration = statistics.median(all_durations)

    def record_task_result(self, worker_id: str, success: bool) -> None:
        """Record task result for failure rate tracking."""
        self._worker_failure_rates[worker_id].append(success)

    def record_resource_usage(self, worker_id: str, cpu: float, memory: float) -> None:
        """Record resource usage for spike detection."""
        self._cpu_history[worker_id].append(cpu)
        self._memory_history[worker_id].append(memory)

    def detect_slow_task(self, task_id: str, duration: float) -> Optional[Anomaly]:
        """
        Detect if a task is running abnormally slow.

        Uses statistical analysis to identify outliers.
        """
        if self._baseline_task_duration is None:
            return None

        # Consider slow if 3x the baseline
        threshold = self._baseline_task_duration * 3

        if duration > threshold:
            anomaly = Anomaly(
                type=AnomalyType.SLOW_TASK,
                severity=AlertSeverity.WARNING,
                description=f"Task {task_id} took {duration:.1f}s, "
                f"which is {duration / self._baseline_task_duration:.1f}x "
                f"the baseline ({self._baseline_task_duration:.1f}s)",
                affected_entity=task_id,
                metrics={
                    "duration": duration,
                    "baseline": self._baseline_task_duration,
                    "ratio": duration / self._baseline_task_duration,
                },
            )
            self._notify_anomaly(anomaly)
            return anomaly

        return None

    def detect_high_failure_rate(self, worker_id: str) -> Optional[Anomaly]:
        """
        Detect if a worker has abnormally high failure rate.
        """
        history = self._worker_failure_rates.get(worker_id)
        if not history or len(history) < 5:
            return None

        failure_rate = 1 - (sum(history) / len(history))

        if failure_rate > 0.5:  # More than 50% failure rate
            severity = AlertSeverity.CRITICAL if failure_rate > 0.8 else AlertSeverity.WARNING
            anomaly = Anomaly(
                type=AnomalyType.HIGH_FAILURE_RATE,
                severity=severity,
                description=f"Worker {worker_id} has {failure_rate * 100:.1f}% failure rate "
                f"over last {len(history)} tasks",
                affected_entity=worker_id,
                metrics={
                    "failure_rate": failure_rate,
                    "sample_size": len(history),
                },
            )
            self._notify_anomaly(anomaly)
            return anomaly

        return None

    def detect_resource_spike(self, worker_id: str) -> Optional[Anomaly]:
        """
        Detect abnormal resource usage spikes.
        """
        cpu_history = self._cpu_history.get(worker_id)
        memory_history = self._memory_history.get(worker_id)

        anomalies = []

        if cpu_history and len(cpu_history) >= 10:
            recent_cpu = list(cpu_history)[-5:]
            if statistics.mean(recent_cpu) > 90:
                anomalies.append(
                    Anomaly(
                        type=AnomalyType.RESOURCE_SPIKE,
                        severity=AlertSeverity.WARNING,
                        description=f"Worker {worker_id} CPU usage spike: "
                        f"{statistics.mean(recent_cpu):.1f}%",
                        affected_entity=worker_id,
                        metrics={
                            "resource": "cpu",
                            "recent_avg": statistics.mean(recent_cpu),
                            "recent_max": max(recent_cpu),
                        },
                    )
                )

        if memory_history and len(memory_history) >= 10:
            recent_memory = list(memory_history)[-5:]
            historical_avg = statistics.mean(list(memory_history)[:-5])

            # Spike if recent is 50% higher than historical average
            if historical_avg > 0 and statistics.mean(recent_memory) > historical_avg * 1.5:
                anomalies.append(
                    Anomaly(
                        type=AnomalyType.RESOURCE_SPIKE,
                        severity=AlertSeverity.WARNING,
                        description=f"Worker {worker_id} memory usage spike: "
                        f"{statistics.mean(recent_memory):.1f}MB "
                        f"(historical avg: {historical_avg:.1f}MB)",
                        affected_entity=worker_id,
                        metrics={
                            "resource": "memory",
                            "recent_avg": statistics.mean(recent_memory),
                            "historical_avg": historical_avg,
                        },
                    )
                )

        for anomaly in anomalies:
            self._notify_anomaly(anomaly)

        return anomalies[0] if anomalies else None

    def detect_stuck_queue(self, queue_size: int, tasks_in_progress: int) -> Optional[Anomaly]:
        """
        Detect if the task queue appears stuck.
        """
        if queue_size > 10 and tasks_in_progress == 0:
            anomaly = Anomaly(
                type=AnomalyType.STUCK_QUEUE,
                severity=AlertSeverity.ERROR,
                description=f"Task queue appears stuck: {queue_size} tasks pending "
                f"but no tasks in progress",
                affected_entity="queue",
                metrics={
                    "queue_size": queue_size,
                    "tasks_in_progress": tasks_in_progress,
                },
            )
            self._notify_anomaly(anomaly)
            return anomaly

        return None

    def get_recent_anomalies(self, limit: int = 10) -> list[Anomaly]:
        """Get most recent anomalies."""
        return sorted(
            self._detected_anomalies, key=lambda a: a.detected_at, reverse=True
        )[:limit]

    def get_anomaly_summary(self) -> dict[str, Any]:
        """Get summary of detected anomalies."""
        by_type = defaultdict(int)
        by_severity = defaultdict(int)

        for anomaly in self._detected_anomalies:
            by_type[anomaly.type.value] += 1
            by_severity[anomaly.severity.value] += 1

        return {
            "total": len(self._detected_anomalies),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "recent": [
                {
                    "type": a.type.value,
                    "severity": a.severity.value,
                    "description": a.description,
                    "detected_at": a.detected_at.isoformat(),
                }
                for a in self.get_recent_anomalies(5)
            ],
        }


# =============================================================================
# Resource Usage Tracking (adv-c-mon-004)
# =============================================================================


@dataclass
class ResourceSnapshot:
    """A point-in-time snapshot of resource usage."""

    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int


class ResourceTracker:
    """
    Tracks system and per-worker resource usage.

    Implements adv-c-mon-004.
    """

    def __init__(self, dashboard: StatusDashboard):
        self.dashboard = dashboard
        self._system_history: deque[ResourceSnapshot] = deque(maxlen=3600)  # 1 hour at 1/sec
        self._worker_history: dict[str, deque[ResourceSnapshot]] = defaultdict(
            lambda: deque(maxlen=3600)
        )
        self._running = False
        self._track_task: Optional[asyncio.Task[None]] = None
        self._baseline: Optional[ResourceSnapshot] = None

    async def start(self, interval: float = 1.0) -> None:
        """Start resource tracking."""
        self._running = True
        self._baseline = self._capture_system_snapshot()
        self._track_task = asyncio.create_task(self._track_loop(interval))

    async def stop(self) -> None:
        """Stop resource tracking."""
        self._running = False
        if self._track_task:
            self._track_task.cancel()
            try:
                await self._track_task
            except asyncio.CancelledError:
                pass

    async def _track_loop(self, interval: float) -> None:
        """Main tracking loop."""
        while self._running:
            try:
                # Capture system snapshot
                snapshot = self._capture_system_snapshot()
                self._system_history.append(snapshot)

                # Capture per-worker snapshots
                for worker_id, health in self.dashboard._worker_health.items():
                    worker_snapshot = ResourceSnapshot(
                        timestamp=datetime.now(),
                        cpu_percent=health.cpu_percent,
                        memory_mb=health.memory_mb,
                        memory_percent=0,  # Would need process-specific calculation
                        disk_usage_percent=0,
                        network_bytes_sent=0,
                        network_bytes_recv=0,
                    )
                    self._worker_history[worker_id].append(worker_snapshot)

                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(interval)

    def _capture_system_snapshot(self) -> ResourceSnapshot:
        """Capture current system resource usage."""
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        net = psutil.net_io_counters()

        return ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu,
            memory_mb=memory.used / (1024 * 1024),
            memory_percent=memory.percent,
            disk_usage_percent=disk.percent,
            network_bytes_sent=net.bytes_sent,
            network_bytes_recv=net.bytes_recv,
        )

    def get_current_usage(self) -> dict[str, Any]:
        """Get current resource usage."""
        if not self._system_history:
            snapshot = self._capture_system_snapshot()
        else:
            snapshot = self._system_history[-1]

        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "cpu_percent": snapshot.cpu_percent,
            "memory_mb": snapshot.memory_mb,
            "memory_percent": snapshot.memory_percent,
            "disk_usage_percent": snapshot.disk_usage_percent,
            "network": {
                "bytes_sent": snapshot.network_bytes_sent,
                "bytes_recv": snapshot.network_bytes_recv,
            },
        }

    def get_usage_history(
        self, duration_minutes: int = 60
    ) -> dict[str, list[dict[str, Any]]]:
        """Get resource usage history."""
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)

        system = [
            {
                "timestamp": s.timestamp.isoformat(),
                "cpu_percent": s.cpu_percent,
                "memory_mb": s.memory_mb,
            }
            for s in self._system_history
            if s.timestamp > cutoff
        ]

        workers = {}
        for worker_id, history in self._worker_history.items():
            workers[worker_id] = [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "cpu_percent": s.cpu_percent,
                    "memory_mb": s.memory_mb,
                }
                for s in history
                if s.timestamp > cutoff
            ]

        return {
            "system": system,
            "workers": workers,
        }

    def get_usage_summary(self) -> dict[str, Any]:
        """Get summary statistics of resource usage."""
        if not self._system_history:
            return {}

        cpu_values = [s.cpu_percent for s in self._system_history]
        memory_values = [s.memory_mb for s in self._system_history]

        return {
            "period": {
                "start": self._system_history[0].timestamp.isoformat(),
                "end": self._system_history[-1].timestamp.isoformat(),
                "samples": len(self._system_history),
            },
            "cpu": {
                "avg": statistics.mean(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "stddev": statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0,
            },
            "memory": {
                "avg": statistics.mean(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
                "current": memory_values[-1] if memory_values else 0,
            },
        }

    def estimate_capacity(self, task_duration: float, worker_count: int) -> dict[str, Any]:
        """
        Estimate system capacity for running tasks.

        Based on current resource usage, estimate how many tasks can run.
        """
        current = self.get_current_usage()

        # Assume each worker needs ~10% CPU and ~500MB memory
        cpu_per_worker = 10.0
        memory_per_worker = 500.0

        available_cpu = 100 - current["cpu_percent"]
        available_memory = (
            psutil.virtual_memory().available / (1024 * 1024)
        )  # MB available

        max_by_cpu = int(available_cpu / cpu_per_worker)
        max_by_memory = int(available_memory / memory_per_worker)
        max_workers = min(max_by_cpu, max_by_memory, worker_count)

        throughput = (60 / task_duration) * max_workers if task_duration > 0 else 0

        return {
            "available_cpu_percent": available_cpu,
            "available_memory_mb": available_memory,
            "max_additional_workers": max_workers,
            "estimated_throughput_per_minute": throughput,
            "bottleneck": "cpu" if max_by_cpu < max_by_memory else "memory",
        }


# =============================================================================
# Execution Timeline Visualization (adv-c-mon-005)
# =============================================================================


class TimelineVisualization:
    """
    Generates execution timeline data for visualization.

    Implements adv-c-mon-005.
    """

    def __init__(self, dashboard: StatusDashboard):
        self.dashboard = dashboard
        self._events: list[TimelineEvent] = []

    def record_event(
        self,
        event_type: str,
        task_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        details: str = "",
        duration: float = 0.0,
    ) -> None:
        """Record a timeline event."""
        event = TimelineEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            task_id=task_id,
            worker_id=worker_id,
            details=details,
            duration=duration,
        )
        self._events.append(event)

    def get_gantt_data(
        self, start_time: Optional[datetime] = None
    ) -> dict[str, Any]:
        """
        Get data formatted for Gantt chart visualization.

        Returns data structure suitable for rendering a Gantt chart.
        """
        if not self.dashboard._task_metrics:
            return {"workers": [], "tasks": []}

        # Determine time range
        if start_time is None:
            all_starts = [m.start_time for m in self.dashboard._task_metrics.values()]
            start_time = min(all_starts) if all_starts else datetime.now()

        # Group tasks by worker
        worker_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for task_id, metrics in self.dashboard._task_metrics.items():
            start_offset = (metrics.start_time - start_time).total_seconds()
            end_time = metrics.end_time or datetime.now()
            duration = (end_time - metrics.start_time).total_seconds()

            worker_tasks[metrics.worker_id].append(
                {
                    "task_id": task_id,
                    "start": start_offset,
                    "duration": duration,
                    "status": metrics.status.value,
                    "start_time": metrics.start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                }
            )

        # Sort tasks within each worker by start time
        for worker_id in worker_tasks:
            worker_tasks[worker_id].sort(key=lambda x: x["start"])

        return {
            "start_time": start_time.isoformat(),
            "workers": list(worker_tasks.keys()),
            "tasks_by_worker": dict(worker_tasks),
            "total_duration": (datetime.now() - start_time).total_seconds(),
        }

    def get_ascii_timeline(self, width: int = 80) -> str:
        """
        Generate an ASCII representation of the execution timeline.

        Returns a text-based visualization.
        """
        gantt = self.get_gantt_data()

        if not gantt["workers"]:
            return "No execution data available."

        lines = []
        total_duration = gantt["total_duration"]
        scale = (width - 20) / total_duration if total_duration > 0 else 1

        # Header
        lines.append("=" * width)
        lines.append("EXECUTION TIMELINE")
        lines.append("=" * width)
        lines.append("")

        # Time scale
        time_line = " " * 15 + "|"
        for i in range(0, int(total_duration) + 1, max(1, int(total_duration / 10))):
            pos = int(i * scale)
            time_line = time_line[:15 + pos] + str(i) + time_line[15 + pos + len(str(i)):]
        lines.append(time_line[:width])
        lines.append(" " * 15 + "+" + "-" * (width - 16))

        # Worker timelines
        for worker_id in gantt["workers"]:
            worker_label = f"{worker_id[:12]:>12} | "
            timeline = [" "] * (width - 16)

            for task in gantt["tasks_by_worker"][worker_id]:
                start_pos = int(task["start"] * scale)
                end_pos = int((task["start"] + task["duration"]) * scale)

                # Draw task bar
                char = "#" if task["status"] == "done" else "X" if task["status"] == "failed" else ">"
                for pos in range(max(0, start_pos), min(len(timeline), end_pos)):
                    timeline[pos] = char

            lines.append(worker_label + "".join(timeline))

        lines.append("")
        lines.append("Legend: # = completed, X = failed, > = in progress")
        lines.append("=" * width)

        return "\n".join(lines)

    def get_worker_utilization(self) -> dict[str, float]:
        """
        Calculate worker utilization percentages.

        Returns dict mapping worker_id to utilization (0-100).
        """
        gantt = self.get_gantt_data()

        if not gantt["workers"] or gantt["total_duration"] == 0:
            return {}

        utilization = {}
        total_duration = gantt["total_duration"]

        for worker_id in gantt["workers"]:
            busy_time = sum(task["duration"] for task in gantt["tasks_by_worker"][worker_id])
            utilization[worker_id] = (busy_time / total_duration) * 100

        return utilization

    def get_concurrency_over_time(
        self, bucket_seconds: float = 1.0
    ) -> list[dict[str, Any]]:
        """
        Get concurrency level over time.

        Returns list of {timestamp, concurrency} data points.
        """
        gantt = self.get_gantt_data()

        if not gantt["workers"]:
            return []

        total_duration = gantt["total_duration"]
        num_buckets = int(total_duration / bucket_seconds) + 1

        concurrency = [0] * num_buckets

        for worker_id in gantt["workers"]:
            for task in gantt["tasks_by_worker"][worker_id]:
                start_bucket = int(task["start"] / bucket_seconds)
                end_bucket = int((task["start"] + task["duration"]) / bucket_seconds)

                for bucket in range(max(0, start_bucket), min(num_buckets, end_bucket + 1)):
                    concurrency[bucket] += 1

        start_time = datetime.fromisoformat(gantt["start_time"])

        return [
            {
                "timestamp": (start_time + timedelta(seconds=i * bucket_seconds)).isoformat(),
                "seconds": i * bucket_seconds,
                "concurrency": concurrency[i],
            }
            for i in range(num_buckets)
        ]

    def export_timeline_json(self) -> str:
        """Export timeline data as JSON."""
        return json.dumps(
            {
                "gantt": self.get_gantt_data(),
                "utilization": self.get_worker_utilization(),
                "concurrency": self.get_concurrency_over_time(5.0),  # 5-second buckets
                "events": [
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "type": e.event_type,
                        "task_id": e.task_id,
                        "worker_id": e.worker_id,
                        "details": e.details,
                        "duration": e.duration,
                    }
                    for e in self._events
                ],
            },
            indent=2,
        )


# =============================================================================
# High-Level Monitor Interface
# =============================================================================


class TaskMonitor:
    """
    High-level interface for all monitoring features.

    Combines dashboard, health checks, anomaly detection, resource tracking,
    and timeline visualization.
    """

    def __init__(self):
        self.dashboard = StatusDashboard()
        self.health_checker = WorkerHealthChecker(self.dashboard)
        self.anomaly_detector = AnomalyDetector(self.dashboard)
        self.resource_tracker = ResourceTracker(self.dashboard)
        self.timeline = TimelineVisualization(self.dashboard)
        self._running = False

    async def start(self) -> None:
        """Start all monitoring components."""
        self._running = True
        await self.health_checker.start()
        await self.resource_tracker.start()

    async def stop(self) -> None:
        """Stop all monitoring components."""
        self._running = False
        await self.health_checker.stop()
        await self.resource_tracker.stop()

    def register_worker(self, worker_id: str, pid: Optional[int] = None) -> None:
        """Register a worker for monitoring."""
        self.dashboard.register_worker(worker_id)
        if pid:
            self.health_checker.register_worker_pid(worker_id, pid)

    def record_task_start(self, task_id: str, worker_id: str) -> None:
        """Record task starting execution."""
        self.dashboard.record_task_start(task_id, worker_id)
        self.timeline.record_event(
            event_type="task_start",
            task_id=task_id,
            worker_id=worker_id,
        )

    def record_task_complete(
        self,
        task_id: str,
        success: bool = True,
        duration: Optional[float] = None,
    ) -> None:
        """Record task completing."""
        self.dashboard.record_task_complete(task_id, success)

        metrics = self.dashboard._task_metrics.get(task_id)
        if metrics:
            actual_duration = duration or metrics.duration_seconds
            worker_id = metrics.worker_id

            self.timeline.record_event(
                event_type="task_complete",
                task_id=task_id,
                worker_id=worker_id,
                details="success" if success else "failed",
                duration=actual_duration,
            )

            # Record for anomaly detection
            self.anomaly_detector.record_task_duration(task_id, actual_duration)
            self.anomaly_detector.record_task_result(worker_id, success)

            # Check for anomalies
            self.anomaly_detector.detect_slow_task(task_id, actual_duration)
            self.anomaly_detector.detect_high_failure_rate(worker_id)

    def update_worker_heartbeat(self, worker_id: str) -> None:
        """Update worker heartbeat."""
        self.dashboard.update_worker_heartbeat(worker_id)

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            "dashboard": self.dashboard.get_dashboard_data(),
            "anomalies": self.anomaly_detector.get_anomaly_summary(),
            "resources": self.resource_tracker.get_usage_summary(),
            "utilization": self.timeline.get_worker_utilization(),
        }

    def get_alerts(self, include_resolved: bool = False) -> list[dict[str, Any]]:
        """Get current alerts."""
        alerts = self.dashboard._alerts
        if not include_resolved:
            alerts = [a for a in alerts if not a.resolved]

        return [
            {
                "id": a.id,
                "type": a.type.value,
                "severity": a.severity.value,
                "message": a.message,
                "timestamp": a.timestamp.isoformat(),
                "acknowledged": a.acknowledged,
                "resolved": a.resolved,
            }
            for a in alerts
        ]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.dashboard._alerts:
            if alert.id == alert_id:
                alert.acknowledge()
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.dashboard._alerts:
            if alert.id == alert_id:
                alert.resolve()
                return True
        return False

    def get_timeline_ascii(self, width: int = 80) -> str:
        """Get ASCII timeline visualization."""
        return self.timeline.get_ascii_timeline(width)

    def export_all(self) -> str:
        """Export all monitoring data as JSON."""
        return json.dumps(
            {
                "exported_at": datetime.now().isoformat(),
                "dashboard": self.dashboard.get_dashboard_data(),
                "anomalies": self.anomaly_detector.get_anomaly_summary(),
                "resources": self.resource_tracker.get_usage_summary(),
                "timeline": json.loads(self.timeline.export_timeline_json()),
            },
            indent=2,
        )
