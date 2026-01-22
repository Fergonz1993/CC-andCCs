"""
Task queue metrics export for monitoring and observability.

Provides queue size, throughput, and timing metrics that can be exported
to monitoring systems like Prometheus or logged for analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class TaskMetrics:
    """Metrics for a single task lifecycle."""

    task_id: str
    created_at: float
    claimed_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "available"

    @property
    def wait_time_seconds(self) -> Optional[float]:
        """Time from creation to claim."""
        if self.claimed_at is not None:
            return self.claimed_at - self.created_at
        return None

    @property
    def execution_time_seconds(self) -> Optional[float]:
        """Time from claim to completion."""
        if self.claimed_at is not None and self.completed_at is not None:
            return self.completed_at - self.claimed_at
        return None

    @property
    def total_time_seconds(self) -> Optional[float]:
        """Total time from creation to completion."""
        if self.completed_at is not None:
            return self.completed_at - self.created_at
        return None


@dataclass
class QueueMetrics:
    """Aggregate metrics for the task queue."""

    total_tasks: int = 0
    available_tasks: int = 0
    claimed_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

    # Throughput metrics
    tasks_completed_last_minute: int = 0
    tasks_completed_last_hour: int = 0

    # Timing metrics (in seconds)
    avg_wait_time: float = 0.0
    avg_execution_time: float = 0.0
    avg_total_time: float = 0.0

    # Queue health
    oldest_pending_task_age_seconds: Optional[float] = None
    active_agents: int = 0

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsCollector:
    """Collects and exports task queue metrics."""

    def __init__(self, metrics_dir: Optional[Path] = None):
        self._task_metrics: dict[str, TaskMetrics] = {}
        self._completion_times: list[float] = []  # timestamps of completions
        self.metrics_dir = metrics_dir

    def record_task_created(self, task_id: str) -> None:
        """Record when a task is created."""
        self._task_metrics[task_id] = TaskMetrics(
            task_id=task_id,
            created_at=time.time(),
            status="available",
        )

    def record_task_claimed(self, task_id: str) -> None:
        """Record when a task is claimed."""
        if task_id in self._task_metrics:
            self._task_metrics[task_id].claimed_at = time.time()
            self._task_metrics[task_id].status = "claimed"

    def record_task_completed(self, task_id: str, success: bool = True) -> None:
        """Record when a task is completed or failed."""
        now = time.time()
        if task_id in self._task_metrics:
            self._task_metrics[task_id].completed_at = now
            self._task_metrics[task_id].status = "done" if success else "failed"
        self._completion_times.append(now)

        # Prune old completion times (keep last hour)
        hour_ago = now - 3600
        self._completion_times = [t for t in self._completion_times if t > hour_ago]

        # Prune old completed TaskMetrics to prevent unbounded memory growth
        # Keep only completed tasks from the last hour for timing calculations
        keys_to_remove = [
            tid for tid, metrics in self._task_metrics.items()
            if metrics.completed_at is not None and metrics.completed_at < hour_ago
        ]
        for tid in keys_to_remove:
            del self._task_metrics[tid]

    def get_queue_metrics(
        self,
        tasks: list[dict[str, Any]],
        agents: list[dict[str, Any]],
    ) -> QueueMetrics:
        """Calculate current queue metrics from task and agent state."""
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        # Count by status
        status_counts: dict[str, int] = {
            "available": 0,
            "claimed": 0,
            "done": 0,
            "failed": 0,
        }
        for task in tasks:
            status = task.get("status", "available")
            if status in status_counts:
                status_counts[status] += 1

        # Calculate throughput
        tasks_last_minute = sum(1 for t in self._completion_times if t > minute_ago)
        tasks_last_hour = sum(1 for t in self._completion_times if t > hour_ago)

        # Calculate timing averages
        wait_times: list[float] = []
        exec_times: list[float] = []
        total_times: list[float] = []
        oldest_pending_age: Optional[float] = None

        for tm in self._task_metrics.values():
            if tm.wait_time_seconds is not None:
                wait_times.append(tm.wait_time_seconds)
            if tm.execution_time_seconds is not None:
                exec_times.append(tm.execution_time_seconds)
            if tm.total_time_seconds is not None:
                total_times.append(tm.total_time_seconds)

            # Track oldest pending task
            if tm.status == "available":
                age = now - tm.created_at
                if oldest_pending_age is None or age > oldest_pending_age:
                    oldest_pending_age = age

        # Count active agents
        active_agents = sum(
            1 for a in agents if a.get("status") == "active"
        )

        return QueueMetrics(
            total_tasks=len(tasks),
            available_tasks=status_counts["available"],
            claimed_tasks=status_counts["claimed"],
            completed_tasks=status_counts["done"],
            failed_tasks=status_counts["failed"],
            tasks_completed_last_minute=tasks_last_minute,
            tasks_completed_last_hour=tasks_last_hour,
            avg_wait_time=sum(wait_times) / len(wait_times) if wait_times else 0.0,
            avg_execution_time=sum(exec_times) / len(exec_times) if exec_times else 0.0,
            avg_total_time=sum(total_times) / len(total_times) if total_times else 0.0,
            oldest_pending_task_age_seconds=oldest_pending_age,
            active_agents=active_agents,
        )

    def export_metrics(
        self,
        metrics: QueueMetrics,
        format: str = "json",
    ) -> str:
        """Export metrics in the specified format."""
        if format == "json":
            return json.dumps(
                {
                    "queue_size": {
                        "total": metrics.total_tasks,
                        "available": metrics.available_tasks,
                        "claimed": metrics.claimed_tasks,
                        "completed": metrics.completed_tasks,
                        "failed": metrics.failed_tasks,
                    },
                    "throughput": {
                        "tasks_per_minute": metrics.tasks_completed_last_minute,
                        "tasks_per_hour": metrics.tasks_completed_last_hour,
                    },
                    "timing_seconds": {
                        "avg_wait_time": round(metrics.avg_wait_time, 3),
                        "avg_execution_time": round(metrics.avg_execution_time, 3),
                        "avg_total_time": round(metrics.avg_total_time, 3),
                        "oldest_pending_age": (
                            round(metrics.oldest_pending_task_age_seconds, 3)
                            if metrics.oldest_pending_task_age_seconds is not None
                            else None
                        ),
                    },
                    "agents": {
                        "active": metrics.active_agents,
                    },
                    "timestamp": metrics.timestamp,
                },
                indent=2,
            )
        elif format == "prometheus":
            lines = [
                "# HELP orchestrator_queue_total Total tasks in queue",
                "# TYPE orchestrator_queue_total gauge",
                f"orchestrator_queue_total {metrics.total_tasks}",
                "# HELP orchestrator_queue_available Available tasks",
                "# TYPE orchestrator_queue_available gauge",
                f"orchestrator_queue_available {metrics.available_tasks}",
                "# HELP orchestrator_queue_claimed Claimed tasks",
                "# TYPE orchestrator_queue_claimed gauge",
                f"orchestrator_queue_claimed {metrics.claimed_tasks}",
                "# HELP orchestrator_queue_completed Completed tasks",
                "# TYPE orchestrator_queue_completed gauge",
                f"orchestrator_queue_completed {metrics.completed_tasks}",
                "# HELP orchestrator_queue_failed Failed tasks",
                "# TYPE orchestrator_queue_failed gauge",
                f"orchestrator_queue_failed {metrics.failed_tasks}",
                "# HELP orchestrator_throughput_per_minute Tasks completed per minute",
                "# TYPE orchestrator_throughput_per_minute gauge",
                f"orchestrator_throughput_per_minute {metrics.tasks_completed_last_minute}",
                "# HELP orchestrator_avg_wait_time_seconds Average wait time",
                "# TYPE orchestrator_avg_wait_time_seconds gauge",
                f"orchestrator_avg_wait_time_seconds {metrics.avg_wait_time:.3f}",
                "# HELP orchestrator_avg_execution_time_seconds Average execution time",
                "# TYPE orchestrator_avg_execution_time_seconds gauge",
                f"orchestrator_avg_execution_time_seconds {metrics.avg_execution_time:.3f}",
                "# HELP orchestrator_active_agents Number of active agents",
                "# TYPE orchestrator_active_agents gauge",
                f"orchestrator_active_agents {metrics.active_agents}",
            ]
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_metrics(self, metrics: QueueMetrics, path: Path) -> None:
        """Save metrics to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.export_metrics(metrics, "json"), encoding="utf-8")
