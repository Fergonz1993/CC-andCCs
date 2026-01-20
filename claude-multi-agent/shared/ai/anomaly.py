"""
Anomaly Detection in Task Execution (adv-ai-004)

Detects unusual patterns in task execution including:
- Abnormal execution times
- Unusual failure rates
- Suspicious patterns
- Resource usage anomalies
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import math


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""

    SLOW_EXECUTION = "slow_execution"
    FAST_EXECUTION = "fast_execution"
    HIGH_FAILURE_RATE = "high_failure_rate"
    SUDDEN_FAILURES = "sudden_failures"
    STUCK_TASK = "stuck_task"
    WORKER_DEGRADATION = "worker_degradation"
    DEPENDENCY_CYCLE = "dependency_cycle"
    RESOURCE_CONTENTION = "resource_contention"
    QUEUE_BUILDUP = "queue_buildup"
    UNUSUAL_PATTERN = "unusual_pattern"


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AnomalyAlert:
    """An alert for a detected anomaly."""

    anomaly_type: AnomalyType
    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    worker_id: Optional[str] = None
    suggested_action: str = ""
    detected_at: datetime = field(default_factory=datetime.now)


class AnomalyDetector:
    """
    Detects anomalies in task execution using statistical methods.

    Uses simple statistical techniques like:
    - Z-score analysis for outlier detection
    - Moving averages for trend detection
    - Threshold-based alerts
    """

    # Configuration thresholds
    Z_SCORE_THRESHOLD = 2.5  # Standard deviations for outlier
    FAILURE_RATE_WARNING = 0.2  # 20% failure rate triggers warning
    FAILURE_RATE_CRITICAL = 0.5  # 50% failure rate is critical
    STUCK_TASK_THRESHOLD_MINUTES = 60  # Task stuck if running > 60 min
    QUEUE_BUILDUP_THRESHOLD = 10  # More than 10 waiting tasks
    MIN_SAMPLES_FOR_STATS = 5  # Minimum samples needed for statistical analysis

    def __init__(self):
        # Historical data storage
        self.execution_times: Dict[str, List[float]] = {}  # task_type -> times
        self.failure_counts: Dict[str, Tuple[int, int]] = {}  # task_type -> (success, fail)
        self.worker_history: Dict[str, List[Dict[str, Any]]] = {}  # worker -> executions
        self.recent_alerts: List[AnomalyAlert] = []

        # Moving averages
        self.queue_depth_history: List[Tuple[datetime, int]] = []
        self.throughput_history: List[Tuple[datetime, int]] = []  # timestamp, completed count

    def analyze_task_execution(
        self,
        task: Dict[str, Any],
        duration_seconds: float,
        success: bool
    ) -> List[AnomalyAlert]:
        """
        Analyze a completed task execution for anomalies.

        Returns list of any detected anomalies.
        """
        alerts = []
        task_id = task.get("id", "unknown")
        task_type = self._get_task_type(task)

        # Record execution
        self._record_execution(task, duration_seconds, success)

        # Check for duration anomalies
        duration_alert = self._check_duration_anomaly(
            task_type, duration_seconds, task_id
        )
        if duration_alert:
            alerts.append(duration_alert)

        # Check for failure rate anomalies
        failure_alert = self._check_failure_rate_anomaly(task_type, task_id)
        if failure_alert:
            alerts.append(failure_alert)

        # Check for sudden failure spike
        spike_alert = self._check_failure_spike(task_type, task_id)
        if spike_alert:
            alerts.append(spike_alert)

        return alerts

    def analyze_worker(
        self,
        worker_id: str,
        task: Dict[str, Any],
        duration_seconds: float,
        success: bool
    ) -> List[AnomalyAlert]:
        """Analyze worker performance for anomalies."""
        alerts = []

        # Record worker execution
        if worker_id not in self.worker_history:
            self.worker_history[worker_id] = []

        self.worker_history[worker_id].append({
            "task_id": task.get("id"),
            "duration": duration_seconds,
            "success": success,
            "timestamp": datetime.now()
        })

        # Keep only last 100 executions
        if len(self.worker_history[worker_id]) > 100:
            self.worker_history[worker_id] = self.worker_history[worker_id][-100:]

        # Check for worker degradation
        degradation_alert = self._check_worker_degradation(worker_id)
        if degradation_alert:
            alerts.append(degradation_alert)

        return alerts

    def analyze_system_state(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]]
    ) -> List[AnomalyAlert]:
        """
        Analyze overall system state for anomalies.

        Should be called periodically to detect system-level issues.
        """
        alerts = []

        # Check for stuck tasks
        stuck_alerts = self._check_stuck_tasks(tasks)
        alerts.extend(stuck_alerts)

        # Check for queue buildup
        queue_alert = self._check_queue_buildup(tasks)
        if queue_alert:
            alerts.append(queue_alert)

        # Check for dependency cycles
        cycle_alert = self._check_dependency_cycles(tasks)
        if cycle_alert:
            alerts.append(cycle_alert)

        # Check for resource contention
        contention_alert = self._check_resource_contention(tasks)
        if contention_alert:
            alerts.append(contention_alert)

        return alerts

    def _record_execution(
        self,
        task: Dict[str, Any],
        duration_seconds: float,
        success: bool
    ) -> None:
        """Record execution data for future analysis."""
        task_type = self._get_task_type(task)

        # Record execution time
        if task_type not in self.execution_times:
            self.execution_times[task_type] = []
        self.execution_times[task_type].append(duration_seconds)

        # Keep only last 100 entries
        if len(self.execution_times[task_type]) > 100:
            self.execution_times[task_type] = self.execution_times[task_type][-100:]

        # Record failure count
        if task_type not in self.failure_counts:
            self.failure_counts[task_type] = (0, 0)

        successes, failures = self.failure_counts[task_type]
        if success:
            self.failure_counts[task_type] = (successes + 1, failures)
        else:
            self.failure_counts[task_type] = (successes, failures + 1)

    def _check_duration_anomaly(
        self,
        task_type: str,
        duration_seconds: float,
        task_id: str
    ) -> Optional[AnomalyAlert]:
        """Check if duration is anomalous."""
        times = self.execution_times.get(task_type, [])

        if len(times) < self.MIN_SAMPLES_FOR_STATS:
            return None

        mean = sum(times) / len(times)
        variance = sum((t - mean) ** 2 for t in times) / len(times)
        std_dev = math.sqrt(variance) if variance > 0 else 1

        z_score = (duration_seconds - mean) / std_dev if std_dev > 0 else 0

        if z_score > self.Z_SCORE_THRESHOLD:
            return AnomalyAlert(
                anomaly_type=AnomalyType.SLOW_EXECUTION,
                severity=AlertSeverity.WARNING,
                message=f"Task took unusually long: {duration_seconds/60:.1f} min vs avg {mean/60:.1f} min",
                details={
                    "duration_seconds": duration_seconds,
                    "mean": mean,
                    "std_dev": std_dev,
                    "z_score": z_score
                },
                task_id=task_id,
                suggested_action="Check for performance issues or blocking operations"
            )

        if z_score < -self.Z_SCORE_THRESHOLD and duration_seconds < mean * 0.5:
            return AnomalyAlert(
                anomaly_type=AnomalyType.FAST_EXECUTION,
                severity=AlertSeverity.INFO,
                message=f"Task completed unusually fast: {duration_seconds/60:.1f} min vs avg {mean/60:.1f} min",
                details={
                    "duration_seconds": duration_seconds,
                    "mean": mean,
                    "std_dev": std_dev,
                    "z_score": z_score
                },
                task_id=task_id,
                suggested_action="Verify task completed fully - may have been short-circuited"
            )

        return None

    def _check_failure_rate_anomaly(
        self,
        task_type: str,
        task_id: str
    ) -> Optional[AnomalyAlert]:
        """Check for high failure rate."""
        counts = self.failure_counts.get(task_type)
        if not counts:
            return None

        successes, failures = counts
        total = successes + failures

        if total < self.MIN_SAMPLES_FOR_STATS:
            return None

        failure_rate = failures / total

        if failure_rate >= self.FAILURE_RATE_CRITICAL:
            return AnomalyAlert(
                anomaly_type=AnomalyType.HIGH_FAILURE_RATE,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical failure rate: {failure_rate*100:.0f}% for task type '{task_type}'",
                details={
                    "task_type": task_type,
                    "successes": successes,
                    "failures": failures,
                    "failure_rate": failure_rate
                },
                task_id=task_id,
                suggested_action="Investigate root cause immediately - consider pausing this task type"
            )

        if failure_rate >= self.FAILURE_RATE_WARNING:
            return AnomalyAlert(
                anomaly_type=AnomalyType.HIGH_FAILURE_RATE,
                severity=AlertSeverity.WARNING,
                message=f"Elevated failure rate: {failure_rate*100:.0f}% for task type '{task_type}'",
                details={
                    "task_type": task_type,
                    "successes": successes,
                    "failures": failures,
                    "failure_rate": failure_rate
                },
                task_id=task_id,
                suggested_action="Review recent failures for common patterns"
            )

        return None

    def _check_failure_spike(
        self,
        task_type: str,
        task_id: str
    ) -> Optional[AnomalyAlert]:
        """Check for sudden increase in failures."""
        times = self.execution_times.get(task_type, [])

        if len(times) < 10:
            return None

        # Compare recent vs historical failure rate
        counts = self.failure_counts.get(task_type, (0, 0))
        total = sum(counts)
        if total < 10:
            return None

        # Check last 5 executions vs previous
        # This is a simplification - in production you'd track success/fail per execution
        recent_times = times[-5:]
        older_times = times[-10:-5] if len(times) >= 10 else times[:-5]

        # Use time variance as proxy for problems
        recent_variance = sum((t - sum(recent_times)/len(recent_times))**2 for t in recent_times) / len(recent_times)
        older_variance = sum((t - sum(older_times)/len(older_times))**2 for t in older_times) / len(older_times) if older_times else 0

        if recent_variance > older_variance * 3 and older_variance > 0:
            return AnomalyAlert(
                anomaly_type=AnomalyType.SUDDEN_FAILURES,
                severity=AlertSeverity.WARNING,
                message=f"Sudden instability detected for task type '{task_type}'",
                details={
                    "recent_variance": recent_variance,
                    "historical_variance": older_variance
                },
                task_id=task_id,
                suggested_action="Check for environmental changes or new code deployments"
            )

        return None

    def _check_worker_degradation(self, worker_id: str) -> Optional[AnomalyAlert]:
        """Check if a worker's performance is degrading."""
        history = self.worker_history.get(worker_id, [])

        if len(history) < 10:
            return None

        # Compare recent vs historical performance
        recent = history[-5:]
        older = history[-10:-5]

        recent_success_rate = sum(1 for h in recent if h["success"]) / len(recent)
        older_success_rate = sum(1 for h in older if h["success"]) / len(older)

        # Check for significant degradation
        if older_success_rate > 0.8 and recent_success_rate < 0.5:
            return AnomalyAlert(
                anomaly_type=AnomalyType.WORKER_DEGRADATION,
                severity=AlertSeverity.WARNING,
                message=f"Worker '{worker_id}' performance degrading: {older_success_rate*100:.0f}% -> {recent_success_rate*100:.0f}%",
                details={
                    "recent_success_rate": recent_success_rate,
                    "historical_success_rate": older_success_rate
                },
                worker_id=worker_id,
                suggested_action="Check worker health and consider restarting"
            )

        # Check for duration increase
        recent_avg_time = sum(h["duration"] for h in recent) / len(recent)
        older_avg_time = sum(h["duration"] for h in older) / len(older)

        if recent_avg_time > older_avg_time * 2:
            return AnomalyAlert(
                anomaly_type=AnomalyType.WORKER_DEGRADATION,
                severity=AlertSeverity.INFO,
                message=f"Worker '{worker_id}' slowing down: avg time {recent_avg_time/60:.1f}min vs {older_avg_time/60:.1f}min",
                details={
                    "recent_avg_time": recent_avg_time,
                    "historical_avg_time": older_avg_time
                },
                worker_id=worker_id,
                suggested_action="Monitor worker - may be overloaded or experiencing issues"
            )

        return None

    def _check_stuck_tasks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[AnomalyAlert]:
        """Check for tasks that appear stuck."""
        alerts = []
        now = datetime.now()

        for task in tasks:
            if task.get("status") != "in_progress":
                continue

            started_at = task.get("started_at") or task.get("claimed_at")
            if not started_at:
                continue

            try:
                if isinstance(started_at, str):
                    started = datetime.fromisoformat(started_at.replace("Z", "+00:00")).replace(tzinfo=None)
                else:
                    started = started_at

                elapsed_minutes = (now - started).total_seconds() / 60

                if elapsed_minutes > self.STUCK_TASK_THRESHOLD_MINUTES:
                    alerts.append(AnomalyAlert(
                        anomaly_type=AnomalyType.STUCK_TASK,
                        severity=AlertSeverity.WARNING,
                        message=f"Task running for {elapsed_minutes:.0f} minutes",
                        details={
                            "started_at": str(started),
                            "elapsed_minutes": elapsed_minutes
                        },
                        task_id=task.get("id"),
                        worker_id=task.get("claimed_by"),
                        suggested_action="Check worker status - task may be stuck or worker may have crashed"
                    ))
            except (ValueError, TypeError):
                pass

        return alerts

    def _check_queue_buildup(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Optional[AnomalyAlert]:
        """Check for queue buildup."""
        available = [t for t in tasks if t.get("status") == "available"]
        queue_depth = len(available)

        # Record history
        self.queue_depth_history.append((datetime.now(), queue_depth))
        if len(self.queue_depth_history) > 100:
            self.queue_depth_history = self.queue_depth_history[-100:]

        if queue_depth > self.QUEUE_BUILDUP_THRESHOLD:
            # Check if it's growing
            if len(self.queue_depth_history) >= 5:
                recent = [d for _, d in self.queue_depth_history[-5:]]
                if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
                    return AnomalyAlert(
                        anomaly_type=AnomalyType.QUEUE_BUILDUP,
                        severity=AlertSeverity.WARNING,
                        message=f"Queue building up: {queue_depth} tasks waiting",
                        details={
                            "queue_depth": queue_depth,
                            "trend": "increasing"
                        },
                        suggested_action="Consider adding more workers or reviewing task throughput"
                    )

            return AnomalyAlert(
                anomaly_type=AnomalyType.QUEUE_BUILDUP,
                severity=AlertSeverity.INFO,
                message=f"Large queue: {queue_depth} tasks waiting",
                details={"queue_depth": queue_depth},
                suggested_action="Monitor queue depth"
            )

        return None

    def _check_dependency_cycles(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Optional[AnomalyAlert]:
        """Check for dependency cycles in tasks."""
        task_map = {t.get("id"): t for t in tasks}

        def has_cycle(task_id: str, visited: set, path: set) -> Optional[List[str]]:
            if task_id in path:
                return list(path)
            if task_id in visited:
                return None

            visited.add(task_id)
            path.add(task_id)

            task = task_map.get(task_id)
            if task:
                for dep in (task.get("dependencies") or []):
                    cycle = has_cycle(dep, visited, path)
                    if cycle:
                        return cycle

            path.remove(task_id)
            return None

        visited: set = set()
        for task_id in task_map:
            cycle = has_cycle(task_id, visited, set())
            if cycle:
                return AnomalyAlert(
                    anomaly_type=AnomalyType.DEPENDENCY_CYCLE,
                    severity=AlertSeverity.ERROR,
                    message=f"Dependency cycle detected involving {len(cycle)} tasks",
                    details={"cycle": cycle},
                    suggested_action="Break dependency cycle to allow tasks to proceed"
                )

        return None

    def _check_resource_contention(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Optional[AnomalyAlert]:
        """Check for resource contention between tasks."""
        in_progress = [t for t in tasks if t.get("status") == "in_progress"]

        if len(in_progress) < 2:
            return None

        # Build file usage map
        file_usage: Dict[str, List[str]] = {}
        for task in in_progress:
            files = task.get("context", {}).get("files") or []
            for f in files:
                if f not in file_usage:
                    file_usage[f] = []
                file_usage[f].append(task.get("id", "unknown"))

        # Find conflicts
        conflicts = {f: tasks for f, tasks in file_usage.items() if len(tasks) > 1}

        if conflicts:
            return AnomalyAlert(
                anomaly_type=AnomalyType.RESOURCE_CONTENTION,
                severity=AlertSeverity.WARNING,
                message=f"Resource contention: {len(conflicts)} files being modified by multiple tasks",
                details={"conflicts": conflicts},
                suggested_action="Consider serializing tasks that modify the same files"
            )

        return None

    def _get_task_type(self, task: Dict[str, Any]) -> str:
        """Get task type for categorization."""
        tags = task.get("tags") or []
        if tags:
            return "|".join(sorted(tags[:3]))
        return "general"

    def get_recent_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 20
    ) -> List[AnomalyAlert]:
        """Get recent alerts, optionally filtered by severity."""
        alerts = self.recent_alerts[-100:]  # Keep last 100

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts[-limit:]

    def clear_old_data(self, days: int = 7) -> None:
        """Clear data older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)

        for worker_id in list(self.worker_history.keys()):
            self.worker_history[worker_id] = [
                h for h in self.worker_history[worker_id]
                if h.get("timestamp", datetime.now()) > cutoff
            ]

        self.queue_depth_history = [
            (ts, d) for ts, d in self.queue_depth_history
            if ts > cutoff
        ]

        self.recent_alerts = [
            a for a in self.recent_alerts
            if a.detected_at > cutoff
        ]
