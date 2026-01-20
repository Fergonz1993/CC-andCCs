"""
Shared Metrics Interface (adv-cross-006)

Provides a common interface for collecting and exporting metrics across
all coordination options. Supports various metric types and exporters.
"""

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"         # Cumulative count (always increasing)
    GAUGE = "gauge"             # Point-in-time value (can go up or down)
    HISTOGRAM = "histogram"     # Distribution of values
    TIMER = "timer"             # Duration measurements
    RATE = "rate"               # Rate of events over time


@dataclass
class MetricValue:
    """A single metric value with timestamp."""
    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
        }


@dataclass
class Metric:
    """A metric definition with its values."""
    name: str
    type: MetricType
    description: str = ""
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    values: List[MetricValue] = field(default_factory=list)

    # For histograms/timers
    buckets: List[float] = field(default_factory=list)
    bucket_counts: Dict[float, int] = field(default_factory=dict)

    # For rate calculations
    window_seconds: float = 60.0
    _events: List[datetime] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "unit": self.unit,
            "labels": self.labels,
        }

        if self.type == MetricType.HISTOGRAM:
            d["buckets"] = self.buckets
            d["bucket_counts"] = {str(k): v for k, v in self.bucket_counts.items()}

        if self.values:
            d["latest_value"] = self.values[-1].to_dict()
            d["value_count"] = len(self.values)

        return d


class MetricsCollector:
    """
    Collects and manages metrics across coordination options.

    Features:
    - Multiple metric types (counter, gauge, histogram, timer, rate)
    - Label support for dimensional metrics
    - Thread-safe operations
    - Automatic aggregation
    """

    def __init__(
        self,
        option: Optional[str] = None,
        prefix: str = "coordination",
        max_values: int = 1000,
    ):
        """
        Initialize the metrics collector.

        Args:
            option: Coordination option ('A', 'B', or 'C')
            prefix: Prefix for all metric names
            max_values: Maximum values to retain per metric
        """
        self.option = option
        self.prefix = prefix
        self.max_values = max_values

        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()
        self._start_time = datetime.now()

        # Pre-define standard coordination metrics
        self._define_standard_metrics()

    def _define_standard_metrics(self) -> None:
        """Define standard metrics for coordination."""
        self.define(
            "tasks_total",
            MetricType.COUNTER,
            "Total number of tasks created",
            labels=["status"],
        )
        self.define(
            "tasks_active",
            MetricType.GAUGE,
            "Number of active tasks",
            labels=["status"],
        )
        self.define(
            "task_duration_seconds",
            MetricType.HISTOGRAM,
            "Task execution duration in seconds",
            unit="seconds",
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 600],
        )
        self.define(
            "task_wait_time_seconds",
            MetricType.HISTOGRAM,
            "Task wait time before execution",
            unit="seconds",
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],
        )
        self.define(
            "agents_active",
            MetricType.GAUGE,
            "Number of active agents",
            labels=["role"],
        )
        self.define(
            "task_throughput",
            MetricType.RATE,
            "Tasks completed per minute",
            unit="tasks/min",
        )
        self.define(
            "claim_attempts",
            MetricType.COUNTER,
            "Number of task claim attempts",
            labels=["result"],
        )
        self.define(
            "sync_operations",
            MetricType.COUNTER,
            "Number of sync operations",
            labels=["direction", "result"],
        )
        self.define(
            "discoveries_total",
            MetricType.COUNTER,
            "Total discoveries shared",
        )
        self.define(
            "queue_depth",
            MetricType.GAUGE,
            "Number of tasks in queue",
            labels=["status"],
        )

    def _get_full_name(self, name: str) -> str:
        """Get the full metric name with prefix."""
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name

    def define(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        unit: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Metric:
        """
        Define a new metric.

        Args:
            name: Metric name (without prefix)
            metric_type: Type of metric
            description: Human-readable description
            unit: Unit of measurement
            labels: Label names for dimensional data
            buckets: Bucket boundaries for histograms
        """
        full_name = self._get_full_name(name)

        with self._lock:
            if full_name in self._metrics:
                return self._metrics[full_name]

            metric = Metric(
                name=full_name,
                type=metric_type,
                description=description,
                unit=unit,
                labels=labels or [],
                buckets=buckets or [],
            )

            if buckets:
                metric.bucket_counts = {b: 0 for b in buckets}
                metric.bucket_counts[float("inf")] = 0

            self._metrics[full_name] = metric
            return metric

    def increment(
        self,
        name: str,
        value: Union[int, float] = 1,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        full_name = self._get_full_name(name)

        with self._lock:
            if full_name not in self._metrics:
                self.define(name, MetricType.COUNTER, labels=list((labels or {}).keys()))

            metric = self._metrics[full_name]
            if metric.type != MetricType.COUNTER:
                raise ValueError(f"Metric {name} is not a counter")

            # Get current value and increment
            current = metric.values[-1].value if metric.values else 0
            metric.values.append(MetricValue(
                value=current + value,
                labels=labels or {},
            ))

            # Trim values if needed
            if len(metric.values) > self.max_values:
                metric.values = metric.values[-self.max_values:]

    def set_gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric value."""
        full_name = self._get_full_name(name)

        with self._lock:
            if full_name not in self._metrics:
                self.define(name, MetricType.GAUGE, labels=list((labels or {}).keys()))

            metric = self._metrics[full_name]
            if metric.type != MetricType.GAUGE:
                raise ValueError(f"Metric {name} is not a gauge")

            metric.values.append(MetricValue(
                value=value,
                labels=labels or {},
            ))

            if len(metric.values) > self.max_values:
                metric.values = metric.values[-self.max_values:]

    def observe(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a value for a histogram metric."""
        full_name = self._get_full_name(name)

        with self._lock:
            if full_name not in self._metrics:
                self.define(
                    name,
                    MetricType.HISTOGRAM,
                    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 300],
                )

            metric = self._metrics[full_name]
            if metric.type not in (MetricType.HISTOGRAM, MetricType.TIMER):
                raise ValueError(f"Metric {name} is not a histogram or timer")

            metric.values.append(MetricValue(
                value=value,
                labels=labels or {},
            ))

            # Update bucket counts
            for bucket in sorted(metric.bucket_counts.keys()):
                if value <= bucket:
                    metric.bucket_counts[bucket] += 1
                    break

            if len(metric.values) > self.max_values:
                metric.values = metric.values[-self.max_values:]

    def record_event(self, name: str) -> None:
        """Record an event for rate calculation."""
        full_name = self._get_full_name(name)

        with self._lock:
            if full_name not in self._metrics:
                self.define(name, MetricType.RATE)

            metric = self._metrics[full_name]
            if metric.type != MetricType.RATE:
                raise ValueError(f"Metric {name} is not a rate metric")

            metric._events.append(datetime.now())

            # Clean up old events
            cutoff = datetime.now() - timedelta(seconds=metric.window_seconds)
            metric._events = [e for e in metric._events if e > cutoff]

    def get_rate(self, name: str) -> float:
        """Get the current rate (events per minute) for a rate metric."""
        full_name = self._get_full_name(name)

        with self._lock:
            if full_name not in self._metrics:
                return 0.0

            metric = self._metrics[full_name]
            if metric.type != MetricType.RATE:
                raise ValueError(f"Metric {name} is not a rate metric")

            # Clean up old events
            cutoff = datetime.now() - timedelta(seconds=metric.window_seconds)
            metric._events = [e for e in metric._events if e > cutoff]

            # Calculate rate (events per minute)
            if not metric._events:
                return 0.0

            return len(metric._events) * 60.0 / metric.window_seconds

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        full_name = self._get_full_name(name)
        return self._metrics.get(full_name)

    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all defined metrics."""
        return dict(self._metrics)

    def get_value(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Union[int, float]]:
        """Get the latest value of a metric."""
        full_name = self._get_full_name(name)

        with self._lock:
            metric = self._metrics.get(full_name)
            if not metric or not metric.values:
                return None

            # If labels specified, find matching value
            if labels:
                for value in reversed(metric.values):
                    if all(value.labels.get(k) == v for k, v in labels.items()):
                        return value.value
                return None

            return metric.values[-1].value

    def get_histogram_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a histogram metric."""
        full_name = self._get_full_name(name)

        with self._lock:
            metric = self._metrics.get(full_name)
            if not metric or metric.type != MetricType.HISTOGRAM:
                return {}

            values = [v.value for v in metric.values]
            if not values:
                return {}

            sorted_values = sorted(values)
            count = len(values)

            return {
                "count": count,
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / count,
                "median": sorted_values[count // 2],
                "p95": sorted_values[int(count * 0.95)] if count > 1 else sorted_values[0],
                "p99": sorted_values[int(count * 0.99)] if count > 1 else sorted_values[0],
                "bucket_counts": dict(metric.bucket_counts),
            }

    def reset(self, name: Optional[str] = None) -> None:
        """Reset a metric or all metrics."""
        with self._lock:
            if name:
                full_name = self._get_full_name(name)
                if full_name in self._metrics:
                    metric = self._metrics[full_name]
                    metric.values = []
                    if metric.bucket_counts:
                        metric.bucket_counts = {k: 0 for k in metric.bucket_counts}
                    metric._events = []
            else:
                for metric in self._metrics.values():
                    metric.values = []
                    if metric.bucket_counts:
                        metric.bucket_counts = {k: 0 for k in metric.bucket_counts}
                    metric._events = []


class MetricsExporter:
    """
    Exports metrics in various formats.

    Supports:
    - JSON export
    - Prometheus text format
    - CSV export
    """

    def __init__(self, collector: MetricsCollector):
        """Initialize with a metrics collector."""
        self.collector = collector

    def to_json(self, pretty: bool = True) -> str:
        """Export metrics as JSON."""
        metrics = self.collector.get_all_metrics()

        data = {
            "timestamp": datetime.now().isoformat(),
            "option": self.collector.option,
            "metrics": {},
        }

        for name, metric in metrics.items():
            metric_data = metric.to_dict()

            # Add computed statistics for histograms
            if metric.type == MetricType.HISTOGRAM:
                metric_data["stats"] = self.collector.get_histogram_stats(
                    name.replace(f"{self.collector.prefix}_", "")
                )

            # Add rate for rate metrics
            if metric.type == MetricType.RATE:
                metric_data["current_rate"] = self.collector.get_rate(
                    name.replace(f"{self.collector.prefix}_", "")
                )

            data["metrics"][name] = metric_data

        if pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data)

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        metrics = self.collector.get_all_metrics()

        for name, metric in metrics.items():
            # Add help text
            if metric.description:
                lines.append(f"# HELP {name} {metric.description}")

            # Add type
            prom_type = {
                MetricType.COUNTER: "counter",
                MetricType.GAUGE: "gauge",
                MetricType.HISTOGRAM: "histogram",
                MetricType.TIMER: "histogram",
                MetricType.RATE: "gauge",
            }.get(metric.type, "gauge")
            lines.append(f"# TYPE {name} {prom_type}")

            if metric.type == MetricType.HISTOGRAM:
                # Export histogram buckets
                stats = self.collector.get_histogram_stats(
                    name.replace(f"{self.collector.prefix}_", "")
                )
                for bucket, count in sorted(metric.bucket_counts.items()):
                    if bucket == float("inf"):
                        lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
                    else:
                        lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')

                lines.append(f'{name}_sum {stats.get("sum", 0)}')
                lines.append(f'{name}_count {stats.get("count", 0)}')

            elif metric.type == MetricType.RATE:
                rate = self.collector.get_rate(
                    name.replace(f"{self.collector.prefix}_", "")
                )
                lines.append(f'{name} {rate}')

            else:
                # Export latest value
                if metric.values:
                    latest = metric.values[-1]
                    labels = ",".join(f'{k}="{v}"' for k, v in latest.labels.items())
                    if labels:
                        lines.append(f'{name}{{{labels}}} {latest.value}')
                    else:
                        lines.append(f'{name} {latest.value}')

            lines.append("")

        return "\n".join(lines)

    def to_csv(self, include_history: bool = False) -> str:
        """Export metrics as CSV."""
        lines = ["timestamp,metric,type,value,labels"]
        metrics = self.collector.get_all_metrics()

        for name, metric in metrics.items():
            if include_history:
                for value in metric.values:
                    labels_str = json.dumps(value.labels) if value.labels else ""
                    lines.append(
                        f'{value.timestamp.isoformat()},{name},{metric.type.value},'
                        f'{value.value},"{labels_str}"'
                    )
            elif metric.values:
                latest = metric.values[-1]
                labels_str = json.dumps(latest.labels) if latest.labels else ""
                lines.append(
                    f'{latest.timestamp.isoformat()},{name},{metric.type.value},'
                    f'{latest.value},"{labels_str}"'
                )

        return "\n".join(lines)

    def save_to_file(
        self,
        filepath: str,
        format: str = "json",
        **kwargs,
    ) -> None:
        """Save metrics to a file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            content = self.to_json(**kwargs)
        elif format == "prometheus":
            content = self.to_prometheus()
        elif format == "csv":
            content = self.to_csv(**kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")

        path.write_text(content)


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None
_global_lock = threading.Lock()


def get_collector(option: Optional[str] = None) -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _global_collector
    with _global_lock:
        if _global_collector is None:
            _global_collector = MetricsCollector(option=option)
        return _global_collector


def reset_collector() -> None:
    """Reset the global metrics collector."""
    global _global_collector
    with _global_lock:
        _global_collector = None


# Convenience functions

def increment(name: str, value: int = 1, **labels) -> None:
    """Increment a counter metric."""
    get_collector().increment(name, value, labels or None)


def set_gauge(name: str, value: Union[int, float], **labels) -> None:
    """Set a gauge metric."""
    get_collector().set_gauge(name, value, labels or None)


def observe(name: str, value: Union[int, float], **labels) -> None:
    """Observe a value for a histogram."""
    get_collector().observe(name, value, labels or None)


def record_event(name: str) -> None:
    """Record an event for rate calculation."""
    get_collector().record_event(name)
