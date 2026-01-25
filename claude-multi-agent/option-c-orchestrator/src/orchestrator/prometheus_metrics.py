"""
Prometheus metrics exporter for Option C orchestrator (OBS-001).

Provides native prometheus_client integration with:
- Task metrics (total, in_progress, duration)
- Agent metrics (count, status)
- HTTP /metrics endpoint
- Optional push gateway support

Usage:
    from orchestrator.prometheus_metrics import (
        PrometheusMetrics,
        start_metrics_server,
    )

    # Initialize metrics
    metrics = PrometheusMetrics()

    # Record events
    metrics.task_created("task-001", priority=1)
    metrics.task_claimed("task-001", "worker-1")
    metrics.task_completed("task-001", duration_seconds=45.2)

    # Start HTTP server for scraping
    start_metrics_server(port=9090)
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, Iterator, List, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# Check if prometheus_client is available
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,
        Info,
        generate_latest,
        push_to_gateway,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed. Install with: pip install prometheus-client"
    )


@dataclass
class MetricsConfig:
    """Configuration for Prometheus metrics."""
    namespace: str = "orchestrator"
    subsystem: str = "tasks"
    enable_default_metrics: bool = True
    # Histogram buckets for task duration (in seconds)
    duration_buckets: tuple = (1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)
    # Push gateway configuration
    push_gateway_url: Optional[str] = None
    push_gateway_job: str = "orchestrator"
    push_interval_seconds: float = 15.0


class PrometheusMetrics:
    """
    Prometheus metrics collector for the orchestrator.

    Exports the following metrics:
    - orchestrator_tasks_total: Counter of all tasks by status
    - orchestrator_tasks_in_progress: Gauge of currently in-progress tasks
    - orchestrator_task_duration_seconds: Histogram of task execution time
    - orchestrator_task_queue_depth: Gauge of tasks waiting to be claimed
    - orchestrator_agent_count: Gauge of registered agents by status
    - orchestrator_agent_heartbeat_timestamp: Gauge of last heartbeat per agent
    """

    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        registry: Optional["CollectorRegistry"] = None,
    ):
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client is required. Install with: pip install prometheus-client"
            )

        self.config = config or MetricsConfig()
        # Use a new registry if not provided to avoid duplicate metric errors
        self.registry = registry if registry is not None else CollectorRegistry()

        # Initialize metrics
        self._init_metrics()

        # Track task start times for duration calculation
        self._task_start_times: Dict[str, float] = {}

        # Push gateway thread
        self._push_thread: Optional[threading.Thread] = None
        self._push_stop_event = threading.Event()

    def _init_metrics(self) -> None:
        """Initialize all Prometheus metrics."""
        ns = self.config.namespace

        # Helper to create or reuse metrics (handles duplicate registration)
        def _counter(name: str, desc: str, labels: List[str]) -> Counter:
            try:
                return Counter(name, desc, labels, registry=self.registry)
            except ValueError:
                # Metric already exists, try to get it from registry
                for collector in self.registry._names_to_collectors.values():
                    if hasattr(collector, '_name') and collector._name == name:
                        return collector
                raise

        def _gauge(name: str, desc: str, labels: List[str] = None) -> Gauge:
            try:
                return Gauge(name, desc, labels or [], registry=self.registry)
            except ValueError:
                for collector in self.registry._names_to_collectors.values():
                    if hasattr(collector, '_name') and collector._name == name:
                        return collector
                raise

        def _histogram(name: str, desc: str, labels: List[str], buckets: tuple) -> Histogram:
            try:
                return Histogram(name, desc, labels, buckets=buckets, registry=self.registry)
            except ValueError:
                for collector in self.registry._names_to_collectors.values():
                    if hasattr(collector, '_name') and collector._name == name:
                        return collector
                raise

        def _info(name: str, desc: str) -> Info:
            try:
                return Info(name, desc, registry=self.registry)
            except ValueError:
                for collector in self.registry._names_to_collectors.values():
                    if hasattr(collector, '_name') and collector._name == name:
                        return collector
                raise

        # Task counters
        self.tasks_total = _counter(
            f"{ns}_tasks_total",
            "Total number of tasks by final status",
            ["status", "priority"],
        )

        self.task_creations_total = _counter(
            f"{ns}_task_creations_total",
            "Total number of tasks created",
            ["priority"],
        )

        # Task gauges
        self.tasks_in_progress = _gauge(
            f"{ns}_tasks_in_progress",
            "Number of tasks currently in progress",
            ["agent_id"],
        )

        self.task_queue_depth = _gauge(
            f"{ns}_task_queue_depth",
            "Number of tasks waiting to be claimed (available status)",
            ["priority"],
        )

        self.tasks_by_status = _gauge(
            f"{ns}_tasks_by_status",
            "Current task count by status",
            ["status"],
        )

        # Task timing
        self.task_duration_seconds = _histogram(
            f"{ns}_task_duration_seconds",
            "Task execution duration in seconds",
            ["priority", "status"],
            self.config.duration_buckets,
        )

        self.task_wait_time_seconds = _histogram(
            f"{ns}_task_wait_time_seconds",
            "Time tasks spend waiting in queue before claim",
            ["priority"],
            (1, 5, 15, 30, 60, 120, 300, 600),
        )

        # Agent metrics
        self.agent_count = _gauge(
            f"{ns}_agent_count",
            "Number of registered agents by status",
            ["status", "role"],
        )

        self.agent_heartbeat_timestamp = _gauge(
            f"{ns}_agent_heartbeat_timestamp_seconds",
            "Unix timestamp of last agent heartbeat",
            ["agent_id"],
        )

        self.agent_tasks_completed = _counter(
            f"{ns}_agent_tasks_completed_total",
            "Total tasks completed per agent",
            ["agent_id"],
        )

        self.agent_tasks_failed = _counter(
            f"{ns}_agent_tasks_failed_total",
            "Total tasks failed per agent",
            ["agent_id"],
        )

        # Throughput metrics
        self.task_throughput = _gauge(
            f"{ns}_task_throughput_per_minute",
            "Tasks completed per minute (rolling average)",
        )

        # Orchestrator info
        self.orchestrator_info = _info(
            f"{ns}_info",
            "Orchestrator version and configuration",
        )

        # Error metrics
        self.errors_total = _counter(
            f"{ns}_errors_total",
            "Total number of errors by type",
            ["error_type"],
        )

    # -------------------------------------------------------------------------
    # Task metrics methods
    # -------------------------------------------------------------------------

    def task_created(self, task_id: str, priority: int = 5) -> None:
        """Record a task creation event."""
        self.task_creations_total.labels(priority=str(priority)).inc()
        self._task_start_times[task_id] = time.time()
        logger.debug("Metric: task_created task_id=%s priority=%d", task_id, priority)

    def task_claimed(self, task_id: str, agent_id: str, wait_time: Optional[float] = None) -> None:
        """Record a task claim event."""
        self.tasks_in_progress.labels(agent_id=agent_id).inc()

        # Record wait time if provided
        if wait_time is not None:
            # Get priority from task_id pattern if possible, default to 5
            priority = "5"
            self.task_wait_time_seconds.labels(priority=priority).observe(wait_time)

        logger.debug("Metric: task_claimed task_id=%s agent_id=%s", task_id, agent_id)

    def task_started(self, task_id: str, agent_id: str) -> None:
        """Record a task start event (transition to in_progress)."""
        # Update start time if not already set
        if task_id not in self._task_start_times:
            self._task_start_times[task_id] = time.time()
        logger.debug("Metric: task_started task_id=%s agent_id=%s", task_id, agent_id)

    def task_completed(
        self,
        task_id: str,
        agent_id: str,
        priority: int = 5,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Record a task completion event."""
        # Calculate duration if not provided
        if duration_seconds is None:
            start_time = self._task_start_times.pop(task_id, None)
            if start_time:
                duration_seconds = time.time() - start_time

        # Record metrics
        self.tasks_total.labels(status="done", priority=str(priority)).inc()
        self.tasks_in_progress.labels(agent_id=agent_id).dec()
        self.agent_tasks_completed.labels(agent_id=agent_id).inc()

        if duration_seconds is not None:
            self.task_duration_seconds.labels(
                priority=str(priority), status="done"
            ).observe(duration_seconds)

        logger.debug(
            "Metric: task_completed task_id=%s agent_id=%s duration=%.2f",
            task_id, agent_id, duration_seconds or 0
        )

    def task_failed(
        self,
        task_id: str,
        agent_id: str,
        priority: int = 5,
        error_type: str = "unknown",
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Record a task failure event."""
        # Calculate duration if not provided
        if duration_seconds is None:
            start_time = self._task_start_times.pop(task_id, None)
            if start_time:
                duration_seconds = time.time() - start_time

        # Record metrics
        self.tasks_total.labels(status="failed", priority=str(priority)).inc()
        self.tasks_in_progress.labels(agent_id=agent_id).dec()
        self.agent_tasks_failed.labels(agent_id=agent_id).inc()
        self.errors_total.labels(error_type=error_type).inc()

        if duration_seconds is not None:
            self.task_duration_seconds.labels(
                priority=str(priority), status="failed"
            ).observe(duration_seconds)

        logger.debug(
            "Metric: task_failed task_id=%s agent_id=%s error_type=%s",
            task_id, agent_id, error_type
        )

    def update_queue_depth(self, available_by_priority: Dict[int, int]) -> None:
        """Update queue depth gauges by priority."""
        for priority, count in available_by_priority.items():
            self.task_queue_depth.labels(priority=str(priority)).set(count)

    def update_tasks_by_status(self, status_counts: Dict[str, int]) -> None:
        """Update task counts by status."""
        for status, count in status_counts.items():
            self.tasks_by_status.labels(status=status).set(count)

    # -------------------------------------------------------------------------
    # Agent metrics methods
    # -------------------------------------------------------------------------

    def agent_registered(self, agent_id: str, role: str = "worker") -> None:
        """Record agent registration."""
        self.agent_count.labels(status="active", role=role).inc()
        self.agent_heartbeat_timestamp.labels(agent_id=agent_id).set(time.time())
        logger.debug("Metric: agent_registered agent_id=%s role=%s", agent_id, role)

    def agent_deregistered(self, agent_id: str, role: str = "worker") -> None:
        """Record agent deregistration."""
        self.agent_count.labels(status="active", role=role).dec()
        self.agent_count.labels(status="inactive", role=role).inc()
        logger.debug("Metric: agent_deregistered agent_id=%s", agent_id)

    def agent_heartbeat(self, agent_id: str) -> None:
        """Record agent heartbeat."""
        self.agent_heartbeat_timestamp.labels(agent_id=agent_id).set(time.time())

    def update_agent_counts(self, counts: Dict[str, Dict[str, int]]) -> None:
        """
        Update agent count gauges.

        Args:
            counts: Dict of {status: {role: count}}
        """
        for status, role_counts in counts.items():
            for role, count in role_counts.items():
                self.agent_count.labels(status=status, role=role).set(count)

    # -------------------------------------------------------------------------
    # Orchestrator info
    # -------------------------------------------------------------------------

    def set_orchestrator_info(
        self,
        version: str,
        mode: str = "async",
        max_workers: int = 3,
        **extra: str,
    ) -> None:
        """Set orchestrator info labels."""
        self.orchestrator_info.info({
            "version": version,
            "mode": mode,
            "max_workers": str(max_workers),
            **extra,
        })

    def update_throughput(self, tasks_per_minute: float) -> None:
        """Update throughput gauge."""
        self.task_throughput.set(tasks_per_minute)

    def record_error(self, error_type: str) -> None:
        """Record an error event."""
        self.errors_total.labels(error_type=error_type).inc()

    # -------------------------------------------------------------------------
    # Context manager for timing
    # -------------------------------------------------------------------------

    @contextmanager
    def time_task(
        self,
        task_id: str,
        agent_id: str,
        priority: int = 5,
    ) -> Iterator[None]:
        """
        Context manager for timing task execution.

        Usage:
            with metrics.time_task("task-001", "worker-1", priority=2):
                # do work
                pass
        """
        start_time = time.time()
        self._task_start_times[task_id] = start_time
        self.tasks_in_progress.labels(agent_id=agent_id).inc()

        try:
            yield
            duration = time.time() - start_time
            self.task_completed(task_id, agent_id, priority, duration)
        except Exception:
            duration = time.time() - start_time
            self.task_failed(task_id, agent_id, priority, "exception", duration)
            raise

    # -------------------------------------------------------------------------
    # Push gateway support
    # -------------------------------------------------------------------------

    def push_to_gateway(self, gateway_url: Optional[str] = None, job: Optional[str] = None) -> None:
        """Push metrics to a Prometheus push gateway."""
        url = gateway_url or self.config.push_gateway_url
        job_name = job or self.config.push_gateway_job

        if not url:
            logger.warning("No push gateway URL configured")
            return

        try:
            push_to_gateway(url, job=job_name, registry=self.registry)
            logger.debug("Pushed metrics to gateway %s", url)
        except Exception as e:
            logger.error("Failed to push metrics to gateway: %s", e)
            self.record_error("push_gateway_failure")

    def start_push_gateway_thread(
        self,
        gateway_url: Optional[str] = None,
        interval: Optional[float] = None,
    ) -> None:
        """Start background thread to periodically push metrics."""
        if self._push_thread and self._push_thread.is_alive():
            logger.warning("Push gateway thread already running")
            return

        url = gateway_url or self.config.push_gateway_url
        push_interval = interval or self.config.push_interval_seconds

        if not url:
            logger.warning("No push gateway URL configured")
            return

        self._push_stop_event.clear()

        def push_loop():
            while not self._push_stop_event.wait(push_interval):
                self.push_to_gateway(url)

        self._push_thread = threading.Thread(target=push_loop, daemon=True)
        self._push_thread.start()
        logger.info("Started push gateway thread (interval=%.1fs)", push_interval)

    def stop_push_gateway_thread(self) -> None:
        """Stop the push gateway background thread."""
        if self._push_thread:
            self._push_stop_event.set()
            self._push_thread.join(timeout=5.0)
            self._push_thread = None
            logger.info("Stopped push gateway thread")

    # -------------------------------------------------------------------------
    # Export metrics
    # -------------------------------------------------------------------------

    def generate_metrics(self) -> bytes:
        """Generate metrics in Prometheus text format."""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get the content type for metrics response."""
        return CONTENT_TYPE_LATEST


# -----------------------------------------------------------------------------
# HTTP Server for /metrics endpoint
# -----------------------------------------------------------------------------

class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for /metrics endpoint."""

    metrics: Optional[PrometheusMetrics] = None

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)

        if parsed.path == "/metrics":
            self._handle_metrics()
        elif parsed.path == "/health":
            self._handle_health()
        else:
            self.send_error(404, "Not Found")

    def _handle_metrics(self) -> None:
        """Handle /metrics endpoint."""
        if self.metrics is None:
            self.send_error(500, "Metrics not initialized")
            return

        try:
            output = self.metrics.generate_metrics()
            self.send_response(200)
            self.send_header("Content-Type", self.metrics.get_content_type())
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)
        except Exception as e:
            logger.error("Error generating metrics: %s", e)
            self.send_error(500, str(e))

    def _handle_health(self) -> None:
        """Handle /health endpoint."""
        import json
        response = json.dumps({
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
        }).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""
        pass


class MetricsServer:
    """HTTP server for Prometheus metrics scraping."""

    def __init__(
        self,
        metrics: PrometheusMetrics,
        host: str = "0.0.0.0",
        port: int = 9090,
    ):
        self.metrics = metrics
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the metrics server in a background thread."""
        if self._server:
            logger.warning("Metrics server already running")
            return

        # Set metrics on handler class
        MetricsHandler.metrics = self.metrics

        self._server = HTTPServer((self.host, self.port), MetricsHandler)

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        logger.info("Metrics server started on http://%s:%d/metrics", self.host, self.port)

    def stop(self) -> None:
        """Stop the metrics server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None
            logger.info("Metrics server stopped")

    def __enter__(self) -> "MetricsServer":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def start_metrics_server(
    port: int = 9090,
    host: str = "0.0.0.0",
    metrics: Optional[PrometheusMetrics] = None,
) -> MetricsServer:
    """
    Start a Prometheus metrics HTTP server.

    Args:
        port: Port to listen on (default 9090)
        host: Host to bind to (default 0.0.0.0)
        metrics: PrometheusMetrics instance (creates new if not provided)

    Returns:
        MetricsServer instance

    Example:
        server = start_metrics_server(port=9090)
        # ... do work ...
        server.stop()
    """
    if metrics is None:
        metrics = PrometheusMetrics()

    server = MetricsServer(metrics, host, port)
    server.start()
    return server


# Singleton instance for easy access
_metrics_instance: Optional[PrometheusMetrics] = None


def get_metrics() -> PrometheusMetrics:
    """Get or create the global PrometheusMetrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client is required. Install with: pip install prometheus-client"
            )
        _metrics_instance = PrometheusMetrics()
    return _metrics_instance


def initialize_metrics(config: Optional[MetricsConfig] = None) -> PrometheusMetrics:
    """Initialize the global PrometheusMetrics instance with config."""
    global _metrics_instance
    if not PROMETHEUS_AVAILABLE:
        raise ImportError(
            "prometheus_client is required. Install with: pip install prometheus-client"
        )
    _metrics_instance = PrometheusMetrics(config)
    return _metrics_instance
