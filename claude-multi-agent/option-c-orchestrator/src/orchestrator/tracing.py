"""
OpenTelemetry tracing integration for Option C orchestrator (OBS-003).

Provides distributed tracing for task lifecycle events:
- Task creation, claim, execution, completion
- Agent registration and heartbeats
- Cross-service trace context propagation

Usage:
    from orchestrator.tracing import (
        OrchestratorTracing,
        initialize_tracing,
        get_tracer,
    )

    # Initialize tracing
    tracing = initialize_tracing(
        service_name="orchestrator",
        otlp_endpoint="http://localhost:4318/v1/traces",
    )

    # Trace task lifecycle
    with tracing.trace_task_lifecycle("task-001", "worker-1") as span:
        # do work
        span.set_attribute("custom.attr", "value")
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Check if OpenTelemetry is available
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, Span
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.context import Context
    from opentelemetry.propagate import set_global_textmap, inject, extract

    # Try to import OTLP exporter
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        OTLP_HTTP_AVAILABLE = True
    except ImportError:
        OTLP_HTTP_AVAILABLE = False

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGrpcSpanExporter
        OTLP_GRPC_AVAILABLE = True
    except ImportError:
        OTLP_GRPC_AVAILABLE = False

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning(
        "opentelemetry not installed. Install with: "
        "pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
    )


T = TypeVar("T")


@dataclass
class TracingConfig:
    """Configuration for OpenTelemetry tracing."""
    service_name: str = "orchestrator"
    service_version: str = "2.1.0"
    environment: str = "development"

    # Exporter configuration
    exporter_type: str = "otlp"  # otlp, otlp-grpc, console, none
    otlp_endpoint: Optional[str] = None  # Default: http://localhost:4318/v1/traces
    otlp_headers: Dict[str, str] = field(default_factory=dict)

    # Processing configuration
    batch_export: bool = True
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout_millis: int = 30000

    # Sampling (1.0 = 100%)
    sampling_ratio: float = 1.0

    # Debug mode
    debug: bool = False

    @classmethod
    def from_env(cls) -> "TracingConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name=os.environ.get("OTEL_SERVICE_NAME", "orchestrator"),
            service_version=os.environ.get("OTEL_SERVICE_VERSION", "2.1.0"),
            environment=os.environ.get("OTEL_ENVIRONMENT", "development"),
            exporter_type=os.environ.get("OTEL_EXPORTER_TYPE", "otlp"),
            otlp_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            batch_export=os.environ.get("OTEL_BATCH_EXPORT", "true").lower() == "true",
            sampling_ratio=float(os.environ.get("OTEL_SAMPLING_RATIO", "1.0")),
            debug=os.environ.get("OTEL_DEBUG", "false").lower() == "true",
        )


class OrchestratorTracing:
    """
    OpenTelemetry tracing for the orchestrator.

    Provides methods to trace:
    - Task lifecycle (create, claim, start, complete, fail)
    - Agent operations (register, heartbeat, deregister)
    - Orchestration operations (coordination start/stop)
    """

    def __init__(self, config: Optional[TracingConfig] = None):
        if not OTEL_AVAILABLE:
            raise ImportError(
                "opentelemetry is required. Install with: "
                "pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
            )

        self.config = config or TracingConfig()
        self._provider: Optional[TracerProvider] = None
        self._tracer: Optional[trace.Tracer] = None
        self._propagator = TraceContextTextMapPropagator()
        self._initialized = False

        # Track active spans for correlation
        self._active_task_spans: Dict[str, Span] = {}

    def initialize(self) -> None:
        """Initialize the tracing provider and exporter."""
        if self._initialized:
            logger.warning("Tracing already initialized")
            return

        # Create resource
        resource = Resource.create({
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            "deployment.environment": self.config.environment,
        })

        # Create provider
        self._provider = TracerProvider(resource=resource)

        # Create and add exporter
        exporter = self._create_exporter()
        if exporter:
            if self.config.batch_export:
                processor = BatchSpanProcessor(
                    exporter,
                    max_queue_size=self.config.max_queue_size,
                    max_export_batch_size=self.config.max_export_batch_size,
                    export_timeout_millis=self.config.export_timeout_millis,
                )
            else:
                processor = SimpleSpanProcessor(exporter)
            self._provider.add_span_processor(processor)

        # Set global provider
        trace.set_tracer_provider(self._provider)

        # Set global propagator
        set_global_textmap(self._propagator)

        # Get tracer
        self._tracer = trace.get_tracer(
            self.config.service_name,
            self.config.service_version,
        )

        self._initialized = True

        if self.config.debug:
            logger.info(
                "Tracing initialized: service=%s version=%s exporter=%s",
                self.config.service_name,
                self.config.service_version,
                self.config.exporter_type,
            )

    def _create_exporter(self) -> Any:
        """Create the span exporter based on configuration."""
        exporter_type = self.config.exporter_type.lower()

        if exporter_type == "none":
            return None

        if exporter_type == "console":
            return ConsoleSpanExporter()

        if exporter_type == "otlp":
            if not OTLP_HTTP_AVAILABLE:
                logger.warning(
                    "OTLP HTTP exporter not available, falling back to console"
                )
                return ConsoleSpanExporter()

            endpoint = self.config.otlp_endpoint or "http://localhost:4318/v1/traces"
            return OTLPSpanExporter(
                endpoint=endpoint,
                headers=self.config.otlp_headers or None,
            )

        if exporter_type == "otlp-grpc":
            if not OTLP_GRPC_AVAILABLE:
                logger.warning(
                    "OTLP gRPC exporter not available, falling back to console"
                )
                return ConsoleSpanExporter()

            endpoint = self.config.otlp_endpoint or "http://localhost:4317"
            return OTLPGrpcSpanExporter(
                endpoint=endpoint,
            )

        logger.warning("Unknown exporter type: %s, using console", exporter_type)
        return ConsoleSpanExporter()

    def shutdown(self) -> None:
        """Shutdown the tracing provider."""
        if self._provider:
            self._provider.shutdown()
            self._provider = None
            self._tracer = None
            self._initialized = False
            logger.info("Tracing shutdown complete")

    @property
    def tracer(self) -> trace.Tracer:
        """Get the tracer instance."""
        if not self._tracer:
            raise RuntimeError("Tracing not initialized. Call initialize() first.")
        return self._tracer

    # -------------------------------------------------------------------------
    # Task tracing
    # -------------------------------------------------------------------------

    def trace_task_create(
        self,
        task_id: str,
        description: str,
        priority: int,
        dependencies: Optional[List[str]] = None,
    ) -> Span:
        """Start a span for task creation."""
        span = self.tracer.start_span(
            "task.create",
            kind=SpanKind.PRODUCER,
            attributes={
                "task.id": task_id,
                "task.description": description[:200],
                "task.priority": priority,
                "task.status": "available",
                "task.dependency_count": len(dependencies or []),
            },
        )
        self._active_task_spans[task_id] = span
        return span

    def trace_task_claim(
        self,
        task_id: str,
        agent_id: str,
        wait_time_seconds: Optional[float] = None,
    ) -> Span:
        """Start a span for task claim."""
        parent_span = self._active_task_spans.get(task_id)
        context = trace.set_span_in_context(parent_span) if parent_span else None

        span = self.tracer.start_span(
            "task.claim",
            kind=SpanKind.CONSUMER,
            context=context,
            attributes={
                "task.id": task_id,
                "agent.id": agent_id,
                "task.status": "claimed",
            },
        )

        if wait_time_seconds is not None:
            span.set_attribute("task.wait_time_seconds", wait_time_seconds)

        return span

    def trace_task_start(
        self,
        task_id: str,
        agent_id: str,
    ) -> Span:
        """Start a span for task execution start."""
        parent_span = self._active_task_spans.get(task_id)
        context = trace.set_span_in_context(parent_span) if parent_span else None

        span = self.tracer.start_span(
            "task.execute",
            kind=SpanKind.INTERNAL,
            context=context,
            attributes={
                "task.id": task_id,
                "agent.id": agent_id,
                "task.status": "in_progress",
            },
        )

        # Store as the active span for this task
        self._active_task_spans[task_id] = span
        return span

    def trace_task_complete(
        self,
        task_id: str,
        agent_id: str,
        duration_seconds: Optional[float] = None,
    ) -> Span:
        """Start a span for task completion."""
        parent_span = self._active_task_spans.get(task_id)
        context = trace.set_span_in_context(parent_span) if parent_span else None

        span = self.tracer.start_span(
            "task.complete",
            kind=SpanKind.INTERNAL,
            context=context,
            attributes={
                "task.id": task_id,
                "agent.id": agent_id,
                "task.status": "done",
                "task.success": True,
            },
        )

        if duration_seconds is not None:
            span.set_attribute("task.duration_seconds", duration_seconds)

        # End the span immediately
        span.end()

        # Clean up active span
        self._active_task_spans.pop(task_id, None)

        return span

    def trace_task_fail(
        self,
        task_id: str,
        agent_id: str,
        error: str,
        duration_seconds: Optional[float] = None,
    ) -> Span:
        """Start a span for task failure."""
        parent_span = self._active_task_spans.get(task_id)
        context = trace.set_span_in_context(parent_span) if parent_span else None

        span = self.tracer.start_span(
            "task.fail",
            kind=SpanKind.INTERNAL,
            context=context,
            attributes={
                "task.id": task_id,
                "agent.id": agent_id,
                "task.status": "failed",
                "task.success": False,
                "task.error": error[:500],
            },
        )

        if duration_seconds is not None:
            span.set_attribute("task.duration_seconds", duration_seconds)

        # Set error status
        span.set_status(Status(StatusCode.ERROR, error[:100]))

        # End the span immediately
        span.end()

        # Clean up active span
        self._active_task_spans.pop(task_id, None)

        return span

    @contextmanager
    def trace_task_lifecycle(
        self,
        task_id: str,
        agent_id: str,
        priority: int = 5,
        description: str = "",
    ) -> Iterator[Span]:
        """
        Context manager for tracing complete task lifecycle.

        Usage:
            with tracing.trace_task_lifecycle("task-001", "worker-1") as span:
                # do work
                span.set_attribute("custom", "value")
        """
        span = self.tracer.start_span(
            "task.lifecycle",
            kind=SpanKind.INTERNAL,
            attributes={
                "task.id": task_id,
                "agent.id": agent_id,
                "task.priority": priority,
                "task.description": description[:200] if description else "",
            },
        )

        self._active_task_spans[task_id] = span
        start_time = datetime.now()

        try:
            yield span
            duration = (datetime.now() - start_time).total_seconds()
            span.set_attribute("task.duration_seconds", duration)
            span.set_attribute("task.status", "done")
            span.set_attribute("task.success", True)
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            span.set_attribute("task.duration_seconds", duration)
            span.set_attribute("task.status", "failed")
            span.set_attribute("task.success", False)
            span.set_attribute("task.error", str(e)[:500])
            span.set_status(Status(StatusCode.ERROR, str(e)[:100]))
            span.record_exception(e)
            raise
        finally:
            span.end()
            self._active_task_spans.pop(task_id, None)

    # -------------------------------------------------------------------------
    # Agent tracing
    # -------------------------------------------------------------------------

    def trace_agent_register(
        self,
        agent_id: str,
        role: str = "worker",
        capabilities: Optional[List[str]] = None,
    ) -> Span:
        """Trace agent registration."""
        span = self.tracer.start_span(
            "agent.register",
            kind=SpanKind.INTERNAL,
            attributes={
                "agent.id": agent_id,
                "agent.role": role,
                "agent.capabilities": ",".join(capabilities or []),
            },
        )
        span.end()
        return span

    def trace_agent_heartbeat(self, agent_id: str) -> Span:
        """Trace agent heartbeat."""
        span = self.tracer.start_span(
            "agent.heartbeat",
            kind=SpanKind.INTERNAL,
            attributes={
                "agent.id": agent_id,
            },
        )
        span.end()
        return span

    def trace_agent_deregister(self, agent_id: str) -> Span:
        """Trace agent deregistration."""
        span = self.tracer.start_span(
            "agent.deregister",
            kind=SpanKind.INTERNAL,
            attributes={
                "agent.id": agent_id,
            },
        )
        span.end()
        return span

    # -------------------------------------------------------------------------
    # Orchestration tracing
    # -------------------------------------------------------------------------

    @contextmanager
    def trace_coordination(
        self,
        goal: str,
        max_workers: int = 3,
    ) -> Iterator[Span]:
        """
        Context manager for tracing a coordination session.

        Usage:
            with tracing.trace_coordination("Build feature X", max_workers=3) as span:
                # run orchestration
                span.set_attribute("tasks.total", 10)
        """
        span = self.tracer.start_span(
            "coordination.run",
            kind=SpanKind.SERVER,
            attributes={
                "coordination.goal": goal[:500],
                "coordination.max_workers": max_workers,
            },
        )

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)[:100]))
            span.record_exception(e)
            raise
        finally:
            span.end()

    # -------------------------------------------------------------------------
    # Context propagation
    # -------------------------------------------------------------------------

    def inject_context(self, carrier: Dict[str, str]) -> None:
        """Inject trace context into carrier (e.g., HTTP headers)."""
        inject(carrier)

    def extract_context(self, carrier: Dict[str, str]) -> Context:
        """Extract trace context from carrier."""
        return extract(carrier)

    def get_trace_id(self, span: Optional[Span] = None) -> Optional[str]:
        """Get the current trace ID as a hex string."""
        if span:
            return format(span.get_span_context().trace_id, "032x")

        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().is_valid:
            return format(current_span.get_span_context().trace_id, "032x")

        return None

    def get_span_id(self, span: Optional[Span] = None) -> Optional[str]:
        """Get the current span ID as a hex string."""
        if span:
            return format(span.get_span_context().span_id, "016x")

        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().is_valid:
            return format(current_span.get_span_context().span_id, "016x")

        return None


# -----------------------------------------------------------------------------
# Singleton and module-level functions
# -----------------------------------------------------------------------------

_tracing_instance: Optional[OrchestratorTracing] = None


def initialize_tracing(
    service_name: str = "orchestrator",
    service_version: str = "2.1.0",
    otlp_endpoint: Optional[str] = None,
    exporter_type: str = "otlp",
    config: Optional[TracingConfig] = None,
    **kwargs: Any,
) -> OrchestratorTracing:
    """
    Initialize the global tracing instance.

    Args:
        service_name: Name of the service
        service_version: Version of the service
        otlp_endpoint: OTLP exporter endpoint
        exporter_type: Type of exporter (otlp, console, none)
        config: Full TracingConfig object (overrides other params)
        **kwargs: Additional config parameters

    Returns:
        Initialized OrchestratorTracing instance
    """
    global _tracing_instance

    if not OTEL_AVAILABLE:
        raise ImportError(
            "opentelemetry is required. Install with: "
            "pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
        )

    if config is None:
        config = TracingConfig(
            service_name=service_name,
            service_version=service_version,
            otlp_endpoint=otlp_endpoint,
            exporter_type=exporter_type,
            **kwargs,
        )

    _tracing_instance = OrchestratorTracing(config)
    _tracing_instance.initialize()

    return _tracing_instance


def get_tracing() -> OrchestratorTracing:
    """Get the global tracing instance."""
    if _tracing_instance is None:
        raise RuntimeError("Tracing not initialized. Call initialize_tracing() first.")
    return _tracing_instance


def get_tracer() -> trace.Tracer:
    """Get the tracer from the global instance."""
    return get_tracing().tracer


def shutdown_tracing() -> None:
    """Shutdown the global tracing instance."""
    global _tracing_instance
    if _tracing_instance:
        _tracing_instance.shutdown()
        _tracing_instance = None


def is_tracing_available() -> bool:
    """Check if OpenTelemetry is available."""
    return OTEL_AVAILABLE
