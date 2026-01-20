import { NodeSDK } from "@opentelemetry/sdk-node";
import { getNodeAutoInstrumentations } from "@opentelemetry/auto-instrumentations-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { JaegerExporter } from "@opentelemetry/exporter-jaeger";
import { ZipkinExporter } from "@opentelemetry/exporter-zipkin";
import { Resource } from "@opentelemetry/resources";
import { SemanticResourceAttributes } from "@opentelemetry/semantic-conventions";
import {
  SimpleSpanProcessor,
  BatchSpanProcessor,
  ConsoleSpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import {
  trace,
  context,
  SpanKind,
  SpanStatusCode,
  Span,
  Tracer,
} from "@opentelemetry/api";
import { config } from "dotenv";

config();

// Configuration
export interface TracingConfig {
  serviceName: string;
  serviceVersion?: string;
  environment?: string;
  exporterType: "otlp" | "jaeger" | "zipkin" | "console";
  exporterEndpoint?: string;
  batchExport?: boolean;
  debug?: boolean;
}

// Default configuration
const defaultConfig: TracingConfig = {
  serviceName: process.env.OTEL_SERVICE_NAME || "claude-coordinator",
  serviceVersion: process.env.OTEL_SERVICE_VERSION || "1.0.0",
  environment: process.env.OTEL_ENVIRONMENT || "development",
  exporterType: (process.env.OTEL_EXPORTER_TYPE as any) || "otlp",
  exporterEndpoint:
    process.env.OTEL_EXPORTER_ENDPOINT || "http://localhost:4318/v1/traces",
  batchExport: process.env.OTEL_BATCH_EXPORT !== "false",
  debug: process.env.OTEL_DEBUG === "true",
};

// Tracing Manager
export class TracingManager {
  private sdk: NodeSDK | null = null;
  private provider: NodeTracerProvider | null = null;
  private tracer: Tracer | null = null;
  private config: TracingConfig;

  constructor(config: Partial<TracingConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  // Initialize OpenTelemetry
  initialize(): void {
    // Create resource
    const resource = new Resource({
      [SemanticResourceAttributes.SERVICE_NAME]: this.config.serviceName,
      [SemanticResourceAttributes.SERVICE_VERSION]:
        this.config.serviceVersion || "1.0.0",
      [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]:
        this.config.environment || "development",
    });

    // Create exporter based on config
    const exporter = this.createExporter();

    // Create provider
    this.provider = new NodeTracerProvider({
      resource,
    });

    // Add span processor
    if (this.config.batchExport) {
      this.provider.addSpanProcessor(new BatchSpanProcessor(exporter));
    } else {
      this.provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
    }

    // Register provider
    this.provider.register();

    // Get tracer
    this.tracer = trace.getTracer(this.config.serviceName);

    // Initialize SDK with auto-instrumentations
    this.sdk = new NodeSDK({
      resource,
      traceExporter: exporter,
      instrumentations: [getNodeAutoInstrumentations()],
    });

    this.sdk.start();

    if (this.config.debug) {
      console.log("OpenTelemetry initialized with config:", this.config);
    }
  }

  private createExporter(): any {
    switch (this.config.exporterType) {
      case "jaeger":
        return new JaegerExporter({
          endpoint: this.config.exporterEndpoint,
        });

      case "zipkin":
        return new ZipkinExporter({
          url: this.config.exporterEndpoint,
        });

      case "console":
        return new ConsoleSpanExporter();

      case "otlp":
      default:
        return new OTLPTraceExporter({
          url: this.config.exporterEndpoint,
        });
    }
  }

  // Shutdown tracing
  async shutdown(): Promise<void> {
    if (this.sdk) {
      await this.sdk.shutdown();
    }
    if (this.provider) {
      await this.provider.shutdown();
    }
  }

  // Get tracer
  getTracer(): Tracer {
    if (!this.tracer) {
      throw new Error("Tracing not initialized. Call initialize() first.");
    }
    return this.tracer;
  }

  // Start a span
  startSpan(
    name: string,
    options: {
      kind?: SpanKind;
      attributes?: Record<string, string | number | boolean>;
      parent?: Span;
    } = {},
  ): Span {
    const tracer = this.getTracer();
    const ctx = options.parent
      ? trace.setSpan(context.active(), options.parent)
      : context.active();

    return tracer.startSpan(
      name,
      {
        kind: options.kind || SpanKind.INTERNAL,
        attributes: options.attributes,
      },
      ctx,
    );
  }

  // End a span
  endSpan(span: Span, error?: Error): void {
    if (error) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      span.recordException(error);
    } else {
      span.setStatus({ code: SpanStatusCode.OK });
    }
    span.end();
  }

  // Wrap a function with tracing
  trace<T>(
    name: string,
    fn: (span: Span) => Promise<T> | T,
    options: {
      kind?: SpanKind;
      attributes?: Record<string, string | number | boolean>;
    } = {},
  ): Promise<T> {
    const span = this.startSpan(name, options);

    return context.with(trace.setSpan(context.active(), span), async () => {
      try {
        const result = await fn(span);
        this.endSpan(span);
        return result;
      } catch (error) {
        this.endSpan(span, error as Error);
        throw error;
      }
    });
  }
}

// Coordinator Tracing - specific traces for coordinator operations
export class CoordinatorTracing {
  private manager: TracingManager;

  constructor(manager: TracingManager) {
    this.manager = manager;
  }

  // Trace task creation
  traceTaskCreate(taskId: string, description: string, priority: number): Span {
    return this.manager.startSpan("task.create", {
      kind: SpanKind.PRODUCER,
      attributes: {
        "task.id": taskId,
        "task.description": description.substring(0, 100),
        "task.priority": priority,
        "task.status": "available",
      },
    });
  }

  // Trace task claim
  traceTaskClaim(taskId: string, workerId: string, parentSpan?: Span): Span {
    return this.manager.startSpan("task.claim", {
      kind: SpanKind.CONSUMER,
      parent: parentSpan,
      attributes: {
        "task.id": taskId,
        "worker.id": workerId,
        "task.status": "claimed",
      },
    });
  }

  // Trace task execution
  traceTaskExecute(taskId: string, workerId: string, parentSpan?: Span): Span {
    return this.manager.startSpan("task.execute", {
      kind: SpanKind.INTERNAL,
      parent: parentSpan,
      attributes: {
        "task.id": taskId,
        "worker.id": workerId,
        "task.status": "in_progress",
      },
    });
  }

  // Trace task completion
  traceTaskComplete(taskId: string, success: boolean, parentSpan?: Span): Span {
    return this.manager.startSpan("task.complete", {
      kind: SpanKind.INTERNAL,
      parent: parentSpan,
      attributes: {
        "task.id": taskId,
        "task.success": success,
        "task.status": success ? "done" : "failed",
      },
    });
  }

  // Trace worker registration
  traceWorkerRegister(workerId: string, capabilities: string[]): Span {
    return this.manager.startSpan("worker.register", {
      kind: SpanKind.INTERNAL,
      attributes: {
        "worker.id": workerId,
        "worker.capabilities": capabilities.join(","),
        "worker.status": "idle",
      },
    });
  }

  // Trace worker heartbeat
  traceWorkerHeartbeat(workerId: string): Span {
    return this.manager.startSpan("worker.heartbeat", {
      kind: SpanKind.INTERNAL,
      attributes: {
        "worker.id": workerId,
      },
    });
  }

  // Trace discovery
  traceDiscoveryAdd(
    discoveryId: string,
    title: string,
    createdBy: string,
  ): Span {
    return this.manager.startSpan("discovery.add", {
      kind: SpanKind.INTERNAL,
      attributes: {
        "discovery.id": discoveryId,
        "discovery.title": title,
        "discovery.created_by": createdBy,
      },
    });
  }

  // Trace coordination start
  traceCoordinationStart(goal: string): Span {
    return this.manager.startSpan("coordination.start", {
      kind: SpanKind.SERVER,
      attributes: {
        "coordination.goal": goal.substring(0, 200),
      },
    });
  }

  // Trace coordination complete
  traceCoordinationComplete(
    tasksCompleted: number,
    tasksFailed: number,
    parentSpan?: Span,
  ): Span {
    return this.manager.startSpan("coordination.complete", {
      kind: SpanKind.SERVER,
      parent: parentSpan,
      attributes: {
        "coordination.tasks_completed": tasksCompleted,
        "coordination.tasks_failed": tasksFailed,
        "coordination.success": tasksFailed === 0,
      },
    });
  }

  // Trace full task lifecycle
  async traceTaskLifecycle<T>(
    taskId: string,
    description: string,
    priority: number,
    workerId: string,
    executeFn: () => Promise<T>,
  ): Promise<T> {
    // Create parent span for entire task lifecycle
    const lifecycleSpan = this.manager.startSpan("task.lifecycle", {
      kind: SpanKind.INTERNAL,
      attributes: {
        "task.id": taskId,
        "task.description": description.substring(0, 100),
        "task.priority": priority,
        "worker.id": workerId,
      },
    });

    try {
      // Create span
      const createSpan = this.traceTaskCreate(taskId, description, priority);
      this.manager.endSpan(createSpan);

      // Claim span
      const claimSpan = this.traceTaskClaim(taskId, workerId, lifecycleSpan);
      this.manager.endSpan(claimSpan);

      // Execute span
      const executeSpan = this.traceTaskExecute(
        taskId,
        workerId,
        lifecycleSpan,
      );

      try {
        const result = await executeFn();

        this.manager.endSpan(executeSpan);

        // Complete span (success)
        const completeSpan = this.traceTaskComplete(
          taskId,
          true,
          lifecycleSpan,
        );
        this.manager.endSpan(completeSpan);

        this.manager.endSpan(lifecycleSpan);
        return result;
      } catch (error) {
        this.manager.endSpan(executeSpan, error as Error);

        // Complete span (failure)
        const completeSpan = this.traceTaskComplete(
          taskId,
          false,
          lifecycleSpan,
        );
        completeSpan.recordException(error as Error);
        this.manager.endSpan(completeSpan);

        this.manager.endSpan(lifecycleSpan, error as Error);
        throw error;
      }
    } catch (error) {
      this.manager.endSpan(lifecycleSpan, error as Error);
      throw error;
    }
  }
}

// Context propagation helpers
export function extractTraceContext(
  headers: Record<string, string>,
): Record<string, string> {
  const traceHeaders: Record<string, string> = {};

  // W3C Trace Context
  if (headers["traceparent"]) {
    traceHeaders["traceparent"] = headers["traceparent"];
  }
  if (headers["tracestate"]) {
    traceHeaders["tracestate"] = headers["tracestate"];
  }

  // B3 headers (Zipkin)
  if (headers["x-b3-traceid"]) {
    traceHeaders["x-b3-traceid"] = headers["x-b3-traceid"];
  }
  if (headers["x-b3-spanid"]) {
    traceHeaders["x-b3-spanid"] = headers["x-b3-spanid"];
  }
  if (headers["x-b3-sampled"]) {
    traceHeaders["x-b3-sampled"] = headers["x-b3-sampled"];
  }

  return traceHeaders;
}

export function injectTraceContext(span: Span): Record<string, string> {
  const headers: Record<string, string> = {};
  const spanContext = span.spanContext();

  // W3C Trace Context format
  const traceparent = `00-${spanContext.traceId}-${spanContext.spanId}-01`;
  headers["traceparent"] = traceparent;

  return headers;
}

// Singleton instance
let tracingManagerInstance: TracingManager | null = null;
let coordinatorTracingInstance: CoordinatorTracing | null = null;

export function initializeTracing(
  config: Partial<TracingConfig> = {},
): TracingManager {
  if (!tracingManagerInstance) {
    tracingManagerInstance = new TracingManager(config);
    tracingManagerInstance.initialize();
    coordinatorTracingInstance = new CoordinatorTracing(tracingManagerInstance);
  }
  return tracingManagerInstance;
}

export function getTracingManager(): TracingManager {
  if (!tracingManagerInstance) {
    throw new Error("Tracing not initialized. Call initializeTracing() first.");
  }
  return tracingManagerInstance;
}

export function getCoordinatorTracing(): CoordinatorTracing {
  if (!coordinatorTracingInstance) {
    throw new Error("Tracing not initialized. Call initializeTracing() first.");
  }
  return coordinatorTracingInstance;
}

export async function shutdownTracing(): Promise<void> {
  if (tracingManagerInstance) {
    await tracingManagerInstance.shutdown();
    tracingManagerInstance = null;
    coordinatorTracingInstance = null;
  }
}

// Main entry point for standalone usage
if (require.main === module) {
  const manager = initializeTracing({
    debug: true,
    exporterType: "console",
  });

  // Example usage
  const coordTracing = getCoordinatorTracing();

  // Create a sample trace
  const taskSpan = coordTracing.traceTaskCreate("task-1", "Sample task", 2);
  manager.endSpan(taskSpan);

  const claimSpan = coordTracing.traceTaskClaim("task-1", "worker-1");
  manager.endSpan(claimSpan);

  const execSpan = coordTracing.traceTaskExecute("task-1", "worker-1");

  setTimeout(() => {
    manager.endSpan(execSpan);

    const completeSpan = coordTracing.traceTaskComplete("task-1", true);
    manager.endSpan(completeSpan);

    console.log("Sample traces created");

    // Shutdown after traces are exported
    setTimeout(async () => {
      await shutdownTracing();
      console.log("Tracing shutdown complete");
    }, 2000);
  }, 1000);
}

export default TracingManager;
