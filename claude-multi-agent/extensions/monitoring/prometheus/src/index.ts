import express from "express";
import * as promClient from "prom-client";
import * as fs from "fs";
import * as path from "path";
import { config } from "dotenv";

config();

// Configuration
const port = parseInt(process.env.METRICS_PORT || "9090");
const coordinationDir = process.env.COORDINATION_DIR || ".coordination";
const collectInterval = parseInt(process.env.COLLECT_INTERVAL || "15000");

// Create Prometheus Registry
const register = new promClient.Registry();

// Add default metrics
promClient.collectDefaultMetrics({ register });

// Custom Metrics

// Task metrics
const tasksTotalGauge = new promClient.Gauge({
  name: "coordinator_tasks_total",
  help: "Total number of tasks",
  registers: [register],
});

const tasksByStatusGauge = new promClient.Gauge({
  name: "coordinator_tasks_by_status",
  help: "Number of tasks by status",
  labelNames: ["status"],
  registers: [register],
});

const tasksByPriorityGauge = new promClient.Gauge({
  name: "coordinator_tasks_by_priority",
  help: "Number of tasks by priority",
  labelNames: ["priority"],
  registers: [register],
});

const tasksCreatedCounter = new promClient.Counter({
  name: "coordinator_tasks_created_total",
  help: "Total number of tasks created",
  registers: [register],
});

const tasksCompletedCounter = new promClient.Counter({
  name: "coordinator_tasks_completed_total",
  help: "Total number of tasks completed",
  registers: [register],
});

const tasksFailedCounter = new promClient.Counter({
  name: "coordinator_tasks_failed_total",
  help: "Total number of tasks failed",
  registers: [register],
});

const taskDurationHistogram = new promClient.Histogram({
  name: "coordinator_task_duration_seconds",
  help: "Task execution duration in seconds",
  labelNames: ["priority", "status"],
  buckets: [1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600],
  registers: [register],
});

const taskQueueWaitTimeHistogram = new promClient.Histogram({
  name: "coordinator_task_queue_wait_seconds",
  help: "Time tasks spend waiting in queue before being claimed",
  buckets: [1, 5, 15, 30, 60, 120, 300, 600],
  registers: [register],
});

// Worker metrics
const workersTotalGauge = new promClient.Gauge({
  name: "coordinator_workers_total",
  help: "Total number of workers",
  registers: [register],
});

const workersByStatusGauge = new promClient.Gauge({
  name: "coordinator_workers_by_status",
  help: "Number of workers by status",
  labelNames: ["status"],
  registers: [register],
});

const workerTasksCompletedGauge = new promClient.Gauge({
  name: "coordinator_worker_tasks_completed",
  help: "Number of tasks completed per worker",
  labelNames: ["worker_id"],
  registers: [register],
});

const workerLastHeartbeatGauge = new promClient.Gauge({
  name: "coordinator_worker_last_heartbeat_timestamp",
  help: "Timestamp of last heartbeat per worker",
  labelNames: ["worker_id"],
  registers: [register],
});

// Discovery metrics
const discoveriesTotalGauge = new promClient.Gauge({
  name: "coordinator_discoveries_total",
  help: "Total number of discoveries",
  registers: [register],
});

const discoveriesByCreatorGauge = new promClient.Gauge({
  name: "coordinator_discoveries_by_creator",
  help: "Number of discoveries by creator",
  labelNames: ["creator"],
  registers: [register],
});

// Coordination metrics
const coordinationUptimeGauge = new promClient.Gauge({
  name: "coordinator_uptime_seconds",
  help: "Time since coordination started in seconds",
  registers: [register],
});

const lastUpdateTimestampGauge = new promClient.Gauge({
  name: "coordinator_last_update_timestamp",
  help: "Timestamp of last state update",
  registers: [register],
});

// Throughput metrics
const taskThroughputGauge = new promClient.Gauge({
  name: "coordinator_task_throughput_per_minute",
  help: "Tasks completed per minute (rolling average)",
  registers: [register],
});

// Data loading functions
function loadTasks(): any[] {
  const tasksPath = path.join(coordinationDir, "tasks.json");
  if (!fs.existsSync(tasksPath)) return [];
  try {
    return JSON.parse(fs.readFileSync(tasksPath, "utf-8")).tasks || [];
  } catch {
    return [];
  }
}

function loadWorkers(): any[] {
  const workersPath = path.join(coordinationDir, "workers.json");
  if (!fs.existsSync(workersPath)) return [];
  try {
    return JSON.parse(fs.readFileSync(workersPath, "utf-8")).workers || [];
  } catch {
    return [];
  }
}

function loadDiscoveries(): any[] {
  const discoveriesPath = path.join(coordinationDir, "discoveries.json");
  if (!fs.existsSync(discoveriesPath)) return [];
  try {
    return (
      JSON.parse(fs.readFileSync(discoveriesPath, "utf-8")).discoveries || []
    );
  } catch {
    return [];
  }
}

// Track previous state for counters
let previousTaskState = new Map<string, any>();
let completedCount = 0;
let failedCount = 0;
let createdCount = 0;
const startTime = Date.now();

// Throughput tracking
const completionTimestamps: number[] = [];

function calculateThroughput(): number {
  const oneMinuteAgo = Date.now() - 60000;
  const recentCompletions = completionTimestamps.filter(
    (t) => t > oneMinuteAgo,
  );
  return recentCompletions.length;
}

// Collect metrics
function collectMetrics(): void {
  const tasks = loadTasks();
  const workers = loadWorkers();
  const discoveries = loadDiscoveries();

  // Task metrics
  tasksTotalGauge.set(tasks.length);

  // Reset status gauges
  const statusCounts: Record<string, number> = {
    available: 0,
    claimed: 0,
    in_progress: 0,
    done: 0,
    failed: 0,
  };

  const priorityCounts: Record<string, number> = {
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
  };

  for (const task of tasks) {
    if (statusCounts[task.status] !== undefined) {
      statusCounts[task.status]++;
    }

    const priority = String(task.priority || 3);
    if (priorityCounts[priority] !== undefined) {
      priorityCounts[priority]++;
    }

    // Check for state changes
    const prevTask = previousTaskState.get(task.id);
    if (!prevTask) {
      // New task
      createdCount++;
      tasksCreatedCounter.inc();
    } else if (prevTask.status !== task.status) {
      // Status changed
      if (task.status === "done") {
        completedCount++;
        tasksCompletedCounter.inc();
        completionTimestamps.push(Date.now());

        // Calculate duration
        if (task.created_at && task.updated_at) {
          const duration =
            (new Date(task.updated_at).getTime() -
              new Date(task.created_at).getTime()) /
            1000;
          taskDurationHistogram.observe(
            { priority: String(task.priority), status: "done" },
            duration,
          );
        }
      } else if (task.status === "failed") {
        failedCount++;
        tasksFailedCounter.inc();
      } else if (task.status === "claimed" && prevTask.status === "available") {
        // Calculate queue wait time
        if (task.created_at) {
          const waitTime =
            (Date.now() - new Date(task.created_at).getTime()) / 1000;
          taskQueueWaitTimeHistogram.observe(waitTime);
        }
      }
    }

    previousTaskState.set(task.id, { ...task });
  }

  // Set status gauges
  for (const [status, count] of Object.entries(statusCounts)) {
    tasksByStatusGauge.set({ status }, count);
  }

  // Set priority gauges
  for (const [priority, count] of Object.entries(priorityCounts)) {
    tasksByPriorityGauge.set({ priority }, count);
  }

  // Worker metrics
  workersTotalGauge.set(workers.length);

  const workerStatusCounts: Record<string, number> = {
    idle: 0,
    busy: 0,
    offline: 0,
  };

  for (const worker of workers) {
    if (workerStatusCounts[worker.status] !== undefined) {
      workerStatusCounts[worker.status]++;
    }

    workerTasksCompletedGauge.set(
      { worker_id: worker.id },
      worker.tasks_completed || 0,
    );

    if (worker.last_heartbeat) {
      workerLastHeartbeatGauge.set(
        { worker_id: worker.id },
        new Date(worker.last_heartbeat).getTime() / 1000,
      );
    }
  }

  for (const [status, count] of Object.entries(workerStatusCounts)) {
    workersByStatusGauge.set({ status }, count);
  }

  // Discovery metrics
  discoveriesTotalGauge.set(discoveries.length);

  const creatorCounts: Record<string, number> = {};
  for (const discovery of discoveries) {
    const creator = discovery.created_by || "unknown";
    creatorCounts[creator] = (creatorCounts[creator] || 0) + 1;
  }

  for (const [creator, count] of Object.entries(creatorCounts)) {
    discoveriesByCreatorGauge.set({ creator }, count);
  }

  // Coordination metrics
  coordinationUptimeGauge.set((Date.now() - startTime) / 1000);
  lastUpdateTimestampGauge.set(Date.now() / 1000);

  // Throughput
  taskThroughputGauge.set(calculateThroughput());

  // Clean old completion timestamps
  const oneMinuteAgo = Date.now() - 60000;
  while (
    completionTimestamps.length > 0 &&
    completionTimestamps[0] < oneMinuteAgo
  ) {
    completionTimestamps.shift();
  }
}

// Create Express app
const app = express();

// Metrics endpoint
app.get("/metrics", async (req, res) => {
  collectMetrics();
  res.set("Content-Type", register.contentType);
  res.end(await register.metrics());
});

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// Status endpoint with JSON metrics
app.get("/status", (req, res) => {
  const tasks = loadTasks();
  const workers = loadWorkers();
  const discoveries = loadDiscoveries();

  res.json({
    tasks: {
      total: tasks.length,
      by_status: {
        available: tasks.filter((t) => t.status === "available").length,
        claimed: tasks.filter((t) => t.status === "claimed").length,
        in_progress: tasks.filter((t) => t.status === "in_progress").length,
        done: tasks.filter((t) => t.status === "done").length,
        failed: tasks.filter((t) => t.status === "failed").length,
      },
      created_total: createdCount,
      completed_total: completedCount,
      failed_total: failedCount,
      throughput_per_minute: calculateThroughput(),
    },
    workers: {
      total: workers.length,
      by_status: {
        idle: workers.filter((w) => w.status === "idle").length,
        busy: workers.filter((w) => w.status === "busy").length,
        offline: workers.filter((w) => w.status === "offline").length,
      },
    },
    discoveries: {
      total: discoveries.length,
    },
    uptime_seconds: (Date.now() - startTime) / 1000,
    timestamp: new Date().toISOString(),
  });
});

// Start periodic collection
setInterval(collectMetrics, collectInterval);

// Initial collection
collectMetrics();

// Start server
app.listen(port, () => {
  console.log(`Prometheus metrics exporter running on port ${port}`);
  console.log(`Metrics available at http://localhost:${port}/metrics`);
  console.log(`Health check at http://localhost:${port}/health`);
  console.log(`JSON status at http://localhost:${port}/status`);
});

// Grafana dashboard JSON (for reference)
export const grafanaDashboard = {
  title: "Claude Coordinator Dashboard",
  panels: [
    {
      title: "Tasks by Status",
      type: "piechart",
      targets: [
        {
          expr: "coordinator_tasks_by_status",
          legendFormat: "{{status}}",
        },
      ],
    },
    {
      title: "Task Throughput",
      type: "graph",
      targets: [
        {
          expr: "coordinator_task_throughput_per_minute",
          legendFormat: "Tasks/min",
        },
      ],
    },
    {
      title: "Workers by Status",
      type: "stat",
      targets: [
        {
          expr: "coordinator_workers_by_status",
          legendFormat: "{{status}}",
        },
      ],
    },
    {
      title: "Task Duration",
      type: "heatmap",
      targets: [
        {
          expr: "rate(coordinator_task_duration_seconds_bucket[5m])",
        },
      ],
    },
    {
      title: "Queue Wait Time",
      type: "histogram",
      targets: [
        {
          expr: "coordinator_task_queue_wait_seconds",
        },
      ],
    },
  ],
};

export { app, register, collectMetrics };
