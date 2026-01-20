/**
 * Health Check Endpoint (adv-b-010)
 *
 * Provides comprehensive health status for the MCP server
 * including storage, agent, task, and memory health checks.
 */

import * as fs from "fs";
import * as path from "path";
import type { HealthStatus, HealthCheck, Task, Agent } from "./types.js";

const VERSION = "1.0.0";
const startTime = Date.now();

export interface HealthCheckContext {
  stateDir: string;
  tasks: Task[];
  agents: Agent[];
  lastActivity: string;
}

/**
 * Check storage health
 */
function checkStorage(stateDir: string): HealthCheck {
  try {
    // Check if state directory exists and is writable
    if (!fs.existsSync(stateDir)) {
      return {
        status: "fail",
        message: "State directory does not exist",
        details: { path: stateDir },
      };
    }

    // Try to write a test file
    const testFile = path.join(stateDir, ".health-check-test");
    try {
      fs.writeFileSync(testFile, "test");
      fs.unlinkSync(testFile);
    } catch (error) {
      return {
        status: "fail",
        message: "State directory is not writable",
        details: {
          path: stateDir,
          error: error instanceof Error ? error.message : String(error),
        },
      };
    }

    // Check state file exists and is readable
    const stateFile = path.join(stateDir, "mcp-state.json");
    if (fs.existsSync(stateFile)) {
      try {
        const stats = fs.statSync(stateFile);
        return {
          status: "pass",
          message: "Storage is healthy",
          details: {
            path: stateDir,
            state_file_size: stats.size,
            last_modified: stats.mtime.toISOString(),
          },
        };
      } catch (error) {
        return {
          status: "warn",
          message: "State file exists but cannot read stats",
          details: {
            path: stateDir,
            error: error instanceof Error ? error.message : String(error),
          },
        };
      }
    }

    return {
      status: "pass",
      message: "Storage is healthy (no state file yet)",
      details: { path: stateDir },
    };
  } catch (error) {
    return {
      status: "fail",
      message: "Storage check failed",
      details: {
        error: error instanceof Error ? error.message : String(error),
      },
    };
  }
}

/**
 * Check agent health
 */
function checkAgents(agents: Agent[]): HealthCheck {
  const now = Date.now();
  const activeThreshold = 60000; // 1 minute

  const totalAgents = agents.length;
  const activeAgents = agents.filter(
    (a) => now - new Date(a.last_heartbeat).getTime() < activeThreshold,
  ).length;
  const staleAgents = totalAgents - activeAgents;

  if (totalAgents === 0) {
    return {
      status: "pass",
      message: "No agents registered",
      details: { total: 0, active: 0, stale: 0 },
    };
  }

  if (staleAgents > 0 && activeAgents === 0) {
    return {
      status: "warn",
      message: "All agents are stale",
      details: { total: totalAgents, active: activeAgents, stale: staleAgents },
    };
  }

  if (staleAgents > totalAgents / 2) {
    return {
      status: "warn",
      message: "More than half of agents are stale",
      details: { total: totalAgents, active: activeAgents, stale: staleAgents },
    };
  }

  return {
    status: "pass",
    message: "Agents are healthy",
    details: { total: totalAgents, active: activeAgents, stale: staleAgents },
  };
}

/**
 * Check task queue health
 */
function checkTasks(tasks: Task[]): HealthCheck {
  const tasksByStatus = {
    available: 0,
    claimed: 0,
    in_progress: 0,
    done: 0,
    failed: 0,
  };

  for (const task of tasks) {
    tasksByStatus[task.status]++;
  }

  const totalTasks = tasks.length;
  const failedRate = totalTasks > 0 ? tasksByStatus.failed / totalTasks : 0;
  const stuckTasks = tasksByStatus.claimed + tasksByStatus.in_progress;

  // Check for stuck tasks (claimed but not progressing)
  const now = Date.now();
  const stuckThreshold = 30 * 60 * 1000; // 30 minutes
  let longRunningTasks = 0;

  for (const task of tasks) {
    if (
      (task.status === "claimed" || task.status === "in_progress") &&
      task.claimed_at
    ) {
      const claimedTime = new Date(task.claimed_at).getTime();
      if (now - claimedTime > stuckThreshold) {
        longRunningTasks++;
      }
    }
  }

  if (failedRate > 0.5) {
    return {
      status: "fail",
      message: "High task failure rate (>50%)",
      details: {
        total: totalTasks,
        by_status: tasksByStatus,
        failure_rate: Math.round(failedRate * 100) + "%",
        long_running: longRunningTasks,
      },
    };
  }

  if (failedRate > 0.2) {
    return {
      status: "warn",
      message: "Elevated task failure rate (>20%)",
      details: {
        total: totalTasks,
        by_status: tasksByStatus,
        failure_rate: Math.round(failedRate * 100) + "%",
        long_running: longRunningTasks,
      },
    };
  }

  if (longRunningTasks > 0) {
    return {
      status: "warn",
      message: `${longRunningTasks} task(s) running longer than 30 minutes`,
      details: {
        total: totalTasks,
        by_status: tasksByStatus,
        failure_rate: Math.round(failedRate * 100) + "%",
        long_running: longRunningTasks,
      },
    };
  }

  return {
    status: "pass",
    message: "Task queue is healthy",
    details: {
      total: totalTasks,
      by_status: tasksByStatus,
      failure_rate: Math.round(failedRate * 100) + "%",
      long_running: longRunningTasks,
    },
  };
}

/**
 * Check memory usage
 */
function checkMemory(): HealthCheck {
  const memUsage = process.memoryUsage();

  // Convert to MB for readability
  const heapUsedMB = Math.round(memUsage.heapUsed / 1024 / 1024);
  const heapTotalMB = Math.round(memUsage.heapTotal / 1024 / 1024);
  const rssMB = Math.round(memUsage.rss / 1024 / 1024);
  const externalMB = Math.round(memUsage.external / 1024 / 1024);

  const heapUsagePercent = (memUsage.heapUsed / memUsage.heapTotal) * 100;

  if (heapUsagePercent > 90) {
    return {
      status: "fail",
      message: "Critical memory usage (>90%)",
      details: {
        heap_used_mb: heapUsedMB,
        heap_total_mb: heapTotalMB,
        rss_mb: rssMB,
        external_mb: externalMB,
        heap_usage_percent: Math.round(heapUsagePercent),
      },
    };
  }

  if (heapUsagePercent > 75) {
    return {
      status: "warn",
      message: "High memory usage (>75%)",
      details: {
        heap_used_mb: heapUsedMB,
        heap_total_mb: heapTotalMB,
        rss_mb: rssMB,
        external_mb: externalMB,
        heap_usage_percent: Math.round(heapUsagePercent),
      },
    };
  }

  return {
    status: "pass",
    message: "Memory usage is healthy",
    details: {
      heap_used_mb: heapUsedMB,
      heap_total_mb: heapTotalMB,
      rss_mb: rssMB,
      external_mb: externalMB,
      heap_usage_percent: Math.round(heapUsagePercent),
    },
  };
}

/**
 * Get comprehensive health status
 */
export function getHealthStatus(context: HealthCheckContext): HealthStatus {
  const storageCheck = checkStorage(context.stateDir);
  const agentsCheck = checkAgents(context.agents);
  const tasksCheck = checkTasks(context.tasks);
  const memoryCheck = checkMemory();

  // Determine overall status
  const checks = [storageCheck, agentsCheck, tasksCheck, memoryCheck];
  let overallStatus: HealthStatus["status"] = "healthy";

  for (const check of checks) {
    if (check.status === "fail") {
      overallStatus = "unhealthy";
      break;
    }
    if (check.status === "warn") {
      overallStatus = "degraded";
    }
  }

  return {
    status: overallStatus,
    uptime_seconds: Math.floor((Date.now() - startTime) / 1000),
    version: VERSION,
    checks: {
      storage: storageCheck,
      agents: agentsCheck,
      tasks: tasksCheck,
      memory: memoryCheck,
    },
    timestamp: new Date().toISOString(),
  };
}

/**
 * Simple liveness check (just returns true if server is running)
 */
export function livenessCheck(): { alive: boolean; timestamp: string } {
  return {
    alive: true,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Readiness check (checks if server is ready to accept requests)
 */
export function readinessCheck(context: HealthCheckContext): {
  ready: boolean;
  reason?: string;
  timestamp: string;
} {
  const storageCheck = checkStorage(context.stateDir);

  if (storageCheck.status === "fail") {
    return {
      ready: false,
      reason: storageCheck.message,
      timestamp: new Date().toISOString(),
    };
  }

  return {
    ready: true,
    timestamp: new Date().toISOString(),
  };
}
