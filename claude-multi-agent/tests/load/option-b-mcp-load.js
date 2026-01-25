/**
 * k6 Load Test: Option B MCP Server
 *
 * Tests the MCP server's task management endpoints for throughput and latency.
 * This test simulates multiple agents interacting with the coordination server.
 *
 * Prerequisites:
 *   - Option B MCP server running with HTTP transport enabled
 *   - k6 installed (https://k6.io/docs/get-started/installation/)
 *
 * Usage:
 *   k6 run option-b-mcp-load.js
 *   k6 run --vus 20 --duration 1m option-b-mcp-load.js
 *   k6 run --out json=results.json option-b-mcp-load.js
 *
 * Environment Variables:
 *   MCP_SERVER_URL: Base URL (default: http://localhost:3001)
 */

import http from "k6/http";
import { check, sleep, group } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";
import {
  randomString,
  randomIntBetween,
} from "https://jslib.k6.io/k6-utils/1.4.0/index.js";

// Custom metrics
const rpcCallDuration = new Trend("rpc_call_duration", true);
const taskOperationDuration = new Trend("task_operation_duration", true);
const agentRegistrationDuration = new Trend(
  "agent_registration_duration",
  true,
);
const rpcSuccessRate = new Rate("rpc_success");
const taskOperationRate = new Rate("task_operation_success");
const rpcCallCount = new Counter("rpc_calls_total");
const taskOperations = new Counter("task_operations_total");

// Configuration
const BASE_URL = __ENV.MCP_SERVER_URL || "http://localhost:3001";

// Test options with thresholds
export const options = {
  scenarios: {
    // Scenario 1: Agent registration and heartbeat
    agent_lifecycle: {
      executor: "per-vu-iterations",
      vus: 10,
      iterations: 5,
      exec: "agentLifecycleScenario",
      startTime: "0s",
      maxDuration: "30s",
    },
    // Scenario 2: Task creation burst
    task_creation_burst: {
      executor: "ramping-vus",
      startVUs: 1,
      stages: [
        { duration: "5s", target: 5 },
        { duration: "15s", target: 15 },
        { duration: "5s", target: 5 },
        { duration: "5s", target: 0 },
      ],
      exec: "taskCreationBurstScenario",
      startTime: "35s",
    },
    // Scenario 3: Concurrent task claiming
    concurrent_claiming: {
      executor: "constant-vus",
      vus: 10,
      duration: "30s",
      exec: "concurrentClaimingScenario",
      startTime: "70s",
    },
    // Scenario 4: Batch operations
    batch_operations: {
      executor: "constant-arrival-rate",
      rate: 5,
      timeUnit: "1s",
      duration: "20s",
      preAllocatedVUs: 5,
      maxVUs: 10,
      exec: "batchOperationsScenario",
      startTime: "105s",
    },
    // Scenario 5: Sustained mixed workload
    sustained_mixed: {
      executor: "constant-arrival-rate",
      rate: 20,
      timeUnit: "1s",
      duration: "45s",
      preAllocatedVUs: 15,
      maxVUs: 30,
      exec: "sustainedMixedScenario",
      startTime: "130s",
    },
  },
  thresholds: {
    // Performance thresholds
    rpc_call_duration: ["p(95) < 500", "avg < 200"],
    task_operation_duration: ["p(95) < 400", "avg < 150"],
    agent_registration_duration: ["p(95) < 300", "avg < 100"],

    // Success rate thresholds
    rpc_success: ["rate > 0.95"],
    task_operation_success: ["rate > 0.90"],

    // HTTP thresholds
    http_req_failed: ["rate < 0.05"],
    http_req_duration: ["p(95) < 1000", "p(99) < 2000"],
  },
};

// Helper: Make RPC call to MCP server
function makeRpcCall(method, params = {}) {
  const startTime = Date.now();

  const payload = {
    jsonrpc: "2.0",
    id: randomIntBetween(1, 1000000),
    method: method,
    params: params,
  };

  const response = http.post(`${BASE_URL}/rpc`, JSON.stringify(payload), {
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
  });

  const duration = Date.now() - startTime;
  rpcCallDuration.add(duration);
  rpcCallCount.add(1);

  const success = check(response, {
    "rpc status 200": (r) => r.status === 200,
    "rpc no error": (r) => {
      try {
        const body = JSON.parse(r.body);
        return !body.error;
      } catch {
        return false;
      }
    },
  });

  rpcSuccessRate.add(success);

  if (success) {
    try {
      return JSON.parse(response.body).result;
    } catch {
      return null;
    }
  }
  return null;
}

// Helper: Call MCP tool
function callTool(toolName, args = {}) {
  const startTime = Date.now();

  const payload = {
    jsonrpc: "2.0",
    id: randomIntBetween(1, 1000000),
    method: "tools/call",
    params: {
      name: toolName,
      arguments: args,
    },
  };

  const response = http.post(`${BASE_URL}/rpc`, JSON.stringify(payload), {
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
  });

  const duration = Date.now() - startTime;
  taskOperationDuration.add(duration);
  taskOperations.add(1);

  const success = check(response, {
    "tool call status 200": (r) => r.status === 200,
    "tool call success": (r) => {
      try {
        const body = JSON.parse(r.body);
        if (body.error) return false;
        const content = body.result?.content?.[0]?.text;
        if (!content) return true; // No content is ok for some tools
        const parsed = JSON.parse(content);
        return parsed.success !== false;
      } catch {
        return true; // Parsing failure doesn't mean call failed
      }
    },
  });

  taskOperationRate.add(success);

  if (success) {
    try {
      const body = JSON.parse(response.body);
      const content = body.result?.content?.[0]?.text;
      return content ? JSON.parse(content) : body.result;
    } catch {
      return null;
    }
  }
  return null;
}

// Scenario 1: Agent lifecycle (register, heartbeat, work)
export function agentLifecycleScenario() {
  const agentId = `agent-${__VU}-${randomString(4)}`;

  group("agent_registration", () => {
    const startTime = Date.now();
    const result = callTool("register_agent", {
      agent_id: agentId,
      role: "worker",
    });
    agentRegistrationDuration.add(Date.now() - startTime);

    check(result, {
      "agent registered": (r) => r && r.success,
    });
  });

  // Send heartbeats
  group("agent_heartbeats", () => {
    for (let i = 0; i < 3; i++) {
      callTool("heartbeat", { agent_id: agentId });
      sleep(1);
    }
  });

  sleep(0.5);
}

// Scenario 2: Task creation burst
export function taskCreationBurstScenario() {
  group("task_creation", () => {
    const taskId = randomString(8);
    const result = callTool("create_task", {
      description: `Burst task ${taskId}`,
      priority: randomIntBetween(1, 10),
      dependencies: [],
    });

    check(result, {
      "task created": (r) => r && (r.success || r.task),
    });
  });

  sleep(0.1);
}

// Scenario 3: Concurrent claiming
export function concurrentClaimingScenario() {
  const agentId = `claimer-${__VU}-${randomString(4)}`;

  group("register_for_claiming", () => {
    callTool("register_agent", {
      agent_id: agentId,
      role: "worker",
    });
  });

  group("claim_and_complete", () => {
    // Create a task first to ensure there's something to claim
    callTool("create_task", {
      description: `Claim test ${randomString(6)}`,
      priority: 5,
    });

    sleep(0.05);

    // Try to claim
    const claimed = callTool("claim_task", { agent_id: agentId });

    if (claimed && (claimed.task || claimed.success)) {
      const taskId = claimed.task?.id;
      if (taskId) {
        sleep(0.1); // Simulate work

        // Start the task
        callTool("start_task", {
          agent_id: agentId,
          task_id: taskId,
        });

        sleep(0.1);

        // Complete the task
        callTool("complete_task", {
          agent_id: agentId,
          task_id: taskId,
          output: `Completed by ${agentId}`,
        });
      }
    }
  });

  sleep(0.2);
}

// Scenario 4: Batch operations
export function batchOperationsScenario() {
  const agentId = `batch-${__VU}-${randomString(4)}`;

  group("batch_task_operations", () => {
    // Create multiple tasks in a batch
    const operations = [];
    for (let i = 0; i < 5; i++) {
      operations.push({
        operation: "create_task",
        params: {
          description: `Batch task ${i} - ${randomString(6)}`,
          priority: randomIntBetween(1, 10),
        },
      });
    }

    const result = callTool("batch_operations", {
      agent_id: agentId,
      operations: operations,
    });

    check(result, {
      "batch operations success": (r) => r && r.success,
      "batch created all tasks": (r) => r && r.successful >= 3,
    });
  });

  sleep(0.5);
}

// Scenario 5: Sustained mixed workload
export function sustainedMixedScenario() {
  const agentId = `mixed-${__VU}-${randomString(4)}`;
  const operationType = randomIntBetween(1, 10);

  group("mixed_workload", () => {
    if (operationType <= 3) {
      // 30% - Create task
      callTool("create_task", {
        description: `Mixed task ${randomString(8)}`,
        priority: randomIntBetween(1, 10),
      });
    } else if (operationType <= 5) {
      // 20% - Claim and complete
      callTool("register_agent", { agent_id: agentId, role: "worker" });
      const claimed = callTool("claim_task", { agent_id: agentId });
      if (claimed && claimed.task) {
        callTool("complete_task", {
          agent_id: agentId,
          task_id: claimed.task.id,
          output: "Mixed workload complete",
        });
      }
    } else if (operationType <= 6) {
      // 10% - Get status
      callTool("get_status", {});
    } else if (operationType <= 8) {
      // 20% - Query tasks
      callTool("query_tasks", {
        status: ["available"],
        page: 1,
        page_size: 20,
      });
    } else if (operationType <= 9) {
      // 10% - Heartbeat
      callTool("register_agent", { agent_id: agentId, role: "worker" });
      callTool("heartbeat", { agent_id: agentId });
    } else {
      // 10% - Health check
      callTool("health_check", { check_type: "liveness" });
    }
  });

  sleep(0.05);
}

// Default function
export default function () {
  sustainedMixedScenario();
}

// Setup: Initialize coordination session
export function setup() {
  // Check health
  const healthResponse = http.get(`${BASE_URL}/health`, {
    headers: { Accept: "application/json" },
  });

  const healthOk = check(healthResponse, {
    "server health check": (r) => r.status === 200 || r.status === 404, // 404 ok if no /health endpoint
  });

  // Initialize coordination session
  const initResult = callTool("init_coordination", {
    goal: "Load testing session",
    master_plan: "Execute load test scenarios",
  });

  console.log(`Load test setup complete. Server: ${BASE_URL}`);
  return { baseUrl: BASE_URL };
}

// Teardown
export function teardown(data) {
  // Get final status
  const status = callTool("get_status", {});
  if (status) {
    console.log(`Final status: ${JSON.stringify(status)}`);
  }
  console.log(`Load test completed against ${data.baseUrl}`);
}
