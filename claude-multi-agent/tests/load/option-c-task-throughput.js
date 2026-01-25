/**
 * k6 Load Test: Option C Orchestrator Task Throughput
 *
 * Tests the task creation, claiming, and completion throughput of
 * the Option C orchestrator through HTTP endpoints.
 *
 * Prerequisites:
 *   - Option C orchestrator running with HTTP API enabled
 *   - k6 installed (https://k6.io/docs/get-started/installation/)
 *
 * Usage:
 *   k6 run option-c-task-throughput.js
 *   k6 run --vus 10 --duration 30s option-c-task-throughput.js
 *   k6 run --out json=results.json option-c-task-throughput.js
 *
 * Environment Variables:
 *   ORCHESTRATOR_URL: Base URL (default: http://localhost:8080)
 */

import http from "k6/http";
import { check, sleep, group } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";
import { randomString } from "https://jslib.k6.io/k6-utils/1.4.0/index.js";

// Custom metrics
const taskCreationDuration = new Trend("task_creation_duration", true);
const taskClaimDuration = new Trend("task_claim_duration", true);
const taskCompletionDuration = new Trend("task_completion_duration", true);
const taskCreationRate = new Rate("task_creation_success");
const taskClaimRate = new Rate("task_claim_success");
const taskCompletionRate = new Rate("task_completion_success");
const tasksCreated = new Counter("tasks_created_total");
const tasksClaimed = new Counter("tasks_claimed_total");
const tasksCompleted = new Counter("tasks_completed_total");

// Configuration
const BASE_URL = __ENV.ORCHESTRATOR_URL || "http://localhost:8080";

// Test options with thresholds
export const options = {
  scenarios: {
    // Scenario 1: Task creation burst
    task_creation_burst: {
      executor: "ramping-vus",
      startVUs: 1,
      stages: [
        { duration: "10s", target: 5 }, // Ramp up
        { duration: "20s", target: 10 }, // Peak burst
        { duration: "10s", target: 0 }, // Ramp down
      ],
      gracefulRampDown: "5s",
      exec: "taskCreationScenario",
      startTime: "0s",
    },
    // Scenario 2: Concurrent claiming
    concurrent_claiming: {
      executor: "constant-vus",
      vus: 5,
      duration: "30s",
      exec: "concurrentClaimingScenario",
      startTime: "45s",
    },
    // Scenario 3: Sustained load (full lifecycle)
    sustained_load: {
      executor: "constant-arrival-rate",
      rate: 10, // 10 iterations per second
      timeUnit: "1s",
      duration: "60s",
      preAllocatedVUs: 10,
      maxVUs: 20,
      exec: "sustainedLoadScenario",
      startTime: "80s",
    },
  },
  thresholds: {
    // Performance thresholds
    task_creation_duration: ["p(95) < 500", "avg < 200"], // p95 < 500ms, avg < 200ms
    task_claim_duration: ["p(95) < 300", "avg < 150"], // p95 < 300ms, avg < 150ms
    task_completion_duration: ["p(95) < 400", "avg < 200"], // p95 < 400ms, avg < 200ms

    // Success rate thresholds
    task_creation_success: ["rate > 0.95"], // >95% success
    task_claim_success: ["rate > 0.90"], // >90% success (claims may fail if no tasks)
    task_completion_success: ["rate > 0.95"], // >95% success

    // HTTP error rate
    http_req_failed: ["rate < 0.05"], // <5% error rate

    // Overall latency
    http_req_duration: ["p(95) < 1000"], // p95 < 1s
  },
};

// Helper function to create headers
function getHeaders() {
  return {
    "Content-Type": "application/json",
    Accept: "application/json",
  };
}

// Helper function to create a task
function createTask(description, priority = 5) {
  const startTime = Date.now();
  const response = http.post(
    `${BASE_URL}/api/tasks`,
    JSON.stringify({
      description: description,
      priority: priority,
      dependencies: [],
    }),
    { headers: getHeaders() },
  );
  const duration = Date.now() - startTime;

  const success = check(response, {
    "task creation status 201": (r) => r.status === 201,
    "task creation has id": (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.id !== undefined;
      } catch {
        return false;
      }
    },
  });

  taskCreationDuration.add(duration);
  taskCreationRate.add(success);
  if (success) {
    tasksCreated.add(1);
  }

  return success ? JSON.parse(response.body) : null;
}

// Helper function to claim a task
function claimTask(agentId) {
  const startTime = Date.now();
  const response = http.post(
    `${BASE_URL}/api/tasks/claim`,
    JSON.stringify({ agent_id: agentId }),
    { headers: getHeaders() },
  );
  const duration = Date.now() - startTime;

  const success = check(response, {
    "task claim status 200": (r) => r.status === 200,
    "task claim has task_id": (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.task_id !== undefined || body.id !== undefined;
      } catch {
        return false;
      }
    },
  });

  taskClaimDuration.add(duration);
  taskClaimRate.add(success);
  if (success) {
    tasksClaimed.add(1);
  }

  return success ? JSON.parse(response.body) : null;
}

// Helper function to complete a task
function completeTask(taskId, agentId, result) {
  const startTime = Date.now();
  const response = http.post(
    `${BASE_URL}/api/tasks/${taskId}/complete`,
    JSON.stringify({
      agent_id: agentId,
      result: result,
    }),
    { headers: getHeaders() },
  );
  const duration = Date.now() - startTime;

  const success = check(response, {
    "task completion status 200": (r) => r.status === 200,
  });

  taskCompletionDuration.add(duration);
  taskCompletionRate.add(success);
  if (success) {
    tasksCompleted.add(1);
  }

  return success;
}

// Helper function to get available tasks count
function getAvailableTasksCount() {
  const response = http.get(`${BASE_URL}/api/tasks?status=available`, {
    headers: getHeaders(),
  });

  if (response.status === 200) {
    try {
      const body = JSON.parse(response.body);
      return body.tasks
        ? body.tasks.length
        : Array.isArray(body)
          ? body.length
          : 0;
    } catch {
      return 0;
    }
  }
  return 0;
}

// Scenario 1: Task creation burst
export function taskCreationScenario() {
  group("task_creation_burst", () => {
    const taskId = randomString(8);
    const description = `Load test task ${taskId} - burst creation`;
    createTask(description, Math.floor(Math.random() * 10) + 1);
    sleep(0.1); // 100ms between creations
  });
}

// Scenario 2: Concurrent claiming
export function concurrentClaimingScenario() {
  group("concurrent_claiming", () => {
    const agentId = `worker-${__VU}-${randomString(4)}`;

    // First create a task if queue is low
    const availableTasks = getAvailableTasksCount();
    if (availableTasks < 5) {
      createTask(`Claim test task ${randomString(8)}`, 5);
    }

    // Try to claim a task
    const claimedTask = claimTask(agentId);

    if (claimedTask) {
      // Simulate work
      sleep(Math.random() * 0.5 + 0.1); // 100-600ms work simulation

      // Complete the task
      const taskId = claimedTask.task_id || claimedTask.id;
      completeTask(taskId, agentId, `Completed by ${agentId}`);
    }

    sleep(0.2); // 200ms between claim attempts
  });
}

// Scenario 3: Sustained load (full task lifecycle)
export function sustainedLoadScenario() {
  group("sustained_load", () => {
    const agentId = `sustained-worker-${__VU}`;
    const taskSuffix = randomString(8);

    // Create task
    const task = createTask(`Sustained load task ${taskSuffix}`, 5);

    if (task) {
      sleep(0.05); // 50ms delay

      // Claim task
      const claimedTask = claimTask(agentId);

      if (claimedTask) {
        sleep(0.1); // 100ms work simulation

        // Complete task
        const taskId = claimedTask.task_id || claimedTask.id;
        completeTask(taskId, agentId, `Sustained test complete`);
      }
    }
  });
}

// Default function (runs if no specific scenario is selected)
export default function () {
  sustainedLoadScenario();
}

// Setup function
export function setup() {
  // Verify orchestrator is accessible
  const response = http.get(`${BASE_URL}/health`, { headers: getHeaders() });
  const healthOk = check(response, {
    "orchestrator health check": (r) => r.status === 200,
  });

  if (!healthOk) {
    console.error(
      `Orchestrator health check failed. Is it running at ${BASE_URL}?`,
    );
  }

  return { baseUrl: BASE_URL };
}

// Teardown function
export function teardown(data) {
  console.log(`Load test completed against ${data.baseUrl}`);
}
