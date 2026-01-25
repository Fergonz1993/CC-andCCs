#!/usr/bin/env node
/**
 * Claude Multi-Agent Coordination MCP Server
 *
 * This MCP server acts as a message broker for coordinating multiple
 * Claude Code instances. It provides:
 * - Task queue management
 * - Real-time task claiming with proper locking
 * - Result submission and aggregation
 * - Shared context/discoveries
 * - Agent heartbeats and status tracking
 *
 * Each Claude Code instance connects to this server and can:
 * - Leaders: Create tasks, monitor progress, aggregate results
 * - Workers: Claim tasks, submit results, share discoveries
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { v4 as uuidv4 } from "uuid";
import * as fs from "fs";
import * as path from "path";

// Import subscription and security modules
import {
  getSubscriptionManager,
  type SubscriptionEventType,
  type SubscriptionFilter,
} from "./subscriptions.js";
import { getWebhookManager } from "./webhooks.js";
import { getSSEManager } from "./sse.js";
import { getSecurityManager, type Permission } from "./security.js";

// Import advanced feature modules (adv-b-001 through adv-b-010)
import { RateLimiter } from "./rate-limiter.js";
import { RequestQueue } from "./request-queue.js";
import {
  getHealthStatus,
  livenessCheck,
  readinessCheck,
} from "./health-check.js";
import {
  createWebSocketTransport,
  type WebSocketTransport,
} from "./websocket-transport.js";
import type {
  TaskFilter,
  PaginationOptions,
  BatchOperation,
  BatchResult,
  AuditLogEntry,
  Transaction,
  TransactionOperation,
} from "./types.js";

// ============================================================================
// State Persistence Configuration
// ============================================================================

const STATE_DIR = process.env.COORDINATION_DIR || ".coordination";
const STATE_FILE = path.join(STATE_DIR, "mcp-state.json");

// Advanced features configuration
const ENABLE_WEBSOCKET = process.env.ENABLE_WEBSOCKET === "true";
const WEBSOCKET_PORT = parseInt(process.env.WEBSOCKET_PORT || "3001", 10);
const ENABLE_RATE_LIMITING = process.env.ENABLE_RATE_LIMITING !== "false";
const RATE_LIMIT_REQUESTS = parseInt(
  process.env.RATE_LIMIT_REQUESTS || "100",
  10,
);
const RATE_LIMIT_WINDOW_MS = parseInt(
  process.env.RATE_LIMIT_WINDOW_MS || "60000",
  10,
);

// Serializable version of state (Map -> Record for JSON)
interface SerializableState {
  master_plan: string;
  goal: string;
  tasks: Task[];
  agents: Record<string, Agent>;
  discoveries: Discovery[];
  created_at: string;
  last_activity: string;
  audit_log?: AuditLogEntry[];
  transactions?: Record<string, Transaction>;
}

function ensureStateDir(): void {
  if (!fs.existsSync(STATE_DIR)) {
    fs.mkdirSync(STATE_DIR, { recursive: true });
  }
}

function loadState(): void {
  try {
    if (fs.existsSync(STATE_FILE)) {
      const data = fs.readFileSync(STATE_FILE, "utf-8");
      const saved: SerializableState = JSON.parse(data);

      state.master_plan = saved.master_plan || "";
      state.goal = saved.goal || "";
      state.tasks = saved.tasks || [];
      state.discoveries = saved.discoveries || [];
      state.created_at = saved.created_at || new Date().toISOString();
      state.last_activity = saved.last_activity || new Date().toISOString();

      // Convert agents Record back to Map
      state.agents = new Map(Object.entries(saved.agents || {}));

      // Load audit log (adv-b-009)
      state.audit_log = saved.audit_log || [];

      // Convert transactions Record back to Map (adv-b-006)
      state.transactions = new Map(
        Object.entries(saved.transactions || {}),
      ) as Map<string, Transaction>;

      console.error(
        `Loaded state from ${STATE_FILE}: ${state.tasks.length} tasks, ${state.agents.size} agents, ${state.audit_log.length} audit entries`,
      );
    }
  } catch (error) {
    console.error(`Failed to load state from ${STATE_FILE}:`, error);
  }
}

function saveState(): void {
  try {
    ensureStateDir();

    // Convert Map to Record for JSON serialization
    const serializable: SerializableState = {
      master_plan: state.master_plan,
      goal: state.goal,
      tasks: state.tasks,
      agents: Object.fromEntries(state.agents),
      discoveries: state.discoveries,
      created_at: state.created_at,
      last_activity: state.last_activity,
      // Save audit log (adv-b-009)
      audit_log: state.audit_log,
      // Save transactions (adv-b-006)
      transactions: Object.fromEntries(state.transactions),
    };

    fs.writeFileSync(STATE_FILE, JSON.stringify(serializable, null, 2));
  } catch (error) {
    console.error(`Failed to save state to ${STATE_FILE}:`, error);
  }
}

// ============================================================================
// Types
// ============================================================================

interface Task {
  id: string;
  description: string;
  status: "available" | "claimed" | "in_progress" | "done" | "failed";
  priority: number;
  claimed_by: string | null;
  dependencies: string[];
  context: {
    files?: string[];
    hints?: string;
    parent_task?: string;
  } | null;
  result: {
    output?: string;
    files_modified?: string[];
    files_created?: string[];
    error?: string;
  } | null;
  created_at: string;
  claimed_at: string | null;
  completed_at: string | null;
  tags?: string[];
}

interface Agent {
  id: string;
  role: "leader" | "worker";
  last_heartbeat: string;
  current_task: string | null;
  tasks_completed: number;
}

interface Discovery {
  id: string;
  agent_id: string;
  content: string;
  tags: string[];
  created_at: string;
}

interface CoordinationState {
  master_plan: string;
  goal: string;
  tasks: Task[];
  agents: Map<string, Agent>;
  discoveries: Discovery[];
  created_at: string;
  last_activity: string;
  audit_log: AuditLogEntry[];
  transactions: Map<string, Transaction>;
}

// ============================================================================
// State Management
// ============================================================================

const state: CoordinationState = {
  master_plan: "",
  goal: "",
  tasks: [],
  agents: new Map(),
  discoveries: [],
  created_at: new Date().toISOString(),
  last_activity: new Date().toISOString(),
  audit_log: [],
  transactions: new Map(),
};

// Advanced features instances (adv-b-001, adv-b-003, adv-b-004)
const rateLimiter = new RateLimiter({
  max_requests: RATE_LIMIT_REQUESTS,
  window_ms: RATE_LIMIT_WINDOW_MS,
  tokens: RATE_LIMIT_REQUESTS,
  last_refill: Date.now(),
});
const requestQueue = new RequestQueue(1000);
let wsTransport: WebSocketTransport | null = null;

// Initialize subscription, webhook, SSE, and security managers
const subscriptionManager = getSubscriptionManager({ batching_enabled: false });
const webhookManager = getWebhookManager();
const sseManager = getSSEManager();
const securityManager = getSecurityManager({
  api_key_enabled: false, // Disabled by default for backward compatibility
  jwt_enabled: false,
  request_signing_enabled: false,
  ip_allowlist_enabled: false,
  rate_limiting_enabled: false,
});

function updateActivity() {
  state.last_activity = new Date().toISOString();
  saveState(); // Persist to disk after every change
}

function generateTaskId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 6);
  return `task-${timestamp}-${random}`;
}

// ============================================================================
// Task Management
// ============================================================================

function createTask(
  description: string,
  priority: number = 5,
  dependencies: string[] = [],
  context: Task["context"] = null,
): Task {
  const task: Task = {
    id: generateTaskId(),
    description,
    status: "available",
    priority,
    claimed_by: null,
    dependencies,
    context,
    result: null,
    created_at: new Date().toISOString(),
    claimed_at: null,
    completed_at: null,
  };
  state.tasks.push(task);
  updateActivity();

  // Publish task created event
  subscriptionManager.publishTaskEvent("task_created", task);
  webhookManager.dispatchTaskEvent("task_created", task);
  sseManager.broadcastTaskEvent("task_created", task);

  return task;
}

function getAvailableTasks(): Task[] {
  const doneIds = new Set(
    state.tasks.filter((t) => t.status === "done").map((t) => t.id),
  );

  return state.tasks.filter(
    (t) =>
      t.status === "available" &&
      t.dependencies.every((dep) => doneIds.has(dep)),
  );
}

function claimTask(agentId: string): Task | null {
  const available = getAvailableTasks();
  if (available.length === 0) return null;

  // Sort by priority (lower number = higher priority)
  available.sort((a, b) => a.priority - b.priority);
  const task = available[0];

  task.status = "claimed";
  task.claimed_by = agentId;
  task.claimed_at = new Date().toISOString();

  // Update agent state
  const agent = state.agents.get(agentId);
  if (agent) {
    agent.current_task = task.id;
    agent.last_heartbeat = new Date().toISOString();
  }

  updateActivity();

  // Publish task claimed event
  subscriptionManager.publishTaskEvent("task_claimed", task);
  webhookManager.dispatchTaskEvent("task_claimed", task);
  sseManager.broadcastTaskEvent("task_claimed", task);

  return task;
}

function startTask(agentId: string, taskId: string): boolean {
  const task = state.tasks.find((t) => t.id === taskId);
  if (!task || task.claimed_by !== agentId) return false;

  task.status = "in_progress";
  updateActivity();

  // Publish task started event
  subscriptionManager.publishTaskEvent("task_started", task);
  webhookManager.dispatchTaskEvent("task_started", task);
  sseManager.broadcastTaskEvent("task_started", task);

  return true;
}

function completeTask(
  agentId: string,
  taskId: string,
  result: Task["result"],
): boolean {
  const task = state.tasks.find((t) => t.id === taskId);
  if (!task || task.claimed_by !== agentId) return false;

  task.status = "done";
  task.result = result;
  task.completed_at = new Date().toISOString();

  // Update agent state
  const agent = state.agents.get(agentId);
  if (agent) {
    agent.current_task = null;
    agent.tasks_completed += 1;
    agent.last_heartbeat = new Date().toISOString();
  }

  updateActivity();

  // Publish task completed event
  subscriptionManager.publishTaskEvent("task_completed", task);
  webhookManager.dispatchTaskEvent("task_completed", task);
  sseManager.broadcastTaskEvent("task_completed", task);

  return true;
}

function failTask(agentId: string, taskId: string, error: string): boolean {
  const task = state.tasks.find((t) => t.id === taskId);
  if (!task || task.claimed_by !== agentId) return false;

  task.status = "failed";
  task.result = { error };
  task.completed_at = new Date().toISOString();

  // Update agent state
  const agent = state.agents.get(agentId);
  if (agent) {
    agent.current_task = null;
    agent.last_heartbeat = new Date().toISOString();
  }

  updateActivity();

  // Publish task failed event
  subscriptionManager.publishTaskEvent("task_failed", task);
  webhookManager.dispatchTaskEvent("task_failed", task);
  sseManager.broadcastTaskEvent("task_failed", task);

  return true;
}

// ============================================================================
// Agent Management
// ============================================================================

function registerAgent(agentId: string, role: "leader" | "worker"): Agent {
  const agent: Agent = {
    id: agentId,
    role,
    last_heartbeat: new Date().toISOString(),
    current_task: null,
    tasks_completed: 0,
  };
  state.agents.set(agentId, agent);
  updateActivity();

  // Publish agent registered event
  subscriptionManager.publishAgentEvent("agent_registered", agent);
  webhookManager.dispatchAgentEvent("agent_registered", agent);
  sseManager.broadcastAgentEvent("agent_registered", agent);

  return agent;
}

function heartbeat(agentId: string): boolean {
  const agent = state.agents.get(agentId);
  if (!agent) return false;

  agent.last_heartbeat = new Date().toISOString();
  return true;
}

// ============================================================================
// Discovery Management
// ============================================================================

function addDiscovery(
  agentId: string,
  content: string,
  tags: string[] = [],
): Discovery {
  const discovery: Discovery = {
    id: uuidv4(),
    agent_id: agentId,
    content,
    tags,
    created_at: new Date().toISOString(),
  };
  state.discoveries.push(discovery);
  updateActivity();

  // Publish discovery added event
  subscriptionManager.publishDiscoveryEvent(discovery);
  webhookManager.dispatchDiscoveryEvent(discovery);
  sseManager.broadcastDiscoveryEvent(discovery);

  return discovery;
}

// ============================================================================
// Status & Reporting
// ============================================================================

function getStatus() {
  const tasksByStatus = {
    available: state.tasks.filter((t) => t.status === "available").length,
    claimed: state.tasks.filter((t) => t.status === "claimed").length,
    in_progress: state.tasks.filter((t) => t.status === "in_progress").length,
    done: state.tasks.filter((t) => t.status === "done").length,
    failed: state.tasks.filter((t) => t.status === "failed").length,
  };

  const agents = Array.from(state.agents.values());

  return {
    goal: state.goal,
    tasks: tasksByStatus,
    total_tasks: state.tasks.length,
    progress_percent:
      state.tasks.length > 0
        ? Math.round((tasksByStatus.done / state.tasks.length) * 100)
        : 0,
    agents: {
      total: agents.length,
      leaders: agents.filter((a) => a.role === "leader").length,
      workers: agents.filter((a) => a.role === "worker").length,
      active: agents.filter(
        (a) => Date.now() - new Date(a.last_heartbeat).getTime() < 60000,
      ).length,
    },
    discoveries_count: state.discoveries.length,
    last_activity: state.last_activity,
  };
}

// ============================================================================
// Advanced Features: Audit Logging (adv-b-009)
// ============================================================================

function addAuditEntry(
  action: string,
  entity_type: "task" | "agent" | "discovery" | "coordination" | "transaction",
  entity_id: string,
  agent_id: string | null,
  details?: Record<string, unknown>,
): AuditLogEntry {
  const entry: AuditLogEntry = {
    id: uuidv4(),
    timestamp: new Date().toISOString(),
    action,
    entity_type,
    entity_id,
    agent_id,
    details,
  };
  state.audit_log.push(entry);

  // Keep audit log size manageable (last 10000 entries)
  if (state.audit_log.length > 10000) {
    state.audit_log = state.audit_log.slice(-10000);
  }

  return entry;
}

function getAuditLog(
  filter?: {
    entity_type?: string;
    entity_id?: string;
    agent_id?: string;
    action?: string;
    since?: string;
  },
  limit: number = 100,
): AuditLogEntry[] {
  let entries = [...state.audit_log];

  if (filter) {
    if (filter.entity_type) {
      entries = entries.filter((e) => e.entity_type === filter.entity_type);
    }
    if (filter.entity_id) {
      entries = entries.filter((e) => e.entity_id === filter.entity_id);
    }
    if (filter.agent_id) {
      entries = entries.filter((e) => e.agent_id === filter.agent_id);
    }
    if (filter.action) {
      entries = entries.filter((e) => e.action === filter.action);
    }
    if (filter.since) {
      const sinceDate = new Date(filter.since).getTime();
      entries = entries.filter(
        (e) => new Date(e.timestamp).getTime() >= sinceDate,
      );
    }
  }

  return entries.slice(-limit);
}

// ============================================================================
// Advanced Features: Query Filtering (adv-b-007)
// ============================================================================

function filterTasks(filter: TaskFilter): Task[] {
  let tasks = [...state.tasks];

  if (filter.status && filter.status.length > 0) {
    tasks = tasks.filter((t) => filter.status!.includes(t.status));
  }

  if (filter.priority_min !== undefined) {
    tasks = tasks.filter((t) => t.priority >= filter.priority_min!);
  }

  if (filter.priority_max !== undefined) {
    tasks = tasks.filter((t) => t.priority <= filter.priority_max!);
  }

  if (filter.claimed_by) {
    tasks = tasks.filter((t) => t.claimed_by === filter.claimed_by);
  }

  if (filter.tags && filter.tags.length > 0) {
    tasks = tasks.filter(
      (t) => t.tags && t.tags.some((tag) => filter.tags!.includes(tag)),
    );
  }

  if (filter.created_after) {
    const afterDate = new Date(filter.created_after).getTime();
    tasks = tasks.filter((t) => new Date(t.created_at).getTime() >= afterDate);
  }

  if (filter.created_before) {
    const beforeDate = new Date(filter.created_before).getTime();
    tasks = tasks.filter((t) => new Date(t.created_at).getTime() <= beforeDate);
  }

  if (filter.search) {
    const search = filter.search.toLowerCase();
    tasks = tasks.filter(
      (t) =>
        t.description.toLowerCase().includes(search) ||
        t.id.toLowerCase().includes(search),
    );
  }

  return tasks;
}

// ============================================================================
// Advanced Features: Pagination (adv-b-008)
// ============================================================================

interface PaginatedResult<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

function paginateItems<T>(
  items: T[],
  options: PaginationOptions,
): PaginatedResult<T> {
  const page = options.page || 1;
  const pageSize = Math.min(options.page_size || 20, 100); // Max 100 items per page
  const total = items.length;
  const totalPages = Math.ceil(total / pageSize);

  // Sort if specified
  if (options.sort_by) {
    const sortDir = options.sort_direction === "desc" ? -1 : 1;
    items = [...items].sort((a: any, b: any) => {
      const aVal = a[options.sort_by!];
      const bVal = b[options.sort_by!];
      if (aVal < bVal) return -1 * sortDir;
      if (aVal > bVal) return 1 * sortDir;
      return 0;
    });
  }

  const startIndex = (page - 1) * pageSize;
  const paginatedItems = items.slice(startIndex, startIndex + pageSize);

  return {
    items: paginatedItems,
    total,
    page,
    page_size: pageSize,
    total_pages: totalPages,
    has_next: page < totalPages,
    has_prev: page > 1,
  };
}

// ============================================================================
// Advanced Features: Batch Operations (adv-b-005)
// ============================================================================

function executeBatchOperations(
  operations: BatchOperation[],
  agentId: string,
): BatchResult[] {
  const results: BatchResult[] = [];

  for (const op of operations) {
    try {
      let success = false;
      let data: unknown = null;
      let error: string | undefined;

      switch (op.operation) {
        case "create_task": {
          const task = createTask(
            op.params.description as string,
            (op.params.priority as number) || 5,
            (op.params.dependencies as string[]) || [],
            op.params.context as Task["context"],
          );
          success = true;
          data = task;
          addAuditEntry("batch_create_task", "task", task.id, agentId, {
            batch: true,
          });
          break;
        }
        case "update_task": {
          const task = state.tasks.find((t) => t.id === op.params.task_id);
          if (task) {
            if (op.params.priority !== undefined)
              task.priority = op.params.priority as number;
            if (op.params.tags !== undefined)
              task.tags = op.params.tags as string[];
            success = true;
            data = task;
            addAuditEntry("batch_update_task", "task", task.id, agentId, {
              batch: true,
            });
          } else {
            error = `Task ${op.params.task_id} not found`;
          }
          break;
        }
        case "delete_task": {
          const index = state.tasks.findIndex(
            (t) => t.id === op.params.task_id,
          );
          if (index !== -1) {
            const deleted = state.tasks.splice(index, 1)[0];
            success = true;
            data = { deleted_id: deleted.id };
            addAuditEntry("batch_delete_task", "task", deleted.id, agentId, {
              batch: true,
            });
          } else {
            error = `Task ${op.params.task_id} not found`;
          }
          break;
        }
        case "claim_task": {
          const claimed = claimTask(agentId);
          if (claimed) {
            success = true;
            data = claimed;
            addAuditEntry("batch_claim_task", "task", claimed.id, agentId, {
              batch: true,
            });
          } else {
            error = "No available tasks";
          }
          break;
        }
        case "complete_task": {
          success = completeTask(
            agentId,
            op.params.task_id as string,
            op.params.result as Task["result"],
          );
          if (success) {
            addAuditEntry(
              "batch_complete_task",
              "task",
              op.params.task_id as string,
              agentId,
              { batch: true },
            );
          } else {
            error = `Cannot complete task ${op.params.task_id}`;
          }
          break;
        }
        default:
          error = `Unknown operation: ${op.operation}`;
      }

      results.push({
        operation_index: operations.indexOf(op),
        success,
        data,
        error,
      });
    } catch (e) {
      results.push({
        operation_index: operations.indexOf(op),
        success: false,
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }

  updateActivity();
  return results;
}

// ============================================================================
// Advanced Features: Transaction Support (adv-b-006)
// ============================================================================

function beginTransaction(agentId: string): Transaction {
  const transaction: Transaction = {
    id: uuidv4(),
    agent_id: agentId,
    operations: [],
    status: "pending",
    created_at: new Date().toISOString(),
  };
  state.transactions.set(transaction.id, transaction);
  addAuditEntry("begin_transaction", "transaction", transaction.id, agentId);
  return transaction;
}

function addTransactionOperation(
  transactionId: string,
  operation: TransactionOperation,
): boolean {
  const transaction = state.transactions.get(transactionId);
  if (!transaction || transaction.status !== "pending") {
    return false;
  }
  transaction.operations.push(operation);
  return true;
}

function commitTransaction(transactionId: string): {
  success: boolean;
  results?: BatchResult[];
  error?: string;
} {
  const transaction = state.transactions.get(transactionId);
  if (!transaction) {
    return { success: false, error: "Transaction not found" };
  }
  if (transaction.status !== "pending") {
    return { success: false, error: `Transaction is ${transaction.status}` };
  }

  // Convert transaction operations to batch operations
  const batchOps: BatchOperation[] = transaction.operations.map((op) => ({
    operation: op.type as BatchOperation["operation"],
    params: op.params,
  }));

  // Execute all operations
  const results = executeBatchOperations(batchOps, transaction.agent_id);

  // Check if all succeeded
  const allSucceeded = results.every((r) => r.success);

  if (allSucceeded) {
    transaction.status = "committed";
    transaction.committed_at = new Date().toISOString();
    addAuditEntry(
      "commit_transaction",
      "transaction",
      transactionId,
      transaction.agent_id,
      { operations_count: results.length },
    );
    return { success: true, results };
  } else {
    // Rollback would be complex - for now we mark as failed
    transaction.status = "failed";
    addAuditEntry(
      "fail_transaction",
      "transaction",
      transactionId,
      transaction.agent_id,
      { failed_operations: results.filter((r) => !r.success).length },
    );
    return {
      success: false,
      results,
      error: "Some operations failed",
    };
  }
}

function rollbackTransaction(transactionId: string): boolean {
  const transaction = state.transactions.get(transactionId);
  if (!transaction || transaction.status !== "pending") {
    return false;
  }
  transaction.status = "rolled_back";
  transaction.operations = [];
  addAuditEntry(
    "rollback_transaction",
    "transaction",
    transactionId,
    transaction.agent_id,
  );
  return true;
}

// ============================================================================
// MCP Server Setup
// ============================================================================

const server = new Server(
  {
    name: "claude-coordination",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  },
);

// ============================================================================
// Tool Definitions
// ============================================================================

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      // === LEADER TOOLS ===
      {
        name: "init_coordination",
        description:
          "Initialize a new coordination session with a goal and master plan",
        inputSchema: {
          type: "object",
          properties: {
            goal: {
              type: "string",
              description: "The overall project goal",
            },
            master_plan: {
              type: "string",
              description: "High-level plan/approach",
            },
          },
          required: ["goal"],
        },
      },
      {
        name: "create_task",
        description: "Create a new task in the queue",
        inputSchema: {
          type: "object",
          properties: {
            description: {
              type: "string",
              description: "Task description",
            },
            priority: {
              type: "number",
              description: "Priority 1-10 (lower = higher priority)",
              default: 5,
            },
            dependencies: {
              type: "array",
              items: { type: "string" },
              description: "Task IDs that must complete first",
            },
            context_files: {
              type: "array",
              items: { type: "string" },
              description: "Relevant file paths",
            },
            hints: {
              type: "string",
              description: "Hints for the worker",
            },
          },
          required: ["description"],
        },
      },
      {
        name: "create_tasks_batch",
        description: "Create multiple tasks at once",
        inputSchema: {
          type: "object",
          properties: {
            tasks: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  description: { type: "string" },
                  priority: { type: "number" },
                  dependencies: {
                    type: "array",
                    items: { type: "string" },
                  },
                  context_files: {
                    type: "array",
                    items: { type: "string" },
                  },
                  hints: { type: "string" },
                },
                required: ["description"],
              },
            },
          },
          required: ["tasks"],
        },
      },
      {
        name: "get_status",
        description: "Get current coordination status and progress",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "get_all_tasks",
        description: "Get all tasks with their current status",
        inputSchema: {
          type: "object",
          properties: {
            status_filter: {
              type: "string",
              enum: [
                "all",
                "available",
                "claimed",
                "in_progress",
                "done",
                "failed",
              ],
              default: "all",
            },
          },
        },
      },
      {
        name: "get_results",
        description: "Get results from completed tasks",
        inputSchema: {
          type: "object",
          properties: {
            task_ids: {
              type: "array",
              items: { type: "string" },
              description: "Specific task IDs (empty for all)",
            },
          },
        },
      },

      // === WORKER TOOLS ===
      {
        name: "register_agent",
        description: "Register as a worker or leader agent",
        inputSchema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "Unique agent identifier (e.g., terminal-2)",
            },
            role: {
              type: "string",
              enum: ["leader", "worker"],
              description: "Agent role",
            },
          },
          required: ["agent_id", "role"],
        },
      },
      {
        name: "claim_task",
        description: "Claim an available task to work on",
        inputSchema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "Your agent ID",
            },
          },
          required: ["agent_id"],
        },
      },
      {
        name: "start_task",
        description: "Mark a claimed task as in_progress",
        inputSchema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "Your agent ID",
            },
            task_id: {
              type: "string",
              description: "Task ID to start",
            },
          },
          required: ["agent_id", "task_id"],
        },
      },
      {
        name: "complete_task",
        description: "Mark a task as completed with results",
        inputSchema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "Your agent ID",
            },
            task_id: {
              type: "string",
              description: "Task ID",
            },
            output: {
              type: "string",
              description: "Summary of what was done",
            },
            files_modified: {
              type: "array",
              items: { type: "string" },
              description: "Files that were modified",
            },
            files_created: {
              type: "array",
              items: { type: "string" },
              description: "Files that were created",
            },
          },
          required: ["agent_id", "task_id", "output"],
        },
      },
      {
        name: "fail_task",
        description: "Mark a task as failed",
        inputSchema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "Your agent ID",
            },
            task_id: {
              type: "string",
              description: "Task ID",
            },
            error: {
              type: "string",
              description: "Error description",
            },
          },
          required: ["agent_id", "task_id", "error"],
        },
      },
      {
        name: "heartbeat",
        description: "Send heartbeat to indicate agent is still active",
        inputSchema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "Your agent ID",
            },
          },
          required: ["agent_id"],
        },
      },

      // === SHARED TOOLS ===
      {
        name: "add_discovery",
        description: "Share a discovery or important finding with other agents",
        inputSchema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "Your agent ID",
            },
            content: {
              type: "string",
              description: "The discovery content",
            },
            tags: {
              type: "array",
              items: { type: "string" },
              description: "Tags for categorization",
            },
          },
          required: ["agent_id", "content"],
        },
      },
      {
        name: "get_discoveries",
        description: "Get shared discoveries from all agents",
        inputSchema: {
          type: "object",
          properties: {
            tags: {
              type: "array",
              items: { type: "string" },
              description: "Filter by tags",
            },
            limit: {
              type: "number",
              description: "Max number of discoveries",
              default: 20,
            },
          },
        },
      },
      {
        name: "get_master_plan",
        description: "Get the master plan and goal",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },

      // === SUBSCRIPTION TOOLS (adv-b-sub-001, adv-b-sub-004, adv-b-sub-005) ===
      {
        name: "subscribe",
        description: "Subscribe to real-time task and coordination events",
        inputSchema: {
          type: "object",
          properties: {
            subscriber_id: {
              type: "string",
              description: "Unique identifier for the subscriber",
            },
            event_types: {
              type: "array",
              items: {
                type: "string",
                enum: [
                  "task_created",
                  "task_updated",
                  "task_claimed",
                  "task_started",
                  "task_completed",
                  "task_failed",
                  "discovery_added",
                  "agent_registered",
                  "agent_heartbeat",
                  "coordination_init",
                ],
              },
              description: "Event types to subscribe to (empty for all)",
            },
            task_status: {
              type: "array",
              items: {
                type: "string",
                enum: ["available", "claimed", "in_progress", "done", "failed"],
              },
              description: "Filter by task status",
            },
            task_tags: {
              type: "array",
              items: { type: "string" },
              description: "Filter by task tags",
            },
            agent_id: {
              type: "string",
              description: "Filter by agent ID",
            },
            priority_min: {
              type: "number",
              description: "Minimum priority filter",
            },
            priority_max: {
              type: "number",
              description: "Maximum priority filter",
            },
          },
          required: ["subscriber_id"],
        },
      },
      {
        name: "unsubscribe",
        description: "Remove a subscription",
        inputSchema: {
          type: "object",
          properties: {
            subscription_id: {
              type: "string",
              description: "Subscription ID to remove",
            },
          },
          required: ["subscription_id"],
        },
      },
      {
        name: "get_subscriptions",
        description: "Get all subscriptions for a subscriber",
        inputSchema: {
          type: "object",
          properties: {
            subscriber_id: {
              type: "string",
              description: "Subscriber ID",
            },
          },
          required: ["subscriber_id"],
        },
      },
      {
        name: "configure_batching",
        description: "Configure notification batching settings",
        inputSchema: {
          type: "object",
          properties: {
            enabled: {
              type: "boolean",
              description: "Enable or disable batching",
            },
            batch_interval_ms: {
              type: "number",
              description: "Batch interval in milliseconds",
            },
            max_batch_size: {
              type: "number",
              description: "Maximum notifications per batch",
            },
          },
          required: ["enabled"],
        },
      },

      // === WEBHOOK TOOLS (adv-b-sub-002) ===
      {
        name: "register_webhook",
        description: "Register a webhook to receive event callbacks",
        inputSchema: {
          type: "object",
          properties: {
            url: {
              type: "string",
              description: "Webhook URL to receive callbacks",
            },
            events: {
              type: "array",
              items: {
                type: "string",
                enum: [
                  "task_created",
                  "task_updated",
                  "task_claimed",
                  "task_started",
                  "task_completed",
                  "task_failed",
                  "discovery_added",
                  "agent_registered",
                  "agent_heartbeat",
                ],
              },
              description: "Events to trigger webhook",
            },
            secret: {
              type: "string",
              description: "Optional secret for signing webhooks",
            },
            headers: {
              type: "object",
              description: "Optional custom headers",
            },
          },
          required: ["url", "events"],
        },
      },
      {
        name: "delete_webhook",
        description: "Delete a webhook",
        inputSchema: {
          type: "object",
          properties: {
            webhook_id: {
              type: "string",
              description: "Webhook ID to delete",
            },
          },
          required: ["webhook_id"],
        },
      },
      {
        name: "list_webhooks",
        description: "List all registered webhooks",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "get_webhook_deliveries",
        description: "Get webhook delivery history",
        inputSchema: {
          type: "object",
          properties: {
            webhook_id: {
              type: "string",
              description: "Optional webhook ID to filter",
            },
            limit: {
              type: "number",
              description: "Maximum deliveries to return",
              default: 50,
            },
          },
        },
      },

      // === SSE TOOLS (adv-b-sub-003) ===
      {
        name: "get_sse_stats",
        description: "Get SSE connection statistics",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "get_event_history",
        description: "Get recent event history for replay",
        inputSchema: {
          type: "object",
          properties: {
            limit: {
              type: "number",
              description: "Maximum events to return",
              default: 50,
            },
          },
        },
      },

      // === SECURITY TOOLS (adv-b-sec-001 through adv-b-sec-005) ===
      {
        name: "create_api_key",
        description: "Create a new API key for authentication",
        inputSchema: {
          type: "object",
          properties: {
            name: {
              type: "string",
              description: "Name for the API key",
            },
            agent_id: {
              type: "string",
              description: "Optional agent ID to associate",
            },
            role: {
              type: "string",
              enum: ["leader", "worker", "admin"],
              description: "Role for the API key",
            },
            permissions: {
              type: "array",
              items: {
                type: "string",
                enum: [
                  "read",
                  "write",
                  "admin",
                  "create_task",
                  "claim_task",
                  "complete_task",
                  "manage_agents",
                  "view_status",
                  "manage_webhooks",
                  "manage_subscriptions",
                ],
              },
              description: "Specific permissions",
            },
            ip_allowlist: {
              type: "array",
              items: { type: "string" },
              description: "IP addresses allowed to use this key",
            },
            expires_in_days: {
              type: "number",
              description: "Days until key expires",
            },
          },
          required: ["name"],
        },
      },
      {
        name: "revoke_api_key",
        description: "Revoke an API key",
        inputSchema: {
          type: "object",
          properties: {
            key_id: {
              type: "string",
              description: "API key ID to revoke",
            },
          },
          required: ["key_id"],
        },
      },
      {
        name: "list_api_keys",
        description: "List all API keys",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "generate_jwt",
        description: "Generate a JWT token for authentication",
        inputSchema: {
          type: "object",
          properties: {
            subject: {
              type: "string",
              description: "Subject (agent ID or identifier)",
            },
            role: {
              type: "string",
              enum: ["leader", "worker", "admin"],
              description: "Role for the token",
            },
            permissions: {
              type: "array",
              items: { type: "string" },
              description: "Permissions to include",
            },
          },
          required: ["subject", "role"],
        },
      },
      {
        name: "validate_jwt",
        description: "Validate a JWT token",
        inputSchema: {
          type: "object",
          properties: {
            token: {
              type: "string",
              description: "JWT token to validate",
            },
          },
          required: ["token"],
        },
      },
      {
        name: "configure_security",
        description: "Configure security settings",
        inputSchema: {
          type: "object",
          properties: {
            api_key_enabled: {
              type: "boolean",
              description: "Enable API key authentication",
            },
            jwt_enabled: {
              type: "boolean",
              description: "Enable JWT authentication",
            },
            request_signing_enabled: {
              type: "boolean",
              description: "Enable request signing",
            },
            ip_allowlist_enabled: {
              type: "boolean",
              description: "Enable IP allowlisting",
            },
            rate_limiting_enabled: {
              type: "boolean",
              description: "Enable rate limiting",
            },
          },
        },
      },
      {
        name: "add_ip_to_allowlist",
        description: "Add an IP address to the global allowlist",
        inputSchema: {
          type: "object",
          properties: {
            ip: {
              type: "string",
              description: "IP address or CIDR range",
            },
          },
          required: ["ip"],
        },
      },
      {
        name: "add_ip_to_denylist",
        description: "Add an IP address to the global denylist",
        inputSchema: {
          type: "object",
          properties: {
            ip: {
              type: "string",
              description: "IP address or CIDR range",
            },
          },
          required: ["ip"],
        },
      },
      {
        name: "get_rate_limit_status",
        description: "Get rate limit status for an API key",
        inputSchema: {
          type: "object",
          properties: {
            key_id: {
              type: "string",
              description: "API key ID",
            },
          },
          required: ["key_id"],
        },
      },
      {
        name: "get_security_stats",
        description: "Get security statistics",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },

      // === ADVANCED FEATURES: Query Filtering (adv-b-007) ===
      {
        name: "query_tasks",
        description:
          "Query tasks with advanced filtering options (status, priority, tags, date range)",
        inputSchema: {
          type: "object",
          properties: {
            status: {
              type: "array",
              items: {
                type: "string",
                enum: ["available", "claimed", "in_progress", "done", "failed"],
              },
              description: "Filter by task statuses",
            },
            priority_min: {
              type: "number",
              description: "Minimum priority (1-10)",
            },
            priority_max: {
              type: "number",
              description: "Maximum priority (1-10)",
            },
            claimed_by: {
              type: "string",
              description: "Filter by agent ID that claimed the task",
            },
            tags: {
              type: "array",
              items: { type: "string" },
              description: "Filter by tags (any match)",
            },
            created_after: {
              type: "string",
              description: "Filter by creation date (ISO 8601)",
            },
            created_before: {
              type: "string",
              description: "Filter by creation date (ISO 8601)",
            },
            search: {
              type: "string",
              description: "Search in task description and ID",
            },
            page: {
              type: "number",
              description: "Page number (default: 1)",
            },
            page_size: {
              type: "number",
              description: "Items per page (default: 20, max: 100)",
            },
            sort_by: {
              type: "string",
              enum: ["created_at", "priority", "status"],
              description: "Sort field",
            },
            sort_direction: {
              type: "string",
              enum: ["asc", "desc"],
              description: "Sort direction",
            },
          },
        },
      },

      // === ADVANCED FEATURES: Batch Operations (adv-b-005) ===
      {
        name: "batch_operations",
        description:
          "Execute multiple operations in a single request (create, update, delete tasks)",
        inputSchema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "Agent ID performing the operations",
            },
            operations: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  operation: {
                    type: "string",
                    enum: [
                      "create_task",
                      "update_task",
                      "delete_task",
                      "claim_task",
                      "complete_task",
                    ],
                    description: "Operation type",
                  },
                  params: {
                    type: "object",
                    description: "Operation parameters",
                  },
                },
                required: ["operation", "params"],
              },
              description: "List of operations to execute",
            },
          },
          required: ["agent_id", "operations"],
        },
      },

      // === ADVANCED FEATURES: Transaction Support (adv-b-006) ===
      {
        name: "begin_transaction",
        description: "Begin a new transaction for atomic operations",
        inputSchema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "Agent ID starting the transaction",
            },
          },
          required: ["agent_id"],
        },
      },
      {
        name: "add_transaction_operation",
        description: "Add an operation to a pending transaction",
        inputSchema: {
          type: "object",
          properties: {
            transaction_id: {
              type: "string",
              description: "Transaction ID",
            },
            operation: {
              type: "object",
              properties: {
                type: {
                  type: "string",
                  enum: [
                    "create_task",
                    "update_task",
                    "delete_task",
                    "claim_task",
                    "complete_task",
                  ],
                  description: "Operation type",
                },
                params: {
                  type: "object",
                  description: "Operation parameters",
                },
              },
              required: ["type", "params"],
            },
          },
          required: ["transaction_id", "operation"],
        },
      },
      {
        name: "commit_transaction",
        description: "Commit a transaction and execute all operations",
        inputSchema: {
          type: "object",
          properties: {
            transaction_id: {
              type: "string",
              description: "Transaction ID to commit",
            },
          },
          required: ["transaction_id"],
        },
      },
      {
        name: "rollback_transaction",
        description: "Rollback a pending transaction",
        inputSchema: {
          type: "object",
          properties: {
            transaction_id: {
              type: "string",
              description: "Transaction ID to rollback",
            },
          },
          required: ["transaction_id"],
        },
      },

      // === ADVANCED FEATURES: Audit Log (adv-b-009) ===
      {
        name: "get_audit_log",
        description: "Get audit log entries with optional filtering",
        inputSchema: {
          type: "object",
          properties: {
            entity_type: {
              type: "string",
              enum: [
                "task",
                "agent",
                "discovery",
                "coordination",
                "transaction",
              ],
              description: "Filter by entity type",
            },
            entity_id: {
              type: "string",
              description: "Filter by entity ID",
            },
            agent_id: {
              type: "string",
              description: "Filter by agent ID",
            },
            action: {
              type: "string",
              description: "Filter by action type",
            },
            since: {
              type: "string",
              description: "Filter entries since this timestamp (ISO 8601)",
            },
            limit: {
              type: "number",
              description: "Maximum entries to return (default: 100)",
            },
          },
        },
      },

      // === ADVANCED FEATURES: Health Check (adv-b-010) ===
      {
        name: "health_check",
        description:
          "Get comprehensive health status of the coordination server",
        inputSchema: {
          type: "object",
          properties: {
            check_type: {
              type: "string",
              enum: ["full", "liveness", "readiness"],
              description: "Type of health check (default: full)",
            },
          },
        },
      },

      // === ADVANCED FEATURES: Rate Limiting (adv-b-003) ===
      {
        name: "check_rate_limit",
        description: "Check rate limit status for an agent",
        inputSchema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "Agent ID to check",
            },
          },
          required: ["agent_id"],
        },
      },

      // === ADVANCED FEATURES: Request Queue (adv-b-004) ===
      {
        name: "get_queue_stats",
        description: "Get request queue statistics",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },

      // === ADVANCED FEATURES: WebSocket Status (adv-b-001) ===
      {
        name: "get_websocket_status",
        description: "Get WebSocket transport status and connected clients",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
    ],
  };
});

// ============================================================================
// Tool Implementation
// ============================================================================

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      // === LEADER TOOLS ===
      case "init_coordination": {
        state.goal = args?.goal as string;
        state.master_plan = (args?.master_plan as string) || "";
        state.created_at = new Date().toISOString();
        updateActivity();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  success: true,
                  goal: state.goal,
                  created_at: state.created_at,
                },
                null,
                2,
              ),
            },
          ],
        };
      }

      case "create_task": {
        const task = createTask(
          args?.description as string,
          (args?.priority as number) || 5,
          (args?.dependencies as string[]) || [],
          {
            files: args?.context_files as string[],
            hints: args?.hints as string,
          },
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success: true, task }, null, 2),
            },
          ],
        };
      }

      case "create_tasks_batch": {
        const tasksInput = args?.tasks as any[] | undefined;
        if (!tasksInput || !Array.isArray(tasksInput)) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  error:
                    "Missing or invalid 'tasks' parameter. Expected an array of task objects.",
                }),
              },
            ],
          };
        }
        const tasks = tasksInput.map((t) =>
          createTask(t.description, t.priority || 5, t.dependencies || [], {
            files: t.context_files,
            hints: t.hints,
          }),
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  success: true,
                  created: tasks.length,
                  task_ids: tasks.map((t) => t.id),
                },
                null,
                2,
              ),
            },
          ],
        };
      }

      case "get_status": {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(getStatus(), null, 2),
            },
          ],
        };
      }

      case "get_all_tasks": {
        const filter = (args?.status_filter as string) || "all";
        let tasks = state.tasks;
        if (filter !== "all") {
          tasks = tasks.filter((t) => t.status === filter);
        }
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ tasks }, null, 2),
            },
          ],
        };
      }

      case "get_results": {
        const taskIds = args?.task_ids as string[];
        let tasks = state.tasks.filter((t) => t.status === "done");
        if (taskIds && taskIds.length > 0) {
          tasks = tasks.filter((t) => taskIds.includes(t.id));
        }
        const results = tasks.map((t) => ({
          task_id: t.id,
          description: t.description,
          result: t.result,
          completed_at: t.completed_at,
        }));
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ results }, null, 2),
            },
          ],
        };
      }

      // === WORKER TOOLS ===
      case "register_agent": {
        const agent = registerAgent(
          args?.agent_id as string,
          args?.role as "leader" | "worker",
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success: true, agent }, null, 2),
            },
          ],
        };
      }

      case "claim_task": {
        const task = claimTask(args?.agent_id as string);
        if (task) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({ success: true, task }, null, 2),
              },
            ],
          };
        }
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: false,
                message: "No available tasks with satisfied dependencies",
              }),
            },
          ],
        };
      }

      case "start_task": {
        const success = startTask(
          args?.agent_id as string,
          args?.task_id as string,
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success }),
            },
          ],
        };
      }

      case "complete_task": {
        const success = completeTask(
          args?.agent_id as string,
          args?.task_id as string,
          {
            output: args?.output as string,
            files_modified: args?.files_modified as string[],
            files_created: args?.files_created as string[],
          },
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success }),
            },
          ],
        };
      }

      case "fail_task": {
        const success = failTask(
          args?.agent_id as string,
          args?.task_id as string,
          args?.error as string,
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success }),
            },
          ],
        };
      }

      case "heartbeat": {
        const success = heartbeat(args?.agent_id as string);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success }),
            },
          ],
        };
      }

      // === SHARED TOOLS ===
      case "add_discovery": {
        const discovery = addDiscovery(
          args?.agent_id as string,
          args?.content as string,
          (args?.tags as string[]) || [],
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success: true, discovery }, null, 2),
            },
          ],
        };
      }

      case "get_discoveries": {
        let discoveries = state.discoveries;
        const tags = args?.tags as string[];
        if (tags && tags.length > 0) {
          discoveries = discoveries.filter((d) =>
            d.tags.some((t) => tags.includes(t)),
          );
        }
        const limit = (args?.limit as number) || 20;
        discoveries = discoveries.slice(-limit);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ discoveries }, null, 2),
            },
          ],
        };
      }

      case "get_master_plan": {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  goal: state.goal,
                  master_plan: state.master_plan,
                  created_at: state.created_at,
                },
                null,
                2,
              ),
            },
          ],
        };
      }

      // === SUBSCRIPTION TOOLS ===
      case "subscribe": {
        const filter: SubscriptionFilter = {
          event_types: args?.event_types as SubscriptionEventType[],
          task_status: args?.task_status as any[],
          task_tags: args?.task_tags as string[],
          agent_id: args?.agent_id as string,
          priority_min: args?.priority_min as number,
          priority_max: args?.priority_max as number,
        };
        const subscription = subscriptionManager.subscribe(
          args?.subscriber_id as string,
          filter,
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success: true, subscription }, null, 2),
            },
          ],
        };
      }

      case "unsubscribe": {
        const success = subscriptionManager.unsubscribe(
          args?.subscription_id as string,
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success }),
            },
          ],
        };
      }

      case "get_subscriptions": {
        const subscriptions = subscriptionManager.getSubscriberSubscriptions(
          args?.subscriber_id as string,
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ subscriptions }, null, 2),
            },
          ],
        };
      }

      case "configure_batching": {
        subscriptionManager.updateConfig({
          batching_enabled: args?.enabled as boolean,
          batch_interval_ms: args?.batch_interval_ms as number,
          max_batch_size: args?.max_batch_size as number,
        });
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: true,
                config: subscriptionManager.getConfig(),
              }),
            },
          ],
        };
      }

      // === WEBHOOK TOOLS ===
      case "register_webhook": {
        const webhook = webhookManager.registerWebhook(
          args?.url as string,
          args?.events as SubscriptionEventType[],
          {
            secret: args?.secret as string,
            headers: args?.headers as Record<string, string>,
          },
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success: true, webhook }, null, 2),
            },
          ],
        };
      }

      case "delete_webhook": {
        const success = webhookManager.deleteWebhook(
          args?.webhook_id as string,
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success }),
            },
          ],
        };
      }

      case "list_webhooks": {
        const webhooks = webhookManager.getAllWebhooks();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ webhooks }, null, 2),
            },
          ],
        };
      }

      case "get_webhook_deliveries": {
        const deliveries = webhookManager.getDeliveryHistory(
          args?.webhook_id as string,
          { limit: (args?.limit as number) || 50 },
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ deliveries }, null, 2),
            },
          ],
        };
      }

      // === SSE TOOLS ===
      case "get_sse_stats": {
        const stats = sseManager.getStats();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ stats }, null, 2),
            },
          ],
        };
      }

      case "get_event_history": {
        const events = sseManager.getEventHistory(
          (args?.limit as number) || 50,
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ events }, null, 2),
            },
          ],
        };
      }

      // === SECURITY TOOLS ===
      case "create_api_key": {
        const apiKey = securityManager.createAPIKey({
          name: args?.name as string,
          agent_id: args?.agent_id as string,
          role: args?.role as "leader" | "worker" | "admin",
          permissions: args?.permissions as Permission[],
          ip_allowlist: args?.ip_allowlist as string[],
          expires_in_days: args?.expires_in_days as number,
        });
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success: true, api_key: apiKey }, null, 2),
            },
          ],
        };
      }

      case "revoke_api_key": {
        const success = securityManager.revokeAPIKey(args?.key_id as string);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success }),
            },
          ],
        };
      }

      case "list_api_keys": {
        const apiKeys = securityManager.getAllAPIKeys();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ api_keys: apiKeys }, null, 2),
            },
          ],
        };
      }

      case "generate_jwt": {
        const token = securityManager.generateJWT(
          args?.subject as string,
          args?.role as "leader" | "worker" | "admin",
          (args?.permissions as Permission[]) || [],
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success: true, token }, null, 2),
            },
          ],
        };
      }

      case "validate_jwt": {
        const result = securityManager.validateJWT(args?.token as string);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case "configure_security": {
        securityManager.updateConfig({
          api_key_enabled: args?.api_key_enabled as boolean,
          jwt_enabled: args?.jwt_enabled as boolean,
          request_signing_enabled: args?.request_signing_enabled as boolean,
          ip_allowlist_enabled: args?.ip_allowlist_enabled as boolean,
          rate_limiting_enabled: args?.rate_limiting_enabled as boolean,
        });
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                success: true,
                config: securityManager.getConfig(),
              }),
            },
          ],
        };
      }

      case "add_ip_to_allowlist": {
        securityManager.addToGlobalAllowlist(args?.ip as string);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success: true }),
            },
          ],
        };
      }

      case "add_ip_to_denylist": {
        securityManager.addToGlobalDenylist(args?.ip as string);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success: true }),
            },
          ],
        };
      }

      case "get_rate_limit_status": {
        const status = securityManager.getRateLimitStatus(
          args?.key_id as string,
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ status }, null, 2),
            },
          ],
        };
      }

      case "get_security_stats": {
        const stats = securityManager.getStats();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ stats }, null, 2),
            },
          ],
        };
      }

      // === ADVANCED FEATURES: Query Filtering (adv-b-007) ===
      case "query_tasks": {
        const taskFilter: TaskFilter = {
          status: args?.status as Task["status"][],
          priority_min: args?.priority_min as number,
          priority_max: args?.priority_max as number,
          claimed_by: args?.claimed_by as string,
          tags: args?.tags as string[],
          created_after: args?.created_after as string,
          created_before: args?.created_before as string,
          search: args?.search as string,
        };
        const paginationOpts: PaginationOptions = {
          page: (args?.page as number) || 1,
          page_size: (args?.page_size as number) || 20,
          sort_by: args?.sort_by as string,
          sort_direction: args?.sort_direction as "asc" | "desc",
        };

        const filteredTasks = filterTasks(taskFilter);
        const result = paginateItems(filteredTasks, paginationOpts);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      // === ADVANCED FEATURES: Batch Operations (adv-b-005) ===
      case "batch_operations": {
        const agentId = args?.agent_id as string;
        const operations = args?.operations as BatchOperation[];

        if (!agentId || !operations || !Array.isArray(operations)) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  error: "Missing required parameters: agent_id and operations",
                }),
              },
            ],
          };
        }

        const results = executeBatchOperations(operations, agentId);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  success: results.every((r) => r.success),
                  results,
                  total_operations: operations.length,
                  successful: results.filter((r) => r.success).length,
                  failed: results.filter((r) => !r.success).length,
                },
                null,
                2,
              ),
            },
          ],
        };
      }

      // === ADVANCED FEATURES: Transaction Support (adv-b-006) ===
      case "begin_transaction": {
        const transaction = beginTransaction(args?.agent_id as string);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                { success: true, transaction_id: transaction.id, transaction },
                null,
                2,
              ),
            },
          ],
        };
      }

      case "add_transaction_operation": {
        const op = args?.operation as TransactionOperation;
        const success = addTransactionOperation(
          args?.transaction_id as string,
          op,
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success }),
            },
          ],
        };
      }

      case "commit_transaction": {
        const commitResult = commitTransaction(args?.transaction_id as string);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(commitResult, null, 2),
            },
          ],
        };
      }

      case "rollback_transaction": {
        const success = rollbackTransaction(args?.transaction_id as string);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ success }),
            },
          ],
        };
      }

      // === ADVANCED FEATURES: Audit Log (adv-b-009) ===
      case "get_audit_log": {
        const auditFilter = {
          entity_type: args?.entity_type as string,
          entity_id: args?.entity_id as string,
          agent_id: args?.agent_id as string,
          action: args?.action as string,
          since: args?.since as string,
        };
        const limit = (args?.limit as number) || 100;
        const entries = getAuditLog(auditFilter, limit);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ entries, count: entries.length }, null, 2),
            },
          ],
        };
      }

      // === ADVANCED FEATURES: Health Check (adv-b-010) ===
      case "health_check": {
        const checkType = (args?.check_type as string) || "full";
        const healthContext = {
          stateDir: STATE_DIR,
          tasks: state.tasks,
          agents: Array.from(state.agents.values()),
          lastActivity: state.last_activity,
        };

        let result;
        switch (checkType) {
          case "liveness":
            result = livenessCheck();
            break;
          case "readiness":
            result = readinessCheck(healthContext);
            break;
          default:
            result = getHealthStatus(healthContext);
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      // === ADVANCED FEATURES: Rate Limiting (adv-b-003) ===
      case "check_rate_limit": {
        if (ENABLE_RATE_LIMITING) {
          const agentId = args?.agent_id as string;
          const limitResult = rateLimiter.checkLimit(agentId);
          const status = rateLimiter.getStatus(agentId);
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(
                  {
                    agent_id: agentId,
                    ...limitResult,
                    ...status,
                    rate_limiting_enabled: true,
                  },
                  null,
                  2,
                ),
              },
            ],
          };
        }
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                rate_limiting_enabled: false,
                message: "Rate limiting is disabled",
              }),
            },
          ],
        };
      }

      // === ADVANCED FEATURES: Request Queue (adv-b-004) ===
      case "get_queue_stats": {
        const queueStats = requestQueue.getStats();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  ...queueStats,
                  is_processing: requestQueue.isProcessing(),
                },
                null,
                2,
              ),
            },
          ],
        };
      }

      // === ADVANCED FEATURES: WebSocket Status (adv-b-001) ===
      case "get_websocket_status": {
        if (wsTransport && wsTransport.isRunning()) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(
                  {
                    enabled: true,
                    running: true,
                    port: WEBSOCKET_PORT,
                    connected_clients: wsTransport.getClientCount(),
                    connected_agents: wsTransport.getConnectedAgents(),
                  },
                  null,
                  2,
                ),
              },
            ],
          };
        }
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                enabled: ENABLE_WEBSOCKET,
                running: false,
                message: ENABLE_WEBSOCKET
                  ? "WebSocket server not started"
                  : "WebSocket transport is disabled",
              }),
            },
          ],
        };
      }

      default:
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ error: `Unknown tool: ${name}` }),
            },
          ],
        };
    }
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
          }),
        },
      ],
    };
  }
});

// ============================================================================
// Resources (for reading state)
// ============================================================================

server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: "coordination://status",
        name: "Coordination Status",
        description: "Current coordination status and progress",
        mimeType: "application/json",
      },
      {
        uri: "coordination://tasks",
        name: "All Tasks",
        description: "All tasks in the queue",
        mimeType: "application/json",
      },
      {
        uri: "coordination://discoveries",
        name: "Discoveries",
        description: "Shared discoveries from all agents",
        mimeType: "application/json",
      },
      {
        uri: "coordination://master-plan",
        name: "Master Plan",
        description: "The master plan and goal",
        mimeType: "application/json",
      },
    ],
  };
});

server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const uri = request.params.uri;

  switch (uri) {
    case "coordination://status":
      return {
        contents: [
          {
            uri,
            mimeType: "application/json",
            text: JSON.stringify(getStatus(), null, 2),
          },
        ],
      };
    case "coordination://tasks":
      return {
        contents: [
          {
            uri,
            mimeType: "application/json",
            text: JSON.stringify({ tasks: state.tasks }, null, 2),
          },
        ],
      };
    case "coordination://discoveries":
      return {
        contents: [
          {
            uri,
            mimeType: "application/json",
            text: JSON.stringify({ discoveries: state.discoveries }, null, 2),
          },
        ],
      };
    case "coordination://master-plan":
      return {
        contents: [
          {
            uri,
            mimeType: "application/json",
            text: JSON.stringify(
              {
                goal: state.goal,
                master_plan: state.master_plan,
                created_at: state.created_at,
              },
              null,
              2,
            ),
          },
        ],
      };
    default:
      throw new Error(`Unknown resource: ${uri}`);
  }
});

// ============================================================================
// Main
// ============================================================================

async function main() {
  // Load any existing state from disk
  loadState();

  // Initialize WebSocket transport if enabled (adv-b-001)
  if (ENABLE_WEBSOCKET) {
    try {
      wsTransport = createWebSocketTransport(WEBSOCKET_PORT);
      await wsTransport.start();
      console.error(`WebSocket transport started on port ${WEBSOCKET_PORT}`);
    } catch (error) {
      console.error(`Failed to start WebSocket transport:`, error);
    }
  }

  // Log advanced features status
  console.error(`Advanced features status:`);
  console.error(
    `  - WebSocket transport: ${ENABLE_WEBSOCKET ? "enabled" : "disabled"}`,
  );
  console.error(
    `  - Rate limiting: ${ENABLE_RATE_LIMITING ? "enabled" : "disabled"}`,
  );
  console.error(`  - Audit logging: enabled`);
  console.error(`  - Batch operations: enabled`);
  console.error(`  - Transaction support: enabled`);
  console.error(`  - Query filtering: enabled`);
  console.error(`  - Pagination: enabled`);
  console.error(`  - Health check: enabled`);

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`Claude Coordination MCP Server running on stdio`);
  console.error(`State file: ${STATE_FILE}`);

  // Graceful shutdown handler (PROD-008)
  const SHUTDOWN_TIMEOUT_MS = 30000; // 30 seconds
  let isShuttingDown = false;

  async function gracefulShutdown(signal: string) {
    if (isShuttingDown) {
      console.error(`Already shutting down, ignoring ${signal}`);
      return;
    }

    isShuttingDown = true;
    const shutdownStart = Date.now();
    console.error(
      `\n[${new Date().toISOString()}] Received ${signal}, initiating graceful shutdown...`,
    );

    // Track active requests by checking in-progress tasks
    const activeTaskCount = state.tasks.filter(
      (t) => t.status === "claimed" || t.status === "in_progress",
    ).length;

    if (activeTaskCount > 0) {
      console.error(
        `[${new Date().toISOString()}] Waiting for ${activeTaskCount} active tasks to complete...`,
      );

      // Wait for active tasks with timeout
      const waitInterval = setInterval(() => {
        const remaining = state.tasks.filter(
          (t) => t.status === "claimed" || t.status === "in_progress",
        ).length;

        const elapsed = Date.now() - shutdownStart;
        if (remaining === 0) {
          console.error(`[${new Date().toISOString()}] All tasks completed`);
          clearInterval(waitInterval);
        } else if (elapsed >= SHUTDOWN_TIMEOUT_MS) {
          console.error(
            `[${new Date().toISOString()}] Timeout reached with ${remaining} tasks still active, forcing shutdown`,
          );
          clearInterval(waitInterval);
        }
      }, 500);

      // Wait up to timeout
      await new Promise<void>((resolve) => {
        const checkComplete = setInterval(() => {
          const remaining = state.tasks.filter(
            (t) => t.status === "claimed" || t.status === "in_progress",
          ).length;
          const elapsed = Date.now() - shutdownStart;

          if (remaining === 0 || elapsed >= SHUTDOWN_TIMEOUT_MS) {
            clearInterval(checkComplete);
            resolve();
          }
        }, 100);
      });
    }

    // Save state before shutdown
    console.error(
      `[${new Date().toISOString()}] Persisting state to ${STATE_FILE}...`,
    );
    try {
      saveState();
      console.error(`[${new Date().toISOString()}] State saved successfully`);
    } catch (error) {
      console.error(
        `[${new Date().toISOString()}] Failed to save state:`,
        error,
      );
    }

    // Stop WebSocket transport
    if (wsTransport) {
      console.error(
        `[${new Date().toISOString()}] Stopping WebSocket transport...`,
      );
      try {
        await wsTransport.stop();
        console.error(
          `[${new Date().toISOString()}] WebSocket transport stopped`,
        );
      } catch (error) {
        console.error(
          `[${new Date().toISOString()}] Error stopping WebSocket transport:`,
          error,
        );
      }
    }

    // Stop request queue
    if (requestQueue) {
      console.error(`[${new Date().toISOString()}] Stopping request queue...`);
      requestQueue.stopProcessing();
    }

    const shutdownDuration = (Date.now() - shutdownStart) / 1000;
    console.error(
      `[${new Date().toISOString()}] Graceful shutdown completed in ${shutdownDuration.toFixed(2)}s`,
    );

    process.exit(0);
  }

  process.on("SIGINT", () => gracefulShutdown("SIGINT"));
  process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
}

main().catch(console.error);
