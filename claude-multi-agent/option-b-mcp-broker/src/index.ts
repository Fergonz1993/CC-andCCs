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

// ============================================================================
// State Persistence Configuration
// ============================================================================

const STATE_DIR = process.env.COORDINATION_DIR || ".coordination";
const STATE_FILE = path.join(STATE_DIR, "mcp-state.json");

// Serializable version of state (Map -> Record for JSON)
interface SerializableState {
  master_plan: string;
  goal: string;
  tasks: Task[];
  agents: Record<string, Agent>;
  discoveries: Discovery[];
  created_at: string;
  last_activity: string;
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

      console.error(
        `Loaded state from ${STATE_FILE}: ${state.tasks.length} tasks, ${state.agents.size} agents`,
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
};

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
  return task;
}

function startTask(agentId: string, taskId: string): boolean {
  const task = state.tasks.find((t) => t.id === taskId);
  if (!task || task.claimed_by !== agentId) return false;

  task.status = "in_progress";
  updateActivity();
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

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`Claude Coordination MCP Server running on stdio`);
  console.error(`State file: ${STATE_FILE}`);
}

main().catch(console.error);
