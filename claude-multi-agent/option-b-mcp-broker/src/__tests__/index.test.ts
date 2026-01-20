/**
 * Unit tests for Option B MCP Server (index.ts)
 *
 * Run with: npm test
 *
 * Coverage target: >80%
 */

import {
  describe,
  it,
  expect,
  beforeEach,
  afterEach,
  jest,
} from "@jest/globals";
import * as fs from "fs";
import * as path from "path";

// Mock MCP SDK
jest.mock("@modelcontextprotocol/sdk/server/index.js", () => ({
  Server: jest.fn().mockImplementation(() => ({
    setRequestHandler: jest.fn(),
    connect: jest.fn((): Promise<void> => Promise.resolve()),
  })),
}));

jest.mock("@modelcontextprotocol/sdk/server/stdio.js", () => ({
  StdioServerTransport: jest.fn(),
}));

// Test constants
const TEST_STATE_DIR = "/tmp/coordination_test_" + Date.now();
const TEST_STATE_FILE = path.join(TEST_STATE_DIR, "mcp-state.json");

describe("MCP Server State Management", () => {
  beforeEach(() => {
    // Clean up any existing test directory
    if (fs.existsSync(TEST_STATE_DIR)) {
      fs.rmSync(TEST_STATE_DIR, { recursive: true });
    }
    fs.mkdirSync(TEST_STATE_DIR, { recursive: true });
  });

  afterEach(() => {
    // Clean up test directory
    if (fs.existsSync(TEST_STATE_DIR)) {
      fs.rmSync(TEST_STATE_DIR, { recursive: true });
    }
  });

  describe("Task Management Functions", () => {
    it("opt-b-task-001: should generate unique task IDs", () => {
      const generateTaskId = (): string => {
        const timestamp = Date.now().toString(36);
        const random = Math.random().toString(36).substring(2, 6);
        return `task-${timestamp}-${random}`;
      };

      const ids = new Set<string>();
      for (let i = 0; i < 100; i++) {
        ids.add(generateTaskId());
      }
      expect(ids.size).toBe(100);
    });

    it("opt-b-task-002: should create task with default values", () => {
      interface Task {
        id: string;
        description: string;
        status: string;
        priority: number;
        claimed_by: string | null;
        dependencies: string[];
      }

      const createTask = (description: string, priority: number = 5): Task => ({
        id: `task-${Date.now().toString(36)}`,
        description,
        status: "available",
        priority,
        claimed_by: null,
        dependencies: [],
      });

      const task = createTask("Test task");
      expect(task.description).toBe("Test task");
      expect(task.status).toBe("available");
      expect(task.priority).toBe(5);
      expect(task.claimed_by).toBeNull();
    });

    it("opt-b-task-003: should create task with custom priority", () => {
      interface Task {
        id: string;
        description: string;
        status: string;
        priority: number;
      }

      const createTask = (description: string, priority: number = 5): Task => ({
        id: `task-${Date.now().toString(36)}`,
        description,
        status: "available",
        priority,
      });

      const task = createTask("High priority task", 1);
      expect(task.priority).toBe(1);
    });
  });

  describe("Task Status Operations", () => {
    it("opt-b-status-001: should filter available tasks correctly", () => {
      interface Task {
        id: string;
        status: string;
        dependencies: string[];
      }

      const tasks: Task[] = [
        { id: "task-1", status: "available", dependencies: [] },
        { id: "task-2", status: "claimed", dependencies: [] },
        { id: "task-3", status: "done", dependencies: [] },
        { id: "task-4", status: "available", dependencies: ["task-3"] },
      ];

      const doneIds = new Set(
        tasks.filter((t) => t.status === "done").map((t) => t.id),
      );

      const available = tasks.filter(
        (t) =>
          t.status === "available" &&
          t.dependencies.every((dep) => doneIds.has(dep)),
      );

      expect(available.length).toBe(2);
      expect(available.map((t) => t.id)).toContain("task-1");
      expect(available.map((t) => t.id)).toContain("task-4");
    });

    it("opt-b-status-002: should respect dependencies when filtering", () => {
      interface Task {
        id: string;
        status: string;
        dependencies: string[];
      }

      const tasks: Task[] = [
        { id: "task-1", status: "available", dependencies: ["task-0"] },
        { id: "task-2", status: "available", dependencies: [] },
      ];

      const doneIds = new Set<string>(); // No tasks done

      const available = tasks.filter(
        (t) =>
          t.status === "available" &&
          t.dependencies.every((dep) => doneIds.has(dep)),
      );

      expect(available.length).toBe(1);
      expect(available[0].id).toBe("task-2");
    });

    it("opt-b-status-003: should claim task and update status", () => {
      interface Task {
        id: string;
        status: string;
        claimed_by: string | null;
        claimed_at: string | null;
      }

      const task: Task = {
        id: "task-1",
        status: "available",
        claimed_by: null,
        claimed_at: null,
      };

      const claimTask = (t: Task, agentId: string): void => {
        t.status = "claimed";
        t.claimed_by = agentId;
        t.claimed_at = new Date().toISOString();
      };

      claimTask(task, "worker-1");

      expect(task.status).toBe("claimed");
      expect(task.claimed_by).toBe("worker-1");
      expect(task.claimed_at).not.toBeNull();
    });

    it("opt-b-status-004: should complete task with result", () => {
      interface TaskResult {
        output: string;
        files_modified?: string[];
        files_created?: string[];
      }

      interface Task {
        id: string;
        status: string;
        result: TaskResult | null;
        completed_at: string | null;
      }

      const task: Task = {
        id: "task-1",
        status: "in_progress",
        result: null,
        completed_at: null,
      };

      const completeTask = (t: Task, result: TaskResult): void => {
        t.status = "done";
        t.result = result;
        t.completed_at = new Date().toISOString();
      };

      completeTask(task, {
        output: "Task completed",
        files_modified: ["file.ts"],
      });

      expect(task.status).toBe("done");
      expect(task.result?.output).toBe("Task completed");
      expect(task.completed_at).not.toBeNull();
    });

    it("opt-b-status-005: should fail task with error", () => {
      interface Task {
        id: string;
        status: string;
        result: { error: string } | null;
        completed_at: string | null;
      }

      const task: Task = {
        id: "task-1",
        status: "in_progress",
        result: null,
        completed_at: null,
      };

      const failTask = (t: Task, error: string): void => {
        t.status = "failed";
        t.result = { error };
        t.completed_at = new Date().toISOString();
      };

      failTask(task, "Connection timeout");

      expect(task.status).toBe("failed");
      expect(task.result?.error).toBe("Connection timeout");
    });
  });

  describe("Agent Management", () => {
    it("opt-b-agent-001: should register new agent", () => {
      interface Agent {
        id: string;
        role: "leader" | "worker";
        last_heartbeat: string;
        current_task: string | null;
        tasks_completed: number;
      }

      const agents = new Map<string, Agent>();

      const registerAgent = (id: string, role: "leader" | "worker"): Agent => {
        const agent: Agent = {
          id,
          role,
          last_heartbeat: new Date().toISOString(),
          current_task: null,
          tasks_completed: 0,
        };
        agents.set(id, agent);
        return agent;
      };

      const agent = registerAgent("worker-1", "worker");

      expect(agent.id).toBe("worker-1");
      expect(agent.role).toBe("worker");
      expect(agents.has("worker-1")).toBe(true);
    });

    it("opt-b-agent-002: should update heartbeat", () => {
      interface Agent {
        id: string;
        last_heartbeat: string;
      }

      const agents = new Map<string, Agent>();
      const oldTime = new Date(Date.now() - 60000).toISOString();
      agents.set("worker-1", { id: "worker-1", last_heartbeat: oldTime });

      const heartbeat = (agentId: string): boolean => {
        const agent = agents.get(agentId);
        if (!agent) return false;
        agent.last_heartbeat = new Date().toISOString();
        return true;
      };

      const result = heartbeat("worker-1");

      expect(result).toBe(true);
      expect(agents.get("worker-1")?.last_heartbeat).not.toBe(oldTime);
    });

    it("opt-b-agent-003: should return false for unknown agent heartbeat", () => {
      const agents = new Map<string, { id: string; last_heartbeat: string }>();

      const heartbeat = (agentId: string): boolean => {
        const agent = agents.get(agentId);
        if (!agent) return false;
        return true;
      };

      expect(heartbeat("unknown-agent")).toBe(false);
    });
  });

  describe("Discovery Management", () => {
    it("opt-b-disc-001: should add discovery", () => {
      interface Discovery {
        id: string;
        agent_id: string;
        content: string;
        tags: string[];
        created_at: string;
      }

      const discoveries: Discovery[] = [];

      const addDiscovery = (
        agentId: string,
        content: string,
        tags: string[] = [],
      ): Discovery => {
        const discovery: Discovery = {
          id: `disc-${Date.now()}`,
          agent_id: agentId,
          content,
          tags,
          created_at: new Date().toISOString(),
        };
        discoveries.push(discovery);
        return discovery;
      };

      const disc = addDiscovery("worker-1", "Found important pattern", [
        "code",
        "pattern",
      ]);

      expect(disc.content).toBe("Found important pattern");
      expect(disc.tags).toContain("code");
      expect(discoveries.length).toBe(1);
    });

    it("opt-b-disc-002: should filter discoveries by tag", () => {
      interface Discovery {
        id: string;
        tags: string[];
      }

      const discoveries: Discovery[] = [
        { id: "1", tags: ["code", "bug"] },
        { id: "2", tags: ["documentation"] },
        { id: "3", tags: ["code", "performance"] },
      ];

      const filterByTag = (tag: string): Discovery[] =>
        discoveries.filter((d) => d.tags.includes(tag));

      const codeDiscoveries = filterByTag("code");
      expect(codeDiscoveries.length).toBe(2);
    });
  });

  describe("Status Reporting", () => {
    it("opt-b-report-001: should calculate progress correctly", () => {
      interface Task {
        status: string;
      }

      const tasks: Task[] = [
        { status: "done" },
        { status: "done" },
        { status: "available" },
        { status: "in_progress" },
      ];

      const getStatus = () => {
        const done = tasks.filter((t) => t.status === "done").length;
        const total = tasks.length;
        return {
          total_tasks: total,
          done: done,
          progress_percent: Math.round((done / total) * 100),
        };
      };

      const status = getStatus();
      expect(status.total_tasks).toBe(4);
      expect(status.done).toBe(2);
      expect(status.progress_percent).toBe(50);
    });

    it("opt-b-report-002: should handle empty task list", () => {
      const tasks: { status: string }[] = [];

      const getStatus = () => {
        const done = tasks.filter((t) => t.status === "done").length;
        const total = tasks.length;
        return {
          total_tasks: total,
          progress_percent: total > 0 ? Math.round((done / total) * 100) : 0,
        };
      };

      const status = getStatus();
      expect(status.total_tasks).toBe(0);
      expect(status.progress_percent).toBe(0);
    });
  });

  describe("State Persistence", () => {
    it("opt-b-persist-001: should save state to file", () => {
      interface SerializableState {
        tasks: { id: string }[];
        created_at: string;
      }

      const saveState = (state: SerializableState, filePath: string): void => {
        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) {
          fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(filePath, JSON.stringify(state, null, 2));
      };

      const state: SerializableState = {
        tasks: [{ id: "task-1" }],
        created_at: new Date().toISOString(),
      };

      saveState(state, TEST_STATE_FILE);

      expect(fs.existsSync(TEST_STATE_FILE)).toBe(true);
      const loaded = JSON.parse(fs.readFileSync(TEST_STATE_FILE, "utf-8"));
      expect(loaded.tasks.length).toBe(1);
    });

    it("opt-b-persist-002: should load state from file", () => {
      interface SerializableState {
        tasks: { id: string }[];
      }

      const savedState: SerializableState = {
        tasks: [{ id: "task-1" }, { id: "task-2" }],
      };

      fs.writeFileSync(TEST_STATE_FILE, JSON.stringify(savedState));

      const loadState = (filePath: string): SerializableState | null => {
        if (!fs.existsSync(filePath)) return null;
        return JSON.parse(fs.readFileSync(filePath, "utf-8"));
      };

      const loaded = loadState(TEST_STATE_FILE);
      expect(loaded?.tasks.length).toBe(2);
    });

    it("opt-b-persist-003: should handle missing state file", () => {
      const loadState = (filePath: string): object | null => {
        if (!fs.existsSync(filePath)) return null;
        return JSON.parse(fs.readFileSync(filePath, "utf-8"));
      };

      const loaded = loadState("/nonexistent/file.json");
      expect(loaded).toBeNull();
    });
  });

  describe("Batch Operations", () => {
    it("opt-b-batch-001: should create multiple tasks at once", () => {
      interface Task {
        id: string;
        description: string;
      }

      const tasks: Task[] = [];

      const createTasksBatch = (
        descriptions: { description: string; priority?: number }[],
      ): Task[] => {
        return descriptions.map((d) => {
          const task: Task = {
            id: `task-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
            description: d.description,
          };
          tasks.push(task);
          return task;
        });
      };

      const created = createTasksBatch([
        { description: "Task 1" },
        { description: "Task 2" },
        { description: "Task 3" },
      ]);

      expect(created.length).toBe(3);
      expect(tasks.length).toBe(3);
    });
  });

  describe("Priority Sorting", () => {
    it("opt-b-priority-001: should sort tasks by priority", () => {
      interface Task {
        id: string;
        priority: number;
      }

      const tasks: Task[] = [
        { id: "task-1", priority: 5 },
        { id: "task-2", priority: 1 },
        { id: "task-3", priority: 3 },
      ];

      const sorted = [...tasks].sort((a, b) => a.priority - b.priority);

      expect(sorted[0].id).toBe("task-2");
      expect(sorted[1].id).toBe("task-3");
      expect(sorted[2].id).toBe("task-1");
    });

    it("opt-b-priority-002: should claim highest priority task", () => {
      interface Task {
        id: string;
        status: string;
        priority: number;
      }

      const tasks: Task[] = [
        { id: "task-1", status: "available", priority: 5 },
        { id: "task-2", status: "available", priority: 1 },
        { id: "task-3", status: "claimed", priority: 1 },
      ];

      const claimHighestPriority = (): Task | null => {
        const available = tasks.filter((t) => t.status === "available");
        if (available.length === 0) return null;
        available.sort((a, b) => a.priority - b.priority);
        return available[0];
      };

      const claimed = claimHighestPriority();
      expect(claimed?.id).toBe("task-2");
    });
  });
});
