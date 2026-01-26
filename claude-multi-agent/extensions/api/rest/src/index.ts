import express, { Request, Response, NextFunction } from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import rateLimit from "express-rate-limit";
import swaggerUi from "swagger-ui-express";
import * as fs from "fs";
import * as path from "path";
import { v4 as uuidv4 } from "uuid";
import { config } from "dotenv";

config();

const isProduction = process.env.NODE_ENV === "production";
const coordinationApiKey = process.env.COORDINATION_API_KEY || "";
const corsOrigins = (process.env.CORS_ALLOWED_ORIGINS || "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

if (isProduction && !coordinationApiKey) {
  throw new Error("COORDINATION_API_KEY is required in production");
}
if (isProduction && corsOrigins.length === 0) {
  throw new Error("CORS_ALLOWED_ORIGINS is required in production");
}

function extractApiKey(req: Request): string | undefined {
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.toLowerCase().startsWith("bearer ")) {
    return authHeader.slice(7).trim();
  }
  const headerKey = req.headers["x-api-key"];
  if (typeof headerKey === "string" && headerKey.trim()) {
    return headerKey.trim();
  }
  return undefined;
}

function requireAuth(req: Request, res: Response, next: NextFunction): void {
  if (!coordinationApiKey) {
    next();
    return;
  }
  if (req.path === "/health") {
    next();
    return;
  }
  const apiKey = extractApiKey(req);
  if (!apiKey || apiKey !== coordinationApiKey) {
    res.status(401).json({ success: false, error: "Unauthorized" });
    return;
  }
  next();
}

// Interfaces
interface Task {
  id: string;
  description: string;
  status: "available" | "claimed" | "in_progress" | "done" | "failed";
  priority: number;
  claimed_by?: string;
  assigned_to?: string;
  dependencies?: string[];
  created_at?: string;
  updated_at?: string;
  result?: string;
  error?: string;
  tags?: string[];
}

interface Worker {
  id: string;
  status: "idle" | "busy" | "offline";
  current_task?: string;
  last_heartbeat?: string;
  tasks_completed?: number;
  capabilities?: string[];
}

interface Discovery {
  id: string;
  title: string;
  content: string;
  created_by: string;
  created_at: string;
  tags?: string[];
}

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  meta?: {
    total?: number;
    page?: number;
    limit?: number;
  };
}

// Coordination Service
class CoordinationService {
  private coordinationDir: string;

  constructor(coordinationDir: string) {
    this.coordinationDir = coordinationDir;
  }

  private ensureDir(dirPath: string): void {
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }
  }

  private writeJsonAtomic(filePath: string, data: unknown): void {
    this.ensureDir(path.dirname(filePath));
    const tempPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
    fs.writeFileSync(tempPath, JSON.stringify(data, null, 2));
    fs.renameSync(tempPath, filePath);
  }

  private normalizeTask(task: Task): Task {
    // Preserve both assigned_to and claimed_by for compatibility with file-based orchestrator
    const claimed_by = task.claimed_by ?? task.assigned_to ?? null;
    return { ...task, claimed_by, assigned_to: claimed_by };
  }

  // Tasks
  getTasks(): Task[] {
    const tasksPath = path.join(this.coordinationDir, "tasks.json");
    if (!fs.existsSync(tasksPath)) {
      return [];
    }
    try {
      const content = fs.readFileSync(tasksPath, "utf-8");
      const tasks = JSON.parse(content).tasks || [];
      const normalized = tasks.map((task: Task) => this.normalizeTask(task));
      return normalized.map((task: Task) => ({
        ...task,
        assigned_to: task.claimed_by ?? task.assigned_to,
      }));
    } catch {
      return [];
    }
  }

  saveTasks(tasks: Task[]): void {
    this.ensureDir(this.coordinationDir);
    const tasksPath = path.join(this.coordinationDir, "tasks.json");
    const normalized = tasks.map((task) => this.normalizeTask(task));
    this.writeJsonAtomic(tasksPath, {
      tasks: normalized,
      updated_at: new Date().toISOString(),
    });
  }

  getTask(id: string): Task | undefined {
    return this.getTasks().find((t) => t.id === id);
  }

  createTask(data: Partial<Task>): Task {
    const tasks = this.getTasks();
    const task: Task = {
      id: data.id || `task-${uuidv4()}`,
      description: data.description || "",
      status: data.status || "available",
      priority: data.priority || 3,
      dependencies: data.dependencies || [],
      tags: data.tags || [],
      created_at: new Date().toISOString(),
    };
    tasks.push(task);
    this.saveTasks(tasks);
    return task;
  }

  updateTask(id: string, updates: Partial<Task>): Task | null {
    const tasks = this.getTasks();
    const index = tasks.findIndex((t) => t.id === id);
    if (index === -1) return null;

    tasks[index] = {
      ...tasks[index],
      ...updates,
      claimed_by:
        updates.claimed_by ?? updates.assigned_to ?? tasks[index].claimed_by,
      updated_at: new Date().toISOString(),
    };
    this.saveTasks(tasks);
    return tasks[index];
  }

  deleteTask(id: string): boolean {
    const tasks = this.getTasks();
    const index = tasks.findIndex((t) => t.id === id);
    if (index === -1) return false;

    tasks.splice(index, 1);
    this.saveTasks(tasks);
    return true;
  }

  claimTask(id: string, workerId: string): Task | null {
    const task = this.getTask(id);
    if (!task || task.status !== "available") return null;

    return this.updateTask(id, {
      status: "claimed",
      claimed_by: workerId,
    });
  }

  completeTask(id: string, result: string): Task | null {
    return this.updateTask(id, {
      status: "done",
      result,
    });
  }

  failTask(id: string, error: string): Task | null {
    return this.updateTask(id, {
      status: "failed",
      error,
    });
  }

  // Workers
  getWorkers(): Worker[] {
    const agentsPath = path.join(this.coordinationDir, "agents.json");
    const workersPath = path.join(this.coordinationDir, "workers.json");
    const filePath = fs.existsSync(agentsPath) ? agentsPath : workersPath;
    if (!fs.existsSync(filePath)) {
      return [];
    }
    try {
      const content = fs.readFileSync(filePath, "utf-8");
      const parsed = JSON.parse(content);
      return parsed.agents || parsed.workers || [];
    } catch {
      return [];
    }
  }

  saveWorkers(workers: Worker[]): void {
    this.ensureDir(this.coordinationDir);
    const agentsPath = path.join(this.coordinationDir, "agents.json");
    this.writeJsonAtomic(agentsPath, { agents: workers });
  }

  registerWorker(data: Partial<Worker>): Worker {
    const workers = this.getWorkers();
    const worker: Worker = {
      id: data.id || `worker-${uuidv4()}`,
      status: data.status || "idle",
      last_heartbeat: new Date().toISOString(),
      tasks_completed: 0,
      capabilities: data.capabilities || [],
    };
    workers.push(worker);
    this.saveWorkers(workers);
    return worker;
  }

  updateWorker(id: string, updates: Partial<Worker>): Worker | null {
    const workers = this.getWorkers();
    const index = workers.findIndex((w) => w.id === id);
    if (index === -1) return null;

    workers[index] = { ...workers[index], ...updates };
    this.saveWorkers(workers);
    return workers[index];
  }

  heartbeat(id: string): Worker | null {
    return this.updateWorker(id, {
      last_heartbeat: new Date().toISOString(),
    });
  }

  // Discoveries
  getDiscoveries(): Discovery[] {
    const discoveriesPath = path.join(this.coordinationDir, "discoveries.json");
    if (!fs.existsSync(discoveriesPath)) {
      return [];
    }
    try {
      const content = fs.readFileSync(discoveriesPath, "utf-8");
      return JSON.parse(content).discoveries || [];
    } catch {
      return [];
    }
  }

  saveDiscoveries(discoveries: Discovery[]): void {
    this.ensureDir(this.coordinationDir);
    const discoveriesPath = path.join(this.coordinationDir, "discoveries.json");
    fs.writeFileSync(discoveriesPath, JSON.stringify({ discoveries }, null, 2));
  }

  addDiscovery(data: Partial<Discovery>): Discovery {
    const discoveries = this.getDiscoveries();
    const discovery: Discovery = {
      id: data.id || `discovery-${uuidv4()}`,
      title: data.title || "",
      content: data.content || "",
      created_by: data.created_by || "api",
      created_at: new Date().toISOString(),
      tags: data.tags || [],
    };
    discoveries.push(discovery);
    this.saveDiscoveries(discoveries);
    return discovery;
  }

  // Status
  getStatus(): any {
    const tasks = this.getTasks();
    const workers = this.getWorkers();

    return {
      tasks: {
        total: tasks.length,
        available: tasks.filter((t) => t.status === "available").length,
        claimed: tasks.filter((t) => t.status === "claimed").length,
        in_progress: tasks.filter((t) => t.status === "in_progress").length,
        done: tasks.filter((t) => t.status === "done").length,
        failed: tasks.filter((t) => t.status === "failed").length,
      },
      workers: {
        total: workers.length,
        idle: workers.filter((w) => w.status === "idle").length,
        busy: workers.filter((w) => w.status === "busy").length,
        offline: workers.filter((w) => w.status === "offline").length,
      },
      timestamp: new Date().toISOString(),
    };
  }
}

// OpenAPI Specification
const openApiSpec = {
  openapi: "3.0.0",
  info: {
    title: "Claude Coordinator REST API",
    version: "1.0.0",
    description: "REST API for Claude Multi-Agent Coordinator",
  },
  servers: [
    {
      url: "/api/v1",
      description: "API v1",
    },
  ],
  paths: {
    "/tasks": {
      get: {
        summary: "List all tasks",
        tags: ["Tasks"],
        parameters: [
          { name: "status", in: "query", schema: { type: "string" } },
          { name: "priority", in: "query", schema: { type: "integer" } },
          { name: "claimed_by", in: "query", schema: { type: "string" } },
          { name: "assigned_to", in: "query", schema: { type: "string" } },
        ],
        responses: {
          "200": { description: "List of tasks" },
        },
      },
      post: {
        summary: "Create a task",
        tags: ["Tasks"],
        requestBody: {
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["description"],
                properties: {
                  description: { type: "string" },
                  priority: { type: "integer", minimum: 1, maximum: 5 },
                  dependencies: { type: "array", items: { type: "string" } },
                  tags: { type: "array", items: { type: "string" } },
                },
              },
            },
          },
        },
        responses: {
          "201": { description: "Task created" },
        },
      },
    },
    "/tasks/{id}": {
      get: {
        summary: "Get task by ID",
        tags: ["Tasks"],
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            schema: { type: "string" },
          },
        ],
        responses: {
          "200": { description: "Task details" },
          "404": { description: "Task not found" },
        },
      },
      put: {
        summary: "Update task",
        tags: ["Tasks"],
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            schema: { type: "string" },
          },
        ],
        responses: {
          "200": { description: "Task updated" },
          "404": { description: "Task not found" },
        },
      },
      delete: {
        summary: "Delete task",
        tags: ["Tasks"],
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            schema: { type: "string" },
          },
        ],
        responses: {
          "204": { description: "Task deleted" },
          "404": { description: "Task not found" },
        },
      },
    },
    "/tasks/{id}/claim": {
      post: {
        summary: "Claim a task",
        tags: ["Tasks"],
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            schema: { type: "string" },
          },
        ],
        requestBody: {
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["worker_id"],
                properties: {
                  worker_id: { type: "string" },
                },
              },
            },
          },
        },
        responses: {
          "200": { description: "Task claimed" },
          "400": { description: "Task not available" },
          "404": { description: "Task not found" },
        },
      },
    },
    "/tasks/{id}/complete": {
      post: {
        summary: "Complete a task",
        tags: ["Tasks"],
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            schema: { type: "string" },
          },
        ],
        requestBody: {
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  result: { type: "string" },
                },
              },
            },
          },
        },
        responses: {
          "200": { description: "Task completed" },
          "404": { description: "Task not found" },
        },
      },
    },
    "/tasks/{id}/fail": {
      post: {
        summary: "Mark task as failed",
        tags: ["Tasks"],
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            schema: { type: "string" },
          },
        ],
        requestBody: {
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  error: { type: "string" },
                },
              },
            },
          },
        },
        responses: {
          "200": { description: "Task marked as failed" },
          "404": { description: "Task not found" },
        },
      },
    },
    "/workers": {
      get: {
        summary: "List all workers",
        tags: ["Workers"],
        responses: {
          "200": { description: "List of workers" },
        },
      },
      post: {
        summary: "Register a worker",
        tags: ["Workers"],
        requestBody: {
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  id: { type: "string" },
                  capabilities: { type: "array", items: { type: "string" } },
                },
              },
            },
          },
        },
        responses: {
          "201": { description: "Worker registered" },
        },
      },
    },
    "/workers/{id}/heartbeat": {
      post: {
        summary: "Send worker heartbeat",
        tags: ["Workers"],
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            schema: { type: "string" },
          },
        ],
        responses: {
          "200": { description: "Heartbeat recorded" },
          "404": { description: "Worker not found" },
        },
      },
    },
    "/discoveries": {
      get: {
        summary: "List all discoveries",
        tags: ["Discoveries"],
        responses: {
          "200": { description: "List of discoveries" },
        },
      },
      post: {
        summary: "Add a discovery",
        tags: ["Discoveries"],
        requestBody: {
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["title", "content"],
                properties: {
                  title: { type: "string" },
                  content: { type: "string" },
                  created_by: { type: "string" },
                  tags: { type: "array", items: { type: "string" } },
                },
              },
            },
          },
        },
        responses: {
          "201": { description: "Discovery added" },
        },
      },
    },
    "/status": {
      get: {
        summary: "Get coordination status",
        tags: ["Status"],
        responses: {
          "200": { description: "Coordination status" },
        },
      },
    },
  },
};

// Create Express app
export function createRestApi(coordinationDir: string): express.Application {
  const app = express();
  const service = new CoordinationService(coordinationDir);

  // Middleware
  app.use(helmet());
  app.use(cors(corsOrigins.length ? { origin: corsOrigins } : undefined));
  app.use(express.json());
  app.use(morgan("combined"));
  app.use(requireAuth);

  // Rate limiting
  const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 1000, // limit each IP to 1000 requests per windowMs
  });
  app.use(limiter);

  // Swagger UI
  app.use("/api-docs", swaggerUi.serve, swaggerUi.setup(openApiSpec));

  // API Routes
  const router = express.Router();

  // Health check
  router.get("/health", (req: Request, res: Response) => {
    res.json({ status: "ok", timestamp: new Date().toISOString() });
  });

  // Status
  router.get("/status", (req: Request, res: Response) => {
    const status = service.getStatus();
    res.json({ success: true, data: status });
  });

  // Tasks
  router.get("/tasks", (req: Request, res: Response) => {
    let tasks = service.getTasks();

    // Filter by status
    if (req.query.status) {
      tasks = tasks.filter((t) => t.status === req.query.status);
    }

    // Filter by priority
    if (req.query.priority) {
      tasks = tasks.filter(
        (t) => t.priority === parseInt(req.query.priority as string),
      );
    }

    // Filter by claimed_by / assigned_to (backward compatible)
    if (req.query.claimed_by || req.query.assigned_to) {
      const claimedBy = (req.query.claimed_by ||
        req.query.assigned_to) as string;
      tasks = tasks.filter((t) => t.claimed_by === claimedBy);
    }

    // Sorting
    tasks.sort((a, b) => a.priority - b.priority);

    res.json({ success: true, data: tasks, meta: { total: tasks.length } });
  });

  router.get("/tasks/:id", (req: Request, res: Response) => {
    const task = service.getTask(req.params.id);
    if (!task) {
      res.status(404).json({ success: false, error: "Task not found" });
      return;
    }
    res.json({ success: true, data: task });
  });

  router.post("/tasks", (req: Request, res: Response) => {
    const task = service.createTask(req.body);
    res.status(201).json({ success: true, data: task });
  });

  router.put("/tasks/:id", (req: Request, res: Response) => {
    const task = service.updateTask(req.params.id, req.body);
    if (!task) {
      res.status(404).json({ success: false, error: "Task not found" });
      return;
    }
    res.json({ success: true, data: task });
  });

  router.delete("/tasks/:id", (req: Request, res: Response) => {
    const deleted = service.deleteTask(req.params.id);
    if (!deleted) {
      res.status(404).json({ success: false, error: "Task not found" });
      return;
    }
    res.status(204).send();
  });

  router.post("/tasks/:id/claim", (req: Request, res: Response) => {
    const { worker_id } = req.body;
    if (!worker_id) {
      res.status(400).json({ success: false, error: "worker_id is required" });
      return;
    }
    const task = service.claimTask(req.params.id, worker_id);
    if (!task) {
      res
        .status(400)
        .json({ success: false, error: "Task not available or not found" });
      return;
    }
    res.json({ success: true, data: task });
  });

  router.post("/tasks/:id/complete", (req: Request, res: Response) => {
    const task = service.completeTask(req.params.id, req.body.result || "");
    if (!task) {
      res.status(404).json({ success: false, error: "Task not found" });
      return;
    }
    res.json({ success: true, data: task });
  });

  router.post("/tasks/:id/fail", (req: Request, res: Response) => {
    const task = service.failTask(req.params.id, req.body.error || "");
    if (!task) {
      res.status(404).json({ success: false, error: "Task not found" });
      return;
    }
    res.json({ success: true, data: task });
  });

  // Workers
  router.get("/workers", (req: Request, res: Response) => {
    const workers = service.getWorkers();
    res.json({ success: true, data: workers, meta: { total: workers.length } });
  });

  router.post("/workers", (req: Request, res: Response) => {
    const worker = service.registerWorker(req.body);
    res.status(201).json({ success: true, data: worker });
  });

  router.put("/workers/:id", (req: Request, res: Response) => {
    const worker = service.updateWorker(req.params.id, req.body);
    if (!worker) {
      res.status(404).json({ success: false, error: "Worker not found" });
      return;
    }
    res.json({ success: true, data: worker });
  });

  router.post("/workers/:id/heartbeat", (req: Request, res: Response) => {
    const worker = service.heartbeat(req.params.id);
    if (!worker) {
      res.status(404).json({ success: false, error: "Worker not found" });
      return;
    }
    res.json({ success: true, data: worker });
  });

  // Discoveries
  router.get("/discoveries", (req: Request, res: Response) => {
    const discoveries = service.getDiscoveries();
    res.json({
      success: true,
      data: discoveries,
      meta: { total: discoveries.length },
    });
  });

  router.post("/discoveries", (req: Request, res: Response) => {
    const discovery = service.addDiscovery(req.body);
    res.status(201).json({ success: true, data: discovery });
  });

  // Mount router
  app.use("/api/v1", router);

  // Error handling
  app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
    console.error(err.stack);
    res.status(500).json({ success: false, error: "Internal server error" });
  });

  return app;
}

// Main entry point
if (require.main === module) {
  const coordinationDir = process.env.COORDINATION_DIR || ".coordination";
  const port = parseInt(process.env.REST_API_PORT || "3002");

  const app = createRestApi(coordinationDir);
  app.listen(port, () => {
    console.log(`REST API server running on port ${port}`);
    console.log(`Swagger UI available at http://localhost:${port}/api-docs`);
  });
}

export { CoordinationService };
