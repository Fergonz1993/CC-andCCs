import * as fs from "fs";
import * as path from "path";

export interface Task {
  id: string;
  description: string;
  status: "available" | "claimed" | "in_progress" | "done" | "failed";
  priority: number;
  assigned_to?: string;
  dependencies?: string[];
  created_at?: string;
  updated_at?: string;
  result?: string;
  error?: string;
}

export interface Worker {
  id: string;
  status: "idle" | "busy" | "offline";
  current_task?: string;
  last_heartbeat?: string;
  tasks_completed?: number;
}

export interface Discovery {
  id: string;
  title: string;
  content: string;
  created_by: string;
  created_at: string;
  tags?: string[];
}

export class CoordinationService {
  private coordPath: string;
  private agentId: string;

  constructor(coordPath: string, agentId: string) {
    this.coordPath = coordPath;
    this.agentId = agentId;
  }

  private ensureDir(dirPath: string): void {
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }
  }

  private getTasksPath(): string {
    return path.join(this.coordPath, "tasks.json");
  }

  private getDiscoveriesPath(): string {
    return path.join(this.coordPath, "context", "discoveries.md");
  }

  private getWorkersPath(): string {
    return path.join(this.coordPath, "workers.json");
  }

  async getTasks(): Promise<Task[]> {
    const tasksPath = this.getTasksPath();
    if (!fs.existsSync(tasksPath)) {
      return [];
    }

    try {
      const content = fs.readFileSync(tasksPath, "utf-8");
      const data = JSON.parse(content);
      return data.tasks || [];
    } catch {
      return [];
    }
  }

  async getWorkers(): Promise<Worker[]> {
    const workersPath = this.getWorkersPath();
    if (!fs.existsSync(workersPath)) {
      return [];
    }

    try {
      const content = fs.readFileSync(workersPath, "utf-8");
      const data = JSON.parse(content);
      return data.workers || [];
    } catch {
      return [];
    }
  }

  async getDiscoveries(): Promise<Discovery[]> {
    const discoveriesPath = this.getDiscoveriesPath();
    if (!fs.existsSync(discoveriesPath)) {
      return [];
    }

    try {
      const content = fs.readFileSync(discoveriesPath, "utf-8");
      return this.parseDiscoveriesMarkdown(content);
    } catch {
      return [];
    }
  }

  private parseDiscoveriesMarkdown(content: string): Discovery[] {
    const discoveries: Discovery[] = [];
    const sections = content.split(/^## /m).filter((s) => s.trim());

    for (const section of sections) {
      const lines = section.split("\n");
      const title = lines[0]?.trim() || "";
      const body = lines.slice(1).join("\n").trim();

      if (title && body) {
        discoveries.push({
          id: this.generateId(),
          title,
          content: body,
          created_by: "unknown",
          created_at: new Date().toISOString(),
        });
      }
    }

    return discoveries;
  }

  private generateId(): string {
    return `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  async createTask(description: string, priority: number): Promise<Task> {
    this.ensureDir(this.coordPath);

    const tasks = await this.getTasks();
    const newTask: Task = {
      id: this.generateId(),
      description,
      status: "available",
      priority,
      created_at: new Date().toISOString(),
    };

    tasks.push(newTask);
    this.saveTasks(tasks);
    return newTask;
  }

  private saveTasks(tasks: Task[]): void {
    const tasksPath = this.getTasksPath();
    fs.writeFileSync(
      tasksPath,
      JSON.stringify({ tasks, updated_at: new Date().toISOString() }, null, 2),
    );
  }

  async claimTask(taskId: string): Promise<Task> {
    const tasks = await this.getTasks();
    const task = tasks.find((t) => t.id === taskId);

    if (!task) {
      throw new Error(`Task not found: ${taskId}`);
    }

    if (task.status !== "available") {
      throw new Error(`Task is not available: ${task.status}`);
    }

    task.status = "claimed";
    task.assigned_to = this.agentId;
    task.updated_at = new Date().toISOString();

    this.saveTasks(tasks);
    this.updateWorkerStatus("busy", taskId);
    return task;
  }

  async completeTask(taskId: string, result: string): Promise<Task> {
    const tasks = await this.getTasks();
    const task = tasks.find((t) => t.id === taskId);

    if (!task) {
      throw new Error(`Task not found: ${taskId}`);
    }

    task.status = "done";
    task.result = result;
    task.updated_at = new Date().toISOString();

    this.saveTasks(tasks);
    this.saveTaskResult(taskId, result);
    this.updateWorkerStatus("idle");
    return task;
  }

  async failTask(taskId: string, error: string): Promise<Task> {
    const tasks = await this.getTasks();
    const task = tasks.find((t) => t.id === taskId);

    if (!task) {
      throw new Error(`Task not found: ${taskId}`);
    }

    task.status = "failed";
    task.error = error;
    task.updated_at = new Date().toISOString();

    this.saveTasks(tasks);
    this.updateWorkerStatus("idle");
    return task;
  }

  private saveTaskResult(taskId: string, result: string): void {
    const resultsDir = path.join(this.coordPath, "results");
    this.ensureDir(resultsDir);

    const resultPath = path.join(resultsDir, `task-${taskId}.md`);
    const content = `# Task Result: ${taskId}\n\nCompleted at: ${new Date().toISOString()}\nCompleted by: ${this.agentId}\n\n## Result\n\n${result}\n`;
    fs.writeFileSync(resultPath, content);
  }

  private updateWorkerStatus(
    status: "idle" | "busy" | "offline",
    currentTask?: string,
  ): void {
    const workersPath = this.getWorkersPath();
    let workers: Worker[] = [];

    if (fs.existsSync(workersPath)) {
      try {
        const content = fs.readFileSync(workersPath, "utf-8");
        workers = JSON.parse(content).workers || [];
      } catch {
        workers = [];
      }
    }

    const existingWorker = workers.find((w) => w.id === this.agentId);
    if (existingWorker) {
      existingWorker.status = status;
      existingWorker.current_task = currentTask;
      existingWorker.last_heartbeat = new Date().toISOString();
      if (status === "idle" && existingWorker.tasks_completed !== undefined) {
        existingWorker.tasks_completed++;
      }
    } else {
      workers.push({
        id: this.agentId,
        status,
        current_task: currentTask,
        last_heartbeat: new Date().toISOString(),
        tasks_completed: 0,
      });
    }

    this.ensureDir(this.coordPath);
    fs.writeFileSync(workersPath, JSON.stringify({ workers }, null, 2));
  }

  async initCoordination(goal: string): Promise<void> {
    this.ensureDir(this.coordPath);
    this.ensureDir(path.join(this.coordPath, "context"));
    this.ensureDir(path.join(this.coordPath, "results"));
    this.ensureDir(path.join(this.coordPath, "logs"));

    // Create master plan
    const masterPlanPath = path.join(this.coordPath, "master-plan.md");
    const masterPlanContent = `# Master Plan\n\nGoal: ${goal}\n\nCreated: ${new Date().toISOString()}\nCoordinator: ${this.agentId}\n\n## Approach\n\nTo be determined during planning phase.\n\n## Tasks\n\nSee tasks.json for task queue.\n`;
    fs.writeFileSync(masterPlanPath, masterPlanContent);

    // Initialize empty tasks
    this.saveTasks([]);

    // Initialize discoveries
    const discoveriesPath = this.getDiscoveriesPath();
    fs.writeFileSync(
      discoveriesPath,
      `# Shared Discoveries\n\nThis file contains discoveries shared between agents.\n`,
    );
  }

  async addDiscovery(title: string, content: string): Promise<Discovery> {
    const discoveriesPath = this.getDiscoveriesPath();
    this.ensureDir(path.dirname(discoveriesPath));

    let existingContent = "";
    if (fs.existsSync(discoveriesPath)) {
      existingContent = fs.readFileSync(discoveriesPath, "utf-8");
    } else {
      existingContent = "# Shared Discoveries\n\n";
    }

    const newEntry = `\n## ${title}\n\n*Added by ${this.agentId} at ${new Date().toISOString()}*\n\n${content}\n`;
    fs.writeFileSync(discoveriesPath, existingContent + newEntry);

    return {
      id: this.generateId(),
      title,
      content,
      created_by: this.agentId,
      created_at: new Date().toISOString(),
    };
  }
}
