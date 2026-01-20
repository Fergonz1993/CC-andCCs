/**
 * SQLite Persistence Backend (adv-b-002)
 *
 * Alternative storage backend using SQLite for better concurrent access
 * and data integrity compared to JSON file storage.
 */

import * as fs from "fs";
import * as path from "path";
import type {
  Task,
  Agent,
  Discovery,
  AuditLogEntry,
  Transaction,
  TaskFilter,
  PaginationOptions,
  PaginatedResult,
  CoordinationState,
} from "./types.js";

// SQLite interface - we use better-sqlite3 style synchronous API
// This module provides a fallback JSON implementation if SQLite is not available

interface SQLiteDatabase {
  prepare(sql: string): SQLiteStatement;
  exec(sql: string): void;
  close(): void;
  transaction<T>(fn: () => T): () => T;
}

interface SQLiteStatement {
  run(...params: unknown[]): { changes: number; lastInsertRowid: number };
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
}

export class SQLiteStore {
  private db: SQLiteDatabase | null = null;
  private dbPath: string;
  private fallbackToJson: boolean = false;
  private jsonState: CoordinationState | null = null;
  private stateDir: string;

  constructor(stateDir: string = ".coordination") {
    this.stateDir = stateDir;
    this.dbPath = path.join(stateDir, "coordination.db");
  }

  async initialize(): Promise<void> {
    // Ensure state directory exists
    if (!fs.existsSync(this.stateDir)) {
      fs.mkdirSync(this.stateDir, { recursive: true });
    }

    try {
      // Try to load better-sqlite3
      const Database = await import("better-sqlite3").then((m) => m.default);
      this.db = new Database(this.dbPath) as unknown as SQLiteDatabase;
      this.createTables();
      console.error(`SQLite store initialized at ${this.dbPath}`);
    } catch {
      console.error(
        "better-sqlite3 not available, falling back to JSON storage",
      );
      this.fallbackToJson = true;
      this.loadJsonState();
    }
  }

  private createTables(): void {
    if (!this.db) return;

    this.db.exec(`
      -- Coordination metadata
      CREATE TABLE IF NOT EXISTS coordination (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        master_plan TEXT DEFAULT '',
        goal TEXT DEFAULT '',
        created_at TEXT NOT NULL,
        last_activity TEXT NOT NULL
      );

      -- Tasks table
      CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'available',
        priority INTEGER NOT NULL DEFAULT 5,
        claimed_by TEXT,
        dependencies TEXT DEFAULT '[]',
        context TEXT,
        result TEXT,
        tags TEXT DEFAULT '[]',
        created_at TEXT NOT NULL,
        claimed_at TEXT,
        completed_at TEXT
      );

      -- Agents table
      CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        role TEXT NOT NULL,
        last_heartbeat TEXT NOT NULL,
        current_task TEXT,
        tasks_completed INTEGER DEFAULT 0
      );

      -- Discoveries table
      CREATE TABLE IF NOT EXISTS discoveries (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        content TEXT NOT NULL,
        tags TEXT DEFAULT '[]',
        created_at TEXT NOT NULL
      );

      -- Audit log table
      CREATE TABLE IF NOT EXISTS audit_log (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        action TEXT NOT NULL,
        agent_id TEXT,
        task_id TEXT,
        details TEXT NOT NULL,
        transaction_id TEXT
      );

      -- Transactions table
      CREATE TABLE IF NOT EXISTS transactions (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        started_at TEXT NOT NULL,
        operations TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'pending'
      );

      -- Create indexes for common queries
      CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
      CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
      CREATE INDEX IF NOT EXISTS idx_tasks_claimed_by ON tasks(claimed_by);
      CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
      CREATE INDEX IF NOT EXISTS idx_audit_log_task_id ON audit_log(task_id);

      -- Initialize coordination row if not exists
      INSERT OR IGNORE INTO coordination (id, created_at, last_activity)
      VALUES (1, datetime('now'), datetime('now'));
    `);
  }

  private loadJsonState(): void {
    const jsonPath = path.join(this.stateDir, "mcp-state.json");
    try {
      if (fs.existsSync(jsonPath)) {
        const data = fs.readFileSync(jsonPath, "utf-8");
        const saved = JSON.parse(data);
        this.jsonState = {
          master_plan: saved.master_plan || "",
          goal: saved.goal || "",
          tasks: saved.tasks || [],
          agents: new Map(Object.entries(saved.agents || {})),
          discoveries: saved.discoveries || [],
          created_at: saved.created_at || new Date().toISOString(),
          last_activity: saved.last_activity || new Date().toISOString(),
          audit_log: saved.audit_log || [],
          transactions: new Map(Object.entries(saved.transactions || {})),
        };
      } else {
        this.jsonState = {
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
      }
    } catch (error) {
      console.error("Failed to load JSON state:", error);
      this.jsonState = {
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
    }
  }

  private saveJsonState(): void {
    if (!this.jsonState) return;
    const jsonPath = path.join(this.stateDir, "mcp-state.json");
    try {
      const serializable = {
        master_plan: this.jsonState.master_plan,
        goal: this.jsonState.goal,
        tasks: this.jsonState.tasks,
        agents: Object.fromEntries(this.jsonState.agents),
        discoveries: this.jsonState.discoveries,
        created_at: this.jsonState.created_at,
        last_activity: this.jsonState.last_activity,
        audit_log: this.jsonState.audit_log,
        transactions: Object.fromEntries(this.jsonState.transactions),
      };
      fs.writeFileSync(jsonPath, JSON.stringify(serializable, null, 2));
    } catch (error) {
      console.error("Failed to save JSON state:", error);
    }
  }

  // ============================================================================
  // Task Operations
  // ============================================================================

  createTask(task: Task): Task {
    if (this.fallbackToJson && this.jsonState) {
      this.jsonState.tasks.push(task);
      this.jsonState.last_activity = new Date().toISOString();
      this.saveJsonState();
      return task;
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare(`
      INSERT INTO tasks (id, description, status, priority, claimed_by, dependencies, context, result, tags, created_at, claimed_at, completed_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      task.id,
      task.description,
      task.status,
      task.priority,
      task.claimed_by,
      JSON.stringify(task.dependencies),
      task.context ? JSON.stringify(task.context) : null,
      task.result ? JSON.stringify(task.result) : null,
      JSON.stringify(task.tags || []),
      task.created_at,
      task.claimed_at,
      task.completed_at,
    );

    this.updateLastActivity();
    return task;
  }

  getTask(taskId: string): Task | null {
    if (this.fallbackToJson && this.jsonState) {
      return this.jsonState.tasks.find((t) => t.id === taskId) || null;
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare("SELECT * FROM tasks WHERE id = ?");
    const row = stmt.get(taskId) as Record<string, unknown> | undefined;

    if (!row) return null;
    return this.rowToTask(row);
  }

  updateTask(taskId: string, updates: Partial<Task>): boolean {
    if (this.fallbackToJson && this.jsonState) {
      const idx = this.jsonState.tasks.findIndex((t) => t.id === taskId);
      if (idx === -1) return false;
      this.jsonState.tasks[idx] = { ...this.jsonState.tasks[idx], ...updates };
      this.jsonState.last_activity = new Date().toISOString();
      this.saveJsonState();
      return true;
    }

    if (!this.db) throw new Error("Database not initialized");

    const fields: string[] = [];
    const values: unknown[] = [];

    if (updates.status !== undefined) {
      fields.push("status = ?");
      values.push(updates.status);
    }
    if (updates.priority !== undefined) {
      fields.push("priority = ?");
      values.push(updates.priority);
    }
    if (updates.claimed_by !== undefined) {
      fields.push("claimed_by = ?");
      values.push(updates.claimed_by);
    }
    if (updates.claimed_at !== undefined) {
      fields.push("claimed_at = ?");
      values.push(updates.claimed_at);
    }
    if (updates.completed_at !== undefined) {
      fields.push("completed_at = ?");
      values.push(updates.completed_at);
    }
    if (updates.result !== undefined) {
      fields.push("result = ?");
      values.push(JSON.stringify(updates.result));
    }
    if (updates.tags !== undefined) {
      fields.push("tags = ?");
      values.push(JSON.stringify(updates.tags));
    }

    if (fields.length === 0) return false;

    values.push(taskId);
    const stmt = this.db.prepare(
      `UPDATE tasks SET ${fields.join(", ")} WHERE id = ?`,
    );
    const result = stmt.run(...values);

    this.updateLastActivity();
    return result.changes > 0;
  }

  deleteTask(taskId: string): boolean {
    if (this.fallbackToJson && this.jsonState) {
      const idx = this.jsonState.tasks.findIndex((t) => t.id === taskId);
      if (idx === -1) return false;
      this.jsonState.tasks.splice(idx, 1);
      this.jsonState.last_activity = new Date().toISOString();
      this.saveJsonState();
      return true;
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare("DELETE FROM tasks WHERE id = ?");
    const result = stmt.run(taskId);

    this.updateLastActivity();
    return result.changes > 0;
  }

  getAllTasks(): Task[] {
    if (this.fallbackToJson && this.jsonState) {
      return [...this.jsonState.tasks];
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare("SELECT * FROM tasks ORDER BY priority ASC");
    const rows = stmt.all() as Record<string, unknown>[];
    return rows.map((row) => this.rowToTask(row));
  }

  // Query filtering (adv-b-007)
  queryTasks(filter: TaskFilter): Task[] {
    if (this.fallbackToJson && this.jsonState) {
      return this.filterTasksInMemory(this.jsonState.tasks, filter);
    }

    if (!this.db) throw new Error("Database not initialized");

    const conditions: string[] = ["1=1"];
    const params: unknown[] = [];

    if (filter.status) {
      if (Array.isArray(filter.status)) {
        conditions.push(
          `status IN (${filter.status.map(() => "?").join(",")})`,
        );
        params.push(...filter.status);
      } else {
        conditions.push("status = ?");
        params.push(filter.status);
      }
    }

    if (filter.priority_min !== undefined) {
      conditions.push("priority >= ?");
      params.push(filter.priority_min);
    }

    if (filter.priority_max !== undefined) {
      conditions.push("priority <= ?");
      params.push(filter.priority_max);
    }

    if (filter.claimed_by) {
      conditions.push("claimed_by = ?");
      params.push(filter.claimed_by);
    }

    if (filter.created_after) {
      conditions.push("created_at >= ?");
      params.push(filter.created_after);
    }

    if (filter.created_before) {
      conditions.push("created_at <= ?");
      params.push(filter.created_before);
    }

    if (filter.has_dependencies !== undefined) {
      if (filter.has_dependencies) {
        conditions.push("dependencies != '[]'");
      } else {
        conditions.push("dependencies = '[]'");
      }
    }

    if (filter.search) {
      conditions.push("description LIKE ?");
      params.push(`%${filter.search}%`);
    }

    const stmt = this.db.prepare(
      `SELECT * FROM tasks WHERE ${conditions.join(" AND ")} ORDER BY priority ASC`,
    );
    const rows = stmt.all(...params) as Record<string, unknown>[];

    let tasks = rows.map((row) => this.rowToTask(row));

    // Filter by tags in memory (JSON field)
    if (filter.tags && filter.tags.length > 0) {
      tasks = tasks.filter(
        (t) => t.tags && filter.tags!.some((tag) => t.tags!.includes(tag)),
      );
    }

    return tasks;
  }

  // Pagination (adv-b-008)
  getTasksPaginated(
    filter: TaskFilter,
    pagination: PaginationOptions,
  ): PaginatedResult<Task> {
    const page = pagination.page || 1;
    const pageSize = Math.min(pagination.page_size || 20, 100);
    const sortBy = pagination.sort_by || "priority";
    const sortDirection = pagination.sort_direction || "asc";

    // Get all filtered tasks first
    const allTasks = this.queryTasks(filter);

    // Sort
    allTasks.sort((a, b) => {
      let aVal: string | number | null;
      let bVal: string | number | null;

      switch (sortBy) {
        case "priority":
          aVal = a.priority;
          bVal = b.priority;
          break;
        case "created_at":
          aVal = a.created_at;
          bVal = b.created_at;
          break;
        case "claimed_at":
          aVal = a.claimed_at;
          bVal = b.claimed_at;
          break;
        case "completed_at":
          aVal = a.completed_at;
          bVal = b.completed_at;
          break;
        default:
          aVal = a.priority;
          bVal = b.priority;
      }

      if (aVal === null && bVal === null) return 0;
      if (aVal === null) return sortDirection === "asc" ? 1 : -1;
      if (bVal === null) return sortDirection === "asc" ? -1 : 1;

      const cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      return sortDirection === "asc" ? cmp : -cmp;
    });

    const total = allTasks.length;
    const totalPages = Math.ceil(total / pageSize);
    const offset = (page - 1) * pageSize;
    const items = allTasks.slice(offset, offset + pageSize);

    return {
      items,
      total,
      page,
      page_size: pageSize,
      total_pages: totalPages,
      has_next: page < totalPages,
      has_prev: page > 1,
    };
  }

  private filterTasksInMemory(tasks: Task[], filter: TaskFilter): Task[] {
    return tasks.filter((t) => {
      if (filter.status) {
        const statuses = Array.isArray(filter.status)
          ? filter.status
          : [filter.status];
        if (!statuses.includes(t.status)) return false;
      }

      if (filter.priority_min !== undefined && t.priority < filter.priority_min)
        return false;
      if (filter.priority_max !== undefined && t.priority > filter.priority_max)
        return false;

      if (filter.claimed_by && t.claimed_by !== filter.claimed_by) return false;

      if (filter.tags && filter.tags.length > 0) {
        if (!t.tags || !filter.tags.some((tag) => t.tags!.includes(tag)))
          return false;
      }

      if (filter.created_after && t.created_at < filter.created_after)
        return false;
      if (filter.created_before && t.created_at > filter.created_before)
        return false;

      if (filter.has_dependencies !== undefined) {
        const hasDeps = t.dependencies.length > 0;
        if (filter.has_dependencies !== hasDeps) return false;
      }

      if (
        filter.search &&
        !t.description.toLowerCase().includes(filter.search.toLowerCase())
      )
        return false;

      return true;
    });
  }

  private rowToTask(row: Record<string, unknown>): Task {
    return {
      id: row.id as string,
      description: row.description as string,
      status: row.status as Task["status"],
      priority: row.priority as number,
      claimed_by: row.claimed_by as string | null,
      dependencies: JSON.parse((row.dependencies as string) || "[]"),
      context: row.context ? JSON.parse(row.context as string) : null,
      result: row.result ? JSON.parse(row.result as string) : null,
      tags: JSON.parse((row.tags as string) || "[]"),
      created_at: row.created_at as string,
      claimed_at: row.claimed_at as string | null,
      completed_at: row.completed_at as string | null,
    };
  }

  // ============================================================================
  // Agent Operations
  // ============================================================================

  createAgent(agent: Agent): Agent {
    if (this.fallbackToJson && this.jsonState) {
      this.jsonState.agents.set(agent.id, agent);
      this.jsonState.last_activity = new Date().toISOString();
      this.saveJsonState();
      return agent;
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO agents (id, role, last_heartbeat, current_task, tasks_completed)
      VALUES (?, ?, ?, ?, ?)
    `);

    stmt.run(
      agent.id,
      agent.role,
      agent.last_heartbeat,
      agent.current_task,
      agent.tasks_completed,
    );

    this.updateLastActivity();
    return agent;
  }

  getAgent(agentId: string): Agent | null {
    if (this.fallbackToJson && this.jsonState) {
      return this.jsonState.agents.get(agentId) || null;
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare("SELECT * FROM agents WHERE id = ?");
    const row = stmt.get(agentId) as Record<string, unknown> | undefined;

    if (!row) return null;
    return this.rowToAgent(row);
  }

  updateAgent(agentId: string, updates: Partial<Agent>): boolean {
    if (this.fallbackToJson && this.jsonState) {
      const agent = this.jsonState.agents.get(agentId);
      if (!agent) return false;
      this.jsonState.agents.set(agentId, { ...agent, ...updates });
      this.jsonState.last_activity = new Date().toISOString();
      this.saveJsonState();
      return true;
    }

    if (!this.db) throw new Error("Database not initialized");

    const fields: string[] = [];
    const values: unknown[] = [];

    if (updates.last_heartbeat !== undefined) {
      fields.push("last_heartbeat = ?");
      values.push(updates.last_heartbeat);
    }
    if (updates.current_task !== undefined) {
      fields.push("current_task = ?");
      values.push(updates.current_task);
    }
    if (updates.tasks_completed !== undefined) {
      fields.push("tasks_completed = ?");
      values.push(updates.tasks_completed);
    }

    if (fields.length === 0) return false;

    values.push(agentId);
    const stmt = this.db.prepare(
      `UPDATE agents SET ${fields.join(", ")} WHERE id = ?`,
    );
    const result = stmt.run(...values);

    return result.changes > 0;
  }

  getAllAgents(): Agent[] {
    if (this.fallbackToJson && this.jsonState) {
      return Array.from(this.jsonState.agents.values());
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare("SELECT * FROM agents");
    const rows = stmt.all() as Record<string, unknown>[];
    return rows.map((row) => this.rowToAgent(row));
  }

  private rowToAgent(row: Record<string, unknown>): Agent {
    return {
      id: row.id as string,
      role: row.role as Agent["role"],
      last_heartbeat: row.last_heartbeat as string,
      current_task: row.current_task as string | null,
      tasks_completed: row.tasks_completed as number,
    };
  }

  // ============================================================================
  // Discovery Operations
  // ============================================================================

  createDiscovery(discovery: Discovery): Discovery {
    if (this.fallbackToJson && this.jsonState) {
      this.jsonState.discoveries.push(discovery);
      this.jsonState.last_activity = new Date().toISOString();
      this.saveJsonState();
      return discovery;
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare(`
      INSERT INTO discoveries (id, agent_id, content, tags, created_at)
      VALUES (?, ?, ?, ?, ?)
    `);

    stmt.run(
      discovery.id,
      discovery.agent_id,
      discovery.content,
      JSON.stringify(discovery.tags),
      discovery.created_at,
    );

    this.updateLastActivity();
    return discovery;
  }

  getDiscoveries(tags?: string[], limit: number = 20): Discovery[] {
    if (this.fallbackToJson && this.jsonState) {
      let discoveries = [...this.jsonState.discoveries];
      if (tags && tags.length > 0) {
        discoveries = discoveries.filter((d) =>
          d.tags.some((t) => tags.includes(t)),
        );
      }
      return discoveries.slice(-limit);
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare(
      `SELECT * FROM discoveries ORDER BY created_at DESC LIMIT ?`,
    );
    const rows = stmt.all(limit) as Record<string, unknown>[];

    const discoveries = rows.map((row) => this.rowToDiscovery(row));

    if (tags && tags.length > 0) {
      return discoveries.filter((d) => d.tags.some((t) => tags.includes(t)));
    }

    return discoveries;
  }

  private rowToDiscovery(row: Record<string, unknown>): Discovery {
    return {
      id: row.id as string,
      agent_id: row.agent_id as string,
      content: row.content as string,
      tags: JSON.parse((row.tags as string) || "[]"),
      created_at: row.created_at as string,
    };
  }

  // ============================================================================
  // Audit Log Operations (adv-b-009)
  // ============================================================================

  addAuditLog(entry: AuditLogEntry): AuditLogEntry {
    if (this.fallbackToJson && this.jsonState) {
      this.jsonState.audit_log.push(entry);
      // Keep only last 10000 entries
      if (this.jsonState.audit_log.length > 10000) {
        this.jsonState.audit_log = this.jsonState.audit_log.slice(-10000);
      }
      this.saveJsonState();
      return entry;
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare(`
      INSERT INTO audit_log (id, timestamp, action, entity_type, entity_id, agent_id, details)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      entry.id,
      entry.timestamp,
      entry.action,
      entry.entity_type,
      entry.entity_id,
      entry.agent_id,
      JSON.stringify(entry.details || {}),
    );

    return entry;
  }

  getAuditLog(
    entityId?: string,
    agentId?: string,
    limit: number = 100,
  ): AuditLogEntry[] {
    if (this.fallbackToJson && this.jsonState) {
      let log = [...this.jsonState.audit_log];
      if (entityId) log = log.filter((e) => e.entity_id === entityId);
      if (agentId) log = log.filter((e) => e.agent_id === agentId);
      return log.slice(-limit);
    }

    if (!this.db) throw new Error("Database not initialized");

    const conditions: string[] = ["1=1"];
    const params: unknown[] = [];

    if (entityId) {
      conditions.push("entity_id = ?");
      params.push(entityId);
    }

    if (agentId) {
      conditions.push("agent_id = ?");
      params.push(agentId);
    }

    params.push(limit);

    const stmt = this.db.prepare(
      `SELECT * FROM audit_log WHERE ${conditions.join(" AND ")} ORDER BY timestamp DESC LIMIT ?`,
    );
    const rows = stmt.all(...params) as Record<string, unknown>[];

    return rows.map((row) => ({
      id: row.id as string,
      timestamp: row.timestamp as string,
      action: row.action as string,
      entity_type: row.entity_type as AuditLogEntry["entity_type"],
      entity_id: row.entity_id as string,
      agent_id: row.agent_id as string | null,
      details: JSON.parse((row.details as string) || "{}"),
    }));
  }

  // ============================================================================
  // Coordination Metadata
  // ============================================================================

  getCoordinationMeta(): {
    master_plan: string;
    goal: string;
    created_at: string;
    last_activity: string;
  } {
    if (this.fallbackToJson && this.jsonState) {
      return {
        master_plan: this.jsonState.master_plan,
        goal: this.jsonState.goal,
        created_at: this.jsonState.created_at,
        last_activity: this.jsonState.last_activity,
      };
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare("SELECT * FROM coordination WHERE id = 1");
    const row = stmt.get() as Record<string, unknown> | undefined;

    if (!row) {
      return {
        master_plan: "",
        goal: "",
        created_at: new Date().toISOString(),
        last_activity: new Date().toISOString(),
      };
    }

    return {
      master_plan: (row.master_plan as string) || "",
      goal: (row.goal as string) || "",
      created_at: row.created_at as string,
      last_activity: row.last_activity as string,
    };
  }

  setCoordinationMeta(goal: string, masterPlan: string): void {
    if (this.fallbackToJson && this.jsonState) {
      this.jsonState.goal = goal;
      this.jsonState.master_plan = masterPlan;
      this.jsonState.last_activity = new Date().toISOString();
      this.saveJsonState();
      return;
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare(`
      UPDATE coordination SET goal = ?, master_plan = ?, last_activity = datetime('now')
      WHERE id = 1
    `);
    stmt.run(goal, masterPlan);
  }

  private updateLastActivity(): void {
    if (this.fallbackToJson && this.jsonState) {
      this.jsonState.last_activity = new Date().toISOString();
      this.saveJsonState();
      return;
    }

    if (!this.db) return;

    const stmt = this.db.prepare(`
      UPDATE coordination SET last_activity = datetime('now') WHERE id = 1
    `);
    stmt.run();
  }

  // ============================================================================
  // Transaction Operations (adv-b-006)
  // ============================================================================

  createTransaction(transaction: Transaction): Transaction {
    if (this.fallbackToJson && this.jsonState) {
      this.jsonState.transactions.set(transaction.id, transaction);
      this.saveJsonState();
      return transaction;
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare(`
      INSERT INTO transactions (id, agent_id, created_at, operations, status, committed_at)
      VALUES (?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      transaction.id,
      transaction.agent_id,
      transaction.created_at,
      JSON.stringify(transaction.operations),
      transaction.status,
      transaction.committed_at || null,
    );

    return transaction;
  }

  getTransaction(transactionId: string): Transaction | null {
    if (this.fallbackToJson && this.jsonState) {
      return this.jsonState.transactions.get(transactionId) || null;
    }

    if (!this.db) throw new Error("Database not initialized");

    const stmt = this.db.prepare("SELECT * FROM transactions WHERE id = ?");
    const row = stmt.get(transactionId) as Record<string, unknown> | undefined;

    if (!row) return null;

    return {
      id: row.id as string,
      agent_id: row.agent_id as string,
      created_at: row.created_at as string,
      operations: JSON.parse((row.operations as string) || "[]"),
      status: row.status as Transaction["status"],
      committed_at: row.committed_at as string | undefined,
    };
  }

  updateTransaction(
    transactionId: string,
    updates: Partial<Transaction>,
  ): boolean {
    if (this.fallbackToJson && this.jsonState) {
      const tx = this.jsonState.transactions.get(transactionId);
      if (!tx) return false;
      this.jsonState.transactions.set(transactionId, { ...tx, ...updates });
      this.saveJsonState();
      return true;
    }

    if (!this.db) throw new Error("Database not initialized");

    const fields: string[] = [];
    const values: unknown[] = [];

    if (updates.status !== undefined) {
      fields.push("status = ?");
      values.push(updates.status);
    }
    if (updates.operations !== undefined) {
      fields.push("operations = ?");
      values.push(JSON.stringify(updates.operations));
    }
    if (updates.committed_at !== undefined) {
      fields.push("committed_at = ?");
      values.push(updates.committed_at);
    }

    if (fields.length === 0) return false;

    values.push(transactionId);
    const stmt = this.db.prepare(
      `UPDATE transactions SET ${fields.join(", ")} WHERE id = ?`,
    );
    const result = stmt.run(...values);

    return result.changes > 0;
  }

  close(): void {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}
