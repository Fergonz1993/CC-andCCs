/**
 * Performance optimization module for Claude Multi-Agent MCP Server.
 *
 * Implements:
 * - adv-perf-001: In-memory caching for task lookups
 * - adv-perf-002: Connection pooling (not applicable for MCP)
 * - adv-perf-003: Lazy loading of task results
 * - adv-perf-004: Batch write operations
 * - adv-perf-005: Index structures for fast task queries
 * - adv-perf-006: Compression for large task data
 * - adv-perf-007: Async I/O optimization
 * - adv-perf-008: Memory-mapped file access (Node.js equivalent)
 * - adv-perf-009: Query result caching with TTL
 * - adv-perf-010: Profiling hooks for bottleneck detection
 */

import * as fs from "fs";
import * as zlib from "zlib";
import { promisify } from "util";

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

interface CacheEntry<T> {
  value: T;
  createdAt: number;
  lastAccessed: number;
  ttl?: number;
}

interface ProfileSample {
  functionName: string;
  startTime: number;
  endTime: number;
  duration: number;
  success: boolean;
  error?: string;
}

interface ProfileStats {
  functionName: string;
  callCount: number;
  totalTime: number;
  minTime: number;
  maxTime: number;
  avgTime: number;
  errorCount: number;
  p95Time: number;
}

// ============================================================================
// adv-perf-001 & adv-perf-009: In-memory Caching with TTL
// ============================================================================

export class TaskCache<K extends string | number, V> {
  private cache: Map<K, CacheEntry<V>> = new Map();
  private maxSize: number;
  private defaultTtl: number;
  private stats = { hits: 0, misses: 0, evictions: 0 };

  constructor(maxSize: number = 1000, defaultTtl: number = 300000) {
    this.maxSize = maxSize;
    this.defaultTtl = defaultTtl; // milliseconds
  }

  get(key: K): V | undefined {
    const entry = this.cache.get(key);

    if (!entry) {
      this.stats.misses++;
      return undefined;
    }

    // Check TTL
    if (entry.ttl && Date.now() - entry.createdAt > entry.ttl) {
      this.cache.delete(key);
      this.stats.misses++;
      return undefined;
    }

    entry.lastAccessed = Date.now();
    this.stats.hits++;
    return entry.value;
  }

  set(key: K, value: V, ttl?: number): void {
    // Evict if at capacity
    if (!this.cache.has(key) && this.cache.size >= this.maxSize) {
      this.evictLRU();
    }

    this.cache.set(key, {
      value,
      createdAt: Date.now(),
      lastAccessed: Date.now(),
      ttl: ttl ?? this.defaultTtl,
    });
  }

  delete(key: K): boolean {
    return this.cache.delete(key);
  }

  has(key: K): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;

    if (entry.ttl && Date.now() - entry.createdAt > entry.ttl) {
      this.cache.delete(key);
      return false;
    }
    return true;
  }

  clear(): void {
    this.cache.clear();
  }

  private evictLRU(): void {
    let oldestKey: K | null = null;
    let oldestTime = Infinity;

    for (const [key, entry] of this.cache.entries()) {
      if (entry.lastAccessed < oldestTime) {
        oldestTime = entry.lastAccessed;
        oldestKey = key;
      }
    }

    if (oldestKey !== null) {
      this.cache.delete(oldestKey);
      this.stats.evictions++;
    }
  }

  getStats() {
    const total = this.stats.hits + this.stats.misses;
    return {
      ...this.stats,
      size: this.cache.size,
      hitRate: total > 0 ? this.stats.hits / total : 0,
    };
  }

  // Invalidate by prefix (for query cache)
  invalidateByPrefix(prefix: string): number {
    let count = 0;
    for (const key of this.cache.keys()) {
      if (String(key).startsWith(prefix)) {
        this.cache.delete(key);
        count++;
      }
    }
    return count;
  }
}

// Singleton query cache
const queryCache = new TaskCache<string, any>(500, 10000); // 10 second TTL

export function getCachedQuery<T>(
  key: string,
  computeFn: () => T,
  ttl?: number,
): T {
  const cached = queryCache.get(key);
  if (cached !== undefined) {
    return cached;
  }

  const result = computeFn();
  queryCache.set(key, result, ttl);
  return result;
}

export function invalidateQueryCache(pattern: string): number {
  return queryCache.invalidateByPrefix(pattern);
}

// ============================================================================
// adv-perf-005: Index Structures for Fast Task Queries
// ============================================================================

export class TaskIndex {
  private tasks: Map<string, Task> = new Map();
  private byStatus: Map<string, Set<string>> = new Map();
  private byAgent: Map<string, Set<string>> = new Map();
  private completedIds: Set<string> = new Set();
  private priorityQueue: Array<{ priority: number; id: string }> = [];
  private priorityQueueDirty = false;

  addTask(task: Task): void {
    this.tasks.set(task.id, task);

    // Index by status
    if (!this.byStatus.has(task.status)) {
      this.byStatus.set(task.status, new Set());
    }
    this.byStatus.get(task.status)!.add(task.id);

    // Index by agent
    if (task.claimed_by) {
      if (!this.byAgent.has(task.claimed_by)) {
        this.byAgent.set(task.claimed_by, new Set());
      }
      this.byAgent.get(task.claimed_by)!.add(task.id);
    }

    // Track completed
    if (task.status === "done") {
      this.completedIds.add(task.id);
    }

    // Add to priority queue if available
    if (task.status === "available") {
      this.priorityQueue.push({ priority: task.priority, id: task.id });
      this.priorityQueueDirty = true;
    }
  }

  updateTask(taskId: string, newTask: Task): void {
    const oldTask = this.tasks.get(taskId);
    if (!oldTask) {
      this.addTask(newTask);
      return;
    }

    // Update status index
    if (oldTask.status !== newTask.status) {
      this.byStatus.get(oldTask.status)?.delete(taskId);
      if (!this.byStatus.has(newTask.status)) {
        this.byStatus.set(newTask.status, new Set());
      }
      this.byStatus.get(newTask.status)!.add(taskId);

      if (newTask.status === "done") {
        this.completedIds.add(taskId);
      }

      // Update priority queue
      if (oldTask.status === "available" && newTask.status !== "available") {
        this.priorityQueue = this.priorityQueue.filter((p) => p.id !== taskId);
      } else if (
        oldTask.status !== "available" &&
        newTask.status === "available"
      ) {
        this.priorityQueue.push({ priority: newTask.priority, id: taskId });
        this.priorityQueueDirty = true;
      }
    }

    // Update agent index
    if (oldTask.claimed_by !== newTask.claimed_by) {
      if (oldTask.claimed_by) {
        this.byAgent.get(oldTask.claimed_by)?.delete(taskId);
      }
      if (newTask.claimed_by) {
        if (!this.byAgent.has(newTask.claimed_by)) {
          this.byAgent.set(newTask.claimed_by, new Set());
        }
        this.byAgent.get(newTask.claimed_by)!.add(taskId);
      }
    }

    this.tasks.set(taskId, newTask);
  }

  removeTask(taskId: string): void {
    const task = this.tasks.get(taskId);
    if (!task) return;

    this.byStatus.get(task.status)?.delete(taskId);
    if (task.claimed_by) {
      this.byAgent.get(task.claimed_by)?.delete(taskId);
    }
    this.completedIds.delete(taskId);
    this.priorityQueue = this.priorityQueue.filter((p) => p.id !== taskId);
    this.tasks.delete(taskId);
  }

  getTask(taskId: string): Task | undefined {
    return this.tasks.get(taskId);
  }

  getByStatus(status: string): Task[] {
    const ids = this.byStatus.get(status);
    if (!ids) return [];
    return Array.from(ids)
      .map((id) => this.tasks.get(id)!)
      .filter(Boolean);
  }

  getByAgent(agentId: string): Task[] {
    const ids = this.byAgent.get(agentId);
    if (!ids) return [];
    return Array.from(ids)
      .map((id) => this.tasks.get(id)!)
      .filter(Boolean);
  }

  getAvailableWithDependenciesSatisfied(): Task[] {
    const available = this.getByStatus("available");
    return available.filter((task) =>
      task.dependencies.every((dep) => this.completedIds.has(dep)),
    );
  }

  getHighestPriorityAvailable(): Task | undefined {
    if (this.priorityQueueDirty) {
      this.priorityQueue.sort((a, b) => a.priority - b.priority);
      this.priorityQueueDirty = false;
    }

    for (const { id } of this.priorityQueue) {
      const task = this.tasks.get(id);
      if (
        task &&
        task.status === "available" &&
        task.dependencies.every((dep) => this.completedIds.has(dep))
      ) {
        return task;
      }
    }
    return undefined;
  }

  areDependenciesSatisfied(taskId: string): boolean {
    const task = this.tasks.get(taskId);
    if (!task) return false;
    return task.dependencies.every((dep) => this.completedIds.has(dep));
  }

  getStats() {
    return {
      totalTasks: this.tasks.size,
      completedTasks: this.completedIds.size,
      statusDistribution: Object.fromEntries(
        Array.from(this.byStatus.entries()).map(([k, v]) => [k, v.size]),
      ),
    };
  }
}

// ============================================================================
// adv-perf-003: Lazy Loading of Task Results
// ============================================================================

export class LazyValue<T> {
  private loader: () => T | Promise<T>;
  private value?: T;
  private loaded = false;
  private loading = false;
  private error?: Error;

  constructor(loader: () => T | Promise<T>) {
    this.loader = loader;
  }

  async get(): Promise<T> {
    if (this.loaded) {
      return this.value!;
    }

    if (this.loading) {
      // Wait for existing load
      while (this.loading) {
        await new Promise((r) => setTimeout(r, 10));
      }
      if (this.error) throw this.error;
      return this.value!;
    }

    this.loading = true;
    try {
      const result = this.loader();
      this.value = result instanceof Promise ? await result : result;
      this.loaded = true;
      return this.value;
    } catch (e) {
      this.error = e as Error;
      throw e;
    } finally {
      this.loading = false;
    }
  }

  isLoaded(): boolean {
    return this.loaded;
  }

  reset(): void {
    this.value = undefined;
    this.loaded = false;
    this.error = undefined;
  }
}

export class LazyTaskResult {
  private resultPath: string;
  private _summary: string;
  private _status: string;
  private _loadedData?: Record<string, any>;

  constructor(
    public taskId: string,
    resultsDir: string,
    summary: string = "",
    status: string = "done",
  ) {
    this.resultPath = `${resultsDir}/${taskId}.json`;
    this._summary = summary;
    this._status = status;
  }

  get summary(): string {
    return this._summary;
  }

  get status(): string {
    return this._status;
  }

  private ensureLoaded(): Record<string, any> {
    if (this._loadedData) return this._loadedData;

    try {
      if (fs.existsSync(this.resultPath)) {
        const content = fs.readFileSync(this.resultPath, "utf-8");
        this._loadedData = JSON.parse(content);
      } else {
        this._loadedData = {};
      }
    } catch {
      this._loadedData = {};
    }

    return this._loadedData || {};
  }

  get output(): string {
    const data = this.ensureLoaded();
    return data.output || data.raw_content || "";
  }

  get filesModified(): string[] {
    const data = this.ensureLoaded();
    return data.files_modified || [];
  }

  get filesCreated(): string[] {
    const data = this.ensureLoaded();
    return data.files_created || [];
  }

  get error(): string | undefined {
    const data = this.ensureLoaded();
    return data.error;
  }

  toDict(includeOutput: boolean = false): Record<string, any> {
    const result: Record<string, any> = {
      taskId: this.taskId,
      status: this._status,
      summary: this._summary,
    };

    if (includeOutput) {
      Object.assign(result, this.ensureLoaded());
    }

    return result;
  }
}

// ============================================================================
// adv-perf-004: Batch Write Operations
// ============================================================================

interface BatchOperation {
  type: "write" | "append";
  path: string;
  data: any;
  timestamp: number;
}

export class BatchWriter {
  private queue: BatchOperation[] = [];
  private maxBatchSize: number;
  private maxDelay: number;
  private timer?: NodeJS.Timeout;
  private bytesQueued = 0;
  private maxBytes: number;
  private stats = {
    operationsBatched: 0,
    batchesFlushed: 0,
    bytesWritten: 0,
  };

  constructor(
    maxBatchSize: number = 100,
    maxDelayMs: number = 1000,
    maxBytes: number = 1024 * 1024,
  ) {
    this.maxBatchSize = maxBatchSize;
    this.maxDelay = maxDelayMs;
    this.maxBytes = maxBytes;
  }

  write(path: string, data: any): void {
    const dataStr = typeof data === "string" ? data : JSON.stringify(data);

    this.queue.push({
      type: "write",
      path,
      data: dataStr,
      timestamp: Date.now(),
    });

    this.bytesQueued += dataStr.length;
    this.stats.operationsBatched++;

    if (!this.timer) {
      this.timer = setTimeout(() => this.flush(), this.maxDelay);
    }

    if (this.shouldFlush()) {
      this.flush();
    }
  }

  append(path: string, data: any): void {
    const dataStr = typeof data === "string" ? data : JSON.stringify(data);

    this.queue.push({
      type: "append",
      path,
      data: dataStr,
      timestamp: Date.now(),
    });

    this.bytesQueued += dataStr.length;
    this.stats.operationsBatched++;

    if (!this.timer) {
      this.timer = setTimeout(() => this.flush(), this.maxDelay);
    }

    if (this.shouldFlush()) {
      this.flush();
    }
  }

  private shouldFlush(): boolean {
    return (
      this.queue.length >= this.maxBatchSize ||
      this.bytesQueued >= this.maxBytes
    );
  }

  flush(): number {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = undefined;
    }

    if (this.queue.length === 0) return 0;

    // Group by path
    const byPath = new Map<string, BatchOperation[]>();
    for (const op of this.queue) {
      if (!byPath.has(op.path)) {
        byPath.set(op.path, []);
      }
      byPath.get(op.path)!.push(op);
    }

    const count = this.queue.length;

    // Process each file
    for (const [path, ops] of byPath.entries()) {
      try {
        const coalesced = this.coalesceOperations(ops);
        if (coalesced) {
          // Ensure directory exists
          const dir = path.substring(0, path.lastIndexOf("/"));
          if (dir && !fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
          }

          fs.writeFileSync(path, coalesced);
          this.stats.bytesWritten += coalesced.length;
        }
      } catch (e) {
        console.error(`Error writing to ${path}:`, e);
      }
    }

    this.stats.batchesFlushed++;
    this.queue = [];
    this.bytesQueued = 0;

    return count;
  }

  private coalesceOperations(ops: BatchOperation[]): string | null {
    if (ops.length === 0) return null;

    // Sort by timestamp
    ops.sort((a, b) => a.timestamp - b.timestamp);

    const writes = ops.filter((op) => op.type === "write");
    const appends = ops.filter((op) => op.type === "append");

    if (writes.length > 0) {
      // Last write wins
      const lastWrite = writes[writes.length - 1];
      const laterAppends = appends.filter(
        (op) => op.timestamp > lastWrite.timestamp,
      );

      if (laterAppends.length > 0) {
        return lastWrite.data + laterAppends.map((op) => op.data).join("");
      }
      return lastWrite.data;
    }

    if (appends.length > 0) {
      return appends.map((op) => op.data).join("");
    }

    return null;
  }

  getStats() {
    return this.stats;
  }

  close(): void {
    this.flush();
  }
}

// ============================================================================
// adv-perf-006: Compression for Large Task Data
// ============================================================================

const gzipAsync = promisify(zlib.gzip);
const gunzipAsync = promisify(zlib.gunzip);

export interface CompressedData {
  _compressed: true;
  algorithm: "gzip" | "zlib";
  originalSize: number;
  compressedSize: number;
  data: string; // Base64 encoded
}

export async function compressData(
  data: any,
  minSizeThreshold: number = 1024,
): Promise<any | CompressedData> {
  const dataStr = typeof data === "string" ? data : JSON.stringify(data);
  const originalSize = Buffer.byteLength(dataStr, "utf8");

  if (originalSize < minSizeThreshold) {
    return data;
  }

  const compressed = await gzipAsync(Buffer.from(dataStr, "utf8"));
  const compressedSize = compressed.length;

  // Only use compression if it reduces size
  if (compressedSize >= originalSize) {
    return data;
  }

  return {
    _compressed: true,
    algorithm: "gzip",
    originalSize,
    compressedSize,
    data: compressed.toString("base64"),
  };
}

export async function decompressData(data: any): Promise<any> {
  if (
    typeof data !== "object" ||
    data === null ||
    !("_compressed" in data) ||
    !data._compressed
  ) {
    return data;
  }

  const compressed = Buffer.from(data.data, "base64");
  const decompressed = await gunzipAsync(compressed);
  const decompressedStr = decompressed.toString("utf8");

  try {
    return JSON.parse(decompressedStr);
  } catch {
    return decompressedStr;
  }
}

export function compressDataSync(
  data: any,
  minSizeThreshold: number = 1024,
): any | CompressedData {
  const dataStr = typeof data === "string" ? data : JSON.stringify(data);
  const originalSize = Buffer.byteLength(dataStr, "utf8");

  if (originalSize < minSizeThreshold) {
    return data;
  }

  const compressed = zlib.gzipSync(Buffer.from(dataStr, "utf8"));
  const compressedSize = compressed.length;

  if (compressedSize >= originalSize) {
    return data;
  }

  return {
    _compressed: true,
    algorithm: "gzip",
    originalSize,
    compressedSize,
    data: compressed.toString("base64"),
  };
}

export function decompressDataSync(data: any): any {
  if (
    typeof data !== "object" ||
    data === null ||
    !("_compressed" in data) ||
    !data._compressed
  ) {
    return data;
  }

  const compressed = Buffer.from(data.data, "base64");
  const decompressed = zlib.gunzipSync(compressed);
  const decompressedStr = decompressed.toString("utf8");

  try {
    return JSON.parse(decompressedStr);
  } catch {
    return decompressedStr;
  }
}

// ============================================================================
// adv-perf-010: Profiling Hooks for Bottleneck Detection
// ============================================================================

class Profiler {
  private enabled: boolean;
  private stats: Map<string, ProfileStats> = new Map();
  private samples: ProfileSample[] = [];
  private maxSamples = 10000;

  constructor(enabled: boolean = true) {
    this.enabled = enabled;
  }

  /**
   * Profile a synchronous function.
   */
  profile<T extends (...args: any[]) => any>(name: string, fn: T): T {
    if (!this.enabled) return fn;

    const self = this;
    return function (this: any, ...args: any[]) {
      const startTime = performance.now();
      let success = true;
      let error: string | undefined;

      try {
        return fn.apply(this, args);
      } catch (e) {
        success = false;
        error = String(e);
        throw e;
      } finally {
        const endTime = performance.now();
        self.recordSample({
          functionName: name,
          startTime,
          endTime,
          duration: endTime - startTime,
          success,
          error,
        });
      }
    } as T;
  }

  /**
   * Profile an async function.
   */
  profileAsync<T extends (...args: any[]) => Promise<any>>(
    name: string,
    fn: T,
  ): T {
    if (!this.enabled) return fn;

    const self = this;
    return async function (this: any, ...args: any[]) {
      const startTime = performance.now();
      let success = true;
      let error: string | undefined;

      try {
        return await fn.apply(this, args);
      } catch (e) {
        success = false;
        error = String(e);
        throw e;
      } finally {
        const endTime = performance.now();
        self.recordSample({
          functionName: name,
          startTime,
          endTime,
          duration: endTime - startTime,
          success,
          error,
        });
      }
    } as T;
  }

  /**
   * Measure a code block.
   */
  async measure<T>(name: string, fn: () => T | Promise<T>): Promise<T> {
    if (!this.enabled) return fn();

    const startTime = performance.now();
    let success = true;
    let error: string | undefined;

    try {
      const result = fn();
      return result instanceof Promise ? await result : result;
    } catch (e) {
      success = false;
      error = String(e);
      throw e;
    } finally {
      const endTime = performance.now();
      this.recordSample({
        functionName: name,
        startTime,
        endTime,
        duration: endTime - startTime,
        success,
        error,
      });
    }
  }

  private recordSample(sample: ProfileSample): void {
    this.samples.push(sample);

    // Keep samples bounded
    if (this.samples.length > this.maxSamples) {
      this.samples = this.samples.slice(-this.maxSamples / 2);
    }

    // Update stats
    let stat = this.stats.get(sample.functionName);
    if (!stat) {
      stat = {
        functionName: sample.functionName,
        callCount: 0,
        totalTime: 0,
        minTime: Infinity,
        maxTime: 0,
        avgTime: 0,
        errorCount: 0,
        p95Time: 0,
      };
      this.stats.set(sample.functionName, stat);
    }

    stat.callCount++;
    stat.totalTime += sample.duration;
    stat.minTime = Math.min(stat.minTime, sample.duration);
    stat.maxTime = Math.max(stat.maxTime, sample.duration);
    stat.avgTime = stat.totalTime / stat.callCount;

    if (!sample.success) {
      stat.errorCount++;
    }

    // Update p95 periodically
    if (stat.callCount % 100 === 0) {
      this.calculatePercentile(stat);
    }
  }

  private calculatePercentile(stat: ProfileStats): void {
    const samples = this.samples
      .filter((s) => s.functionName === stat.functionName)
      .map((s) => s.duration)
      .sort((a, b) => a - b);

    if (samples.length > 0) {
      stat.p95Time = samples[Math.floor(samples.length * 0.95)];
    }
  }

  getStats(): Map<string, ProfileStats> {
    return this.stats;
  }

  getTopFunctions(
    n: number = 10,
    by: "totalTime" | "avgTime" | "callCount" = "totalTime",
  ): ProfileStats[] {
    const statsList = Array.from(this.stats.values());

    switch (by) {
      case "totalTime":
        statsList.sort((a, b) => b.totalTime - a.totalTime);
        break;
      case "avgTime":
        statsList.sort((a, b) => b.avgTime - a.avgTime);
        break;
      case "callCount":
        statsList.sort((a, b) => b.callCount - a.callCount);
        break;
    }

    return statsList.slice(0, n);
  }

  reset(): void {
    this.stats.clear();
    this.samples = [];
  }

  report(): string {
    const lines = [
      "=".repeat(60),
      "PROFILING REPORT",
      "=".repeat(60),
      "",
      "Top 10 Functions by Total Time:",
      "-".repeat(40),
    ];

    for (const stat of this.getTopFunctions(10, "totalTime")) {
      lines.push(
        `  ${stat.functionName.padEnd(30)} ` +
          `total=${stat.totalTime.toFixed(2)}ms ` +
          `avg=${stat.avgTime.toFixed(2)}ms ` +
          `calls=${stat.callCount}`,
      );
    }

    lines.push("", "=".repeat(60));
    return lines.join("\n");
  }
}

// Global profiler instance
let globalProfiler: Profiler | null = null;

export function getProfiler(enabled: boolean = true): Profiler {
  if (!globalProfiler) {
    globalProfiler = new Profiler(enabled);
  }
  return globalProfiler;
}

// ============================================================================
// Integration: Performance-Enhanced State Manager
// ============================================================================

export class PerformanceEnhancedState {
  private taskCache: TaskCache<string, Task>;
  private taskIndex: TaskIndex;
  private batchWriter: BatchWriter;
  private profiler: Profiler;
  private stateFile: string;

  // Declare methods
  getTask: (taskId: string) => Task | undefined;
  addTask: (task: Task) => void;
  updateTask: (taskId: string, updates: Partial<Task>) => Task | undefined;
  getAvailableTasks: () => Task[];
  claimHighestPriorityTask: (agentId: string) => Task | undefined;

  constructor(stateFile: string) {
    this.stateFile = stateFile;
    this.taskCache = new TaskCache(1000);
    this.taskIndex = new TaskIndex();
    this.batchWriter = new BatchWriter();
    this.profiler = getProfiler();

    // Initialize profiled methods after profiler is available
    this.getTask = this.profiler.profile(
      "getTask",
      (taskId: string): Task | undefined => {
        // Check cache first
        const cached = this.taskCache.get(taskId);
        if (cached) return cached;

        // Check index
        const task = this.taskIndex.getTask(taskId);
        if (task) {
          this.taskCache.set(taskId, task);
        }
        return task;
      },
    );

    this.addTask = this.profiler.profile("addTask", (task: Task): void => {
      this.taskIndex.addTask(task);
      this.taskCache.set(task.id, task);
    });

    this.updateTask = this.profiler.profile(
      "updateTask",
      (taskId: string, updates: Partial<Task>): Task | undefined => {
        const task = this.taskIndex.getTask(taskId);
        if (!task) return undefined;

        const updated = { ...task, ...updates };
        this.taskIndex.updateTask(taskId, updated);
        this.taskCache.set(taskId, updated);

        // Batch the write
        this.batchWriter.write(
          this.stateFile,
          JSON.stringify({ tasks: Array.from(this.getAllTasks()) }, null, 2),
        );

        return updated;
      },
    );

    this.getAvailableTasks = this.profiler.profile(
      "getAvailableTasks",
      (): Task[] => {
        return getCachedQuery(
          "available_tasks",
          () => this.taskIndex.getAvailableWithDependenciesSatisfied(),
          5000, // 5 second TTL
        );
      },
    );

    this.claimHighestPriorityTask = this.profiler.profile(
      "claimHighestPriorityTask",
      (agentId: string): Task | undefined => {
        const task = this.taskIndex.getHighestPriorityAvailable();
        if (!task) return undefined;

        const updated = {
          ...task,
          status: "claimed" as const,
          claimed_by: agentId,
          claimed_at: new Date().toISOString(),
        };

        this.updateTask(task.id, updated);
        invalidateQueryCache("available");
        return updated;
      },
    );
  }

  *getAllTasks(): Iterable<Task> {
    for (const [, task] of this.taskIndex["tasks"]) {
      yield task;
    }
  }

  getPerformanceReport(): string {
    const cacheStats = this.taskCache.getStats();
    const indexStats = this.taskIndex.getStats();
    const writerStats = this.batchWriter.getStats();

    return [
      "Performance Report",
      "=".repeat(40),
      "",
      "Cache Stats:",
      `  Hits: ${cacheStats.hits}`,
      `  Misses: ${cacheStats.misses}`,
      `  Hit Rate: ${(cacheStats.hitRate * 100).toFixed(1)}%`,
      `  Size: ${cacheStats.size}`,
      "",
      "Index Stats:",
      `  Total Tasks: ${indexStats.totalTasks}`,
      `  Completed: ${indexStats.completedTasks}`,
      "",
      "Batch Writer Stats:",
      `  Operations Batched: ${writerStats.operationsBatched}`,
      `  Batches Flushed: ${writerStats.batchesFlushed}`,
      `  Bytes Written: ${writerStats.bytesWritten}`,
      "",
      this.profiler.report(),
    ].join("\n");
  }

  close(): void {
    this.batchWriter.close();
  }
}

export default {
  TaskCache,
  TaskIndex,
  LazyValue,
  LazyTaskResult,
  BatchWriter,
  compressData,
  decompressData,
  compressDataSync,
  decompressDataSync,
  getProfiler,
  getCachedQuery,
  invalidateQueryCache,
  PerformanceEnhancedState,
};
