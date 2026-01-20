/**
 * Reliability Integration for Option B (MCP Server)
 *
 * Implements reliability patterns in TypeScript for the MCP broker:
 * - Circuit breaker for external calls
 * - Automatic retry with jitter
 * - Fallback strategies
 * - Deadlock detection
 * - Data consistency validation
 * - Backup and restore
 * - Leader election
 * - Split-brain prevention
 * - Graceful degradation
 * - Self-healing mechanisms
 */

// ============================================================================
// Circuit Breaker (adv-rel-001)
// ============================================================================

export enum CircuitState {
  CLOSED = "closed",
  OPEN = "open",
  HALF_OPEN = "half_open",
}

export interface CircuitBreakerConfig {
  failureThreshold: number;
  successThreshold: number;
  timeoutMs: number;
  windowMs: number;
  halfOpenMaxCalls: number;
}

export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failures: number[] = [];
  private halfOpenSuccesses = 0;
  private halfOpenFailures = 0;
  private halfOpenCalls = 0;
  private lastStateChange = Date.now();
  private stats = {
    totalCalls: 0,
    successfulCalls: 0,
    failedCalls: 0,
    rejectedCalls: 0,
  };

  constructor(
    public readonly name: string,
    private config: CircuitBreakerConfig = {
      failureThreshold: 5,
      successThreshold: 3,
      timeoutMs: 30000,
      windowMs: 60000,
      halfOpenMaxCalls: 3,
    },
    private onStateChange?: (
      name: string,
      oldState: CircuitState,
      newState: CircuitState,
    ) => void,
  ) {}

  get currentState(): CircuitState {
    this.checkStateTransition();
    return this.state;
  }

  get isOpen(): boolean {
    return this.currentState === CircuitState.OPEN;
  }

  canExecute(): boolean {
    this.checkStateTransition();

    if (this.state === CircuitState.CLOSED) {
      return true;
    } else if (this.state === CircuitState.HALF_OPEN) {
      if (this.halfOpenCalls < this.config.halfOpenMaxCalls) {
        this.halfOpenCalls++;
        return true;
      }
      return false;
    }
    return false;
  }

  recordSuccess(): void {
    this.stats.totalCalls++;
    this.stats.successfulCalls++;

    if (this.state === CircuitState.HALF_OPEN) {
      this.halfOpenSuccesses++;
      this.halfOpenCalls = Math.max(0, this.halfOpenCalls - 1);

      if (this.halfOpenSuccesses >= this.config.successThreshold) {
        this.transitionTo(CircuitState.CLOSED);
      }
    }
  }

  recordFailure(error?: Error): void {
    this.stats.totalCalls++;
    this.stats.failedCalls++;

    const now = Date.now();
    this.failures.push(now);
    this.cleanupOldFailures();

    if (this.state === CircuitState.HALF_OPEN) {
      this.halfOpenFailures++;
      this.halfOpenCalls = Math.max(0, this.halfOpenCalls - 1);
      this.transitionTo(CircuitState.OPEN);
    } else if (this.state === CircuitState.CLOSED) {
      if (this.failures.length >= this.config.failureThreshold) {
        this.transitionTo(CircuitState.OPEN);
      }
    }

    console.error(
      `CircuitBreaker ${this.name}: failure recorded (state=${this.state}, failures=${this.failures.length})`,
    );
  }

  reset(): void {
    this.transitionTo(CircuitState.CLOSED);
    this.failures = [];
    this.resetHalfOpenCounters();
  }

  private checkStateTransition(): void {
    if (this.state === CircuitState.OPEN) {
      const elapsed = Date.now() - this.lastStateChange;
      if (elapsed >= this.config.timeoutMs) {
        this.transitionTo(CircuitState.HALF_OPEN);
      }
    }
  }

  private transitionTo(newState: CircuitState): void {
    if (this.state === newState) return;

    const oldState = this.state;
    this.state = newState;
    this.lastStateChange = Date.now();

    if (newState === CircuitState.HALF_OPEN) {
      this.resetHalfOpenCounters();
    } else if (newState === CircuitState.CLOSED) {
      this.failures = [];
    }

    console.log(`CircuitBreaker ${this.name}: ${oldState} -> ${newState}`);

    if (this.onStateChange) {
      try {
        this.onStateChange(this.name, oldState, newState);
      } catch (e) {
        console.error("CircuitBreaker callback error:", e);
      }
    }
  }

  private resetHalfOpenCounters(): void {
    this.halfOpenSuccesses = 0;
    this.halfOpenFailures = 0;
    this.halfOpenCalls = 0;
  }

  private cleanupOldFailures(): void {
    const cutoff = Date.now() - this.config.windowMs;
    this.failures = this.failures.filter((t) => t > cutoff);
  }

  toJSON(): object {
    return {
      name: this.name,
      state: this.state,
      config: this.config,
      stats: this.stats,
      currentFailures: this.failures.length,
    };
  }
}

// ============================================================================
// Retry with Jitter (adv-rel-002)
// ============================================================================

export interface RetryConfig {
  maxAttempts: number;
  initialDelayMs: number;
  maxDelayMs: number;
  jitterFactor: number;
}

export class RetryWithJitter {
  constructor(
    private config: RetryConfig = {
      maxAttempts: 3,
      initialDelayMs: 1000,
      maxDelayMs: 60000,
      jitterFactor: 0.5,
    },
    private onRetry?: (attempt: number, error: Error, delayMs: number) => void,
  ) {}

  calculateDelay(attempt: number): number {
    const baseDelay = this.config.initialDelayMs * Math.pow(2, attempt);
    const cappedDelay = Math.min(baseDelay, this.config.maxDelayMs);

    if (this.config.jitterFactor > 0) {
      const minDelay = cappedDelay * (1 - this.config.jitterFactor);
      return minDelay + Math.random() * (cappedDelay - minDelay);
    }

    return cappedDelay;
  }

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < this.config.maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;

        if (attempt + 1 >= this.config.maxAttempts) {
          throw lastError;
        }

        const delay = this.calculateDelay(attempt);

        if (this.onRetry) {
          try {
            this.onRetry(attempt + 1, lastError, delay);
          } catch (e) {
            console.error("Retry callback error:", e);
          }
        }

        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }

    throw lastError;
  }
}

// ============================================================================
// Fallback Strategies (adv-rel-003)
// ============================================================================

export interface FallbackResult<T> {
  value: T | null;
  fallbackUsed: boolean;
  fallbackType?: string;
  originalError?: Error;
}

export abstract class FallbackStrategy<T> {
  constructor(public readonly name: string) {}

  abstract execute(error: Error, context: Record<string, unknown>): T | null;
}

export class CacheFallback<T> extends FallbackStrategy<T> {
  private cache = new Map<string, { value: T; timestamp: number }>();

  constructor(private maxAgeMs = 3600000) {
    super("cache");
  }

  setCache(key: string, value: T): void {
    this.cache.set(key, { value, timestamp: Date.now() });
  }

  execute(error: Error, context: Record<string, unknown>): T | null {
    const key = (context.cacheKey as string) || "default";
    const entry = this.cache.get(key);

    if (entry && Date.now() - entry.timestamp < this.maxAgeMs) {
      console.log(`CacheFallback: returning cached value for key '${key}'`);
      return entry.value;
    }

    return null;
  }
}

export class DefaultFallback<T> extends FallbackStrategy<T> {
  constructor(private defaultValue: T) {
    super("default");
  }

  execute(): T {
    return this.defaultValue;
  }
}

export class FallbackChain<T> {
  private strategies: FallbackStrategy<T>[] = [];

  add(strategy: FallbackStrategy<T>): FallbackChain<T> {
    this.strategies.push(strategy);
    return this;
  }

  execute(
    error: Error,
    context: Record<string, unknown> = {},
  ): FallbackResult<T> {
    for (const strategy of this.strategies) {
      try {
        const value = strategy.execute(error, context);
        if (value !== null) {
          return {
            value,
            fallbackUsed: true,
            fallbackType: strategy.name,
            originalError: error,
          };
        }
      } catch (e) {
        console.warn(`FallbackChain: strategy '${strategy.name}' failed:`, e);
        continue;
      }
    }

    return {
      value: null,
      fallbackUsed: false,
      originalError: error,
    };
  }
}

// ============================================================================
// Deadlock Detection (adv-rel-004)
// ============================================================================

export interface DeadlockInfo {
  type: "circular_dependency" | "stale_lock" | "orphaned_task";
  tasksInvolved: string[];
  detectedAt: Date;
  details: Record<string, unknown>;
}

export class DeadlockDetector {
  private tasks = new Map<
    string,
    {
      status: string;
      claimedBy?: string;
      dependencies: Set<string>;
      claimedAt?: Date;
    }
  >();
  private activeWorkers = new Map<string, Date>();
  private detectedDeadlocks: DeadlockInfo[] = [];
  private intervalId?: ReturnType<typeof setInterval>;

  constructor(
    private staleThresholdMs = 300000,
    private checkIntervalMs = 30000,
    private onDeadlock?: (deadlock: DeadlockInfo) => void,
  ) {}

  registerTask(
    taskId: string,
    dependencies: string[] = [],
    status = "available",
  ): void {
    this.tasks.set(taskId, {
      status,
      dependencies: new Set(dependencies),
    });
  }

  updateTask(taskId: string, status?: string, claimedBy?: string): void {
    const task = this.tasks.get(taskId);
    if (task) {
      if (status) task.status = status;
      if (claimedBy) {
        task.claimedBy = claimedBy;
        task.claimedAt = new Date();
      }
    }
  }

  registerWorker(workerId: string): void {
    this.activeWorkers.set(workerId, new Date());
  }

  workerHeartbeat(workerId: string): void {
    this.activeWorkers.set(workerId, new Date());
  }

  detectCircularDependencies(): DeadlockInfo[] {
    const deadlocks: DeadlockInfo[] = [];
    const visited = new Set<string>();
    const recStack = new Set<string>();

    const dfs = (taskId: string, path: string[]): string[] | null => {
      visited.add(taskId);
      recStack.add(taskId);
      path.push(taskId);

      const task = this.tasks.get(taskId);
      if (!task) {
        recStack.delete(taskId);
        path.pop();
        return null;
      }

      for (const depId of task.dependencies) {
        if (!visited.has(depId)) {
          const result = dfs(depId, path);
          if (result) return result;
        } else if (recStack.has(depId)) {
          const idx = path.indexOf(depId);
          return [...path.slice(idx), depId];
        }
      }

      recStack.delete(taskId);
      path.pop();
      return null;
    };

    for (const taskId of this.tasks.keys()) {
      if (!visited.has(taskId)) {
        const cycle = dfs(taskId, []);
        if (cycle) {
          deadlocks.push({
            type: "circular_dependency",
            tasksInvolved: cycle,
            detectedAt: new Date(),
            details: { cycle: cycle.join(" -> ") },
          });
        }
      }
    }

    return deadlocks;
  }

  detectStaleLocks(): DeadlockInfo[] {
    const deadlocks: DeadlockInfo[] = [];
    const now = Date.now();

    for (const [taskId, task] of this.tasks) {
      if (
        (task.status === "claimed" || task.status === "in_progress") &&
        task.claimedAt
      ) {
        const age = now - task.claimedAt.getTime();
        if (age > this.staleThresholdMs) {
          deadlocks.push({
            type: "stale_lock",
            tasksInvolved: [taskId],
            detectedAt: new Date(),
            details: {
              claimedBy: task.claimedBy,
              claimedAt: task.claimedAt.toISOString(),
              ageMs: age,
            },
          });
        }
      }
    }

    return deadlocks;
  }

  detectOrphanedTasks(): DeadlockInfo[] {
    const deadlocks: DeadlockInfo[] = [];
    const now = Date.now();
    const workerTimeout = 60000;

    const inactiveWorkers = new Set<string>();
    for (const [workerId, lastSeen] of this.activeWorkers) {
      if (now - lastSeen.getTime() > workerTimeout) {
        inactiveWorkers.add(workerId);
      }
    }

    for (const [taskId, task] of this.tasks) {
      if (
        (task.status === "claimed" || task.status === "in_progress") &&
        task.claimedBy &&
        inactiveWorkers.has(task.claimedBy)
      ) {
        deadlocks.push({
          type: "orphaned_task",
          tasksInvolved: [taskId],
          detectedAt: new Date(),
          details: {
            claimedBy: task.claimedBy,
            workerLastSeen: this.activeWorkers
              .get(task.claimedBy)
              ?.toISOString(),
          },
        });
      }
    }

    return deadlocks;
  }

  runDetection(): DeadlockInfo[] {
    const allDeadlocks = [
      ...this.detectCircularDependencies(),
      ...this.detectStaleLocks(),
      ...this.detectOrphanedTasks(),
    ];

    for (const deadlock of allDeadlocks) {
      console.warn(
        `Deadlock detected: ${deadlock.type}`,
        deadlock.tasksInvolved,
      );
      this.detectedDeadlocks.push(deadlock);

      if (this.onDeadlock) {
        try {
          this.onDeadlock(deadlock);
        } catch (e) {
          console.error("Deadlock callback error:", e);
        }
      }
    }

    return allDeadlocks;
  }

  startMonitoring(): void {
    if (this.intervalId) return;
    this.intervalId = setInterval(
      () => this.runDetection(),
      this.checkIntervalMs,
    );
  }

  stopMonitoring(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
  }
}

// ============================================================================
// Data Consistency Validation (adv-rel-005)
// ============================================================================

export interface ValidationError {
  field: string;
  message: string;
  severity: "error" | "warning";
}

export interface ValidationReport {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
  checksum?: string;
}

export class DataConsistencyValidator {
  private static VALID_TRANSITIONS: Record<string, Set<string>> = {
    pending: new Set(["available", "cancelled"]),
    available: new Set(["claimed", "cancelled"]),
    claimed: new Set(["in_progress", "available", "failed"]),
    in_progress: new Set(["done", "failed", "available"]),
    done: new Set(),
    failed: new Set(["available"]),
  };

  validateTask(task: Record<string, unknown>): ValidationReport {
    const errors: ValidationError[] = [];
    const warnings: ValidationError[] = [];

    // Required fields
    const required = ["id", "description", "status", "priority"];
    for (const field of required) {
      if (!(field in task)) {
        errors.push({
          field,
          message: `Required field '${field}' is missing`,
          severity: "error",
        });
      }
    }

    // Validate status
    const status = task.status as string;
    if (status && !(status in DataConsistencyValidator.VALID_TRANSITIONS)) {
      errors.push({
        field: "status",
        message: `Invalid status: ${status}`,
        severity: "error",
      });
    }

    // Validate priority
    const priority = task.priority as number;
    if (priority !== undefined) {
      if (!Number.isInteger(priority) || priority < 1 || priority > 10) {
        errors.push({
          field: "priority",
          message: `Priority must be integer 1-10, got: ${priority}`,
          severity: "error",
        });
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
    };
  }

  validateTransition(
    currentStatus: string,
    newStatus: string,
  ): ValidationReport {
    const errors: ValidationError[] = [];

    const validNext = DataConsistencyValidator.VALID_TRANSITIONS[currentStatus];
    if (!validNext) {
      errors.push({
        field: "current_status",
        message: `Invalid current status: ${currentStatus}`,
        severity: "error",
      });
    } else if (!validNext.has(newStatus)) {
      errors.push({
        field: "status",
        message: `Invalid transition: ${currentStatus} -> ${newStatus}`,
        severity: "error",
      });
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings: [],
    };
  }

  computeChecksum(data: Record<string, unknown>): string {
    const json = JSON.stringify(data, Object.keys(data).sort());
    // Simple hash for TypeScript
    let hash = 0;
    for (let i = 0; i < json.length; i++) {
      const char = json.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(8, "0");
  }
}

// ============================================================================
// Backup Manager (adv-rel-006)
// ============================================================================

export interface BackupInfo {
  id: string;
  createdAt: Date;
  checksum: string;
  size: number;
}

export class BackupManager {
  private backups: Map<string, { data: unknown; info: BackupInfo }> = new Map();
  private maxBackups = 10;
  private autoBackupInterval?: ReturnType<typeof setInterval>;

  constructor(private onBackup?: (info: BackupInfo) => void) {}

  createBackup(state: unknown, description = ""): BackupInfo {
    const id = `backup-${Date.now()}`;
    const json = JSON.stringify(state);
    const checksum = this.computeChecksum(json);

    const info: BackupInfo = {
      id,
      createdAt: new Date(),
      checksum,
      size: json.length,
    };

    this.backups.set(id, { data: state, info });
    this.cleanupOldBackups();

    console.log(`Created backup: ${id} (${info.size} bytes)`);

    if (this.onBackup) {
      try {
        this.onBackup(info);
      } catch (e) {
        console.error("Backup callback error:", e);
      }
    }

    return info;
  }

  restore(backupId: string): unknown | null {
    const backup = this.backups.get(backupId);
    if (backup) {
      console.log(`Restored backup: ${backupId}`);
      return JSON.parse(JSON.stringify(backup.data));
    }
    return null;
  }

  listBackups(): BackupInfo[] {
    return Array.from(this.backups.values())
      .map((b) => b.info)
      .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  }

  getLatestBackup(): BackupInfo | null {
    const list = this.listBackups();
    return list.length > 0 ? list[0] : null;
  }

  startAutoBackup(getState: () => unknown, intervalMs = 300000): void {
    if (this.autoBackupInterval) return;
    this.autoBackupInterval = setInterval(() => {
      try {
        this.createBackup(getState(), "Auto backup");
      } catch (e) {
        console.error("Auto backup failed:", e);
      }
    }, intervalMs);
  }

  stopAutoBackup(): void {
    if (this.autoBackupInterval) {
      clearInterval(this.autoBackupInterval);
      this.autoBackupInterval = undefined;
    }
  }

  private cleanupOldBackups(): void {
    while (this.backups.size > this.maxBackups) {
      const oldest = this.listBackups().pop();
      if (oldest) {
        this.backups.delete(oldest.id);
      }
    }
  }

  private computeChecksum(data: string): string {
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(8, "0");
  }
}

// ============================================================================
// Graceful Degradation (adv-rel-009)
// ============================================================================

export enum DegradationLevel {
  NORMAL = "normal",
  LIGHT = "light",
  MODERATE = "moderate",
  HEAVY = "heavy",
  CRITICAL = "critical",
  EMERGENCY = "emergency",
}

export class GracefulDegradation {
  private currentLevel = DegradationLevel.NORMAL;
  private lastLevelChange = new Date();
  private recoveryDelayMs = 60000;

  private policies: Record<
    DegradationLevel,
    { maxWorkers: number; maxTasksPerMinute: number }
  > = {
    [DegradationLevel.NORMAL]: { maxWorkers: 10, maxTasksPerMinute: 100 },
    [DegradationLevel.LIGHT]: { maxWorkers: 8, maxTasksPerMinute: 80 },
    [DegradationLevel.MODERATE]: { maxWorkers: 5, maxTasksPerMinute: 50 },
    [DegradationLevel.HEAVY]: { maxWorkers: 3, maxTasksPerMinute: 20 },
    [DegradationLevel.CRITICAL]: { maxWorkers: 1, maxTasksPerMinute: 5 },
    [DegradationLevel.EMERGENCY]: { maxWorkers: 0, maxTasksPerMinute: 0 },
  };

  constructor(
    private onDegrade?: (
      oldLevel: DegradationLevel,
      newLevel: DegradationLevel,
    ) => void,
    private onRecover?: (
      oldLevel: DegradationLevel,
      newLevel: DegradationLevel,
    ) => void,
  ) {}

  get level(): DegradationLevel {
    return this.currentLevel;
  }

  get maxWorkers(): number {
    return this.policies[this.currentLevel].maxWorkers;
  }

  get rateLimit(): number {
    return this.policies[this.currentLevel].maxTasksPerMinute;
  }

  setLevel(level: DegradationLevel): void {
    if (level === this.currentLevel) return;

    const oldLevel = this.currentLevel;
    const levels = Object.values(DegradationLevel);
    const isDegrading = levels.indexOf(level) > levels.indexOf(oldLevel);

    this.currentLevel = level;
    this.lastLevelChange = new Date();

    console.log(`Degradation: ${oldLevel} -> ${level}`);

    if (isDegrading && this.onDegrade) {
      try {
        this.onDegrade(oldLevel, level);
      } catch (e) {
        console.error("onDegrade callback error:", e);
      }
    } else if (!isDegrading && this.onRecover) {
      try {
        this.onRecover(oldLevel, level);
      } catch (e) {
        console.error("onRecover callback error:", e);
      }
    }
  }

  degrade(): DegradationLevel {
    const levels = Object.values(DegradationLevel);
    const currentIdx = levels.indexOf(this.currentLevel);

    if (currentIdx < levels.length - 1) {
      this.setLevel(levels[currentIdx + 1]);
    }

    return this.currentLevel;
  }

  recover(): DegradationLevel {
    const elapsed = Date.now() - this.lastLevelChange.getTime();
    if (elapsed < this.recoveryDelayMs) {
      return this.currentLevel;
    }

    const levels = Object.values(DegradationLevel);
    const currentIdx = levels.indexOf(this.currentLevel);

    if (currentIdx > 0) {
      this.setLevel(levels[currentIdx - 1]);
    }

    return this.currentLevel;
  }

  emergencyStop(): void {
    this.setLevel(DegradationLevel.EMERGENCY);
  }

  reset(): void {
    this.setLevel(DegradationLevel.NORMAL);
  }
}

// ============================================================================
// Self-Healing Manager (adv-rel-010)
// ============================================================================

export enum HealthStatus {
  HEALTHY = "healthy",
  DEGRADED = "degraded",
  UNHEALTHY = "unhealthy",
  UNKNOWN = "unknown",
}

export interface HealthCheckResult {
  name: string;
  status: HealthStatus;
  message: string;
  checkedAt: Date;
}

export abstract class HealthCheck {
  protected consecutiveFailures = 0;
  protected consecutiveSuccesses = 0;
  protected currentStatus = HealthStatus.UNKNOWN;
  protected lastResult?: HealthCheckResult;

  constructor(
    public readonly name: string,
    protected intervalMs = 30000,
    protected failureThreshold = 3,
    protected successThreshold = 1,
  ) {}

  get status(): HealthStatus {
    return this.currentStatus;
  }

  abstract check(): HealthCheckResult;
  abstract recover(): { success: boolean; message: string };

  runCheck(): HealthCheckResult {
    const result = this.check();
    this.lastResult = result;

    if (result.status === HealthStatus.HEALTHY) {
      this.consecutiveFailures = 0;
      this.consecutiveSuccesses++;

      if (this.consecutiveSuccesses >= this.successThreshold) {
        this.currentStatus = HealthStatus.HEALTHY;
      }
    } else if (result.status === HealthStatus.UNHEALTHY) {
      this.consecutiveSuccesses = 0;
      this.consecutiveFailures++;

      if (this.consecutiveFailures >= this.failureThreshold) {
        this.currentStatus = HealthStatus.UNHEALTHY;
      }
    } else if (result.status === HealthStatus.DEGRADED) {
      this.currentStatus = HealthStatus.DEGRADED;
    }

    return result;
  }
}

export class SelfHealingManager {
  private checks = new Map<string, HealthCheck>();
  private recoveryAttempts = new Map<string, number>();
  private lastRecovery = new Map<string, Date>();
  private intervalId?: ReturnType<typeof setInterval>;

  constructor(
    private recoveryCooldownMs = 60000,
    private maxRecoveryAttempts = 3,
    private onUnhealthy?: (result: HealthCheckResult) => void,
    private onRecovery?: (
      name: string,
      success: boolean,
      message: string,
    ) => void,
  ) {}

  registerCheck(check: HealthCheck): void {
    this.checks.set(check.name, check);
    this.recoveryAttempts.set(check.name, 0);
  }

  runAllChecks(): Map<string, HealthCheckResult> {
    const results = new Map<string, HealthCheckResult>();

    for (const [name, check] of this.checks) {
      const result = check.runCheck();
      results.set(name, result);

      if (result.status === HealthStatus.UNHEALTHY) {
        console.warn(`Health check failed: ${name} - ${result.message}`);

        if (this.onUnhealthy) {
          try {
            this.onUnhealthy(result);
          } catch (e) {
            console.error("onUnhealthy callback error:", e);
          }
        }
      }
    }

    return results;
  }

  attemptRecovery(
    checkName: string,
  ): { success: boolean; message: string } | null {
    const check = this.checks.get(checkName);
    if (!check) return null;

    const last = this.lastRecovery.get(checkName);
    if (last && Date.now() - last.getTime() < this.recoveryCooldownMs) {
      return null;
    }

    const attempts = this.recoveryAttempts.get(checkName) || 0;
    if (attempts >= this.maxRecoveryAttempts) {
      console.warn(`Max recovery attempts reached for ${checkName}`);
      return null;
    }

    console.log(
      `Attempting recovery for ${checkName} (attempt ${attempts + 1})`,
    );

    const result = check.recover();
    this.lastRecovery.set(checkName, new Date());

    if (result.success) {
      this.recoveryAttempts.set(checkName, 0);
    } else {
      this.recoveryAttempts.set(checkName, attempts + 1);
    }

    if (this.onRecovery) {
      try {
        this.onRecovery(checkName, result.success, result.message);
      } catch (e) {
        console.error("onRecovery callback error:", e);
      }
    }

    return result;
  }

  healAll(): void {
    const results = this.runAllChecks();

    for (const [name, result] of results) {
      if (result.status === HealthStatus.UNHEALTHY) {
        this.attemptRecovery(name);
      }
    }
  }

  startMonitoring(minIntervalMs = 5000): void {
    if (this.intervalId) return;
    this.intervalId = setInterval(() => this.healAll(), minIntervalMs);
    console.log("Started self-healing monitoring");
  }

  stopMonitoring(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
      console.log("Stopped self-healing monitoring");
    }
  }

  getHealthSummary(): Record<string, unknown> {
    const checks: Record<string, unknown> = {};

    for (const [name, check] of this.checks) {
      checks[name] = {
        status: check.status,
        recoveryAttempts: this.recoveryAttempts.get(name) || 0,
      };
    }

    const healthy = Array.from(this.checks.values()).filter(
      (c) => c.status === HealthStatus.HEALTHY,
    ).length;

    return {
      overallStatus:
        healthy === this.checks.size
          ? HealthStatus.HEALTHY
          : healthy > 0
            ? HealthStatus.DEGRADED
            : HealthStatus.UNHEALTHY,
      healthyChecks: healthy,
      totalChecks: this.checks.size,
      checks,
    };
  }
}

// ============================================================================
// Leader Election (adv-rel-007) - Simplified for in-memory
// ============================================================================

export class LeaderElection {
  private isLeader = false;
  private term = 0;
  private lastHeartbeat = new Date();

  constructor(
    public readonly nodeId: string,
    private onBecomeLeader?: () => void,
    private onLoseLeadership?: () => void,
  ) {}

  get currentTerm(): number {
    return this.term;
  }

  get leader(): boolean {
    return this.isLeader;
  }

  becomeLeader(): boolean {
    if (this.isLeader) return true;

    this.isLeader = true;
    this.term++;
    this.lastHeartbeat = new Date();

    console.log(`Node ${this.nodeId} became leader (term ${this.term})`);

    if (this.onBecomeLeader) {
      try {
        this.onBecomeLeader();
      } catch (e) {
        console.error("onBecomeLeader callback error:", e);
      }
    }

    return true;
  }

  stepDown(): void {
    if (!this.isLeader) return;

    this.isLeader = false;
    console.log(`Node ${this.nodeId} stepped down from leadership`);

    if (this.onLoseLeadership) {
      try {
        this.onLoseLeadership();
      } catch (e) {
        console.error("onLoseLeadership callback error:", e);
      }
    }
  }

  heartbeat(): void {
    this.lastHeartbeat = new Date();
  }

  getStatus(): Record<string, unknown> {
    return {
      nodeId: this.nodeId,
      isLeader: this.isLeader,
      term: this.term,
      lastHeartbeat: this.lastHeartbeat.toISOString(),
    };
  }
}

// ============================================================================
// Coordination Reliability Manager
// ============================================================================

export class CoordinationReliability {
  public readonly circuitBreaker: CircuitBreaker;
  public readonly retry: RetryWithJitter;
  public readonly fallbackChain: FallbackChain<unknown>;
  public readonly deadlockDetector: DeadlockDetector;
  public readonly validator: DataConsistencyValidator;
  public readonly backupManager: BackupManager;
  public readonly degradation: GracefulDegradation;
  public readonly selfHealing: SelfHealingManager;
  public readonly leaderElection: LeaderElection;

  constructor(nodeId: string) {
    this.circuitBreaker = new CircuitBreaker("mcp-operations");
    this.retry = new RetryWithJitter();
    this.fallbackChain = new FallbackChain<unknown>()
      .add(new CacheFallback())
      .add(new DefaultFallback({ tasks: [], agents: {} }));
    this.deadlockDetector = new DeadlockDetector();
    this.validator = new DataConsistencyValidator();
    this.backupManager = new BackupManager();
    this.degradation = new GracefulDegradation();
    this.selfHealing = new SelfHealingManager();
    this.leaderElection = new LeaderElection(nodeId);
  }

  start(getState: () => unknown): void {
    this.deadlockDetector.startMonitoring();
    this.backupManager.startAutoBackup(getState);
    this.selfHealing.startMonitoring();
  }

  stop(): void {
    this.deadlockDetector.stopMonitoring();
    this.backupManager.stopAutoBackup();
    this.selfHealing.stopMonitoring();
  }

  getHealthStatus(): Record<string, unknown> {
    return {
      circuitBreaker: this.circuitBreaker.toJSON(),
      degradation: {
        level: this.degradation.level,
        maxWorkers: this.degradation.maxWorkers,
        rateLimit: this.degradation.rateLimit,
      },
      selfHealing: this.selfHealing.getHealthSummary(),
      leader: this.leaderElection.getStatus(),
      backups: this.backupManager.listBackups().length,
    };
  }
}
