/**
 * Rate Limiting Middleware for MCP Server (PROD-004)
 *
 * Implements rate limiting using a hybrid token bucket + sliding window algorithm.
 * Provides per-client/IP rate limiting with configurable limits.
 *
 * Features:
 * - Token bucket algorithm for burst handling
 * - Sliding window for accurate rate tracking
 * - Per-client/IP limits
 * - Standard rate limit headers (X-RateLimit-*)
 * - Configurable limits and windows
 * - Automatic cleanup of stale entries
 */

import * as crypto from "crypto";

// ============================================================================
// Types
// ============================================================================

export interface RateLimitConfig {
  /** Maximum requests per window (default: 100) */
  maxRequests: number;
  /** Time window in milliseconds (default: 60000 = 1 minute) */
  windowMs: number;
  /** Burst limit - max requests in a short burst (default: 10) */
  burstLimit: number;
  /** Burst window in milliseconds (default: 1000 = 1 second) */
  burstWindowMs: number;
  /** Whether rate limiting is enabled (default: true) */
  enabled: boolean;
  /** Skip rate limiting for certain client IDs */
  skipList: string[];
  /** Custom key generator function */
  keyGenerator?: (request: RateLimitRequest) => string;
  /** Callback when rate limit is exceeded */
  onRateLimitExceeded?: (clientId: string, info: RateLimitInfo) => void;
}

export interface RateLimitRequest {
  /** Client identifier (IP, agent ID, API key, etc.) */
  clientId?: string;
  /** IP address of the client */
  ip?: string;
  /** Agent ID if available */
  agentId?: string;
  /** Tool name being called */
  toolName?: string;
  /** Request timestamp */
  timestamp?: number;
}

export interface RateLimitResult {
  /** Whether the request is allowed */
  allowed: boolean;
  /** Number of remaining requests in the current window */
  remaining: number;
  /** Maximum requests allowed in the window */
  limit: number;
  /** Timestamp when the rate limit resets (Unix ms) */
  resetAt: number;
  /** Seconds until the rate limit resets */
  retryAfter?: number;
  /** Current window usage count */
  currentCount: number;
  /** Information about the rate limit state */
  info: RateLimitInfo;
}

export interface RateLimitInfo {
  /** Client identifier used for rate limiting */
  clientId: string;
  /** Requests made in current window */
  windowCount: number;
  /** Requests made in current burst window */
  burstCount: number;
  /** Whether this is a burst violation */
  burstViolation: boolean;
  /** Whether this is a window violation */
  windowViolation: boolean;
  /** Window start timestamp */
  windowStart: number;
  /** Window end timestamp */
  windowEnd: number;
}

export interface RateLimitHeaders {
  "X-RateLimit-Limit": string;
  "X-RateLimit-Remaining": string;
  "X-RateLimit-Reset": string;
  "X-RateLimit-Policy"?: string;
  "Retry-After"?: string;
}

// Internal bucket structure for sliding window
interface SlidingWindowBucket {
  timestamps: number[];
  burstTimestamps: number[];
  lastCleanup: number;
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_RATE_LIMIT_CONFIG: RateLimitConfig = {
  maxRequests: 100,
  windowMs: 60000, // 1 minute
  burstLimit: 10,
  burstWindowMs: 1000, // 1 second
  enabled: true,
  skipList: [],
};

// ============================================================================
// Rate Limit Middleware Class
// ============================================================================

export class RateLimitMiddleware {
  private config: RateLimitConfig;
  private buckets: Map<string, SlidingWindowBucket> = new Map();
  private cleanupInterval: NodeJS.Timeout | null = null;
  private readonly CLEANUP_INTERVAL_MS = 60000; // Clean up every minute

  constructor(config: Partial<RateLimitConfig> = {}) {
    this.config = { ...DEFAULT_RATE_LIMIT_CONFIG, ...config };
    this.startCleanupInterval();
  }

  /**
   * Check if a request should be rate limited
   */
  checkLimit(request: RateLimitRequest): RateLimitResult {
    const now = request.timestamp || Date.now();
    const clientId = this.getClientId(request);

    // Check if rate limiting is disabled or client is in skip list
    if (!this.config.enabled || this.config.skipList.includes(clientId)) {
      return this.createAllowedResult(clientId, now);
    }

    // Get or create bucket for this client
    let bucket = this.buckets.get(clientId);
    if (!bucket) {
      bucket = {
        timestamps: [],
        burstTimestamps: [],
        lastCleanup: now,
      };
      this.buckets.set(clientId, bucket);
    }

    // Clean up old timestamps from sliding window
    const windowStart = now - this.config.windowMs;
    const burstWindowStart = now - this.config.burstWindowMs;

    bucket.timestamps = bucket.timestamps.filter((ts) => ts > windowStart);
    bucket.burstTimestamps = bucket.burstTimestamps.filter(
      (ts) => ts > burstWindowStart,
    );

    // Check burst limit
    const burstCount = bucket.burstTimestamps.length;
    const burstViolation = burstCount >= this.config.burstLimit;

    // Check window limit
    const windowCount = bucket.timestamps.length;
    const windowViolation = windowCount >= this.config.maxRequests;

    const resetAt = windowStart + this.config.windowMs + this.config.windowMs;
    const remaining = Math.max(0, this.config.maxRequests - windowCount);

    const info: RateLimitInfo = {
      clientId,
      windowCount,
      burstCount,
      burstViolation,
      windowViolation,
      windowStart,
      windowEnd: windowStart + this.config.windowMs,
    };

    // If either limit is exceeded, deny the request
    if (burstViolation || windowViolation) {
      const retryAfter = burstViolation
        ? Math.ceil(this.config.burstWindowMs / 1000)
        : Math.ceil((resetAt - now) / 1000);

      if (this.config.onRateLimitExceeded) {
        this.config.onRateLimitExceeded(clientId, info);
      }

      return {
        allowed: false,
        remaining: 0,
        limit: this.config.maxRequests,
        resetAt,
        retryAfter,
        currentCount: windowCount,
        info,
      };
    }

    // Request is allowed - record the timestamp
    bucket.timestamps.push(now);
    bucket.burstTimestamps.push(now);

    return {
      allowed: true,
      remaining: remaining - 1, // Account for this request
      limit: this.config.maxRequests,
      resetAt,
      currentCount: windowCount + 1,
      info,
    };
  }

  /**
   * Consume multiple tokens at once (for batch operations)
   */
  consumeTokens(request: RateLimitRequest, count: number): RateLimitResult {
    const now = request.timestamp || Date.now();
    const clientId = this.getClientId(request);

    if (!this.config.enabled || this.config.skipList.includes(clientId)) {
      return this.createAllowedResult(clientId, now);
    }

    let bucket = this.buckets.get(clientId);
    if (!bucket) {
      bucket = {
        timestamps: [],
        burstTimestamps: [],
        lastCleanup: now,
      };
      this.buckets.set(clientId, bucket);
    }

    // Clean up old timestamps
    const windowStart = now - this.config.windowMs;
    bucket.timestamps = bucket.timestamps.filter((ts) => ts > windowStart);

    const windowCount = bucket.timestamps.length;
    const available = this.config.maxRequests - windowCount;
    const resetAt = windowStart + this.config.windowMs + this.config.windowMs;

    const info: RateLimitInfo = {
      clientId,
      windowCount,
      burstCount: 0,
      burstViolation: false,
      windowViolation: windowCount + count > this.config.maxRequests,
      windowStart,
      windowEnd: windowStart + this.config.windowMs,
    };

    if (count > available) {
      const retryAfter = Math.ceil((resetAt - now) / 1000);

      if (this.config.onRateLimitExceeded) {
        this.config.onRateLimitExceeded(clientId, info);
      }

      return {
        allowed: false,
        remaining: Math.max(0, available),
        limit: this.config.maxRequests,
        resetAt,
        retryAfter,
        currentCount: windowCount,
        info,
      };
    }

    // Consume tokens
    for (let i = 0; i < count; i++) {
      bucket.timestamps.push(now);
    }

    return {
      allowed: true,
      remaining: available - count,
      limit: this.config.maxRequests,
      resetAt,
      currentCount: windowCount + count,
      info,
    };
  }

  /**
   * Get rate limit headers for a response
   */
  getHeaders(result: RateLimitResult): RateLimitHeaders {
    const headers: RateLimitHeaders = {
      "X-RateLimit-Limit": result.limit.toString(),
      "X-RateLimit-Remaining": Math.max(0, result.remaining).toString(),
      "X-RateLimit-Reset": Math.floor(result.resetAt / 1000).toString(),
      "X-RateLimit-Policy": `${this.config.maxRequests};w=${Math.floor(this.config.windowMs / 1000)}`,
    };

    if (!result.allowed && result.retryAfter !== undefined) {
      headers["Retry-After"] = result.retryAfter.toString();
    }

    return headers;
  }

  /**
   * Get current status for a client
   */
  getClientStatus(clientId: string): {
    requests: number;
    remaining: number;
    limit: number;
    resetIn: number;
  } | null {
    const bucket = this.buckets.get(clientId);
    if (!bucket) {
      return {
        requests: 0,
        remaining: this.config.maxRequests,
        limit: this.config.maxRequests,
        resetIn: 0,
      };
    }

    const now = Date.now();
    const windowStart = now - this.config.windowMs;
    const activeTimestamps = bucket.timestamps.filter((ts) => ts > windowStart);
    const oldestTimestamp =
      activeTimestamps.length > 0 ? Math.min(...activeTimestamps) : now;

    return {
      requests: activeTimestamps.length,
      remaining: Math.max(0, this.config.maxRequests - activeTimestamps.length),
      limit: this.config.maxRequests,
      resetIn: Math.max(0, oldestTimestamp + this.config.windowMs - now),
    };
  }

  /**
   * Reset rate limit for a specific client
   */
  reset(clientId: string): boolean {
    return this.buckets.delete(clientId);
  }

  /**
   * Reset all rate limits
   */
  resetAll(): void {
    this.buckets.clear();
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<RateLimitConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): RateLimitConfig {
    return { ...this.config };
  }

  /**
   * Add client to skip list (e.g., for internal services)
   */
  addToSkipList(clientId: string): void {
    if (!this.config.skipList.includes(clientId)) {
      this.config.skipList.push(clientId);
    }
  }

  /**
   * Remove client from skip list
   */
  removeFromSkipList(clientId: string): void {
    const index = this.config.skipList.indexOf(clientId);
    if (index !== -1) {
      this.config.skipList.splice(index, 1);
    }
  }

  /**
   * Get statistics about rate limiting
   */
  getStats(): {
    totalClients: number;
    activeClients: number;
    totalRequests: number;
    config: RateLimitConfig;
  } {
    const now = Date.now();
    const windowStart = now - this.config.windowMs;
    let activeClients = 0;
    let totalRequests = 0;

    for (const bucket of this.buckets.values()) {
      const activeTimestamps = bucket.timestamps.filter(
        (ts) => ts > windowStart,
      );
      if (activeTimestamps.length > 0) {
        activeClients++;
        totalRequests += activeTimestamps.length;
      }
    }

    return {
      totalClients: this.buckets.size,
      activeClients,
      totalRequests,
      config: { ...this.config },
    };
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.buckets.clear();
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private getClientId(request: RateLimitRequest): string {
    // Use custom key generator if provided
    if (this.config.keyGenerator) {
      return this.config.keyGenerator(request);
    }

    // Priority: explicit clientId > agentId > IP > hash of request
    if (request.clientId) {
      return request.clientId;
    }
    if (request.agentId) {
      return `agent:${request.agentId}`;
    }
    if (request.ip) {
      return `ip:${request.ip}`;
    }

    // Fallback: generate a hash from available data
    const data = JSON.stringify({
      toolName: request.toolName,
      timestamp: request.timestamp,
    });
    return `anon:${crypto.createHash("sha256").update(data).digest("hex").substring(0, 16)}`;
  }

  private createAllowedResult(clientId: string, now: number): RateLimitResult {
    return {
      allowed: true,
      remaining: this.config.maxRequests,
      limit: this.config.maxRequests,
      resetAt: now + this.config.windowMs,
      currentCount: 0,
      info: {
        clientId,
        windowCount: 0,
        burstCount: 0,
        burstViolation: false,
        windowViolation: false,
        windowStart: now,
        windowEnd: now + this.config.windowMs,
      },
    };
  }

  private startCleanupInterval(): void {
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, this.CLEANUP_INTERVAL_MS);

    // Don't prevent process from exiting
    if (this.cleanupInterval.unref) {
      this.cleanupInterval.unref();
    }
  }

  private cleanup(): number {
    const now = Date.now();
    const threshold = this.config.windowMs * 2;
    let removed = 0;

    for (const [clientId, bucket] of this.buckets) {
      // Remove completely stale buckets
      if (
        bucket.timestamps.length === 0 ||
        now - Math.max(...bucket.timestamps) > threshold
      ) {
        this.buckets.delete(clientId);
        removed++;
        continue;
      }

      // Clean up old timestamps in active buckets
      const windowStart = now - this.config.windowMs;
      bucket.timestamps = bucket.timestamps.filter((ts) => ts > windowStart);
      bucket.burstTimestamps = [];
    }

    return removed;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a rate limit middleware instance
 */
export function createRateLimitMiddleware(
  config: Partial<RateLimitConfig> = {},
): RateLimitMiddleware {
  return new RateLimitMiddleware(config);
}

// ============================================================================
// Default Instance
// ============================================================================

let defaultInstance: RateLimitMiddleware | null = null;

/**
 * Get the default rate limit middleware instance
 */
export function getRateLimitMiddleware(
  config?: Partial<RateLimitConfig>,
): RateLimitMiddleware {
  if (!defaultInstance) {
    defaultInstance = new RateLimitMiddleware(config);
  }
  return defaultInstance;
}

/**
 * Reset the default instance (useful for testing)
 */
export function resetRateLimitMiddleware(): void {
  if (defaultInstance) {
    defaultInstance.destroy();
    defaultInstance = null;
  }
}
