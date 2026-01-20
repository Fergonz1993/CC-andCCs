/**
 * Rate Limiting per Client (adv-b-003)
 *
 * Implements token bucket algorithm for rate limiting requests
 * per agent/client.
 */

import type { RateLimitConfig, RateLimitBucket } from "./types.js";

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  reset_at: number;
  retry_after?: number;
}

export class RateLimiter {
  private buckets: Map<string, RateLimitBucket> = new Map();
  private config: RateLimitConfig;

  constructor(config?: Partial<RateLimitConfig>) {
    this.config = {
      max_requests: config?.max_requests ?? 100,
      window_ms: config?.window_ms ?? 60000, // 1 minute
      tokens: config?.tokens ?? 100,
      last_refill: Date.now(),
    };
  }

  /**
   * Check if a request is allowed for the given client
   * Uses token bucket algorithm
   */
  checkLimit(clientId: string): RateLimitResult {
    const now = Date.now();
    let bucket = this.buckets.get(clientId);

    if (!bucket) {
      // New client, create fresh bucket
      bucket = {
        tokens: this.config.max_requests,
        last_refill: now,
      };
      this.buckets.set(clientId, bucket);
    }

    // Refill tokens based on time elapsed
    const elapsed = now - bucket.last_refill;
    const refillRate = this.config.max_requests / this.config.window_ms;
    const tokensToAdd = Math.floor(elapsed * refillRate);

    if (tokensToAdd > 0) {
      bucket.tokens = Math.min(
        this.config.max_requests,
        bucket.tokens + tokensToAdd,
      );
      bucket.last_refill = now;
    }

    // Check if request is allowed
    if (bucket.tokens >= 1) {
      bucket.tokens -= 1;
      return {
        allowed: true,
        remaining: Math.floor(bucket.tokens),
        reset_at: now + this.config.window_ms,
      };
    }

    // Rate limited
    const timeUntilRefill = Math.ceil((1 - bucket.tokens) / refillRate);

    return {
      allowed: false,
      remaining: 0,
      reset_at: now + this.config.window_ms,
      retry_after: timeUntilRefill,
    };
  }

  /**
   * Consume tokens for a client (for batch operations)
   */
  consumeTokens(clientId: string, count: number): RateLimitResult {
    const now = Date.now();
    let bucket = this.buckets.get(clientId);

    if (!bucket) {
      bucket = {
        tokens: this.config.max_requests,
        last_refill: now,
      };
      this.buckets.set(clientId, bucket);
    }

    // Refill tokens
    const elapsed = now - bucket.last_refill;
    const refillRate = this.config.max_requests / this.config.window_ms;
    const tokensToAdd = Math.floor(elapsed * refillRate);

    if (tokensToAdd > 0) {
      bucket.tokens = Math.min(
        this.config.max_requests,
        bucket.tokens + tokensToAdd,
      );
      bucket.last_refill = now;
    }

    if (bucket.tokens >= count) {
      bucket.tokens -= count;
      return {
        allowed: true,
        remaining: Math.floor(bucket.tokens),
        reset_at: now + this.config.window_ms,
      };
    }

    const timeUntilRefill = Math.ceil((count - bucket.tokens) / refillRate);

    return {
      allowed: false,
      remaining: Math.floor(bucket.tokens),
      reset_at: now + this.config.window_ms,
      retry_after: timeUntilRefill,
    };
  }

  /**
   * Get current status for a client
   */
  getStatus(clientId: string): {
    tokens: number;
    max_tokens: number;
    window_ms: number;
  } {
    const bucket = this.buckets.get(clientId);
    return {
      tokens: bucket ? Math.floor(bucket.tokens) : this.config.max_requests,
      max_tokens: this.config.max_requests,
      window_ms: this.config.window_ms,
    };
  }

  /**
   * Reset rate limit for a specific client
   */
  reset(clientId: string): void {
    this.buckets.delete(clientId);
  }

  /**
   * Clear all rate limit buckets
   */
  clear(): void {
    this.buckets.clear();
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<RateLimitConfig>): void {
    if (config.max_requests !== undefined) {
      this.config.max_requests = config.max_requests;
    }
    if (config.window_ms !== undefined) {
      this.config.window_ms = config.window_ms;
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): RateLimitConfig {
    return { ...this.config };
  }

  /**
   * Clean up stale buckets (older than 2x window)
   */
  cleanup(): number {
    const now = Date.now();
    const threshold = this.config.window_ms * 2;
    let removed = 0;

    for (const [clientId, bucket] of this.buckets) {
      if (now - bucket.last_refill > threshold) {
        this.buckets.delete(clientId);
        removed++;
      }
    }

    return removed;
  }
}

// Singleton instance with default config
export const defaultRateLimiter = new RateLimiter();
