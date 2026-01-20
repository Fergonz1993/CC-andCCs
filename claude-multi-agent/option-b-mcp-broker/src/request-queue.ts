/**
 * Request Queuing with Priority (adv-b-004)
 *
 * Implements a priority queue for requests to handle high-load scenarios
 * and ensure fair processing based on priority.
 */

import { v4 as uuidv4 } from "uuid";
import type { QueuedRequest } from "./types.js";

export interface QueueStats {
  total_queued: number;
  by_priority: Record<number, number>;
  oldest_request_age_ms: number | null;
  processing_rate: number;
}

export class RequestQueue {
  private queue: QueuedRequest[] = [];
  private maxSize: number;
  private processing: boolean = false;
  private processedCount: number = 0;
  private lastProcessedTime: number = Date.now();

  constructor(maxSize: number = 1000) {
    this.maxSize = maxSize;
  }

  /**
   * Add a request to the queue
   * Returns a promise that resolves when the request is processed
   */
  enqueue<T>(
    agentId: string,
    toolName: string,
    args: Record<string, unknown>,
    priority: number = 5,
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      if (this.queue.length >= this.maxSize) {
        reject(new Error("Queue is full, please retry later"));
        return;
      }

      const request: QueuedRequest = {
        id: uuidv4(),
        priority,
        timestamp: Date.now(),
        agent_id: agentId,
        tool_name: toolName,
        args,
        resolve: resolve as (value: unknown) => void,
        reject,
      };

      // Insert in priority order (lower number = higher priority)
      // Use binary search for efficient insertion
      const insertIndex = this.findInsertIndex(priority);
      this.queue.splice(insertIndex, 0, request);
    });
  }

  /**
   * Binary search to find insertion index maintaining priority order
   */
  private findInsertIndex(priority: number): number {
    let left = 0;
    let right = this.queue.length;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (this.queue[mid].priority <= priority) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    return left;
  }

  /**
   * Get the next request to process (highest priority, oldest)
   */
  dequeue(): QueuedRequest | null {
    if (this.queue.length === 0) return null;

    const request = this.queue.shift()!;
    this.processedCount++;
    this.lastProcessedTime = Date.now();

    return request;
  }

  /**
   * Peek at the next request without removing it
   */
  peek(): QueuedRequest | null {
    return this.queue.length > 0 ? this.queue[0] : null;
  }

  /**
   * Get queue statistics
   */
  getStats(): QueueStats {
    const byPriority: Record<number, number> = {};

    for (const req of this.queue) {
      byPriority[req.priority] = (byPriority[req.priority] || 0) + 1;
    }

    const now = Date.now();
    const elapsed = (now - this.lastProcessedTime) / 1000;
    const processingRate = elapsed > 0 ? this.processedCount / elapsed : 0;

    return {
      total_queued: this.queue.length,
      by_priority: byPriority,
      oldest_request_age_ms:
        this.queue.length > 0 ? now - this.queue[0].timestamp : null,
      processing_rate: Math.round(processingRate * 100) / 100,
    };
  }

  /**
   * Get queue size
   */
  size(): number {
    return this.queue.length;
  }

  /**
   * Check if queue is empty
   */
  isEmpty(): boolean {
    return this.queue.length === 0;
  }

  /**
   * Cancel a specific request by ID
   */
  cancel(requestId: string): boolean {
    const index = this.queue.findIndex((r) => r.id === requestId);
    if (index === -1) return false;

    const [request] = this.queue.splice(index, 1);
    request.reject(new Error("Request cancelled"));
    return true;
  }

  /**
   * Cancel all requests for a specific agent
   */
  cancelForAgent(agentId: string): number {
    let cancelled = 0;
    const remaining: QueuedRequest[] = [];

    for (const request of this.queue) {
      if (request.agent_id === agentId) {
        request.reject(new Error("Request cancelled"));
        cancelled++;
      } else {
        remaining.push(request);
      }
    }

    this.queue = remaining;
    return cancelled;
  }

  /**
   * Clear the entire queue
   */
  clear(): void {
    for (const request of this.queue) {
      request.reject(new Error("Queue cleared"));
    }
    this.queue = [];
  }

  /**
   * Process queued requests with a handler function
   */
  async startProcessing(
    handler: (request: QueuedRequest) => Promise<unknown>,
    concurrency: number = 1,
  ): Promise<void> {
    if (this.processing) return;
    this.processing = true;

    const workers: Promise<void>[] = [];

    for (let i = 0; i < concurrency; i++) {
      workers.push(this.processWorker(handler));
    }

    await Promise.all(workers);
    this.processing = false;
  }

  private async processWorker(
    handler: (request: QueuedRequest) => Promise<unknown>,
  ): Promise<void> {
    while (this.processing && this.queue.length > 0) {
      const request = this.dequeue();
      if (!request) break;

      try {
        const result = await handler(request);
        request.resolve(result);
      } catch (error) {
        request.reject(
          error instanceof Error ? error : new Error(String(error)),
        );
      }
    }
  }

  /**
   * Stop processing
   */
  stopProcessing(): void {
    this.processing = false;
  }

  /**
   * Check if currently processing
   */
  isProcessing(): boolean {
    return this.processing;
  }

  /**
   * Get all requests for a specific agent
   */
  getRequestsForAgent(agentId: string): QueuedRequest[] {
    return this.queue.filter((r) => r.agent_id === agentId);
  }

  /**
   * Boost priority of a specific request
   */
  boostPriority(requestId: string, newPriority: number): boolean {
    const index = this.queue.findIndex((r) => r.id === requestId);
    if (index === -1) return false;

    const request = this.queue[index];
    if (newPriority >= request.priority) return false;

    // Remove and re-insert with new priority
    this.queue.splice(index, 1);
    request.priority = newPriority;
    const newIndex = this.findInsertIndex(newPriority);
    this.queue.splice(newIndex, 0, request);

    return true;
  }
}

// Singleton instance
export const defaultRequestQueue = new RequestQueue();
