/**
 * Subscription System for Claude Multi-Agent Coordination MCP Server
 *
 * Features:
 * - adv-b-sub-001: Real-time task update notifications
 * - adv-b-sub-004: Subscription filtering by task type
 * - adv-b-sub-005: Notification batching
 */

import { EventEmitter } from "events";
import { v4 as uuidv4 } from "uuid";
import type { Task, Discovery, Agent, TaskStatus } from "./types.js";

// ============================================================================
// Types
// ============================================================================

export type SubscriptionEventType =
  | "task_created"
  | "task_updated"
  | "task_claimed"
  | "task_started"
  | "task_completed"
  | "task_failed"
  | "discovery_added"
  | "agent_registered"
  | "agent_heartbeat"
  | "coordination_init";

export interface SubscriptionFilter {
  /** Filter by specific event types */
  event_types?: SubscriptionEventType[];
  /** Filter by task status */
  task_status?: TaskStatus[];
  /** Filter by task tags */
  task_tags?: string[];
  /** Filter by agent ID */
  agent_id?: string;
  /** Filter by task priority range */
  priority_min?: number;
  priority_max?: number;
}

export interface Subscription {
  id: string;
  subscriber_id: string;
  filter: SubscriptionFilter;
  created_at: string;
  callback?: (notification: Notification) => void;
}

export interface Notification {
  id: string;
  subscription_id: string;
  event_type: SubscriptionEventType;
  payload: unknown;
  timestamp: string;
}

export interface BatchedNotifications {
  batch_id: string;
  subscriber_id: string;
  notifications: Notification[];
  start_time: string;
  end_time: string;
}

export interface SubscriptionConfig {
  /** Enable notification batching */
  batching_enabled: boolean;
  /** Batch interval in milliseconds */
  batch_interval_ms: number;
  /** Maximum notifications per batch */
  max_batch_size: number;
  /** Maximum pending notifications before forcing flush */
  max_pending_notifications: number;
}

// ============================================================================
// Subscription Manager Class
// ============================================================================

export class SubscriptionManager extends EventEmitter {
  private subscriptions: Map<string, Subscription> = new Map();
  private subscriberSubscriptions: Map<string, Set<string>> = new Map();
  private pendingNotifications: Map<string, Notification[]> = new Map();
  private batchTimers: Map<string, NodeJS.Timeout> = new Map();
  private config: SubscriptionConfig;

  constructor(config: Partial<SubscriptionConfig> = {}) {
    super();
    this.config = {
      batching_enabled: config.batching_enabled ?? false,
      batch_interval_ms: config.batch_interval_ms ?? 1000,
      max_batch_size: config.max_batch_size ?? 100,
      max_pending_notifications: config.max_pending_notifications ?? 500,
    };
  }

  // --------------------------------------------------------------------------
  // Subscription Management
  // --------------------------------------------------------------------------

  /**
   * Create a new subscription
   */
  subscribe(
    subscriberId: string,
    filter: SubscriptionFilter = {},
    callback?: (notification: Notification) => void,
  ): Subscription {
    const subscription: Subscription = {
      id: uuidv4(),
      subscriber_id: subscriberId,
      filter,
      created_at: new Date().toISOString(),
      callback,
    };

    this.subscriptions.set(subscription.id, subscription);

    // Track subscriptions by subscriber
    if (!this.subscriberSubscriptions.has(subscriberId)) {
      this.subscriberSubscriptions.set(subscriberId, new Set());
    }
    this.subscriberSubscriptions.get(subscriberId)!.add(subscription.id);

    // Initialize pending notifications for this subscriber
    if (!this.pendingNotifications.has(subscriberId)) {
      this.pendingNotifications.set(subscriberId, []);
    }

    this.emit("subscription_created", subscription);
    return subscription;
  }

  /**
   * Remove a subscription
   */
  unsubscribe(subscriptionId: string): boolean {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) return false;

    this.subscriptions.delete(subscriptionId);

    // Remove from subscriber tracking
    const subscriberSubs = this.subscriberSubscriptions.get(
      subscription.subscriber_id,
    );
    if (subscriberSubs) {
      subscriberSubs.delete(subscriptionId);
      if (subscriberSubs.size === 0) {
        this.subscriberSubscriptions.delete(subscription.subscriber_id);
        this.pendingNotifications.delete(subscription.subscriber_id);

        // Clear any pending batch timer
        const timer = this.batchTimers.get(subscription.subscriber_id);
        if (timer) {
          clearTimeout(timer);
          this.batchTimers.delete(subscription.subscriber_id);
        }
      }
    }

    this.emit("subscription_removed", subscription);
    return true;
  }

  /**
   * Remove all subscriptions for a subscriber
   */
  unsubscribeAll(subscriberId: string): number {
    const subs = this.subscriberSubscriptions.get(subscriberId);
    if (!subs) return 0;

    let count = 0;
    for (const subId of subs) {
      if (this.unsubscribe(subId)) count++;
    }
    return count;
  }

  /**
   * Get a subscription by ID
   */
  getSubscription(subscriptionId: string): Subscription | undefined {
    return this.subscriptions.get(subscriptionId);
  }

  /**
   * Get all subscriptions for a subscriber
   */
  getSubscriberSubscriptions(subscriberId: string): Subscription[] {
    const subIds = this.subscriberSubscriptions.get(subscriberId);
    if (!subIds) return [];
    return Array.from(subIds)
      .map((id) => this.subscriptions.get(id))
      .filter((sub): sub is Subscription => sub !== undefined);
  }

  /**
   * Get all subscriptions
   */
  getAllSubscriptions(): Subscription[] {
    return Array.from(this.subscriptions.values());
  }

  /**
   * Update subscription filter
   */
  updateFilter(subscriptionId: string, filter: SubscriptionFilter): boolean {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) return false;

    subscription.filter = { ...subscription.filter, ...filter };
    this.emit("subscription_updated", subscription);
    return true;
  }

  // --------------------------------------------------------------------------
  // Notification Publishing
  // --------------------------------------------------------------------------

  /**
   * Publish an event to all matching subscriptions
   */
  publish(eventType: SubscriptionEventType, payload: unknown): void {
    const matchingSubscriptions = this.findMatchingSubscriptions(
      eventType,
      payload,
    );

    for (const subscription of matchingSubscriptions) {
      const notification: Notification = {
        id: uuidv4(),
        subscription_id: subscription.id,
        event_type: eventType,
        payload,
        timestamp: new Date().toISOString(),
      };

      if (this.config.batching_enabled) {
        this.addToBatch(subscription.subscriber_id, notification);
      } else {
        this.deliverNotification(subscription, notification);
      }
    }
  }

  /**
   * Publish a task event
   */
  publishTaskEvent(eventType: SubscriptionEventType, task: Task): void {
    this.publish(eventType, {
      task_id: task.id,
      description: task.description,
      status: task.status,
      priority: task.priority,
      claimed_by: task.claimed_by,
      tags: task.tags,
      result: task.result,
    });
  }

  /**
   * Publish a discovery event
   */
  publishDiscoveryEvent(discovery: Discovery): void {
    this.publish("discovery_added", {
      discovery_id: discovery.id,
      agent_id: discovery.agent_id,
      content: discovery.content,
      tags: discovery.tags,
    });
  }

  /**
   * Publish an agent event
   */
  publishAgentEvent(
    eventType: "agent_registered" | "agent_heartbeat",
    agent: Agent,
  ): void {
    this.publish(eventType, {
      agent_id: agent.id,
      role: agent.role,
      current_task: agent.current_task,
      tasks_completed: agent.tasks_completed,
    });
  }

  // --------------------------------------------------------------------------
  // Filtering Logic
  // --------------------------------------------------------------------------

  /**
   * Find all subscriptions that match an event
   */
  private findMatchingSubscriptions(
    eventType: SubscriptionEventType,
    payload: unknown,
  ): Subscription[] {
    return Array.from(this.subscriptions.values()).filter((subscription) =>
      this.matchesFilter(subscription.filter, eventType, payload),
    );
  }

  /**
   * Check if an event matches a subscription filter
   */
  private matchesFilter(
    filter: SubscriptionFilter,
    eventType: SubscriptionEventType,
    payload: unknown,
  ): boolean {
    // Check event type filter
    if (filter.event_types && filter.event_types.length > 0) {
      if (!filter.event_types.includes(eventType)) {
        return false;
      }
    }

    // Type guard for task-related payloads
    const taskPayload = payload as {
      status?: TaskStatus;
      tags?: string[];
      priority?: number;
      agent_id?: string;
      claimed_by?: string;
    } | null;

    // Check task status filter
    if (filter.task_status && filter.task_status.length > 0) {
      if (
        !taskPayload?.status ||
        !filter.task_status.includes(taskPayload.status)
      ) {
        return false;
      }
    }

    // Check task tags filter
    if (filter.task_tags && filter.task_tags.length > 0) {
      const payloadTags = taskPayload?.tags || [];
      const hasMatchingTag = filter.task_tags.some((tag) =>
        payloadTags.includes(tag),
      );
      if (!hasMatchingTag) {
        return false;
      }
    }

    // Check agent ID filter
    if (filter.agent_id) {
      const agentId = taskPayload?.agent_id || taskPayload?.claimed_by;
      if (agentId !== filter.agent_id) {
        return false;
      }
    }

    // Check priority range filter
    if (
      filter.priority_min !== undefined ||
      filter.priority_max !== undefined
    ) {
      const priority = taskPayload?.priority;
      if (priority === undefined) {
        return false;
      }
      if (filter.priority_min !== undefined && priority < filter.priority_min) {
        return false;
      }
      if (filter.priority_max !== undefined && priority > filter.priority_max) {
        return false;
      }
    }

    return true;
  }

  // --------------------------------------------------------------------------
  // Notification Batching
  // --------------------------------------------------------------------------

  /**
   * Add a notification to the batch queue
   */
  private addToBatch(subscriberId: string, notification: Notification): void {
    const pending = this.pendingNotifications.get(subscriberId) || [];
    pending.push(notification);
    this.pendingNotifications.set(subscriberId, pending);

    // Force flush if max pending reached
    if (pending.length >= this.config.max_pending_notifications) {
      this.flushBatch(subscriberId);
      return;
    }

    // Start batch timer if not already running
    if (!this.batchTimers.has(subscriberId)) {
      const timer = setTimeout(() => {
        this.flushBatch(subscriberId);
      }, this.config.batch_interval_ms);
      this.batchTimers.set(subscriberId, timer);
    }
  }

  /**
   * Flush pending notifications for a subscriber
   */
  flushBatch(subscriberId: string): void {
    // Clear timer
    const timer = this.batchTimers.get(subscriberId);
    if (timer) {
      clearTimeout(timer);
      this.batchTimers.delete(subscriberId);
    }

    // Get pending notifications
    const pending = this.pendingNotifications.get(subscriberId) || [];
    if (pending.length === 0) return;

    // Clear pending
    this.pendingNotifications.set(subscriberId, []);

    // Split into batches if needed
    const batches: Notification[][] = [];
    for (let i = 0; i < pending.length; i += this.config.max_batch_size) {
      batches.push(pending.slice(i, i + this.config.max_batch_size));
    }

    // Deliver batches
    for (const batchNotifications of batches) {
      const batch: BatchedNotifications = {
        batch_id: uuidv4(),
        subscriber_id: subscriberId,
        notifications: batchNotifications,
        start_time: batchNotifications[0].timestamp,
        end_time: batchNotifications[batchNotifications.length - 1].timestamp,
      };

      this.emit("batch_ready", batch);

      // Deliver to subscription callbacks
      const subscriptions = this.getSubscriberSubscriptions(subscriberId);
      for (const sub of subscriptions) {
        if (sub.callback) {
          for (const notification of batchNotifications) {
            if (notification.subscription_id === sub.id) {
              try {
                sub.callback(notification);
              } catch (error) {
                this.emit("callback_error", { subscription: sub, error });
              }
            }
          }
        }
      }
    }
  }

  /**
   * Flush all pending batches
   */
  flushAllBatches(): void {
    for (const subscriberId of this.pendingNotifications.keys()) {
      this.flushBatch(subscriberId);
    }
  }

  // --------------------------------------------------------------------------
  // Notification Delivery
  // --------------------------------------------------------------------------

  /**
   * Deliver a single notification immediately
   */
  private deliverNotification(
    subscription: Subscription,
    notification: Notification,
  ): void {
    // Emit event for external handlers
    this.emit("notification", notification);

    // Call subscription callback if provided
    if (subscription.callback) {
      try {
        subscription.callback(notification);
      } catch (error) {
        this.emit("callback_error", { subscription, notification, error });
      }
    }
  }

  // --------------------------------------------------------------------------
  // Configuration
  // --------------------------------------------------------------------------

  /**
   * Update batching configuration
   */
  updateConfig(config: Partial<SubscriptionConfig>): void {
    this.config = { ...this.config, ...config };
    this.emit("config_updated", this.config);
  }

  /**
   * Get current configuration
   */
  getConfig(): SubscriptionConfig {
    return { ...this.config };
  }

  /**
   * Enable or disable batching
   */
  setBatchingEnabled(enabled: boolean): void {
    if (this.config.batching_enabled && !enabled) {
      // Flush all pending when disabling
      this.flushAllBatches();
    }
    this.config.batching_enabled = enabled;
  }

  // --------------------------------------------------------------------------
  // Statistics
  // --------------------------------------------------------------------------

  /**
   * Get subscription statistics
   */
  getStats(): {
    total_subscriptions: number;
    total_subscribers: number;
    pending_notifications: number;
    batching_enabled: boolean;
  } {
    let pendingCount = 0;
    for (const pending of this.pendingNotifications.values()) {
      pendingCount += pending.length;
    }

    return {
      total_subscriptions: this.subscriptions.size,
      total_subscribers: this.subscriberSubscriptions.size,
      pending_notifications: pendingCount,
      batching_enabled: this.config.batching_enabled,
    };
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    // Clear all batch timers
    for (const timer of this.batchTimers.values()) {
      clearTimeout(timer);
    }
    this.batchTimers.clear();
    this.subscriptions.clear();
    this.subscriberSubscriptions.clear();
    this.pendingNotifications.clear();
    this.removeAllListeners();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let subscriptionManagerInstance: SubscriptionManager | null = null;

export function getSubscriptionManager(
  config?: Partial<SubscriptionConfig>,
): SubscriptionManager {
  if (!subscriptionManagerInstance) {
    subscriptionManagerInstance = new SubscriptionManager(config);
  }
  return subscriptionManagerInstance;
}

export function resetSubscriptionManager(): void {
  if (subscriptionManagerInstance) {
    subscriptionManagerInstance.destroy();
    subscriptionManagerInstance = null;
  }
}
