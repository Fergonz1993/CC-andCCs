/**
 * Webhook System for Claude Multi-Agent Coordination MCP Server
 *
 * Features:
 * - adv-b-sub-002: Webhook callbacks for task events
 */

import { v4 as uuidv4 } from "uuid";
import * as crypto from "crypto";
import type { Task, Discovery, Agent, TaskStatus } from "./types.js";
import type { SubscriptionEventType } from "./subscriptions.js";

// ============================================================================
// Types
// ============================================================================

export interface WebhookConfig {
  id: string;
  url: string;
  secret: string;
  events: SubscriptionEventType[];
  enabled: boolean;
  created_at: string;
  headers?: Record<string, string>;
  retry_config: WebhookRetryConfig;
  filter?: WebhookFilter;
}

export interface WebhookFilter {
  task_status?: TaskStatus[];
  task_tags?: string[];
  agent_id?: string;
  priority_min?: number;
  priority_max?: number;
}

export interface WebhookRetryConfig {
  max_retries: number;
  initial_delay_ms: number;
  max_delay_ms: number;
  backoff_multiplier: number;
}

export interface WebhookDelivery {
  id: string;
  webhook_id: string;
  event_type: SubscriptionEventType;
  payload: unknown;
  status: "pending" | "success" | "failed" | "retrying";
  attempts: number;
  created_at: string;
  last_attempt_at: string | null;
  next_retry_at: string | null;
  response_status?: number;
  response_body?: string;
  error?: string;
}

export interface WebhookPayload {
  event_id: string;
  event_type: SubscriptionEventType;
  timestamp: string;
  data: unknown;
}

// ============================================================================
// Webhook Manager Class
// ============================================================================

export class WebhookManager {
  private webhooks: Map<string, WebhookConfig> = new Map();
  private deliveries: Map<string, WebhookDelivery> = new Map();
  private retryTimers: Map<string, NodeJS.Timeout> = new Map();
  private deliveryHistory: WebhookDelivery[] = [];
  private maxHistorySize: number = 1000;

  constructor() {}

  // --------------------------------------------------------------------------
  // Webhook Registration
  // --------------------------------------------------------------------------

  /**
   * Register a new webhook
   */
  registerWebhook(
    url: string,
    events: SubscriptionEventType[],
    options: {
      secret?: string;
      headers?: Record<string, string>;
      retry_config?: Partial<WebhookRetryConfig>;
      filter?: WebhookFilter;
    } = {},
  ): WebhookConfig {
    const webhook: WebhookConfig = {
      id: uuidv4(),
      url,
      secret: options.secret || this.generateSecret(),
      events,
      enabled: true,
      created_at: new Date().toISOString(),
      headers: options.headers,
      retry_config: {
        max_retries: options.retry_config?.max_retries ?? 3,
        initial_delay_ms: options.retry_config?.initial_delay_ms ?? 1000,
        max_delay_ms: options.retry_config?.max_delay_ms ?? 60000,
        backoff_multiplier: options.retry_config?.backoff_multiplier ?? 2,
      },
      filter: options.filter,
    };

    this.webhooks.set(webhook.id, webhook);
    return webhook;
  }

  /**
   * Update a webhook configuration
   */
  updateWebhook(
    webhookId: string,
    updates: Partial<Omit<WebhookConfig, "id" | "created_at">>,
  ): WebhookConfig | null {
    const webhook = this.webhooks.get(webhookId);
    if (!webhook) return null;

    const updated = { ...webhook, ...updates };
    this.webhooks.set(webhookId, updated);
    return updated;
  }

  /**
   * Delete a webhook
   */
  deleteWebhook(webhookId: string): boolean {
    const webhook = this.webhooks.get(webhookId);
    if (!webhook) return false;

    // Cancel any pending retries
    for (const [deliveryId, timer] of this.retryTimers.entries()) {
      const delivery = this.deliveries.get(deliveryId);
      if (delivery?.webhook_id === webhookId) {
        clearTimeout(timer);
        this.retryTimers.delete(deliveryId);
      }
    }

    this.webhooks.delete(webhookId);
    return true;
  }

  /**
   * Get a webhook by ID
   */
  getWebhook(webhookId: string): WebhookConfig | undefined {
    return this.webhooks.get(webhookId);
  }

  /**
   * Get all webhooks
   */
  getAllWebhooks(): WebhookConfig[] {
    return Array.from(this.webhooks.values());
  }

  /**
   * Enable or disable a webhook
   */
  setWebhookEnabled(webhookId: string, enabled: boolean): boolean {
    const webhook = this.webhooks.get(webhookId);
    if (!webhook) return false;
    webhook.enabled = enabled;
    return true;
  }

  // --------------------------------------------------------------------------
  // Event Dispatching
  // --------------------------------------------------------------------------

  /**
   * Dispatch an event to all matching webhooks
   */
  async dispatchEvent(
    eventType: SubscriptionEventType,
    data: unknown,
  ): Promise<WebhookDelivery[]> {
    const matchingWebhooks = this.findMatchingWebhooks(eventType, data);
    const deliveries: WebhookDelivery[] = [];

    for (const webhook of matchingWebhooks) {
      const delivery = await this.sendWebhook(webhook, eventType, data);
      deliveries.push(delivery);
    }

    return deliveries;
  }

  /**
   * Dispatch a task event
   */
  async dispatchTaskEvent(
    eventType: SubscriptionEventType,
    task: Task,
  ): Promise<WebhookDelivery[]> {
    return this.dispatchEvent(eventType, {
      task_id: task.id,
      description: task.description,
      status: task.status,
      priority: task.priority,
      claimed_by: task.claimed_by,
      tags: task.tags,
      result: task.result,
      created_at: task.created_at,
      claimed_at: task.claimed_at,
      completed_at: task.completed_at,
    });
  }

  /**
   * Dispatch a discovery event
   */
  async dispatchDiscoveryEvent(
    discovery: Discovery,
  ): Promise<WebhookDelivery[]> {
    return this.dispatchEvent("discovery_added", {
      discovery_id: discovery.id,
      agent_id: discovery.agent_id,
      content: discovery.content,
      tags: discovery.tags,
      created_at: discovery.created_at,
    });
  }

  /**
   * Dispatch an agent event
   */
  async dispatchAgentEvent(
    eventType: "agent_registered" | "agent_heartbeat",
    agent: Agent,
  ): Promise<WebhookDelivery[]> {
    return this.dispatchEvent(eventType, {
      agent_id: agent.id,
      role: agent.role,
      current_task: agent.current_task,
      tasks_completed: agent.tasks_completed,
      last_heartbeat: agent.last_heartbeat,
    });
  }

  // --------------------------------------------------------------------------
  // Webhook Delivery
  // --------------------------------------------------------------------------

  /**
   * Send a webhook request
   */
  private async sendWebhook(
    webhook: WebhookConfig,
    eventType: SubscriptionEventType,
    data: unknown,
  ): Promise<WebhookDelivery> {
    const delivery: WebhookDelivery = {
      id: uuidv4(),
      webhook_id: webhook.id,
      event_type: eventType,
      payload: data,
      status: "pending",
      attempts: 0,
      created_at: new Date().toISOString(),
      last_attempt_at: null,
      next_retry_at: null,
    };

    this.deliveries.set(delivery.id, delivery);

    await this.attemptDelivery(delivery, webhook);
    return delivery;
  }

  /**
   * Attempt to deliver a webhook
   */
  private async attemptDelivery(
    delivery: WebhookDelivery,
    webhook: WebhookConfig,
  ): Promise<void> {
    delivery.attempts++;
    delivery.last_attempt_at = new Date().toISOString();

    const payload: WebhookPayload = {
      event_id: delivery.id,
      event_type: delivery.event_type,
      timestamp: new Date().toISOString(),
      data: delivery.payload,
    };

    const body = JSON.stringify(payload);
    const signature = this.signPayload(body, webhook.secret);

    try {
      const response = await fetch(webhook.url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Webhook-Signature": signature,
          "X-Webhook-Event": delivery.event_type,
          "X-Webhook-Delivery-ID": delivery.id,
          ...webhook.headers,
        },
        body,
      });

      delivery.response_status = response.status;

      if (response.ok) {
        delivery.status = "success";
        try {
          delivery.response_body = await response.text();
        } catch {
          // Ignore response body errors
        }
      } else {
        delivery.status = "failed";
        try {
          delivery.response_body = await response.text();
        } catch {
          // Ignore response body errors
        }
        delivery.error = `HTTP ${response.status}: ${response.statusText}`;
        this.scheduleRetry(delivery, webhook);
      }
    } catch (error) {
      delivery.status = "failed";
      delivery.error = error instanceof Error ? error.message : String(error);
      this.scheduleRetry(delivery, webhook);
    }

    // Archive to history
    this.archiveDelivery(delivery);
  }

  /**
   * Schedule a retry for a failed delivery
   */
  private scheduleRetry(
    delivery: WebhookDelivery,
    webhook: WebhookConfig,
  ): void {
    if (delivery.attempts >= webhook.retry_config.max_retries) {
      delivery.status = "failed";
      return;
    }

    delivery.status = "retrying";

    // Calculate delay with exponential backoff
    const delay = Math.min(
      webhook.retry_config.initial_delay_ms *
        Math.pow(
          webhook.retry_config.backoff_multiplier,
          delivery.attempts - 1,
        ),
      webhook.retry_config.max_delay_ms,
    );

    delivery.next_retry_at = new Date(Date.now() + delay).toISOString();

    const timer = setTimeout(async () => {
      this.retryTimers.delete(delivery.id);
      await this.attemptDelivery(delivery, webhook);
    }, delay);

    this.retryTimers.set(delivery.id, timer);
  }

  /**
   * Archive a delivery to history
   */
  private archiveDelivery(delivery: WebhookDelivery): void {
    // Add to history
    this.deliveryHistory.push({ ...delivery });

    // Trim history if needed
    while (this.deliveryHistory.length > this.maxHistorySize) {
      this.deliveryHistory.shift();
    }
  }

  // --------------------------------------------------------------------------
  // Filtering
  // --------------------------------------------------------------------------

  /**
   * Find webhooks that match an event
   */
  private findMatchingWebhooks(
    eventType: SubscriptionEventType,
    data: unknown,
  ): WebhookConfig[] {
    return Array.from(this.webhooks.values()).filter((webhook) => {
      if (!webhook.enabled) return false;
      if (!webhook.events.includes(eventType)) return false;
      if (webhook.filter && !this.matchesFilter(webhook.filter, data)) {
        return false;
      }
      return true;
    });
  }

  /**
   * Check if data matches a webhook filter
   */
  private matchesFilter(filter: WebhookFilter, data: unknown): boolean {
    const payload = data as {
      status?: TaskStatus;
      tags?: string[];
      priority?: number;
      agent_id?: string;
      claimed_by?: string;
    } | null;

    if (filter.task_status && filter.task_status.length > 0) {
      if (!payload?.status || !filter.task_status.includes(payload.status)) {
        return false;
      }
    }

    if (filter.task_tags && filter.task_tags.length > 0) {
      const tags = payload?.tags || [];
      if (!filter.task_tags.some((tag) => tags.includes(tag))) {
        return false;
      }
    }

    if (filter.agent_id) {
      const agentId = payload?.agent_id || payload?.claimed_by;
      if (agentId !== filter.agent_id) {
        return false;
      }
    }

    if (
      filter.priority_min !== undefined ||
      filter.priority_max !== undefined
    ) {
      const priority = payload?.priority;
      if (priority === undefined) return false;
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
  // Signature Generation
  // --------------------------------------------------------------------------

  /**
   * Generate a signature for a payload
   */
  private signPayload(payload: string, secret: string): string {
    const hmac = crypto.createHmac("sha256", secret);
    hmac.update(payload);
    return `sha256=${hmac.digest("hex")}`;
  }

  /**
   * Verify a webhook signature
   */
  verifySignature(payload: string, signature: string, secret: string): boolean {
    const expected = this.signPayload(payload, secret);
    try {
      return crypto.timingSafeEqual(
        Buffer.from(signature),
        Buffer.from(expected),
      );
    } catch {
      return false;
    }
  }

  /**
   * Generate a random secret
   */
  private generateSecret(): string {
    return crypto.randomBytes(32).toString("hex");
  }

  // --------------------------------------------------------------------------
  // Delivery History
  // --------------------------------------------------------------------------

  /**
   * Get delivery history for a webhook
   */
  getDeliveryHistory(
    webhookId?: string,
    options: { limit?: number; status?: WebhookDelivery["status"] } = {},
  ): WebhookDelivery[] {
    let history = this.deliveryHistory;

    if (webhookId) {
      history = history.filter((d) => d.webhook_id === webhookId);
    }

    if (options.status) {
      history = history.filter((d) => d.status === options.status);
    }

    const limit = options.limit || 100;
    return history.slice(-limit);
  }

  /**
   * Get a specific delivery
   */
  getDelivery(deliveryId: string): WebhookDelivery | undefined {
    return this.deliveries.get(deliveryId);
  }

  /**
   * Retry a failed delivery manually
   */
  async retryDelivery(deliveryId: string): Promise<boolean> {
    const delivery = this.deliveries.get(deliveryId);
    if (
      !delivery ||
      delivery.status === "pending" ||
      delivery.status === "retrying"
    ) {
      return false;
    }

    const webhook = this.webhooks.get(delivery.webhook_id);
    if (!webhook) return false;

    delivery.attempts = 0; // Reset attempts
    await this.attemptDelivery(delivery, webhook);
    return true;
  }

  // --------------------------------------------------------------------------
  // Statistics
  // --------------------------------------------------------------------------

  /**
   * Get webhook statistics
   */
  getStats(): {
    total_webhooks: number;
    enabled_webhooks: number;
    total_deliveries: number;
    successful_deliveries: number;
    failed_deliveries: number;
    pending_retries: number;
  } {
    const deliveries = Array.from(this.deliveries.values());
    return {
      total_webhooks: this.webhooks.size,
      enabled_webhooks: Array.from(this.webhooks.values()).filter(
        (w) => w.enabled,
      ).length,
      total_deliveries: deliveries.length,
      successful_deliveries: deliveries.filter((d) => d.status === "success")
        .length,
      failed_deliveries: deliveries.filter((d) => d.status === "failed").length,
      pending_retries: this.retryTimers.size,
    };
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    for (const timer of this.retryTimers.values()) {
      clearTimeout(timer);
    }
    this.retryTimers.clear();
    this.webhooks.clear();
    this.deliveries.clear();
    this.deliveryHistory = [];
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let webhookManagerInstance: WebhookManager | null = null;

export function getWebhookManager(): WebhookManager {
  if (!webhookManagerInstance) {
    webhookManagerInstance = new WebhookManager();
  }
  return webhookManagerInstance;
}

export function resetWebhookManager(): void {
  if (webhookManagerInstance) {
    webhookManagerInstance.destroy();
    webhookManagerInstance = null;
  }
}
