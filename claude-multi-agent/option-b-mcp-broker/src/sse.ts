/**
 * Server-Sent Events (SSE) Stream for Claude Multi-Agent Coordination MCP Server
 *
 * Features:
 * - adv-b-sub-003: Server-sent events (SSE) stream
 *
 * Note: This module provides SSE stream management that can be integrated
 * with an HTTP server for real-time event streaming to clients.
 */

import { EventEmitter } from "events";
import { v4 as uuidv4 } from "uuid";
import type { Task, Discovery, Agent, TaskStatus } from "./types.js";
import type { SubscriptionEventType } from "./subscriptions.js";

// ============================================================================
// Types
// ============================================================================

export interface SSEClient {
  id: string;
  agent_id: string | null;
  connected_at: string;
  last_event_at: string | null;
  filter: SSEFilter;
  write: (data: string) => boolean;
  close: () => void;
}

export interface SSEFilter {
  event_types?: SubscriptionEventType[];
  task_status?: TaskStatus[];
  task_tags?: string[];
  agent_id?: string;
  priority_min?: number;
  priority_max?: number;
}

export interface SSEEvent {
  id: string;
  event: SubscriptionEventType;
  data: unknown;
  retry?: number;
}

export interface SSEConfig {
  /** Heartbeat interval in milliseconds */
  heartbeat_interval_ms: number;
  /** Retry interval sent to clients */
  client_retry_ms: number;
  /** Maximum clients */
  max_clients: number;
  /** Enable compression */
  compression_enabled: boolean;
}

// ============================================================================
// SSE Manager Class
// ============================================================================

export class SSEManager extends EventEmitter {
  private clients: Map<string, SSEClient> = new Map();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private eventHistory: SSEEvent[] = [];
  private maxHistorySize: number = 100;
  private config: SSEConfig;
  private eventCounter: number = 0;

  constructor(config: Partial<SSEConfig> = {}) {
    super();
    this.config = {
      heartbeat_interval_ms: config.heartbeat_interval_ms ?? 30000,
      client_retry_ms: config.client_retry_ms ?? 3000,
      max_clients: config.max_clients ?? 100,
      compression_enabled: config.compression_enabled ?? false,
    };

    // Start heartbeat
    this.startHeartbeat();
  }

  // --------------------------------------------------------------------------
  // Client Management
  // --------------------------------------------------------------------------

  /**
   * Register a new SSE client
   */
  addClient(
    write: (data: string) => boolean,
    close: () => void,
    options: {
      agent_id?: string;
      filter?: SSEFilter;
      last_event_id?: string;
    } = {},
  ): SSEClient | null {
    if (this.clients.size >= this.config.max_clients) {
      return null;
    }

    const client: SSEClient = {
      id: uuidv4(),
      agent_id: options.agent_id || null,
      connected_at: new Date().toISOString(),
      last_event_at: null,
      filter: options.filter || {},
      write,
      close,
    };

    this.clients.set(client.id, client);

    // Send initial connection event
    this.sendToClient(client, {
      id: this.generateEventId(),
      event: "coordination_init" as SubscriptionEventType,
      data: { client_id: client.id, connected: true },
      retry: this.config.client_retry_ms,
    });

    // Replay missed events if last_event_id provided
    if (options.last_event_id) {
      this.replayEvents(client, options.last_event_id);
    }

    this.emit("client_connected", client);
    return client;
  }

  /**
   * Remove an SSE client
   */
  removeClient(clientId: string): boolean {
    const client = this.clients.get(clientId);
    if (!client) return false;

    try {
      client.close();
    } catch {
      // Ignore close errors
    }

    this.clients.delete(clientId);
    this.emit("client_disconnected", client);
    return true;
  }

  /**
   * Get a client by ID
   */
  getClient(clientId: string): SSEClient | undefined {
    return this.clients.get(clientId);
  }

  /**
   * Get all clients
   */
  getAllClients(): SSEClient[] {
    return Array.from(this.clients.values());
  }

  /**
   * Update client filter
   */
  updateClientFilter(clientId: string, filter: SSEFilter): boolean {
    const client = this.clients.get(clientId);
    if (!client) return false;
    client.filter = { ...client.filter, ...filter };
    return true;
  }

  // --------------------------------------------------------------------------
  // Event Broadcasting
  // --------------------------------------------------------------------------

  /**
   * Broadcast an event to all matching clients
   */
  broadcast(eventType: SubscriptionEventType, data: unknown): void {
    const event: SSEEvent = {
      id: this.generateEventId(),
      event: eventType,
      data,
    };

    // Store in history
    this.addToHistory(event);

    // Send to matching clients
    for (const client of this.clients.values()) {
      if (this.matchesFilter(client.filter, eventType, data)) {
        this.sendToClient(client, event);
      }
    }
  }

  /**
   * Broadcast a task event
   */
  broadcastTaskEvent(eventType: SubscriptionEventType, task: Task): void {
    this.broadcast(eventType, {
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
   * Broadcast a discovery event
   */
  broadcastDiscoveryEvent(discovery: Discovery): void {
    this.broadcast("discovery_added", {
      discovery_id: discovery.id,
      agent_id: discovery.agent_id,
      content: discovery.content,
      tags: discovery.tags,
      created_at: discovery.created_at,
    });
  }

  /**
   * Broadcast an agent event
   */
  broadcastAgentEvent(
    eventType: "agent_registered" | "agent_heartbeat",
    agent: Agent,
  ): void {
    this.broadcast(eventType, {
      agent_id: agent.id,
      role: agent.role,
      current_task: agent.current_task,
      tasks_completed: agent.tasks_completed,
      last_heartbeat: agent.last_heartbeat,
    });
  }

  /**
   * Send an event to a specific client
   */
  sendToClient(client: SSEClient, event: SSEEvent): boolean {
    const sseMessage = this.formatSSEMessage(event);

    try {
      const success = client.write(sseMessage);
      if (success) {
        client.last_event_at = new Date().toISOString();
      }
      return success;
    } catch (error) {
      // Client disconnected, remove them
      this.removeClient(client.id);
      return false;
    }
  }

  // --------------------------------------------------------------------------
  // SSE Message Formatting
  // --------------------------------------------------------------------------

  /**
   * Format an event as an SSE message
   */
  private formatSSEMessage(event: SSEEvent): string {
    const lines: string[] = [];

    // Event ID
    lines.push(`id: ${event.id}`);

    // Event type
    lines.push(`event: ${event.event}`);

    // Retry interval (optional)
    if (event.retry !== undefined) {
      lines.push(`retry: ${event.retry}`);
    }

    // Data (can be multiline)
    const dataStr = JSON.stringify(event.data);
    lines.push(`data: ${dataStr}`);

    // SSE messages end with double newline
    return lines.join("\n") + "\n\n";
  }

  /**
   * Format a comment (heartbeat/keepalive)
   */
  private formatComment(comment: string): string {
    return `: ${comment}\n\n`;
  }

  // --------------------------------------------------------------------------
  // Filtering
  // --------------------------------------------------------------------------

  /**
   * Check if an event matches a client filter
   */
  private matchesFilter(
    filter: SSEFilter,
    eventType: SubscriptionEventType,
    data: unknown,
  ): boolean {
    // Check event type filter
    if (filter.event_types && filter.event_types.length > 0) {
      if (!filter.event_types.includes(eventType)) {
        return false;
      }
    }

    const payload = data as {
      status?: TaskStatus;
      tags?: string[];
      priority?: number;
      agent_id?: string;
      claimed_by?: string;
    } | null;

    // Check task status filter
    if (filter.task_status && filter.task_status.length > 0) {
      if (!payload?.status || !filter.task_status.includes(payload.status)) {
        return false;
      }
    }

    // Check task tags filter
    if (filter.task_tags && filter.task_tags.length > 0) {
      const tags = payload?.tags || [];
      if (!filter.task_tags.some((tag) => tags.includes(tag))) {
        return false;
      }
    }

    // Check agent ID filter
    if (filter.agent_id) {
      const agentId = payload?.agent_id || payload?.claimed_by;
      if (agentId !== filter.agent_id) {
        return false;
      }
    }

    // Check priority range filter
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
  // Event History & Replay
  // --------------------------------------------------------------------------

  /**
   * Add event to history
   */
  private addToHistory(event: SSEEvent): void {
    this.eventHistory.push(event);
    while (this.eventHistory.length > this.maxHistorySize) {
      this.eventHistory.shift();
    }
  }

  /**
   * Replay events from a specific event ID
   */
  private replayEvents(client: SSEClient, lastEventId: string): void {
    const startIndex = this.eventHistory.findIndex((e) => e.id === lastEventId);
    if (startIndex === -1) {
      // Event not found, send all history
      for (const event of this.eventHistory) {
        if (this.matchesFilter(client.filter, event.event, event.data)) {
          this.sendToClient(client, event);
        }
      }
    } else {
      // Replay from after the last received event
      for (let i = startIndex + 1; i < this.eventHistory.length; i++) {
        const event = this.eventHistory[i];
        if (this.matchesFilter(client.filter, event.event, event.data)) {
          this.sendToClient(client, event);
        }
      }
    }
  }

  /**
   * Get event history
   */
  getEventHistory(limit?: number): SSEEvent[] {
    const events = [...this.eventHistory];
    return limit ? events.slice(-limit) : events;
  }

  // --------------------------------------------------------------------------
  // Heartbeat
  // --------------------------------------------------------------------------

  /**
   * Start the heartbeat interval
   */
  private startHeartbeat(): void {
    if (this.heartbeatInterval) return;

    this.heartbeatInterval = setInterval(() => {
      this.sendHeartbeat();
    }, this.config.heartbeat_interval_ms);
  }

  /**
   * Stop the heartbeat interval
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  /**
   * Send heartbeat to all clients
   */
  private sendHeartbeat(): void {
    const comment = this.formatComment(`heartbeat ${new Date().toISOString()}`);

    for (const client of this.clients.values()) {
      try {
        client.write(comment);
      } catch {
        // Client disconnected
        this.removeClient(client.id);
      }
    }
  }

  // --------------------------------------------------------------------------
  // Utilities
  // --------------------------------------------------------------------------

  /**
   * Generate a unique event ID
   */
  private generateEventId(): string {
    this.eventCounter++;
    return `${Date.now()}-${this.eventCounter}`;
  }

  /**
   * Get SSE headers for HTTP response
   */
  static getHeaders(): Record<string, string> {
    return {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    };
  }

  // --------------------------------------------------------------------------
  // Configuration
  // --------------------------------------------------------------------------

  /**
   * Update configuration
   */
  updateConfig(config: Partial<SSEConfig>): void {
    const restartHeartbeat =
      config.heartbeat_interval_ms !== undefined &&
      config.heartbeat_interval_ms !== this.config.heartbeat_interval_ms;

    this.config = { ...this.config, ...config };

    if (restartHeartbeat) {
      this.stopHeartbeat();
      this.startHeartbeat();
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): SSEConfig {
    return { ...this.config };
  }

  // --------------------------------------------------------------------------
  // Statistics
  // --------------------------------------------------------------------------

  /**
   * Get SSE statistics
   */
  getStats(): {
    connected_clients: number;
    max_clients: number;
    events_in_history: number;
    heartbeat_interval_ms: number;
  } {
    return {
      connected_clients: this.clients.size,
      max_clients: this.config.max_clients,
      events_in_history: this.eventHistory.length,
      heartbeat_interval_ms: this.config.heartbeat_interval_ms,
    };
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    this.stopHeartbeat();

    // Close all clients
    for (const client of this.clients.values()) {
      try {
        client.close();
      } catch {
        // Ignore close errors
      }
    }

    this.clients.clear();
    this.eventHistory = [];
    this.removeAllListeners();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let sseManagerInstance: SSEManager | null = null;

export function getSSEManager(config?: Partial<SSEConfig>): SSEManager {
  if (!sseManagerInstance) {
    sseManagerInstance = new SSEManager(config);
  }
  return sseManagerInstance;
}

export function resetSSEManager(): void {
  if (sseManagerInstance) {
    sseManagerInstance.destroy();
    sseManagerInstance = null;
  }
}
