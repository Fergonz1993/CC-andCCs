/**
 * WebSocket Transport for Real-time Updates (adv-b-001)
 *
 * Provides real-time push notifications for task updates,
 * agent status changes, and coordination events.
 */

import { WebSocketServer, WebSocket } from "ws";
import { v4 as uuidv4 } from "uuid";
import type {
  WebSocketClient,
  WebSocketEventType,
  WebSocketMessage,
  Task,
  Agent,
  Discovery,
} from "./types.js";

export interface WebSocketTransportConfig {
  port: number;
  heartbeatInterval?: number;
  reconnectTimeout?: number;
}

export class WebSocketTransport {
  private wss: WebSocketServer | null = null;
  private clients: Map<string, WebSocketClient> = new Map();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private config: WebSocketTransportConfig;

  constructor(config: WebSocketTransportConfig) {
    this.config = {
      ...config,
      heartbeatInterval: config.heartbeatInterval ?? 30000,
      reconnectTimeout: config.reconnectTimeout ?? 60000,
    };
  }

  /**
   * Start the WebSocket server
   */
  start(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.wss = new WebSocketServer({ port: this.config.port });

        this.wss.on("connection", (ws, req) => {
          this.handleConnection(ws, req);
        });

        this.wss.on("error", (error) => {
          console.error("WebSocket server error:", error);
        });

        // Start heartbeat checker
        this.startHeartbeat();

        console.error(`WebSocket server started on port ${this.config.port}`);
        resolve();
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Stop the WebSocket server
   */
  stop(): Promise<void> {
    return new Promise((resolve) => {
      if (this.heartbeatInterval) {
        clearInterval(this.heartbeatInterval);
        this.heartbeatInterval = null;
      }

      if (this.wss) {
        // Close all client connections
        for (const client of this.clients.values()) {
          try {
            client.send(
              JSON.stringify({
                type: "server_shutdown",
                payload: { message: "Server shutting down" },
                timestamp: new Date().toISOString(),
              }),
            );
          } catch {
            // Ignore send errors during shutdown
          }
        }

        this.wss.close(() => {
          this.clients.clear();
          resolve();
        });
      } else {
        resolve();
      }
    });
  }

  /**
   * Handle new WebSocket connection
   */
  private handleConnection(
    ws: WebSocket,
    _req: import("http").IncomingMessage,
  ): void {
    const clientId = uuidv4();

    const client: WebSocketClient = {
      id: clientId,
      agent_id: null,
      subscriptions: new Set(["task_update", "coordination_status"]),
      send: (message: string) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(message);
        }
      },
    };

    this.clients.set(clientId, client);

    // Send welcome message
    client.send(
      JSON.stringify({
        type: "connected",
        payload: { client_id: clientId },
        timestamp: new Date().toISOString(),
      }),
    );

    ws.on("message", (data) => {
      this.handleMessage(client, data.toString());
    });

    ws.on("close", () => {
      this.clients.delete(clientId);
    });

    ws.on("error", (error) => {
      console.error(`WebSocket client ${clientId} error:`, error);
      this.clients.delete(clientId);
    });

    // Handle pong for heartbeat
    ws.on("pong", () => {
      // Client is alive
    });
  }

  /**
   * Handle incoming message from client
   */
  private handleMessage(client: WebSocketClient, data: string): void {
    try {
      const message = JSON.parse(data);

      switch (message.type) {
        case "subscribe":
          if (Array.isArray(message.events)) {
            for (const event of message.events) {
              client.subscriptions.add(event);
            }
          }
          client.send(
            JSON.stringify({
              type: "subscribed",
              payload: { events: Array.from(client.subscriptions) },
              timestamp: new Date().toISOString(),
            }),
          );
          break;

        case "unsubscribe":
          if (Array.isArray(message.events)) {
            for (const event of message.events) {
              client.subscriptions.delete(event);
            }
          }
          client.send(
            JSON.stringify({
              type: "unsubscribed",
              payload: { events: Array.from(client.subscriptions) },
              timestamp: new Date().toISOString(),
            }),
          );
          break;

        case "identify":
          if (message.agent_id) {
            client.agent_id = message.agent_id;
            client.send(
              JSON.stringify({
                type: "identified",
                payload: { agent_id: client.agent_id },
                timestamp: new Date().toISOString(),
              }),
            );
          }
          break;

        case "ping":
          client.send(
            JSON.stringify({
              type: "pong",
              payload: {},
              timestamp: new Date().toISOString(),
            }),
          );
          break;

        default:
          client.send(
            JSON.stringify({
              type: "error",
              payload: { message: `Unknown message type: ${message.type}` },
              timestamp: new Date().toISOString(),
            }),
          );
      }
    } catch (error) {
      client.send(
        JSON.stringify({
          type: "error",
          payload: { message: "Invalid JSON message" },
          timestamp: new Date().toISOString(),
        }),
      );
    }
  }

  /**
   * Start heartbeat interval to check client connections
   */
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      this.broadcast("heartbeat", { server_time: new Date().toISOString() });
    }, this.config.heartbeatInterval);
  }

  /**
   * Broadcast message to all connected clients with matching subscription
   */
  broadcast(eventType: WebSocketEventType, payload: unknown): void {
    const message: WebSocketMessage = {
      type: eventType,
      payload,
      timestamp: new Date().toISOString(),
    };

    const messageStr = JSON.stringify(message);

    for (const client of this.clients.values()) {
      if (client.subscriptions.has(eventType)) {
        try {
          client.send(messageStr);
        } catch {
          // Client might be disconnected
        }
      }
    }
  }

  /**
   * Send message to a specific agent
   */
  sendToAgent(
    agentId: string,
    eventType: WebSocketEventType,
    payload: unknown,
  ): boolean {
    const message: WebSocketMessage = {
      type: eventType,
      payload,
      timestamp: new Date().toISOString(),
    };

    const messageStr = JSON.stringify(message);

    for (const client of this.clients.values()) {
      if (client.agent_id === agentId) {
        try {
          client.send(messageStr);
          return true;
        } catch {
          return false;
        }
      }
    }

    return false;
  }

  /**
   * Notify all clients about a task update
   */
  notifyTaskUpdate(task: Task, action: string): void {
    this.broadcast("task_update", {
      action,
      task_id: task.id,
      status: task.status,
      priority: task.priority,
      claimed_by: task.claimed_by,
      description: task.description,
    });
  }

  /**
   * Notify all clients about an agent update
   */
  notifyAgentUpdate(agent: Agent, action: string): void {
    this.broadcast("agent_update", {
      action,
      agent_id: agent.id,
      role: agent.role,
      current_task: agent.current_task,
      tasks_completed: agent.tasks_completed,
    });
  }

  /**
   * Notify all clients about a new discovery
   */
  notifyDiscovery(discovery: Discovery): void {
    this.broadcast("discovery_added", {
      discovery_id: discovery.id,
      agent_id: discovery.agent_id,
      content: discovery.content,
      tags: discovery.tags,
    });
  }

  /**
   * Notify all clients about coordination status change
   */
  notifyCoordinationStatus(status: unknown): void {
    this.broadcast("coordination_status", status);
  }

  /**
   * Get number of connected clients
   */
  getClientCount(): number {
    return this.clients.size;
  }

  /**
   * Get list of connected agent IDs
   */
  getConnectedAgents(): string[] {
    const agents: string[] = [];
    for (const client of this.clients.values()) {
      if (client.agent_id) {
        agents.push(client.agent_id);
      }
    }
    return agents;
  }

  /**
   * Check if server is running
   */
  isRunning(): boolean {
    return this.wss !== null;
  }
}

// Factory function to create WebSocket transport
export function createWebSocketTransport(
  port: number = 3001,
): WebSocketTransport {
  return new WebSocketTransport({ port });
}
