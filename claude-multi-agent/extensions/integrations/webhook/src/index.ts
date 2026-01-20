import express, { Request, Response, NextFunction } from "express";
import * as crypto from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as chokidar from "chokidar";
import fetch from "node-fetch";
import { config } from "dotenv";

config();

// Interfaces
interface Task {
  id: string;
  description: string;
  status: "available" | "claimed" | "in_progress" | "done" | "failed";
  priority: number;
  assigned_to?: string;
  dependencies?: string[];
  created_at?: string;
  updated_at?: string;
  result?: string;
  error?: string;
}

interface WebhookConfig {
  id: string;
  url: string;
  secret?: string;
  events: WebhookEvent[];
  active: boolean;
  headers?: Record<string, string>;
  retryCount?: number;
  retryDelay?: number;
  createdAt: string;
}

type WebhookEvent =
  | "task.created"
  | "task.claimed"
  | "task.started"
  | "task.completed"
  | "task.failed"
  | "task.cancelled"
  | "discovery.added"
  | "worker.registered"
  | "worker.disconnected"
  | "coordination.started"
  | "coordination.completed";

interface WebhookPayload {
  event: WebhookEvent;
  timestamp: string;
  data: any;
  signature?: string;
}

interface WebhookDelivery {
  id: string;
  webhookId: string;
  event: WebhookEvent;
  payload: any;
  status: "pending" | "success" | "failed";
  attempts: number;
  lastAttempt?: string;
  response?: {
    status: number;
    body: string;
  };
  error?: string;
}

// Webhook Manager
export class WebhookManager {
  private webhooks: Map<string, WebhookConfig> = new Map();
  private deliveries: WebhookDelivery[] = [];
  private configPath: string;
  private coordinationDir: string;
  private watcher?: chokidar.FSWatcher;
  private lastTaskState: Map<string, Task> = new Map();

  constructor(coordinationDir: string) {
    this.coordinationDir = coordinationDir;
    this.configPath = path.join(coordinationDir, "webhooks.json");
    this.loadWebhooks();
  }

  // Load webhooks from config file
  private loadWebhooks(): void {
    if (fs.existsSync(this.configPath)) {
      try {
        const content = fs.readFileSync(this.configPath, "utf-8");
        const data = JSON.parse(content);
        for (const webhook of data.webhooks || []) {
          this.webhooks.set(webhook.id, webhook);
        }
      } catch (error) {
        console.error("Failed to load webhooks:", error);
      }
    }
  }

  // Save webhooks to config file
  private saveWebhooks(): void {
    const dir = path.dirname(this.configPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(
      this.configPath,
      JSON.stringify({ webhooks: Array.from(this.webhooks.values()) }, null, 2),
    );
  }

  // Register a new webhook
  registerWebhook(
    config: Omit<WebhookConfig, "id" | "createdAt">,
  ): WebhookConfig {
    const id = `webhook-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const webhook: WebhookConfig = {
      ...config,
      id,
      createdAt: new Date().toISOString(),
    };
    this.webhooks.set(id, webhook);
    this.saveWebhooks();
    return webhook;
  }

  // Unregister a webhook
  unregisterWebhook(id: string): boolean {
    const deleted = this.webhooks.delete(id);
    if (deleted) {
      this.saveWebhooks();
    }
    return deleted;
  }

  // Update webhook
  updateWebhook(
    id: string,
    updates: Partial<Omit<WebhookConfig, "id" | "createdAt">>,
  ): WebhookConfig | null {
    const webhook = this.webhooks.get(id);
    if (!webhook) {
      return null;
    }

    const updated = { ...webhook, ...updates };
    this.webhooks.set(id, updated);
    this.saveWebhooks();
    return updated;
  }

  // Get all webhooks
  getWebhooks(): WebhookConfig[] {
    return Array.from(this.webhooks.values());
  }

  // Get webhook by ID
  getWebhook(id: string): WebhookConfig | undefined {
    return this.webhooks.get(id);
  }

  // Generate signature for payload
  private generateSignature(payload: string, secret: string): string {
    return crypto.createHmac("sha256", secret).update(payload).digest("hex");
  }

  // Verify incoming webhook signature
  verifySignature(payload: string, signature: string, secret: string): boolean {
    const expected = this.generateSignature(payload, secret);
    return crypto.timingSafeEqual(
      Buffer.from(signature),
      Buffer.from(expected),
    );
  }

  // Send webhook
  async sendWebhook(
    webhook: WebhookConfig,
    event: WebhookEvent,
    data: any,
  ): Promise<WebhookDelivery> {
    const deliveryId = `delivery-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const payload: WebhookPayload = {
      event,
      timestamp: new Date().toISOString(),
      data,
    };

    const payloadStr = JSON.stringify(payload);

    const delivery: WebhookDelivery = {
      id: deliveryId,
      webhookId: webhook.id,
      event,
      payload,
      status: "pending",
      attempts: 0,
    };

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "X-Webhook-Event": event,
      "X-Webhook-Delivery": deliveryId,
      ...webhook.headers,
    };

    if (webhook.secret) {
      headers["X-Webhook-Signature"] =
        `sha256=${this.generateSignature(payloadStr, webhook.secret)}`;
    }

    const maxRetries = webhook.retryCount || 3;
    const retryDelay = webhook.retryDelay || 1000;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      delivery.attempts = attempt;
      delivery.lastAttempt = new Date().toISOString();

      try {
        const response = await fetch(webhook.url, {
          method: "POST",
          headers,
          body: payloadStr,
          timeout: 30000,
        });

        const responseBody = await response.text();
        delivery.response = {
          status: response.status,
          body: responseBody.substring(0, 1000),
        };

        if (response.ok) {
          delivery.status = "success";
          break;
        } else {
          delivery.error = `HTTP ${response.status}: ${responseBody.substring(0, 200)}`;
          if (attempt < maxRetries) {
            await this.delay(retryDelay * attempt);
          }
        }
      } catch (error) {
        delivery.error = `${error}`;
        if (attempt < maxRetries) {
          await this.delay(retryDelay * attempt);
        }
      }
    }

    if (delivery.status === "pending") {
      delivery.status = "failed";
    }

    this.deliveries.push(delivery);

    // Keep only last 1000 deliveries
    if (this.deliveries.length > 1000) {
      this.deliveries = this.deliveries.slice(-1000);
    }

    return delivery;
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // Trigger webhooks for an event
  async trigger(event: WebhookEvent, data: any): Promise<WebhookDelivery[]> {
    const deliveries: WebhookDelivery[] = [];

    for (const webhook of this.webhooks.values()) {
      if (webhook.active && webhook.events.includes(event)) {
        const delivery = await this.sendWebhook(webhook, event, data);
        deliveries.push(delivery);
      }
    }

    return deliveries;
  }

  // Get delivery history
  getDeliveries(webhookId?: string): WebhookDelivery[] {
    if (webhookId) {
      return this.deliveries.filter((d) => d.webhookId === webhookId);
    }
    return this.deliveries;
  }

  // Load tasks from coordination directory
  private loadTasks(): Task[] {
    const tasksPath = path.join(this.coordinationDir, "tasks.json");
    if (!fs.existsSync(tasksPath)) {
      return [];
    }

    try {
      const content = fs.readFileSync(tasksPath, "utf-8");
      const data = JSON.parse(content);
      return data.tasks || [];
    } catch {
      return [];
    }
  }

  // Start watching for changes
  startWatching(): void {
    const tasksPath = path.join(this.coordinationDir, "tasks.json");
    const discoveriesPath = path.join(
      this.coordinationDir,
      "context",
      "discoveries.md",
    );

    // Initialize task state
    const tasks = this.loadTasks();
    for (const task of tasks) {
      this.lastTaskState.set(task.id, { ...task });
    }

    this.watcher = chokidar.watch([tasksPath, discoveriesPath], {
      persistent: true,
      ignoreInitial: true,
    });

    this.watcher.on("change", async (filePath) => {
      if (filePath.endsWith("tasks.json")) {
        await this.handleTasksChange();
      } else if (filePath.endsWith("discoveries.md")) {
        await this.trigger("discovery.added", { file: filePath });
      }
    });
  }

  private async handleTasksChange(): Promise<void> {
    const tasks = this.loadTasks();

    for (const task of tasks) {
      const lastState = this.lastTaskState.get(task.id);

      if (!lastState) {
        // New task
        await this.trigger("task.created", task);
      } else if (lastState.status !== task.status) {
        // Status changed
        switch (task.status) {
          case "claimed":
            await this.trigger("task.claimed", task);
            break;
          case "in_progress":
            await this.trigger("task.started", task);
            break;
          case "done":
            await this.trigger("task.completed", task);
            break;
          case "failed":
            await this.trigger("task.failed", task);
            break;
        }
      }

      this.lastTaskState.set(task.id, { ...task });
    }
  }

  stopWatching(): void {
    if (this.watcher) {
      this.watcher.close();
    }
  }
}

// Express server for webhook management
export function createWebhookServer(
  manager: WebhookManager,
  port: number = 3001,
): express.Application {
  const app = express();
  app.use(express.json());

  // CORS middleware
  app.use((req: Request, res: Response, next: NextFunction) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header(
      "Access-Control-Allow-Methods",
      "GET, POST, PUT, DELETE, OPTIONS",
    );
    res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    if (req.method === "OPTIONS") {
      res.sendStatus(200);
      return;
    }
    next();
  });

  // List webhooks
  app.get("/webhooks", (req: Request, res: Response) => {
    res.json({ webhooks: manager.getWebhooks() });
  });

  // Get webhook by ID
  app.get("/webhooks/:id", (req: Request, res: Response) => {
    const webhook = manager.getWebhook(req.params.id);
    if (!webhook) {
      res.status(404).json({ error: "Webhook not found" });
      return;
    }
    res.json(webhook);
  });

  // Create webhook
  app.post("/webhooks", (req: Request, res: Response) => {
    const { url, secret, events, headers, retryCount, retryDelay } = req.body;

    if (!url || !events || !Array.isArray(events)) {
      res.status(400).json({ error: "url and events are required" });
      return;
    }

    const webhook = manager.registerWebhook({
      url,
      secret,
      events,
      active: true,
      headers,
      retryCount,
      retryDelay,
    });

    res.status(201).json(webhook);
  });

  // Update webhook
  app.put("/webhooks/:id", (req: Request, res: Response) => {
    const webhook = manager.updateWebhook(req.params.id, req.body);
    if (!webhook) {
      res.status(404).json({ error: "Webhook not found" });
      return;
    }
    res.json(webhook);
  });

  // Delete webhook
  app.delete("/webhooks/:id", (req: Request, res: Response) => {
    const deleted = manager.unregisterWebhook(req.params.id);
    if (!deleted) {
      res.status(404).json({ error: "Webhook not found" });
      return;
    }
    res.status(204).send();
  });

  // Test webhook
  app.post("/webhooks/:id/test", async (req: Request, res: Response) => {
    const webhook = manager.getWebhook(req.params.id);
    if (!webhook) {
      res.status(404).json({ error: "Webhook not found" });
      return;
    }

    const delivery = await manager.sendWebhook(webhook, "task.created", {
      id: "test-task",
      description: "Test webhook delivery",
      status: "available",
      priority: 3,
      created_at: new Date().toISOString(),
    });

    res.json(delivery);
  });

  // Get deliveries
  app.get("/webhooks/:id/deliveries", (req: Request, res: Response) => {
    const deliveries = manager.getDeliveries(req.params.id);
    res.json({ deliveries });
  });

  // Get all deliveries
  app.get("/deliveries", (req: Request, res: Response) => {
    const deliveries = manager.getDeliveries();
    res.json({ deliveries });
  });

  // Manually trigger event
  app.post("/trigger", async (req: Request, res: Response) => {
    const { event, data } = req.body;

    if (!event || !data) {
      res.status(400).json({ error: "event and data are required" });
      return;
    }

    const deliveries = await manager.trigger(event, data);
    res.json({ deliveries });
  });

  // Health check
  app.get("/health", (req: Request, res: Response) => {
    res.json({ status: "ok" });
  });

  return app;
}

// Main entry point
if (require.main === module) {
  const coordinationDir = process.env.COORDINATION_DIR || ".coordination";
  const port = parseInt(process.env.WEBHOOK_PORT || "3001");

  const manager = new WebhookManager(coordinationDir);
  manager.startWatching();

  const app = createWebhookServer(manager, port);
  app.listen(port, () => {
    console.log(`Webhook server running on port ${port}`);
  });

  process.on("SIGINT", () => {
    manager.stopWatching();
    process.exit(0);
  });
}

export default WebhookManager;
