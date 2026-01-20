import { App, LogLevel } from "@slack/bolt";
import { WebClient } from "@slack/web-api";
import * as fs from "fs";
import * as path from "path";
import * as chokidar from "chokidar";
import { config } from "dotenv";

config();

// Configuration interface
interface SlackBotConfig {
  slackBotToken: string;
  slackSigningSecret: string;
  slackAppToken?: string;
  coordinationDir: string;
  notificationChannel: string;
  port?: number;
}

// Task interface
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

// Slack Bot for Claude Coordinator
export class ClaudeCoordinatorSlackBot {
  private app: App;
  private webClient: WebClient;
  private config: SlackBotConfig;
  private watcher?: chokidar.FSWatcher;
  private lastTaskState: Map<string, Task> = new Map();

  constructor(config: SlackBotConfig) {
    this.config = config;
    this.webClient = new WebClient(config.slackBotToken);

    this.app = new App({
      token: config.slackBotToken,
      signingSecret: config.slackSigningSecret,
      appToken: config.slackAppToken,
      socketMode: !!config.slackAppToken,
      port: config.port || 3000,
      logLevel: LogLevel.INFO,
    });

    this.setupCommands();
    this.setupActions();
    this.setupEvents();
  }

  private setupCommands(): void {
    // List tasks command
    this.app.command("/tasks", async ({ command, ack, respond }) => {
      await ack();

      const tasks = this.loadTasks();
      const statusFilter = command.text?.trim() || "all";

      let filteredTasks = tasks;
      if (statusFilter !== "all") {
        filteredTasks = tasks.filter((t) => t.status === statusFilter);
      }

      if (filteredTasks.length === 0) {
        await respond({
          text: `No tasks found${statusFilter !== "all" ? ` with status: ${statusFilter}` : ""}.`,
        });
        return;
      }

      const blocks = this.formatTaskListBlocks(filteredTasks);
      await respond({ blocks });
    });

    // Create task command
    this.app.command("/create-task", async ({ command, ack, respond }) => {
      await ack();

      const text = command.text?.trim();
      if (!text) {
        await respond({ text: "Please provide a task description." });
        return;
      }

      // Parse: "description" or "description | priority"
      const parts = text.split("|").map((p) => p.trim());
      const description = parts[0];
      const priority = parseInt(parts[1]) || 3;

      const task = this.createTask(description, priority);
      await respond({
        text: `Task created successfully!`,
        blocks: this.formatTaskBlocks(task, "Task Created"),
      });

      await this.notifyChannel(
        `New task created by <@${command.user_id}>`,
        this.formatTaskBlocks(task, "New Task"),
      );
    });

    // Claim task command
    this.app.command("/claim-task", async ({ command, ack, respond }) => {
      await ack();

      const taskId = command.text?.trim();
      if (!taskId) {
        await respond({ text: "Please provide a task ID." });
        return;
      }

      try {
        const task = this.claimTask(taskId, command.user_id);
        await respond({
          text: `Task claimed successfully!`,
          blocks: this.formatTaskBlocks(task, "Task Claimed"),
        });

        await this.notifyChannel(
          `Task claimed by <@${command.user_id}>`,
          this.formatTaskBlocks(task, "Task Claimed"),
        );
      } catch (error) {
        await respond({ text: `Error: ${error}` });
      }
    });

    // Complete task command
    this.app.command("/complete-task", async ({ command, ack, respond }) => {
      await ack();

      const parts = command.text?.split("|").map((p) => p.trim()) || [];
      const taskId = parts[0];
      const result = parts[1] || "Completed via Slack";

      if (!taskId) {
        await respond({
          text: "Please provide a task ID. Usage: /complete-task task-id | result",
        });
        return;
      }

      try {
        const task = this.completeTask(taskId, result);
        await respond({
          text: `Task completed!`,
          blocks: this.formatTaskBlocks(task, "Task Completed"),
        });

        await this.notifyChannel(
          `Task completed by <@${command.user_id}>`,
          this.formatTaskBlocks(task, "Task Completed"),
        );
      } catch (error) {
        await respond({ text: `Error: ${error}` });
      }
    });

    // Status command
    this.app.command("/coord-status", async ({ command, ack, respond }) => {
      await ack();

      const tasks = this.loadTasks();
      const statusCounts = {
        available: tasks.filter((t) => t.status === "available").length,
        claimed: tasks.filter((t) => t.status === "claimed").length,
        in_progress: tasks.filter((t) => t.status === "in_progress").length,
        done: tasks.filter((t) => t.status === "done").length,
        failed: tasks.filter((t) => t.status === "failed").length,
      };

      await respond({
        blocks: [
          {
            type: "header",
            text: {
              type: "plain_text",
              text: "Coordination Status",
            },
          },
          {
            type: "section",
            fields: [
              {
                type: "mrkdwn",
                text: `*Available:* ${statusCounts.available}`,
              },
              { type: "mrkdwn", text: `*Claimed:* ${statusCounts.claimed}` },
              {
                type: "mrkdwn",
                text: `*In Progress:* ${statusCounts.in_progress}`,
              },
              { type: "mrkdwn", text: `*Done:* ${statusCounts.done}` },
              { type: "mrkdwn", text: `*Failed:* ${statusCounts.failed}` },
              { type: "mrkdwn", text: `*Total:* ${tasks.length}` },
            ],
          },
        ],
      });
    });
  }

  private setupActions(): void {
    // Handle claim button clicks
    this.app.action("claim_task", async ({ body, ack, respond }) => {
      await ack();

      const actionBody = body as any;
      const taskId = actionBody.actions[0].value;
      const userId = body.user.id;

      try {
        const task = this.claimTask(taskId, userId);
        await respond({
          text: `Task claimed!`,
          blocks: this.formatTaskBlocks(task, "Task Claimed"),
          replace_original: true,
        });

        await this.notifyChannel(
          `Task claimed by <@${userId}>`,
          this.formatTaskBlocks(task, "Task Claimed"),
        );
      } catch (error) {
        await respond({ text: `Error claiming task: ${error}` });
      }
    });

    // Handle complete button clicks
    this.app.action("complete_task", async ({ body, ack, client }) => {
      await ack();

      const actionBody = body as any;
      const taskId = actionBody.actions[0].value;

      // Open a modal to get the result
      await client.views.open({
        trigger_id: actionBody.trigger_id,
        view: {
          type: "modal",
          callback_id: "complete_task_modal",
          private_metadata: taskId,
          title: {
            type: "plain_text",
            text: "Complete Task",
          },
          submit: {
            type: "plain_text",
            text: "Complete",
          },
          blocks: [
            {
              type: "input",
              block_id: "result_block",
              element: {
                type: "plain_text_input",
                action_id: "result_input",
                multiline: true,
                placeholder: {
                  type: "plain_text",
                  text: "Enter the task result...",
                },
              },
              label: {
                type: "plain_text",
                text: "Result",
              },
            },
          ],
        },
      });
    });

    // Handle modal submission
    this.app.view("complete_task_modal", async ({ ack, body, view }) => {
      await ack();

      const taskId = view.private_metadata;
      const result =
        view.state.values.result_block.result_input.value ||
        "Completed via Slack";
      const userId = body.user.id;

      try {
        const task = this.completeTask(taskId, result);
        await this.notifyChannel(
          `Task completed by <@${userId}>`,
          this.formatTaskBlocks(task, "Task Completed"),
        );
      } catch (error) {
        console.error("Error completing task:", error);
      }
    });
  }

  private setupEvents(): void {
    // Handle app mentions
    this.app.event("app_mention", async ({ event, say }) => {
      const text = event.text.toLowerCase();

      if (text.includes("status")) {
        const tasks = this.loadTasks();
        const available = tasks.filter((t) => t.status === "available").length;
        const inProgress = tasks.filter(
          (t) => t.status === "in_progress" || t.status === "claimed",
        ).length;
        const done = tasks.filter((t) => t.status === "done").length;

        await say(
          `Current status: ${available} available, ${inProgress} in progress, ${done} completed.`,
        );
      } else if (text.includes("help")) {
        await say({
          blocks: [
            {
              type: "section",
              text: {
                type: "mrkdwn",
                text:
                  "*Claude Coordinator Commands:*\n" +
                  "- `/tasks [status]` - List tasks (optionally filter by status)\n" +
                  "- `/create-task description | priority` - Create a new task\n" +
                  "- `/claim-task task-id` - Claim a task\n" +
                  "- `/complete-task task-id | result` - Complete a task\n" +
                  "- `/coord-status` - View coordination status",
              },
            },
          ],
        });
      } else {
        await say(
          'Hello! I\'m the Claude Coordinator bot. Mention me with "help" to see available commands.',
        );
      }
    });
  }

  private loadTasks(): Task[] {
    const tasksPath = path.join(this.config.coordinationDir, "tasks.json");
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

  private saveTasks(tasks: Task[]): void {
    const tasksPath = path.join(this.config.coordinationDir, "tasks.json");
    fs.writeFileSync(
      tasksPath,
      JSON.stringify({ tasks, updated_at: new Date().toISOString() }, null, 2),
    );
  }

  private createTask(description: string, priority: number): Task {
    const tasks = this.loadTasks();
    const newTask: Task = {
      id: `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      description,
      status: "available",
      priority,
      created_at: new Date().toISOString(),
    };

    tasks.push(newTask);
    this.saveTasks(tasks);
    return newTask;
  }

  private claimTask(taskId: string, userId: string): Task {
    const tasks = this.loadTasks();
    const task = tasks.find((t) => t.id === taskId);

    if (!task) {
      throw new Error(`Task not found: ${taskId}`);
    }

    if (task.status !== "available") {
      throw new Error(`Task is not available: ${task.status}`);
    }

    task.status = "claimed";
    task.assigned_to = `slack:${userId}`;
    task.updated_at = new Date().toISOString();

    this.saveTasks(tasks);
    return task;
  }

  private completeTask(taskId: string, result: string): Task {
    const tasks = this.loadTasks();
    const task = tasks.find((t) => t.id === taskId);

    if (!task) {
      throw new Error(`Task not found: ${taskId}`);
    }

    task.status = "done";
    task.result = result;
    task.updated_at = new Date().toISOString();

    this.saveTasks(tasks);
    return task;
  }

  private formatTaskBlocks(task: Task, title: string): any[] {
    const statusEmoji = {
      available: ":white_circle:",
      claimed: ":large_orange_circle:",
      in_progress: ":large_blue_circle:",
      done: ":white_check_mark:",
      failed: ":x:",
    };

    const blocks: any[] = [
      {
        type: "header",
        text: {
          type: "plain_text",
          text: title,
        },
      },
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: `*${task.description}*`,
        },
        fields: [
          { type: "mrkdwn", text: `*ID:* \`${task.id}\`` },
          {
            type: "mrkdwn",
            text: `*Status:* ${statusEmoji[task.status]} ${task.status}`,
          },
          { type: "mrkdwn", text: `*Priority:* ${task.priority}` },
          {
            type: "mrkdwn",
            text: `*Assigned:* ${task.assigned_to || "Unassigned"}`,
          },
        ],
      },
    ];

    // Add action buttons for available tasks
    if (task.status === "available") {
      blocks.push({
        type: "actions",
        elements: [
          {
            type: "button",
            text: { type: "plain_text", text: "Claim Task" },
            style: "primary",
            action_id: "claim_task",
            value: task.id,
          },
        ],
      });
    }

    // Add complete button for claimed/in_progress tasks
    if (task.status === "claimed" || task.status === "in_progress") {
      blocks.push({
        type: "actions",
        elements: [
          {
            type: "button",
            text: { type: "plain_text", text: "Complete Task" },
            style: "primary",
            action_id: "complete_task",
            value: task.id,
          },
        ],
      });
    }

    return blocks;
  }

  private formatTaskListBlocks(tasks: Task[]): any[] {
    const blocks: any[] = [
      {
        type: "header",
        text: {
          type: "plain_text",
          text: `Tasks (${tasks.length})`,
        },
      },
    ];

    const statusEmoji = {
      available: ":white_circle:",
      claimed: ":large_orange_circle:",
      in_progress: ":large_blue_circle:",
      done: ":white_check_mark:",
      failed: ":x:",
    };

    for (const task of tasks.slice(0, 10)) {
      blocks.push({
        type: "section",
        text: {
          type: "mrkdwn",
          text: `${statusEmoji[task.status]} *${task.description}*\n\`${task.id}\` | Priority: ${task.priority}`,
        },
        accessory:
          task.status === "available"
            ? {
                type: "button",
                text: { type: "plain_text", text: "Claim" },
                action_id: "claim_task",
                value: task.id,
              }
            : undefined,
      });
    }

    if (tasks.length > 10) {
      blocks.push({
        type: "context",
        elements: [
          {
            type: "mrkdwn",
            text: `_Showing 10 of ${tasks.length} tasks_`,
          },
        ],
      });
    }

    return blocks;
  }

  private async notifyChannel(text: string, blocks?: any[]): Promise<void> {
    try {
      await this.webClient.chat.postMessage({
        channel: this.config.notificationChannel,
        text,
        blocks,
      });
    } catch (error) {
      console.error("Failed to send notification:", error);
    }
  }

  // Watch for file changes and notify
  public startWatching(): void {
    const tasksPath = path.join(this.config.coordinationDir, "tasks.json");

    // Initialize state
    const tasks = this.loadTasks();
    for (const task of tasks) {
      this.lastTaskState.set(task.id, { ...task });
    }

    this.watcher = chokidar.watch(tasksPath, {
      persistent: true,
      ignoreInitial: true,
    });

    this.watcher.on("change", async () => {
      const tasks = this.loadTasks();

      for (const task of tasks) {
        const lastState = this.lastTaskState.get(task.id);

        if (!lastState) {
          // New task
          await this.notifyChannel(
            "New task added",
            this.formatTaskBlocks(task, "New Task"),
          );
        } else if (lastState.status !== task.status) {
          // Status changed
          await this.notifyChannel(
            `Task status changed: ${lastState.status} -> ${task.status}`,
            this.formatTaskBlocks(task, "Task Updated"),
          );
        }

        this.lastTaskState.set(task.id, { ...task });
      }
    });
  }

  public stopWatching(): void {
    if (this.watcher) {
      this.watcher.close();
    }
  }

  public async start(): Promise<void> {
    await this.app.start();
    console.log(`Slack bot is running on port ${this.config.port || 3000}`);
    this.startWatching();
  }
}

// Main entry point
if (require.main === module) {
  const config: SlackBotConfig = {
    slackBotToken: process.env.SLACK_BOT_TOKEN || "",
    slackSigningSecret: process.env.SLACK_SIGNING_SECRET || "",
    slackAppToken: process.env.SLACK_APP_TOKEN,
    coordinationDir: process.env.COORDINATION_DIR || ".coordination",
    notificationChannel: process.env.SLACK_CHANNEL || "#claude-coordinator",
    port: parseInt(process.env.PORT || "3000"),
  };

  if (!config.slackBotToken || !config.slackSigningSecret) {
    console.error(
      "Missing required environment variables: SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET",
    );
    process.exit(1);
  }

  const bot = new ClaudeCoordinatorSlackBot(config);
  bot.start().catch(console.error);
}

export default ClaudeCoordinatorSlackBot;
