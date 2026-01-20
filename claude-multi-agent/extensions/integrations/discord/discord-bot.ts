/**
 * Discord Bot Integration (ext-008)
 *
 * Provides a Discord bot for interacting with Claude Multi-Agent Coordination.
 * Allows teams to monitor tasks, claim work, and receive notifications via Discord.
 */

// ============================================================================
// Types
// ============================================================================

export interface DiscordConfig {
  botToken: string;
  guildId?: string;
  channelId?: string;
  enableCommands?: boolean;
  enableNotifications?: boolean;
  commandPrefix?: string;
}

export interface DiscordMessage {
  content: string;
  embeds?: DiscordEmbed[];
  components?: any[];
}

export interface DiscordEmbed {
  title?: string;
  description?: string;
  color?: number;
  fields?: Array<{ name: string; value: string; inline?: boolean }>;
  footer?: { text: string };
  timestamp?: string;
}

export interface DiscordCommand {
  name: string;
  description: string;
  handler: (interaction: any) => Promise<void>;
}

// ============================================================================
// Discord API Client
// ============================================================================

export class DiscordClient {
  private botToken: string;
  private baseUrl = "https://discord.com/api/v10";

  constructor(botToken: string) {
    this.botToken = botToken;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: any,
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: {
        Authorization: `Bot ${this.botToken}`,
        "Content-Type": "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Discord API error: ${response.status} - ${error}`);
    }

    if (response.status === 204) {
      return {} as T;
    }

    return response.json();
  }

  async sendMessage(channelId: string, message: DiscordMessage): Promise<any> {
    return this.request("POST", `/channels/${channelId}/messages`, message);
  }

  async editMessage(
    channelId: string,
    messageId: string,
    message: DiscordMessage,
  ): Promise<any> {
    return this.request(
      "PATCH",
      `/channels/${channelId}/messages/${messageId}`,
      message,
    );
  }

  async createReaction(
    channelId: string,
    messageId: string,
    emoji: string,
  ): Promise<void> {
    await this.request(
      "PUT",
      `/channels/${channelId}/messages/${messageId}/reactions/${emoji}/@me`,
    );
  }

  async getGuild(guildId: string): Promise<any> {
    return this.request("GET", `/guilds/${guildId}`);
  }

  async getChannel(channelId: string): Promise<any> {
    return this.request("GET", `/channels/${channelId}`);
  }

  async registerCommand(
    applicationId: string,
    guildId: string | undefined,
    command: {
      name: string;
      description: string;
      options?: any[];
    },
  ): Promise<any> {
    const path = guildId
      ? `/applications/${applicationId}/guilds/${guildId}/commands`
      : `/applications/${applicationId}/commands`;

    return this.request("POST", path, command);
  }

  async respondToInteraction(
    interactionId: string,
    interactionToken: string,
    response: { type: number; data: DiscordMessage },
  ): Promise<void> {
    await this.request(
      "POST",
      `/interactions/${interactionId}/${interactionToken}/callback`,
      response,
    );
  }
}

// ============================================================================
// Discord Bot Service
// ============================================================================

export class DiscordBot {
  private client: DiscordClient;
  private config: DiscordConfig;
  private commands: Map<string, DiscordCommand> = new Map();
  private messageCache: Map<string, string> = new Map(); // taskId -> messageId

  constructor(config: DiscordConfig) {
    this.config = config;
    this.client = new DiscordClient(config.botToken);
  }

  /**
   * Initialize the bot and register commands
   */
  async initialize(applicationId: string): Promise<void> {
    if (this.config.enableCommands !== false) {
      await this.registerDefaultCommands(applicationId);
    }
  }

  /**
   * Register default slash commands
   */
  private async registerDefaultCommands(applicationId: string): Promise<void> {
    const commands = [
      {
        name: "tasks",
        description: "List all available tasks",
      },
      {
        name: "task",
        description: "Get details about a specific task",
        options: [
          {
            name: "id",
            description: "Task ID",
            type: 3, // STRING
            required: true,
          },
        ],
      },
      {
        name: "claim",
        description: "Claim a task",
        options: [
          {
            name: "id",
            description: "Task ID",
            type: 3,
            required: true,
          },
          {
            name: "agent",
            description: "Agent ID",
            type: 3,
            required: true,
          },
        ],
      },
      {
        name: "status",
        description: "Get coordination system status",
      },
      {
        name: "agents",
        description: "List all registered agents",
      },
    ];

    for (const command of commands) {
      try {
        await this.client.registerCommand(
          applicationId,
          this.config.guildId,
          command,
        );
      } catch (error) {
        console.error(`Failed to register command ${command.name}:`, error);
      }
    }
  }

  /**
   * Send a task notification to Discord
   */
  async notifyTaskCreated(task: {
    id: string;
    description: string;
    priority: number;
    status: string;
  }): Promise<void> {
    if (!this.config.channelId || this.config.enableNotifications === false) {
      return;
    }

    const embed = this.buildTaskEmbed(task, "New Task Available");
    const message = await this.client.sendMessage(this.config.channelId, {
      content: "🆕 **New Task**",
      embeds: [embed],
    });

    this.messageCache.set(task.id, message.id);
  }

  /**
   * Notify when a task is claimed
   */
  async notifyTaskClaimed(task: {
    id: string;
    description: string;
    assigned_to: string;
  }): Promise<void> {
    if (!this.config.channelId || this.config.enableNotifications === false) {
      return;
    }

    const messageId = this.messageCache.get(task.id);
    if (messageId) {
      // Update existing message
      const embed: DiscordEmbed = {
        title: "Task Claimed",
        description: task.description,
        color: 0xffa500, // Orange
        fields: [
          { name: "Task ID", value: task.id, inline: true },
          { name: "Assigned To", value: task.assigned_to, inline: true },
          { name: "Status", value: "Claimed", inline: true },
        ],
        timestamp: new Date().toISOString(),
      };

      await this.client.editMessage(this.config.channelId, messageId, {
        content: "⚡ **Task Claimed**",
        embeds: [embed],
      });
    } else {
      // Send new message
      await this.client.sendMessage(this.config.channelId, {
        content: `⚡ **Task Claimed** by ${task.assigned_to}`,
        embeds: [
          {
            description: task.description,
            fields: [{ name: "Task ID", value: task.id }],
            color: 0xffa500,
          },
        ],
      });
    }
  }

  /**
   * Notify when a task is completed
   */
  async notifyTaskCompleted(task: {
    id: string;
    description: string;
    assigned_to?: string;
  }): Promise<void> {
    if (!this.config.channelId || this.config.enableNotifications === false) {
      return;
    }

    const embed: DiscordEmbed = {
      title: "✅ Task Completed",
      description: task.description,
      color: 0x00ff00, // Green
      fields: [
        { name: "Task ID", value: task.id, inline: true },
        ...(task.assigned_to
          ? [{ name: "Completed By", value: task.assigned_to, inline: true }]
          : []),
      ],
      timestamp: new Date().toISOString(),
    };

    await this.client.sendMessage(this.config.channelId, {
      embeds: [embed],
    });

    this.messageCache.delete(task.id);
  }

  /**
   * Notify when a task fails
   */
  async notifyTaskFailed(task: {
    id: string;
    description: string;
    error?: string;
  }): Promise<void> {
    if (!this.config.channelId || this.config.enableNotifications === false) {
      return;
    }

    const embed: DiscordEmbed = {
      title: "❌ Task Failed",
      description: task.description,
      color: 0xff0000, // Red
      fields: [
        { name: "Task ID", value: task.id, inline: true },
        ...(task.error
          ? [{ name: "Error", value: task.error.substring(0, 1024) }]
          : []),
      ],
      timestamp: new Date().toISOString(),
    };

    await this.client.sendMessage(this.config.channelId, {
      embeds: [embed],
    });

    this.messageCache.delete(task.id);
  }

  /**
   * Send a status update
   */
  async sendStatusUpdate(stats: {
    totalTasks: number;
    availableTasks: number;
    inProgressTasks: number;
    completedTasks: number;
    activeAgents: number;
  }): Promise<void> {
    if (!this.config.channelId) return;

    const embed: DiscordEmbed = {
      title: "📊 Coordination Status",
      color: 0x0099ff, // Blue
      fields: [
        {
          name: "Total Tasks",
          value: stats.totalTasks.toString(),
          inline: true,
        },
        {
          name: "Available",
          value: stats.availableTasks.toString(),
          inline: true,
        },
        {
          name: "In Progress",
          value: stats.inProgressTasks.toString(),
          inline: true,
        },
        {
          name: "Completed",
          value: stats.completedTasks.toString(),
          inline: true,
        },
        {
          name: "Active Agents",
          value: stats.activeAgents.toString(),
          inline: true,
        },
      ],
      timestamp: new Date().toISOString(),
    };

    await this.client.sendMessage(this.config.channelId, { embeds: [embed] });
  }

  /**
   * Send a list of tasks
   */
  async sendTaskList(
    tasks: Array<{
      id: string;
      description: string;
      status: string;
      priority: number;
    }>,
  ): Promise<void> {
    if (!this.config.channelId) return;

    if (tasks.length === 0) {
      await this.client.sendMessage(this.config.channelId, {
        content: "No tasks available.",
      });
      return;
    }

    const taskLines = tasks.slice(0, 10).map((task) => {
      const priorityEmoji = this.getPriorityEmoji(task.priority);
      const statusEmoji = this.getStatusEmoji(task.status);
      return `${statusEmoji} ${priorityEmoji} **${task.id}** - ${task.description.substring(0, 60)}...`;
    });

    const embed: DiscordEmbed = {
      title: "📋 Available Tasks",
      description: taskLines.join("\n"),
      color: 0x0099ff,
      footer: {
        text:
          tasks.length > 10
            ? `Showing 10 of ${tasks.length} tasks`
            : `${tasks.length} task(s)`,
      },
    };

    await this.client.sendMessage(this.config.channelId, { embeds: [embed] });
  }

  /**
   * Send agent list
   */
  async sendAgentList(
    agents: Array<{
      id: string;
      status: string;
      currentTask?: string;
      lastHeartbeat: string;
    }>,
  ): Promise<void> {
    if (!this.config.channelId) return;

    if (agents.length === 0) {
      await this.client.sendMessage(this.config.channelId, {
        content: "No agents registered.",
      });
      return;
    }

    const agentLines = agents.slice(0, 10).map((agent) => {
      const statusEmoji = agent.status === "active" ? "🟢" : "🔴";
      const taskInfo = agent.currentTask ? ` (Task: ${agent.currentTask})` : "";
      return `${statusEmoji} **${agent.id}**${taskInfo}`;
    });

    const embed: DiscordEmbed = {
      title: "🤖 Registered Agents",
      description: agentLines.join("\n"),
      color: 0x00ff00,
      footer: {
        text:
          agents.length > 10
            ? `Showing 10 of ${agents.length} agents`
            : `${agents.length} agent(s)`,
      },
    };

    await this.client.sendMessage(this.config.channelId, { embeds: [embed] });
  }

  /**
   * Build an embed for a task
   */
  private buildTaskEmbed(
    task: {
      id: string;
      description: string;
      priority: number;
      status: string;
      assigned_to?: string;
    },
    title: string,
  ): DiscordEmbed {
    const priorityEmoji = this.getPriorityEmoji(task.priority);
    const statusEmoji = this.getStatusEmoji(task.status);

    return {
      title,
      description: task.description,
      color: this.getColorForPriority(task.priority),
      fields: [
        { name: "Task ID", value: task.id, inline: true },
        {
          name: "Priority",
          value: `${priorityEmoji} ${task.priority}`,
          inline: true,
        },
        {
          name: "Status",
          value: `${statusEmoji} ${task.status}`,
          inline: true,
        },
        ...(task.assigned_to
          ? [{ name: "Assigned To", value: task.assigned_to, inline: true }]
          : []),
      ],
      timestamp: new Date().toISOString(),
    };
  }

  private getPriorityEmoji(priority: number): string {
    switch (priority) {
      case 1:
        return "🔴";
      case 2:
        return "🟠";
      case 3:
        return "🟡";
      case 4:
        return "🟢";
      case 5:
        return "🔵";
      default:
        return "⚪";
    }
  }

  private getStatusEmoji(status: string): string {
    switch (status) {
      case "available":
        return "🆕";
      case "claimed":
        return "👤";
      case "in_progress":
        return "⚡";
      case "done":
        return "✅";
      case "failed":
        return "❌";
      default:
        return "❓";
    }
  }

  private getColorForPriority(priority: number): number {
    switch (priority) {
      case 1:
        return 0xff0000; // Red
      case 2:
        return 0xff6600; // Orange
      case 3:
        return 0xffcc00; // Yellow
      case 4:
        return 0x00ff00; // Green
      case 5:
        return 0x0099ff; // Blue
      default:
        return 0x999999; // Gray
    }
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createDiscordBot(config: DiscordConfig): DiscordBot {
  return new DiscordBot(config);
}
