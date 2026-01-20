/**
 * Notion Integration (ext-007)
 *
 * Integrates Claude Multi-Agent Coordination with Notion for knowledge management.
 * Creates and updates pages in Notion database for tasks and discoveries.
 */

// ============================================================================
// Types
// ============================================================================

export interface NotionConfig {
  apiKey: string;
  databaseId: string;
  syncEnabled?: boolean;
  autoCreatePages?: boolean;
  discoveriesDatabaseId?: string;
}

export interface NotionPage {
  id: string;
  url: string;
  created_time: string;
  last_edited_time: string;
  properties: Record<string, any>;
}

export interface NotionDatabase {
  id: string;
  title: Array<{ plain_text: string }>;
  properties: Record<string, any>;
}

export interface TaskToNotionMapping {
  taskId: string;
  notionPageId: string;
  notionUrl: string;
  syncedAt: string;
  lastSyncDirection: "toNotion" | "fromNotion";
}

// ============================================================================
// Notion API Client
// ============================================================================

export class NotionClient {
  private apiKey: string;
  private baseUrl = "https://api.notion.com/v1";
  private notionVersion = "2022-06-28";

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: any,
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
        "Notion-Version": this.notionVersion,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Notion API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  async getDatabase(databaseId: string): Promise<NotionDatabase> {
    return this.request<NotionDatabase>("GET", `/databases/${databaseId}`);
  }

  async queryDatabase(
    databaseId: string,
    filter?: any,
    sorts?: any[],
  ): Promise<{ results: NotionPage[] }> {
    return this.request<{ results: NotionPage[] }>(
      "POST",
      `/databases/${databaseId}/query`,
      {
        filter,
        sorts,
      },
    );
  }

  async getPage(pageId: string): Promise<NotionPage> {
    return this.request<NotionPage>("GET", `/pages/${pageId}`);
  }

  async createPage(params: {
    databaseId: string;
    properties: Record<string, any>;
    children?: any[];
  }): Promise<NotionPage> {
    return this.request<NotionPage>("POST", "/pages", {
      parent: { database_id: params.databaseId },
      properties: params.properties,
      children: params.children,
    });
  }

  async updatePage(
    pageId: string,
    properties: Record<string, any>,
  ): Promise<NotionPage> {
    return this.request<NotionPage>("PATCH", `/pages/${pageId}`, {
      properties,
    });
  }

  async appendBlockChildren(pageId: string, children: any[]): Promise<void> {
    await this.request("PATCH", `/blocks/${pageId}/children`, { children });
  }

  async getPageContent(pageId: string): Promise<{ results: any[] }> {
    return this.request<{ results: any[] }>(
      "GET",
      `/blocks/${pageId}/children`,
    );
  }
}

// ============================================================================
// Notion Integration Service
// ============================================================================

export class NotionIntegration {
  private client: NotionClient;
  private config: NotionConfig;
  private mappings: Map<string, TaskToNotionMapping> = new Map();

  constructor(config: NotionConfig) {
    this.config = config;
    this.client = new NotionClient(config.apiKey);
  }

  /**
   * Initialize and verify database access
   */
  async initialize(): Promise<void> {
    try {
      await this.client.getDatabase(this.config.databaseId);
    } catch (error) {
      throw new Error(
        `Failed to access Notion database: ${error}. Make sure the integration has access to the database.`,
      );
    }
  }

  /**
   * Sync a task to Notion
   */
  async syncTaskToNotion(task: {
    id: string;
    description: string;
    status: string;
    priority: number;
    assigned_to?: string;
    context?: { hints?: string; files?: string[] };
    dependencies?: string[];
  }): Promise<TaskToNotionMapping> {
    // Check if already mapped
    const existing = this.mappings.get(task.id);

    const properties = this.buildTaskProperties(task);

    if (existing) {
      // Update existing page
      await this.client.updatePage(existing.notionPageId, properties);

      existing.syncedAt = new Date().toISOString();
      existing.lastSyncDirection = "toNotion";
      return existing;
    }

    // Create new page
    const children = this.buildTaskContent(task);
    const page = await this.client.createPage({
      databaseId: this.config.databaseId,
      properties,
      children,
    });

    const mapping: TaskToNotionMapping = {
      taskId: task.id,
      notionPageId: page.id,
      notionUrl: page.url,
      syncedAt: new Date().toISOString(),
      lastSyncDirection: "toNotion",
    };

    this.mappings.set(task.id, mapping);
    return mapping;
  }

  /**
   * Create a discovery page in Notion
   */
  async createDiscoveryPage(discovery: {
    id: string;
    title: string;
    content: string;
    tags?: string[];
    relatedTasks?: string[];
    agent?: string;
  }): Promise<NotionPage> {
    const databaseId =
      this.config.discoveriesDatabaseId || this.config.databaseId;

    const properties: Record<string, any> = {
      Name: {
        title: [{ text: { content: discovery.title } }],
      },
      Status: {
        select: { name: "Active" },
      },
      Type: {
        select: { name: "Discovery" },
      },
      "Discovery ID": {
        rich_text: [{ text: { content: discovery.id } }],
      },
    };

    if (discovery.agent) {
      properties["Agent"] = {
        rich_text: [{ text: { content: discovery.agent } }],
      };
    }

    if (discovery.tags && discovery.tags.length > 0) {
      properties["Tags"] = {
        multi_select: discovery.tags.map((tag) => ({ name: tag })),
      };
    }

    const children = [
      {
        object: "block",
        type: "heading_2",
        heading_2: {
          rich_text: [{ text: { content: "Discovery Details" } }],
        },
      },
      {
        object: "block",
        type: "paragraph",
        paragraph: {
          rich_text: [{ text: { content: discovery.content } }],
        },
      },
    ];

    if (discovery.relatedTasks && discovery.relatedTasks.length > 0) {
      children.push(
        {
          object: "block",
          type: "heading_3",
          heading_3: {
            rich_text: [{ text: { content: "Related Tasks" } }],
          },
        },
        {
          object: "block",
          type: "bulleted_list_item",
          bulleted_list_item: {
            rich_text: discovery.relatedTasks.map((taskId) => ({
              text: { content: taskId },
            })),
          },
        },
      );
    }

    return this.client.createPage({
      databaseId,
      properties,
      children,
    });
  }

  /**
   * Sync Notion page changes back to task
   */
  async syncNotionToTask(page: NotionPage): Promise<{
    id: string;
    description: string;
    status: string;
    priority: number;
  } | null> {
    // Extract task ID from properties
    const taskIdProp = page.properties["Task ID"];
    if (!taskIdProp || !taskIdProp.rich_text?.[0]?.plain_text) {
      return null;
    }

    const taskId = taskIdProp.rich_text[0].plain_text;

    // Extract other properties
    const nameProp = page.properties["Name"];
    const description = nameProp?.title?.[0]?.plain_text || "";

    const statusProp = page.properties["Status"];
    const notionStatus = statusProp?.select?.name || "To Do";
    const status = this.mapNotionStatusToTask(notionStatus);

    const priorityProp = page.properties["Priority"];
    const notionPriority = priorityProp?.select?.name || "Medium";
    const priority = this.mapNotionPriorityToTask(notionPriority);

    // Update mapping
    const mapping = this.mappings.get(taskId);
    if (mapping) {
      mapping.syncedAt = new Date().toISOString();
      mapping.lastSyncDirection = "fromNotion";
    }

    return {
      id: taskId,
      description,
      status,
      priority,
    };
  }

  /**
   * Query tasks from Notion database
   */
  async queryTasks(filter?: {
    status?: string;
    priority?: string;
    assignedTo?: string;
  }): Promise<NotionPage[]> {
    let notionFilter: any = undefined;

    if (filter) {
      const conditions: any[] = [];

      if (filter.status) {
        conditions.push({
          property: "Status",
          select: { equals: this.mapTaskStatusToNotion(filter.status) },
        });
      }

      if (filter.priority) {
        conditions.push({
          property: "Priority",
          select: { equals: this.mapTaskPriorityToNotion(filter.priority) },
        });
      }

      if (filter.assignedTo) {
        conditions.push({
          property: "Assigned To",
          rich_text: { contains: filter.assignedTo },
        });
      }

      if (conditions.length > 0) {
        notionFilter = { and: conditions };
      }
    }

    const result = await this.client.queryDatabase(
      this.config.databaseId,
      notionFilter,
      [{ property: "Created", direction: "descending" }],
    );

    return result.results;
  }

  /**
   * Get Notion page URL for a task
   */
  getPageUrl(taskId: string): string | null {
    const mapping = this.mappings.get(taskId);
    return mapping?.notionUrl || null;
  }

  /**
   * Get all mappings
   */
  getMappings(): TaskToNotionMapping[] {
    return Array.from(this.mappings.values());
  }

  /**
   * Build Notion properties for a task
   */
  private buildTaskProperties(task: {
    id: string;
    description: string;
    status: string;
    priority: number;
    assigned_to?: string;
    dependencies?: string[];
  }): Record<string, any> {
    const properties: Record<string, any> = {
      Name: {
        title: [{ text: { content: task.description.substring(0, 2000) } }],
      },
      Status: {
        select: { name: this.mapTaskStatusToNotion(task.status) },
      },
      Priority: {
        select: { name: this.mapTaskPriorityToNotion(task.priority) },
      },
      "Task ID": {
        rich_text: [{ text: { content: task.id } }],
      },
    };

    if (task.assigned_to) {
      properties["Assigned To"] = {
        rich_text: [{ text: { content: task.assigned_to } }],
      };
    }

    if (task.dependencies && task.dependencies.length > 0) {
      properties["Dependencies"] = {
        rich_text: [{ text: { content: task.dependencies.join(", ") } }],
      };
    }

    return properties;
  }

  /**
   * Build Notion content blocks for a task
   */
  private buildTaskContent(task: {
    id: string;
    description: string;
    context?: { hints?: string; files?: string[] };
  }): any[] {
    const blocks: any[] = [];

    // Add description
    blocks.push({
      object: "block",
      type: "heading_2",
      heading_2: {
        rich_text: [{ text: { content: "Task Description" } }],
      },
    });

    blocks.push({
      object: "block",
      type: "paragraph",
      paragraph: {
        rich_text: [{ text: { content: task.description } }],
      },
    });

    // Add hints if present
    if (task.context?.hints) {
      blocks.push({
        object: "block",
        type: "heading_3",
        heading_3: {
          rich_text: [{ text: { content: "Hints" } }],
        },
      });

      blocks.push({
        object: "block",
        type: "paragraph",
        paragraph: {
          rich_text: [{ text: { content: task.context.hints } }],
        },
      });
    }

    // Add files if present
    if (task.context?.files && task.context.files.length > 0) {
      blocks.push({
        object: "block",
        type: "heading_3",
        heading_3: {
          rich_text: [{ text: { content: "Related Files" } }],
        },
      });

      for (const file of task.context.files.slice(0, 10)) {
        blocks.push({
          object: "block",
          type: "bulleted_list_item",
          bulleted_list_item: {
            rich_text: [{ text: { content: file, code: true } }],
          },
        });
      }
    }

    return blocks;
  }

  private mapTaskStatusToNotion(status: string): string {
    switch (status) {
      case "available":
        return "To Do";
      case "claimed":
        return "In Progress";
      case "in_progress":
        return "In Progress";
      case "done":
        return "Done";
      case "failed":
        return "Blocked";
      default:
        return "To Do";
    }
  }

  private mapNotionStatusToTask(notionStatus: string): string {
    switch (notionStatus) {
      case "To Do":
        return "available";
      case "In Progress":
        return "in_progress";
      case "Done":
        return "done";
      case "Blocked":
        return "failed";
      default:
        return "available";
    }
  }

  private mapTaskPriorityToNotion(priority: number): string {
    switch (priority) {
      case 1:
        return "Urgent";
      case 2:
        return "High";
      case 3:
        return "Medium";
      case 4:
        return "Low";
      case 5:
        return "Low";
      default:
        return "Medium";
    }
  }

  private mapNotionPriorityToTask(notionPriority: string): number {
    switch (notionPriority) {
      case "Urgent":
        return 1;
      case "High":
        return 2;
      case "Medium":
        return 3;
      case "Low":
        return 4;
      default:
        return 3;
    }
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createNotionIntegration(
  config: NotionConfig,
): NotionIntegration {
  return new NotionIntegration(config);
}
