/**
 * Jira Integration (ext-006)
 *
 * Integrates Claude Multi-Agent Coordination with Jira for issue tracking.
 * Syncs tasks bidirectionally between the coordination system and Jira.
 */

// ============================================================================
// Types
// ============================================================================

export interface JiraConfig {
  host: string; // e.g., "yourcompany.atlassian.net"
  email: string;
  apiToken: string;
  projectKey: string;
  syncEnabled?: boolean;
  defaultIssueType?: string;
  customFieldMappings?: Record<string, string>;
}

export interface JiraIssue {
  id: string;
  key: string;
  self: string;
  fields: {
    summary: string;
    description?: string;
    status: {
      id: string;
      name: string;
      statusCategory: {
        key: "new" | "indeterminate" | "done";
      };
    };
    priority?: {
      id: string;
      name: string;
    };
    assignee?: {
      accountId: string;
      displayName: string;
      emailAddress: string;
    };
    issuetype: {
      id: string;
      name: string;
    };
    created: string;
    updated: string;
    labels: string[];
  };
}

export interface JiraWebhookPayload {
  webhookEvent: string;
  issue_event_type_name?: string;
  issue: JiraIssue;
  user: {
    accountId: string;
    displayName: string;
  };
  changelog?: {
    items: Array<{
      field: string;
      fromString: string;
      toString: string;
    }>;
  };
}

export interface TaskToJiraMapping {
  taskId: string;
  jiraIssueId: string;
  jiraKey: string;
  syncedAt: string;
  lastSyncDirection: "toJira" | "fromJira";
}

// ============================================================================
// Jira API Client
// ============================================================================

export class JiraClient {
  private baseUrl: string;
  private authHeader: string;

  constructor(host: string, email: string, apiToken: string) {
    this.baseUrl = `https://${host}/rest/api/3`;
    this.authHeader = `Basic ${Buffer.from(`${email}:${apiToken}`).toString("base64")}`;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: any,
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: {
        Authorization: this.authHeader,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Jira API error: ${response.status} - ${error}`);
    }

    if (response.status === 204) {
      return {} as T;
    }

    return response.json();
  }

  async getIssue(issueIdOrKey: string): Promise<JiraIssue | null> {
    try {
      return await this.request<JiraIssue>("GET", `/issue/${issueIdOrKey}`);
    } catch {
      return null;
    }
  }

  async createIssue(params: {
    projectKey: string;
    summary: string;
    description?: string;
    issueType?: string;
    priority?: string;
    labels?: string[];
  }): Promise<JiraIssue> {
    const result = await this.request<{ id: string; key: string }>(
      "POST",
      "/issue",
      {
        fields: {
          project: { key: params.projectKey },
          summary: params.summary,
          description: params.description
            ? {
                type: "doc",
                version: 1,
                content: [
                  {
                    type: "paragraph",
                    content: [{ type: "text", text: params.description }],
                  },
                ],
              }
            : undefined,
          issuetype: { name: params.issueType || "Task" },
          priority: params.priority ? { name: params.priority } : undefined,
          labels: params.labels,
        },
      },
    );

    // Fetch the full issue
    return (await this.getIssue(result.key))!;
  }

  async updateIssue(
    issueIdOrKey: string,
    params: {
      summary?: string;
      description?: string;
      priority?: string;
      labels?: string[];
    },
  ): Promise<void> {
    const fields: any = {};

    if (params.summary) {
      fields.summary = params.summary;
    }

    if (params.description) {
      fields.description = {
        type: "doc",
        version: 1,
        content: [
          {
            type: "paragraph",
            content: [{ type: "text", text: params.description }],
          },
        ],
      };
    }

    if (params.priority) {
      fields.priority = { name: params.priority };
    }

    if (params.labels) {
      fields.labels = params.labels;
    }

    await this.request("PUT", `/issue/${issueIdOrKey}`, { fields });
  }

  async transitionIssue(
    issueIdOrKey: string,
    transitionId: string,
  ): Promise<void> {
    await this.request("POST", `/issue/${issueIdOrKey}/transitions`, {
      transition: { id: transitionId },
    });
  }

  async getTransitions(
    issueIdOrKey: string,
  ): Promise<Array<{ id: string; name: string; to: { name: string } }>> {
    const result = await this.request<{
      transitions: Array<{ id: string; name: string; to: { name: string } }>;
    }>("GET", `/issue/${issueIdOrKey}/transitions`);
    return result.transitions;
  }

  async getProjects(): Promise<
    Array<{ id: string; key: string; name: string }>
  > {
    return this.request<Array<{ id: string; key: string; name: string }>>(
      "GET",
      "/project",
    );
  }

  async getIssueTypes(
    projectKey: string,
  ): Promise<Array<{ id: string; name: string; subtask: boolean }>> {
    const result = await this.request<{
      issueTypes: Array<{ id: string; name: string; subtask: boolean }>;
    }>("GET", `/project/${projectKey}`);
    return result.issueTypes;
  }

  async searchIssues(jql: string, maxResults = 50): Promise<JiraIssue[]> {
    const result = await this.request<{ issues: JiraIssue[] }>(
      "POST",
      "/search",
      {
        jql,
        maxResults,
        fields: [
          "summary",
          "description",
          "status",
          "priority",
          "assignee",
          "issuetype",
          "created",
          "updated",
          "labels",
        ],
      },
    );
    return result.issues;
  }
}

// ============================================================================
// Jira Integration Service
// ============================================================================

export class JiraIntegration {
  private client: JiraClient;
  private config: JiraConfig;
  private mappings: Map<string, TaskToJiraMapping> = new Map();
  private transitionCache: Map<string, Array<{ id: string; name: string }>> =
    new Map();

  constructor(config: JiraConfig) {
    this.config = config;
    this.client = new JiraClient(config.host, config.email, config.apiToken);
  }

  /**
   * Sync a task to Jira
   */
  async syncTaskToJira(task: {
    id: string;
    description: string;
    status: string;
    priority: number;
    context?: { hints?: string };
  }): Promise<TaskToJiraMapping> {
    // Check if already mapped
    const existing = this.mappings.get(task.id);

    if (existing) {
      // Update existing issue
      await this.client.updateIssue(existing.jiraKey, {
        summary: task.description.substring(0, 255),
        description: this.formatTaskDescription(task),
        priority: this.mapPriority(task.priority),
        labels: ["claude-coordination"],
      });

      // Update status via transition if needed
      await this.updateIssueStatus(existing.jiraKey, task.status);

      existing.syncedAt = new Date().toISOString();
      existing.lastSyncDirection = "toJira";
      return existing;
    }

    // Create new issue
    const issue = await this.client.createIssue({
      projectKey: this.config.projectKey,
      summary: task.description.substring(0, 255),
      description: this.formatTaskDescription(task),
      issueType: this.config.defaultIssueType || "Task",
      priority: this.mapPriority(task.priority),
      labels: ["claude-coordination"],
    });

    const mapping: TaskToJiraMapping = {
      taskId: task.id,
      jiraIssueId: issue.id,
      jiraKey: issue.key,
      syncedAt: new Date().toISOString(),
      lastSyncDirection: "toJira",
    };

    this.mappings.set(task.id, mapping);
    return mapping;
  }

  /**
   * Sync a Jira issue to a task
   */
  async syncJiraToTask(issue: JiraIssue): Promise<{
    id: string;
    description: string;
    status: string;
    priority: number;
  }> {
    const status = this.mapJiraStatusToTask(
      issue.fields.status.statusCategory.key,
    );
    const priority = this.mapJiraPriorityToTask(
      issue.fields.priority?.name || "Medium",
    );

    return {
      id: issue.key,
      description: issue.fields.summary,
      status,
      priority,
    };
  }

  /**
   * Handle webhook from Jira
   */
  async handleWebhook(payload: JiraWebhookPayload): Promise<{
    processed: boolean;
    taskUpdate?: any;
  }> {
    const issueKey = payload.issue.key;

    // Find mapping for this issue
    let mapping: TaskToJiraMapping | undefined;
    for (const m of this.mappings.values()) {
      if (m.jiraKey === issueKey) {
        mapping = m;
        break;
      }
    }

    if (!mapping) {
      return { processed: false };
    }

    // Check if this is a status change
    const statusChange = payload.changelog?.items.find(
      (i) => i.field === "status",
    );

    const taskUpdate = {
      id: mapping.taskId,
      status: this.mapJiraStatusToTask(
        payload.issue.fields.status.statusCategory.key,
      ),
      priority: this.mapJiraPriorityToTask(
        payload.issue.fields.priority?.name || "Medium",
      ),
      statusChanged: !!statusChange,
      updatedFromJira: true,
    };

    mapping.syncedAt = new Date().toISOString();
    mapping.lastSyncDirection = "fromJira";

    return { processed: true, taskUpdate };
  }

  /**
   * Update issue status via transition
   */
  private async updateIssueStatus(
    issueKey: string,
    targetStatus: string,
  ): Promise<void> {
    // Get available transitions
    let transitions = this.transitionCache.get(issueKey);
    if (!transitions) {
      const fetched = await this.client.getTransitions(issueKey);
      transitions = fetched.map((t) => ({ id: t.id, name: t.to.name }));
      this.transitionCache.set(issueKey, transitions);
    }

    // Map target status to Jira status name
    const targetJiraStatus = this.mapTaskStatusToJira(targetStatus);

    // Find matching transition
    const transition = transitions.find(
      (t) => t.name.toLowerCase() === targetJiraStatus.toLowerCase(),
    );

    if (transition) {
      await this.client.transitionIssue(issueKey, transition.id);
      // Clear cache after transition
      this.transitionCache.delete(issueKey);
    }
  }

  /**
   * Get Jira issue URL for a task
   */
  getIssueUrl(taskId: string): string | null {
    const mapping = this.mappings.get(taskId);
    if (!mapping) return null;
    return `https://${this.config.host}/browse/${mapping.jiraKey}`;
  }

  /**
   * Get all mappings
   */
  getMappings(): TaskToJiraMapping[] {
    return Array.from(this.mappings.values());
  }

  /**
   * Search for related Jira issues
   */
  async searchRelatedIssues(query: string): Promise<JiraIssue[]> {
    const jql = `project = ${this.config.projectKey} AND (summary ~ "${query}" OR description ~ "${query}") ORDER BY updated DESC`;
    return this.client.searchIssues(jql, 10);
  }

  private formatTaskDescription(task: {
    id: string;
    description: string;
    context?: { hints?: string };
  }): string {
    let desc = task.description;

    if (task.context?.hints) {
      desc += `\n\nHints:\n${task.context.hints}`;
    }

    desc += `\n\n---\nSynced from Claude Multi-Agent Coordination\nTask ID: ${task.id}`;

    return desc;
  }

  private mapPriority(taskPriority: number): string {
    // Task priority: 1 (highest) to 5 (lowest)
    // Jira priority: Highest, High, Medium, Low, Lowest
    switch (taskPriority) {
      case 1:
        return "Highest";
      case 2:
        return "High";
      case 3:
        return "Medium";
      case 4:
        return "Low";
      case 5:
        return "Lowest";
      default:
        return "Medium";
    }
  }

  private mapJiraPriorityToTask(jiraPriority: string): number {
    switch (jiraPriority.toLowerCase()) {
      case "highest":
        return 1;
      case "high":
        return 2;
      case "medium":
        return 3;
      case "low":
        return 4;
      case "lowest":
        return 5;
      default:
        return 3;
    }
  }

  private mapJiraStatusToTask(statusCategory: string): string {
    switch (statusCategory) {
      case "new":
        return "available";
      case "indeterminate":
        return "in_progress";
      case "done":
        return "done";
      default:
        return "available";
    }
  }

  private mapTaskStatusToJira(taskStatus: string): string {
    switch (taskStatus) {
      case "available":
        return "To Do";
      case "claimed":
        return "To Do";
      case "in_progress":
        return "In Progress";
      case "done":
        return "Done";
      case "failed":
        return "Done";
      default:
        return "To Do";
    }
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createJiraIntegration(config: JiraConfig): JiraIntegration {
  return new JiraIntegration(config);
}
