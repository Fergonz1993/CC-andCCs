/**
 * Linear Integration (ext-005)
 *
 * Integrates Claude Multi-Agent Coordination with Linear for issue tracking.
 * Syncs tasks bidirectionally between the coordination system and Linear.
 */

import * as crypto from "crypto";

// ============================================================================
// Types
// ============================================================================

export interface LinearConfig {
  apiKey: string;
  teamId: string;
  projectId?: string;
  syncEnabled?: boolean;
  autoCreateIssues?: boolean;
  webhookSecret?: string;
}

export interface LinearIssue {
  id: string;
  identifier: string;
  title: string;
  description?: string;
  priority: number; // 0 = no priority, 1 = urgent, 2 = high, 3 = medium, 4 = low
  state: {
    id: string;
    name: string;
    type: "backlog" | "unstarted" | "started" | "completed" | "canceled";
  };
  assignee?: {
    id: string;
    name: string;
    email: string;
  };
  labels: Array<{ id: string; name: string; color: string }>;
  createdAt: string;
  updatedAt: string;
  url: string;
}

export interface LinearWebhookPayload {
  action: "create" | "update" | "remove";
  type: "Issue" | "Comment" | "Project";
  data: any;
  createdAt: string;
  url: string;
}

export interface TaskToLinearMapping {
  taskId: string;
  linearIssueId: string;
  linearIdentifier: string;
  syncedAt: string;
  lastSyncDirection: "toLinear" | "fromLinear";
}

// ============================================================================
// Linear API Client
// ============================================================================

export class LinearClient {
  private apiKey: string;
  private baseUrl = "https://api.linear.app/graphql";

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  private async graphql<T>(
    query: string,
    variables?: Record<string, any>,
  ): Promise<T> {
    const response = await fetch(this.baseUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: this.apiKey,
      },
      body: JSON.stringify({ query, variables }),
    });

    if (!response.ok) {
      throw new Error(
        `Linear API error: ${response.status} ${response.statusText}`,
      );
    }

    const result = await response.json();
    if (result.errors) {
      throw new Error(`GraphQL error: ${result.errors[0].message}`);
    }

    return result.data;
  }

  async getTeams(): Promise<Array<{ id: string; name: string; key: string }>> {
    const query = `
      query {
        teams {
          nodes {
            id
            name
            key
          }
        }
      }
    `;
    const data = await this.graphql<{ teams: { nodes: any[] } }>(query);
    return data.teams.nodes;
  }

  async getIssue(issueId: string): Promise<LinearIssue | null> {
    const query = `
      query GetIssue($id: String!) {
        issue(id: $id) {
          id
          identifier
          title
          description
          priority
          state {
            id
            name
            type
          }
          assignee {
            id
            name
            email
          }
          labels {
            nodes {
              id
              name
              color
            }
          }
          createdAt
          updatedAt
          url
        }
      }
    `;

    try {
      const data = await this.graphql<{ issue: any }>(query, { id: issueId });
      if (!data.issue) return null;

      return {
        ...data.issue,
        labels: data.issue.labels?.nodes || [],
      };
    } catch {
      return null;
    }
  }

  async createIssue(params: {
    teamId: string;
    title: string;
    description?: string;
    priority?: number;
    projectId?: string;
    labelIds?: string[];
  }): Promise<LinearIssue> {
    const mutation = `
      mutation CreateIssue($input: IssueCreateInput!) {
        issueCreate(input: $input) {
          success
          issue {
            id
            identifier
            title
            description
            priority
            state {
              id
              name
              type
            }
            createdAt
            updatedAt
            url
          }
        }
      }
    `;

    const data = await this.graphql<{
      issueCreate: { success: boolean; issue: any };
    }>(mutation, {
      input: {
        teamId: params.teamId,
        title: params.title,
        description: params.description,
        priority: params.priority || 0,
        projectId: params.projectId,
        labelIds: params.labelIds,
      },
    });

    if (!data.issueCreate.success) {
      throw new Error("Failed to create Linear issue");
    }

    return {
      ...data.issueCreate.issue,
      labels: [],
    };
  }

  async updateIssue(
    issueId: string,
    params: {
      title?: string;
      description?: string;
      priority?: number;
      stateId?: string;
      assigneeId?: string;
    },
  ): Promise<LinearIssue> {
    const mutation = `
      mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
        issueUpdate(id: $id, input: $input) {
          success
          issue {
            id
            identifier
            title
            description
            priority
            state {
              id
              name
              type
            }
            assignee {
              id
              name
              email
            }
            updatedAt
            url
          }
        }
      }
    `;

    const data = await this.graphql<{
      issueUpdate: { success: boolean; issue: any };
    }>(mutation, { id: issueId, input: params });

    return {
      ...data.issueUpdate.issue,
      labels: [],
    };
  }

  async getWorkflowStates(
    teamId: string,
  ): Promise<Array<{ id: string; name: string; type: string }>> {
    const query = `
      query GetStates($teamId: String!) {
        team(id: $teamId) {
          states {
            nodes {
              id
              name
              type
            }
          }
        }
      }
    `;

    const data = await this.graphql<{ team: { states: { nodes: any[] } } }>(
      query,
      {
        teamId,
      },
    );
    return data.team.states.nodes;
  }
}

// ============================================================================
// Linear Integration Service
// ============================================================================

export class LinearIntegration {
  private client: LinearClient;
  private config: LinearConfig;
  private mappings: Map<string, TaskToLinearMapping> = new Map();
  private stateMapping: Map<string, string> = new Map(); // task status -> linear state id

  constructor(config: LinearConfig) {
    this.config = config;
    this.client = new LinearClient(config.apiKey);
  }

  /**
   * Initialize the integration, loading workflow states
   */
  async initialize(): Promise<void> {
    const states = await this.client.getWorkflowStates(this.config.teamId);

    // Map task statuses to Linear states
    for (const state of states) {
      switch (state.type) {
        case "backlog":
          this.stateMapping.set("available", state.id);
          break;
        case "unstarted":
          this.stateMapping.set("claimed", state.id);
          break;
        case "started":
          this.stateMapping.set("in_progress", state.id);
          break;
        case "completed":
          this.stateMapping.set("done", state.id);
          break;
        case "canceled":
          this.stateMapping.set("failed", state.id);
          break;
      }
    }
  }

  /**
   * Sync a task to Linear
   */
  async syncTaskToLinear(task: {
    id: string;
    description: string;
    status: string;
    priority: number;
    context?: { hints?: string };
  }): Promise<TaskToLinearMapping> {
    // Check if already mapped
    const existing = this.mappings.get(task.id);

    if (existing) {
      // Update existing issue
      const stateId = this.stateMapping.get(task.status);
      await this.client.updateIssue(existing.linearIssueId, {
        title: task.description.substring(0, 100),
        description: this.formatTaskDescription(task),
        priority: this.mapPriority(task.priority),
        stateId,
      });

      existing.syncedAt = new Date().toISOString();
      existing.lastSyncDirection = "toLinear";
      return existing;
    }

    // Create new issue
    const issue = await this.client.createIssue({
      teamId: this.config.teamId,
      title: task.description.substring(0, 100),
      description: this.formatTaskDescription(task),
      priority: this.mapPriority(task.priority),
      projectId: this.config.projectId,
    });

    const mapping: TaskToLinearMapping = {
      taskId: task.id,
      linearIssueId: issue.id,
      linearIdentifier: issue.identifier,
      syncedAt: new Date().toISOString(),
      lastSyncDirection: "toLinear",
    };

    this.mappings.set(task.id, mapping);
    return mapping;
  }

  /**
   * Sync a Linear issue to a task
   */
  async syncLinearToTask(issue: LinearIssue): Promise<{
    id: string;
    description: string;
    status: string;
    priority: number;
  }> {
    const status = this.mapLinearStateToStatus(issue.state.type);
    const priority = this.mapLinearPriorityToTask(issue.priority);

    return {
      id: issue.id,
      description: issue.title,
      status,
      priority,
    };
  }

  /**
   * Handle webhook from Linear
   */
  async handleWebhook(
    payload: LinearWebhookPayload,
    signature?: string,
  ): Promise<{ processed: boolean; taskUpdate?: any }> {
    // Verify signature if webhook secret is configured
    if (this.config.webhookSecret && signature) {
      const expectedSignature = crypto
        .createHmac("sha256", this.config.webhookSecret)
        .update(JSON.stringify(payload))
        .digest("hex");

      if (signature !== expectedSignature) {
        return { processed: false };
      }
    }

    if (payload.type !== "Issue") {
      return { processed: false };
    }

    // Find mapping for this issue
    let mapping: TaskToLinearMapping | undefined;
    for (const m of this.mappings.values()) {
      if (m.linearIssueId === payload.data.id) {
        mapping = m;
        break;
      }
    }

    if (!mapping) {
      return { processed: false };
    }

    // Convert Linear update to task update
    const taskUpdate = {
      id: mapping.taskId,
      status: this.mapLinearStateToStatus(
        payload.data.state?.type || "backlog",
      ),
      priority: this.mapLinearPriorityToTask(payload.data.priority || 0),
      updatedFromLinear: true,
    };

    mapping.syncedAt = new Date().toISOString();
    mapping.lastSyncDirection = "fromLinear";

    return { processed: true, taskUpdate };
  }

  /**
   * Get Linear issue URL for a task
   */
  getIssueUrl(taskId: string): string | null {
    const mapping = this.mappings.get(taskId);
    if (!mapping) return null;
    return `https://linear.app/issue/${mapping.linearIdentifier}`;
  }

  /**
   * Get all mappings
   */
  getMappings(): TaskToLinearMapping[] {
    return Array.from(this.mappings.values());
  }

  private formatTaskDescription(task: {
    id: string;
    description: string;
    context?: { hints?: string };
  }): string {
    let desc = task.description;

    if (task.context?.hints) {
      desc += `\n\n**Hints:**\n${task.context.hints}`;
    }

    desc += `\n\n---\n*Synced from Claude Multi-Agent Coordination*\nTask ID: \`${task.id}\``;

    return desc;
  }

  private mapPriority(taskPriority: number): number {
    // Task priority: 1 (highest) to 5 (lowest)
    // Linear priority: 0 (no priority), 1 (urgent), 2 (high), 3 (medium), 4 (low)
    switch (taskPriority) {
      case 1:
        return 1; // Urgent
      case 2:
        return 2; // High
      case 3:
        return 3; // Medium
      case 4:
      case 5:
        return 4; // Low
      default:
        return 0; // No priority
    }
  }

  private mapLinearPriorityToTask(linearPriority: number): number {
    switch (linearPriority) {
      case 1:
        return 1; // Urgent -> 1
      case 2:
        return 2; // High -> 2
      case 3:
        return 3; // Medium -> 3
      case 4:
        return 4; // Low -> 4
      default:
        return 5; // No priority -> 5
    }
  }

  private mapLinearStateToStatus(stateType: string): string {
    switch (stateType) {
      case "backlog":
        return "available";
      case "unstarted":
        return "claimed";
      case "started":
        return "in_progress";
      case "completed":
        return "done";
      case "canceled":
        return "failed";
      default:
        return "available";
    }
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createLinearIntegration(
  config: LinearConfig,
): LinearIntegration {
  return new LinearIntegration(config);
}
