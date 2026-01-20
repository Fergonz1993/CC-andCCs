import * as core from "@actions/core";
import * as github from "@actions/github";
import { Octokit } from "@octokit/rest";
import * as fs from "fs";
import * as path from "path";

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
  github_issue?: number;
  github_pr?: number;
}

// GitHub Actions integration for Claude Coordinator
export class GitHubActionsIntegration {
  private octokit: Octokit;
  private owner: string;
  private repo: string;
  private coordinationDir: string;

  constructor(
    token: string,
    owner: string,
    repo: string,
    coordinationDir: string = ".coordination",
  ) {
    this.octokit = new Octokit({ auth: token });
    this.owner = owner;
    this.repo = repo;
    this.coordinationDir = coordinationDir;
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

  // Save tasks to coordination directory
  private saveTasks(tasks: Task[]): void {
    const tasksPath = path.join(this.coordinationDir, "tasks.json");
    const dir = path.dirname(tasksPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(
      tasksPath,
      JSON.stringify({ tasks, updated_at: new Date().toISOString() }, null, 2),
    );
  }

  // Create task from GitHub issue
  async createTaskFromIssue(issueNumber: number): Promise<Task> {
    const { data: issue } = await this.octokit.issues.get({
      owner: this.owner,
      repo: this.repo,
      issue_number: issueNumber,
    });

    const tasks = this.loadTasks();

    // Check if task already exists for this issue
    const existingTask = tasks.find((t) => t.github_issue === issueNumber);
    if (existingTask) {
      return existingTask;
    }

    // Determine priority from labels
    let priority = 3;
    for (const label of issue.labels) {
      const labelName = typeof label === "string" ? label : label.name || "";
      if (labelName.includes("priority-1") || labelName.includes("critical")) {
        priority = 1;
      } else if (
        labelName.includes("priority-2") ||
        labelName.includes("high")
      ) {
        priority = 2;
      } else if (
        labelName.includes("priority-4") ||
        labelName.includes("low")
      ) {
        priority = 4;
      } else if (
        labelName.includes("priority-5") ||
        labelName.includes("lowest")
      ) {
        priority = 5;
      }
    }

    const newTask: Task = {
      id: `gh-issue-${issueNumber}-${Date.now()}`,
      description: `[GH#${issueNumber}] ${issue.title}`,
      status: "available",
      priority,
      created_at: new Date().toISOString(),
      github_issue: issueNumber,
    };

    tasks.push(newTask);
    this.saveTasks(tasks);

    // Add comment to issue
    await this.octokit.issues.createComment({
      owner: this.owner,
      repo: this.repo,
      issue_number: issueNumber,
      body: `This issue has been added to the Claude Coordinator task queue.\n\nTask ID: \`${newTask.id}\`\nPriority: ${priority}`,
    });

    return newTask;
  }

  // Update issue when task status changes
  async syncTaskToIssue(taskId: string): Promise<void> {
    const tasks = this.loadTasks();
    const task = tasks.find((t) => t.id === taskId);

    if (!task || !task.github_issue) {
      return;
    }

    const statusLabels: Record<string, string> = {
      available: "coord:available",
      claimed: "coord:claimed",
      in_progress: "coord:in-progress",
      done: "coord:done",
      failed: "coord:failed",
    };

    // Get current labels
    const { data: issue } = await this.octokit.issues.get({
      owner: this.owner,
      repo: this.repo,
      issue_number: task.github_issue,
    });

    // Remove old coordination labels
    const labels = issue.labels
      .map((l) => (typeof l === "string" ? l : l.name || ""))
      .filter((l) => !l.startsWith("coord:"));

    // Add new status label
    labels.push(statusLabels[task.status]);

    await this.octokit.issues.update({
      owner: this.owner,
      repo: this.repo,
      issue_number: task.github_issue,
      labels,
      state: task.status === "done" ? "closed" : "open",
    });

    // Add comment for status updates
    if (task.status === "done" && task.result) {
      await this.octokit.issues.createComment({
        owner: this.owner,
        repo: this.repo,
        issue_number: task.github_issue,
        body: `Task completed by Claude Coordinator.\n\n**Result:**\n${task.result}`,
      });
    } else if (task.status === "failed" && task.error) {
      await this.octokit.issues.createComment({
        owner: this.owner,
        repo: this.repo,
        issue_number: task.github_issue,
        body: `Task failed in Claude Coordinator.\n\n**Error:**\n${task.error}`,
      });
    }
  }

  // Link PR to task
  async linkPRToTask(taskId: string, prNumber: number): Promise<void> {
    const tasks = this.loadTasks();
    const task = tasks.find((t) => t.id === taskId);

    if (!task) {
      throw new Error(`Task not found: ${taskId}`);
    }

    task.github_pr = prNumber;
    task.updated_at = new Date().toISOString();
    this.saveTasks(tasks);

    // Add comment to PR
    await this.octokit.issues.createComment({
      owner: this.owner,
      repo: this.repo,
      issue_number: prNumber,
      body: `This PR is linked to Claude Coordinator task: \`${taskId}\`\n\n**Task:** ${task.description}`,
    });

    // If task has associated issue, link them
    if (task.github_issue) {
      await this.octokit.issues.createComment({
        owner: this.owner,
        repo: this.repo,
        issue_number: task.github_issue,
        body: `PR #${prNumber} has been linked to this task.`,
      });
    }
  }

  // Create status check for coordination
  async createStatusCheck(
    sha: string,
    status: "queued" | "in_progress" | "completed",
    conclusion?: "success" | "failure" | "neutral",
  ): Promise<void> {
    await this.octokit.checks.create({
      owner: this.owner,
      repo: this.repo,
      name: "Claude Coordinator",
      head_sha: sha,
      status,
      conclusion,
      output: {
        title: "Coordination Status",
        summary: this.getStatusSummary(),
      },
    });
  }

  // Get summary of current coordination status
  private getStatusSummary(): string {
    const tasks = this.loadTasks();
    const statusCounts = {
      available: tasks.filter((t) => t.status === "available").length,
      claimed: tasks.filter((t) => t.status === "claimed").length,
      in_progress: tasks.filter((t) => t.status === "in_progress").length,
      done: tasks.filter((t) => t.status === "done").length,
      failed: tasks.filter((t) => t.status === "failed").length,
    };

    return `## Coordination Status

| Status | Count |
|--------|-------|
| Available | ${statusCounts.available} |
| Claimed | ${statusCounts.claimed} |
| In Progress | ${statusCounts.in_progress} |
| Done | ${statusCounts.done} |
| Failed | ${statusCounts.failed} |
| **Total** | **${tasks.length}** |
`;
  }

  // Sync all tasks with GitHub issues
  async syncAllTasks(): Promise<void> {
    const tasks = this.loadTasks();

    for (const task of tasks) {
      if (task.github_issue) {
        await this.syncTaskToIssue(task.id);
      }
    }
  }

  // Import all open issues as tasks
  async importOpenIssues(label?: string): Promise<Task[]> {
    const query: any = {
      owner: this.owner,
      repo: this.repo,
      state: "open",
      per_page: 100,
    };

    if (label) {
      query.labels = label;
    }

    const { data: issues } = await this.octokit.issues.listForRepo(query);

    const newTasks: Task[] = [];
    for (const issue of issues) {
      // Skip pull requests
      if (issue.pull_request) {
        continue;
      }

      const task = await this.createTaskFromIssue(issue.number);
      newTasks.push(task);
    }

    return newTasks;
  }

  // Create GitHub issue from task
  async createIssueFromTask(taskId: string): Promise<number> {
    const tasks = this.loadTasks();
    const task = tasks.find((t) => t.id === taskId);

    if (!task) {
      throw new Error(`Task not found: ${taskId}`);
    }

    if (task.github_issue) {
      return task.github_issue;
    }

    const { data: issue } = await this.octokit.issues.create({
      owner: this.owner,
      repo: this.repo,
      title: task.description,
      body: `Created from Claude Coordinator task.\n\n**Task ID:** \`${task.id}\`\n**Priority:** ${task.priority}\n**Status:** ${task.status}`,
      labels: [`priority-${task.priority}`, `coord:${task.status}`],
    });

    task.github_issue = issue.number;
    task.updated_at = new Date().toISOString();
    this.saveTasks(tasks);

    return issue.number;
  }
}

// GitHub Action entry point
async function run(): Promise<void> {
  try {
    const token = core.getInput("github-token", { required: true });
    const action = core.getInput("action", { required: true });
    const coordinationDir =
      core.getInput("coordination-dir") || ".coordination";

    const context = github.context;
    const integration = new GitHubActionsIntegration(
      token,
      context.repo.owner,
      context.repo.repo,
      coordinationDir,
    );

    switch (action) {
      case "import-issues": {
        const label = core.getInput("label");
        const tasks = await integration.importOpenIssues(label || undefined);
        core.setOutput("tasks-created", tasks.length);
        core.info(`Imported ${tasks.length} issues as tasks`);
        break;
      }

      case "sync-tasks": {
        await integration.syncAllTasks();
        core.info("Synced all tasks with GitHub issues");
        break;
      }

      case "create-task-from-issue": {
        const issueNumber = parseInt(
          core.getInput("issue-number", { required: true }),
        );
        const task = await integration.createTaskFromIssue(issueNumber);
        core.setOutput("task-id", task.id);
        core.info(`Created task ${task.id} from issue #${issueNumber}`);
        break;
      }

      case "link-pr": {
        const taskId = core.getInput("task-id", { required: true });
        const prNumber = parseInt(
          core.getInput("pr-number", { required: true }),
        );
        await integration.linkPRToTask(taskId, prNumber);
        core.info(`Linked PR #${prNumber} to task ${taskId}`);
        break;
      }

      case "status-check": {
        const sha = core.getInput("sha") || context.sha;
        const status = core.getInput("status") as
          | "queued"
          | "in_progress"
          | "completed";
        const conclusion = core.getInput("conclusion") as
          | "success"
          | "failure"
          | "neutral"
          | undefined;
        await integration.createStatusCheck(sha, status, conclusion);
        core.info("Created status check");
        break;
      }

      default:
        core.setFailed(`Unknown action: ${action}`);
    }
  } catch (error) {
    core.setFailed(`Action failed: ${error}`);
  }
}

// Run if executed directly
if (require.main === module) {
  run();
}

export { GitHubActionsIntegration, run };
