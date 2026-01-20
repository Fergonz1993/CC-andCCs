import * as vscode from "vscode";
import { CoordinationService, Task } from "./coordinationService";

export class TaskItem extends vscode.TreeItem {
  constructor(
    public readonly task: Task,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
  ) {
    super(task.description, collapsibleState);

    this.id = task.id;
    this.tooltip = `${task.description}\nStatus: ${task.status}\nPriority: ${task.priority}`;
    this.description = `[P${task.priority}] ${task.status}`;

    // Set context value for menu filtering
    switch (task.status) {
      case "available":
        this.contextValue = "availableTask";
        this.iconPath = new vscode.ThemeIcon(
          "circle-outline",
          new vscode.ThemeColor("charts.green"),
        );
        break;
      case "claimed":
        this.contextValue = "claimedTask";
        this.iconPath = new vscode.ThemeIcon(
          "circle-filled",
          new vscode.ThemeColor("charts.orange"),
        );
        break;
      case "in_progress":
        this.contextValue = "inProgressTask";
        this.iconPath = new vscode.ThemeIcon(
          "sync~spin",
          new vscode.ThemeColor("charts.blue"),
        );
        break;
      case "done":
        this.contextValue = "doneTask";
        this.iconPath = new vscode.ThemeIcon(
          "check",
          new vscode.ThemeColor("charts.gray"),
        );
        break;
      case "failed":
        this.contextValue = "failedTask";
        this.iconPath = new vscode.ThemeIcon(
          "error",
          new vscode.ThemeColor("charts.red"),
        );
        break;
    }

    this.command = {
      command: "claude-coordinator.viewTaskDetails",
      title: "View Task Details",
      arguments: [this],
    };
  }
}

export class TaskProvider implements vscode.TreeDataProvider<TaskItem> {
  private _onDidChangeTreeData: vscode.EventEmitter<
    TaskItem | undefined | null | void
  > = new vscode.EventEmitter<TaskItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<
    TaskItem | undefined | null | void
  > = this._onDidChangeTreeData.event;

  constructor(private coordinationService: CoordinationService) {}

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: TaskItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: TaskItem): Promise<TaskItem[]> {
    if (element) {
      return [];
    }

    const tasks = await this.coordinationService.getTasks();

    // Sort by priority (lowest number = highest priority), then by status
    const statusOrder = {
      in_progress: 0,
      claimed: 1,
      available: 2,
      failed: 3,
      done: 4,
    };

    tasks.sort((a, b) => {
      const statusDiff = statusOrder[a.status] - statusOrder[b.status];
      if (statusDiff !== 0) return statusDiff;
      return a.priority - b.priority;
    });

    return tasks.map(
      (task) => new TaskItem(task, vscode.TreeItemCollapsibleState.None),
    );
  }
}
