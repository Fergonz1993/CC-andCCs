import * as vscode from "vscode";
import { CoordinationService, Worker } from "./coordinationService";

export class WorkerItem extends vscode.TreeItem {
  constructor(
    public readonly worker: Worker,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
  ) {
    super(worker.id, collapsibleState);

    this.id = worker.id;
    this.tooltip = `Worker: ${worker.id}\nStatus: ${worker.status}\nTasks Completed: ${worker.tasks_completed || 0}`;
    this.description = worker.status;

    switch (worker.status) {
      case "idle":
        this.iconPath = new vscode.ThemeIcon(
          "circle-outline",
          new vscode.ThemeColor("charts.green"),
        );
        break;
      case "busy":
        this.iconPath = new vscode.ThemeIcon(
          "sync~spin",
          new vscode.ThemeColor("charts.blue"),
        );
        this.description = `busy - ${worker.current_task || "unknown task"}`;
        break;
      case "offline":
        this.iconPath = new vscode.ThemeIcon(
          "circle-slash",
          new vscode.ThemeColor("charts.gray"),
        );
        break;
    }
  }
}

export class WorkerProvider implements vscode.TreeDataProvider<WorkerItem> {
  private _onDidChangeTreeData: vscode.EventEmitter<
    WorkerItem | undefined | null | void
  > = new vscode.EventEmitter<WorkerItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<
    WorkerItem | undefined | null | void
  > = this._onDidChangeTreeData.event;

  constructor(private coordinationService: CoordinationService) {}

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: WorkerItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: WorkerItem): Promise<WorkerItem[]> {
    if (element) {
      return [];
    }

    const workers = await this.coordinationService.getWorkers();

    // Sort by status (busy first, then idle, then offline)
    const statusOrder = { busy: 0, idle: 1, offline: 2 };
    workers.sort((a, b) => statusOrder[a.status] - statusOrder[b.status]);

    return workers.map(
      (worker) => new WorkerItem(worker, vscode.TreeItemCollapsibleState.None),
    );
  }
}
