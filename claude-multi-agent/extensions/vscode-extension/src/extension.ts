import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
import { TaskProvider, TaskItem } from "./taskProvider";
import { WorkerProvider } from "./workerProvider";
import { DiscoveryProvider } from "./discoveryProvider";
import { CoordinationService } from "./coordinationService";

let coordinationService: CoordinationService;
let taskProvider: TaskProvider;
let workerProvider: WorkerProvider;
let discoveryProvider: DiscoveryProvider;
let refreshInterval: NodeJS.Timeout | undefined;

export function activate(context: vscode.ExtensionContext) {
  console.log("Claude Multi-Agent Coordinator extension is now active");

  const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
  if (!workspaceFolder) {
    vscode.window.showWarningMessage(
      "No workspace folder open. Claude Coordinator features limited.",
    );
    return;
  }

  const config = vscode.workspace.getConfiguration("claude-coordinator");
  const coordinationDir = config.get<string>(
    "coordinationDir",
    ".coordination",
  );
  const agentId = config.get<string>("agentId", "vscode-agent");
  const autoRefresh = config.get<boolean>("autoRefresh", true);
  const refreshIntervalMs = config.get<number>("refreshInterval", 5000);

  const coordPath = path.join(workspaceFolder.uri.fsPath, coordinationDir);

  // Initialize coordination service
  coordinationService = new CoordinationService(coordPath, agentId);

  // Initialize providers
  taskProvider = new TaskProvider(coordinationService);
  workerProvider = new WorkerProvider(coordinationService);
  discoveryProvider = new DiscoveryProvider(coordinationService);

  // Register tree data providers
  vscode.window.registerTreeDataProvider("taskList", taskProvider);
  vscode.window.registerTreeDataProvider("workerStatus", workerProvider);
  vscode.window.registerTreeDataProvider("discoveries", discoveryProvider);

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand("claude-coordinator.refreshTasks", () => {
      taskProvider.refresh();
      workerProvider.refresh();
      discoveryProvider.refresh();
    }),

    vscode.commands.registerCommand(
      "claude-coordinator.createTask",
      async () => {
        const description = await vscode.window.showInputBox({
          prompt: "Enter task description",
          placeHolder: "Task description...",
        });
        if (!description) return;

        const priorityStr = await vscode.window.showQuickPick(
          ["1 (Highest)", "2", "3", "4", "5 (Lowest)"],
          {
            placeHolder: "Select priority",
          },
        );
        const priority = priorityStr ? parseInt(priorityStr[0]) : 3;

        try {
          await coordinationService.createTask(description, priority);
          taskProvider.refresh();
          vscode.window.showInformationMessage("Task created successfully");
        } catch (error) {
          vscode.window.showErrorMessage(`Failed to create task: ${error}`);
        }
      },
    ),

    vscode.commands.registerCommand(
      "claude-coordinator.claimTask",
      async (item: TaskItem) => {
        try {
          await coordinationService.claimTask(item.task.id);
          taskProvider.refresh();
          vscode.window.showInformationMessage(
            `Claimed task: ${item.task.description}`,
          );
        } catch (error) {
          vscode.window.showErrorMessage(`Failed to claim task: ${error}`);
        }
      },
    ),

    vscode.commands.registerCommand(
      "claude-coordinator.completeTask",
      async (item: TaskItem) => {
        const result = await vscode.window.showInputBox({
          prompt: "Enter completion result (optional)",
          placeHolder: "Task result...",
        });

        try {
          await coordinationService.completeTask(
            item.task.id,
            result || "Completed via VS Code",
          );
          taskProvider.refresh();
          vscode.window.showInformationMessage(
            `Completed task: ${item.task.description}`,
          );
        } catch (error) {
          vscode.window.showErrorMessage(`Failed to complete task: ${error}`);
        }
      },
    ),

    vscode.commands.registerCommand(
      "claude-coordinator.failTask",
      async (item: TaskItem) => {
        const reason = await vscode.window.showInputBox({
          prompt: "Enter failure reason",
          placeHolder: "Failure reason...",
        });

        try {
          await coordinationService.failTask(
            item.task.id,
            reason || "Failed via VS Code",
          );
          taskProvider.refresh();
          vscode.window.showErrorMessage(
            `Failed task: ${item.task.description}`,
          );
        } catch (error) {
          vscode.window.showErrorMessage(
            `Failed to mark task as failed: ${error}`,
          );
        }
      },
    ),

    vscode.commands.registerCommand(
      "claude-coordinator.viewTaskDetails",
      async (item: TaskItem) => {
        const task = item.task;
        const panel = vscode.window.createWebviewPanel(
          "taskDetails",
          `Task: ${task.id}`,
          vscode.ViewColumn.One,
          {},
        );

        panel.webview.html = getTaskDetailsHtml(task);
      },
    ),

    vscode.commands.registerCommand(
      "claude-coordinator.initCoordination",
      async () => {
        const goal = await vscode.window.showInputBox({
          prompt: "Enter coordination goal",
          placeHolder: "Project goal...",
        });
        if (!goal) return;

        try {
          await coordinationService.initCoordination(goal);
          taskProvider.refresh();
          vscode.window.showInformationMessage(
            "Coordination initialized successfully",
          );
        } catch (error) {
          vscode.window.showErrorMessage(
            `Failed to initialize coordination: ${error}`,
          );
        }
      },
    ),

    vscode.commands.registerCommand(
      "claude-coordinator.addDiscovery",
      async () => {
        const title = await vscode.window.showInputBox({
          prompt: "Enter discovery title",
          placeHolder: "Discovery title...",
        });
        if (!title) return;

        const content = await vscode.window.showInputBox({
          prompt: "Enter discovery content",
          placeHolder: "Discovery content...",
        });
        if (!content) return;

        try {
          await coordinationService.addDiscovery(title, content);
          discoveryProvider.refresh();
          vscode.window.showInformationMessage("Discovery added successfully");
        } catch (error) {
          vscode.window.showErrorMessage(`Failed to add discovery: ${error}`);
        }
      },
    ),
  );

  // Set up auto-refresh
  if (autoRefresh) {
    refreshInterval = setInterval(() => {
      taskProvider.refresh();
      workerProvider.refresh();
      discoveryProvider.refresh();
    }, refreshIntervalMs);

    context.subscriptions.push({
      dispose: () => {
        if (refreshInterval) {
          clearInterval(refreshInterval);
        }
      },
    });
  }

  // Watch for file changes in coordination directory
  if (fs.existsSync(coordPath)) {
    const watcher = vscode.workspace.createFileSystemWatcher(
      new vscode.RelativePattern(coordPath, "**/*"),
    );

    watcher.onDidChange(() => {
      taskProvider.refresh();
      workerProvider.refresh();
      discoveryProvider.refresh();
    });

    context.subscriptions.push(watcher);
  }

  // Create status bar item
  const statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    100,
  );
  statusBarItem.text = "$(pulse) Claude Coordinator";
  statusBarItem.tooltip = "Claude Multi-Agent Coordinator";
  statusBarItem.command = "claude-coordinator.refreshTasks";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);
}

function getTaskDetailsHtml(task: any): string {
  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Task Details</title>
        <style>
            body {
                font-family: var(--vscode-font-family);
                padding: 20px;
                color: var(--vscode-foreground);
                background-color: var(--vscode-editor-background);
            }
            .field {
                margin-bottom: 15px;
            }
            .label {
                font-weight: bold;
                color: var(--vscode-textLink-foreground);
            }
            .value {
                margin-top: 5px;
                padding: 10px;
                background-color: var(--vscode-textBlockQuote-background);
                border-radius: 4px;
            }
            .status {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                text-transform: uppercase;
            }
            .status.available { background-color: #4CAF50; color: white; }
            .status.claimed { background-color: #FF9800; color: white; }
            .status.in_progress { background-color: #2196F3; color: white; }
            .status.done { background-color: #9E9E9E; color: white; }
            .status.failed { background-color: #F44336; color: white; }
        </style>
    </head>
    <body>
        <h1>Task: ${task.id}</h1>
        <div class="field">
            <div class="label">Description</div>
            <div class="value">${task.description}</div>
        </div>
        <div class="field">
            <div class="label">Status</div>
            <div class="value">
                <span class="status ${task.status}">${task.status}</span>
            </div>
        </div>
        <div class="field">
            <div class="label">Priority</div>
            <div class="value">${task.priority} ${task.priority === 1 ? "(Highest)" : task.priority === 5 ? "(Lowest)" : ""}</div>
        </div>
        ${
          task.assigned_to
            ? `
        <div class="field">
            <div class="label">Assigned To</div>
            <div class="value">${task.assigned_to}</div>
        </div>
        `
            : ""
        }
        ${
          task.dependencies && task.dependencies.length > 0
            ? `
        <div class="field">
            <div class="label">Dependencies</div>
            <div class="value">${task.dependencies.join(", ")}</div>
        </div>
        `
            : ""
        }
        ${
          task.created_at
            ? `
        <div class="field">
            <div class="label">Created At</div>
            <div class="value">${new Date(task.created_at).toLocaleString()}</div>
        </div>
        `
            : ""
        }
        ${
          task.result
            ? `
        <div class="field">
            <div class="label">Result</div>
            <div class="value">${task.result}</div>
        </div>
        `
            : ""
        }
    </body>
    </html>
    `;
}

export function deactivate() {
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }
}
