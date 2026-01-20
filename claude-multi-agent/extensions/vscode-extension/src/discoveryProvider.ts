import * as vscode from "vscode";
import { CoordinationService, Discovery } from "./coordinationService";

export class DiscoveryItem extends vscode.TreeItem {
  constructor(
    public readonly discovery: Discovery,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
  ) {
    super(discovery.title, collapsibleState);

    this.id = discovery.id;
    this.tooltip = `${discovery.title}\n\n${discovery.content}\n\nBy: ${discovery.created_by}`;
    this.description = `by ${discovery.created_by}`;
    this.iconPath = new vscode.ThemeIcon("lightbulb");
  }
}

export class DiscoveryProvider implements vscode.TreeDataProvider<DiscoveryItem> {
  private _onDidChangeTreeData: vscode.EventEmitter<
    DiscoveryItem | undefined | null | void
  > = new vscode.EventEmitter<DiscoveryItem | undefined | null | void>();
  readonly onDidChangeTreeData: vscode.Event<
    DiscoveryItem | undefined | null | void
  > = this._onDidChangeTreeData.event;

  constructor(private coordinationService: CoordinationService) {}

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: DiscoveryItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: DiscoveryItem): Promise<DiscoveryItem[]> {
    if (element) {
      return [];
    }

    const discoveries = await this.coordinationService.getDiscoveries();

    return discoveries.map(
      (discovery) =>
        new DiscoveryItem(discovery, vscode.TreeItemCollapsibleState.None),
    );
  }
}
