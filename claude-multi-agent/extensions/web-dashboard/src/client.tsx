import React, { useCallback, useEffect } from "react";
import { createRoot } from "react-dom/client";
import {
  Action,
  ActionSchema,
  DataModel,
  UIElement,
  UITree,
  action,
  createCatalog,
} from "@json-render/core";
import {
  ActionProvider,
  ComponentRegistry,
  ComponentRenderProps,
  DataProvider,
  Renderer,
  flatToTree,
  useActions,
  useData,
  useDataValue,
} from "@json-render/react";
import { z } from "zod";

// Type declaration for injected dashboard config from server
declare global {
  interface Window {
    __DASHBOARD_CONFIG__?: { apiKey?: string };
  }
}

// Get API key injected by server for authenticated requests
function getApiKey(): string | undefined {
  return window.__DASHBOARD_CONFIG__?.apiKey || undefined;
}

type Task = {
  id: string;
  description: string;
  status: string;
  priority: number;
  assigned_to?: string;
};

type Worker = {
  id: string;
  status: string;
  current_task?: string;
  tasks_completed?: number;
};

type Discovery = {
  title: string;
  content: string;
  created_by: string;
  created_at: string;
};

type ApiStatus = {
  tasks?: {
    available?: number;
    claimed?: number;
    in_progress?: number;
    done?: number;
    failed?: number;
  };
  workers?: {
    idle?: number;
    busy?: number;
    offline?: number;
  };
  timestamp?: string;
};

type DashboardStatus = {
  tasks: {
    available: number;
    claimed: number;
    in_progress: number;
    done: number;
    failed: number;
    active: number;
  };
  workers: {
    idle: number;
    busy: number;
    offline: number;
    active: number;
  };
  timestamp?: string;
};

const catalog = createCatalog({
  name: "dashboard",
  components: {
    Page: { props: z.object({}), hasChildren: true },
    Header: { props: z.object({ title: z.string() }) },
    Container: { props: z.object({}), hasChildren: true },
    StatsGrid: { props: z.object({}), hasChildren: true },
    StatCard: {
      props: z.object({
        label: z.string(),
        valuePath: z.string(),
        tone: z
          .enum(["available", "in-progress", "done", "failed", "workers"])
          .optional(),
      }),
    },
    MainGrid: { props: z.object({}), hasChildren: true },
    Column: { props: z.object({}), hasChildren: true },
    Panel: {
      props: z.object({
        title: z.string(),
        action: ActionSchema.optional(),
      }),
      hasChildren: true,
    },
    TaskList: {
      props: z.object({
        itemsPath: z.string(),
        emptyText: z.string(),
      }),
    },
    WorkerList: {
      props: z.object({
        itemsPath: z.string(),
        emptyText: z.string(),
      }),
    },
    DiscoveryList: {
      props: z.object({
        itemsPath: z.string(),
        emptyText: z.string(),
        limit: z.number().int().positive().optional(),
      }),
    },
    Timestamp: {
      props: z.object({
        label: z.string(),
      }),
    },
  },
  actions: {
    refresh: {
      description: "Refresh dashboard data",
    },
  },
});

type ElementWithParent = UIElement & { parentKey?: string | null };

const dashboardTree = buildDashboardTree();
const dashboardTreeValidation = catalog.validateTree(dashboardTree);
const validatedTree = dashboardTreeValidation.data ?? dashboardTree;
if (!dashboardTreeValidation.success) {
  console.error("Invalid dashboard UI tree", dashboardTreeValidation.error);
}

type HeaderProps = { title: string };
type StatCardProps = {
  label: string;
  valuePath: string;
  tone?: "available" | "in-progress" | "done" | "failed" | "workers";
};
type PanelProps = { title: string; action?: Action };
type ListProps = { itemsPath: string; emptyText: string };
type DiscoveryListProps = ListProps & { limit?: number };
type TimestampProps = { label: string };

const registry: ComponentRegistry = {
  Page: ({ children }) => <div className="page">{children}</div>,
  Header: ({ element }: ComponentRenderProps<HeaderProps>) => {
    const connected = useDataValue<boolean>("/connection/connected") ?? false;
    return (
      <header className="header">
        <h1>{(element.props as HeaderProps).title}</h1>
        <div className="connection-status">
          <span className={`status-dot${connected ? " connected" : ""}`}></span>
          <span>{connected ? "Connected" : "Disconnected"}</span>
        </div>
      </header>
    );
  },
  Container: ({ children }) => <div className="container">{children}</div>,
  StatsGrid: ({ children }) => <div className="stats-grid">{children}</div>,
  StatCard: ({ element }: ComponentRenderProps<StatCardProps>) => {
    const props = element.props as StatCardProps;
    const value = useDataValue<number>(props.valuePath);
    const toneClass = props.tone ? ` ${props.tone}` : "";
    return (
      <div className={`stat-card${toneClass}`}>
        <h3>{props.label}</h3>
        <div className="value">{typeof value === "number" ? value : 0}</div>
      </div>
    );
  },
  MainGrid: ({ children }) => <div className="main-grid">{children}</div>,
  Column: ({ children }) => <div className="column">{children}</div>,
  Panel: ({
    element,
    children,
    onAction,
  }: ComponentRenderProps<PanelProps>) => {
    const props = element.props as PanelProps;
    const actionConfig = props.action;
    return (
      <div className="panel">
        <div className="panel-header">
          <h2>{props.title}</h2>
          {actionConfig ? (
            <button
              className="refresh-btn"
              onClick={() => onAction?.(actionConfig)}
              type="button"
            >
              Refresh
            </button>
          ) : null}
        </div>
        <div className="panel-content">{children}</div>
      </div>
    );
  },
  TaskList: ({ element }: ComponentRenderProps<ListProps>) => {
    const props = element.props as ListProps;
    const tasksValue = useDataValue<Task[]>(props.itemsPath);
    const tasks = Array.isArray(tasksValue) ? tasksValue : [];
    const statusOrder: Record<string, number> = {
      in_progress: 0,
      claimed: 1,
      available: 2,
      failed: 3,
      done: 4,
    };

    if (tasks.length === 0) {
      return (
        <div className="task-list">
          <div className="empty-state">{props.emptyText}</div>
        </div>
      );
    }

    const sorted = [...tasks].sort((a, b) => {
      const statusDelta =
        (statusOrder[a.status] ?? 99) - (statusOrder[b.status] ?? 99);
      if (statusDelta !== 0) return statusDelta;
      return a.priority - b.priority;
    });

    return (
      <div className="task-list">
        {sorted.map((task) => {
          const status = task.status ?? "unknown";
          const priority =
            typeof task.priority === "number" ? task.priority : 0;
          return (
            <div className="task-item" key={task.id}>
              <div className="task-info">
                <div className="task-description">{task.description ?? ""}</div>
                <div className="task-meta">
                  <span>ID: {task.id}</span>
                  {task.assigned_to ? (
                    <span>Assigned: {task.assigned_to}</span>
                  ) : null}
                </div>
              </div>
              <span className="priority-badge">P{priority}</span>
              <span className={`status-badge status-${status}`}>
                {status.replace("_", " ")}
              </span>
            </div>
          );
        })}
      </div>
    );
  },
  WorkerList: ({ element }: ComponentRenderProps<ListProps>) => {
    const props = element.props as ListProps;
    const workersValue = useDataValue<Worker[]>(props.itemsPath);
    const workers = Array.isArray(workersValue) ? workersValue : [];

    if (workers.length === 0) {
      return (
        <div className="worker-list">
          <div className="empty-state">{props.emptyText}</div>
        </div>
      );
    }

    return (
      <div className="worker-list">
        {workers.map((worker) => (
          <div className="worker-item" key={worker.id}>
            <div className="worker-info">
              <h4>{worker.id}</h4>
              <p>
                {worker.current_task
                  ? `Working on: ${worker.current_task}`
                  : "No active task"}
              </p>
              <p>Tasks completed: {worker.tasks_completed ?? 0}</p>
            </div>
            <div
              className={`worker-status ${worker.status || "offline"}`}
              title={worker.status}
            ></div>
          </div>
        ))}
      </div>
    );
  },
  DiscoveryList: ({ element }: ComponentRenderProps<DiscoveryListProps>) => {
    const props = element.props as DiscoveryListProps;
    const discoveriesValue = useDataValue<Discovery[]>(props.itemsPath);
    const discoveries = Array.isArray(discoveriesValue) ? discoveriesValue : [];
    const limit = props.limit ?? 10;

    if (discoveries.length === 0) {
      return (
        <div className="discovery-list">
          <div className="empty-state">{props.emptyText}</div>
        </div>
      );
    }

    return (
      <div className="discovery-list">
        {discoveries.slice(0, limit).map((discovery, index) => {
          const title = discovery.title || "Discovery";
          const rawContent = discovery.content || "";
          const content =
            rawContent.length > 200
              ? `${rawContent.slice(0, 200)}...`
              : rawContent;
          const createdAt = discovery.created_at
            ? new Date(discovery.created_at).toLocaleString()
            : "";
          const key = `${title}-${index}`;
          return (
            <div className="discovery-item" key={key}>
              <h4>{title}</h4>
              <p>{content}</p>
              <div className="discovery-meta">
                By {discovery.created_by}
                {createdAt ? ` - ${createdAt}` : ""}
              </div>
            </div>
          );
        })}
      </div>
    );
  },
  Timestamp: ({ element }: ComponentRenderProps<TimestampProps>) => {
    const props = element.props as TimestampProps;
    const lastUpdated = useDataValue<string>("/lastUpdated");
    const display = lastUpdated
      ? new Date(lastUpdated).toLocaleTimeString()
      : "-";
    return (
      <p className="timestamp">
        {props.label}: {display}
      </p>
    );
  },
};

const initialData: DataModel = {
  status: normalizeStatus({}),
  tasks: [],
  workers: [],
  discoveries: [],
  connection: { connected: false },
  lastUpdated: null,
};

function DashboardController() {
  const { update } = useData();
  const { registerHandler } = useActions();

  const applyStatus = useCallback(
    (status: ApiStatus | undefined) => {
      const normalized = normalizeStatus(status);
      const timestamp = status?.timestamp ?? new Date().toISOString();
      update({
        "/status": normalized,
        "/lastUpdated": timestamp,
      });
    },
    [update],
  );

  const applyPayload = useCallback(
    (payload: {
      status?: ApiStatus;
      tasks?: Task[];
      workers?: Worker[];
      discoveries?: Discovery[];
    }) => {
      const tasks = Array.isArray(payload.tasks) ? payload.tasks : [];
      const workers = Array.isArray(payload.workers) ? payload.workers : [];
      const discoveries = Array.isArray(payload.discoveries)
        ? payload.discoveries
        : [];
      const normalized = normalizeStatus(payload.status);
      const timestamp = payload.status?.timestamp ?? new Date().toISOString();

      update({
        "/status": normalized,
        "/tasks": tasks,
        "/workers": workers,
        "/discoveries": discoveries,
        "/lastUpdated": timestamp,
      });
    },
    [update],
  );

  const refresh = useCallback(async () => {
    const [status, tasksResponse, workersResponse, discoveriesResponse] =
      await Promise.all([
        fetchJSON<ApiStatus>("/api/status"),
        fetchJSON<{ tasks: Task[] }>("/api/tasks"),
        fetchJSON<{ workers: Worker[] }>("/api/workers"),
        fetchJSON<{ discoveries: Discovery[] }>("/api/discoveries"),
      ]);

    applyPayload({
      status,
      tasks: tasksResponse.tasks,
      workers: workersResponse.workers,
      discoveries: discoveriesResponse.discoveries,
    });
  }, [applyPayload]);

  useEffect(() => {
    registerHandler("refresh", refresh);
  }, [refresh, registerHandler]);

  useEffect(() => {
    refresh().catch((error) => {
      console.error("Failed to refresh dashboard", error);
    });
  }, [refresh]);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let closed = false;

    const connect = () => {
      if (closed) return;
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const apiKey = getApiKey();
      const wsUrl = apiKey
        ? `${protocol}://${window.location.host}/ws?token=${encodeURIComponent(apiKey)}`
        : `${protocol}://${window.location.host}/ws`;
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        update({ "/connection/connected": true });
      };

      ws.onclose = () => {
        update({ "/connection/connected": false });
        if (!closed) {
          reconnectTimer = setTimeout(connect, 5000);
        }
      };

      ws.onerror = () => {
        update({ "/connection/connected": false });
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data as string);
          if (!message || typeof message !== "object") return;
          switch (message.type) {
            case "init":
              applyPayload({
                status: message.data?.status,
                tasks: message.data?.tasks,
                workers: message.data?.workers,
                discoveries: message.data?.discoveries,
              });
              break;
            case "status":
              applyStatus(message.data);
              break;
            case "tasks":
              update({
                "/tasks": Array.isArray(message.data) ? message.data : [],
                "/lastUpdated": new Date().toISOString(),
              });
              break;
            case "workers":
              update({
                "/workers": Array.isArray(message.data) ? message.data : [],
                "/lastUpdated": new Date().toISOString(),
              });
              break;
            case "discoveries":
              update({
                "/discoveries": Array.isArray(message.data) ? message.data : [],
                "/lastUpdated": new Date().toISOString(),
              });
              break;
            default:
              break;
          }
        } catch (error) {
          console.warn("Failed to parse dashboard message", error);
        }
      };
    };

    connect();

    return () => {
      closed = true;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
    };
  }, [applyPayload, applyStatus, update]);

  return <Renderer tree={validatedTree} registry={registry} />;
}

function App() {
  return (
    <DataProvider initialData={initialData}>
      <ActionProvider>
        <DashboardController />
      </ActionProvider>
    </DataProvider>
  );
}

function buildDashboardTree(): UITree {
  const elements: ElementWithParent[] = [
    { key: "page", type: "Page", props: {}, parentKey: null },
    {
      key: "header",
      type: "Header",
      props: { title: "Claude Coordinator Dashboard" },
      parentKey: "page",
    },
    { key: "container", type: "Container", props: {}, parentKey: "page" },
    { key: "stats", type: "StatsGrid", props: {}, parentKey: "container" },
    {
      key: "stat-available",
      type: "StatCard",
      props: {
        label: "Available",
        valuePath: "/status/tasks/available",
        tone: "available",
      },
      parentKey: "stats",
    },
    {
      key: "stat-in-progress",
      type: "StatCard",
      props: {
        label: "In Progress",
        valuePath: "/status/tasks/active",
        tone: "in-progress",
      },
      parentKey: "stats",
    },
    {
      key: "stat-done",
      type: "StatCard",
      props: {
        label: "Done",
        valuePath: "/status/tasks/done",
        tone: "done",
      },
      parentKey: "stats",
    },
    {
      key: "stat-failed",
      type: "StatCard",
      props: {
        label: "Failed",
        valuePath: "/status/tasks/failed",
        tone: "failed",
      },
      parentKey: "stats",
    },
    {
      key: "stat-workers",
      type: "StatCard",
      props: {
        label: "Workers Active",
        valuePath: "/status/workers/active",
        tone: "workers",
      },
      parentKey: "stats",
    },
    { key: "main-grid", type: "MainGrid", props: {}, parentKey: "container" },
    {
      key: "tasks-panel",
      type: "Panel",
      props: { title: "Tasks", action: action.simple("refresh") },
      parentKey: "main-grid",
    },
    {
      key: "tasks-list",
      type: "TaskList",
      props: { itemsPath: "/tasks", emptyText: "No tasks found" },
      parentKey: "tasks-panel",
    },
    { key: "side-column", type: "Column", props: {}, parentKey: "main-grid" },
    {
      key: "workers-panel",
      type: "Panel",
      props: { title: "Workers" },
      parentKey: "side-column",
    },
    {
      key: "workers-list",
      type: "WorkerList",
      props: { itemsPath: "/workers", emptyText: "No workers registered" },
      parentKey: "workers-panel",
    },
    {
      key: "discoveries-panel",
      type: "Panel",
      props: { title: "Discoveries" },
      parentKey: "side-column",
    },
    {
      key: "discoveries-list",
      type: "DiscoveryList",
      props: {
        itemsPath: "/discoveries",
        emptyText: "No discoveries yet",
        limit: 10,
      },
      parentKey: "discoveries-panel",
    },
    {
      key: "timestamp",
      type: "Timestamp",
      props: { label: "Last updated" },
      parentKey: "container",
    },
  ];

  return flatToTree(elements);
}

function normalizeStatus(status?: ApiStatus): DashboardStatus {
  const tasks = status?.tasks ?? {};
  const workers = status?.workers ?? {};
  const available = toNumber(tasks.available);
  const claimed = toNumber(tasks.claimed);
  const inProgress = toNumber(tasks.in_progress);
  const done = toNumber(tasks.done);
  const failed = toNumber(tasks.failed);
  const idle = toNumber(workers.idle);
  const busy = toNumber(workers.busy);
  const offline = toNumber(workers.offline);

  return {
    tasks: {
      available,
      claimed,
      in_progress: inProgress,
      done,
      failed,
      active: claimed + inProgress,
    },
    workers: {
      idle,
      busy,
      offline,
      active: idle + busy,
    },
    timestamp: status?.timestamp,
  };
}

function toNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

async function fetchJSON<T>(url: string): Promise<T> {
  const headers: Record<string, string> = {};
  const apiKey = getApiKey();
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  const response = await fetch(url, { headers });
  if (!response.ok) {
    throw new Error(`Request failed for ${url}: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Dashboard root element not found");
}

createRoot(rootElement).render(<App />);
