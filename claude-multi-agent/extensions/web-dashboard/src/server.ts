import express from "express";
import cors from "cors";
import { WebSocketServer, WebSocket } from "ws";
import { createServer } from "http";
import * as fs from "fs";
import * as path from "path";
import * as chokidar from "chokidar";
import { config } from "dotenv";

config();

const isProduction = process.env.NODE_ENV === "production";
const dashboardApiKey = process.env.COORDINATION_API_KEY || "";
const corsOrigins = (process.env.CORS_ALLOWED_ORIGINS || "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

if (isProduction && !dashboardApiKey) {
  throw new Error("COORDINATION_API_KEY is required in production");
}
if (isProduction && corsOrigins.length === 0) {
  throw new Error("CORS_ALLOWED_ORIGINS is required in production");
}

const app = express();
const port = parseInt(process.env.DASHBOARD_PORT || "3004");
const coordinationDir = process.env.COORDINATION_DIR || ".coordination";

function extractApiKey(
  headers: Record<string, string | string[] | undefined>,
): string | undefined {
  const authHeader = headers.authorization;
  if (
    typeof authHeader === "string" &&
    authHeader.toLowerCase().startsWith("bearer ")
  ) {
    return authHeader.slice(7).trim();
  }
  const apiKeyHeader = headers["x-api-key"];
  if (typeof apiKeyHeader === "string" && apiKeyHeader.trim()) {
    return apiKeyHeader.trim();
  }
  return undefined;
}

function requireAuth(
  req: express.Request,
  res: express.Response,
  next: express.NextFunction,
): void {
  if (!dashboardApiKey) {
    next();
    return;
  }
  if (!req.path.startsWith("/api")) {
    next();
    return;
  }
  const apiKey = extractApiKey(
    req.headers as Record<string, string | string[] | undefined>,
  );
  if (!apiKey || apiKey !== dashboardApiKey) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }
  next();
}

app.use(cors(corsOrigins.length ? { origin: corsOrigins } : undefined));
app.use(express.json());
app.use(requireAuth);

// Serve static files (excluding index.html which is served dynamically)
app.use(express.static(path.join(__dirname, "../public"), { index: false }));

// Serve index.html dynamically with injected API key for authenticated dashboard access
app.get("/", (_req, res) => {
  const indexPath = path.join(__dirname, "../public/index.html");
  if (!fs.existsSync(indexPath)) {
    res.status(404).send("Dashboard not found");
    return;
  }
  let html = fs.readFileSync(indexPath, "utf-8");
  // Inject API key into the page so the client can authenticate requests
  const configScript = `<script>window.__DASHBOARD_CONFIG__ = { apiKey: ${JSON.stringify(dashboardApiKey || "")} };</script>`;
  html = html.replace("</head>", `${configScript}</head>`);
  res.type("html").send(html);
});

// Data loading functions
function loadTasks(): any[] {
  const tasksPath = path.join(coordinationDir, "tasks.json");
  if (!fs.existsSync(tasksPath)) return [];
  try {
    return JSON.parse(fs.readFileSync(tasksPath, "utf-8")).tasks || [];
  } catch {
    return [];
  }
}

function loadWorkers(): any[] {
  const agentsPath = path.join(coordinationDir, "agents.json");
  const workersPath = path.join(coordinationDir, "workers.json");
  const filePath = fs.existsSync(agentsPath) ? agentsPath : workersPath;
  if (!fs.existsSync(filePath)) return [];
  try {
    const parsed = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    return parsed.agents || parsed.workers || [];
  } catch {
    return [];
  }
}

function loadDiscoveries(): any[] {
  const discoveriesPath = path.join(coordinationDir, "discoveries.json");
  if (!fs.existsSync(discoveriesPath)) return [];
  try {
    return (
      JSON.parse(fs.readFileSync(discoveriesPath, "utf-8")).discoveries || []
    );
  } catch {
    return [];
  }
}

function getStatus(): any {
  const tasks = loadTasks();
  const workers = loadWorkers();

  return {
    tasks: {
      total: tasks.length,
      available: tasks.filter((t) => t.status === "available").length,
      claimed: tasks.filter((t) => t.status === "claimed").length,
      in_progress: tasks.filter((t) => t.status === "in_progress").length,
      done: tasks.filter((t) => t.status === "done").length,
      failed: tasks.filter((t) => t.status === "failed").length,
    },
    workers: {
      total: workers.length,
      idle: workers.filter((w) => w.status === "idle").length,
      busy: workers.filter((w) => w.status === "busy").length,
      offline: workers.filter((w) => w.status === "offline").length,
    },
    timestamp: new Date().toISOString(),
  };
}

// API endpoints
app.get("/api/status", (req, res) => {
  res.json(getStatus());
});

app.get("/api/tasks", (req, res) => {
  res.json({ tasks: loadTasks() });
});

app.get("/api/workers", (req, res) => {
  res.json({ workers: loadWorkers() });
});

app.get("/api/discoveries", (req, res) => {
  res.json({ discoveries: loadDiscoveries() });
});

// Serve the dashboard HTML
app.get("/", (req, res) => {
  const indexPath = path.join(__dirname, "../public/index.html");
  const bundlePath = path.join(__dirname, "../public/dashboard.js");
  if (fs.existsSync(indexPath) && fs.existsSync(bundlePath)) {
    res.sendFile(indexPath);
    return;
  }
  res.send(getDashboardHTML());
});

// Create HTTP server
const server = createServer(app);

// WebSocket server for real-time updates
const wss = new WebSocketServer({ server, path: "/ws" });
const clients = new Set<WebSocket>();

wss.on("connection", (ws, req) => {
  if (dashboardApiKey) {
    // Check headers first, then query param for browser WebSocket compatibility
    let apiKey = extractApiKey(
      req.headers as Record<string, string | string[] | undefined>,
    );
    if (!apiKey && req.url) {
      const url = new URL(req.url, `http://${req.headers.host}`);
      apiKey = url.searchParams.get("token") || undefined;
    }
    if (!apiKey || apiKey !== dashboardApiKey) {
      ws.close(1008, "Unauthorized");
      return;
    }
  }
  clients.add(ws);
  console.log("Client connected");

  // Send initial data
  ws.send(
    JSON.stringify({
      type: "init",
      data: {
        status: getStatus(),
        tasks: loadTasks(),
        workers: loadWorkers(),
        discoveries: loadDiscoveries(),
      },
    }),
  );

  ws.on("close", () => {
    clients.delete(ws);
    console.log("Client disconnected");
  });
});

// Broadcast to all clients
function broadcast(message: any): void {
  const data = JSON.stringify(message);
  for (const client of clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(data);
    }
  }
}

// Watch for file changes
const watcher = chokidar.watch(
  [
    path.join(coordinationDir, "tasks.json"),
    path.join(coordinationDir, "agents.json"),
    path.join(coordinationDir, "workers.json"),
    path.join(coordinationDir, "discoveries.json"),
  ],
  {
    persistent: true,
    ignoreInitial: true,
  },
);

watcher.on("change", (filePath) => {
  console.log(`File changed: ${filePath}`);

  if (filePath.endsWith("tasks.json")) {
    broadcast({ type: "tasks", data: loadTasks() });
  } else if (
    filePath.endsWith("agents.json") ||
    filePath.endsWith("workers.json")
  ) {
    broadcast({ type: "workers", data: loadWorkers() });
  } else if (filePath.endsWith("discoveries.json")) {
    broadcast({ type: "discoveries", data: loadDiscoveries() });
  }

  broadcast({ type: "status", data: getStatus() });
});

// Start server
server.listen(port, () => {
  console.log(`Dashboard running at http://localhost:${port}`);
  console.log(`WebSocket at ws://localhost:${port}/ws`);
});

// Dashboard HTML - using textContent-based rendering for security
function getDashboardHTML(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude Coordinator Dashboard</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent: #e94560;
            --success: #4caf50;
            --warning: #ff9800;
            --info: #2196f3;
            --danger: #f44336;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }
        .header {
            background: var(--bg-secondary);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--bg-card);
        }
        .header h1 { font-size: 1.5rem; color: var(--accent); }
        .connection-status { display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--danger); }
        .status-dot.connected { background: var(--success); }
        .container { padding: 2rem; max-width: 1600px; margin: 0 auto; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
        .stat-card { background: var(--bg-card); border-radius: 8px; padding: 1.5rem; text-align: center; }
        .stat-card h3 { font-size: 0.875rem; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 0.5rem; }
        .stat-card .value { font-size: 2.5rem; font-weight: bold; }
        .stat-card.available .value { color: var(--success); }
        .stat-card.in-progress .value { color: var(--info); }
        .stat-card.done .value { color: var(--text-secondary); }
        .stat-card.failed .value { color: var(--danger); }
        .main-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 2rem; }
        @media (max-width: 1024px) { .main-grid { grid-template-columns: 1fr; } }
        .panel { background: var(--bg-secondary); border-radius: 8px; overflow: hidden; }
        .panel-header { background: var(--bg-card); padding: 1rem 1.5rem; display: flex; justify-content: space-between; align-items: center; }
        .panel-header h2 { font-size: 1rem; font-weight: 600; }
        .panel-content { padding: 1rem; max-height: 500px; overflow-y: auto; }
        .task-list, .worker-list, .discovery-list { display: flex; flex-direction: column; gap: 0.75rem; }
        .task-item, .worker-item, .discovery-item { background: var(--bg-card); border-radius: 6px; padding: 1rem; }
        .task-item { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
        .task-info { flex: 1; }
        .task-description { margin-bottom: 0.5rem; }
        .task-meta { display: flex; gap: 1rem; font-size: 0.75rem; color: var(--text-secondary); }
        .status-badge { padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; }
        .status-available { background: rgba(76, 175, 80, 0.2); color: var(--success); }
        .status-claimed { background: rgba(255, 152, 0, 0.2); color: var(--warning); }
        .status-in_progress { background: rgba(33, 150, 243, 0.2); color: var(--info); }
        .status-done { background: rgba(158, 158, 158, 0.2); color: var(--text-secondary); }
        .status-failed { background: rgba(244, 67, 54, 0.2); color: var(--danger); }
        .priority-badge { background: var(--accent); color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
        .worker-item { display: flex; justify-content: space-between; align-items: center; }
        .worker-info h4 { margin-bottom: 0.25rem; }
        .worker-info p { font-size: 0.75rem; color: var(--text-secondary); }
        .worker-status { width: 10px; height: 10px; border-radius: 50%; }
        .worker-status.idle { background: var(--success); }
        .worker-status.busy { background: var(--info); }
        .worker-status.offline { background: var(--text-secondary); }
        .discovery-item h4 { margin-bottom: 0.5rem; color: var(--accent); }
        .discovery-item p { font-size: 0.875rem; color: var(--text-secondary); }
        .discovery-meta { margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-secondary); }
        .empty-state { text-align: center; padding: 2rem; color: var(--text-secondary); }
        .refresh-btn { background: var(--accent); color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; font-size: 0.875rem; }
        .refresh-btn:hover { opacity: 0.9; }
        .timestamp { font-size: 0.75rem; color: var(--text-secondary); }
    </style>
</head>
<body>
    <header class="header">
        <h1>Claude Coordinator Dashboard</h1>
        <div class="connection-status">
            <span class="status-dot" id="connectionDot"></span>
            <span id="connectionText">Disconnected</span>
        </div>
    </header>
    <div class="container">
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card available"><h3>Available</h3><div class="value" id="statAvailable">0</div></div>
            <div class="stat-card in-progress"><h3>In Progress</h3><div class="value" id="statInProgress">0</div></div>
            <div class="stat-card done"><h3>Done</h3><div class="value" id="statDone">0</div></div>
            <div class="stat-card failed"><h3>Failed</h3><div class="value" id="statFailed">0</div></div>
            <div class="stat-card"><h3>Workers Active</h3><div class="value" id="statWorkers">0</div></div>
        </div>
        <div class="main-grid">
            <div class="panel">
                <div class="panel-header"><h2>Tasks</h2><button class="refresh-btn" onclick="refresh()">Refresh</button></div>
                <div class="panel-content"><div class="task-list" id="taskList"><div class="empty-state">Loading tasks...</div></div></div>
            </div>
            <div>
                <div class="panel" style="margin-bottom: 1rem;">
                    <div class="panel-header"><h2>Workers</h2></div>
                    <div class="panel-content"><div class="worker-list" id="workerList"><div class="empty-state">Loading workers...</div></div></div>
                </div>
                <div class="panel">
                    <div class="panel-header"><h2>Discoveries</h2></div>
                    <div class="panel-content"><div class="discovery-list" id="discoveryList"><div class="empty-state">Loading discoveries...</div></div></div>
                </div>
            </div>
        </div>
        <p class="timestamp" style="margin-top: 1rem;">Last updated: <span id="lastUpdated">-</span></p>
    </div>
    <script>
        let ws;
        let reconnectInterval;
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + window.location.host + '/ws');
            ws.onopen = () => {
                document.getElementById('connectionDot').classList.add('connected');
                document.getElementById('connectionText').textContent = 'Connected';
                if (reconnectInterval) { clearInterval(reconnectInterval); reconnectInterval = null; }
            };
            ws.onclose = () => {
                document.getElementById('connectionDot').classList.remove('connected');
                document.getElementById('connectionText').textContent = 'Disconnected';
                if (!reconnectInterval) { reconnectInterval = setInterval(connect, 5000); }
            };
            ws.onmessage = (event) => { handleMessage(JSON.parse(event.data)); };
        }
        function handleMessage(message) {
            switch (message.type) {
                case 'init':
                    updateStatus(message.data.status);
                    renderTasks(message.data.tasks);
                    renderWorkers(message.data.workers);
                    renderDiscoveries(message.data.discoveries);
                    break;
                case 'status': updateStatus(message.data); break;
                case 'tasks': renderTasks(message.data); break;
                case 'workers': renderWorkers(message.data); break;
                case 'discoveries': renderDiscoveries(message.data); break;
            }
            document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
        }
        function updateStatus(status) {
            document.getElementById('statAvailable').textContent = status.tasks.available;
            document.getElementById('statInProgress').textContent = status.tasks.claimed + status.tasks.in_progress;
            document.getElementById('statDone').textContent = status.tasks.done;
            document.getElementById('statFailed').textContent = status.tasks.failed;
            document.getElementById('statWorkers').textContent = status.workers.idle + status.workers.busy;
        }
        function createTaskElement(task) {
            const item = document.createElement('div');
            item.className = 'task-item';
            const info = document.createElement('div');
            info.className = 'task-info';
            const desc = document.createElement('div');
            desc.className = 'task-description';
            desc.textContent = task.description;
            const meta = document.createElement('div');
            meta.className = 'task-meta';
            const idSpan = document.createElement('span');
            idSpan.textContent = 'ID: ' + task.id;
            meta.appendChild(idSpan);
            if (task.assigned_to) {
                const assignedSpan = document.createElement('span');
                assignedSpan.textContent = 'Assigned: ' + task.assigned_to;
                meta.appendChild(assignedSpan);
            }
            info.appendChild(desc);
            info.appendChild(meta);
            const priority = document.createElement('span');
            priority.className = 'priority-badge';
            priority.textContent = 'P' + task.priority;
            const status = document.createElement('span');
            status.className = 'status-badge status-' + task.status;
            status.textContent = task.status.replace('_', ' ');
            item.appendChild(info);
            item.appendChild(priority);
            item.appendChild(status);
            return item;
        }
        function renderTasks(tasks) {
            const container = document.getElementById('taskList');
            container.textContent = '';
            if (!tasks || tasks.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'empty-state';
                empty.textContent = 'No tasks found';
                container.appendChild(empty);
                return;
            }
            const statusOrder = { in_progress: 0, claimed: 1, available: 2, failed: 3, done: 4 };
            tasks.sort((a, b) => (statusOrder[a.status] - statusOrder[b.status]) || (a.priority - b.priority));
            tasks.forEach(task => container.appendChild(createTaskElement(task)));
        }
        function renderWorkers(workers) {
            const container = document.getElementById('workerList');
            container.textContent = '';
            if (!workers || workers.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'empty-state';
                empty.textContent = 'No workers registered';
                container.appendChild(empty);
                return;
            }
            workers.forEach(worker => {
                const item = document.createElement('div');
                item.className = 'worker-item';
                const info = document.createElement('div');
                info.className = 'worker-info';
                const h4 = document.createElement('h4');
                h4.textContent = worker.id;
                const p1 = document.createElement('p');
                p1.textContent = worker.current_task ? 'Working on: ' + worker.current_task : 'No active task';
                const p2 = document.createElement('p');
                p2.textContent = 'Tasks completed: ' + (worker.tasks_completed || 0);
                info.appendChild(h4);
                info.appendChild(p1);
                info.appendChild(p2);
                const statusDiv = document.createElement('div');
                statusDiv.className = 'worker-status ' + worker.status;
                statusDiv.title = worker.status;
                item.appendChild(info);
                item.appendChild(statusDiv);
                container.appendChild(item);
            });
        }
        function renderDiscoveries(discoveries) {
            const container = document.getElementById('discoveryList');
            container.textContent = '';
            if (!discoveries || discoveries.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'empty-state';
                empty.textContent = 'No discoveries yet';
                container.appendChild(empty);
                return;
            }
            discoveries.slice(0, 10).forEach(discovery => {
                const item = document.createElement('div');
                item.className = 'discovery-item';
                const h4 = document.createElement('h4');
                h4.textContent = discovery.title;
                const p = document.createElement('p');
                const content = discovery.content.length > 200 ? discovery.content.substring(0, 200) + '...' : discovery.content;
                p.textContent = content;
                const meta = document.createElement('div');
                meta.className = 'discovery-meta';
                meta.textContent = 'By ' + discovery.created_by + ' - ' + new Date(discovery.created_at).toLocaleString();
                item.appendChild(h4);
                item.appendChild(p);
                item.appendChild(meta);
                container.appendChild(item);
            });
        }
        function refresh() {
            fetch('/api/status').then(res => res.json()).then(data => updateStatus(data));
            fetch('/api/tasks').then(res => res.json()).then(data => renderTasks(data.tasks));
            fetch('/api/workers').then(res => res.json()).then(data => renderWorkers(data.workers));
            fetch('/api/discoveries').then(res => res.json()).then(data => renderDiscoveries(data.discoveries));
            document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
        }
        connect();
    </script>
</body>
</html>`;
}

export { app, server };
