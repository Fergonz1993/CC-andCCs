# Claude Coordinator Extensions

This directory contains extensions and integrations for the Claude Multi-Agent Coordinator system.

## Directory Structure

```
extensions/
├── vscode-extension/      # VS Code extension for task management
├── web-dashboard/         # Real-time web dashboard
├── integrations/
│   ├── slack/            # Slack bot for notifications
│   ├── github/           # GitHub Actions integration
│   └── webhook/          # Webhook support for external systems
├── api/
│   ├── rest/             # REST API wrapper
│   └── graphql/          # GraphQL API layer
├── cli-plugins/          # CLI plugin system
└── monitoring/
    ├── prometheus/       # Prometheus metrics export
    └── opentelemetry/    # OpenTelemetry tracing
```

## Extensions Overview

### 1. VS Code Extension (`vscode-extension/`)

A full-featured VS Code extension for managing coordination tasks.

**Features:**
- Task list sidebar with real-time updates
- Worker status panel
- Discovery viewer
- Inline task actions (claim, complete, fail)
- Auto-refresh and file watching

**Installation:**
```bash
cd extensions/vscode-extension
npm install
npm run build
# Then install the extension in VS Code
```

### 2. Web Dashboard (`web-dashboard/`)

A real-time web dashboard for monitoring coordination status.

**Features:**
- Real-time task board with WebSocket updates
- Worker status cards
- Discovery list
- Metrics overview

**Usage:**
```bash
cd extensions/web-dashboard
npm install
npm run build
npm start
# Open http://localhost:3004
```

### 3. Slack Integration (`integrations/slack/`)

A Slack bot for receiving notifications and managing tasks.

**Features:**
- Slash commands: `/tasks`, `/create-task`, `/claim-task`, `/complete-task`, `/coord-status`
- Interactive messages with claim/complete buttons
- Real-time notifications on task status changes
- App mentions support

**Configuration:**
```bash
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_SIGNING_SECRET=...
export SLACK_APP_TOKEN=xapp-... # For Socket Mode
export SLACK_CHANNEL=#claude-coordinator
export COORDINATION_DIR=.coordination
```

### 4. GitHub Actions Integration (`integrations/github/`)

GitHub Actions for CI/CD integration.

**Features:**
- Create tasks from GitHub issues
- Sync task status to issues
- Link PRs to tasks
- Status checks for coordination

**Usage in workflow:**
```yaml
- uses: ./extensions/integrations/github
  with:
    github-token: ${{ secrets.GITHUB_TOKEN }}
    action: import-issues
    label: coordination
```

### 5. Webhook Support (`integrations/webhook/`)

Configurable webhooks for external system integration.

**Features:**
- Register/unregister webhooks via API
- Event filtering
- Payload customization with signatures
- Retry on failure with exponential backoff

**Events:**
- `task.created`, `task.claimed`, `task.started`, `task.completed`, `task.failed`
- `discovery.added`
- `worker.registered`, `worker.disconnected`
- `coordination.started`, `coordination.completed`

**API:**
```bash
# Register webhook
curl -X POST http://localhost:3001/webhooks \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/hook", "events": ["task.completed"], "secret": "..."}'
```

### 6. REST API (`api/rest/`)

Complete REST API wrapper with OpenAPI documentation.

**Features:**
- Full CRUD for tasks, workers, discoveries
- Task lifecycle operations (claim, complete, fail)
- Worker registration and heartbeat
- Swagger UI at `/api-docs`

**Endpoints:**
- `GET/POST /api/v1/tasks`
- `GET/PUT/DELETE /api/v1/tasks/:id`
- `POST /api/v1/tasks/:id/claim`
- `POST /api/v1/tasks/:id/complete`
- `GET/POST /api/v1/workers`
- `GET/POST /api/v1/discoveries`
- `GET /api/v1/status`

### 7. GraphQL API (`api/graphql/`)

GraphQL API with subscriptions for real-time updates.

**Features:**
- Full queries for tasks, workers, discoveries
- Mutations for all operations
- Subscriptions for real-time events
- WebSocket support

**Example query:**
```graphql
query {
  tasks(filter: { status: available }) {
    id
    description
    priority
    status
  }
}
```

### 8. CLI Plugins (`cli-plugins/`)

Extensible plugin system for the CLI.

**Features:**
- Plugin lifecycle hooks
- Custom commands per plugin
- Plugin API for task/worker/discovery operations
- Plugin scaffold generator

**Commands:**
```bash
coord plugin list           # List installed plugins
coord plugin install <path> # Install a plugin
coord plugin create <name>  # Create plugin scaffold
coord plugin run <cmd>      # Run plugin command
```

### 9. Prometheus Metrics (`monitoring/prometheus/`)

Prometheus metrics exporter for monitoring.

**Metrics:**
- `coordinator_tasks_total` - Total tasks
- `coordinator_tasks_by_status` - Tasks by status
- `coordinator_tasks_completed_total` - Counter of completed tasks
- `coordinator_task_duration_seconds` - Task execution duration histogram
- `coordinator_workers_total` - Total workers
- `coordinator_workers_by_status` - Workers by status
- `coordinator_task_throughput_per_minute` - Tasks completed per minute

**Endpoints:**
- `GET /metrics` - Prometheus format
- `GET /status` - JSON format
- `GET /health` - Health check

### 10. OpenTelemetry Tracing (`monitoring/opentelemetry/`)

Distributed tracing with OpenTelemetry.

**Features:**
- Automatic instrumentation
- Task lifecycle tracing
- Context propagation
- Multiple exporter support (OTLP, Jaeger, Zipkin)

**Spans:**
- `task.create`, `task.claim`, `task.execute`, `task.complete`
- `worker.register`, `worker.heartbeat`
- `discovery.add`
- `coordination.start`, `coordination.complete`

## Environment Variables

Common environment variables across extensions:

| Variable | Description | Default |
|----------|-------------|---------|
| `COORDINATION_DIR` | Path to coordination directory | `.coordination` |
| `REST_API_PORT` | REST API server port | `3002` |
| `GRAPHQL_PORT` | GraphQL server port | `3003` |
| `DASHBOARD_PORT` | Dashboard server port | `3004` |
| `WEBHOOK_PORT` | Webhook server port | `3001` |
| `METRICS_PORT` | Prometheus metrics port | `9090` |
| `OTEL_SERVICE_NAME` | OpenTelemetry service name | `claude-coordinator` |
| `OTEL_EXPORTER_TYPE` | Tracing exporter type | `otlp` |

## Development

Each extension can be built and run independently:

```bash
cd extensions/<extension-name>
npm install
npm run build
npm start
```

For development with auto-reload:
```bash
npm run dev
```
