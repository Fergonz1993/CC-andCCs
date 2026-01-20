# OPTION B: MCP Server Broker - Detailed Plan

## Executive Summary

Option B uses the **Model Context Protocol (MCP)** to create a shared coordination server that all Claude Code instances connect to. The server provides tools for task management, and Claude Code instances call these tools directly. Real-time, typed, and native to Claude Code.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCP COORDINATION SERVER                           │
│                         (Node.js process via stdio)                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         IN-MEMORY STATE                              │   │
│  │                                                                      │   │
│  │  CoordinationState {                                                 │   │
│  │    goal: string                                                      │   │
│  │    master_plan: string                                               │   │
│  │    tasks: Task[]                                                     │   │
│  │    agents: Map<string, Agent>                                        │   │
│  │    discoveries: Discovery[]                                          │   │
│  │  }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ init_coord  │ │ create_task │ │ claim_task  │ │ complete_task       │   │
│  │ Tool        │ │ Tool        │ │ Tool        │ │ Tool                │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ get_status  │ │ add_disc    │ │ get_tasks   │ │ register_agent      │   │
│  │ Tool        │ │ Tool        │ │ Tool        │ │ Tool                │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────┘   │
│                                                                             │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
     ┌──────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐
     │ Terminal 1  │    │ Terminal 2  │    │ Terminal 3  │
     │             │    │             │    │             │
     │ Claude Code │    │ Claude Code │    │ Claude Code │
     │     +       │    │     +       │    │     +       │
     │ MCP Client  │    │ MCP Client  │    │ MCP Client  │
     │             │    │             │    │             │
     │  LEADER     │    │  WORKER     │    │  WORKER     │
     └─────────────┘    └─────────────┘    └─────────────┘
```

---

## How MCP Works

### What is MCP?

The **Model Context Protocol** is Anthropic's standard for extending Claude with external tools and resources. Claude Code has built-in MCP support.

### MCP Server Lifecycle

```
1. Claude Code starts
2. Reads ~/.claude/mcp.json for server configs
3. Spawns MCP server as child process
4. Communicates via stdin/stdout (JSON-RPC)
5. Server provides tools that Claude can call
6. Tools execute and return results
```

### Our Server Architecture

```typescript
// Server provides these capabilities
{
  capabilities: {
    tools: {},      // Functions Claude can call
    resources: {}   // Data Claude can read
  }
}
```

---

## Tool Reference

### Leader Tools

#### `init_coordination`
Initialize a new coordination session.

```typescript
// Input
{
  goal: string,           // "Build a REST API for user management"
  master_plan?: string    // Optional high-level approach
}

// Output
{
  success: true,
  goal: "Build a REST API...",
  created_at: "2025-01-19T20:00:00Z"
}
```

#### `create_task`
Add a single task to the queue.

```typescript
// Input
{
  description: string,      // "Create User model with Prisma"
  priority?: number,        // 1-10, default 5
  dependencies?: string[],  // ["task-abc123"]
  context_files?: string[], // ["src/models/"]
  hints?: string           // "Use TypeScript"
}

// Output
{
  success: true,
  task: {
    id: "task-xyz789",
    description: "Create User model...",
    status: "available",
    priority: 1,
    ...
  }
}
```

#### `create_tasks_batch`
Add multiple tasks at once.

```typescript
// Input
{
  tasks: [
    { description: "Task 1", priority: 1 },
    { description: "Task 2", priority: 2, dependencies: ["task-1"] },
    ...
  ]
}

// Output
{
  success: true,
  created: 5,
  task_ids: ["task-abc", "task-def", ...]
}
```

#### `get_status`
Get coordination progress summary.

```typescript
// Input: none

// Output
{
  goal: "Build a REST API...",
  tasks: {
    available: 2,
    claimed: 1,
    in_progress: 1,
    done: 3,
    failed: 0
  },
  total_tasks: 7,
  progress_percent: 43,
  agents: {
    total: 3,
    leaders: 1,
    workers: 2,
    active: 3
  },
  discoveries_count: 2,
  last_activity: "2025-01-19T20:15:00Z"
}
```

#### `get_all_tasks`
Get all tasks with optional status filter.

```typescript
// Input
{
  status_filter?: "all" | "available" | "claimed" | "in_progress" | "done" | "failed"
}

// Output
{
  tasks: [
    { id: "task-001", description: "...", status: "done", ... },
    { id: "task-002", description: "...", status: "in_progress", ... },
    ...
  ]
}
```

#### `get_results`
Get results from completed tasks.

```typescript
// Input
{
  task_ids?: string[]  // Empty for all completed tasks
}

// Output
{
  results: [
    {
      task_id: "task-001",
      description: "Create User model",
      result: {
        output: "Implemented User model with...",
        files_modified: ["prisma/schema.prisma"],
        files_created: ["src/models/user.ts"]
      },
      completed_at: "2025-01-19T20:10:00Z"
    },
    ...
  ]
}
```

### Worker Tools

#### `register_agent`
Register as a participant in the coordination.

```typescript
// Input
{
  agent_id: string,            // "terminal-2"
  role: "leader" | "worker"
}

// Output
{
  success: true,
  agent: {
    id: "terminal-2",
    role: "worker",
    last_heartbeat: "2025-01-19T20:00:00Z",
    current_task: null,
    tasks_completed: 0
  }
}
```

#### `claim_task`
Claim an available task.

```typescript
// Input
{
  agent_id: string  // "terminal-2"
}

// Output (success)
{
  success: true,
  task: {
    id: "task-xyz",
    description: "...",
    status: "claimed",
    claimed_by: "terminal-2",
    priority: 1,
    context: { files: [...], hints: "..." }
  }
}

// Output (no tasks available)
{
  success: false,
  message: "No available tasks with satisfied dependencies"
}
```

#### `start_task`
Mark a claimed task as in progress.

```typescript
// Input
{
  agent_id: string,
  task_id: string
}

// Output
{
  success: true
}
```

#### `complete_task`
Mark a task as done with results.

```typescript
// Input
{
  agent_id: string,
  task_id: string,
  output: string,              // Summary of what was done
  files_modified?: string[],
  files_created?: string[]
}

// Output
{
  success: true
}
```

#### `fail_task`
Mark a task as failed.

```typescript
// Input
{
  agent_id: string,
  task_id: string,
  error: string  // What went wrong
}

// Output
{
  success: true
}
```

#### `heartbeat`
Signal that agent is still active.

```typescript
// Input
{
  agent_id: string
}

// Output
{
  success: true
}
```

### Shared Tools

#### `add_discovery`
Share an important finding.

```typescript
// Input
{
  agent_id: string,
  content: string,      // The discovery
  tags?: string[]       // ["database", "security"]
}

// Output
{
  success: true,
  discovery: {
    id: "disc-abc123",
    agent_id: "terminal-2",
    content: "Found that...",
    tags: ["database"],
    created_at: "2025-01-19T20:10:00Z"
  }
}
```

#### `get_discoveries`
Get shared discoveries.

```typescript
// Input
{
  tags?: string[],  // Filter by tags
  limit?: number    // Max results, default 20
}

// Output
{
  discoveries: [
    {
      id: "disc-abc",
      agent_id: "terminal-2",
      content: "...",
      tags: ["database"],
      created_at: "..."
    },
    ...
  ]
}
```

#### `get_master_plan`
Get the goal and plan.

```typescript
// Input: none

// Output
{
  goal: "Build a REST API...",
  master_plan: "We'll use Express with TypeScript...",
  created_at: "2025-01-19T20:00:00Z"
}
```

---

## Resource Reference

MCP also supports **resources** - read-only data endpoints.

| URI | Description |
|-----|-------------|
| `coordination://status` | Current status (same as get_status tool) |
| `coordination://tasks` | All tasks |
| `coordination://discoveries` | All discoveries |
| `coordination://master-plan` | Goal and plan |

Claude can read these like:
```
Read the resource at coordination://status
```

---

## Setup Instructions

### Step 1: Build the Server

```bash
cd option-b-mcp-broker
npm install
npm run build
```

This creates `dist/index.js`.

### Step 2: Configure Claude Code

Edit `~/.claude/mcp.json` (create if doesn't exist):

```json
{
  "mcpServers": {
    "coordination": {
      "command": "node",
      "args": ["/Users/fernandogonzalez/Code-and-development/CC-and-CCs/claude-multi-agent/option-b-mcp-broker/dist/index.js"]
    }
  }
}
```

### Step 3: Restart Claude Code

Each terminal running Claude Code will now:
1. Spawn the MCP server
2. Have access to all coordination tools

**STATE PERSISTENCE**: ✅ **IMPLEMENTED**

The server now automatically:
- Loads state from `.coordination/mcp-state.json` on startup
- Saves state after every mutation (task creation, claiming, completion, etc.)
- All terminals share the same state file

You can customize the state directory with the `COORDINATION_DIR` environment variable:
```json
{
  "mcpServers": {
    "coordination": {
      "command": "node",
      "args": ["/path/to/dist/index.js"],
      "env": {
        "COORDINATION_DIR": "/custom/path/.coordination"
      }
    }
  }
}
```

### Step 4: Verify Setup

In Claude Code, try:
```
Call the get_status coordination tool
```

If configured correctly, you'll see the status output.

---

## Detailed Workflow

### Phase 1: Initialization (Terminal 1 - Leader)

**Step 1.1: Register as leader**
```
Call register_agent with agent_id="leader" and role="leader"
```

**Step 1.2: Initialize coordination**
```
Call init_coordination with goal="Build a REST API for user management"
and master_plan="We'll use Express.js with TypeScript, Prisma for ORM,
and JWT for authentication."
```

**Step 1.3: Create tasks**
```
Call create_tasks_batch with tasks=[
  {
    "description": "Set up Express.js project with TypeScript config",
    "priority": 1,
    "context_files": ["package.json", "tsconfig.json"],
    "hints": "Use express, typescript, ts-node-dev"
  },
  {
    "description": "Create Prisma schema with User model",
    "priority": 1,
    "context_files": ["prisma/schema.prisma"],
    "hints": "Fields: id, email, passwordHash, name, createdAt, updatedAt"
  },
  {
    "description": "Implement JWT authentication utilities",
    "priority": 2,
    "dependencies": [],
    "context_files": ["src/utils/auth.ts"],
    "hints": "generateToken, verifyToken, hashPassword, comparePassword"
  },
  {
    "description": "Create auth routes: POST /login, POST /register",
    "priority": 2,
    "context_files": ["src/routes/auth.ts"],
    "hints": "Validate input, return JWT on success"
  },
  {
    "description": "Write tests for authentication flow",
    "priority": 3,
    "context_files": ["src/__tests__/auth.test.ts"],
    "hints": "Test login, register, invalid credentials"
  }
]
```

**Step 1.4: Monitor**
```
Call get_status to see current progress
```

### Phase 2: Worker Execution (Terminals 2 & 3)

**Step 2.1: Register as worker**
```
Call register_agent with agent_id="worker-2" and role="worker"
```

**Step 2.2: Get context**
```
Call get_master_plan to understand the goal
Call get_discoveries to see any shared knowledge
```

**Step 2.3: Claim a task**
```
Call claim_task with agent_id="worker-2"
```

Server automatically:
- Finds highest-priority available task
- Checks dependencies are satisfied
- Marks it as claimed

**Step 2.4: Start working**
```
Call start_task with agent_id="worker-2" and task_id="task-xxx"
```

**Step 2.5: Do the work**

Worker actually implements the task using Claude Code's normal capabilities.

**Step 2.6: Share discoveries**
```
Call add_discovery with agent_id="worker-2" and
content="The existing codebase uses ESM modules, not CommonJS.
All imports should use 'import' syntax." and tags=["typescript", "setup"]
```

**Step 2.7: Complete the task**
```
Call complete_task with agent_id="worker-2", task_id="task-xxx",
output="Created Express.js project with TypeScript. Added dev dependencies
including ts-node-dev for hot reloading. Configured tsconfig for ES2020 target.",
files_created=["package.json", "tsconfig.json", "src/index.ts"]
```

**Step 2.8: Claim next task**

Repeat from Step 2.3.

### Phase 3: Completion (Leader)

**Step 3.1: Check progress**
```
Call get_status
```

**Step 3.2: Get all results**
```
Call get_results
```

**Step 3.3: Review discoveries**
```
Call get_discoveries
```

**Step 3.4: Aggregate and finalize**

Leader reviews all work, runs final tests, makes any needed adjustments.

---

## State Persistence (Required Modification)

To make multiple terminals share state, add file persistence:

```typescript
// Add to src/index.ts

const STATE_FILE = '.coordination/mcp-state.json';

function loadState(): CoordinationState {
  try {
    if (fs.existsSync(STATE_FILE)) {
      return JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8'));
    }
  } catch (e) {
    console.error('Failed to load state:', e);
  }
  return initialState();
}

function saveState(): void {
  fs.mkdirSync(path.dirname(STATE_FILE), { recursive: true });
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

// Call saveState() after every mutation
function createTask(...) {
  // ... existing code ...
  saveState();
  return task;
}
```

---

## Tool Invocation Examples

### From Claude Code

Claude Code will naturally invoke tools when you describe what you want:

```
You: "Create a new task to implement the login endpoint with priority 2"

Claude: I'll create that task for you.
[Calls create_task tool with appropriate parameters]
Task created: task-abc123
```

### Explicit Tool Calls

You can also be explicit:

```
You: "Use the create_task tool to add: description='Implement logout', priority=3"

Claude: [Calls create_task with exact parameters]
```

---

## Advantages

1. **Native integration** - Uses Claude Code's built-in MCP support
2. **Type-safe** - JSON schema validation on all inputs
3. **Real-time** - No polling, immediate responses
4. **Extensible** - Easy to add new tools
5. **Discoverable** - Claude can list available tools

## Limitations

1. **Server per terminal** - Default setup spawns separate server per terminal
2. **State sharing requires modification** - Need to add persistence
3. **No built-in auth** - All terminals have equal access
4. **Memory-only by default** - State lost on restart

---

## File Listing

```
option-b-mcp-broker/
├── package.json           # Dependencies
├── tsconfig.json          # TypeScript config
├── src/
│   └── index.ts           # Main MCP server (900+ lines)
├── dist/
│   └── index.js           # Compiled JavaScript
├── mcp-config.example.json # Example Claude Code config
└── demo-usage.md          # Usage guide
```

---

## Comparison with Option A

| Aspect | Option A (Files) | Option B (MCP) |
|--------|------------------|----------------|
| Latency | 1-2s polling | Instant |
| Setup | None | Build + config |
| State visibility | Human-readable files | JSON responses |
| Debugging | cat/grep files | Tool call logs |
| Extension | Edit Python | Edit TypeScript |
| Multi-machine | Via shared filesystem | Via network (with mods) |
