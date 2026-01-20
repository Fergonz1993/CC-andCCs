# MCP Tools Reference

This document provides detailed reference for all Model Context Protocol (MCP) tools provided by the Option B coordination server.

## Overview

The MCP server provides tools organized into three categories:
- **Leader Tools**: For task creation and coordination management
- **Worker Tools**: For claiming and executing tasks
- **Shared Tools**: For discoveries and status checks

## Connection Setup

Add to your Claude Code MCP configuration (`~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "multi-agent-coordinator": {
      "command": "node",
      "args": ["/path/to/option-b-mcp-broker/dist/index.js"],
      "env": {
        "COORDINATION_DIR": "/path/to/working/directory"
      }
    }
  }
}
```

## Leader Tools

### init_coordination

Initialize a new coordination session with a goal and master plan.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "goal": {
      "type": "string",
      "description": "The overall project goal"
    },
    "master_plan": {
      "type": "string",
      "description": "High-level plan/approach"
    }
  },
  "required": ["goal"]
}
```

**Example:**
```json
{
  "goal": "Build a REST API for user management",
  "master_plan": "1. Design data models\n2. Implement endpoints\n3. Add authentication"
}
```

**Response:**
```json
{
  "success": true,
  "goal": "Build a REST API for user management",
  "created_at": "2024-01-15T10:30:00.000Z"
}
```

---

### create_task

Create a new task in the queue.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "description": {
      "type": "string",
      "description": "Task description"
    },
    "priority": {
      "type": "number",
      "description": "Priority 1-10 (lower = higher priority)",
      "default": 5
    },
    "dependencies": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Task IDs that must complete first"
    },
    "context_files": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Relevant file paths"
    },
    "hints": {
      "type": "string",
      "description": "Hints for the worker"
    }
  },
  "required": ["description"]
}
```

**Example:**
```json
{
  "description": "Implement user login endpoint",
  "priority": 1,
  "dependencies": ["task-xxx-001"],
  "context_files": ["src/auth/login.ts"],
  "hints": "Use JWT for tokens, bcrypt for passwords"
}
```

**Response:**
```json
{
  "success": true,
  "task": {
    "id": "task-xxx-002",
    "description": "Implement user login endpoint",
    "status": "available",
    "priority": 1,
    "dependencies": ["task-xxx-001"],
    "context": {
      "files": ["src/auth/login.ts"],
      "hints": "Use JWT for tokens, bcrypt for passwords"
    },
    "created_at": "2024-01-15T10:35:00.000Z"
  }
}
```

---

### create_tasks_batch

Create multiple tasks at once.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "tasks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": { "type": "string" },
          "priority": { "type": "number" },
          "dependencies": { "type": "array", "items": { "type": "string" } },
          "context_files": { "type": "array", "items": { "type": "string" } },
          "hints": { "type": "string" }
        },
        "required": ["description"]
      }
    }
  },
  "required": ["tasks"]
}
```

**Example:**
```json
{
  "tasks": [
    { "description": "Create User model", "priority": 1 },
    { "description": "Create login endpoint", "priority": 2, "dependencies": ["task-1"] },
    { "description": "Create logout endpoint", "priority": 3, "dependencies": ["task-2"] }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "created": 3,
  "task_ids": ["task-xxx-001", "task-xxx-002", "task-xxx-003"]
}
```

---

### get_status

Get current coordination status and progress.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {}
}
```

**Response:**
```json
{
  "goal": "Build a REST API",
  "tasks": {
    "available": 5,
    "claimed": 1,
    "in_progress": 2,
    "done": 10,
    "failed": 0
  },
  "total_tasks": 18,
  "progress_percent": 55,
  "agents": {
    "total": 3,
    "leaders": 1,
    "workers": 2,
    "active": 3
  },
  "discoveries_count": 7,
  "last_activity": "2024-01-15T11:00:00.000Z"
}
```

---

### get_all_tasks

Get all tasks with their current status.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "status_filter": {
      "type": "string",
      "enum": ["all", "available", "claimed", "in_progress", "done", "failed"],
      "default": "all"
    }
  }
}
```

**Response:**
```json
{
  "tasks": [
    {
      "id": "task-xxx-001",
      "description": "Create User model",
      "status": "done",
      "priority": 1,
      "claimed_by": "terminal-2",
      "result": { "output": "Created User model with validation" }
    },
    ...
  ]
}
```

---

### get_results

Get results from completed tasks.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "task_ids": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Specific task IDs (empty for all)"
    }
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "task_id": "task-xxx-001",
      "description": "Create User model",
      "result": {
        "output": "Created User model with validation",
        "files_created": ["src/models/user.ts"]
      },
      "completed_at": "2024-01-15T10:45:00.000Z"
    }
  ]
}
```

---

## Worker Tools

### register_agent

Register as a worker or leader agent.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "agent_id": {
      "type": "string",
      "description": "Unique agent identifier (e.g., terminal-2)"
    },
    "role": {
      "type": "string",
      "enum": ["leader", "worker"],
      "description": "Agent role"
    }
  },
  "required": ["agent_id", "role"]
}
```

**Example:**
```json
{
  "agent_id": "terminal-2",
  "role": "worker"
}
```

**Response:**
```json
{
  "success": true,
  "agent": {
    "id": "terminal-2",
    "role": "worker",
    "last_heartbeat": "2024-01-15T10:30:00.000Z",
    "current_task": null,
    "tasks_completed": 0
  }
}
```

---

### claim_task

Claim an available task to work on.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "agent_id": {
      "type": "string",
      "description": "Your agent ID"
    }
  },
  "required": ["agent_id"]
}
```

**Response (success):**
```json
{
  "success": true,
  "task": {
    "id": "task-xxx-002",
    "description": "Implement login endpoint",
    "status": "claimed",
    "priority": 1,
    "claimed_by": "terminal-2",
    "claimed_at": "2024-01-15T10:35:00.000Z",
    "context": {
      "files": ["src/auth/login.ts"],
      "hints": "Use JWT"
    }
  }
}
```

**Response (no tasks available):**
```json
{
  "success": false,
  "message": "No available tasks with satisfied dependencies"
}
```

---

### start_task

Mark a claimed task as in_progress.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "agent_id": {
      "type": "string",
      "description": "Your agent ID"
    },
    "task_id": {
      "type": "string",
      "description": "Task ID to start"
    }
  },
  "required": ["agent_id", "task_id"]
}
```

**Response:**
```json
{
  "success": true
}
```

---

### complete_task

Mark a task as completed with results.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "agent_id": {
      "type": "string",
      "description": "Your agent ID"
    },
    "task_id": {
      "type": "string",
      "description": "Task ID"
    },
    "output": {
      "type": "string",
      "description": "Summary of what was done"
    },
    "files_modified": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Files that were modified"
    },
    "files_created": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Files that were created"
    }
  },
  "required": ["agent_id", "task_id", "output"]
}
```

**Example:**
```json
{
  "agent_id": "terminal-2",
  "task_id": "task-xxx-002",
  "output": "Implemented login endpoint with JWT authentication",
  "files_modified": ["src/routes/index.ts"],
  "files_created": ["src/auth/login.ts", "src/auth/jwt.ts"]
}
```

**Response:**
```json
{
  "success": true
}
```

---

### fail_task

Mark a task as failed.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "agent_id": {
      "type": "string",
      "description": "Your agent ID"
    },
    "task_id": {
      "type": "string",
      "description": "Task ID"
    },
    "error": {
      "type": "string",
      "description": "Error description"
    }
  },
  "required": ["agent_id", "task_id", "error"]
}
```

**Response:**
```json
{
  "success": true
}
```

---

### heartbeat

Send heartbeat to indicate agent is still active.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "agent_id": {
      "type": "string",
      "description": "Your agent ID"
    }
  },
  "required": ["agent_id"]
}
```

**Response:**
```json
{
  "success": true
}
```

---

## Shared Tools

### add_discovery

Share a discovery or important finding with other agents.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "agent_id": {
      "type": "string",
      "description": "Your agent ID"
    },
    "content": {
      "type": "string",
      "description": "The discovery content"
    },
    "tags": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Tags for categorization"
    }
  },
  "required": ["agent_id", "content"]
}
```

**Example:**
```json
{
  "agent_id": "terminal-2",
  "content": "Found existing auth middleware in src/middleware/auth.ts - can be reused",
  "tags": ["auth", "existing-code"]
}
```

**Response:**
```json
{
  "success": true,
  "discovery": {
    "id": "uuid-xxx",
    "agent_id": "terminal-2",
    "content": "Found existing auth middleware...",
    "tags": ["auth", "existing-code"],
    "created_at": "2024-01-15T10:40:00.000Z"
  }
}
```

---

### get_discoveries

Get shared discoveries from all agents.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "tags": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Filter by tags"
    },
    "limit": {
      "type": "number",
      "description": "Max number of discoveries",
      "default": 20
    }
  }
}
```

**Response:**
```json
{
  "discoveries": [
    {
      "id": "uuid-xxx",
      "agent_id": "terminal-2",
      "content": "Found existing auth middleware...",
      "tags": ["auth", "existing-code"],
      "created_at": "2024-01-15T10:40:00.000Z"
    }
  ]
}
```

---

### get_master_plan

Get the master plan and goal.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {}
}
```

**Response:**
```json
{
  "goal": "Build a REST API for user management",
  "master_plan": "1. Design data models\n2. Implement endpoints...",
  "created_at": "2024-01-15T10:30:00.000Z"
}
```

---

## MCP Resources

The server also exposes resources for reading state:

| URI | Description |
|-----|-------------|
| `coordination://status` | Current coordination status |
| `coordination://tasks` | All tasks in the queue |
| `coordination://discoveries` | Shared discoveries |
| `coordination://master-plan` | Master plan and goal |

Resources can be read using the MCP resource reading capability.

## Error Handling

All tools return errors in a consistent format:

```json
{
  "error": "Error message describing what went wrong"
}
```

Common errors:
- `"Unknown tool: xxx"` - Tool name not recognized
- `"No available tasks with satisfied dependencies"` - No claimable tasks
- Task not found or not claimed by the requesting agent
