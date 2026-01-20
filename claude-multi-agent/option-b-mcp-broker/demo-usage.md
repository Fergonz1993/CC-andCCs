# MCP Server Coordination Demo - Usage Guide

This guide walks you through using the MCP Server (Option B) for multi-agent Claude Code coordination.

## Overview

The MCP Server acts as a broker that enables multiple Claude Code instances to:
- Coordinate on shared tasks
- Claim and complete work items
- Share context and artifacts
- Communicate via messages

## Prerequisites

- Node.js 18+ installed
- Claude Code CLI installed (`claude` command available)
- This repository cloned locally

## Step 1: Install Dependencies

```bash
cd <project-root>/option-b-mcp-broker
npm install
```

## Step 2: Build the Server

```bash
npm run build
```

This compiles the TypeScript to JavaScript in the `dist/` folder.

## Step 3: Configure Claude Code

Copy the example config to your Claude Code settings:

```bash
# Copy to your home directory Claude Code config
cp mcp-config.example.json ~/.claude/mcp.json

# Or merge with existing config if you have other MCP servers
```

Alternatively, add this to your existing `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "multi-agent-coordinator": {
      "command": "node",
      "args": ["<path-to-project>/option-b-mcp-broker/dist/index.js"],
      "env": {
        "COORDINATION_DIR": "<coordination-dir>"
      }
    }
  }
}
```

> **Note:** Replace `<path-to-project>` with your local repository path and `<coordination-dir>` with a directory for coordination state (e.g., `/tmp/claude-coordination` or a project-local `.coordination` folder).

## Step 4: Verify MCP Server is Available

Start a new Claude Code session and check that the tools are available:

```bash
claude
```

Then ask:
```
What MCP tools do you have available for coordination?
```

You should see tools like:
- `init_coordination`
- `create_task`
- `claim_task`
- `complete_task`
- `get_task_status`
- `send_message`
- `get_messages`
- `share_artifact`
- `get_artifacts`

## Step 5: Demo - Multi-Agent Task Coordination

### Terminal 1: Leader Agent

Open a terminal and start Claude Code:

```bash
claude
```

Initialize coordination and create tasks:

```
Initialize a coordination session called "api-refactor" with description "Refactoring the API layer for better performance". Then create three tasks:

1. Task: "analyze-endpoints" - Analyze all API endpoints and identify slow ones
2. Task: "optimize-queries" - Optimize database queries in identified endpoints
3. Task: "add-caching" - Add Redis caching layer for frequently accessed data

After creating the tasks, share an artifact with the project context.
```

Example leader prompts:

```
# Check overall progress
Get the status of all tasks in our coordination session.

# Send guidance to workers
Send a message to all agents: "Focus on the /api/users endpoint first - it has the highest traffic"

# Share context
Share an artifact called "architecture-notes" with content describing the current system architecture.

# Review completed work
Get all artifacts shared by workers to review their findings.
```

### Terminal 2: Worker Agent 1

Open a second terminal:

```bash
claude
```

Join and claim work:

```
Join the "api-refactor" coordination session. List available tasks and claim the "analyze-endpoints" task. Then check for any messages from the leader.
```

Example worker prompts:

```
# Claim a task
Claim the task "analyze-endpoints" and mark myself as worker-1.

# Check for instructions
Get all messages in our coordination session.

# Share findings
Share an artifact called "endpoint-analysis" with my findings about slow endpoints.

# Complete the task
Complete the task "analyze-endpoints" with a summary of what was found.

# Look for more work
Get task status to see what other tasks are available to claim.
```

### Terminal 3: Worker Agent 2

Open a third terminal:

```bash
claude
```

```
Join the "api-refactor" coordination session as worker-2. Check what tasks are available and claim one that hasn't been taken.
```

## Available MCP Tools Reference

### `init_coordination`
Initialize a new coordination session.

**Parameters:**
- `session_id` (string): Unique identifier for the session
- `description` (string): Description of the coordination goal

**Example:**
```
Initialize coordination with session_id "my-project" and description "Building a new feature"
```

### `create_task`
Create a new task in the coordination session.

**Parameters:**
- `session_id` (string): The coordination session ID
- `task_id` (string): Unique task identifier
- `description` (string): What the task involves
- `dependencies` (array, optional): List of task IDs this depends on

**Example:**
```
Create a task with id "implement-auth" in session "my-project" with description "Implement JWT authentication"
```

### `claim_task`
Claim a task for this agent to work on.

**Parameters:**
- `session_id` (string): The coordination session ID
- `task_id` (string): The task to claim
- `agent_id` (string): Identifier for this agent

**Example:**
```
Claim task "implement-auth" in session "my-project" as agent "worker-1"
```

### `complete_task`
Mark a task as completed.

**Parameters:**
- `session_id` (string): The coordination session ID
- `task_id` (string): The task to complete
- `result` (string): Summary of what was accomplished

**Example:**
```
Complete task "implement-auth" with result "Implemented JWT auth with refresh tokens, added middleware"
```

### `get_task_status`
Get status of all tasks or a specific task.

**Parameters:**
- `session_id` (string): The coordination session ID
- `task_id` (string, optional): Specific task to check

**Example:**
```
Get task status for session "my-project"
```

### `send_message`
Send a message to other agents.

**Parameters:**
- `session_id` (string): The coordination session ID
- `from_agent` (string): Sender identifier
- `message` (string): The message content
- `to_agent` (string, optional): Specific recipient (broadcast if omitted)

**Example:**
```
Send message from "leader" to all agents: "Please prioritize security tasks"
```

### `get_messages`
Retrieve messages from the coordination session.

**Parameters:**
- `session_id` (string): The coordination session ID
- `since_timestamp` (number, optional): Only get messages after this time

**Example:**
```
Get all messages for session "my-project"
```

### `share_artifact`
Share a file or data artifact with other agents.

**Parameters:**
- `session_id` (string): The coordination session ID
- `artifact_id` (string): Unique identifier for the artifact
- `content` (string): The artifact content
- `metadata` (object, optional): Additional metadata

**Example:**
```
Share artifact "api-schema" with the OpenAPI schema content
```

### `get_artifacts`
Retrieve shared artifacts.

**Parameters:**
- `session_id` (string): The coordination session ID
- `artifact_id` (string, optional): Specific artifact to retrieve

**Example:**
```
Get all artifacts from session "my-project"
```

## Coordination Patterns

### Pattern 1: Leader-Worker

1. Leader initializes session and creates tasks
2. Workers claim tasks and work independently
3. Workers share artifacts with findings
4. Leader reviews and sends follow-up messages
5. Workers complete tasks with summaries

### Pattern 2: Pipeline

1. Create tasks with dependencies
2. First worker completes initial task
3. Next worker claims dependent task (now unblocked)
4. Artifacts flow through the pipeline

### Pattern 3: Collaborative Review

1. Worker shares artifact for review
2. Other workers get artifacts and send feedback messages
3. Original worker updates artifact based on feedback
4. Task is completed when consensus reached

## Troubleshooting

### MCP tools not showing up

1. Verify the config path is correct in `~/.claude/mcp.json`
2. Check that the server was built: `ls dist/index.js`
3. Restart Claude Code to reload MCP config

### Coordination state not persisting

Check that the `COORDINATION_DIR` exists and is writable:

```bash
mkdir -p /tmp/claude-coordination
ls -la /tmp/claude-coordination
```

### Multiple sessions interfering

Use unique `session_id` values for different coordination efforts.

## File Locations

- MCP Server source: `<repo-root>/option-b-mcp-broker/src/`
- Built server: `<repo-root>/option-b-mcp-broker/dist/index.js`
- Coordination data: `${COORDINATION_DIR}` (configurable via environment variable, defaults to `.coordination`)
- Claude config: `~/.claude/mcp.json` (or `${HOME}/.claude/mcp.json`)
