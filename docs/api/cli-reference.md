# CLI Reference

This document provides complete reference for all command-line interfaces in the Claude Multi-Agent Coordination System.

## Option A: File-Based Coordination CLI

The `coordination.py` script provides commands for both leaders and workers.

### Leader Commands

#### Initialize Coordination

```bash
python coordination.py leader init <goal> [--approach <approach>]
```

Creates the `.coordination/` directory structure and initializes the master plan.

**Arguments:**
- `goal` (required): The overall project goal
- `--approach`: High-level approach description

**Example:**
```bash
python coordination.py leader init "Build a REST API" --approach "Use Express.js with TypeScript"
```

**Output:**
```
Created .coordination/ directory structure
Written: .coordination/master-plan.md
```

---

#### Add Task

```bash
python coordination.py leader add-task <description> [options]
```

Add a new task to the queue.

**Arguments:**
- `description` (required): Task description

**Options:**
- `-p, --priority <int>`: Priority 1-10 (default: 5, lower = higher priority)
- `-d, --depends <task-ids>`: Space-separated dependency task IDs
- `-f, --files <paths>`: Space-separated relevant file paths
- `--hints <text>`: Hints for the worker

**Examples:**
```bash
# Simple task
python coordination.py leader add-task "Create User model" -p 1

# Task with dependencies
python coordination.py leader add-task "Implement login" -p 2 -d task-001 task-002

# Task with context
python coordination.py leader add-task "Add validation" -p 3 -f src/models/user.ts --hints "Use zod"
```

**Output:**
```
Added task: task-20240115-a1b2
  Priority: 1
  Dependencies: []
```

---

#### Show Status

```bash
python coordination.py leader status
```

Display current coordination status.

**Output:**
```
============================================================
COORDINATION STATUS
============================================================

AVAILABLE (3):
  [1] task-001: Create User model...
  [2] task-002: Create login endpoint...
  [3] task-003: Add validation...

IN_PROGRESS (1):
  [2] task-004: Implement JWT... [by terminal-2]

DONE (5):
  [1] task-005: Setup project...
  [2] task-006: Create database schema...
  ...

------------------------------------------------------------
Progress: 5/9 tasks complete (55%)
============================================================
```

---

#### Aggregate Results

```bash
python coordination.py leader aggregate
```

Aggregate all completed task results into a summary file.

**Output:**
```
Aggregated 5 results into .coordination/summary.md
```

---

### Worker Commands

#### Register Agent

```bash
python coordination.py register --agent-id <id> [--capabilities <caps>]
```

Register an agent in the coordination system.

**Arguments:**
- `--agent-id` (required): Unique agent identifier
- `--capabilities`: Comma-separated capabilities

**Example:**
```bash
python coordination.py register --agent-id terminal-2 --capabilities "python,testing"
```

---

#### Claim Task

```bash
python coordination.py worker claim <terminal-id>
```

Claim the highest priority available task.

**Arguments:**
- `terminal-id` (required): Your terminal identifier

**Example:**
```bash
python coordination.py worker claim terminal-2
```

**Output:**
```
Claimed task: task-20240115-a1b2
  Description: Create User model
  Priority: 1
  Context: {"files": ["src/models/user.ts"], "hints": "Use TypeScript"}
```

---

#### Start Task

```bash
python coordination.py worker start <terminal-id> <task-id>
```

Mark a claimed task as in_progress.

**Arguments:**
- `terminal-id`: Your terminal identifier
- `task-id`: The task ID to start

**Example:**
```bash
python coordination.py worker start terminal-2 task-20240115-a1b2
```

---

#### Complete Task

```bash
python coordination.py worker complete <terminal-id> <task-id> <output> [options]
```

Mark a task as completed.

**Arguments:**
- `terminal-id`: Your terminal identifier
- `task-id`: The task ID
- `output`: Summary of what was done

**Options:**
- `-m, --modified <files>`: Space-separated modified files
- `-c, --created <files>`: Space-separated created files

**Example:**
```bash
python coordination.py worker complete terminal-2 task-001 "Created User model with validation" \
  -c src/models/user.ts -m src/index.ts
```

**Output:**
```
Completed task: task-20240115-a1b2
Written: .coordination/results/task-20240115-a1b2.md
```

---

#### Fail Task

```bash
python coordination.py worker fail <terminal-id> <task-id> <reason>
```

Mark a task as failed.

**Arguments:**
- `terminal-id`: Your terminal identifier
- `task-id`: The task ID
- `reason`: Failure reason

**Example:**
```bash
python coordination.py worker fail terminal-2 task-001 "Missing dependency: bcrypt not installed"
```

---

#### List Available Tasks

```bash
python coordination.py worker list
```

List all tasks available for claiming.

**Output:**
```
Available tasks (3):
  [1] task-001: Create User model...
  [2] task-002: Create login endpoint...
  [3] task-003: Add validation...
```

---

## Option C: Orchestrator CLI

The `orchestrate` command provides full programmatic control.

### Run Orchestration

```bash
orchestrate run <goal> [options]
```

Run the full orchestration flow.

**Arguments:**
- `goal` (required): The overall project goal

**Options:**
- `-w, --workers <n>`: Number of worker agents (default: 3)
- `--no-plan`: Skip leader planning, use predefined tasks
- `--tasks <file>`: JSON file with predefined tasks
- `--timeout <seconds>`: Task timeout (default: 600)
- `--model <name>`: Claude model to use

**Examples:**
```bash
# Full automatic mode
orchestrate run "Build a REST API" -w 3

# With predefined tasks
orchestrate run "Build API" --no-plan --tasks tasks.json

# With custom timeout
orchestrate run "Complex refactoring" -w 5 --timeout 900
```

---

### Initialize Config

```bash
orchestrate init <goal>
```

Create an orchestration configuration file.

**Example:**
```bash
orchestrate init "Add authentication feature"
```

Creates `orchestration.json` with:
```json
{
  "goal": "Add authentication feature",
  "max_workers": 3,
  "model": "claude-sonnet-4-20250514",
  "tasks": []
}
```

---

### Create Task

```bash
orchestrate create-task <description> [options]
```

Add a task to the configuration.

**Options:**
- `-p, --priority <n>`: Priority 1-10
- `-d, --depends <ids>`: Dependency task IDs

**Example:**
```bash
orchestrate create-task "Create JWT utility" -p 1
orchestrate create-task "Add auth middleware" -p 2 -d task-1
```

---

### Run from Config

```bash
orchestrate from-config <config-file>
```

Run orchestration from a configuration file.

**Example:**
```bash
orchestrate from-config orchestration.json
```

---

### Status

```bash
orchestrate status
```

Show current orchestration status (if running).

---

## Environment Variables

### Option A

| Variable | Description | Default |
|----------|-------------|---------|
| `COORDINATION_DIR` | Directory for coordination files | `.coordination` |

### Option B

| Variable | Description | Default |
|----------|-------------|---------|
| `COORDINATION_DIR` | Directory for state file | `.coordination` |

### Option C

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_MODEL` | Claude model to use | `claude-sonnet-4-20250514` |
| `MAX_WORKERS` | Default worker count | `3` |
| `TASK_TIMEOUT` | Default task timeout | `600` |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Task not found |
| 4 | Claim failed (race condition) |
| 5 | Timeout |

---

## Task File Format

For predefined tasks (`--tasks` option):

```json
[
  {
    "description": "Create User model",
    "priority": 1
  },
  {
    "description": "Create login endpoint",
    "priority": 2,
    "dependencies": ["task-1"],
    "context_files": ["src/auth/"],
    "hints": "Use JWT for tokens"
  }
]
```
