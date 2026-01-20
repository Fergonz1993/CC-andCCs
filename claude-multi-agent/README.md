# Claude Multi-Agent Coordination System

Coordinate multiple Claude Code instances across terminals to work together on complex tasks. Terminal 1 leads, Terminals 2 and 3 follow and help.

## The Problem

Each Claude Code instance runs in isolation with no native inter-process communication. But they share the **same filesystem**, which becomes your coordination layer.

## Three Solutions

| Option | Complexity | Latency | Best For |
|--------|------------|---------|----------|
| **A. File-based** | Low | ~1-2s | Quick MVP, simple tasks |
| **B. MCP Broker** | Medium | Real-time | Production workflow |
| **C. Orchestrator** | High | Real-time | Full programmatic control |

---

## Quick Start

### Option A: File-Based (Simplest - Start Here)

```bash
cd option-a-file-based

# Terminal 1 (Leader)
python coordination.py leader init "Build a REST API"
python coordination.py leader add-task "Create User model" -p 1
python coordination.py leader add-task "Create login endpoint" -p 2
python coordination.py leader status

# Terminal 2 (Worker)
python coordination.py worker claim terminal-2
# ... do the work ...
python coordination.py worker complete terminal-2 task-xxx "Done!"

# Terminal 3 (Worker)
python coordination.py worker claim terminal-3
```

### Option B: MCP Server

```bash
cd option-b-mcp-broker
npm install
npm run build

# Add to your Claude Code MCP config
# Then all terminals share the same task queue via MCP tools
```

### Option C: External Orchestrator (Full Control)

```bash
cd option-c-orchestrator
pip install -e .

# Let leader plan automatically
orchestrate run "Build a user authentication system" -w 3

# Or with predefined tasks
orchestrate run "Build API" --no-plan --tasks tasks.json
```

---

## Architecture

```
                    ┌─────────────────────┐
                    │    Orchestrator     │
                    │  (Option C only)    │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Terminal 1   │     │  Terminal 2   │     │  Terminal 3   │
│    LEADER     │     │    WORKER     │     │    WORKER     │
│               │     │               │     │               │
│ • Plans work  │     │ • Claims task │     │ • Claims task │
│ • Creates     │     │ • Executes    │     │ • Executes    │
│   tasks       │     │ • Reports     │     │ • Reports     │
│ • Aggregates  │     │   results     │     │   results     │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Coordination     │
                    │     Layer         │
                    │                   │
                    │ • File-based (A)  │
                    │ • MCP Server (B)  │
                    │ • Orchestrator(C) │
                    └───────────────────┘
```

---

## Option A: File-Based Coordination

### How It Works

1. Leader writes tasks to `.coordination/tasks.json`
2. Workers poll for available tasks
3. Workers claim by updating task status
4. Workers write results to `.coordination/results/`
5. Leader aggregates completed work

### Directory Structure

```
.coordination/
├── master-plan.md      # Overall goal and approach
├── tasks.json          # Task queue (source of truth)
├── context/
│   └── discoveries.md  # Shared findings
├── logs/
│   ├── leader.log
│   ├── terminal-2.log
│   └── terminal-3.log
└── results/
    ├── task-001.md
    └── task-002.md
```

### Task Format

```json
{
  "tasks": [
    {
      "id": "task-001",
      "description": "Implement user validation",
      "status": "available",
      "claimed_by": null,
      "priority": 1,
      "dependencies": [],
      "context": {
        "files": ["src/models/user.ts"],
        "hints": "Use zod for validation"
      }
    }
  ]
}
```

### Prompts for Claude Code

**Terminal 1 (Leader)**:
```
You are the lead agent. Create .coordination/ folder with master-plan.md
and tasks.json. Write the overall project goal to master-plan.md. Break
work into discrete tasks in tasks.json with fields: id, description,
status (available/claimed/done), claimed_by, priority. Monitor the
results/ folder for completed work.
```

**Terminal 2 & 3 (Workers)**:
```
You are a worker agent (terminal-2). Read .coordination/master-plan.md
for context. Check tasks.json for tasks with status "available". Claim
one by updating status to "claimed" and claimed_by to "terminal-2".
Execute the task, write output to results/{task-id}.md, update status
to "done". Repeat until no tasks remain.
```

---

## Option B: MCP Server Broker

A real-time message broker that all Claude Code instances connect to.

### Tools Provided

| Tool | Description |
|------|-------------|
| `init_coordination` | Set goal and master plan |
| `create_task` | Add a task to the queue |
| `claim_task` | Claim an available task |
| `complete_task` | Mark task done with results |
| `get_status` | Get progress summary |
| `add_discovery` | Share findings |

### Setup

1. Build the server:
   ```bash
   cd option-b-mcp-broker
   npm install
   npm run build
   ```

2. Add to Claude Code config (`~/.claude/mcp.json`):
   ```json
   {
     "mcpServers": {
       "coordination": {
         "command": "node",
         "args": ["/path/to/option-b-mcp-broker/dist/index.js"]
       }
     }
   }
   ```

3. Restart Claude Code - all instances now share the queue.

---

## Option C: External Orchestrator

Full programmatic control over multiple Claude Code instances.

### Features

- Spawn and manage Claude Code processes
- Automatic task distribution
- Dependency tracking
- Result aggregation
- Progress monitoring
- State persistence

### Installation

```bash
cd option-c-orchestrator
pip install -e .
```

### CLI Usage

```bash
# Full automatic mode - leader plans, workers execute
orchestrate run "Build a REST API for user management" -w 3

# With predefined tasks
orchestrate run "Build API" --no-plan --tasks my-tasks.json

# Initialize config file
orchestrate init "Add authentication feature"
orchestrate create-task "Create JWT utility" -p 1
orchestrate create-task "Add auth middleware" -p 2 -d task-1
orchestrate from-config orchestration.json
```

### Programmatic Usage

```python
import asyncio
from orchestrator import Orchestrator

async def main():
    orch = Orchestrator(
        working_directory="./my-project",
        max_workers=3,
        model="claude-sonnet-4-20250514",
    )

    await orch.initialize("Build user authentication")

    # Add tasks
    orch.add_task("Create User model", priority=1)
    orch.add_task("Implement login endpoint", priority=2)
    orch.add_task("Write tests", priority=3, dependencies=["task-1"])

    # Run
    result = await orch.run_with_predefined_tasks()
    print(f"Completed: {result['tasks_completed']}")

asyncio.run(main())
```

### Leader-Driven Planning

```python
# Let the leader analyze and create tasks automatically
result = await orch.run_with_leader_planning()
```

The leader will:
1. Analyze the codebase
2. Break down the goal into tasks
3. Assign priorities and dependencies
4. Coordinate workers
5. Aggregate results

---

## Best Practices

### Task Design

- **Atomic**: Each task should be completable independently
- **Specific**: Clear description of what needs to be done
- **Bounded**: ~5-15 minutes of work per task
- **Contextual**: Include relevant file paths

### Avoiding Race Conditions

- Workers should claim specific tasks, not "next available"
- Re-read state after claiming to verify
- Use file locks (Option A) or the MCP server (Option B)

### Sharing Context

- Write discoveries to shared context files
- Include relevant findings in task results
- Update master plan as understanding evolves

---

## Comparison

| Feature | Option A | Option B | Option C |
|---------|----------|----------|----------|
| Setup time | 5 min | 15 min | 10 min |
| Real-time | No (polling) | Yes | Yes |
| Race condition handling | Manual | Built-in | Built-in |
| Programmatic control | Limited | Via MCP | Full |
| Process management | Manual | Manual | Automatic |
| Best for | Quick tests | Production | Automation |

---

## Examples

See the `examples/` directory in each option folder for complete working examples.

---

## Troubleshooting

### "Claude Code CLI not found"
```bash
npm install -g @anthropic-ai/claude-code
```

### Race conditions in Option A
Use the file locking in `coordination.py` or switch to Option B/C.

### Workers not picking up tasks
- Check that dependencies are satisfied
- Verify task status is "available"
- Check worker logs for errors

### Timeout errors
Increase `task_timeout` for complex tasks.

---

## License

MIT
