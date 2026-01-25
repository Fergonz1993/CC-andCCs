---
layout: default
title: Getting Started
nav_order: 2
---

# Getting Started

Get up and running with multi-agent coordination in under 5 minutes.

---

## Prerequisites

- Python 3.8+ (for Options A and C)
- Node.js 18+ (for Option B)
- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`)

---

## Option A: File-Based

The simplest option. No installation required - just Python standard library.

### Setup

```bash
git clone https://github.com/your-org/claude-multi-agent.git
cd claude-multi-agent/option-a-file-based
```

### Your First Coordination

**Terminal 1 - Leader:**

```bash
# Initialize a coordination session
python coordination.py leader init "Build a REST API for user management"

# Create tasks for your team
python coordination.py leader add-task "Set up Express.js project" -p 1
python coordination.py leader add-task "Create User model" -p 1
python coordination.py leader add-task "Implement GET /users" -p 2 -d task-001 task-002
python coordination.py leader add-task "Implement POST /users" -p 2 -d task-002
python coordination.py leader add-task "Write tests" -p 3 -d task-003 task-004

# Check status
python coordination.py leader status
```

**Terminal 2 - Worker:**

```bash
cd claude-multi-agent/option-a-file-based

# Claim the highest priority available task
python coordination.py worker claim terminal-2

# After completing the work
python coordination.py worker complete terminal-2 task-001 \
  "Set up Express.js with TypeScript" \
  -c package.json tsconfig.json src/index.ts

# Claim next task
python coordination.py worker claim terminal-2
```

**Terminal 3 - Worker:**

```bash
cd claude-multi-agent/option-a-file-based

# Work in parallel with Terminal 2
python coordination.py worker claim terminal-3
```

### When Done

```bash
# Aggregate all results
python coordination.py leader aggregate

# View summary
cat .coordination/summary.md
```

---

## Option B: MCP Server

Real-time coordination through Model Context Protocol.

### Setup

```bash
cd claude-multi-agent/option-b-mcp-broker
npm install
npm run build
```

### Configure Claude Code

Add to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "multi-agent-coordinator": {
      "command": "node",
      "args": ["/full/path/to/option-b-mcp-broker/dist/index.js"],
      "env": {
        "COORDINATION_DIR": "/path/to/your/project"
      }
    }
  }
}
```

Restart Claude Code to load the MCP server.

### Your First Coordination

Now in Claude Code, you can use MCP tools directly:

**Terminal 1 - Leader:**

```
Use the init_coordination tool:
- goal: "Build authentication system"
- master_plan: "1. Create JWT utilities\n2. Add middleware\n3. Protect routes"

Use the create_tasks_batch tool:
- tasks: [
    {"description": "Create JWT sign/verify functions", "priority": 1},
    {"description": "Implement auth middleware", "priority": 2, "dependencies": ["task-1"]},
    {"description": "Create login endpoint", "priority": 2, "dependencies": ["task-1"]},
    {"description": "Write tests", "priority": 3, "dependencies": ["task-2", "task-3"]}
  ]

Use the get_status tool to monitor progress.
```

**Terminal 2 - Worker:**

```
Use the register_agent tool:
- agent_id: "terminal-2"
- role: "worker"

Use the claim_task tool:
- agent_id: "terminal-2"

After completing work, use the complete_task tool:
- agent_id: "terminal-2"
- task_id: "task-xxx"
- output: "Created JWT utilities with sign/verify"
- files_created: ["src/auth/jwt.ts"]
```

---

## Option C: External Orchestrator

Full programmatic control with automatic worker management.

### Setup

```bash
cd claude-multi-agent/option-c-orchestrator
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Your First Coordination

**CLI - Automatic Mode:**

```bash
# Let the leader plan everything
orchestrate run "Build a REST API for user management" -w 3
```

**CLI - Predefined Tasks:**

```bash
# Create a tasks file
cat > tasks.json << 'EOF'
[
  {"description": "Create User model", "priority": 1},
  {"description": "Implement CRUD endpoints", "priority": 2, "dependencies": ["task-1"]},
  {"description": "Write tests", "priority": 3, "dependencies": ["task-2"]}
]
EOF

# Run with predefined tasks
orchestrate run "Build API" --no-plan --tasks tasks.json -w 2
```

**Python Script:**

```python
import asyncio
from orchestrator import Orchestrator

async def main():
    orch = Orchestrator(
        working_directory="./my-project",
        max_workers=3,
    )

    await orch.initialize("Build user authentication")

    # Add tasks
    t1 = orch.add_task("Create User model", priority=1)
    t2 = orch.add_task("Implement login", priority=2, dependencies=[t1.id])
    t3 = orch.add_task("Write tests", priority=3, dependencies=[t2.id])

    # Run
    result = await orch.run_with_predefined_tasks()
    print(f"Completed: {result['tasks_completed']} tasks")

asyncio.run(main())
```

---

## Choosing an Option

| Need | Recommended |
|------|-------------|
| Quick prototype | Option A |
| Learning the system | Option A |
| Real-time coordination | Option B |
| Production workflow | Option B |
| CI/CD automation | Option C |
| Programmatic control | Option C |
| Complex dependencies | Option B or C |
| Minimal setup | Option A |

---

## Next Steps

1. Read the [User Guide](guides/user-guide.md) for detailed workflows
2. Explore [Code Examples](examples/README.md)
3. Check the [API Reference](api/cli-reference.md)
4. Review [Best Practices](guides/user-guide.md#best-practices)

---

## Troubleshooting

### "Claude Code CLI not found"

```bash
npm install -g @anthropic-ai/claude-code
```

### Tasks not being claimed

- Check dependencies are satisfied
- Verify task status is "available"
- Review worker logs in `.coordination/logs/`

### Race conditions in Option A

Use the built-in file locking or switch to Option B/C for atomic operations.

See [Troubleshooting Guide](guides/troubleshooting.md) for more solutions.
