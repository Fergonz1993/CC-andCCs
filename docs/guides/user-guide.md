# User Guide

This comprehensive guide walks you through using the Claude Multi-Agent Coordination System for real-world development tasks.

## Table of Contents

1. [Introduction](#introduction)
2. [Choosing an Option](#choosing-an-option)
3. [Option A: File-Based Coordination](#option-a-file-based-coordination)
4. [Option B: MCP Server](#option-b-mcp-server)
5. [Option C: External Orchestrator](#option-c-external-orchestrator)
6. [Common Workflows](#common-workflows)
7. [Best Practices](#best-practices)

---

## Introduction

The Claude Multi-Agent Coordination System enables multiple Claude Code instances to work together on complex development tasks. Think of it as having a team of AI developers:

- **Leader (Terminal 1)**: Plans the work, creates tasks, monitors progress, and aggregates results
- **Workers (Terminals 2, 3, ...)**: Claim tasks, execute them, and report back

This guide will help you get started with each coordination option and apply them to real projects.

---

## Choosing an Option

| Scenario | Recommended Option |
|----------|-------------------|
| Quick prototype or learning | Option A (File-Based) |
| Production workflow with real-time needs | Option B (MCP Server) |
| Automated CI/CD pipelines | Option C (Orchestrator) |
| Simple task parallelization | Option A |
| Complex dependency chains | Option B or C |
| Programmatic control needed | Option C |

---

## Option A: File-Based Coordination

Best for: Quick setup, simple tasks, learning the system.

### Setup

No installation needed - just Python 3.7+ standard library.

```bash
cd option-a-file-based
```

### Workflow Example: Building a REST API

**Terminal 1 (Leader)**

```bash
# 1. Initialize the coordination
python coordination.py leader init "Build a REST API for user management"

# 2. Create tasks
python coordination.py leader add-task "Create database schema" -p 1
python coordination.py leader add-task "Implement User model" -p 1
python coordination.py leader add-task "Create GET /users endpoint" -p 2 -d task-001 task-002
python coordination.py leader add-task "Create POST /users endpoint" -p 2 -d task-002
python coordination.py leader add-task "Create PUT /users/:id endpoint" -p 3 -d task-003 task-004
python coordination.py leader add-task "Create DELETE /users/:id endpoint" -p 3 -d task-003 task-004
python coordination.py leader add-task "Write API tests" -p 4 -d task-003 task-004 task-005 task-006

# 3. Monitor progress
python coordination.py leader status

# 4. When done, aggregate results
python coordination.py leader aggregate
```

**Terminal 2 (Worker)**

```bash
# 1. Claim a task
python coordination.py worker claim terminal-2

# Output:
# Claimed task: task-20240115-001
#   Description: Create database schema
#   Priority: 1

# 2. Execute the task (use Claude Code normally)
# ... work on creating database schema ...

# 3. Mark as complete
python coordination.py worker complete terminal-2 task-20240115-001 \
  "Created PostgreSQL schema with users table" \
  -c migrations/001_create_users.sql -m prisma/schema.prisma

# 4. Claim next task
python coordination.py worker claim terminal-2
```

**Terminal 3 (Worker)**

```bash
# Same workflow as Terminal 2
python coordination.py worker claim terminal-3
# ... work ...
python coordination.py worker complete terminal-3 task-20240115-002 "..."
```

### Monitoring Progress

```bash
# Check status anytime
python coordination.py leader status

# View available tasks
python coordination.py worker list

# Check specific task result
cat .coordination/results/task-20240115-001.md
```

---

## Option B: MCP Server

Best for: Production workflows, real-time coordination, team environments.

### Setup

```bash
cd option-b-mcp-broker
npm install
npm run build
```

Add to Claude Code config (`~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "coordination": {
      "command": "node",
      "args": ["/path/to/option-b-mcp-broker/dist/index.js"],
      "env": {
        "COORDINATION_DIR": "/path/to/your/project"
      }
    }
  }
}
```

Restart Claude Code - all instances now share the coordination server.

### Workflow Example: Adding Authentication

**Terminal 1 (Leader)**

In Claude Code, the leader uses MCP tools directly:

```
Use the init_coordination tool with:
- goal: "Add JWT authentication to the API"
- master_plan: "1. Create auth utilities\n2. Add middleware\n3. Protect routes"
```

```
Use the register_agent tool with:
- agent_id: "leader-1"
- role: "leader"
```

```
Use the create_tasks_batch tool with:
- tasks: [
    {"description": "Create JWT utility module", "priority": 1},
    {"description": "Implement auth middleware", "priority": 2, "dependencies": ["task-1"]},
    {"description": "Create login endpoint", "priority": 2, "dependencies": ["task-1"]},
    {"description": "Create logout endpoint", "priority": 3, "dependencies": ["task-3"]},
    {"description": "Protect existing routes", "priority": 3, "dependencies": ["task-2"]},
    {"description": "Write auth tests", "priority": 4, "dependencies": ["task-2", "task-3", "task-4", "task-5"]}
  ]
```

Monitor progress:
```
Use the get_status tool
```

**Terminal 2 (Worker)**

```
Use the register_agent tool with:
- agent_id: "terminal-2"
- role: "worker"
```

```
Use the claim_task tool with:
- agent_id: "terminal-2"
```

After completing work:
```
Use the complete_task tool with:
- agent_id: "terminal-2"
- task_id: "task-xxx"
- output: "Created JWT utility with sign/verify functions"
- files_created: ["src/auth/jwt.ts"]
```

Share discoveries:
```
Use the add_discovery tool with:
- agent_id: "terminal-2"
- content: "Found existing bcrypt utility in src/utils/crypto.ts - can reuse for password hashing"
- tags: ["auth", "existing-code"]
```

---

## Option C: External Orchestrator

Best for: Automation, CI/CD pipelines, programmatic control.

### Setup

```bash
cd option-c-orchestrator
pip install -e .
```

### Workflow Example: Full Automation

**CLI Usage**

```bash
# Full automatic mode - leader plans, workers execute
orchestrate run "Build a user authentication system" -w 3

# With predefined tasks
cat > tasks.json << 'EOF'
[
  {"description": "Create User model", "priority": 1},
  {"description": "Implement login endpoint", "priority": 2, "dependencies": ["task-1"]},
  {"description": "Implement logout endpoint", "priority": 3, "dependencies": ["task-2"]},
  {"description": "Write tests", "priority": 4, "dependencies": ["task-2", "task-3"]}
]
EOF

orchestrate run "Build auth" --no-plan --tasks tasks.json
```

**Python Script**

```python
import asyncio
from orchestrator import Orchestrator

async def main():
    orch = Orchestrator(
        working_directory="./my-project",
        max_workers=3,
        verbose=True,
    )

    await orch.initialize("Refactor database layer")

    # Add tasks
    t1 = orch.add_task("Analyze current database code", priority=1)
    t2 = orch.add_task("Design new schema", priority=1)
    t3 = orch.add_task("Create migration scripts", priority=2, dependencies=[t1.id, t2.id])
    t4 = orch.add_task("Update model code", priority=2, dependencies=[t3.id])
    t5 = orch.add_task("Update tests", priority=3, dependencies=[t4.id])

    # Run
    result = await orch.run_with_predefined_tasks()

    print(f"Completed: {result['tasks_completed']} tasks")
    print(f"Failed: {result['tasks_failed']} tasks")
    print(f"\nSummary:\n{result['summary']}")

asyncio.run(main())
```

**With Callbacks**

```python
from orchestrator import Orchestrator
from orchestrator.models import Task, Discovery

def on_task_complete(task: Task):
    print(f"[DONE] {task.id}: {task.description[:50]}")
    # Send notification, update dashboard, etc.

def on_discovery(discovery: Discovery):
    print(f"[DISCOVERY] {discovery.content}")
    # Log to shared knowledge base

orch = Orchestrator(
    on_task_complete=on_task_complete,
    on_discovery=on_discovery,
)
```

---

## Common Workflows

### Web Application Development

```
Goal: "Build a todo list web application"

Tasks (in dependency order):
1. Set up project structure (priority 1)
2. Create database schema (priority 1)
3. Implement API endpoints (priority 2, depends: 1, 2)
4. Build frontend components (priority 2, depends: 1)
5. Integrate frontend with API (priority 3, depends: 3, 4)
6. Add authentication (priority 3, depends: 3)
7. Write tests (priority 4, depends: 5, 6)
8. Write documentation (priority 5, depends: 7)
```

### Code Refactoring

```
Goal: "Refactor legacy authentication module"

Tasks:
1. Analyze current auth code and document issues
2. Design new auth architecture
3. Create new auth module structure
4. Migrate login functionality
5. Migrate logout functionality
6. Migrate password reset
7. Update all call sites
8. Write comprehensive tests
9. Remove legacy code
```

### Bug Investigation

```
Goal: "Investigate and fix performance issues in /api/search"

Tasks:
1. Profile current endpoint performance
2. Analyze database queries
3. Review caching strategy
4. Implement query optimizations
5. Add database indexes
6. Implement caching
7. Load test improvements
8. Document changes
```

---

## Best Practices

### Task Design

1. **Be Specific**: Include all necessary context
   - Bad: "Fix the bug"
   - Good: "Fix null pointer exception in UserService.getUser() when user ID doesn't exist"

2. **Right Size**: 5-15 minutes of work per task
   - Too small: "Add import statement"
   - Too large: "Build entire frontend"
   - Right: "Implement user login form with validation"

3. **Clear Dependencies**: Be explicit about what must complete first
   ```bash
   python coordination.py leader add-task "Implement login" -d task-001 task-002
   ```

4. **Include Context**: Provide relevant files and hints
   ```bash
   python coordination.py leader add-task "Add validation" \
     -f src/models/user.ts src/utils/validate.ts \
     --hints "Use existing validateEmail function"
   ```

### Coordination

1. **Read First**: Always read the master plan before starting
2. **Check Dependencies**: Verify dependencies are complete before claiming
3. **Share Discoveries**: Write important findings to shared context
4. **Log Everything**: Detailed logs help debug issues later

### Error Handling

1. **Mark Failures**: Don't leave tasks stuck in in_progress
   ```bash
   python coordination.py worker fail terminal-2 task-001 "Missing npm dependency"
   ```

2. **Provide Details**: Error messages should be actionable
   - Bad: "It didn't work"
   - Good: "TypeScript compilation failed: Property 'name' does not exist on type 'User'"

3. **Suggest Fixes**: If you know what went wrong, include a fix suggestion

### Performance

1. **Batch Related Tasks**: Group related work to minimize context switching
2. **Parallelize Independent Work**: Tasks without dependencies can run concurrently
3. **Avoid Bottlenecks**: Don't make everything depend on one task
