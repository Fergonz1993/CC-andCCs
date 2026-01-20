# Migration Guide

This guide helps you migrate between the different coordination options or upgrade to newer versions.

## Table of Contents

1. [Migration Overview](#migration-overview)
2. [Option A to Option B](#option-a-to-option-b)
3. [Option B to Option C](#option-b-to-option-c)
4. [Option A to Option C](#option-a-to-option-c)
5. [Downgrade Paths](#downgrade-paths)
6. [Version Upgrades](#version-upgrades)

---

## Migration Overview

### When to Migrate

| From | To | When |
|------|-----|------|
| Option A | Option B | Need real-time updates, production use |
| Option A | Option C | Need programmatic control, automation |
| Option B | Option C | Need process management, complex workflows |
| Any | Newer Version | Bug fixes, new features |

### Migration Considerations

- **Data Compatibility**: Task schemas are compatible across options
- **In-Progress Tasks**: Complete or reassign before migration
- **State Transfer**: Export/import mechanisms available
- **Downtime**: Some migrations require brief coordination pause

---

## Option A to Option B

Migrate from file-based coordination to MCP server.

### Prerequisites

- Node.js 18+ installed
- Option B MCP server built and configured
- Current Option A session with tasks

### Step 1: Export Option A State

```python
# export_to_mcp.py
import json
from pathlib import Path
from datetime import datetime

def export_option_a_state():
    """Export Option A state to MCP-compatible format."""

    # Load Option A data
    tasks_file = Path('.coordination/tasks.json')
    if not tasks_file.exists():
        print("No Option A state found")
        return None

    with open(tasks_file) as f:
        option_a = json.load(f)

    # Load master plan
    master_plan = ""
    master_plan_file = Path('.coordination/master-plan.md')
    if master_plan_file.exists():
        master_plan = master_plan_file.read_text()

    # Convert to MCP format
    mcp_state = {
        "master_plan": master_plan,
        "goal": option_a.get('goal', 'Migrated from Option A'),
        "tasks": [],
        "agents": {},
        "discoveries": [],
        "created_at": option_a.get('created_at', datetime.now().isoformat()),
        "last_activity": option_a.get('last_updated', datetime.now().isoformat())
    }

    # Convert tasks
    for task in option_a.get('tasks', []):
        mcp_task = {
            "id": task['id'],
            "description": task['description'],
            "status": task['status'],
            "priority": task.get('priority', 5),
            "claimed_by": task.get('claimed_by'),
            "dependencies": task.get('dependencies', []),
            "context": task.get('context'),
            "result": task.get('result'),
            "created_at": task.get('created_at'),
            "claimed_at": task.get('claimed_at'),
            "completed_at": task.get('completed_at')
        }
        mcp_state['tasks'].append(mcp_task)

    # Load discoveries if exist
    discoveries_file = Path('.coordination/context/discoveries.md')
    if discoveries_file.exists():
        content = discoveries_file.read_text()
        # Parse discoveries from markdown (simplified)
        lines = [l.strip() for l in content.split('\n') if l.strip() and not l.startswith('#')]
        for i, line in enumerate(lines):
            mcp_state['discoveries'].append({
                "id": f"discovery-{i}",
                "agent_id": "migrated",
                "content": line,
                "tags": ["migrated"],
                "created_at": datetime.now().isoformat()
            })

    return mcp_state

if __name__ == '__main__':
    state = export_option_a_state()
    if state:
        output_file = '.coordination/mcp-state.json'
        with open(output_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Exported to {output_file}")
        print(f"Tasks: {len(state['tasks'])}")
        print(f"Discoveries: {len(state['discoveries'])}")
```

### Step 2: Configure MCP Server

```bash
# Build MCP server
cd option-b-mcp-broker
npm install
npm run build

# Configure Claude Code
cat >> ~/.claude/mcp.json << 'EOF'
{
  "mcpServers": {
    "multi-agent-coordinator": {
      "command": "node",
      "args": ["/path/to/option-b-mcp-broker/dist/index.js"],
      "env": {
        "COORDINATION_DIR": "/path/to/your/project/.coordination"
      }
    }
  }
}
EOF
```

### Step 3: Start MCP Server

Restart Claude Code. The MCP server will load the exported state automatically.

### Step 4: Verify Migration

```
Use the get_status tool
```

Should show the migrated tasks.

### Step 5: Re-register Agents

Each Claude Code instance needs to register:

```
Use the register_agent tool with:
- agent_id: "terminal-1"
- role: "leader"
```

---

## Option B to Option C

Migrate from MCP server to Python orchestrator.

### Prerequisites

- Python 3.8+ installed
- Option C orchestrator installed
- Current Option B session with tasks

### Step 1: Export Option B State

```python
# export_mcp_to_orchestrator.py
import json
from pathlib import Path

def export_mcp_state():
    """Export MCP state to Orchestrator format."""

    mcp_file = Path('.coordination/mcp-state.json')
    if not mcp_file.exists():
        print("No MCP state found")
        return None

    with open(mcp_file) as f:
        mcp_state = json.load(f)

    # Orchestrator uses Pydantic models, but state is JSON-compatible
    orch_state = {
        "goal": mcp_state.get('goal', ''),
        "master_plan": mcp_state.get('master_plan', ''),
        "working_directory": str(Path('.').resolve()),
        "tasks": [],
        "discoveries": [],
        "max_parallel_workers": 3,
        "task_timeout_seconds": 600,
        "created_at": mcp_state.get('created_at'),
        "last_activity": mcp_state.get('last_activity')
    }

    # Convert tasks
    for task in mcp_state.get('tasks', []):
        orch_task = {
            "id": task['id'],
            "description": task['description'],
            "status": task['status'],
            "priority": task.get('priority', 5),
            "claimed_by": task.get('claimed_by'),
            "dependencies": task.get('dependencies', []),
            "context": {
                "files": task.get('context', {}).get('files', []),
                "hints": task.get('context', {}).get('hints', ''),
                "parent_task": task.get('context', {}).get('parent_task')
            } if task.get('context') else None,
            "result": task.get('result'),
            "created_at": task.get('created_at'),
            "claimed_at": task.get('claimed_at'),
            "completed_at": task.get('completed_at')
        }
        orch_state['tasks'].append(orch_task)

    # Convert discoveries
    for disc in mcp_state.get('discoveries', []):
        orch_disc = {
            "agent_id": disc.get('agent_id', 'unknown'),
            "content": disc.get('content', ''),
            "related_task": None,
            "tags": disc.get('tags', []),
            "created_at": disc.get('created_at')
        }
        orch_state['discoveries'].append(orch_disc)

    return orch_state

if __name__ == '__main__':
    state = export_mcp_state()
    if state:
        output_file = 'orchestrator-state.json'
        with open(output_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Exported to {output_file}")
```

### Step 2: Import into Orchestrator

```python
# import_to_orchestrator.py
import asyncio
from orchestrator import Orchestrator

async def import_and_resume():
    orch = Orchestrator(
        working_directory=".",
        max_workers=3
    )

    # Load migrated state
    orch.load_state('orchestrator-state.json')

    print(f"Loaded goal: {orch.state.goal}")
    print(f"Tasks: {len(orch.state.tasks)}")

    # Resume execution if there are pending tasks
    pending = [t for t in orch.state.tasks if t.status in ('available', 'claimed', 'in_progress')]
    if pending:
        print(f"Resuming with {len(pending)} pending tasks...")
        result = await orch.run_with_predefined_tasks()
        print(f"Completed: {result['tasks_completed']}")

asyncio.run(import_and_resume())
```

---

## Option A to Option C

Direct migration from file-based to orchestrator.

### Quick Migration Script

```python
# migrate_a_to_c.py
import asyncio
import json
from pathlib import Path
from orchestrator import Orchestrator
from orchestrator.models import Task, TaskStatus, TaskContext

async def migrate_option_a_to_c():
    # Load Option A state
    tasks_file = Path('.coordination/tasks.json')
    with open(tasks_file) as f:
        option_a = json.load(f)

    master_plan_file = Path('.coordination/master-plan.md')
    master_plan = master_plan_file.read_text() if master_plan_file.exists() else ""

    # Extract goal from master plan (simplified)
    goal = option_a.get('goal', 'Migrated from Option A')
    for line in master_plan.split('\n'):
        if 'objective' in line.lower() or 'goal' in line.lower():
            goal = line.replace('#', '').replace('Objective', '').replace('Goal', '').strip()
            break

    # Create orchestrator
    orch = Orchestrator(
        working_directory=".",
        max_workers=3,
        verbose=True
    )

    await orch.initialize(goal, master_plan)

    # Import tasks
    for task_data in option_a.get('tasks', []):
        # Skip completed tasks
        if task_data['status'] in ('done', 'failed'):
            continue

        task = Task(
            id=task_data['id'],
            description=task_data['description'],
            status=TaskStatus(task_data['status']),
            priority=task_data.get('priority', 5),
            dependencies=task_data.get('dependencies', []),
            context=TaskContext(
                files=task_data.get('context', {}).get('files', []),
                hints=task_data.get('context', {}).get('hints', '')
            ) if task_data.get('context') else TaskContext()
        )
        orch.state.tasks.append(task)

    print(f"Imported {len(orch.state.tasks)} tasks")

    # Save state
    orch.save_state('orchestrator-state.json')

    # Optionally continue execution
    if input("Continue execution? (y/n): ").lower() == 'y':
        result = await orch.run_with_predefined_tasks()
        print(f"Completed: {result['tasks_completed']}")

if __name__ == '__main__':
    asyncio.run(migrate_option_a_to_c())
```

---

## Downgrade Paths

### Option C to Option A

```python
# downgrade_c_to_a.py
import json
from pathlib import Path

def downgrade_to_option_a():
    # Load orchestrator state
    with open('orchestrator-state.json') as f:
        orch_state = json.load(f)

    # Create Option A structure
    Path('.coordination').mkdir(exist_ok=True)
    Path('.coordination/context').mkdir(exist_ok=True)
    Path('.coordination/logs').mkdir(exist_ok=True)
    Path('.coordination/results').mkdir(exist_ok=True)

    # Convert to Option A format
    option_a = {
        "version": "1.0",
        "created_at": orch_state.get('created_at'),
        "last_updated": orch_state.get('last_activity'),
        "tasks": []
    }

    for task in orch_state.get('tasks', []):
        a_task = {
            "id": task['id'],
            "description": task['description'],
            "status": task['status'],
            "priority": task.get('priority', 5),
            "claimed_by": task.get('claimed_by'),
            "dependencies": task.get('dependencies', []),
            "context": task.get('context'),
            "result": task.get('result'),
            "created_at": task.get('created_at'),
            "claimed_at": task.get('claimed_at'),
            "completed_at": task.get('completed_at')
        }
        option_a['tasks'].append(a_task)

    # Write files
    with open('.coordination/tasks.json', 'w') as f:
        json.dump(option_a, f, indent=2)

    # Write master plan
    master_plan = f"""# Master Plan

## Objective
{orch_state.get('goal', 'Downgraded from Option C')}

## Plan
{orch_state.get('master_plan', '')}
"""
    with open('.coordination/master-plan.md', 'w') as f:
        f.write(master_plan)

    print(f"Downgraded {len(option_a['tasks'])} tasks to Option A")

if __name__ == '__main__':
    downgrade_to_option_a()
```

### Option B to Option A

Similar approach - convert mcp-state.json to tasks.json format.

---

## Version Upgrades

### Schema Migrations

When upgrading, check for schema changes:

```python
# migrate_schema.py
import json
from pathlib import Path

CURRENT_VERSION = "2.0"

def migrate_schema(filepath: str):
    with open(filepath) as f:
        data = json.load(f)

    version = data.get('version', '1.0')

    if version == CURRENT_VERSION:
        print("Already at current version")
        return

    # Version 1.0 -> 1.1: Add tags field to tasks
    if version == '1.0':
        for task in data.get('tasks', []):
            if 'tags' not in task:
                task['tags'] = []
        version = '1.1'

    # Version 1.1 -> 2.0: Add retry fields
    if version == '1.1':
        for task in data.get('tasks', []):
            if 'max_retries' not in task:
                task['max_retries'] = 3
            if 'retry_count' not in task:
                task['retry_count'] = 0
        version = '2.0'

    data['version'] = version

    # Backup original
    backup = Path(filepath).with_suffix('.bak')
    Path(filepath).rename(backup)

    # Write migrated
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Migrated from {data.get('version', '1.0')} to {version}")
    print(f"Backup saved to {backup}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        migrate_schema(sys.argv[1])
    else:
        migrate_schema('.coordination/tasks.json')
```

### Testing After Migration

```bash
# Verify file structure
ls -la .coordination/

# Validate JSON
python3 -c "import json; json.load(open('.coordination/tasks.json'))"

# Check task count
python3 -c "
import json
with open('.coordination/tasks.json') as f:
    data = json.load(f)
print(f'Tasks: {len(data.get(\"tasks\", []))}')
"

# Run a status check
python coordination.py leader status  # Option A
# or
orchestrate status  # Option C
```
