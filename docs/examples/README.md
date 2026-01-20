# Code Examples

This directory contains working code examples for the Claude Multi-Agent Coordination System.

## Directory Structure

```
examples/
├── README.md                    # This file
├── workflows.md                 # Common workflow patterns
├── option-a/                    # File-based examples
│   ├── basic-coordination.py
│   ├── batch-tasks.py
│   └── custom-worker.py
├── option-b/                    # MCP server examples
│   ├── leader-workflow.md
│   └── worker-workflow.md
├── option-c/                    # Orchestrator examples
│   ├── basic_usage.py
│   ├── with_callbacks.py
│   └── ci_integration.py
└── scripts/                     # Utility scripts
    ├── export_state.py
    ├── import_state.py
    └── benchmark.py
```

## Quick Examples

### Option A: File-Based

```python
#!/usr/bin/env python3
"""Basic Option A coordination example."""

import sys
sys.path.insert(0, '../option-a-file-based')
from coordination import leader_init, leader_add_task, worker_claim, worker_complete

# Initialize
leader_init("Build a simple web page", "Create HTML, CSS, and JavaScript files")

# Add tasks
leader_add_task("Create HTML structure", priority=1)
leader_add_task("Add CSS styling", priority=2, dependencies=["task-1"])
leader_add_task("Add JavaScript interactivity", priority=2, dependencies=["task-1"])
leader_add_task("Test in browser", priority=3, dependencies=["task-2", "task-3"])

# Worker claims and completes (in separate terminal)
task = worker_claim("worker-1")
if task:
    # ... do work ...
    worker_complete("worker-1", task.id, "Created index.html with basic structure",
                   files_created=["index.html"])
```

### Option B: MCP Server

```markdown
## Leader Workflow

1. Initialize coordination:
   ```
   Use init_coordination with goal="Build REST API" and master_plan="Design, implement, test"
   ```

2. Create tasks:
   ```
   Use create_tasks_batch with tasks=[
     {"description": "Design API schema", "priority": 1},
     {"description": "Implement endpoints", "priority": 2, "dependencies": ["task-1"]},
     {"description": "Write tests", "priority": 3, "dependencies": ["task-2"]}
   ]
   ```

3. Monitor:
   ```
   Use get_status
   ```

## Worker Workflow

1. Register:
   ```
   Use register_agent with agent_id="worker-1" and role="worker"
   ```

2. Claim and work:
   ```
   Use claim_task with agent_id="worker-1"
   ```

3. Complete:
   ```
   Use complete_task with agent_id="worker-1", task_id="...", output="Done"
   ```
```

### Option C: Orchestrator

```python
#!/usr/bin/env python3
"""Basic Option C orchestrator example."""

import asyncio
from orchestrator import Orchestrator

async def main():
    # Create orchestrator
    orch = Orchestrator(
        working_directory="./my-project",
        max_workers=3,
        verbose=True
    )

    # Initialize
    await orch.initialize("Build user authentication")

    # Add tasks
    orch.add_task("Create User model", priority=1)
    orch.add_task("Implement login", priority=2, dependencies=["task-1"])
    orch.add_task("Implement logout", priority=2, dependencies=["task-1"])
    orch.add_task("Write tests", priority=3, dependencies=["task-2", "task-3"])

    # Run
    result = await orch.run_with_predefined_tasks()

    print(f"Completed: {result['tasks_completed']}")
    print(f"Summary: {result['summary']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Example Categories

### Basic Usage
- [Option A Basic](option-a/basic-coordination.py) - Simple file-based coordination
- [Option B Workflow](option-b/leader-workflow.md) - MCP-based coordination
- [Option C Basic](option-c/basic_usage.py) - Programmatic orchestration

### Advanced Patterns
- [Batch Task Creation](option-a/batch-tasks.py) - Create many tasks at once
- [Custom Worker Logic](option-a/custom-worker.py) - Extend worker behavior
- [Callbacks and Hooks](option-c/with_callbacks.py) - React to events
- [CI/CD Integration](option-c/ci_integration.py) - Automate in pipelines

### Utility Scripts
- [Export State](scripts/export_state.py) - Export coordination state
- [Import State](scripts/import_state.py) - Import from backup
- [Benchmark](scripts/benchmark.py) - Performance testing

## Running Examples

### Prerequisites

```bash
# Option A: Just Python
python3 --version  # 3.8+

# Option B: Node.js and built MCP server
cd option-b-mcp-broker && npm install && npm run build

# Option C: Install orchestrator
cd option-c-orchestrator && pip install -e .
```

### Running

```bash
# Option A
cd examples/option-a
python basic-coordination.py

# Option C
cd examples/option-c
python basic_usage.py
```

## Contributing Examples

We welcome contributions! Please:

1. Include a docstring explaining what the example demonstrates
2. Include error handling for robustness
3. Add comments for non-obvious code
4. Test your example before submitting
5. Update this README with your example
