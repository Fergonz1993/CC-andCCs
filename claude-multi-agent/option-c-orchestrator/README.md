# Claude Multi-Agent Orchestrator (Option C)

External orchestrator for coordinating multiple Claude Code instances with full programmatic control.

## Features

- **Hybrid Backend**: Automatic selection between file-based (sync) and async orchestrators
- **Leader-Worker Pattern**: Automatic task planning and distribution
- **Dependency Tracking**: Task DAG with dependency resolution
- **Retry Policies**: Configurable retry with exponential backoff
- **Metrics Export**: Task queue size and throughput monitoring
- **Structured Logging**: JSON-formatted events with correlation IDs

## Installation

```bash
# Standard installation
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## CLI Usage

```bash
# Let leader plan automatically with 3 workers
orchestrate run "Build a user authentication system" -w 3

# With predefined tasks
orchestrate run "Build API" --no-plan --tasks tasks.json

# Initialize a configuration file
orchestrate init "Add authentication feature"
orchestrate create-task "Create JWT utility" -p 1
orchestrate from-config orchestration.json
```

## Programmatic Usage

```python
import asyncio
from orchestrator import Orchestrator

async def main():
    # Async mode (production)
    orch = Orchestrator(
        working_directory="./my-project",
        max_workers=3,
        model="claude-sonnet-4-20250514",
    )
    await orch.initialize("Build user authentication")
    result = await orch.run_with_leader_planning()

    # File-based mode (testing/debugging)
    orch = Orchestrator(coordination_dir="./.coordination", goal="Test task")
    orch.add_task("Task 1", priority=1)
    task = orch.claim_task("task-xxx", "agent-1")

asyncio.run(main())
```

## Testing

```bash
# All tests
pytest

# Property-based tests
pytest tests/test_property_based.py -v --hypothesis-show-statistics

# With coverage
pytest --cov=orchestrator --cov-report=html
```

## Security Audit

```bash
pip-audit
```

See [main README](../README.md) for complete documentation.
