# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **multi-agent coordination system** enabling multiple Claude Code terminals to work together on complex tasks. Uses a leader-worker pattern where Terminal 1 plans and aggregates, while Terminals 2+ claim and execute tasks.

## Architecture

Three coordination options under `claude-multi-agent/`:

| Option | Directory | Language | Method |
|--------|-----------|----------|--------|
| A | `option-a-file-based/` | Python | Filesystem polling via `.coordination/` |
| B | `option-b-mcp-broker/` | TypeScript | MCP server with shared state |
| C | `option-c-orchestrator/` | Python | External process management |

### Shared Modules (`shared/`)

Cross-cutting utilities used by all options:
- `shared/schemas/task.schema.json` - Canonical task JSON structure
- `shared/prompts/` - System prompts for leader and worker agents
- `shared/security/` - Auth, encryption, rate limiting, audit
- `shared/reliability/` - Circuit breaker, retry, fallback, self-healing
- `shared/performance/` - Caching, compression, async I/O, profiling
- `shared/ai/` - Task prioritization, decomposition, anomaly detection
- `shared/cross_option/` - Migration, sync, unified CLI, plugins

### Progress Tracking

Development progress is tracked via `TODO_RALPH.md` which contains the active backlog organized by priority. Completed items from previous releases are archived in collapsible sections. Check `AGENTS.md` for current verification status.

## Build & Test Commands

### Option A: File-Based (Python)
```bash
cd claude-multi-agent/option-a-file-based

# Run CLI
python coordination.py --help
python coordination.py leader init "Goal description"
python coordination.py worker claim terminal-2

# Run tests
pytest                              # All tests
pytest tests/test_coordination.py   # Single file
pytest -k "test_claim"              # Single test by name
pytest --cov                        # With coverage (requires pytest-cov)
```

### Option B: MCP Server (TypeScript)
```bash
cd claude-multi-agent/option-b-mcp-broker

npm install
npm run build          # Compile TypeScript
npm run dev            # Run with tsx (development)
npm start              # Run compiled JS
npm run watch          # Watch mode compilation

# Tests
npm test                             # All tests
npm test -- --testPathPattern=index  # Single file
npm test -- -t "claim"               # Single test by name
npm run test:coverage                # With coverage
npm run test:watch                   # Watch mode

# Type checking
npx tsc --noEmit

# Security audit
npm audit
```

MCP config for `~/.claude/mcp.json`:
```json
{
  "mcpServers": {
    "multi-agent-coordinator": {
      "command": "node",
      "args": ["<path>/option-b-mcp-broker/dist/index.js"],
      "env": { "COORDINATION_DIR": "<working-dir>" }
    }
  }
}
```

### Option C: Orchestrator (Python)
```bash
cd claude-multi-agent/option-c-orchestrator

# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# CLI
orchestrate --help
orchestrate run "Goal" -w 3                          # Auto-plan with 3 workers
orchestrate run "Goal" --no-plan --tasks tasks.json  # Predefined tasks

# Tests
pytest                                    # All tests
pytest tests/test_orchestrator_unit.py    # Single file
pytest -k "test_task_claim"               # Single test by name
pytest tests/test_property_based.py       # Property-based tests (hypothesis)

# Security audit
pip-audit                                 # Check for vulnerable dependencies
```

### Integration Tests
```bash
cd claude-multi-agent
pytest tests/integration/                 # All integration tests
pytest tests/integration/test_workflow.py # End-to-end workflow
pytest tests/integration/test_chaos.py    # Chaos/resilience tests
```

## Task Lifecycle

`available` → `claimed` → `in_progress` → `done` | `failed`

Task schema requires: `id`, `description`, `status`, `priority` (1=highest). Dependencies are task IDs that must complete first.

## Coordination Directory Structure (Options A & C)

```
.coordination/
├── master-plan.md          # Overall goal
├── tasks.json              # Task queue (source of truth)
├── context/discoveries.md  # Shared findings
├── logs/                   # Per-agent logs
└── results/                # Completed task outputs (task-{id}.md)
```

## Key Implementation Files

| Component | File |
|-----------|------|
| File-based CLI | `option-a-file-based/coordination.py` |
| MCP server | `option-b-mcp-broker/src/index.ts` |
| Orchestrator core | `option-c-orchestrator/src/orchestrator/orchestrator.py` |
| Async orchestrator | `option-c-orchestrator/src/orchestrator/async_orchestrator.py` |
| Task/Agent models | `option-c-orchestrator/src/orchestrator/models.py` |
| Configuration | `option-c-orchestrator/src/orchestrator/config.py` |
| CLI interface | `option-c-orchestrator/src/orchestrator/cli.py` |
| Agent subprocess | `option-c-orchestrator/src/orchestrator/agent.py` |

## Race Condition Handling

- **Option A**: Workers must re-read `tasks.json` after claiming to verify success
- **Options B & C**: Built-in atomic operations

## CI/CD

GitHub Actions workflow in `.github/workflows/test.yml` runs tests across all three options. The workflow validates:
- Option A Python tests
- Option B TypeScript build and Jest tests
- Option C pytest suite including property-based tests
- Security audits (pip-audit, npm audit)

---

## Long-Running Development Harness

Uses Anthropic's "Effective Harnesses for Long-Running Agents" methodology for continuous development.

| File | Purpose |
|------|---------|
| `init.sh` | Environment setup (run first in new session) |
| `TODO_RALPH.md` | Active backlog and archived releases |
| `claude-progress.txt` | Progress tracking between sessions |
| `CODING_AGENT.md` | Complete session workflow instructions |

### Quick Start
```bash
./init.sh                    # Setup all environments
cat claude-progress.txt      # Check current progress
```

### Development Workflow
1. Read `claude-progress.txt` for context
2. Run `./init.sh` if environment needs setup
3. Verify existing tests still pass
4. Pick ONE item from `TODO_RALPH.md` backlog
5. Implement and verify with tests
6. Update `TODO_RALPH.md`, commit

See `CODING_AGENT.md` for detailed session instructions.

### Quick Verification Commands
```bash
# Option C import check
cd claude-multi-agent/option-c-orchestrator && source .venv/bin/activate
python -c "from orchestrator import Orchestrator; from orchestrator.models import Task; print('OK')"

# Option B build check
cd claude-multi-agent/option-b-mcp-broker && ls dist/index.js

# Option A CLI check
cd claude-multi-agent/option-a-file-based && python coordination.py --help

# Run all tests
cd claude-multi-agent/option-c-orchestrator && pytest
cd claude-multi-agent/option-b-mcp-broker && npm test
```
