# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **multi-agent coordination system** for Claude Code instances. It enables multiple Claude Code terminals to work together on complex tasks by sharing a coordination layer (filesystem, MCP server, or external orchestrator).

## Architecture

Three coordination options exist, each in its own directory under `claude-multi-agent/`:

| Option | Directory | Language | Coordination Method |
|--------|-----------|----------|---------------------|
| A | `option-a-file-based/` | Python | File polling via `.coordination/` |
| B | `option-b-mcp-broker/` | TypeScript | MCP server with shared state |
| C | `option-c-orchestrator/` | Python | External process management |

**Leader-Worker Pattern**: Terminal 1 acts as leader (plans work, creates tasks, aggregates results). Terminals 2+ are workers (claim tasks, execute, report back).

## Build & Run Commands

### Option A: File-Based
```bash
cd claude-multi-agent/option-a-file-based
python coordination.py leader init "Goal description"
python coordination.py leader add-task "Task description" -p 1
python coordination.py worker claim terminal-2
```

### Option B: MCP Server
```bash
cd claude-multi-agent/option-b-mcp-broker
npm install
npm run build          # Compile TypeScript
npm run dev            # Run with tsx (development)
npm start              # Run compiled JS
```

Add to `~/.claude/mcp.json`:
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

### Option C: Orchestrator
```bash
cd claude-multi-agent/option-c-orchestrator
pip install -e .
orchestrate run "Goal" -w 3              # Leader plans automatically
orchestrate run "Goal" --no-plan --tasks tasks.json  # Predefined tasks
pytest                                    # Run tests (requires dev deps)
```

## Key Files

- `shared/schemas/task.schema.json` - Canonical task JSON structure
- `shared/prompts/leader.md` - System prompt for leader agents
- `shared/prompts/worker.md` - System prompt for worker agents
- `option-a-file-based/coordination.py` - CLI for file-based coordination
- `option-b-mcp-broker/src/index.ts` - MCP server implementation
- `option-c-orchestrator/src/orchestrator/orchestrator.py` - Main orchestration logic

## Task Lifecycle

`available` → `claimed` → `in_progress` → `done` | `failed`

Tasks must specify: `id`, `description`, `status`, `priority` (1=highest). Dependencies are other task IDs that must complete first.

## Coordination Directory Structure (Options A & C)

```
.coordination/
├── master-plan.md      # Overall goal and approach
├── tasks.json          # Task queue (source of truth)
├── context/discoveries.md  # Shared findings between agents
├── logs/               # Per-agent activity logs
└── results/            # Completed task outputs (task-{id}.md)
```

## Race Condition Handling

- **Option A**: Workers must re-read `tasks.json` after claiming to verify success
- **Options B & C**: Built-in atomic operations
