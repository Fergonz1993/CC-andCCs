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

---

## Long-Running Development Harness

This project uses the "Effective Harnesses for Long-Running Agents" methodology from Anthropic for continuous development across multiple Claude sessions.

### Harness Files

| File | Purpose |
|------|---------|
| `init.sh` | Environment setup script (run first) |
| `feature_list.json` | 210 features to implement and verify |
| `claude-progress.txt` | Progress tracking between sessions |
| `CODING_AGENT.md` | Complete session instructions |

### Quick Start

```bash
# 1. Setup environment
chmod +x init.sh && ./init.sh

# 2. Check current progress
cat claude-progress.txt

# 3. See remaining features
cat feature_list.json | python3 -c "
import json,sys
d=json.load(sys.stdin)
passing=sum(1 for f in d['features'] if f['passes'])
print(f'Progress: {passing}/{len(d[\"features\"])} features passing')
"
```

### Development Workflow

1. **Start**: Read `claude-progress.txt` to orient yourself
2. **Setup**: Run `./init.sh` if environment needs setup
3. **Verify**: Check existing tests still pass
4. **Implement**: Pick ONE feature from `feature_list.json`
5. **Update**: Mark `"passes": true` when verified
6. **Commit**: Commit with clear message
7. **Document**: Update `claude-progress.txt`
8. **Repeat**: Continue until context fills

### Feature Categories

- `option-a-*`: File-based coordination (50 features)
- `option-b-*`: MCP server (45 features)
- `option-c-*`: Python orchestrator (105 features)
- `integration`: End-to-end tests (10 features)

### Progress Tracking

The `claude-progress.txt` file tracks:
- Overall feature counts per option
- Priority queue of what to work on next
- Known issues and their fixes
- Session history with summaries
- Test results log

### For New Sessions

Read `CODING_AGENT.md` for detailed instructions on:
- Getting oriented in a new session
- Running verification tests
- Choosing features to implement
- Updating progress files
- Committing changes cleanly
