# ADR-006: State Persistence Strategy

## Status

Accepted

## Date

2024-01-15

## Context

Coordination state (tasks, agent registrations, discoveries) must persist across process restarts and be recoverable in case of crashes. Different options have different persistence needs and constraints.

## Decision

Each coordination option implements state persistence appropriate to its complexity level:

### Option A: File-Based Persistence

State is inherently file-based:
- `tasks.json` - Task queue (atomic writes with file locking)
- `agents.json` - Agent registry
- `master-plan.md` - Project context
- `results/*.md` - Task results

**Persistence characteristics:**
- Write-on-change: Every state change writes to disk
- Crash recovery: Last written state is always current
- Backup: Manual or via filesystem snapshots

### Option B: MCP Server Persistence

Single state file with JSON serialization:
- `.coordination/mcp-state.json` - Complete server state

**Persistence characteristics:**
- Write-after-every-operation: `updateActivity()` saves state
- Atomic writes: Full state written at once
- Startup loading: State loaded on server start
- Crash recovery: Lose only in-flight operation

### Option C: Orchestrator Persistence

Pydantic models serialized to JSON:
- State file path configurable
- `save_state()` / `load_state()` methods

**Persistence characteristics:**
- Explicit saves: User controls when to persist
- Rich model: Full typing and validation
- Checkpoint support: Can save/restore at any point

### Common Schema

All options use compatible task schema:
```json
{
  "id": "task-xxx",
  "description": "...",
  "status": "available|claimed|in_progress|done|failed",
  "priority": 1,
  "claimed_by": "agent-id",
  "dependencies": ["task-yyy"],
  "context": {"files": [], "hints": ""},
  "result": {"output": "", "files_modified": []},
  "created_at": "ISO-8601",
  "claimed_at": "ISO-8601",
  "completed_at": "ISO-8601"
}
```

## Alternatives Considered

### Alternative 1: SQLite Database
Embedded SQL database for all options.
- **Pros**: ACID transactions, query capability
- **Cons**: Additional dependency, harder to inspect

### Alternative 2: Redis/External Store
Shared Redis instance for state.
- **Pros**: Fast, distributed capability
- **Cons**: Requires external service, operational complexity

### Alternative 3: Event Sourcing
Store events, derive state.
- **Pros**: Full audit trail, time-travel debugging
- **Cons**: Complex implementation, overkill for this use case

## Consequences

### Positive
- State survives restarts and crashes
- Can inspect/debug by reading files
- No external dependencies for persistence
- Migration between options possible (compatible schema)
- Easy backup (just copy files)

### Negative
- JSON parsing overhead on every read
- Large state files may slow down operations
- No query capability (must load all tasks)
- File corruption requires manual recovery

### Neutral
- State always visible on disk
- Manual editing possible (use with caution)
- Migration requires careful schema handling

## References

- [JSON Serialization](https://www.json.org)
- [Pydantic Models](https://docs.pydantic.dev)
