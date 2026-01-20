# ADR-003: File-Based Coordination for Option A

## Status

Accepted

## Date

2024-01-15

## Context

Option A needs a simple, zero-dependency coordination mechanism. Since all Claude Code instances share the same filesystem, we can leverage file operations for inter-process communication.

## Decision

Option A uses the filesystem as the coordination layer with the following structure:

```
.coordination/
├── master-plan.md          # Overall goal and approach (leader writes)
├── tasks.json              # Task queue - source of truth
├── agents.json             # Registered agents and capabilities
├── context/
│   └── discoveries.md      # Shared findings between agents
├── logs/
│   ├── leader.log          # Leader activity log
│   ├── terminal-2.log      # Worker 2 activity log
│   └── terminal-3.log      # Worker 3 activity log
└── results/
    ├── task-001.md         # Task 1 result
    └── task-002.md         # Task 2 result
```

### Key Design Decisions

1. **JSON for Task Queue**: Structured data for easy parsing and manipulation
2. **Markdown for Results**: Human-readable, easy to aggregate
3. **File Locking**: Use `fcntl` (Unix) or `msvcrt` (Windows) for atomic operations
4. **Polling**: Workers poll `tasks.json` at intervals (default 1-2 seconds)
5. **Claim Verification**: After claiming, re-read to verify no race condition occurred

### Task Claiming Protocol

```
1. Read tasks.json
2. Find available task with satisfied dependencies
3. Write claim (status=claimed, claimed_by=self)
4. Re-read tasks.json
5. If claimed_by != self, retry with different task
6. Else, proceed with execution
```

## Alternatives Considered

### Alternative 1: SQLite Database
Use SQLite for coordination state.
- **Pros**: ACID transactions, better concurrent access
- **Cons**: Additional dependency, harder to inspect manually

### Alternative 2: Named Pipes / FIFOs
Use OS-level IPC mechanisms.
- **Pros**: Lower latency, OS-managed
- **Cons**: Platform-specific, more complex setup

### Alternative 3: Socket-Based Communication
Local sockets for coordination.
- **Pros**: Real-time, bidirectional
- **Cons**: Requires running service, more complex

## Consequences

### Positive
- Zero external dependencies (just Python standard library)
- Easy to debug (can inspect files directly)
- Survives process restarts (state is persisted)
- Works on all platforms (with appropriate locking)
- Familiar concepts for developers

### Negative
- Polling adds latency (1-2 seconds typical)
- File locking can be tricky across platforms
- Race conditions possible if locking fails
- Not suitable for very high-frequency updates
- Disk I/O can be a bottleneck

### Neutral
- State is always visible on disk
- Manual intervention possible (edit JSON directly)

## References

- [File Locking in Python](https://docs.python.org/3/library/fcntl.html)
- [Advisory vs Mandatory Locking](https://en.wikipedia.org/wiki/File_locking)
