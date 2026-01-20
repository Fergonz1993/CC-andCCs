# ADR-007: Race Condition Handling

## Status

Accepted

## Date

2024-01-15

## Context

Multiple workers may attempt to claim the same task simultaneously. Without proper handling, this can lead to:
- Multiple workers working on the same task
- Lost updates when concurrent writes occur
- Inconsistent state between workers

Each coordination option must handle race conditions appropriately.

## Decision

Each option implements race condition prevention at the appropriate level:

### Option A: Optimistic Locking with Verification

File-based coordination uses a **claim-then-verify** pattern:

```python
def worker_claim(terminal_id: str):
    # 1. Acquire file lock
    with file_lock(TASKS_FILE, exclusive=True):
        # 2. Read current state
        data = load_tasks()

        # 3. Find and claim task
        for task in data["tasks"]:
            if task["status"] == "available":
                task["status"] = "claimed"
                task["claimed_by"] = terminal_id
                task["claimed_at"] = now_iso()
                break

        # 4. Write state
        save_tasks(data)

    # 5. CRITICAL: Re-read and verify claim succeeded
    data = load_tasks()
    for task in data["tasks"]:
        if task["id"] == task_id:
            if task["claimed_by"] == terminal_id:
                return task  # Success!
            else:
                # Another worker claimed it first
                return worker_claim(terminal_id)  # Try another task
```

**Key elements:**
- File locking (`fcntl.LOCK_EX`) prevents concurrent writes
- Re-read after claim detects if another worker won
- Retry logic finds another task if claim failed

### Option B: Server-Side Atomicity

MCP server handles coordination in-memory with single-threaded processing:

```typescript
function claimTask(agentId: string): Task | null {
    const available = getAvailableTasks();
    if (available.length === 0) return null;

    // Atomic: single-threaded, no concurrent access
    const task = available[0];
    task.status = "claimed";
    task.claimed_by = agentId;
    task.claimed_at = new Date().toISOString();

    updateActivity();  // Persist to disk
    return task;
}
```

**Key elements:**
- Single server process handles all requests
- No concurrent access to in-memory state
- MCP requests are processed sequentially
- Persistence happens after state change

### Option C: Async Locking

Python orchestrator uses asyncio locks:

```python
async def claim_task(self, agent_id: str) -> Optional[Task]:
    async with self._task_lock:
        available = self.state.get_available_tasks()
        if not available:
            return None

        task = available[0]
        task.claim(agent_id)
        return task
```

**Key elements:**
- `asyncio.Lock()` serializes access to task queue
- Async context manager ensures lock release
- State changes are atomic within lock

## Alternatives Considered

### Alternative 1: Database Transactions
Use database with transaction support.
- **Pros**: Well-understood semantics, ACID guarantees
- **Cons**: Requires database, more complex setup

### Alternative 2: Distributed Locking
Use distributed lock service (etcd, Redis).
- **Pros**: Works across machines
- **Cons**: External dependency, operational complexity

### Alternative 3: Version Numbers (CAS)
Compare-and-swap with version tracking.
- **Pros**: No locks needed, can detect conflicts
- **Cons**: Requires retry logic, potential livelock

## Consequences

### Positive
- Workers cannot claim the same task
- State remains consistent
- Each option uses appropriate mechanism
- Failures are detected and handled

### Negative
- File locking can fail on some filesystems (NFS)
- MCP server is single point of coordination
- Contention under high load may cause retries
- Complexity in verification logic

### Neutral
- Workers must handle claim failures gracefully
- Retry logic is standard pattern
- Logging helps debug coordination issues

## References

- [File Locking - Python fcntl](https://docs.python.org/3/library/fcntl.html)
- [Optimistic Concurrency Control](https://en.wikipedia.org/wiki/Optimistic_concurrency_control)
- [Python asyncio.Lock](https://docs.python.org/3/library/asyncio-sync.html)
