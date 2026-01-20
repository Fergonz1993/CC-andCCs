# ADR-005: Task Status Lifecycle

## Status

Accepted

## Date

2024-01-15

## Context

Tasks progress through various states from creation to completion. We need a clear, consistent lifecycle that handles normal completion, failures, and edge cases.

## Decision

Tasks follow this status lifecycle:

```
                    ┌──────────────────────────────────────┐
                    │                                      │
                    ▼                                      │
┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────┐ │
│ created │───>│available│───>│   claimed   │───>│ done │ │
└─────────┘    └────┬────┘    └──────┬──────┘    └──────┘ │
                    │                │                     │
                    │                ▼                     │
                    │         ┌─────────────┐              │
                    │         │ in_progress │──────────────┘
                    │         └──────┬──────┘
                    │                │
                    │                ▼
                    │         ┌─────────┐
                    └────────>│ failed  │
                              └────┬────┘
                                   │
                                   ▼ (optional retry)
                              ┌──────────┐
                              │available │
                              └──────────┘
```

### Status Definitions

| Status | Description | Who Sets It |
|--------|-------------|-------------|
| `available` | Task is ready to be claimed | Leader (on create), System (on retry) |
| `claimed` | A worker has claimed this task | Worker |
| `in_progress` | Worker is actively working on task | Worker |
| `done` | Task completed successfully | Worker |
| `failed` | Task failed during execution | Worker |
| `cancelled` | Task was cancelled (optional) | Leader |

### Transition Rules

1. **available -> claimed**: Worker calls `claim_task` with their agent_id
2. **claimed -> in_progress**: Worker calls `start_task` after setup
3. **in_progress -> done**: Worker calls `complete_task` with results
4. **in_progress -> failed**: Worker calls `fail_task` with error
5. **failed -> available**: System retries (if retries remaining)
6. **any -> cancelled**: Leader cancels task (cleanup required)

### Task Metadata by State

```json
{
  "id": "task-xxx",
  "description": "...",
  "status": "in_progress",
  "priority": 1,
  "claimed_by": "terminal-2",      // Set on claim
  "claimed_at": "2024-01-15T...",  // Set on claim
  "started_at": "2024-01-15T...",  // Set on start (optional)
  "completed_at": "2024-01-15T...", // Set on done/failed
  "result": {...}                   // Set on done/failed
}
```

## Alternatives Considered

### Alternative 1: Simple Two-State (pending/done)
Just pending and done states.
- **Pros**: Simple
- **Cons**: No visibility into progress, hard to handle failures

### Alternative 2: More Granular States
Additional states like queued, validated, reviewing.
- **Pros**: More visibility
- **Cons**: Complex transitions, overkill for most cases

### Alternative 3: Custom State Machine
User-defined states and transitions.
- **Pros**: Maximum flexibility
- **Cons**: Inconsistent, hard to build tooling around

## Consequences

### Positive
- Clear progression visible at all times
- Easy to identify stuck or failed tasks
- Supports retry logic naturally
- Workers know exact expectations
- Enables progress tracking/metrics

### Negative
- Must handle all state transitions correctly
- Invalid transitions should be rejected
- More metadata to track and persist

### Neutral
- All options use the same lifecycle
- Timestamps enable duration tracking
- Result storage consistent across states

## References

- [Finite State Machines](https://en.wikipedia.org/wiki/Finite-state_machine)
- [Task Queues](https://en.wikipedia.org/wiki/Job_queue)
