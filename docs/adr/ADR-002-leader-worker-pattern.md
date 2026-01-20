# ADR-002: Leader-Worker Pattern

## Status

Accepted

## Date

2024-01-15

## Context

Multi-agent systems require some form of coordination to prevent conflicts, ensure work distribution, and aggregate results. We need to decide how agents will organize themselves and who has authority over different aspects of the coordination.

## Decision

We adopt a **Leader-Worker Pattern** where:

### Leader Agent (Terminal 1)
- Plans and decomposes the overall goal into discrete tasks
- Creates and manages the task queue
- Monitors progress and handles failures
- Aggregates final results
- Has exclusive write access to certain coordination files (master plan, task creation)

### Worker Agents (Terminals 2, 3, ...)
- Read the master plan to understand context
- Claim available tasks from the queue
- Execute claimed tasks independently
- Report results and share discoveries
- Only update their own task status (not create new tasks)

### Coordination Rules
1. Only the leader creates new tasks
2. Workers can only modify tasks they have claimed
3. Workers must verify claims succeeded (race condition handling)
4. Discoveries are shared via a common file/mechanism

## Alternatives Considered

### Alternative 1: Peer-to-Peer Coordination
All agents are equal and coordinate through consensus.
- **Pros**: No single point of failure, more democratic
- **Cons**: Complex coordination, potential for conflicts, harder to reason about

### Alternative 2: Central Scheduler
A separate process (not Claude Code) manages all coordination.
- **Pros**: Clear separation of concerns, simpler agent logic
- **Cons**: Additional infrastructure, not self-contained

### Alternative 3: Self-Organizing Agents
Agents dynamically elect leaders and reorganize as needed.
- **Pros**: More flexible, handles leader failures
- **Cons**: Very complex, unpredictable behavior

## Consequences

### Positive
- Clear responsibility assignment
- Easy to reason about and debug
- Natural fit for how humans organize teams
- Leader can maintain overall coherence
- Workers operate independently, enabling parallelism

### Negative
- Leader is a single point of failure
- Leader can become a bottleneck for large task sets
- Uneven work distribution possible
- Less flexibility than peer-to-peer approaches

### Neutral
- Workers need to poll or subscribe for updates
- Task granularity affects efficiency (too fine = overhead, too coarse = underutilization)

## References

- [Leader-Follower Pattern](https://en.wikipedia.org/wiki/Leader/follower_pattern)
- [Multi-Agent Systems](https://en.wikipedia.org/wiki/Multi-agent_system)
