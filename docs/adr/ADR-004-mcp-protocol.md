# ADR-004: MCP Protocol for Option B

## Status

Accepted

## Date

2024-01-15

## Context

Option B needs real-time coordination without the latency of file polling. The Model Context Protocol (MCP) is the standard way for Claude Code to interact with external services and provides a natural fit for a coordination broker.

## Decision

Option B implements an MCP server that acts as a message broker for coordinating Claude Code instances.

### MCP Server Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Claude Code    │     │  Claude Code    │     │  Claude Code    │
│   (Leader)      │     │   (Worker 1)    │     │   (Worker 2)    │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │    MCP Protocol       │    MCP Protocol       │
         │   (stdio transport)   │   (stdio transport)   │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   MCP Coordination      │
                    │       Server            │
                    │                         │
                    │  • Task Queue           │
                    │  • Agent Registry       │
                    │  • Discovery Store      │
                    │  • State Persistence    │
                    └─────────────────────────┘
```

### MCP Tools Provided

| Tool | Role | Description |
|------|------|-------------|
| `init_coordination` | Leader | Set goal and master plan |
| `create_task` | Leader | Add a task to the queue |
| `create_tasks_batch` | Leader | Add multiple tasks at once |
| `get_status` | All | Get progress summary |
| `get_all_tasks` | All | List tasks with filters |
| `get_results` | Leader | Get completed task results |
| `register_agent` | All | Register as leader or worker |
| `claim_task` | Worker | Claim an available task |
| `start_task` | Worker | Mark task as in_progress |
| `complete_task` | Worker | Mark task done with results |
| `fail_task` | Worker | Mark task as failed |
| `heartbeat` | All | Signal agent is still active |
| `add_discovery` | All | Share a finding |
| `get_discoveries` | All | Get shared discoveries |
| `get_master_plan` | All | Get goal and plan |

### State Persistence

State is persisted to `.coordination/mcp-state.json` after every change, ensuring recovery after restarts.

## Alternatives Considered

### Alternative 1: REST API Server
Standard HTTP REST API for coordination.
- **Pros**: Widely understood, easy to debug with curl
- **Cons**: Not integrated with Claude Code, requires HTTP client

### Alternative 2: GraphQL API
GraphQL endpoint for flexible queries.
- **Pros**: Efficient data fetching, strong typing
- **Cons**: More complex, overkill for this use case

### Alternative 3: gRPC Service
Binary protocol for efficiency.
- **Pros**: High performance, streaming support
- **Cons**: Less accessible, requires code generation

## Consequences

### Positive
- Native integration with Claude Code
- Real-time coordination (no polling latency)
- Atomic operations (no race conditions)
- Rich tool ecosystem (typed inputs/outputs)
- Single server handles all coordination

### Negative
- Requires Node.js runtime
- MCP server must be running
- All instances must connect to same server
- Server is single point of failure
- More complex setup than file-based

### Neutral
- State still persisted to disk
- Can inspect state file manually
- Tools are self-documenting

## References

- [Model Context Protocol Specification](https://modelcontextprotocol.io)
- [MCP SDK for Node.js](https://www.npmjs.com/package/@modelcontextprotocol/sdk)
