# ADR-001: Three Coordination Options

## Status

Accepted

## Date

2024-01-15

## Context

Claude Code instances run in isolation with no native inter-process communication. We need a way to coordinate multiple Claude Code terminals working on complex tasks together. Different users have different needs:

- Some need a quick, simple solution for one-off tasks
- Some need production-grade coordination with real-time updates
- Some need full programmatic control for automation pipelines

A single solution cannot optimally serve all these use cases.

## Decision

We will provide **three coordination options**, each optimized for different use cases:

### Option A: File-Based Coordination
- Uses the shared filesystem as the coordination layer
- Workers poll for changes via JSON files
- Lowest complexity, easiest to understand and debug
- Best for quick prototypes and simple tasks

### Option B: MCP Server Broker
- Implements a Model Context Protocol (MCP) server
- All Claude Code instances connect to the same server
- Real-time updates via MCP tools
- Medium complexity, suitable for production workflows

### Option C: External Orchestrator
- Python-based process management
- Spawns and controls Claude Code instances programmatically
- Full automation support with dependency tracking
- Highest complexity, maximum control

## Alternatives Considered

### Alternative 1: Single Universal Solution
A single coordination approach that tries to satisfy all use cases.
- **Pros**: Simpler codebase, single learning curve
- **Cons**: Compromises on all fronts, complex configuration

### Alternative 2: Plugin Architecture
A core system with pluggable coordination backends.
- **Pros**: Extensible, clean separation
- **Cons**: More complex initial implementation, harder to maintain

## Consequences

### Positive
- Users can choose the option that best fits their needs
- Each option can be optimized for its specific use case
- Lower barrier to entry (start with Option A, grow into B/C)
- Easier to test and maintain isolated options

### Negative
- Three codebases to maintain
- Users must decide which option to use
- Some code duplication across options
- Documentation must cover all three

### Neutral
- Shared concepts (tasks, agents, discoveries) work across all options
- Common schema definitions can be reused

## References

- [Claude Code Documentation](https://claude.ai/code)
- [Model Context Protocol Specification](https://modelcontextprotocol.io)
