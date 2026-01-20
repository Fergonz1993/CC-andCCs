# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the Claude Multi-Agent Coordination System. ADRs document significant architectural decisions made during the development of this project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences. ADRs help:

- Record the reasoning behind decisions for future reference
- Onboard new team members by providing decision history
- Facilitate discussion about proposed changes
- Prevent revisiting already-decided issues without new information

## ADR Template

Each ADR follows this template:

```markdown
# ADR-XXX: Title

## Status
[Proposed | Accepted | Deprecated | Superseded by ADR-XXX]

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult to do because of this change?
```

## Index of ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](ADR-001-coordination-options.md) | Three Coordination Options | Accepted | 2024-01 |
| [ADR-002](ADR-002-leader-worker-pattern.md) | Leader-Worker Pattern | Accepted | 2024-01 |
| [ADR-003](ADR-003-file-based-coordination.md) | File-Based Coordination for Option A | Accepted | 2024-01 |
| [ADR-004](ADR-004-mcp-protocol.md) | MCP Protocol for Option B | Accepted | 2024-01 |
| [ADR-005](ADR-005-task-lifecycle.md) | Task Status Lifecycle | Accepted | 2024-01 |
| [ADR-006](ADR-006-state-persistence.md) | State Persistence Strategy | Accepted | 2024-01 |
| [ADR-007](ADR-007-race-condition-handling.md) | Race Condition Handling | Accepted | 2024-01 |

## Creating a New ADR

1. Copy the template from `ADR-TEMPLATE.md`
2. Name it `ADR-XXX-short-title.md` where XXX is the next number
3. Fill in all sections
4. Submit for review
5. Update this index once accepted
