---
layout: default
title: System Overview
parent: Architecture
nav_order: 1
---

# System Architecture Overview

This document provides a high-level overview of the Claude Multi-Agent Coordination System architecture.

---

## System Overview Diagram

```mermaid
flowchart TB
    subgraph User["User Environment"]
        T1[Terminal 1<br/>LEADER]
        T2[Terminal 2<br/>WORKER]
        T3[Terminal 3<br/>WORKER]
        TN[Terminal N<br/>WORKER]
    end

    subgraph Coordination["Coordination Layer"]
        subgraph OptionA["Option A: File-Based"]
            FS[(Filesystem<br/>.coordination/)]
            FL[File Locking]
        end

        subgraph OptionB["Option B: MCP Server"]
            MCP[MCP Server]
            MS[(In-Memory State)]
            MP[JSON Persistence]
        end

        subgraph OptionC["Option C: Orchestrator"]
            ORCH[Orchestrator Process]
            PM[Process Manager]
            AS[(Async State)]
        end
    end

    subgraph Shared["Shared Components"]
        SCHEMA[Task Schema]
        PROMPTS[System Prompts]
        SEC[Security Module]
        REL[Reliability Module]
    end

    T1 --> OptionA
    T1 --> OptionB
    T1 --> OptionC

    T2 --> OptionA
    T2 --> OptionB
    T2 --> OptionC

    T3 --> OptionA
    T3 --> OptionB
    T3 --> OptionC

    TN --> OptionA
    TN --> OptionB
    TN --> OptionC

    OptionA --> Shared
    OptionB --> Shared
    OptionC --> Shared

    style T1 fill:#e1f5fe
    style T2 fill:#fff3e0
    style T3 fill:#fff3e0
    style TN fill:#fff3e0
```

---

## Core Architecture Principles

### 1. Leader-Worker Pattern

The system implements a leader-worker pattern where:

- **One Leader** (Terminal 1): Plans work, creates tasks, monitors progress, aggregates results
- **Multiple Workers** (Terminals 2+): Claim tasks, execute them, report results back

This pattern enables:
- Clear separation of concerns
- Parallel task execution
- Centralized coordination
- Scalable worker pool

### 2. Shared Filesystem as Communication Layer

All coordination options leverage the filesystem as the common ground:

```
.coordination/
├── master-plan.md          # Goal and approach
├── tasks.json              # Task queue (source of truth)
├── context/
│   └── discoveries.md      # Shared findings
├── logs/                   # Per-agent logs
│   ├── leader.log
│   ├── terminal-2.log
│   └── terminal-3.log
└── results/                # Task outputs
    ├── task-001.md
    └── task-002.md
```

### 3. Task-Centric Design

Tasks are the fundamental unit of work:

```json
{
  "id": "task-20240115-a1b2",
  "description": "Implement user login endpoint",
  "status": "available",
  "priority": 1,
  "dependencies": ["task-001"],
  "context": {
    "files": ["src/auth/login.ts"],
    "hints": "Use JWT for tokens"
  }
}
```

### 4. Option Flexibility

Three coordination options serve different needs:

| Aspect | Option A | Option B | Option C |
|--------|----------|----------|----------|
| Communication | File polling | MCP protocol | Process IPC |
| Latency | 1-2 seconds | Real-time | Real-time |
| Setup | None | MCP config | pip install |
| Control | Manual | Semi-auto | Full auto |
| Best for | Prototypes | Production | Automation |

---

## Component Interaction

### Option A: File-Based Flow

```mermaid
sequenceDiagram
    participant L as Leader
    participant FS as Filesystem
    participant W as Worker

    L->>FS: Write tasks.json
    L->>FS: Write master-plan.md

    loop Task Processing
        W->>FS: Read tasks.json
        W->>FS: Claim task (lock + update)
        W->>W: Execute task
        W->>FS: Write results/task-xxx.md
        W->>FS: Update tasks.json (done)
    end

    L->>FS: Read results/*
    L->>L: Aggregate results
```

### Option B: MCP Server Flow

```mermaid
sequenceDiagram
    participant L as Leader
    participant MCP as MCP Server
    participant W as Worker

    L->>MCP: init_coordination
    L->>MCP: create_tasks_batch

    loop Task Processing
        W->>MCP: claim_task
        MCP-->>W: Task details
        W->>W: Execute task
        W->>MCP: complete_task
        W->>MCP: add_discovery (if any)
    end

    L->>MCP: get_status
    L->>MCP: get_results
```

### Option C: Orchestrator Flow

```mermaid
sequenceDiagram
    participant U as User
    participant O as Orchestrator
    participant L as Leader Agent
    participant W1 as Worker 1
    participant W2 as Worker 2

    U->>O: orchestrate run "Goal"
    O->>L: Spawn leader process
    L->>O: Task plan
    O->>W1: Spawn worker
    O->>W2: Spawn worker

    par Worker 1
        O->>W1: Assign task
        W1->>O: Task complete
    and Worker 2
        O->>W2: Assign task
        W2->>O: Task complete
    end

    O->>U: Aggregated results
```

---

## Data Flow

### Task State Machine

```mermaid
stateDiagram-v2
    [*] --> available: Task created

    available --> claimed: Worker claims
    claimed --> in_progress: Worker starts
    claimed --> available: Timeout/release

    in_progress --> done: Success
    in_progress --> failed: Error
    in_progress --> available: Timeout

    failed --> available: Retry
    done --> [*]
    failed --> [*]: Max retries
```

### Dependency Resolution

Tasks with dependencies are only claimable when all dependencies are satisfied:

```mermaid
graph LR
    A[Task 1: Setup] --> C[Task 3: Implement]
    B[Task 2: Design] --> C
    C --> D[Task 4: Test]
    C --> E[Task 5: Document]

    style A fill:#90EE90
    style B fill:#90EE90
    style C fill:#FFE4B5
    style D fill:#FFFFFF
    style E fill:#FFFFFF
```

Legend:
- Green: Completed
- Orange: In Progress
- White: Blocked (dependencies not met)

---

## Security Architecture

```mermaid
flowchart LR
    subgraph Security["Security Layer"]
        AUTH[Authentication]
        AUTHZ[Authorization]
        AUDIT[Audit Logging]
        ENC[Encryption]
        RATE[Rate Limiting]
    end

    subgraph Coordination
        FS[File-Based]
        MCP[MCP Server]
        ORCH[Orchestrator]
    end

    AUTH --> FS
    AUTH --> MCP
    AUTH --> ORCH

    AUTHZ --> FS
    AUTHZ --> MCP
    AUTHZ --> ORCH

    AUDIT --> FS
    AUDIT --> MCP
    AUDIT --> ORCH
```

See [Security Guide](../guides/security.md) for detailed security practices.

---

## Reliability Architecture

```mermaid
flowchart TB
    subgraph Reliability["Reliability Components"]
        CB[Circuit Breaker]
        RETRY[Retry with Backoff]
        FB[Fallback]
        HB[Heartbeat]
        HEAL[Self-Healing]
    end

    subgraph State["State Management"]
        CKPT[Checkpoints]
        BACKUP[Backups]
        RECOVER[Recovery]
    end

    CB --> RETRY
    RETRY --> FB
    HB --> HEAL
    HEAL --> RECOVER
    CKPT --> RECOVER
    BACKUP --> RECOVER
```

See [ADR-007](../adr/ADR-007-race-condition-handling.md) for race condition handling details.

---

## Scaling Considerations

### Horizontal Scaling

- **Workers**: Add more terminal instances
- **Option B**: Single MCP server handles multiple workers
- **Option C**: Orchestrator spawns/manages workers automatically

### Vertical Scaling

- **Task Complexity**: Adjust timeouts for complex tasks
- **Queue Depth**: Monitor queue size metrics
- **Memory**: Option B/C keep state in memory

### Performance Tips

1. Keep tasks atomic (5-15 min each)
2. Minimize dependencies to enable parallelism
3. Use Option B/C for high-throughput scenarios
4. Monitor metrics for bottlenecks

See [Performance Tuning Guide](../guides/performance-tuning.md) for optimization strategies.

---

## Next Steps

- [Task Lifecycle](task-lifecycle.md) - Detailed task state documentation
- [Component Diagram](components.md) - Component relationships
- [ADRs](../adr/README.md) - Architecture Decision Records
