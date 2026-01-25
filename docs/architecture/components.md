---
layout: default
title: Component Diagram
parent: Architecture
nav_order: 3
---

# Component Diagram

This document shows the relationships between components across all three coordination options.

---

## High-Level Component View

```mermaid
graph TB
    subgraph External["External Interfaces"]
        CLI[CLI Interface]
        API[MCP API]
        LIB[Python Library]
    end

    subgraph Core["Core Components"]
        subgraph OptionA["Option A: File-Based"]
            ACLI[coordination.py]
            AFL[File Locking]
            AFS[Filesystem I/O]
        end

        subgraph OptionB["Option B: MCP Server"]
            BMCP[MCP Server]
            BTOOLS[MCP Tools]
            BSTATE[State Manager]
        end

        subgraph OptionC["Option C: Orchestrator"]
            CORCH[Orchestrator]
            CASYNC[AsyncOrchestrator]
            CFILE[FileOrchestrator]
            CAGENT[Agent Manager]
        end
    end

    subgraph Shared["Shared Components"]
        SCHEMA[Task Schema]
        MODELS[Data Models]
        PROMPTS[System Prompts]
        SEC[Security]
        REL[Reliability]
        PERF[Performance]
        AI[AI Features]
    end

    subgraph Storage["Storage Layer"]
        TASKS[(tasks.json)]
        PLAN[(master-plan.md)]
        RESULTS[(results/)]
        LOGS[(logs/)]
        METRICS[(metrics/)]
    end

    CLI --> ACLI
    CLI --> CORCH
    API --> BMCP
    LIB --> CORCH

    ACLI --> AFL
    AFL --> AFS
    AFS --> TASKS
    AFS --> PLAN
    AFS --> RESULTS
    AFS --> LOGS

    BMCP --> BTOOLS
    BTOOLS --> BSTATE
    BSTATE --> TASKS

    CORCH --> CASYNC
    CORCH --> CFILE
    CASYNC --> CAGENT
    CFILE --> TASKS

    ACLI --> SCHEMA
    BMCP --> SCHEMA
    CORCH --> SCHEMA

    ACLI --> MODELS
    BMCP --> MODELS
    CORCH --> MODELS

    CORCH --> SEC
    CORCH --> REL
    CORCH --> PERF
    CORCH --> AI

    style OptionA fill:#e3f2fd
    style OptionB fill:#fff3e0
    style OptionC fill:#e8f5e9
```

---

## Option A: File-Based Components

```mermaid
graph TB
    subgraph CLI["CLI Layer"]
        MAIN[coordination.py]
        LEADER[Leader Commands]
        WORKER[Worker Commands]
        UTIL[Utility Commands]
    end

    subgraph Core["Core Layer"]
        TASK[Task Model]
        LOCK[File Lock Manager]
        LOG[Logger]
        BACKUP[Backup Manager]
        METRIC[Metrics Collector]
    end

    subgraph Storage["Storage Layer"]
        TASKS[(tasks.json)]
        PLAN[(master-plan.md)]
        DISC[(discoveries.md)]
        RES[(results/)]
        LOGS[(logs/)]
        CKPT[(checkpoints/)]
    end

    MAIN --> LEADER
    MAIN --> WORKER
    MAIN --> UTIL

    LEADER --> TASK
    WORKER --> TASK
    LEADER --> LOG
    WORKER --> LOG

    TASK --> LOCK
    LOCK --> TASKS
    TASK --> BACKUP
    BACKUP --> CKPT

    LOG --> LOGS
    LEADER --> PLAN
    WORKER --> RES
    WORKER --> DISC

    LEADER --> METRIC
    METRIC --> TASKS
```

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| CLI Parser | `coordination.py` | Command-line interface |
| Task Model | `coordination.py` | Task dataclass and operations |
| File Lock | `coordination.py` | `file_lock()` context manager |
| Backup Manager | `coordination.py` | `create_backup()`, `cleanup_old_backups()` |
| Metrics | `coordination.py` | `record_task_created()`, `get_metrics()` |

---

## Option B: MCP Server Components

```mermaid
graph TB
    subgraph Transport["Transport Layer"]
        STDIO[Stdio Transport]
        HTTP[HTTP Transport]
    end

    subgraph MCP["MCP Layer"]
        SERVER[MCP Server]
        TOOLS[Tool Registry]
        RESOURCES[Resource Registry]
    end

    subgraph Tools["Tool Implementations"]
        INIT[init_coordination]
        CREATE[create_task]
        CLAIM[claim_task]
        COMPLETE[complete_task]
        STATUS[get_status]
        DISC[add_discovery]
    end

    subgraph State["State Management"]
        COORD[CoordinationState]
        TASK[Task Manager]
        AGENT[Agent Manager]
        DISCOVER[Discovery Manager]
    end

    subgraph Storage["Persistence"]
        JSON[(state.json)]
    end

    STDIO --> SERVER
    HTTP --> SERVER

    SERVER --> TOOLS
    SERVER --> RESOURCES

    TOOLS --> INIT
    TOOLS --> CREATE
    TOOLS --> CLAIM
    TOOLS --> COMPLETE
    TOOLS --> STATUS
    TOOLS --> DISC

    INIT --> COORD
    CREATE --> TASK
    CLAIM --> TASK
    CLAIM --> AGENT
    COMPLETE --> TASK
    STATUS --> COORD
    DISC --> DISCOVER

    COORD --> JSON
    TASK --> COORD
    AGENT --> COORD
    DISCOVER --> COORD
```

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| MCP Server | `src/index.ts` | Server setup and transport |
| Tool Registry | `src/index.ts` | Tool definitions |
| State Manager | `src/index.ts` | In-memory state + persistence |
| Tool Handlers | `src/index.ts` | Individual tool implementations |

---

## Option C: Orchestrator Components

```mermaid
graph TB
    subgraph CLI["CLI Layer"]
        MAIN[cli.py]
        RUN[run command]
        INIT[init command]
        STAT[status command]
    end

    subgraph Orchestrator["Orchestrator Layer"]
        ORCH[orchestrator.py]
        ASYNC[async_orchestrator.py]
        FILE[FileOrchestrator]
    end

    subgraph Agent["Agent Layer"]
        AGENT[agent.py]
        PROC[Process Manager]
        HB[Heartbeat Monitor]
    end

    subgraph Models["Model Layer"]
        TASK[Task]
        STATE[CoordinationState]
        RESULT[TaskResult]
        CONTEXT[TaskContext]
        DISCOVERY[Discovery]
    end

    subgraph Support["Support Layer"]
        PLAN[planner.py]
        MON[monitor.py]
        ADV[advanced.py]
        REL[reliability.py]
        SEC[security_integration.py]
    end

    subgraph Config["Configuration"]
        CFG[config.py]
        LOG[logging_config.py]
    end

    MAIN --> RUN
    MAIN --> INIT
    MAIN --> STAT

    RUN --> ORCH
    INIT --> ORCH
    STAT --> ORCH

    ORCH --> ASYNC
    ORCH --> FILE
    ASYNC --> AGENT
    AGENT --> PROC
    AGENT --> HB

    ORCH --> TASK
    ORCH --> STATE
    ASYNC --> RESULT
    ASYNC --> CONTEXT
    ASYNC --> DISCOVERY

    ORCH --> PLAN
    ORCH --> MON
    ORCH --> ADV
    ORCH --> REL
    ORCH --> SEC

    ORCH --> CFG
    ORCH --> LOG
```

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Orchestrator | `orchestrator.py` | Main coordinator, backend selection |
| AsyncOrchestrator | `async_orchestrator.py` | Async execution, process spawning |
| FileOrchestrator | (internal) | File-based state management |
| Agent | `agent.py` | Claude Code subprocess management |
| Planner | `planner.py` | Task planning and decomposition |
| Monitor | `monitor.py` | Progress tracking and metrics |
| Models | `models.py` | Task, State, Result dataclasses |
| Config | `config.py` | Configuration management |

---

## Shared Components

```mermaid
graph TB
    subgraph Schemas["Schema Layer"]
        TASK_SCHEMA[task.schema.json]
    end

    subgraph Prompts["Prompt Layer"]
        LEADER_P[leader.md]
        WORKER_P[worker.md]
    end

    subgraph Security["Security Layer"]
        AUTH[Authentication]
        AUTHZ[Authorization]
        AUDIT[Audit Logging]
        ENC[Encryption]
        RATE[Rate Limiting]
    end

    subgraph Reliability["Reliability Layer"]
        CB[Circuit Breaker]
        RETRY[Retry Logic]
        FB[Fallback]
        LEADER_E[Leader Election]
    end

    subgraph Performance["Performance Layer"]
        CACHE[Caching]
        COMPRESS[Compression]
        ASYNC_IO[Async I/O]
        PROFILE[Profiling]
    end

    subgraph AI["AI Layer"]
        PRIORITY[Task Prioritization]
        DECOMP[Task Decomposition]
        ANOMALY[Anomaly Detection]
    end

    TASK_SCHEMA --> Security
    TASK_SCHEMA --> Reliability
    TASK_SCHEMA --> Performance
    TASK_SCHEMA --> AI
```

### Shared Component Locations

| Component | Directory | Purpose |
|-----------|-----------|---------|
| Task Schema | `shared/schemas/` | Canonical task JSON structure |
| System Prompts | `shared/prompts/` | Leader and worker prompts |
| Security | `shared/security/` | Auth, encryption, audit |
| Reliability | `shared/reliability/` | Circuit breaker, retry, fallback |
| Performance | `shared/performance/` | Caching, async I/O |
| AI Features | `shared/ai/` | Prioritization, decomposition |

---

## Cross-Option Integration

```mermaid
graph LR
    subgraph Options["Coordination Options"]
        A[Option A]
        B[Option B]
        C[Option C]
    end

    subgraph CrossOption["Cross-Option Support"]
        MIG[Migration Tool]
        SYNC[State Sync]
        CLI[Unified CLI]
        PLUGIN[Plugin System]
    end

    A <--> MIG
    B <--> MIG
    C <--> MIG

    A <--> SYNC
    B <--> SYNC
    C <--> SYNC

    CLI --> A
    CLI --> B
    CLI --> C

    PLUGIN --> A
    PLUGIN --> B
    PLUGIN --> C
```

### Cross-Option Tools

| Tool | Location | Purpose |
|------|----------|---------|
| Migration | `shared/cross_option/` | Migrate state between options |
| State Sync | `shared/cross_option/` | Synchronize state across options |
| Unified CLI | `shared/cross_option/` | Single CLI for all options |
| Plugin System | `shared/cross_option/` | Extensibility framework |

---

## Data Flow Between Components

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Orchestrator
    participant Agent
    participant Storage

    User->>CLI: orchestrate run "Goal"
    CLI->>Orchestrator: initialize(goal)
    Orchestrator->>Storage: Load/create state

    Orchestrator->>Agent: Spawn leader
    Agent->>Orchestrator: Task plan
    Orchestrator->>Storage: Save tasks

    loop For each worker
        Orchestrator->>Agent: Spawn worker
        Agent->>Orchestrator: Request task
        Orchestrator->>Storage: Claim task
        Storage-->>Orchestrator: Task details
        Orchestrator-->>Agent: Task details
        Agent->>Agent: Execute
        Agent->>Orchestrator: Complete task
        Orchestrator->>Storage: Update status
    end

    Orchestrator->>Storage: Final state
    Orchestrator-->>CLI: Results
    CLI-->>User: Summary
```

---

## See Also

- [System Overview](overview.md)
- [Task Lifecycle](task-lifecycle.md)
- [ADR Index](../adr/README.md)
