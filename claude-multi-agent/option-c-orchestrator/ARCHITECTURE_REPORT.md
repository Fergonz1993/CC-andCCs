# Option C Orchestrator - Architecture Report

**Date:** January 20, 2026
**Scope:** Complete analysis of the multi-agent coordination system
**Status:** Review Complete

---

## Executive Summary

Option C Orchestrator is a Python-based multi-agent coordination system that enables multiple Claude Code terminals to work together on complex tasks. It uses a **leader-worker pattern** where a leader agent plans and decomposes work, while worker agents execute tasks in parallel.

### Key Findings

| Category | Status | Details |
|----------|--------|---------|
| Architecture | ✅ Sound | Clean separation of concerns across 6 modules |
| Functionality | ✅ Complete | 10 advanced planning features implemented |
| Reliability | ⚠️ Issues | 4 race conditions, 2 resource leaks identified |
| Error Handling | ⚠️ Gaps | 3 silent failure points found |

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Analysis](#2-component-analysis)
3. [Data Flow](#3-data-flow)
4. [Task Lifecycle](#4-task-lifecycle)
5. [Critical Issues](#5-critical-issues)
6. [Recommendations](#6-recommendations)
7. [Appendix: Code References](#7-appendix-code-references)

---

## 1. Architecture Overview

### 1.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                │
│                        (cli.py)                                  │
│  Commands: run, init, from-config, create_task, status          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                           │
│          (orchestrator.py, async_orchestrator.py)               │
│  • Hybrid facade (file-backed vs async)                         │
│  • Task queue management                                         │
│  • Agent coordination                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌───────────────────┐ ┌─────────────┐ ┌─────────────────┐
│   Agent Layer     │ │  Planning   │ │   Data Models   │
│   (agent.py)      │ │ (planner.py)│ │   (models.py)   │
│                   │ │             │ │                 │
│ • ClaudeCodeAgent │ │ • TaskDAG   │ │ • Task          │
│ • AgentPool       │ │ • Critical  │ │ • Agent         │
│ • Subprocess mgmt │ │   Path      │ │ • Coordination  │
│                   │ │ • Resource  │ │   State         │
│                   │ │   Solver    │ │                 │
└───────────────────┘ └─────────────┘ └─────────────────┘
```

### 1.2 File Structure

```
src/orchestrator/
├── __init__.py
├── models.py              # Data structures (Task, Agent, CoordinationState)
├── orchestrator.py        # Hybrid orchestrator facade
├── async_orchestrator.py  # Production async orchestrator
├── agent.py               # Claude Code subprocess management
├── planner.py             # DAG-based task planning (10 features)
└── cli.py                 # Typer-based CLI interface
```

### 1.3 Design Patterns Used

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Facade** | `orchestrator.py:259-274` | Unified API for file-backed and async modes |
| **Factory** | `agent.py:469-482` | AgentPool creates leader/worker agents |
| **Observer** | `async_orchestrator.py:55-56` | Callbacks for task completion/discovery |
| **State Machine** | `models.py:14-22` | TaskStatus enum manages lifecycle |
| **Strategy** | `planner.py:436-523` | Resource solver uses list scheduling algorithm |

---

## 2. Component Analysis

### 2.1 Data Models (`models.py`)

**Purpose:** Define the core data structures using Pydantic for validation and serialization.

#### Key Classes

| Class | Lines | Purpose |
|-------|-------|---------|
| `TaskStatus` | 14-22 | Enum: PENDING, AVAILABLE, CLAIMED, IN_PROGRESS, DONE, FAILED, BLOCKED |
| `Task` | 52-116 | Unit of work with dependencies, priority, result tracking |
| `Agent` | 128-147 | Represents a Claude Code instance |
| `CoordinationState` | 160-214 | Complete system state snapshot |

#### Task Model Fields

```python
class Task(BaseModel):
    id: str                          # Auto-generated UUID
    description: str                 # What needs to be done
    status: TaskStatus               # Current lifecycle state
    priority: int                    # 1=highest, 10=lowest
    claimed_by: Optional[str]        # Agent ID
    dependencies: List[str]          # Task IDs that must complete first
    context: TaskContext             # Files, hints, metadata
    result: Optional[TaskResult]     # Output when complete
    attempts: int                    # Retry tracking
    max_attempts: int = 3            # Retry limit
```

#### Key Methods

- `can_start(completed_task_ids)` - Check if dependencies satisfied
- `claim(agent_id)` - Transition to CLAIMED state
- `complete(result)` - Transition to DONE state
- `fail(error)` - Transition to FAILED state
- `reset()` - Reset for retry (⚠️ race condition)

---

### 2.2 Orchestrator (`orchestrator.py`)

**Purpose:** Provide synchronous, file-backed coordination for tests and simple workflows.

#### Class Hierarchy

```
Orchestrator (Facade)
    │
    ├── FileOrchestrator (coordination_dir provided)
    │   └── Uses JSON files: tasks.json, agents.json, discoveries.json
    │
    └── AsyncOrchestrator (no coordination_dir)
        └── In-memory state with async execution
```

#### File-Based State Management

| File | Key | Contents |
|------|-----|----------|
| `tasks.json` | `tasks` | Array of task objects |
| `agents.json` | `agents` | Array of agent registrations |
| `discoveries.json` | `discoveries` | Shared findings between agents |
| `master-plan.md` | - | High-level goal description |

#### Key Operations

| Method | Lines | Description |
|--------|-------|-------------|
| `add_task()` | 105-128 | Create task with validation |
| `claim_task()` | 145-165 | Assign task to agent (⚠️ TOCTOU race) |
| `complete_task()` | 167-179 | Mark task done with result |
| `fail_task()` | 181-193 | Mark task failed with error |
| `register_agent()` | 199-217 | Register or reactivate agent |

---

### 2.3 Agent Management (`agent.py`)

**Purpose:** Spawn and manage Claude Code subprocesses.

#### ClaudeCodeAgent Class (Lines 28-446)

**Subprocess Spawning (Lines 99-126):**

The agent uses `asyncio.create_subprocess_exec()` which is the secure method for process creation - arguments are passed directly without shell interpretation, preventing injection vulnerabilities.

**Communication Protocol:**
1. Send prompt via stdin (line 205)
2. Read stream-json responses via stdout (lines 342-382)
3. Parse JSON messages with `type` field:
   - `type: "assistant"` → Extract text content
   - `type: "result"` → End of response

**Graceful Shutdown (Lines 155-171):**
```
stdin.close() → wait(5s) → terminate() → wait(2s) → kill()
```

#### AgentPool Class (Lines 449-531)

| Method | Purpose |
|--------|---------|
| `start_leader()` | Create leader agent singleton |
| `start_worker(id)` | Create worker up to `max_workers` |
| `stop_all()` | Graceful shutdown all agents |
| `get_idle_worker()` | Find worker not executing task (⚠️ race) |

---

### 2.4 Task Planner (`planner.py`)

**Purpose:** Advanced task planning with 10 features.

#### Feature Matrix

| ID | Feature | Class | Lines |
|----|---------|-------|-------|
| adv-c-plan-001 | DAG-based execution | `TaskDAG` | 76-237 |
| adv-c-plan-002 | Critical path analysis | `CriticalPathAnalyzer` | 245-366 |
| adv-c-plan-003 | Resource constraints | `ResourceConstraintSolver` | 383-545 |
| adv-c-plan-004 | Parallel optimization | `ParallelExecutionOptimizer` | 553-662 |
| adv-c-plan-005 | Affinity grouping | `AffinityGrouper` | 670-745 |
| adv-c-plan-006 | Milestone tracking | `MilestoneTracker` | 776-867 |
| adv-c-plan-007 | Plan versioning | `PlanVersionControl` | 886-980 |
| adv-c-plan-008 | What-if analysis | `ScenarioAnalyzer` | 1000-1108 |
| adv-c-plan-009 | Auto-adjustment | `PlanAdjuster` | 1126-1259 |
| adv-c-plan-010 | Export/import | `PlanExporter` | 1275-1399 |

#### Key Algorithms

**Cycle Detection (Lines 135-177):**
- Uses DFS with three-color marking (WHITE, GRAY, BLACK)
- Returns cycle path if found

**Topological Sort (Lines 185-219):**
- Kahn's algorithm
- O(V + E) complexity
- Deterministic ordering via alphabetical tie-breaking

**Critical Path Analysis (Lines 257-313):**
- Forward pass: Compute earliest start/finish
- Backward pass: Compute latest start/finish
- Slack = Latest Start - Earliest Start
- Critical path = tasks with zero slack

**Resource Scheduling (Lines 436-523):**
- List scheduling algorithm
- Priority based on slack (critical tasks first)
- Respects resource capacity constraints

---

### 2.5 Async Orchestrator (`async_orchestrator.py`)

**Purpose:** Production-mode orchestration with concurrent task execution.

#### Three-Phase Execution Model

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  PHASE 1:       │     │  PHASE 2:       │     │  PHASE 3:       │
│  PLANNING       │ ──▶ │  EXECUTION      │ ──▶ │  AGGREGATION    │
│                 │     │                 │     │                 │
│ Leader creates  │     │ Workers claim   │     │ Leader          │
│ task list from  │     │ and execute     │     │ summarizes      │
│ goal            │     │ tasks parallel  │     │ results         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

#### Task Distribution Loop (Lines 332-372)

```python
while self._running:
    # 1. Collect completed async tasks
    for completed in [t for t in active_tasks.values() if t.done()]:
        await complete_task() or fail_task()

    # 2. Check termination condition
    if no_pending and no_active:
        break

    # 3. Assign tasks to idle workers
    for worker in idle_workers:
        task = await claim_task(worker.id)  # Uses _task_lock
        asyncio.create_task(execute_on_worker(task))

    await asyncio.sleep(1)  # Polling interval
```

---

### 2.6 CLI Interface (`cli.py`)

**Purpose:** User entry point using Typer framework.

#### Commands

| Command | Usage | Description |
|---------|-------|-------------|
| `run` | `orchestrate run "Goal" -w 3` | Execute with leader planning |
| `run --no-plan` | `orchestrate run "Goal" --no-plan -f tasks.json` | Execute predefined tasks |
| `init` | `orchestrate init "Goal"` | Create config file |
| `from-config` | `orchestrate from-config config.json` | Run from config |
| `create_task` | `orchestrate create-task "Do X" -p 1` | Add task to config |
| `status` | `orchestrate status` | Show current config |
| `example` | `orchestrate example` | Show usage examples |

---

## 3. Data Flow

### 3.1 Complete Execution Flow

```
User: orchestrate run "Build an API" -w 3
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ CLI (cli.py:34-107)                                             │
│ • Parse: goal="Build an API", workers=3                         │
│ • Create AsyncOrchestrator                                      │
│ • asyncio.run(execute())                                        │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Initialize (async_orchestrator.py:89-108)                       │
│ • state.goal = "Build an API"                                   │
│ • state.created_at = now()                                      │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Start Agents (async_orchestrator.py:110-126)                    │
│ • AgentPool.start_leader()                                      │
│   └── ClaudeCodeAgent("leader").start()                         │
│       └── asyncio.create_subprocess_exec("claude", ...)         │
│ • AgentPool.start_worker("worker-1")                            │
│ • AgentPool.start_worker("worker-2")                            │
│ • AgentPool.start_worker("worker-3")                            │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Planning Phase (async_orchestrator.py:263-281)                  │
│ • leader.send_prompt(planning_prompt)                           │
│ • _parse_plan_response() → Extract JSON task array              │
│ • add_task() for each task                                      │
│                                                                 │
│ Example output from leader:                                     │
│ [                                                               │
│   {"description": "Set up Express server", "priority": 1},     │
│   {"description": "Create user routes", "priority": 2,         │
│    "dependencies": ["task-abc123"]},                            │
│   ...                                                           │
│ ]                                                               │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Execution Loop (async_orchestrator.py:332-372)                  │
│                                                                 │
│ Iteration 1:                                                    │
│   worker-1 claims task-abc123 (priority 1, no deps)             │
│   worker-2 claims task-def456 (priority 1, no deps)             │
│   worker-3 idle (no more priority-1 tasks without deps)         │
│                                                                 │
│ Iteration 2:                                                    │
│   task-abc123 completes → worker-1 claims task-ghi789           │
│   task-def456 still running                                     │
│   worker-3 still idle                                           │
│                                                                 │
│ ... continues until all tasks complete ...                      │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Aggregation (async_orchestrator.py:388-421)                     │
│ • Collect all TaskResults                                       │
│ • leader.send_prompt(aggregation_prompt)                        │
│ • Return summary                                                │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Cleanup (async_orchestrator.py:128-134)                         │
│ • AgentPool.stop_all()                                          │
│   └── Each agent: stdin.close() → terminate() → kill()         │
│ • Display results to user                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Task Lifecycle

### 4.1 State Machine

```
                    ┌──────────────────────────────────────────────────┐
                    │                                                  │
                    ▼                                                  │
┌─────────┐    ┌───────────┐    ┌─────────┐    ┌─────────────┐    ┌───┴───┐
│ PENDING │───▶│ AVAILABLE │───▶│ CLAIMED │───▶│ IN_PROGRESS │───▶│ DONE  │
└─────────┘    └───────────┘    └─────────┘    └─────────────┘    └───────┘
     │              │                │                │
     │              │                │                │
     │              ▼                ▼                ▼
     │         ┌─────────┐      ┌─────────┐      ┌────────┐
     └────────▶│ BLOCKED │      │  (retry)│─────▶│ FAILED │
               └─────────┘      └─────────┘      └────────┘
                    │                                  │
                    │     (dependencies satisfied)     │
                    └──────────────────────────────────┘
```

### 4.2 State Transitions

| From | To | Trigger | Method |
|------|-----|---------|--------|
| PENDING | AVAILABLE | Dependencies check | Automatic |
| PENDING | BLOCKED | Dependencies not met | Automatic |
| BLOCKED | AVAILABLE | Dependencies satisfied | `get_available_tasks()` |
| AVAILABLE | CLAIMED | Worker requests task | `claim_task()` |
| CLAIMED | IN_PROGRESS | Work begins | `task.start()` |
| IN_PROGRESS | DONE | Success | `task.complete(result)` |
| IN_PROGRESS | FAILED | Error | `task.fail(error)` |
| FAILED | AVAILABLE | Retry (attempts < max) | `task.reset()` |

---

## 5. Critical Issues

### 5.1 Race Conditions (Priority 1)

#### Issue RC-1: TOCTOU in `claim_task()`

**Location:** `orchestrator.py:145-165`

**Problem:** The claim_task function reads state, checks availability, then modifies - without holding a lock throughout. Another process could claim the same task between the check and the modification.

**Impact:** Two workers can claim the same task, leading to duplicate work.

**Fix:** Use file locking with `fcntl.flock()` or atomic rename pattern.

---

#### Issue RC-2: Missing Lock in Task Completion

**Location:** `async_orchestrator.py:198-230, 232-244`

**Problem:** The `complete_task()` and `fail_task()` methods don't use `self._task_lock` even though `claim_task()` does.

**Impact:** Concurrent modifications corrupt state.

**Fix:** Add async lock to both methods.

---

#### Issue RC-3: Race in `get_idle_worker()`

**Location:** `agent.py:519-524`

**Problem:** The method checks if a worker is idle and returns it, but the worker could be assigned between the check and return.

**Impact:** Same worker assigned to multiple tasks.

**Fix:** Use atomic assignment with lock or compare-and-swap.

---

#### Issue RC-4: Non-atomic `Task.reset()`

**Location:** `models.py:108-116`

**Problem:** Multiple fields modified without synchronization.

**Impact:** Task can be claimed mid-reset.

**Fix:** Use compare-and-swap or lock.

---

### 5.2 Resource Leaks (Priority 2)

#### Issue RL-1: Process Leak on Exception

**Location:** `agent.py:134-137`

**Problem:** Exception during start doesn't clean up partially started process.

**Impact:** Zombie processes accumulate.

**Fix:** Add cleanup in finally block or exception handler.

---

#### Issue RL-2: Stderr Task Leak

**Location:** `agent.py:430-446`

**Problem:** Background stderr reader task may not be cleaned on error.

**Impact:** Orphaned asyncio tasks.

**Fix:** Log exception, ensure task cleanup.

---

### 5.3 Error Handling (Priority 3)

#### Issue EH-1: Bare Exception Swallows Errors

**Location:** `agent.py:379, 445`

**Problem:** Bare `except:` blocks break without logging.

**Fix:** Log before breaking.

---

#### Issue EH-2: Corrupted JSON Loses Data

**Location:** `orchestrator.py:69-74`

**Problem:** Corrupted JSON silently returns empty list.

**Fix:** Create backup before overwriting.

---

#### Issue EH-3: Datetime Serialization

**Location:** `async_orchestrator.py:523-526`

**Problem:** Using `default=str` loses datetime precision.

**Fix:** Use explicit `.isoformat()` serialization.

---

### 5.4 Configuration Issues (Priority 4)

#### Issue CF-1: Hardcoded Model

**Location:** Multiple files

**Problem:** Model `"claude-sonnet-4-20250514"` hardcoded in 4 places.

**Fix:** Centralize in config module or environment variable.

---

#### Issue CF-2: No Worker Count Validation

**Location:** `async_orchestrator.py:49-79`

**Problem:** `max_workers` not validated to be >= 1.

**Fix:** Add validation.

---

### 5.5 Edge Cases (Priority 5)

#### Issue EC-1: Missing Parent Directory

**Location:** `cli.py:98-99`

**Problem:** File write fails if parent directory doesn't exist.

**Fix:** Use `Path.mkdir(parents=True, exist_ok=True)`.

---

## 6. Recommendations

### 6.1 Immediate Actions (P1)

1. **Add file locking to `FileOrchestrator.claim_task()`** using `fcntl.flock()`
2. **Add `_task_lock` to `complete_task()` and `fail_task()`** in async orchestrator
3. **Add process cleanup** in agent `start()` exception handler

### 6.2 Short-term Improvements (P2-P3)

4. **Add logging** throughout - replace print statements
5. **Create backup files** before JSON overwrites
6. **Validate configuration** (max_workers >= 1, timeout > 0)
7. **Use explicit datetime serialization** with `.isoformat()`

### 6.3 Long-term Enhancements

8. **Centralize configuration** in a config module
9. **Add metrics/observability** for production monitoring
10. **Add integration tests** for race conditions
11. **Consider using SQLite** instead of JSON files for file-based mode

---

## 7. Appendix: Code References

### 7.1 Key File Locations

| Component | File | Key Lines |
|-----------|------|-----------|
| Task model | `models.py` | 52-116 |
| Task status enum | `models.py` | 14-22 |
| File orchestrator | `orchestrator.py` | 40-257 |
| Claim task (race) | `orchestrator.py` | 145-165 |
| Claude agent | `agent.py` | 28-446 |
| Process spawn | `agent.py` | 99-126 |
| Agent pool | `agent.py` | 449-531 |
| Task DAG | `planner.py` | 76-237 |
| Critical path | `planner.py` | 245-366 |
| Async orchestrator | `async_orchestrator.py` | 38-540 |
| Task loop | `async_orchestrator.py` | 332-372 |
| CLI commands | `cli.py` | 33-316 |

### 7.2 Test Coverage Gaps

The following areas need additional testing:

- [ ] Race condition in `claim_task()` with multiple processes
- [ ] Process cleanup on exception
- [ ] Corrupted JSON file recovery
- [ ] Worker count edge cases (0, negative)
- [ ] Task dependency cycles
- [ ] Timeout handling

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-20 | Claude Opus 4.5 | Initial analysis |

---

*Report generated as part of Option C Orchestrator code review.*
