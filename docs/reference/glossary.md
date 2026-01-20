# Glossary of Terms

This glossary defines key terms and concepts used throughout the Claude Multi-Agent Coordination System documentation.

---

## A

### Agent
A Claude Code instance participating in coordination. Can be either a leader or worker.

### Agent ID
A unique identifier for an agent, typically including the terminal number (e.g., `terminal-2`, `worker-1`).

### Agent Pool
In Option C, the collection of Claude Code processes managed by the orchestrator.

### Available (Status)
Task status indicating the task is ready to be claimed by a worker.

---

## B

### Batch Operation
Creating or modifying multiple tasks in a single operation for efficiency.

---

## C

### Claim
The action of a worker agent reserving a task for execution. Transitions task from `available` to `claimed`.

### Claimed (Status)
Task status indicating a worker has reserved this task but hasn't started execution.

### Claude Code
Anthropic's command-line tool for AI-assisted coding. The agents in this system.

### Coordination Directory
The `.coordination/` directory containing all coordination state and files.

### Coordination Layer
The mechanism by which agents communicate. Options: file-based, MCP server, or orchestrator.

---

## D

### Dependencies
Tasks that must complete before another task can be claimed. Expressed as a list of task IDs.

### Discovery
An important finding shared between agents to maintain alignment. Can include code patterns found, bugs discovered, or architectural insights.

### Done (Status)
Task status indicating successful completion.

---

## E

### Exclusive Lock
File lock that prevents all other processes from accessing the file. Used during writes.

---

## F

### Failed (Status)
Task status indicating the task could not be completed successfully.

### File Lock
Mechanism to prevent concurrent access to files. Uses `fcntl` on Unix, `msvcrt` on Windows.

---

## G

### Goal
The overall objective of a coordination session. Set during initialization.

---

## H

### Heartbeat
Periodic signal from an agent indicating it's still active. Used for detecting stale workers.

### Hints
Contextual information provided with a task to help the worker. Part of task context.

---

## I

### In Progress (Status)
Task status indicating active execution by a worker.

### Idempotent
An operation that produces the same result regardless of how many times it's executed.

---

## L

### Leader
The coordinating agent (typically Terminal 1). Responsible for:
- Planning and decomposing work
- Creating tasks
- Monitoring progress
- Aggregating results

### Lifecycle
The progression of a task through statuses: available -> claimed -> in_progress -> done/failed.

---

## M

### Master Plan
High-level documentation of the goal, approach, and strategy. Stored in `master-plan.md`.

### MCP (Model Context Protocol)
Anthropic's protocol for Claude tools and integrations. Used by Option B for coordination.

### MCP Server
A server implementing the Model Context Protocol. In Option B, acts as the coordination broker.

---

## O

### Orchestrator
The Python class in Option C that manages the entire coordination process programmatically.

### Option A
File-based coordination. Simplest approach using shared filesystem.

### Option B
MCP Server coordination. Real-time coordination via Model Context Protocol.

### Option C
External Orchestrator coordination. Full programmatic control with process management.

---

## P

### Polling
Repeatedly checking for changes at intervals. Used by workers in Option A to detect new tasks.

### Priority
Task urgency level (1-10, where 1 is highest). Lower numbers are claimed first.

---

## R

### Race Condition
Situation where multiple agents attempt the same operation simultaneously, potentially causing conflicts.

### Result
The output of a completed task, including summary, modified files, and created files.

### Retry
Re-attempting a failed task, potentially with modifications.

---

## S

### Shared Lock
File lock that allows multiple readers but blocks writers.

### State
The complete coordination data including tasks, agents, discoveries, and configuration.

### State Persistence
Saving coordination state to disk for recovery and continuity.

### Stale Worker
A worker that hasn't sent a heartbeat within the timeout period.

### Status
Current state of a task in its lifecycle (available, claimed, in_progress, done, failed).

### Subtask
A task created during the execution of another task, typically to handle unexpected complexity.

---

## T

### Task
A discrete unit of work to be executed by a worker. Contains:
- ID (unique identifier)
- Description (what to do)
- Status (lifecycle state)
- Priority (urgency)
- Dependencies (prerequisite tasks)
- Context (helpful information)
- Result (output when complete)

### Task Context
Additional information provided with a task:
- Files: Relevant file paths
- Hints: Suggestions for the worker
- Parent Task: For subtasks

### Task Queue
The collection of all tasks, managed by the coordination layer.

### Terminal
A shell/command-line instance. Each Claude Code instance runs in its own terminal.

### Throughput
Rate of task completion, typically measured in tasks per minute.

### Timeout
Maximum duration for an operation before it's considered failed.

---

## V

### Verification
The step after claiming where a worker re-reads state to confirm their claim succeeded.

---

## W

### Worker
An agent that executes tasks (typically Terminals 2, 3, etc.). Responsible for:
- Claiming available tasks
- Executing task requirements
- Reporting results
- Sharing discoveries

### Working Directory
The base directory for the project being worked on.

---

## Notation

### Task ID Format
`task-{timestamp}-{random}` (e.g., `task-20240115-a1b2`)

### ISO 8601
Date/time format used throughout: `YYYY-MM-DDTHH:MM:SS.sssZ`

### Priority Scale
1-10 where:
- 1-3: High priority (do first)
- 4-6: Medium priority (normal)
- 7-10: Low priority (do last)

### Status Values
- `available`: Ready to claim
- `claimed`: Reserved by worker
- `in_progress`: Being executed
- `done`: Successfully completed
- `failed`: Execution failed
- `cancelled`: Cancelled by leader (optional)
