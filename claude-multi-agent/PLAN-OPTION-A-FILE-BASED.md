# OPTION A: File-Based Coordination - Detailed Plan

## Executive Summary

Option A uses the **filesystem as a message queue**. Multiple Claude Code instances coordinate by reading and writing to shared JSON and Markdown files. No external dependencies, no servers to run - just files.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SHARED FILESYSTEM                            │
│                                                                     │
│  .coordination/                                                     │
│  ├── master-plan.md      ←── Leader writes overall strategy        │
│  ├── tasks.json          ←── Source of truth for all tasks         │
│  ├── context/                                                       │
│  │   └── discoveries.md  ←── Shared knowledge base                 │
│  ├── logs/                                                          │
│  │   ├── leader.log      ←── Leader activity log                   │
│  │   ├── terminal-2.log  ←── Worker 2 activity log                 │
│  │   └── terminal-3.log  ←── Worker 3 activity log                 │
│  └── results/                                                       │
│      ├── task-001.md     ←── Completed task results                │
│      └── task-002.md                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
    │Terminal 1│          │Terminal 2│          │Terminal 3│
    │  LEADER  │          │  WORKER  │          │  WORKER  │
    │          │          │          │          │          │
    │Claude    │          │Claude    │          │Claude    │
    │Code      │          │Code      │          │Code      │
    └──────────┘          └──────────┘          └──────────┘
```

---

## Component Details

### 1. The Coordination Directory (`.coordination/`)

This is the "shared memory" between all agents.

| File/Directory | Purpose | Who Writes | Who Reads |
|----------------|---------|------------|-----------|
| `master-plan.md` | High-level goal and strategy | Leader only | Everyone |
| `tasks.json` | Task queue with status tracking | Leader creates, Workers update status | Everyone |
| `context/discoveries.md` | Important findings to share | Anyone | Everyone |
| `logs/{agent-id}.log` | Activity audit trail | Each agent writes their own | Anyone (debugging) |
| `results/{task-id}.md` | Completed work output | Worker who completed task | Leader for aggregation |

### 2. Task Lifecycle

```
┌──────────────┐
│   CREATED    │  Leader creates task in tasks.json
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  AVAILABLE   │  Task ready to be claimed (dependencies satisfied)
└──────┬───────┘
       │  Worker claims by updating status + claimed_by
       ▼
┌──────────────┐
│   CLAIMED    │  Worker has reserved this task
└──────┬───────┘
       │  Worker starts execution
       ▼
┌──────────────┐
│ IN_PROGRESS  │  Work is actively being done
└──────┬───────┘
       │
       ├─────────────────────┐
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│     DONE     │      │    FAILED    │
└──────────────┘      └──────────────┘
```

### 3. Task JSON Schema

```json
{
  "version": "1.0",
  "created_at": "2025-01-19T20:00:00Z",
  "last_updated": "2025-01-19T20:15:00Z",
  "tasks": [
    {
      "id": "task-20250119200000-a1b2",
      "description": "Implement user validation function",
      "status": "available",
      "priority": 1,
      "claimed_by": null,
      "dependencies": [],
      "context": {
        "files": ["src/models/user.ts", "src/utils/validation.ts"],
        "hints": "Use zod for schema validation"
      },
      "result": null,
      "created_at": "2025-01-19T20:00:00Z",
      "claimed_at": null,
      "completed_at": null
    },
    {
      "id": "task-20250119200001-c3d4",
      "description": "Write unit tests for validation",
      "status": "available",
      "priority": 2,
      "claimed_by": null,
      "dependencies": ["task-20250119200000-a1b2"],
      "context": {
        "files": ["src/__tests__/"],
        "hints": "Cover edge cases: empty strings, invalid emails"
      },
      "result": null,
      "created_at": "2025-01-19T20:00:00Z",
      "claimed_at": null,
      "completed_at": null
    }
  ]
}
```

---

## Detailed Workflow

### Phase 1: Initialization (Leader - Terminal 1)

**Step 1.1: Create coordination structure**
```bash
python coordination.py leader init "Build a REST API for user management"
```

This creates:
- `.coordination/` directory tree
- `master-plan.md` with the goal
- Empty `tasks.json`
- `context/discoveries.md` placeholder

**Step 1.2: Leader analyzes and plans**

The leader (Claude Code in Terminal 1) should:
1. Read the codebase to understand existing patterns
2. Break down the goal into discrete tasks
3. Identify dependencies between tasks
4. Estimate priorities

**Step 1.3: Create tasks**
```bash
# High priority, no dependencies - can start immediately
python coordination.py leader add-task "Create User model with id, name, email, passwordHash fields" \
  -p 1 \
  --files src/models/ \
  --hints "Use Prisma schema, add timestamps"

# High priority, no dependencies - can run in parallel with above
python coordination.py leader add-task "Implement password hashing utility with bcrypt" \
  -p 1 \
  --files src/utils/ \
  --hints "Use bcrypt, 12 rounds"

# Medium priority, depends on both above tasks
python coordination.py leader add-task "Create POST /users registration endpoint" \
  -p 2 \
  --depends task-20250119200000-xxxx task-20250119200001-yyyy \
  --files src/routes/users.ts \
  --hints "Validate input, hash password, return user without password"

# Lower priority, depends on registration
python coordination.py leader add-task "Write integration tests for user registration" \
  -p 3 \
  --depends task-20250119200002-zzzz \
  --files src/__tests__/ \
  --hints "Test success case, duplicate email, invalid input"
```

**Step 1.4: Verify task queue**
```bash
python coordination.py leader status
```

### Phase 2: Worker Onboarding (Terminals 2 & 3)

**Step 2.1: Worker reads context**

Each worker should first understand the project:

```bash
# Read the master plan
cat .coordination/master-plan.md

# Read any shared discoveries
cat .coordination/context/discoveries.md

# See available tasks
python coordination.py worker list
```

**Step 2.2: Claim a task**
```bash
python coordination.py worker claim terminal-2
```

This atomically:
1. Reads `tasks.json`
2. Finds highest-priority available task with satisfied dependencies
3. Updates status to "claimed" and sets `claimed_by`
4. Re-reads to verify claim succeeded (race condition check)

**Step 2.3: Start working**
```bash
python coordination.py worker start terminal-2 task-20250119200000-xxxx
```

Updates status to "in_progress".

### Phase 3: Task Execution (Workers)

**Step 3.1: Do the actual work**

The worker (Claude Code) now:
1. Reads the task description and context
2. Examines relevant files
3. Implements the solution
4. Tests their work

**Step 3.2: Document discoveries**

If the worker learns something important:
```bash
echo "## Discovery: Database Connection

Found that the database connection pool is configured in src/config/db.ts.
Max connections: 10. Connection timeout: 5000ms.

This affects how we should handle concurrent user registrations.
" >> .coordination/context/discoveries.md
```

**Step 3.3: Complete the task**
```bash
python coordination.py worker complete terminal-2 task-20250119200000-xxxx \
  "Implemented User model with Prisma schema including id, name, email, passwordHash, createdAt, updatedAt fields. Added database migration." \
  --modified prisma/schema.prisma \
  --created src/models/user.ts prisma/migrations/20250119_add_user/
```

This:
1. Updates task status to "done"
2. Writes detailed result to `.coordination/results/task-xxx.md`
3. Logs the completion

**Step 3.4: Claim next task**

Worker repeats from Step 2.2 until no tasks remain.

### Phase 4: Aggregation (Leader)

**Step 4.1: Monitor progress**
```bash
# Continuous monitoring
watch -n 5 'python coordination.py leader status'
```

**Step 4.2: Handle failures**

If a task fails, the leader can:
```bash
# Check what went wrong
cat .coordination/results/task-xxx.md

# Reset the task for retry (if appropriate)
# Edit tasks.json to set status back to "available"
```

**Step 4.3: Aggregate results**
```bash
python coordination.py leader aggregate
```

Creates `.coordination/summary.md` with all results combined.

**Step 4.4: Final review**

Leader reviews the aggregate, runs tests, makes final adjustments.

---

## Race Condition Handling

### The Problem

Two workers might try to claim the same task simultaneously:

```
Time    Terminal-2              Terminal-3
────    ──────────              ──────────
T1      Read tasks.json         Read tasks.json
T2      See task-001 available  See task-001 available
T3      Update to claimed       Update to claimed
T4      Write tasks.json        Write tasks.json
T5      ??? Who wins ???
```

### The Solution

The `coordination.py` script uses **file locking**:

```python
@contextmanager
def file_lock(filepath: Path, exclusive: bool = True):
    lock_file = filepath.with_suffix(filepath.suffix + ".lock")
    lock_file.touch(exist_ok=True)

    with open(lock_file, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

Additionally, after claiming, the worker **re-reads** to verify:

```python
# Claim the task
data["tasks"][i]["claimed_by"] = terminal_id
save_tasks(data)

# Verify claim succeeded
data = load_tasks()
if data["tasks"][i]["claimed_by"] == terminal_id:
    # Success!
else:
    # Someone else got it, try again
    return worker_claim(terminal_id)
```

---

## Prompts for Claude Code

### Leader Prompt (Terminal 1)

```
You are the LEAD AGENT in a multi-agent coordination system.

## Your Responsibilities
1. Plan and decompose work into discrete tasks
2. Create tasks using: python coordination.py leader add-task "description" -p PRIORITY
3. Monitor progress: python coordination.py leader status
4. Aggregate results when complete

## Coordination Files
- .coordination/master-plan.md - Your high-level plan (you write this)
- .coordination/tasks.json - Task queue (you create tasks, workers update status)
- .coordination/context/discoveries.md - Shared knowledge (read and contribute)
- .coordination/results/ - Completed work (read to aggregate)

## Task Design Guidelines
- Each task should be completable in 5-15 minutes
- Include relevant file paths in --files
- Use --depends for ordering constraints
- Priority 1 = highest, 10 = lowest

## Current Goal
{PASTE YOUR GOAL HERE}

Start by analyzing the codebase, then create a task breakdown.
```

### Worker Prompt (Terminals 2 & 3)

```
You are WORKER AGENT "{TERMINAL_ID}" in a multi-agent coordination system.

## Your Responsibilities
1. Claim available tasks: python coordination.py worker claim {TERMINAL_ID}
2. Execute the task fully
3. Complete with results: python coordination.py worker complete {TERMINAL_ID} TASK_ID "summary" --modified FILE1 --created FILE2

## Coordination Files
- .coordination/master-plan.md - Read for overall context
- .coordination/tasks.json - Task queue (claim and update your tasks)
- .coordination/context/discoveries.md - Read for context, add important findings
- .coordination/results/ - Write your completed work here

## Work Loop
1. Read master-plan.md and discoveries.md for context
2. Run: python coordination.py worker list
3. Claim a task
4. Execute it completely
5. Document results
6. Repeat until no tasks remain

## Guidelines
- Only work on YOUR claimed task
- If you discover something important, add it to discoveries.md
- If a task is unclear, complete what you can and note issues in the result
- Don't modify files outside your task's scope

Start by reading the master plan, then claim your first task.
```

---

## Failure Modes and Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Worker crashes mid-task | Task stuck in "in_progress" | Leader manually resets task status |
| Worker claims but never starts | Task stuck in "claimed" for too long | Leader resets after timeout |
| File corruption | JSON parse error | Restore from git or manual fix |
| Dependency deadlock | Circular dependency | Leader restructures task dependencies |
| All workers finish, tasks remain | Available tasks with unsatisfied deps | Leader checks and fixes dependencies |

---

## Monitoring Commands

```bash
# Watch task status live
watch -n 2 'python coordination.py leader status'

# See all logs
tail -f .coordination/logs/*.log

# Check specific task result
cat .coordination/results/task-xxx.md

# Count tasks by status
jq '.tasks | group_by(.status) | map({status: .[0].status, count: length})' .coordination/tasks.json
```

---

## Advantages

1. **Zero dependencies** - Just Python and filesystem
2. **Transparent** - All state is human-readable files
3. **Debuggable** - Can manually inspect/edit any file
4. **Git-friendly** - Can commit coordination state
5. **Offline** - Works without network

## Limitations

1. **Polling latency** - Workers must poll for new tasks (1-2s delay)
2. **Manual race handling** - File locks help but aren't perfect
3. **No real-time updates** - Must refresh to see changes
4. **Single machine** - Requires shared filesystem

---

## File Listing

```
option-a-file-based/
├── coordination.py      # Main CLI (500 lines)
├── worker-loop.sh       # Auto-polling watcher script
├── demo.sh              # Demo script
└── .coordination/       # Created at runtime
    ├── master-plan.md
    ├── tasks.json
    ├── context/
    │   └── discoveries.md
    ├── logs/
    └── results/
```
