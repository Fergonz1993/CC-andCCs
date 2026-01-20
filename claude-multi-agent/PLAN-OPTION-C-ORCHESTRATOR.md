# OPTION C: External Orchestrator - Detailed Plan

## Executive Summary

Option C is a **Python-based orchestrator** that programmatically spawns, controls, and coordinates multiple Claude Code subprocess instances. You don't manually open terminals - the orchestrator does everything. Maximum automation, full programmatic control.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATOR                                    │
│                         (Python asyncio process)                             │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        CoordinationState                                │ │
│  │  goal: str                                                              │ │
│  │  master_plan: str                                                       │ │
│  │  tasks: List[Task]          ◄── Pydantic models                        │ │
│  │  agents: Dict[str, Agent]                                               │ │
│  │  discoveries: List[Discovery]                                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          AgentPool                                    │   │
│  │                                                                       │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │   │
│  │   │   Leader    │    │  Worker-1   │    │  Worker-2   │   ...        │   │
│  │   │   Agent     │    │   Agent     │    │   Agent     │              │   │
│  │   │             │    │             │    │             │              │   │
│  │   │ subprocess  │    │ subprocess  │    │ subprocess  │              │   │
│  │   │ stdin/out   │    │ stdin/out   │    │ stdin/out   │              │   │
│  │   └─────────────┘    └─────────────┘    └─────────────┘              │   │
│  │         │                  │                  │                       │   │
│  └─────────┼──────────────────┼──────────────────┼───────────────────────┘   │
│            │                  │                  │                           │
└────────────┼──────────────────┼──────────────────┼───────────────────────────┘
             │                  │                  │
             ▼                  ▼                  ▼
      ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
      │ claude CLI  │    │ claude CLI  │    │ claude CLI  │
      │  process    │    │  process    │    │  process    │
      │             │    │             │    │             │
      │ --print     │    │ --print     │    │ --print     │
      │ --output-   │    │ --output-   │    │ --output-   │
      │  format     │    │  format     │    │  format     │
      │  stream-json│    │  stream-json│    │  stream-json│
      └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Security Note

The orchestrator uses `asyncio.create_subprocess_exec()` which:
- Executes commands **without shell interpretation** (like Node.js `execFile`)
- Passes arguments directly to the process
- Prevents command injection vulnerabilities
- Is the recommended safe way to spawn subprocesses in Python

---

## Component Deep Dive

### 1. Pydantic Models (`models.py`)

Type-safe data structures with validation.

#### Task Model

```python
class TaskStatus(str, Enum):
    PENDING = "pending"
    AVAILABLE = "available"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"

class Task(BaseModel):
    id: str = Field(default_factory=lambda: f"task-{uuid4().hex[:8]}")
    description: str
    status: TaskStatus = TaskStatus.AVAILABLE
    priority: int = Field(default=5, ge=1, le=10)

    # Assignment
    claimed_by: Optional[str] = None

    # Dependencies
    dependencies: List[str] = Field(default_factory=list)

    # Context
    context: TaskContext = Field(default_factory=TaskContext)
    result: Optional[TaskResult] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    claimed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Retry tracking
    attempts: int = 0
    max_attempts: int = 3

    def can_start(self, completed_ids: set[str]) -> bool:
        """Check if dependencies are satisfied."""
        return all(dep in completed_ids for dep in self.dependencies)

    def claim(self, agent_id: str) -> None:
        self.status = TaskStatus.CLAIMED
        self.claimed_by = agent_id
        self.claimed_at = datetime.now()
        self.attempts += 1
```

#### TaskResult Model

```python
class TaskResult(BaseModel):
    output: str                              # Summary of work done
    files_modified: List[str] = []
    files_created: List[str] = []
    files_deleted: List[str] = []
    error: Optional[str] = None              # If failed
    subtasks_created: List[str] = []         # New tasks spawned
    discoveries: List[str] = []              # Important findings
```

#### Agent Model

```python
class AgentRole(str, Enum):
    LEADER = "leader"
    WORKER = "worker"
    SPECIALIST = "specialist"

class Agent(BaseModel):
    id: str
    role: AgentRole
    is_active: bool = False
    current_task: Optional[str] = None
    pid: Optional[int] = None
    last_heartbeat: datetime = Field(default_factory=datetime.now)
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)
```

### 2. ClaudeCodeAgent (`agent.py`)

Manages a single Claude Code subprocess.

#### Spawning a Process

```python
async def start(self) -> bool:
    """Start Claude Code subprocess (safe, no shell)."""
    self._process = await asyncio.create_subprocess_exec(
        "claude",                         # Command
        "--print",                        # Arg 1
        "--output-format", "stream-json", # Args 2-3
        "--model", self.model,            # Args 4-5
        "--verbose",                      # Arg 6
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=self.working_directory,
        env={**os.environ, "CLAUDE_CODE_AGENT_ID": self.agent_id},
    )
    self._is_running = True
    return True
```

#### Sending Prompts

```python
async def send_prompt(self, prompt: str, timeout: float = 600.0) -> str:
    """Send prompt and wait for complete response."""
    # Write to stdin
    prompt_bytes = (prompt + "\n").encode("utf-8")
    self._process.stdin.write(prompt_bytes)
    await self._process.stdin.drain()

    # Read response with timeout
    response = await asyncio.wait_for(
        self._read_response(),
        timeout=timeout
    )
    return response
```

#### Parsing Responses

```python
async def _read_response(self) -> str:
    """Read JSON stream from stdout."""
    response_parts = []

    while True:
        line = await self._process.stdout.readline()
        if not line:
            break

        decoded = line.decode("utf-8").strip()

        # Parse stream-json format
        try:
            data = json.loads(decoded)
            if data.get("type") == "assistant":
                # Extract text content
                for block in data.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        response_parts.append(block.get("text", ""))
            elif data.get("type") == "result":
                # End of response
                break
        except json.JSONDecodeError:
            response_parts.append(decoded)

    return "\n".join(response_parts)
```

#### Task Execution

```python
async def execute_task(self, task: Task) -> TaskResult:
    """Execute a task and return result."""
    self._current_task = task.id

    # Build prompt from task
    prompt = self._build_task_prompt(task)

    try:
        response = await self.send_prompt(prompt)
        result = self._parse_task_response(response, task)
        self._tasks_completed += 1
        return result
    except asyncio.TimeoutError:
        self._tasks_failed += 1
        return TaskResult(output="", error="Task timed out")
    finally:
        self._current_task = None
```

### 3. AgentPool

Manages multiple agents.

```python
class AgentPool:
    def __init__(self, working_directory: str, max_workers: int, model: str):
        self._agents: dict[str, ClaudeCodeAgent] = {}
        self._leader: Optional[ClaudeCodeAgent] = None

    async def start_leader(self) -> ClaudeCodeAgent:
        """Start the leader agent."""
        self._leader = ClaudeCodeAgent(
            agent_id="leader",
            role=AgentRole.LEADER,
            working_directory=self.working_directory,
        )
        await self._leader.start()
        return self._leader

    async def start_worker(self, worker_id: str = None) -> ClaudeCodeAgent:
        """Start a worker agent."""
        worker_id = worker_id or f"worker-{len(self._agents) + 1}"
        agent = ClaudeCodeAgent(
            agent_id=worker_id,
            role=AgentRole.WORKER,
            working_directory=self.working_directory,
        )
        await agent.start()
        self._agents[worker_id] = agent
        return agent

    def get_idle_worker(self) -> Optional[ClaudeCodeAgent]:
        """Get a worker not currently executing a task."""
        for agent in self._agents.values():
            if agent.is_running and agent._current_task is None:
                return agent
        return None

    async def stop_all(self) -> None:
        """Gracefully stop all agents."""
        tasks = [a.stop() for a in self._agents.values()]
        if self._leader:
            tasks.append(self._leader.stop())
        await asyncio.gather(*tasks, return_exceptions=True)
```

### 4. Orchestrator (`orchestrator.py`)

The main coordinator.

```python
class Orchestrator:
    def __init__(
        self,
        working_directory: str = ".",
        max_workers: int = 3,
        model: str = "claude-sonnet-4-20250514",
        task_timeout: int = 600,
    ):
        self.state = CoordinationState(
            working_directory=working_directory,
            max_parallel_workers=max_workers,
        )
        self._pool = AgentPool(working_directory, max_workers, model)
        self._running = False
        self._task_lock = asyncio.Lock()
```

---

## Execution Modes

### Mode 1: Leader-Driven Planning

The orchestrator asks the leader to create tasks automatically.

```python
async def run_with_leader_planning(self) -> dict:
    """Full automatic flow."""
    await self.start()

    try:
        # Phase 1: Leader creates plan
        plan_prompt = self._build_planning_prompt()
        leader = self._pool._leader
        plan_response = await leader.send_prompt(plan_prompt, timeout=300)
        tasks = self._parse_plan_response(plan_response)

        # Add tasks to queue
        for t in tasks:
            self.add_task(**t)

        # Phase 2: Execute tasks
        await self._run_task_loop()

        # Phase 3: Aggregate
        summary = await self._aggregate_results()

        return {
            "tasks_completed": ...,
            "summary": summary,
        }
    finally:
        await self.stop()
```

### Mode 2: Predefined Tasks

You provide the tasks, orchestrator executes them.

```python
async def run_with_predefined_tasks(self) -> dict:
    """Execute predefined tasks."""
    await self.start()
    try:
        await self._run_task_loop()
        summary = await self._aggregate_results()
        return {"summary": summary}
    finally:
        await self.stop()
```

---

## The Task Loop

The core execution engine:

```python
async def _run_task_loop(self) -> None:
    """Distribute tasks to workers until done."""
    active_tasks: dict[str, asyncio.Task] = {}  # task_id -> async task

    while self._running:
        # 1. Check for completed async tasks
        completed = [tid for tid, t in active_tasks.items() if t.done()]
        for tid in completed:
            try:
                result = active_tasks[tid].result()
                await self.complete_task(tid, result)
            except Exception as e:
                await self.fail_task(tid, str(e))
            del active_tasks[tid]

        # 2. Check if all done
        pending = [t for t in self.state.tasks
                   if t.status in (AVAILABLE, CLAIMED, IN_PROGRESS)]
        if not pending and not active_tasks:
            break  # All complete!

        # 3. Assign tasks to idle workers
        for worker in self._pool._agents.values():
            if not worker.is_running or worker._current_task:
                continue  # Worker busy

            task = await self.claim_task(worker.agent_id)
            if task:
                # Start async execution
                async_task = asyncio.create_task(
                    self._execute_task_on_worker(worker, task)
                )
                active_tasks[task.id] = async_task

        # 4. Show progress
        self._show_progress()
        await asyncio.sleep(1)
```

---

## CLI Interface (`cli.py`)

Built with Typer for a polished CLI.

### Commands

| Command | Description |
|---------|-------------|
| `orchestrate run GOAL` | Run with leader planning |
| `orchestrate run GOAL --no-plan --tasks FILE` | Run with predefined tasks |
| `orchestrate init GOAL` | Create config file |
| `orchestrate create-task DESC` | Add task to config |
| `orchestrate from-config FILE` | Run from config |
| `orchestrate status` | Show config status |
| `orchestrate example` | Show example usage |

### Examples

```bash
# Full automatic - leader plans, workers execute
orchestrate run "Build a REST API for user management" -w 3

# With predefined tasks
orchestrate run "Build API" --no-plan --tasks my-tasks.json

# Step by step
orchestrate init "Add authentication"
orchestrate create-task "Create User model" -p 1
orchestrate create-task "Implement login" -p 2 -d task-1
orchestrate from-config orchestration.json
```

---

## Config File Format

```json
{
  "goal": "Build a user authentication system",
  "working_directory": "/path/to/project",
  "max_workers": 3,
  "model": "claude-sonnet-4-20250514",
  "task_timeout": 600,
  "tasks": [
    {
      "description": "Create User model with Prisma",
      "priority": 1,
      "context_files": ["prisma/schema.prisma"],
      "hints": "Include id, email, passwordHash, timestamps"
    },
    {
      "description": "Implement password hashing utility",
      "priority": 1,
      "context_files": ["src/utils/password.ts"]
    },
    {
      "description": "Create login endpoint",
      "priority": 2,
      "dependencies": ["task-1", "task-2"]
    }
  ]
}
```

---

## Programmatic Usage

### Basic Example

```python
import asyncio
from orchestrator import Orchestrator

async def main():
    orch = Orchestrator(
        working_directory="./my-project",
        max_workers=3,
    )

    await orch.initialize("Build a REST API")

    # Add tasks
    orch.add_task("Create User model", priority=1)
    orch.add_task("Create API routes", priority=2)
    orch.add_task("Write tests", priority=3, dependencies=["task-1"])

    # Run
    result = await orch.run_with_predefined_tasks()
    print(f"Completed: {result['tasks_completed']}")

asyncio.run(main())
```

### Leader-Driven Example

```python
async def main():
    orch = Orchestrator(working_directory=".", max_workers=3)

    await orch.initialize(
        goal="Add dark mode to the application",
        master_plan="Create theme context, toggle, CSS vars, localStorage"
    )

    # Let leader create tasks automatically
    result = await orch.run_with_leader_planning()
    print(result['summary'])
```

### With Callbacks

```python
def on_task_complete(task):
    print(f"✓ {task.id}: {task.description[:30]}...")

def on_discovery(discovery):
    print(f"💡 {discovery.content[:50]}...")

async def main():
    orch = Orchestrator(working_directory=".")
    orch.on_task_complete = on_task_complete
    orch.on_discovery = on_discovery

    await orch.initialize("Refactor authentication")
    result = await orch.run_with_leader_planning()
```

---

## Execution Timeline

```
Time    Orchestrator              Leader              Worker-1            Worker-2
─────   ────────────              ──────              ────────            ────────
T0      start()
T1      spawn leader              [running]
T2      spawn worker-1                                [running]
T3      spawn worker-2                                                    [running]
T4      send planning prompt ───► analyzes...
T5                                returns tasks
T6      parse & queue tasks
T7      assign task-001 ─────────────────────────────► working
T8      assign task-002 ──────────────────────────────────────────────────► working
T9                                                    completes
T10     mark done, assign task-003 ──────────────────► working
T11                                                                       completes
T12     mark done, assign task-004 ────────────────────────────────────────► working
...
Tn      all done
Tn+1    aggregate with leader ───► summarizes
Tn+2    stop_all()                [stopped]           [stopped]           [stopped]
```

---

## Error Handling

| Error | Detection | Recovery |
|-------|-----------|----------|
| Task timeout | asyncio.TimeoutError | Mark failed, continue |
| Worker crash | Process exit | Restart worker, retry task |
| All workers busy | No idle workers | Wait, poll again |
| Dependency cycle | Detection on add | Reject task |
| Max retries | attempts >= max_attempts | Skip task, log error |

---

## State Persistence

```python
# Save for later
orch.save_state("progress.json")

# Resume later
orch.load_state("progress.json")
await orch.run_with_predefined_tasks()  # Continues from where it left off
```

---

## Advantages

1. **Full automation** - No manual terminal management
2. **Programmatic control** - Embed in scripts, CI/CD
3. **Type safety** - Pydantic models catch errors early
4. **True parallelism** - Concurrent task execution
5. **Rich CLI** - Beautiful terminal output
6. **Extensible** - Callbacks, custom logic

## Limitations

1. **Python dependency** - Requires Python 3.10+
2. **Process overhead** - Multiple Claude Code processes
3. **Memory usage** - Each agent is separate process
4. **Complexity** - More code than Options A/B

---

## File Listing

```
option-c-orchestrator/
├── pyproject.toml                    # Package config
├── requirements.txt                  # Dependencies
├── README.md                         # Quick start
├── demo-config.json                  # Example config
├── src/orchestrator/
│   ├── __init__.py                  # Package exports
│   ├── models.py                    # Pydantic models
│   ├── agent.py                     # ClaudeCodeAgent
│   ├── orchestrator.py              # Main Orchestrator
│   └── cli.py                       # Typer CLI
└── examples/
    ├── basic_usage.py
    └── leader_planning.py
```

---

## Quick Comparison

| Aspect | Option A | Option B | Option C |
|--------|----------|----------|----------|
| Automation | Manual | Semi-auto | **Full auto** |
| Setup | None | Build server | pip install |
| Terminals | Manual open | Manual open | **Auto spawned** |
| Control | Prompts | Tool calls | **Python API** |
| Parallelism | Manual | Manual | **Automatic** |
| Best for | Learning | Integration | **Production** |
