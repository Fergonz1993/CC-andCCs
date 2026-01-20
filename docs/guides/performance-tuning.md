# Performance Tuning Guide

This guide covers strategies for optimizing the Claude Multi-Agent Coordination System for your specific workloads.

## Table of Contents

1. [Understanding Performance](#understanding-performance)
2. [Task Design Optimization](#task-design-optimization)
3. [Option-Specific Tuning](#option-specific-tuning)
4. [Scaling Strategies](#scaling-strategies)
5. [Monitoring and Metrics](#monitoring-and-metrics)
6. [Benchmarking](#benchmarking)

---

## Understanding Performance

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Task Throughput** | Tasks completed per minute | Depends on task complexity |
| **Claim Latency** | Time from available to claimed | < 2 seconds (Option A), < 100ms (B, C) |
| **Coordination Overhead** | Time spent on coordination vs. work | < 10% of total time |
| **Worker Utilization** | % time workers are actively working | > 80% |
| **Queue Depth** | Number of available tasks waiting | Low and stable |

### Performance Factors

```
Total Time = Task Execution Time + Coordination Overhead + Wait Time

Where:
- Task Execution Time: Time Claude Code spends on actual work
- Coordination Overhead: Time spent claiming, reporting, syncing
- Wait Time: Time waiting for dependencies or available tasks
```

---

## Task Design Optimization

### Right-Sizing Tasks

**Too Small (< 2 minutes)**
- High coordination overhead relative to work
- Frequent context switching
- Many task claims/completions

**Too Large (> 20 minutes)**
- Reduced parallelization opportunity
- Higher risk of timeout
- Slower feedback loop

**Optimal (5-15 minutes)**
- Good balance of work vs. overhead
- Allows meaningful parallel execution
- Manageable failure scope

### Dependency Optimization

**Anti-Pattern: Sequential Chain**
```
Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 5
```
Only one worker active at a time.

**Better: Diamond Pattern**
```
        Task 2 ---\
Task 1 <           > Task 4
        Task 3 ---/
```
Multiple workers active after Task 1.

**Best: Wide Parallelism**
```
        Task 2
        Task 3
Task 1  Task 4  Task 6
        Task 5
```
Maximize concurrent work.

### Batching Strategy

```python
# Instead of many tiny tasks
for file in files:
    add_task(f"Process {file}")  # 100 tasks, 30 seconds each

# Batch related files
for batch in chunks(files, 10):
    add_task(f"Process {', '.join(batch)}")  # 10 tasks, 5 minutes each
```

---

## Option-Specific Tuning

### Option A: File-Based

#### Reduce File I/O

```python
# Bad: Reading entire file for each check
while True:
    tasks = load_tasks()  # Full file read
    available = get_available(tasks)
    if not available:
        time.sleep(1)
    ...

# Better: Check file modification time first
last_mtime = 0
cached_tasks = None

while True:
    current_mtime = os.path.getmtime(TASKS_FILE)
    if current_mtime > last_mtime:
        cached_tasks = load_tasks()
        last_mtime = current_mtime

    available = get_available(cached_tasks)
    ...
```

#### Optimize JSON Handling

```python
# For large task files, use streaming JSON
import ijson  # pip install ijson

def get_available_tasks_streaming():
    with open(TASKS_FILE, 'rb') as f:
        for task in ijson.items(f, 'tasks.item'):
            if task['status'] == 'available':
                yield task
```

#### File Locking Performance

```python
# Quick lock timeouts for high contention
import time

def try_lock_with_timeout(filepath, timeout=1.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with file_lock(filepath, exclusive=True):
                return True
        except BlockingIOError:
            time.sleep(0.01)  # 10ms backoff
    return False
```

### Option B: MCP Server

#### Memory Management

```typescript
// Periodic cleanup of old data
function cleanupOldData() {
  const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000; // 7 days

  // Archive completed tasks
  const archive = state.tasks.filter(
    t => t.status === 'done' && new Date(t.completed_at).getTime() < cutoff
  );

  // Keep only recent
  state.tasks = state.tasks.filter(
    t => t.status !== 'done' || new Date(t.completed_at).getTime() >= cutoff
  );

  // Limit discoveries
  state.discoveries = state.discoveries.slice(-100);
}

// Run periodically
setInterval(cleanupOldData, 60 * 60 * 1000); // Every hour
```

#### Efficient State Persistence

```typescript
// Debounce saves for rapid updates
let saveTimeout: NodeJS.Timeout | null = null;

function debouncedSave() {
  if (saveTimeout) clearTimeout(saveTimeout);
  saveTimeout = setTimeout(() => {
    saveState();
    saveTimeout = null;
  }, 100); // Wait 100ms before saving
}
```

#### Connection Pooling

For HTTP transport (if implemented):

```typescript
import { Agent } from 'http';

const agent = new Agent({
  keepAlive: true,
  maxSockets: 10,
  maxFreeSockets: 5,
});
```

### Option C: Orchestrator

#### Worker Pool Sizing

```python
import os

# Base on available CPU cores
cpu_count = os.cpu_count() or 4
optimal_workers = min(cpu_count - 1, 5)  # Leave 1 core for orchestrator

orch = Orchestrator(max_workers=optimal_workers)
```

#### Async Optimization

```python
# Process results in batches
async def _run_task_loop(self):
    active_tasks = {}

    while self._running:
        # Batch check for completed tasks
        done_ids = [tid for tid, t in active_tasks.items() if t.done()]

        # Process all completions at once
        results = await asyncio.gather(*[
            self._process_completion(tid, active_tasks[tid])
            for tid in done_ids
        ], return_exceptions=True)

        for tid in done_ids:
            del active_tasks[tid]

        # ... rest of loop
```

#### Memory-Efficient State

```python
from dataclasses import dataclass, field
from typing import Iterator

@dataclass
class CoordinationState:
    # Use __slots__ for memory efficiency
    __slots__ = ['goal', 'tasks', '_task_index']

    goal: str
    tasks: list
    _task_index: dict = field(default_factory=dict, repr=False)

    def get_task(self, task_id: str) -> Task:
        # O(1) lookup instead of O(n)
        if not self._task_index:
            self._task_index = {t.id: t for t in self.tasks}
        return self._task_index.get(task_id)
```

---

## Scaling Strategies

### Horizontal Scaling

#### Multiple Coordination Sessions

For very large projects, split into multiple sessions:

```bash
# Session 1: Backend
orchestrate run "Build API endpoints" -w 3

# Session 2: Frontend (separate directory)
orchestrate run "Build UI components" -w 3

# Session 3: Tests (can reference both)
orchestrate run "Write integration tests" -w 2
```

#### Distributed MCP Servers

For Option B across machines:

```
Machine A: MCP Server (central state)
Machine B: Claude Code workers -> connect to Machine A
Machine C: Claude Code workers -> connect to Machine A
```

Configure with network transport instead of stdio.

### Vertical Scaling

#### Increase Task Parallelism

```python
# More workers for I/O-bound tasks
orch = Orchestrator(max_workers=10)

# Fewer workers for CPU-intensive tasks
orch = Orchestrator(max_workers=3)
```

#### Optimize Claude Code Performance

```bash
# Use faster model for simple tasks
orchestrate run "Simple refactoring" --model claude-haiku

# Use more capable model for complex tasks
orchestrate run "Architecture redesign" --model claude-sonnet-4-20250514
```

---

## Monitoring and Metrics

### Built-in Status

```python
status = orch.get_status()

print(f"Progress: {status['progress']['percent_complete']}%")
print(f"Queue depth: {status['progress']['by_status']['available']}")
print(f"Active workers: {len([a for a in status['agents'] if a['current_task']])}")
```

### Custom Metrics Collection

```python
import time
from dataclasses import dataclass, field

@dataclass
class Metrics:
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0
    claim_times: list = field(default_factory=list)

    @property
    def avg_claim_time(self):
        return sum(self.claim_times) / len(self.claim_times) if self.claim_times else 0

    @property
    def success_rate(self):
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 0

metrics = Metrics()

def on_task_complete(task):
    metrics.tasks_completed += 1
    if task.claimed_at and task.completed_at:
        duration = (task.completed_at - task.claimed_at).total_seconds()
        metrics.total_execution_time += duration

orch = Orchestrator(on_task_complete=on_task_complete)
```

### Logging for Performance Analysis

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coordination-perf.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('coordination')

def on_task_complete(task):
    duration = (task.completed_at - task.claimed_at).total_seconds()
    logger.info(f"TASK_COMPLETE task_id={task.id} duration={duration:.2f}s worker={task.claimed_by}")
```

---

## Benchmarking

### Simple Benchmark Script

```python
import asyncio
import time
from orchestrator import Orchestrator

async def benchmark(num_tasks: int, num_workers: int):
    orch = Orchestrator(max_workers=num_workers, verbose=False)
    await orch.initialize("Benchmark test")

    # Create simple tasks
    for i in range(num_tasks):
        orch.add_task(f"Task {i}: echo test", priority=5)

    start = time.time()
    result = await orch.run_with_predefined_tasks()
    elapsed = time.time() - start

    print(f"Tasks: {num_tasks}, Workers: {num_workers}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {num_tasks/elapsed:.2f} tasks/second")
    print(f"Completed: {result['tasks_completed']}, Failed: {result['tasks_failed']}")

# Run benchmarks
asyncio.run(benchmark(10, 1))
asyncio.run(benchmark(10, 3))
asyncio.run(benchmark(10, 5))
```

### Coordination Overhead Measurement

```python
import time

# Measure claim time
start = time.time()
task = await orch.claim_task("test-agent")
claim_time = time.time() - start
print(f"Claim latency: {claim_time*1000:.2f}ms")

# Measure completion time
start = time.time()
await orch.complete_task(task.id, TaskResult(output="done"))
complete_time = time.time() - start
print(f"Complete latency: {complete_time*1000:.2f}ms")
```

### Performance Targets

| Configuration | Tasks/Minute | Claim Latency | Notes |
|--------------|--------------|---------------|-------|
| Option A, 3 workers | 10-20 | 100-500ms | File I/O bound |
| Option B, 3 workers | 30-60 | 10-50ms | Memory operations |
| Option C, 5 workers | 20-40 | 20-100ms | Process management |

Actual performance depends heavily on task complexity and system resources.
