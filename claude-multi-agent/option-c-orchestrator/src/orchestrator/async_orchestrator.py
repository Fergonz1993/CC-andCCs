"""
Main Orchestrator - Coordinates multiple Claude Code agents.

This is the brain of the multi-agent system. It:
1. Manages the agent pool (leader + workers)
2. Distributes tasks to available workers
3. Collects and aggregates results
4. Handles failures and retries
5. Maintains shared state and discoveries
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .models import (
    Task,
    TaskStatus,
    TaskResult,
    TaskContext,
    Discovery,
    CoordinationState,
)
from .agent import ClaudeCodeAgent, AgentPool
from .config import DEFAULT_MODEL


console = Console()


class Orchestrator:
    """
    Coordinates multiple Claude Code agents to work on a shared goal.

    The orchestrator manages:
    - A leader agent that plans and decomposes work
    - Multiple worker agents that execute tasks in parallel
    - A shared task queue with dependency tracking
    - Result aggregation and discovery sharing
    """

    def __init__(
        self,
        working_directory: str = ".",
        max_workers: int = 3,
        model: str = DEFAULT_MODEL,
        task_timeout: int = 600,
        on_task_complete: Optional[Callable[[Task], None]] = None,
        on_discovery: Optional[Callable[[Discovery], None]] = None,
        verbose: bool = True,
    ):
        # Validate configuration
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if task_timeout <= 0:
            raise ValueError("task_timeout must be positive")

        self.working_directory = Path(working_directory).resolve()
        self.max_workers = max_workers
        self.model = model
        self.task_timeout = task_timeout
        self.on_task_complete = on_task_complete
        self.on_discovery = on_discovery
        self.verbose = verbose

        # State
        self.state = CoordinationState(
            working_directory=str(self.working_directory),
            max_parallel_workers=max_workers,
            task_timeout_seconds=task_timeout,
        )

        # Agent pool
        self._pool = AgentPool(
            working_directory=str(self.working_directory),
            max_workers=max_workers,
            model=model,
        )

        # Control
        self._running = False
        self._task_lock = asyncio.Lock()

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self, goal: str, master_plan: str = "") -> None:
        """
        Initialize the orchestration session.

        Args:
            goal: The overall project goal
            master_plan: Optional high-level plan (if empty, leader will create one)
        """
        self.state.goal = goal
        self.state.master_plan = master_plan
        self.state.created_at = datetime.now()

        if self.verbose:
            console.print(Panel(
                f"[bold blue]Goal:[/] {goal}\n"
                f"[bold blue]Working Directory:[/] {self.working_directory}\n"
                f"[bold blue]Max Workers:[/] {self.max_workers}",
                title="Orchestrator Initialized",
                border_style="blue",
            ))

    async def start(self) -> None:
        """Start the orchestrator and all agents."""
        self._running = True

        # Start leader
        if self.verbose:
            console.print("[yellow]Starting leader agent...[/]")
        await self._pool.start_leader()

        # Start workers
        for i in range(self.max_workers):
            if self.verbose:
                console.print(f"[yellow]Starting worker-{i+1}...[/]")
            await self._pool.start_worker(f"worker-{i+1}")

        if self.verbose:
            console.print("[green]All agents started![/]")

    async def stop(self) -> None:
        """Stop all agents and clean up."""
        self._running = False
        await self._pool.stop_all()

        if self.verbose:
            console.print("[yellow]Orchestrator stopped.[/]")

    # =========================================================================
    # Task Management
    # =========================================================================

    def add_task(
        self,
        description: str,
        priority: int = 5,
        dependencies: Optional[list[str]] = None,
        context_files: Optional[list[str]] = None,
        hints: str = "",
    ) -> Task:
        """Add a new task to the queue."""
        task = Task(
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            context=TaskContext(
                files=context_files or [],
                hints=hints,
            ),
        )

        self.state.add_task(task)

        if self.verbose:
            console.print(f"[green]Added task:[/] {task.id} - {description[:50]}...")

        return task

    def add_tasks_batch(self, tasks: list[dict[str, Any]]) -> list[Task]:
        """Add multiple tasks at once."""
        created = []
        for t in tasks:
            task = self.add_task(
                description=t["description"],
                priority=t.get("priority", 5),
                dependencies=t.get("dependencies"),
                context_files=t.get("context_files"),
                hints=t.get("hints", ""),
            )
            created.append(task)
        return created

    async def claim_task(self, agent_id: str) -> Optional[Task]:
        """Claim an available task for an agent."""
        async with self._task_lock:
            available = self.state.get_available_tasks()
            if not available:
                return None

            # Sort by priority
            available.sort(key=lambda t: t.priority)
            task = available[0]

            task.claim(agent_id)

            if self.verbose:
                console.print(f"[cyan]{agent_id}[/] claimed task [bold]{task.id}[/]")

            return task

    async def complete_task(self, task_id: str, result: TaskResult) -> bool:
        """Mark a task as completed."""
        async with self._task_lock:
            task = self.state.get_task(task_id)
            if not task:
                return False

            task.complete(result)
            self.state.last_activity = datetime.now()

            # Handle discoveries
            for disc_content in result.discoveries:
                discovery = Discovery(
                    agent_id=task.claimed_by or "unknown",
                    content=disc_content,
                    related_task=task_id,
                )
                self.state.discoveries.append(discovery)

                if self.on_discovery:
                    self.on_discovery(discovery)

            # Handle subtasks
            for subtask_id in result.subtasks_created:
                if self.verbose:
                    console.print(f"[yellow]New subtask created:[/] {subtask_id}")

        if self.on_task_complete:
            self.on_task_complete(task)

        if self.verbose:
            console.print(f"[green]Task completed:[/] {task_id}")

        return True

    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed."""
        async with self._task_lock:
            task = self.state.get_task(task_id)
            if not task:
                return False

            task.fail(error)
            self.state.last_activity = datetime.now()

        if self.verbose:
            console.print(f"[red]Task failed:[/] {task_id} - {error}")

        return True

    # =========================================================================
    # Execution
    # =========================================================================

    async def run_with_leader_planning(self) -> dict[str, Any]:
        """
        Full orchestration flow:
        1. Leader creates a plan and tasks
        2. Workers execute tasks in parallel
        3. Results are aggregated
        """
        if not self.state.goal:
            raise ValueError("No goal set. Call initialize() first.")

        await self.start()

        try:
            # Phase 1: Leader creates plan
            if self.verbose:
                console.print("\n[bold blue]Phase 1: Planning[/]")

            plan_prompt = self._build_planning_prompt()
            leader = self._pool._leader

            if not leader:
                raise RuntimeError("Leader not started")

            plan_response = await leader.send_prompt(plan_prompt, timeout=300)
            tasks = self._parse_plan_response(plan_response)

            if self.verbose:
                console.print(f"[green]Leader created {len(tasks)} tasks[/]")

            # Add tasks to queue
            for t in tasks:
                self.add_task(**t)

            # Phase 2: Execute tasks
            if self.verbose:
                console.print("\n[bold blue]Phase 2: Execution[/]")

            await self._run_task_loop()

            # Phase 3: Aggregate results
            if self.verbose:
                console.print("\n[bold blue]Phase 3: Aggregation[/]")

            summary = await self._aggregate_results()

            return {
                "goal": self.state.goal,
                "tasks_completed": len([t for t in self.state.tasks if t.status == TaskStatus.DONE]),
                "tasks_failed": len([t for t in self.state.tasks if t.status == TaskStatus.FAILED]),
                "discoveries": len(self.state.discoveries),
                "summary": summary,
            }

        finally:
            await self.stop()

    async def run_with_predefined_tasks(self) -> dict[str, Any]:
        """
        Execute predefined tasks without leader planning.

        Use this when you already have a task list.
        """
        if not self.state.tasks:
            raise ValueError("No tasks defined. Add tasks first.")

        await self.start()

        try:
            await self._run_task_loop()
            summary = await self._aggregate_results()

            return {
                "goal": self.state.goal,
                "tasks_completed": len([t for t in self.state.tasks if t.status == TaskStatus.DONE]),
                "tasks_failed": len([t for t in self.state.tasks if t.status == TaskStatus.FAILED]),
                "discoveries": len(self.state.discoveries),
                "summary": summary,
            }

        finally:
            await self.stop()

    async def _run_task_loop(self) -> None:
        """Main loop that distributes tasks to workers."""
        active_tasks: dict[str, asyncio.Task[Any]] = {}

        while self._running:
            # Check for completed async tasks
            completed = [tid for tid, t in active_tasks.items() if t.done()]
            for tid in completed:
                try:
                    result = active_tasks[tid].result()
                    await self.complete_task(tid, result)
                except Exception as e:
                    await self.fail_task(tid, str(e))
                del active_tasks[tid]

            # Check if all tasks are done
            pending = [t for t in self.state.tasks if t.status in (
                TaskStatus.AVAILABLE, TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS
            )]

            if not pending and not active_tasks:
                break

            # Assign available tasks to idle workers
            for worker in self._pool._agents.values():
                if not worker.is_running or worker._current_task:
                    continue

                task = await self.claim_task(worker.agent_id)
                if task:
                    # Start task execution
                    async_task = asyncio.create_task(
                        self._execute_task_on_worker(worker, task)
                    )
                    active_tasks[task.id] = async_task

            # Show progress
            if self.verbose:
                self._show_progress()

            await asyncio.sleep(1)

    async def _execute_task_on_worker(
        self,
        worker: ClaudeCodeAgent,
        task: Task
    ) -> TaskResult:
        """Execute a task on a specific worker."""
        task.start()

        try:
            result = await worker.execute_task(task)
            return result
        except Exception as e:
            return TaskResult(output="", error=str(e))

    async def _aggregate_results(self) -> str:
        """Have the leader aggregate all results into a summary."""
        leader = self._pool._leader
        if not leader or not leader.is_running:
            # Manual aggregation
            completed = [t for t in self.state.tasks if t.status == TaskStatus.DONE]
            return f"Completed {len(completed)} tasks. See individual results for details."

        # Build aggregation prompt
        results_text = []
        for task in self.state.tasks:
            if task.result:
                results_text.append(f"## {task.id}\n{task.result.output[:500]}...")

        prompt = f"""
Please provide a summary of the completed work.

## Goal
{self.state.goal}

## Completed Tasks
{chr(10).join(results_text)}

## Discoveries
{chr(10).join(d.content for d in self.state.discoveries)}

Provide a concise summary of what was accomplished.
"""

        try:
            summary = await leader.send_prompt(prompt, timeout=120)
            return summary
        except Exception as e:
            return f"Error generating summary: {e}"

    # =========================================================================
    # Planning
    # =========================================================================

    def _build_planning_prompt(self) -> str:
        """Build the prompt for the leader to create a plan."""
        return f"""
You are the lead agent coordinating a team of {self.max_workers} worker agents.

## Goal
{self.state.goal}

## Your Task
1. Analyze what needs to be done
2. Break the work into discrete, parallelizable tasks
3. Output tasks in the following JSON format:

```json
[
  {{
    "description": "Clear description of what needs to be done",
    "priority": 1,
    "dependencies": [],
    "context_files": ["path/to/relevant/file.ts"],
    "hints": "Any helpful hints"
  }},
  ...
]
```

## Guidelines
- Each task should be completable by one agent in ~5-15 minutes
- Use dependencies to ensure correct ordering
- Priority 1 = highest, 10 = lowest
- Include relevant file paths in context_files
- Tasks with no dependencies can run in parallel

Output ONLY the JSON array, no other text.
"""

    def _parse_plan_response(self, response: str) -> list[dict[str, Any]]:
        """Parse the leader's plan response into tasks."""
        # Try to extract JSON from the response
        try:
            # Look for JSON array in the response
            start = response.find('[')
            end = response.rfind(']') + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                tasks = json.loads(json_str)
                return tasks

        except json.JSONDecodeError:
            pass

        # Fallback: create a single task with the goal
        if self.verbose:
            console.print("[yellow]Could not parse plan, creating single task[/]")

        return [{
            "description": self.state.goal,
            "priority": 1,
        }]

    # =========================================================================
    # Display
    # =========================================================================

    def _show_progress(self) -> None:
        """Display current progress."""
        progress = self.state.get_progress()

        table = Table(title="Task Progress", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Tasks", str(progress["total_tasks"]))
        table.add_row("Completed", str(progress["by_status"].get("done", 0)))
        table.add_row("In Progress", str(progress["by_status"].get("in_progress", 0)))
        table.add_row("Available", str(progress["by_status"].get("available", 0)))
        table.add_row("Failed", str(progress["by_status"].get("failed", 0)))
        table.add_row("Progress", f"{progress['percent_complete']}%")

        console.print(table)

    def get_status(self) -> dict[str, Any]:
        """Get current orchestration status."""
        return {
            "goal": self.state.goal,
            "progress": self.state.get_progress(),
            "agents": [a.to_model().model_dump() for a in self._pool.get_all_agents()],
            "discoveries": [d.model_dump() for d in self.state.discoveries],
            "last_activity": self.state.last_activity.isoformat(),
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_state(self, filepath: str) -> None:
        """Save current state to a JSON file."""
        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        state_data = self.state.model_dump(mode="json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False, default=str)

        if self.verbose:
            console.print(f"[green]State saved to {filepath}[/]")

    def load_state(self, filepath: str) -> None:
        """Load state from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.state = CoordinationState.model_validate(data)

        if self.verbose:
            console.print(f"[green]State loaded from {filepath}[/]")

    # =========================================================================
    # Health Check (PROD-007)
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive health check of the orchestrator.

        Implements: PROD-007 - Health check endpoint for Option C

        Returns:
            Dictionary with health status and details
        """
        checks = {}

        # Check working directory accessibility
        try:
            test_file = self.working_directory / ".health-check-test"
            test_file.write_text("health check")
            test_file.unlink()
            checks["filesystem"] = {
                "status": "pass",
                "message": "Working directory is accessible",
                "details": {"path": str(self.working_directory)}
            }
        except (IOError, OSError) as e:
            checks["filesystem"] = {
                "status": "fail",
                "message": "Working directory is not accessible",
                "details": {"path": str(self.working_directory), "error": str(e)}
            }

        # Check internal state consistency
        try:
            progress = self.state.get_progress()
            task_ids = [t.id for t in self.state.tasks]
            duplicate_ids = [tid for tid in task_ids if task_ids.count(tid) > 1]

            if duplicate_ids:
                checks["state_consistency"] = {
                    "status": "warn",
                    "message": f"Found {len(set(duplicate_ids))} duplicate task IDs",
                    "details": {"duplicates": list(set(duplicate_ids))}
                }
            else:
                checks["state_consistency"] = {
                    "status": "pass",
                    "message": "State is consistent",
                    "details": {
                        "total_tasks": progress["total_tasks"],
                        "by_status": progress["by_status"],
                        "percent_complete": progress["percent_complete"]
                    }
                }
        except Exception as e:
            checks["state_consistency"] = {
                "status": "fail",
                "message": "State consistency check failed",
                "details": {"error": str(e)}
            }

        # Check agent pool health
        try:
            agents = self._pool.get_all_agents()
            running_agents = [a for a in agents if a.is_running]
            checks["agent_pool"] = {
                "status": "pass" if len(running_agents) > 0 or len(agents) == 0 else "warn",
                "message": f"{len(running_agents)}/{len(agents)} agents running",
                "details": {
                    "total_agents": len(agents),
                    "running": len(running_agents),
                    "agent_ids": [a.agent_id for a in agents]
                }
            }
        except Exception as e:
            checks["agent_pool"] = {
                "status": "fail",
                "message": "Agent pool check failed",
                "details": {"error": str(e)}
            }

        # Determine overall status
        statuses = [c.get("status", "fail") for c in checks.values()]
        if "fail" in statuses:
            overall_status = "unhealthy"
        elif "warn" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return {
            "status": overall_status,
            "checks": checks,
            "timestamp": datetime.now().isoformat(),
            "running": self._running,
            "version": "2.2.0"
        }

    # =========================================================================
    # Graceful Shutdown (PROD-008)
    # =========================================================================

    async def graceful_shutdown(
        self,
        timeout: float = 30.0,
        persist_state: bool = True,
        state_filepath: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Gracefully shutdown the orchestrator.

        Implements: PROD-008 - Graceful shutdown handler for Option C

        Args:
            timeout: Maximum seconds to wait for active tasks to complete
            persist_state: Whether to save state before shutdown
            state_filepath: Path to save state (default: .coordination/orchestrator-state.json)

        Returns:
            Dictionary with shutdown summary
        """
        if self.verbose:
            console.print("[yellow]Initiating graceful shutdown...[/]")

        shutdown_start = datetime.now()
        shutdown_log = []

        # Mark as shutting down (stop accepting new tasks)
        self._running = False
        shutdown_log.append(f"[{datetime.now().isoformat()}] Stopped accepting new tasks")

        # Wait for active tasks to complete (with timeout)
        active_tasks = [
            t for t in self.state.tasks
            if t.status in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS)
        ]

        if active_tasks:
            if self.verbose:
                console.print(f"[yellow]Waiting for {len(active_tasks)} active tasks to complete...[/]")

            wait_start = datetime.now()
            while active_tasks and (datetime.now() - wait_start).total_seconds() < timeout:
                await asyncio.sleep(0.5)
                active_tasks = [
                    t for t in self.state.tasks
                    if t.status in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS)
                ]

            if active_tasks:
                shutdown_log.append(
                    f"[{datetime.now().isoformat()}] Timeout reached with {len(active_tasks)} tasks still active"
                )
                # Mark remaining active tasks as failed due to shutdown
                for task in active_tasks:
                    task.fail("Forced shutdown - task did not complete in time")
                    shutdown_log.append(f"[{datetime.now().isoformat()}] Force-failed task {task.id}")
            else:
                shutdown_log.append(f"[{datetime.now().isoformat()}] All active tasks completed")

        # Persist state if requested
        if persist_state:
            state_path = state_filepath or str(
                self.working_directory / ".coordination" / "orchestrator-state.json"
            )
            try:
                self.save_state(state_path)
                shutdown_log.append(f"[{datetime.now().isoformat()}] State persisted to {state_path}")
            except Exception as e:
                shutdown_log.append(f"[{datetime.now().isoformat()}] Failed to persist state: {e}")

        # Stop all agents
        try:
            await self._pool.stop_all()
            shutdown_log.append(f"[{datetime.now().isoformat()}] All agents stopped")
        except Exception as e:
            shutdown_log.append(f"[{datetime.now().isoformat()}] Error stopping agents: {e}")

        shutdown_duration = (datetime.now() - shutdown_start).total_seconds()

        if self.verbose:
            console.print(f"[green]Graceful shutdown completed in {shutdown_duration:.2f}s[/]")

        return {
            "success": True,
            "duration_seconds": shutdown_duration,
            "forced_task_count": len([
                t for t in self.state.tasks
                if t.status == TaskStatus.FAILED and "Forced shutdown" in (t.error or "")
            ]),
            "log": shutdown_log,
            "timestamp": datetime.now().isoformat()
        }
