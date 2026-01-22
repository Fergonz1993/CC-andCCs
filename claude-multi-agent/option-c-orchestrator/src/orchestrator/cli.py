"""
CLI interface for the Claude Multi-Agent Orchestrator.

Provides commands to:
- Initialize orchestration sessions
- Run with automatic planning or predefined tasks
- Monitor progress
- Manage agents
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from .async_orchestrator import Orchestrator as AsyncOrchestrator
from .config import DEFAULT_MODEL, DEFAULT_MAX_WORKERS, DEFAULT_TASK_TIMEOUT

app = typer.Typer(
    name="orchestrate",
    help="Multi-agent Claude Code orchestration system",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    goal: str = typer.Argument(..., help="The goal to accomplish"),
    workers: int = typer.Option(DEFAULT_MAX_WORKERS, "--workers", "-w", help="Number of worker agents"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Model to use"),
    timeout: int = typer.Option(DEFAULT_TASK_TIMEOUT, "--timeout", "-t", help="Task timeout in seconds"),
    cwd: str = typer.Option(".", "--cwd", "-C", help="Working directory"),
    plan: bool = typer.Option(True, "--plan/--no-plan", help="Let leader create plan"),
    tasks_file: Optional[str] = typer.Option(None, "--tasks", "-f", help="JSON file with predefined tasks"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save results to file"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """
    Run multi-agent orchestration to accomplish a goal.

    Examples:
        # Let leader plan and execute
        orchestrate run "Build a REST API for user management"

        # Use predefined tasks
        orchestrate run "Build API" --no-plan --tasks tasks.json

        # With more workers
        orchestrate run "Refactor authentication" -w 5
    """
    orchestrator = AsyncOrchestrator(
        working_directory=cwd,
        max_workers=workers,
        model=model,
        task_timeout=timeout,
        verbose=not quiet,
    )

    async def execute():
        await orchestrator.initialize(goal)

        if tasks_file:
            # Load predefined tasks
            with open(tasks_file) as f:
                tasks = json.load(f)
            orchestrator.add_tasks_batch(tasks)
            result = await orchestrator.run_with_predefined_tasks()
        elif plan:
            result = await orchestrator.run_with_leader_planning()
        else:
            console.print("[red]No tasks provided. Use --plan or --tasks[/]")
            raise typer.Exit(1)

        return result

    try:
        result = asyncio.run(execute())

        # Show results
        console.print(Panel(
            f"[bold green]Completed![/]\n\n"
            f"Tasks Completed: {result['tasks_completed']}\n"
            f"Tasks Failed: {result['tasks_failed']}\n"
            f"Discoveries: {result['discoveries']}\n\n"
            f"[bold]Summary:[/]\n{result['summary'][:500]}...",
            title="Orchestration Results",
            border_style="green",
        ))

        if output:
            # EC-1 fix: Ensure parent directory exists
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            console.print(f"[green]Results saved to {output}[/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Stopping agents...[/]")
        asyncio.run(orchestrator.stop())
    except (RuntimeError, OSError, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from e


@app.command()
def init(
    goal: str = typer.Argument(..., help="Project goal"),
    cwd: str = typer.Option(".", "--cwd", "-C", help="Working directory"),
    output: str = typer.Option("orchestration.json", "--output", "-o", help="Output file"),
):
    """
    Initialize an orchestration config file without running.

    This creates a JSON file you can edit before running.
    """
    config = {
        "goal": goal,
        "working_directory": str(Path(cwd).resolve()),
        "max_workers": DEFAULT_MAX_WORKERS,
        "model": DEFAULT_MODEL,
        "task_timeout": DEFAULT_TASK_TIMEOUT,
        "tasks": [],
    }

    with open(output, 'w') as f:
        json.dump(config, f, indent=2)

    console.print(f"[green]Config created: {output}[/]")
    console.print("Edit the file to add tasks, then run with:")
    console.print(f"  orchestrate from-config {output}")


@app.command("from-config")
def from_config(
    config_file: str = typer.Argument(..., help="Path to config JSON"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """
    Run orchestration from a config file.
    """
    with open(config_file) as f:
        config = json.load(f)

    orchestrator = AsyncOrchestrator(
        working_directory=config.get("working_directory", "."),
        max_workers=config.get("max_workers", DEFAULT_MAX_WORKERS),
        model=config.get("model", DEFAULT_MODEL),
        task_timeout=config.get("task_timeout", DEFAULT_TASK_TIMEOUT),
        verbose=not quiet,
    )

    async def execute():
        await orchestrator.initialize(config["goal"], config.get("master_plan", ""))

        if config.get("tasks"):
            orchestrator.add_tasks_batch(config["tasks"])
            return await orchestrator.run_with_predefined_tasks()
        else:
            return await orchestrator.run_with_leader_planning()

    try:
        result = asyncio.run(execute())

        console.print(Panel(
            f"[bold green]Completed![/]\n\n"
            f"Tasks: {result['tasks_completed']} done, {result['tasks_failed']} failed\n"
            f"Discoveries: {result['discoveries']}",
            title="Results",
            border_style="green",
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
        asyncio.run(orchestrator.stop())


@app.command()
def create_task(
    description: str = typer.Argument(..., help="Task description"),
    config_file: str = typer.Option("orchestration.json", "--config", "-c"),
    priority: int = typer.Option(5, "--priority", "-p", help="Priority 1-10"),
    depends: Optional[list[str]] = typer.Option(None, "--depends", "-d", help="Dependencies"),
    files: Optional[list[str]] = typer.Option(None, "--files", "-f", help="Context files"),
    hints: str = typer.Option("", "--hints", help="Hints for worker"),
):
    """
    Add a task to an existing config file.
    """
    with open(config_file) as f:
        config = json.load(f)

    task = {
        "description": description,
        "priority": priority,
    }

    if depends:
        task["dependencies"] = depends
    if files:
        task["context_files"] = files
    if hints:
        task["hints"] = hints

    config.setdefault("tasks", []).append(task)

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    console.print(f"[green]Task added to {config_file}[/]")


@app.command()
def status(
    config_file: str = typer.Option("orchestration.json", "--config", "-c"),
):
    """
    Show current orchestration status.
    """
    with open(config_file) as f:
        config = json.load(f)

    console.print(Panel(
        f"[bold]Goal:[/] {config.get('goal', 'Not set')}\n"
        f"[bold]Workers:[/] {config.get('max_workers', 3)}\n"
        f"[bold]Model:[/] {config.get('model', 'default')}\n"
        f"[bold]Tasks:[/] {len(config.get('tasks', []))}",
        title="Orchestration Config",
    ))

    tasks = config.get("tasks", [])
    if tasks:
        table = Table(title="Tasks")
        table.add_column("#", style="cyan")
        table.add_column("Description")
        table.add_column("Priority", justify="center")
        table.add_column("Dependencies")

        for i, task in enumerate(tasks, 1):
            deps = ", ".join(task.get("dependencies", [])) or "-"
            table.add_row(
                str(i),
                task["description"][:50] + "...",
                str(task.get("priority", 5)),
                deps,
            )

        console.print(table)


@app.command()
def example():
    """
    Show example usage and task file format.
    """
    example_config = {
        "goal": "Build a user authentication system",
        "working_directory": "/path/to/project",
        "max_workers": DEFAULT_MAX_WORKERS,
        "model": DEFAULT_MODEL,
        "task_timeout": DEFAULT_TASK_TIMEOUT,
        "tasks": [
            {
                "description": "Create User model with email, password hash, and timestamps",
                "priority": 1,
                "context_files": ["src/models/"],
                "hints": "Use TypeScript with Prisma"
            },
            {
                "description": "Implement password hashing utility",
                "priority": 1,
                "context_files": ["src/utils/"]
            },
            {
                "description": "Create login endpoint POST /auth/login",
                "priority": 2,
                "dependencies": ["task-1", "task-2"],
                "context_files": ["src/routes/auth.ts"]
            },
            {
                "description": "Create registration endpoint POST /auth/register",
                "priority": 2,
                "dependencies": ["task-1", "task-2"]
            },
            {
                "description": "Write unit tests for auth endpoints",
                "priority": 3,
                "dependencies": ["task-3", "task-4"]
            }
        ]
    }

    console.print(Panel(
        Syntax(json.dumps(example_config, indent=2), "json"),
        title="Example Config (orchestration.json)",
    ))

    console.print("\n[bold]Quick Start:[/]")
    console.print("  1. orchestrate init 'Build auth system'")
    console.print("  2. orchestrate create-task 'Create User model' -p 1")
    console.print("  3. orchestrate from-config orchestration.json")
    console.print("\n[bold]Or let the leader plan:[/]")
    console.print("  orchestrate run 'Build a user authentication system'")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
