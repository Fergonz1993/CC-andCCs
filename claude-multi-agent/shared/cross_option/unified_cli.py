"""
Unified CLI Wrapper (adv-cross-004)

Provides a single command-line interface that works with all three
coordination options. Abstracts away the differences between options
and provides a consistent user experience.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .task_adapter import (
    UniversalTask,
    TaskStatus,
    TaskContext,
    TaskResult,
    AdapterFactory,
)
from .migration import MigrationTool, MigrationResult
from .sync import TaskSynchronizer, SyncDirection, ConflictResolution


class CoordinationOption(Enum):
    """Available coordination options."""
    FILE_BASED = "A"
    MCP_SERVER = "B"
    ORCHESTRATOR = "C"
    AUTO = "auto"


@dataclass
class CLIConfig:
    """Configuration for the unified CLI."""
    default_option: CoordinationOption = CoordinationOption.AUTO
    coordination_dir: str = ".coordination"
    verbose: bool = False
    color_output: bool = True
    json_output: bool = False

    @classmethod
    def from_env(cls) -> "CLIConfig":
        """Create config from environment variables."""
        return cls(
            default_option=CoordinationOption(
                os.environ.get("COORD_OPTION", "auto")
            ) if os.environ.get("COORD_OPTION") else CoordinationOption.AUTO,
            coordination_dir=os.environ.get("COORD_DIR", ".coordination"),
            verbose=os.environ.get("COORD_VERBOSE", "").lower() in ("1", "true", "yes"),
            color_output=os.environ.get("COORD_COLOR", "1").lower() in ("1", "true", "yes"),
            json_output=os.environ.get("COORD_JSON", "").lower() in ("1", "true", "yes"),
        )

    @classmethod
    def from_file(cls, path: str) -> "CLIConfig":
        """Load config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            default_option=CoordinationOption(data.get("default_option", "auto")),
            coordination_dir=data.get("coordination_dir", ".coordination"),
            verbose=data.get("verbose", False),
            color_output=data.get("color_output", True),
            json_output=data.get("json_output", False),
        )


class UnifiedCLI:
    """
    Unified command-line interface for all coordination options.

    Provides consistent commands across Options A, B, and C:
    - init: Initialize a coordination session
    - add-task: Add a new task
    - list-tasks: List all tasks
    - claim: Claim a task (workers)
    - complete: Mark a task complete
    - fail: Mark a task failed
    - status: Show coordination status
    - migrate: Migrate between options
    - sync: Synchronize between options
    """

    def __init__(self, config: Optional[CLIConfig] = None):
        """Initialize the CLI with optional configuration."""
        self.config = config or CLIConfig.from_env()
        self._option: Optional[CoordinationOption] = None
        self._adapter = None

    def detect_option(self, path: str = ".") -> CoordinationOption:
        """
        Auto-detect which coordination option is in use.

        Checks for:
        - Option A: .coordination/tasks.json
        - Option B: .coordination/mcp-state.json or MCP server running
        - Option C: State file from Python orchestrator
        """
        coord_dir = Path(path) / self.config.coordination_dir

        # Check for Option A markers
        if (coord_dir / "tasks.json").exists():
            return CoordinationOption.FILE_BASED

        # Check for Option B markers
        if (coord_dir / "mcp-state.json").exists():
            return CoordinationOption.MCP_SERVER

        # Check for Option C markers
        if (coord_dir / "state.json").exists():
            return CoordinationOption.ORCHESTRATOR

        # Default to file-based
        return CoordinationOption.FILE_BASED

    def get_option(self) -> CoordinationOption:
        """Get the current coordination option."""
        if self._option is None:
            if self.config.default_option == CoordinationOption.AUTO:
                self._option = self.detect_option()
            else:
                self._option = self.config.default_option
        return self._option

    def set_option(self, option: CoordinationOption) -> None:
        """Set the coordination option to use."""
        self._option = option
        self._adapter = AdapterFactory.get_adapter(option.value)

    def _get_state_path(self) -> str:
        """Get the path to the state file/directory."""
        option = self.get_option()
        coord_dir = Path(self.config.coordination_dir)

        if option == CoordinationOption.FILE_BASED:
            return str(coord_dir / "tasks.json")
        elif option == CoordinationOption.MCP_SERVER:
            return str(coord_dir / "mcp-state.json")
        else:
            return str(coord_dir / "state.json")

    def _output(self, data: Any, message: str = "") -> None:
        """Output data in the configured format."""
        if self.config.json_output:
            if isinstance(data, dict) or isinstance(data, list):
                print(json.dumps(data, indent=2, default=str))
            else:
                print(json.dumps({"result": str(data)}, indent=2))
        else:
            if message:
                if self.config.color_output:
                    print(f"\033[92m{message}\033[0m")  # Green
                else:
                    print(message)
            elif isinstance(data, dict) or isinstance(data, list):
                print(json.dumps(data, indent=2, default=str))
            else:
                print(data)

    def _error(self, message: str) -> None:
        """Output an error message."""
        if self.config.json_output:
            print(json.dumps({"error": message}))
        else:
            if self.config.color_output:
                print(f"\033[91mError: {message}\033[0m", file=sys.stderr)  # Red
            else:
                print(f"Error: {message}", file=sys.stderr)

    # =========================================================================
    # Commands
    # =========================================================================

    def cmd_init(self, goal: str, approach: str = "") -> int:
        """Initialize a new coordination session."""
        option = self.get_option()
        coord_dir = Path(self.config.coordination_dir)

        # Create directory structure
        coord_dir.mkdir(parents=True, exist_ok=True)
        (coord_dir / "context").mkdir(exist_ok=True)
        (coord_dir / "logs").mkdir(exist_ok=True)
        (coord_dir / "results").mkdir(exist_ok=True)

        # Create initial state
        if option == CoordinationOption.FILE_BASED:
            state = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "tasks": [],
            }
            with open(coord_dir / "tasks.json", "w") as f:
                json.dump(state, f, indent=2)

            # Create master plan
            plan = f"""# Master Plan

## Objective
{goal}

## Approach
{approach or "To be determined."}

## Status
**Phase**: Planning
**Started**: {datetime.now().isoformat()}
"""
            (coord_dir / "master-plan.md").write_text(plan)

            # Create discoveries file
            (coord_dir / "context" / "discoveries.md").write_text(
                "# Shared Discoveries\n\nImportant findings go here.\n"
            )

        elif option == CoordinationOption.MCP_SERVER:
            state = {
                "master_plan": approach,
                "goal": goal,
                "tasks": [],
                "agents": {},
                "discoveries": [],
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
            }
            with open(coord_dir / "mcp-state.json", "w") as f:
                json.dump(state, f, indent=2)

        else:  # Option C
            state = {
                "goal": goal,
                "master_plan": approach,
                "tasks": [],
                "discoveries": [],
                "working_directory": str(Path.cwd()),
                "max_parallel_workers": 3,
                "task_timeout_seconds": 600,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
            }
            with open(coord_dir / "state.json", "w") as f:
                json.dump(state, f, indent=2, default=str)

        self._output(
            {"goal": goal, "option": option.value},
            f"Initialized coordination (Option {option.value}) for: {goal}"
        )
        return 0

    def cmd_add_task(
        self,
        description: str,
        priority: int = 5,
        dependencies: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        hints: str = "",
    ) -> int:
        """Add a new task."""
        import hashlib
        import time

        # Generate task ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:4]
        task_id = f"task-{timestamp}-{random_suffix}"

        task = UniversalTask(
            id=task_id,
            description=description,
            status=TaskStatus.AVAILABLE,
            priority=priority,
            dependencies=dependencies or [],
            context=TaskContext(
                files=files or [],
                hints=hints,
            ) if files or hints else None,
            created_at=datetime.now(),
        )

        # Load and update state
        adapter = AdapterFactory.get_adapter(self.get_option().value)
        state_path = self._get_state_path()

        tasks = adapter.load_tasks(state_path)
        tasks.append(task)
        adapter.save_tasks(tasks, state_path)

        self._output(
            {"task_id": task_id, "description": description},
            f"Added task {task_id}"
        )
        return 0

    def cmd_list_tasks(
        self,
        status_filter: Optional[str] = None,
        limit: int = 50,
    ) -> int:
        """List tasks."""
        adapter = AdapterFactory.get_adapter(self.get_option().value)
        tasks = adapter.load_tasks(self._get_state_path())

        if status_filter:
            try:
                filter_status = TaskStatus.from_string(status_filter)
                tasks = [t for t in tasks if t.status == filter_status]
            except ValueError:
                self._error(f"Invalid status: {status_filter}")
                return 1

        # Sort by priority
        tasks.sort(key=lambda t: t.priority)
        tasks = tasks[:limit]

        if self.config.json_output:
            self._output([t.to_dict() for t in tasks])
        else:
            print(f"\nTasks ({len(tasks)}):")
            print("-" * 60)
            for task in tasks:
                status_color = {
                    TaskStatus.AVAILABLE: "\033[32m",  # Green
                    TaskStatus.CLAIMED: "\033[33m",    # Yellow
                    TaskStatus.IN_PROGRESS: "\033[34m", # Blue
                    TaskStatus.DONE: "\033[90m",       # Gray
                    TaskStatus.FAILED: "\033[31m",     # Red
                }.get(task.status, "")
                reset = "\033[0m" if self.config.color_output else ""
                status_color = status_color if self.config.color_output else ""

                claimed = f" [{task.claimed_by}]" if task.claimed_by else ""
                print(f"  [{task.priority}] {task.id}: {status_color}{task.status.value}{reset}{claimed}")
                print(f"      {task.description[:50]}...")

        return 0

    def cmd_claim(self, agent_id: str) -> int:
        """Claim an available task."""
        adapter = AdapterFactory.get_adapter(self.get_option().value)
        state_path = self._get_state_path()
        tasks = adapter.load_tasks(state_path)

        # Find done task IDs for dependency checking
        done_ids = {t.id for t in tasks if t.status == TaskStatus.DONE}

        # Find available tasks with satisfied dependencies
        available = [
            t for t in tasks
            if t.status == TaskStatus.AVAILABLE
            and all(dep in done_ids for dep in t.dependencies)
        ]

        if not available:
            self._output(
                {"claimed": False, "message": "No available tasks"},
                "No available tasks with satisfied dependencies."
            )
            return 0

        # Claim highest priority task
        available.sort(key=lambda t: t.priority)
        task = available[0]

        # Update task
        for t in tasks:
            if t.id == task.id:
                t.status = TaskStatus.CLAIMED
                t.claimed_by = agent_id
                t.claimed_at = datetime.now()
                break

        adapter.save_tasks(tasks, state_path)

        self._output(
            {"claimed": True, "task_id": task.id, "description": task.description},
            f"Claimed task {task.id}: {task.description[:50]}..."
        )
        return 0

    def cmd_complete(
        self,
        task_id: str,
        agent_id: str,
        output: str,
        files_modified: Optional[List[str]] = None,
        files_created: Optional[List[str]] = None,
    ) -> int:
        """Mark a task as complete."""
        adapter = AdapterFactory.get_adapter(self.get_option().value)
        state_path = self._get_state_path()
        tasks = adapter.load_tasks(state_path)

        found = False
        for task in tasks:
            if task.id == task_id:
                if task.claimed_by != agent_id:
                    self._error(f"Task {task_id} not claimed by {agent_id}")
                    return 1

                task.status = TaskStatus.DONE
                task.completed_at = datetime.now()
                task.result = TaskResult(
                    output=output,
                    files_modified=files_modified or [],
                    files_created=files_created or [],
                )
                found = True
                break

        if not found:
            self._error(f"Task {task_id} not found")
            return 1

        adapter.save_tasks(tasks, state_path)

        self._output(
            {"completed": True, "task_id": task_id},
            f"Completed task {task_id}"
        )
        return 0

    def cmd_fail(self, task_id: str, agent_id: str, reason: str) -> int:
        """Mark a task as failed."""
        adapter = AdapterFactory.get_adapter(self.get_option().value)
        state_path = self._get_state_path()
        tasks = adapter.load_tasks(state_path)

        found = False
        for task in tasks:
            if task.id == task_id:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.result = TaskResult(error=reason)
                found = True
                break

        if not found:
            self._error(f"Task {task_id} not found")
            return 1

        adapter.save_tasks(tasks, state_path)

        self._output(
            {"failed": True, "task_id": task_id, "reason": reason},
            f"Task {task_id} marked as failed: {reason}"
        )
        return 0

    def cmd_status(self) -> int:
        """Show coordination status."""
        adapter = AdapterFactory.get_adapter(self.get_option().value)
        tasks = adapter.load_tasks(self._get_state_path())

        by_status = {}
        for task in tasks:
            status = task.status.value
            by_status[status] = by_status.get(status, 0) + 1

        total = len(tasks)
        done = by_status.get("done", 0)
        progress = (done / total * 100) if total else 0

        status_data = {
            "option": self.get_option().value,
            "total_tasks": total,
            "by_status": by_status,
            "progress_percent": round(progress, 1),
        }

        if self.config.json_output:
            self._output(status_data)
        else:
            print("\n" + "=" * 60)
            print(f"COORDINATION STATUS (Option {self.get_option().value})")
            print("=" * 60)

            for status in ["available", "claimed", "in_progress", "done", "failed"]:
                count = by_status.get(status, 0)
                if count > 0:
                    print(f"  {status.upper()}: {count}")

            print("-" * 60)
            print(f"Progress: {done}/{total} tasks complete ({progress:.1f}%)")
            print("=" * 60 + "\n")

        return 0

    def cmd_migrate(
        self,
        source_option: str,
        target_option: str,
        source_path: str,
        target_path: str,
    ) -> int:
        """Migrate between coordination options."""
        tool = MigrationTool()
        result = tool.migrate(source_path, target_path, source_option, target_option)

        if self.config.json_output:
            self._output(result.to_dict())
        else:
            if result.success:
                print(f"Migration successful!")
                print(f"  Tasks migrated: {result.tasks_migrated}")
                print(f"  Discoveries migrated: {result.discoveries_migrated}")
                print(f"  Agents migrated: {result.agents_migrated}")
                if result.backup_path:
                    print(f"  Backup created: {result.backup_path}")
            else:
                print(f"Migration failed:")
                for error in result.errors:
                    print(f"  - {error}")

        return 0 if result.success else 1

    def cmd_sync(
        self,
        source_option: str,
        target_option: str,
        source_path: str,
        target_path: str,
        direction: str = "bidirectional",
    ) -> int:
        """Synchronize between coordination options."""
        sync_direction = {
            "source_to_target": SyncDirection.SOURCE_TO_TARGET,
            "target_to_source": SyncDirection.TARGET_TO_SOURCE,
            "bidirectional": SyncDirection.BIDIRECTIONAL,
        }.get(direction, SyncDirection.BIDIRECTIONAL)

        synchronizer = TaskSynchronizer(
            source_option,
            target_option,
            conflict_resolution=ConflictResolution.NEWEST_WINS,
        )
        result = synchronizer.sync(source_path, target_path, sync_direction)

        if self.config.json_output:
            self._output(result.to_dict())
        else:
            if result.success:
                print(f"Sync successful!")
                print(f"  Tasks synced: {result.tasks_synced}")
                print(f"  Tasks created: {result.tasks_created}")
                print(f"  Tasks updated: {result.tasks_updated}")
                if result.conflicts_found > 0:
                    print(f"  Conflicts: {result.conflicts_resolved}/{result.conflicts_found} resolved")
            else:
                print(f"Sync failed:")
                for error in result.errors:
                    print(f"  - {error}")

        return 0 if result.success else 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the unified CLI."""
    parser = argparse.ArgumentParser(
        prog="coord",
        description="Unified CLI for Claude Multi-Agent Coordination",
    )

    parser.add_argument(
        "--option", "-o",
        choices=["A", "B", "C", "auto"],
        default="auto",
        help="Coordination option to use (default: auto-detect)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable color output",
    )
    parser.add_argument(
        "--dir", "-d",
        default=".coordination",
        help="Coordination directory (default: .coordination)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize coordination")
    init_parser.add_argument("goal", help="Project goal")
    init_parser.add_argument("--approach", "-a", default="", help="High-level approach")

    # add-task command
    add_parser = subparsers.add_parser("add-task", help="Add a task")
    add_parser.add_argument("description", help="Task description")
    add_parser.add_argument("--priority", "-p", type=int, default=5, help="Priority (1-10)")
    add_parser.add_argument("--depends", "-d", nargs="*", help="Dependency task IDs")
    add_parser.add_argument("--files", "-f", nargs="*", help="Relevant files")
    add_parser.add_argument("--hints", default="", help="Hints for worker")

    # list-tasks command
    list_parser = subparsers.add_parser("list-tasks", help="List tasks")
    list_parser.add_argument("--status", "-s", help="Filter by status")
    list_parser.add_argument("--limit", "-l", type=int, default=50, help="Max tasks to show")

    # claim command
    claim_parser = subparsers.add_parser("claim", help="Claim a task")
    claim_parser.add_argument("agent_id", help="Your agent ID")

    # complete command
    complete_parser = subparsers.add_parser("complete", help="Complete a task")
    complete_parser.add_argument("task_id", help="Task ID")
    complete_parser.add_argument("agent_id", help="Your agent ID")
    complete_parser.add_argument("output", help="Task output summary")
    complete_parser.add_argument("--modified", "-m", nargs="*", help="Files modified")
    complete_parser.add_argument("--created", "-c", nargs="*", help="Files created")

    # fail command
    fail_parser = subparsers.add_parser("fail", help="Mark task as failed")
    fail_parser.add_argument("task_id", help="Task ID")
    fail_parser.add_argument("agent_id", help="Your agent ID")
    fail_parser.add_argument("reason", help="Failure reason")

    # status command
    subparsers.add_parser("status", help="Show coordination status")

    # migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate between options")
    migrate_parser.add_argument("--from", dest="source_option", required=True, help="Source option")
    migrate_parser.add_argument("--to", dest="target_option", required=True, help="Target option")
    migrate_parser.add_argument("--source-path", required=True, help="Source state path")
    migrate_parser.add_argument("--target-path", required=True, help="Target state path")

    # sync command
    sync_parser = subparsers.add_parser("sync", help="Sync between options")
    sync_parser.add_argument("--source-option", required=True, help="Source option")
    sync_parser.add_argument("--target-option", required=True, help="Target option")
    sync_parser.add_argument("--source-path", required=True, help="Source state path")
    sync_parser.add_argument("--target-path", required=True, help="Target state path")
    sync_parser.add_argument(
        "--direction",
        choices=["source_to_target", "target_to_source", "bidirectional"],
        default="bidirectional",
        help="Sync direction",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the unified CLI."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    config = CLIConfig(
        default_option=CoordinationOption(parsed.option) if parsed.option != "auto" else CoordinationOption.AUTO,
        coordination_dir=parsed.dir,
        verbose=parsed.verbose,
        color_output=not parsed.no_color,
        json_output=parsed.json,
    )

    cli = UnifiedCLI(config)

    if parsed.option != "auto":
        cli.set_option(CoordinationOption(parsed.option))

    if parsed.command == "init":
        return cli.cmd_init(parsed.goal, parsed.approach)
    elif parsed.command == "add-task":
        return cli.cmd_add_task(
            parsed.description,
            parsed.priority,
            parsed.depends,
            parsed.files,
            parsed.hints,
        )
    elif parsed.command == "list-tasks":
        return cli.cmd_list_tasks(parsed.status, parsed.limit)
    elif parsed.command == "claim":
        return cli.cmd_claim(parsed.agent_id)
    elif parsed.command == "complete":
        return cli.cmd_complete(
            parsed.task_id,
            parsed.agent_id,
            parsed.output,
            parsed.modified,
            parsed.created,
        )
    elif parsed.command == "fail":
        return cli.cmd_fail(parsed.task_id, parsed.agent_id, parsed.reason)
    elif parsed.command == "status":
        return cli.cmd_status()
    elif parsed.command == "migrate":
        return cli.cmd_migrate(
            parsed.source_option,
            parsed.target_option,
            parsed.source_path,
            parsed.target_path,
        )
    elif parsed.command == "sync":
        return cli.cmd_sync(
            parsed.source_option,
            parsed.target_option,
            parsed.source_path,
            parsed.target_path,
            parsed.direction,
        )
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
