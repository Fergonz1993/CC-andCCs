#!/usr/bin/env python3
"""
File-based Multi-Agent Coordination System for Claude Code

This module provides utilities for coordinating multiple Claude Code instances
through a shared filesystem. Designed for minimal setup - just files and JSON.

Usage:
    # As a leader
    python coordination.py leader init "Build a REST API"
    python coordination.py leader add-task "Implement user model" --priority 1
    python coordination.py leader status

    # As a worker
    python coordination.py worker claim terminal-2
    python coordination.py worker complete task-001 "Implemented successfully"
    python coordination.py worker fail task-001 "Missing dependency"
"""

import json
import os
import sys
import fcntl
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import argparse
import time

# Configuration
COORDINATION_DIR = Path(".coordination")
TASKS_FILE = COORDINATION_DIR / "tasks.json"
MASTER_PLAN_FILE = COORDINATION_DIR / "master-plan.md"
DISCOVERIES_FILE = COORDINATION_DIR / "context" / "discoveries.md"
LOGS_DIR = COORDINATION_DIR / "logs"
RESULTS_DIR = COORDINATION_DIR / "results"


@dataclass
class Task:
    id: str
    description: str
    status: str  # available, claimed, in_progress, done, failed
    priority: int
    claimed_by: Optional[str] = None
    dependencies: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    claimed_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove None values for cleaner JSON
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            id=data["id"],
            description=data["description"],
            status=data["status"],
            priority=data.get("priority", 5),
            claimed_by=data.get("claimed_by"),
            dependencies=data.get("dependencies"),
            context=data.get("context"),
            result=data.get("result"),
            created_at=data.get("created_at"),
            claimed_at=data.get("claimed_at"),
            completed_at=data.get("completed_at"),
        )


@contextmanager
def file_lock(filepath: Path, exclusive: bool = True):
    """
    File locking to prevent race conditions.

    Uses fcntl on Unix systems and msvcrt on Windows.
    Falls back to a no-op lock if neither is available.
    """
    lock_file = filepath.with_suffix(filepath.suffix + ".lock")
    lock_file.touch(exist_ok=True)

    with open(lock_file, "r+") as f:
        try:
            if sys.platform == "win32":
                # Windows locking using msvcrt
                import msvcrt
                if exclusive:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # Windows doesn't have shared locks via msvcrt, use exclusive
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                # Unix locking using fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            if sys.platform == "win32":
                import msvcrt
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass  # Already unlocked
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def generate_task_id() -> str:
    """Generate a unique task ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:4]
    return f"task-{timestamp}-{random_suffix}"


def now_iso() -> str:
    """Get current time in ISO format."""
    return datetime.now().isoformat()


def ensure_coordination_structure():
    """Create the coordination directory structure if it doesn't exist."""
    COORDINATION_DIR.mkdir(exist_ok=True)
    (COORDINATION_DIR / "context").mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    if not TASKS_FILE.exists():
        save_tasks({"version": "1.0", "created_at": now_iso(), "last_updated": now_iso(), "tasks": []})

    if not DISCOVERIES_FILE.exists():
        DISCOVERIES_FILE.write_text("# Shared Discoveries\n\nImportant findings go here.\n")


def load_tasks() -> dict:
    """Load the tasks file with locking."""
    if not TASKS_FILE.exists():
        return {"version": "1.0", "created_at": now_iso(), "last_updated": now_iso(), "tasks": []}

    with file_lock(TASKS_FILE, exclusive=False):
        return json.loads(TASKS_FILE.read_text())


def save_tasks(data: dict):
    """Save the tasks file with locking."""
    data["last_updated"] = now_iso()
    with file_lock(TASKS_FILE, exclusive=True):
        TASKS_FILE.write_text(json.dumps(data, indent=2))


def log_action(agent_id: str, action: str, details: str = ""):
    """Log an action to the agent's log file."""
    log_file = LOGS_DIR / f"{agent_id}.log"
    timestamp = now_iso()
    entry = f"[{timestamp}] {action}"
    if details:
        entry += f": {details}"
    entry += "\n"

    with open(log_file, "a") as f:
        f.write(entry)


# ============================================================================
# LEADER COMMANDS
# ============================================================================

def leader_init(goal: str, approach: str = ""):
    """Initialize a new coordination session."""
    ensure_coordination_structure()

    plan_content = f"""# Master Plan

## Objective
{goal}

## Approach
{approach if approach else "To be determined based on codebase analysis."}

## Success Criteria
- [ ] All tasks completed successfully
- [ ] Code quality maintained
- [ ] Tests passing

## Task Breakdown
Tasks are tracked in `tasks.json`. This plan provides high-level context.

## Status
**Phase**: Planning
**Started**: {now_iso()}

## Notes
Add important notes here as the project progresses.
"""

    MASTER_PLAN_FILE.write_text(plan_content)
    log_action("leader", "INIT", f"Goal: {goal}")
    print(f"✓ Initialized coordination for: {goal}")
    print(f"  Master plan: {MASTER_PLAN_FILE}")
    print(f"  Tasks file: {TASKS_FILE}")


def leader_add_task(
    description: str,
    priority: int = 5,
    dependencies: Optional[List[str]] = None,
    context_files: Optional[List[str]] = None,
    hints: str = ""
) -> str:
    """Add a new task to the queue."""
    ensure_coordination_structure()

    task = Task(
        id=generate_task_id(),
        description=description,
        status="available",
        priority=priority,
        dependencies=dependencies or [],
        context={
            "files": context_files or [],
            "hints": hints
        } if context_files or hints else None,
        created_at=now_iso()
    )

    data = load_tasks()
    data["tasks"].append(task.to_dict())
    save_tasks(data)

    log_action("leader", "ADD_TASK", f"{task.id}: {description[:50]}")
    print(f"✓ Added task {task.id}")
    return task.id


def leader_status():
    """Show current coordination status."""
    data = load_tasks()
    tasks = [Task.from_dict(t) for t in data["tasks"]]

    by_status = {}
    for task in tasks:
        by_status.setdefault(task.status, []).append(task)

    print("\n" + "=" * 60)
    print("COORDINATION STATUS")
    print("=" * 60)

    for status in ["available", "claimed", "in_progress", "done", "failed"]:
        status_tasks = by_status.get(status, [])
        if status_tasks:
            print(f"\n{status.upper()} ({len(status_tasks)}):")
            for t in sorted(status_tasks, key=lambda x: x.priority):
                claimed = f" [by {t.claimed_by}]" if t.claimed_by else ""
                print(f"  [{t.priority}] {t.id}: {t.description[:40]}...{claimed}")

    print("\n" + "-" * 60)
    total = len(tasks)
    done = len(by_status.get("done", []))
    print(f"Progress: {done}/{total} tasks complete ({100*done//total if total else 0}%)")
    print("=" * 60 + "\n")


def leader_aggregate():
    """Aggregate all completed results into a summary."""
    results = []
    for result_file in RESULTS_DIR.glob("*.md"):
        results.append({
            "task_id": result_file.stem,
            "content": result_file.read_text()
        })

    if not results:
        print("No results to aggregate yet.")
        return

    summary_file = COORDINATION_DIR / "summary.md"
    content = f"# Aggregated Results\n\nGenerated: {now_iso()}\n\n"

    for r in results:
        content += f"---\n\n## {r['task_id']}\n\n{r['content']}\n\n"

    summary_file.write_text(content)
    print(f"✓ Aggregated {len(results)} results to {summary_file}")


# ============================================================================
# AGENT REGISTRATION
# ============================================================================

AGENTS_FILE = COORDINATION_DIR / "agents.json"


def load_agents() -> dict:
    """Load the agents file."""
    if not AGENTS_FILE.exists():
        return {"agents": []}
    with file_lock(AGENTS_FILE, exclusive=False):
        return json.loads(AGENTS_FILE.read_text())


def save_agents(data: dict):
    """Save the agents file with locking."""
    with file_lock(AGENTS_FILE, exclusive=True):
        AGENTS_FILE.write_text(json.dumps(data, indent=2))


def register_agent(agent_id: str, capabilities: str = ""):
    """Register a new agent in the coordination system."""
    ensure_coordination_structure()

    data = load_agents()

    # Check if agent already exists
    for agent in data["agents"]:
        if agent["id"] == agent_id:
            agent["capabilities"] = capabilities
            agent["last_seen"] = now_iso()
            save_agents(data)
            print(f"✓ Updated agent {agent_id}")
            return

    # Add new agent
    agent = {
        "id": agent_id,
        "capabilities": capabilities,
        "registered_at": now_iso(),
        "last_seen": now_iso(),
    }
    data["agents"].append(agent)
    save_agents(data)

    log_action(agent_id, "REGISTERED", f"capabilities: {capabilities}")
    print(f"✓ Registered agent {agent_id}")
    if capabilities:
        print(f"  Capabilities: {capabilities}")


# ============================================================================
# WORKER COMMANDS
# ============================================================================

def worker_claim(terminal_id: str) -> Optional[Task]:
    """Claim an available task."""
    data = load_tasks()
    tasks = [Task.from_dict(t) for t in data["tasks"]]
    done_ids = {t.id for t in tasks if t.status == "done"}

    # Find available task with satisfied dependencies
    available = [
        t for t in tasks
        if t.status == "available"
        and all(dep in done_ids for dep in (t.dependencies or []))
    ]

    if not available:
        print("No available tasks with satisfied dependencies.")
        return None

    # Pick highest priority (lowest number)
    task = min(available, key=lambda t: t.priority)

    # Claim it
    for i, t in enumerate(data["tasks"]):
        if t["id"] == task.id:
            data["tasks"][i]["status"] = "claimed"
            data["tasks"][i]["claimed_by"] = terminal_id
            data["tasks"][i]["claimed_at"] = now_iso()
            break

    save_tasks(data)

    # Verify claim succeeded (re-read to check for race condition)
    data = load_tasks()
    for t in data["tasks"]:
        if t["id"] == task.id:
            if t.get("claimed_by") == terminal_id:
                log_action(terminal_id, "CLAIMED", task.id)
                print(f"✓ Claimed task {task.id}")
                print(f"  Description: {task.description}")
                print(f"  Priority: {task.priority}")
                if task.context:
                    print(f"  Context: {json.dumps(task.context, indent=4)}")
                return Task.from_dict(t)
            else:
                print(f"✗ Race condition: task {task.id} was claimed by {t.get('claimed_by')}")
                return worker_claim(terminal_id)  # Try again

    return None


def worker_start(terminal_id: str, task_id: str):
    """Mark a claimed task as in_progress."""
    data = load_tasks()

    for i, t in enumerate(data["tasks"]):
        if t["id"] == task_id and t.get("claimed_by") == terminal_id:
            data["tasks"][i]["status"] = "in_progress"
            save_tasks(data)
            log_action(terminal_id, "STARTED", task_id)
            print(f"✓ Started task {task_id}")
            return

    print(f"✗ Cannot start task {task_id} - not claimed by {terminal_id}")


def worker_complete(terminal_id: str, task_id: str, output: str, files_modified: List[str] = None, files_created: List[str] = None):
    """Mark a task as complete and write results."""
    data = load_tasks()
    task = None

    for i, t in enumerate(data["tasks"]):
        if t["id"] == task_id:
            if t.get("claimed_by") != terminal_id:
                print(f"✗ Task {task_id} not claimed by {terminal_id}")
                return

            data["tasks"][i]["status"] = "done"
            data["tasks"][i]["completed_at"] = now_iso()
            data["tasks"][i]["result"] = {
                "output": output,
                "files_modified": files_modified or [],
                "files_created": files_created or []
            }
            task = t
            break

    if not task:
        print(f"✗ Task {task_id} not found")
        return

    save_tasks(data)

    # Write result file
    result_content = f"""# Task: {task_id}

## Description
{task['description']}

## Completed By
{terminal_id}

## Completed At
{now_iso()}

## Output
{output}

## Files Modified
{chr(10).join(f'- {f}' for f in (files_modified or [])) or 'None'}

## Files Created
{chr(10).join(f'- {f}' for f in (files_created or [])) or 'None'}

## Status
SUCCESS
"""

    result_file = RESULTS_DIR / f"{task_id}.md"
    result_file.write_text(result_content)

    log_action(terminal_id, "COMPLETED", task_id)
    print(f"✓ Completed task {task_id}")
    print(f"  Result file: {result_file}")


def worker_fail(terminal_id: str, task_id: str, reason: str):
    """Mark a task as failed."""
    data = load_tasks()
    task = None

    for i, t in enumerate(data["tasks"]):
        if t["id"] == task_id:
            data["tasks"][i]["status"] = "failed"
            data["tasks"][i]["completed_at"] = now_iso()
            data["tasks"][i]["result"] = {"error": reason}
            task = t
            break

    if not task:
        print(f"✗ Task {task_id} not found")
        return

    save_tasks(data)

    # Write failure report
    result_content = f"""# Task: {task_id}

## Description
{task['description']}

## Attempted By
{terminal_id}

## Failed At
{now_iso()}

## Failure Reason
{reason}

## Status
FAILED
"""

    result_file = RESULTS_DIR / f"{task_id}.md"
    result_file.write_text(result_content)

    log_action(terminal_id, "FAILED", f"{task_id}: {reason}")
    print(f"✗ Task {task_id} marked as failed: {reason}")


def worker_list_available():
    """List available tasks."""
    data = load_tasks()
    tasks = [Task.from_dict(t) for t in data["tasks"]]
    done_ids = {t.id for t in tasks if t.status == "done"}

    available = [
        t for t in tasks
        if t.status == "available"
        and all(dep in done_ids for dep in (t.dependencies or []))
    ]

    if not available:
        print("No available tasks.")
        return

    print(f"\nAvailable tasks ({len(available)}):")
    for t in sorted(available, key=lambda x: x.priority):
        print(f"  [{t.priority}] {t.id}: {t.description[:60]}...")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Coordination CLI")
    subparsers = parser.add_subparsers(dest="role", help="Role (leader or worker)")

    # Leader commands
    leader_parser = subparsers.add_parser("leader", help="Leader commands")
    leader_sub = leader_parser.add_subparsers(dest="command")

    init_cmd = leader_sub.add_parser("init", help="Initialize coordination")
    init_cmd.add_argument("goal", help="Project goal")
    init_cmd.add_argument("--approach", default="", help="High-level approach")

    add_cmd = leader_sub.add_parser("add-task", help="Add a task")
    add_cmd.add_argument("description", help="Task description")
    add_cmd.add_argument("--priority", "-p", type=int, default=5, help="Priority (1-10, lower=higher)")
    add_cmd.add_argument("--depends", "-d", nargs="*", help="Dependency task IDs")
    add_cmd.add_argument("--files", "-f", nargs="*", help="Relevant files")
    add_cmd.add_argument("--hints", default="", help="Hints for the worker")

    leader_sub.add_parser("status", help="Show status")
    leader_sub.add_parser("aggregate", help="Aggregate results")

    # Register command (for agents)
    register_parser = subparsers.add_parser("register", help="Register an agent")
    register_parser.add_argument("--agent-id", required=True, help="Unique agent identifier")
    register_parser.add_argument("--capabilities", default="", help="Comma-separated capabilities")

    # Worker commands
    worker_parser = subparsers.add_parser("worker", help="Worker commands")
    worker_sub = worker_parser.add_subparsers(dest="command")

    claim_cmd = worker_sub.add_parser("claim", help="Claim a task")
    claim_cmd.add_argument("terminal_id", help="Your terminal ID")

    start_cmd = worker_sub.add_parser("start", help="Start working on claimed task")
    start_cmd.add_argument("terminal_id", help="Your terminal ID")
    start_cmd.add_argument("task_id", help="Task ID")

    complete_cmd = worker_sub.add_parser("complete", help="Complete a task")
    complete_cmd.add_argument("terminal_id", help="Your terminal ID")
    complete_cmd.add_argument("task_id", help="Task ID")
    complete_cmd.add_argument("output", help="Task output/summary")
    complete_cmd.add_argument("--modified", "-m", nargs="*", help="Files modified")
    complete_cmd.add_argument("--created", "-c", nargs="*", help="Files created")

    fail_cmd = worker_sub.add_parser("fail", help="Mark task as failed")
    fail_cmd.add_argument("terminal_id", help="Your terminal ID")
    fail_cmd.add_argument("task_id", help="Task ID")
    fail_cmd.add_argument("reason", help="Failure reason")

    worker_sub.add_parser("list", help="List available tasks")

    args = parser.parse_args()

    if args.role == "leader":
        if args.command == "init":
            leader_init(args.goal, args.approach)
        elif args.command == "add-task":
            leader_add_task(args.description, args.priority, args.depends, args.files, args.hints)
        elif args.command == "status":
            leader_status()
        elif args.command == "aggregate":
            leader_aggregate()
        else:
            leader_parser.print_help()

    elif args.role == "register":
        register_agent(args.agent_id, args.capabilities)

    elif args.role == "worker":
        if args.command == "claim":
            worker_claim(args.terminal_id)
        elif args.command == "start":
            worker_start(args.terminal_id, args.task_id)
        elif args.command == "complete":
            worker_complete(args.terminal_id, args.task_id, args.output, args.modified, args.created)
        elif args.command == "fail":
            worker_fail(args.terminal_id, args.task_id, args.reason)
        elif args.command == "list":
            worker_list_available()
        else:
            worker_parser.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
