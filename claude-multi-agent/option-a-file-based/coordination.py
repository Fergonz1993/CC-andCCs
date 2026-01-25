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
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
import argparse
import time

# Try to import yaml for batch task creation
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Configuration
COORDINATION_DIR = Path(".coordination")
TASKS_FILE = COORDINATION_DIR / "tasks.json"
MASTER_PLAN_FILE = COORDINATION_DIR / "master-plan.md"
DISCOVERIES_FILE = COORDINATION_DIR / "context" / "discoveries.md"
LOGS_DIR = COORDINATION_DIR / "logs"
RESULTS_DIR = COORDINATION_DIR / "results"
TEMPLATES_DIR = COORDINATION_DIR / "templates"
GROUPS_FILE = COORDINATION_DIR / "groups.json"

# Recovery feature directories and files
CHECKPOINTS_DIR = COORDINATION_DIR / "checkpoints"
BACKUPS_DIR = COORDINATION_DIR / "backups"
TRANSACTION_LOG_FILE = COORDINATION_DIR / "transaction.log"
HEARTBEAT_FILE = COORDINATION_DIR / "heartbeats.json"

# Metrics feature files
METRICS_FILE = COORDINATION_DIR / "metrics.json"
METRICS_HISTORY_DIR = COORDINATION_DIR / "metrics_history"

# Recovery configuration constants
BACKUP_RETENTION_COUNT = 10  # Keep last N backups
HEARTBEAT_TIMEOUT_SECONDS = 60  # Worker heartbeat timeout
ORPHAN_TASK_TIMEOUT_MINUTES = 15  # Timeout for orphan task detection
CHECKPOINT_INTERVAL_SECONDS = 300  # Auto-checkpoint every 5 minutes

# Metrics configuration constants
METRICS_PORT = 9100  # Prometheus metrics port
METRICS_HISTORY_RETENTION_HOURS = 24  # Keep 24 hours of metrics history
THROUGHPUT_WINDOW_SECONDS = 60  # Window for throughput calculations

# Advanced configuration constants
PRIORITY_BOOST_THRESHOLD_MINUTES = 5  # Boost priority if waiting > 5 minutes
PRIORITY_BOOST_AMOUNT = 1  # Reduce priority value by this much (lower = higher priority)
STALE_WORKER_TIMEOUT_MINUTES = 10  # Consider worker stale after 10 minutes of no activity
DEFAULT_TASK_TIMEOUT_MINUTES = 60  # Default timeout for tasks
DEFAULT_MAX_RETRIES = 3  # Default maximum retries for failed tasks
INITIAL_RETRY_DELAY_SECONDS = 30  # Initial delay before retry (exponential backoff)


@dataclass
class Task:
    """
    Represents a task in the coordination system.

    Status lifecycle: available -> claimed -> in_progress -> done | failed | cancelled

    Attributes:
        id: Unique task identifier
        description: Human-readable task description
        status: Current task status
        priority: Task priority (1-10, lower = higher priority)
        claimed_by: ID of the worker that claimed this task
        dependencies: List of task IDs that must complete before this task
        context: Additional context for task execution
        result: Task execution result
        created_at: ISO timestamp when task was created
        claimed_at: ISO timestamp when task was claimed
        completed_at: ISO timestamp when task was completed
        tags: List of tags for filtering and categorization
        parent_id: ID of parent task (for subtasks)
        estimated_duration: Estimated duration in minutes
        actual_duration: Actual duration in minutes (calculated on completion)
        required_capabilities: Capabilities required to work on this task
        max_retries: Maximum number of retry attempts
        retry_count: Current retry count
        retry_delay: Delay before next retry in seconds
        next_retry_at: ISO timestamp when next retry is allowed
        timeout_minutes: Task-specific timeout in minutes
        group_id: ID of the task group (for parallel execution)
        priority_boost: Current priority boost from wait time
    """
    id: str
    description: str
    status: str  # available, claimed, in_progress, done, failed, cancelled
    priority: int
    claimed_by: Optional[str] = None
    dependencies: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    claimed_at: Optional[str] = None
    completed_at: Optional[str] = None
    # Advanced features
    tags: Optional[List[str]] = None
    parent_id: Optional[str] = None
    estimated_duration: Optional[int] = None  # in minutes
    actual_duration: Optional[int] = None  # in minutes
    required_capabilities: Optional[List[str]] = None
    max_retries: Optional[int] = None
    retry_count: Optional[int] = None
    retry_delay: Optional[int] = None  # in seconds
    next_retry_at: Optional[str] = None
    timeout_minutes: Optional[int] = None
    group_id: Optional[str] = None
    priority_boost: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert task to dictionary, removing None values for cleaner JSON."""
        d = asdict(self)
        # Remove None values for cleaner JSON
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Create a Task from a dictionary."""
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
            tags=data.get("tags"),
            parent_id=data.get("parent_id"),
            estimated_duration=data.get("estimated_duration"),
            actual_duration=data.get("actual_duration"),
            required_capabilities=data.get("required_capabilities"),
            max_retries=data.get("max_retries"),
            retry_count=data.get("retry_count"),
            retry_delay=data.get("retry_delay"),
            next_retry_at=data.get("next_retry_at"),
            timeout_minutes=data.get("timeout_minutes"),
            group_id=data.get("group_id"),
            priority_boost=data.get("priority_boost"),
        )

    def get_effective_priority(self) -> int:
        """Get effective priority including any boosts."""
        base = self.priority
        boost = self.priority_boost or 0
        return max(1, base - boost)  # Lower number = higher priority, minimum 1


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
    TEMPLATES_DIR.mkdir(exist_ok=True)

    # Recovery directories
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    BACKUPS_DIR.mkdir(exist_ok=True)

    # Metrics directories
    METRICS_HISTORY_DIR.mkdir(exist_ok=True)

    if not TASKS_FILE.exists():
        save_tasks({"version": "1.0", "created_at": now_iso(), "last_updated": now_iso(), "tasks": []})

    if not DISCOVERIES_FILE.exists():
        DISCOVERIES_FILE.write_text("# Shared Discoveries\n\nImportant findings go here.\n")

    if not GROUPS_FILE.exists():
        save_groups({"version": "1.0", "created_at": now_iso(), "groups": []})

    # Initialize metrics file
    if not METRICS_FILE.exists():
        init_metrics()


def load_tasks() -> dict:
    """Load the tasks file with locking."""
    if not TASKS_FILE.exists():
        return {"version": "1.0", "created_at": now_iso(), "last_updated": now_iso(), "tasks": []}

    with file_lock(TASKS_FILE, exclusive=False):
        return json.loads(TASKS_FILE.read_text())


def load_tasks_no_lock() -> dict:
    """
    Load the tasks file WITHOUT acquiring a lock.

    IMPORTANT: Only use this inside a file_lock() context to avoid TOCTOU races.
    This helper exists for atomic transactions that need to read and write
    within a single lock context.
    """
    if not TASKS_FILE.exists():
        return {"version": "1.0", "created_at": now_iso(), "last_updated": now_iso(), "tasks": []}
    return json.loads(TASKS_FILE.read_text())


def save_tasks_no_lock(data: dict):
    """
    Save the tasks file WITHOUT acquiring a lock.

    IMPORTANT: Only use this inside a file_lock() context to avoid TOCTOU races.
    This helper exists for atomic transactions that need to read and write
    within a single lock context.

    Note: Does NOT create backups - caller should handle backup if needed.
    """
    data["last_updated"] = now_iso()
    TASKS_FILE.write_text(json.dumps(data, indent=2) + "\n")


def cleanup_old_backups():
    """Remove old backups beyond the retention count."""
    backups = sorted(BACKUPS_DIR.glob("tasks_backup_*.json"))
    while len(backups) > BACKUP_RETENTION_COUNT:
        oldest = backups.pop(0)
        oldest.unlink()
        log_action("system", "BACKUP_DELETED", str(oldest))


def create_backup():
    """
    Create a timestamped backup of the current state.
    Implements: adv-a-011 - Automatic state backup on every change
    """
    ensure_coordination_structure()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_file = BACKUPS_DIR / f"tasks_backup_{timestamp}.json"

    if TASKS_FILE.exists():
        data = load_tasks()
        data["backup_created"] = now_iso()
        with file_lock(backup_file, exclusive=True):
            backup_file.write_text(json.dumps(data, indent=2))

        # Maintain rolling backup window
        cleanup_old_backups()
        log_action("system", "BACKUP_CREATED", str(backup_file))
        return backup_file
    return None


def save_tasks(data: dict):
    """Save the tasks file with locking and automatic backup (adv-a-011)."""
    # Create backup before modifying state (adv-a-011: Automatic state backup)
    if TASKS_FILE.exists():
        create_backup()

    data["last_updated"] = now_iso()
    with file_lock(TASKS_FILE, exclusive=True):
        TASKS_FILE.write_text(json.dumps(data, indent=2) + "\n")  # Add trailing newline


def load_groups() -> dict:
    """Load the groups file with locking."""
    if not GROUPS_FILE.exists():
        return {"version": "1.0", "created_at": now_iso(), "groups": []}

    with file_lock(GROUPS_FILE, exclusive=False):
        return json.loads(GROUPS_FILE.read_text())


def save_groups(data: dict):
    """Save the groups file with locking."""
    data["last_updated"] = now_iso()
    with file_lock(GROUPS_FILE, exclusive=True):
        GROUPS_FILE.write_text(json.dumps(data, indent=2))


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
    hints: str = "",
    tags: Optional[List[str]] = None,
    parent_id: Optional[str] = None,
    estimated_duration: Optional[int] = None,
    required_capabilities: Optional[List[str]] = None,
    max_retries: Optional[int] = None,
    timeout_minutes: Optional[int] = None,
    group_id: Optional[str] = None,
) -> str:
    """
    Add a new task to the queue.

    Args:
        description: Human-readable task description
        priority: Task priority (1-10, lower = higher priority)
        dependencies: List of task IDs that must complete before this task
        context_files: List of relevant file paths
        hints: Additional hints for the worker
        tags: List of tags for categorization and filtering
        parent_id: ID of the parent task (for subtasks)
        estimated_duration: Estimated duration in minutes
        required_capabilities: List of capabilities required to work on this task
        max_retries: Maximum retry attempts on failure
        timeout_minutes: Task-specific timeout in minutes
        group_id: Task group ID for parallel execution with barrier sync

    Returns:
        The generated task ID
    """
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
        created_at=now_iso(),
        tags=tags,
        parent_id=parent_id,
        estimated_duration=estimated_duration,
        required_capabilities=required_capabilities,
        max_retries=max_retries,
        timeout_minutes=timeout_minutes,
        group_id=group_id,
    )

    data = load_tasks()
    data["tasks"].append(task.to_dict())
    save_tasks(data)

    # Record metrics after saving (adv-a-016)
    record_task_created()

    log_action("leader", "ADD_TASK", f"{task.id}: {description[:50]}")
    print(f"✓ Added task {task.id}")
    return task.id


def leader_status():
    """Show current coordination status including all task states."""
    data = load_tasks()
    tasks = [Task.from_dict(t) for t in data["tasks"]]

    by_status = {}
    for task in tasks:
        by_status.setdefault(task.status, []).append(task)

    print("\n" + "=" * 60)
    print("COORDINATION STATUS")
    print("=" * 60)

    for status in ["available", "claimed", "in_progress", "done", "failed", "cancelled"]:
        status_tasks = by_status.get(status, [])
        if status_tasks:
            print(f"\n{status.upper()} ({len(status_tasks)}):")
            for t in sorted(status_tasks, key=lambda x: x.get_effective_priority()):
                claimed = f" [by {t.claimed_by}]" if t.claimed_by else ""
                tags_str = f" [{', '.join(t.tags)}]" if t.tags else ""
                boost = f" (+{t.priority_boost} boost)" if t.priority_boost else ""
                print(f"  [{t.get_effective_priority()}] {t.id}: {t.description[:35]}...{claimed}{tags_str}{boost}")

    print("\n" + "-" * 60)
    total = len(tasks)
    done = len(by_status.get("done", []))
    cancelled = len(by_status.get("cancelled", []))
    active = total - done - cancelled - len(by_status.get("failed", []))
    print(f"Progress: {done}/{total} tasks complete ({100*done//total if total else 0}%)")
    if cancelled:
        print(f"Cancelled: {cancelled} tasks")
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
# RECOVERY FEATURES (adv-a-011 to adv-a-015)
# Note: create_backup() and cleanup_old_backups() moved before save_tasks()
# ============================================================================

def restore_from_backup(backup_file: Optional[Path] = None) -> bool:
    """
    Restore state from a backup file.
    Implements: adv-a-012 - Crash recovery from backup

    Args:
        backup_file: Specific backup file to restore from, or None for latest

    Returns:
        True if restore was successful, False otherwise
    """
    ensure_coordination_structure()

    if backup_file is None:
        backups = sorted(BACKUPS_DIR.glob("tasks_backup_*.json"))
        if not backups:
            print("No backup files found.")
            return False
        backup_file = backups[-1]  # Get latest backup

    if not backup_file.exists():
        print(f"Backup file not found: {backup_file}")
        return False

    try:
        with file_lock(backup_file, exclusive=False):
            data = json.loads(backup_file.read_text())

        # Validate backup integrity
        if "tasks" not in data:
            print("Invalid backup file: missing 'tasks' field")
            return False

        data["restored_from"] = str(backup_file)
        data["restored_at"] = now_iso()
        save_tasks(data)

        log_action("system", "RESTORED_FROM_BACKUP", str(backup_file))
        print(f"✓ Restored from backup: {backup_file}")
        return True
    except json.JSONDecodeError as e:
        print(f"Failed to restore: corrupted backup file - {e}")
        return False


def detect_crash() -> bool:
    """
    Detect if the tasks.json file is corrupted.
    Implements: adv-a-012 - Automatic crash detection

    Returns:
        True if corruption detected, False if file is valid
    """
    if not TASKS_FILE.exists():
        return False

    try:
        with file_lock(TASKS_FILE, exclusive=False):
            data = json.loads(TASKS_FILE.read_text())
        # Validate structure
        if "tasks" not in data or not isinstance(data["tasks"], list):
            return True
        return False
    except (json.JSONDecodeError, IOError):
        return True


def auto_recover() -> bool:
    """
    Automatically detect crash and recover from backup.

    Returns:
        True if recovery was performed, False otherwise
    """
    if detect_crash():
        log_action("system", "CRASH_DETECTED", "Attempting auto-recovery")
        print("! Crash detected - attempting recovery from backup...")
        return restore_from_backup()
    return False


def write_transaction_log(action: str, data: Dict[str, Any]):
    """
    Write to the transaction log for state changes.
    Implements: adv-a-013 - Transaction log for all state changes
    """
    ensure_coordination_structure()
    entry = {
        "timestamp": now_iso(),
        "action": action,
        "data": data
    }
    with open(TRANSACTION_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def replay_transaction_log(from_timestamp: Optional[str] = None) -> List[Dict]:
    """
    Replay transaction log entries.

    Args:
        from_timestamp: Only replay entries after this timestamp

    Returns:
        List of transaction entries
    """
    if not TRANSACTION_LOG_FILE.exists():
        return []

    entries = []
    with open(TRANSACTION_LOG_FILE, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if from_timestamp is None or entry["timestamp"] > from_timestamp:
                    entries.append(entry)
    return entries


def compact_transaction_log(keep_hours: int = 24):
    """
    Compact the transaction log by removing old entries.

    Args:
        keep_hours: Keep entries from the last N hours
    """
    if not TRANSACTION_LOG_FILE.exists():
        return

    cutoff = datetime.now() - timedelta(hours=keep_hours)
    cutoff_str = cutoff.isoformat()

    entries = []
    with open(TRANSACTION_LOG_FILE, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry["timestamp"] > cutoff_str:
                    entries.append(entry)

    with open(TRANSACTION_LOG_FILE, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    log_action("system", "TRANSACTION_LOG_COMPACTED", f"Kept {len(entries)} entries")


def save_checkpoint(checkpoint_name: Optional[str] = None) -> Path:
    """
    Save a coordination checkpoint.
    Implements: adv-a-015 - Checkpoint and resume coordination

    Args:
        checkpoint_name: Optional name for the checkpoint

    Returns:
        Path to the checkpoint file
    """
    ensure_coordination_structure()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = checkpoint_name or f"checkpoint_{timestamp}"
    checkpoint_file = CHECKPOINTS_DIR / f"{name}.json"

    checkpoint_data = {
        "checkpoint_name": name,
        "created_at": now_iso(),
        "tasks": load_tasks() if TASKS_FILE.exists() else {"tasks": []},
        "agents": load_agents() if AGENTS_FILE.exists() else {"agents": []},
        "groups": load_groups() if GROUPS_FILE.exists() else {"groups": []},
        "checksum": None  # Will be calculated
    }

    # Calculate checksum for integrity validation
    content = json.dumps(checkpoint_data, sort_keys=True)
    checkpoint_data["checksum"] = hashlib.sha256(content.encode()).hexdigest()

    with file_lock(checkpoint_file, exclusive=True):
        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))

    log_action("system", "CHECKPOINT_SAVED", str(checkpoint_file))
    print(f"✓ Checkpoint saved: {checkpoint_file}")
    return checkpoint_file


def restore_checkpoint(checkpoint_name: str) -> bool:
    """
    Restore from a checkpoint.

    Args:
        checkpoint_name: Name of the checkpoint to restore

    Returns:
        True if restore was successful
    """
    ensure_coordination_structure()
    checkpoint_file = CHECKPOINTS_DIR / f"{checkpoint_name}.json"

    if not checkpoint_file.exists():
        # Try with .json extension
        if not checkpoint_name.endswith(".json"):
            checkpoint_file = CHECKPOINTS_DIR / checkpoint_name
        if not checkpoint_file.exists():
            print(f"Checkpoint not found: {checkpoint_name}")
            return False

    try:
        with file_lock(checkpoint_file, exclusive=False):
            checkpoint_data = json.loads(checkpoint_file.read_text())

        # Validate checkpoint integrity
        stored_checksum = checkpoint_data.pop("checksum", None)
        if stored_checksum:
            content = json.dumps(checkpoint_data, sort_keys=True)
            calculated_checksum = hashlib.sha256(content.encode()).hexdigest()
            if stored_checksum != calculated_checksum:
                print("! Warning: Checkpoint integrity check failed")

        # Restore state
        if "tasks" in checkpoint_data:
            save_tasks(checkpoint_data["tasks"])
        if "agents" in checkpoint_data:
            save_agents(checkpoint_data["agents"])
        if "groups" in checkpoint_data:
            save_groups(checkpoint_data["groups"])

        log_action("system", "CHECKPOINT_RESTORED", str(checkpoint_file))
        print(f"✓ Restored from checkpoint: {checkpoint_name}")
        return True
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to restore checkpoint: {e}")
        return False


def list_checkpoints() -> List[Dict]:
    """List all available checkpoints."""
    ensure_coordination_structure()
    checkpoints = []
    for f in sorted(CHECKPOINTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            checkpoints.append({
                "name": data.get("checkpoint_name", f.stem),
                "created_at": data.get("created_at"),
                "file": str(f)
            })
        except json.JSONDecodeError:
            continue
    return checkpoints


# Heartbeat monitoring for workers (adv-a-014)
def load_heartbeats() -> Dict:
    """Load heartbeat data."""
    if not HEARTBEAT_FILE.exists():
        return {"workers": {}}
    with file_lock(HEARTBEAT_FILE, exclusive=False):
        return json.loads(HEARTBEAT_FILE.read_text())


def save_heartbeats(data: Dict):
    """Save heartbeat data."""
    with file_lock(HEARTBEAT_FILE, exclusive=True):
        HEARTBEAT_FILE.write_text(json.dumps(data, indent=2))


def worker_heartbeat(worker_id: str, task_id: Optional[str] = None):
    """
    Send a heartbeat from a worker.
    Implements: adv-a-014 - Worker heartbeat monitoring

    Args:
        worker_id: The worker's identifier
        task_id: Currently working task ID (optional)
    """
    ensure_coordination_structure()
    data = load_heartbeats()

    data["workers"][worker_id] = {
        "last_heartbeat": now_iso(),
        "current_task": task_id,
        "status": "active"
    }
    save_heartbeats(data)


def check_worker_health() -> Dict[str, str]:
    """
    Check health of all workers based on heartbeats.

    Returns:
        Dict mapping worker_id to health status
    """
    data = load_heartbeats()
    now = datetime.now()
    health = {}

    for worker_id, info in data.get("workers", {}).items():
        last_hb = datetime.fromisoformat(info["last_heartbeat"])
        age_seconds = (now - last_hb).total_seconds()

        if age_seconds < HEARTBEAT_TIMEOUT_SECONDS:
            health[worker_id] = "healthy"
        elif age_seconds < HEARTBEAT_TIMEOUT_SECONDS * 2:
            health[worker_id] = "warning"
        else:
            health[worker_id] = "dead"

    return health


def get_dead_workers() -> List[str]:
    """Get list of dead workers based on heartbeat timeout."""
    health = check_worker_health()
    return [w for w, status in health.items() if status == "dead"]


# Orphan task cleanup (adv-a-014)
def detect_orphan_tasks() -> List[Task]:
    """
    Detect tasks claimed by dead workers.
    Implements: adv-a-014 - Orphaned task detection and cleanup

    Returns:
        List of orphaned tasks
    """
    data = load_tasks()
    tasks = [Task.from_dict(t) for t in data["tasks"]]
    dead_workers = set(get_dead_workers())

    orphans = []
    now = datetime.now()

    for task in tasks:
        if task.status in ["claimed", "in_progress"] and task.claimed_by:
            # Check if worker is dead
            if task.claimed_by in dead_workers:
                orphans.append(task)
            # Also check based on claimed_at timeout
            elif task.claimed_at:
                claimed_time = datetime.fromisoformat(task.claimed_at)
                age_minutes = (now - claimed_time).total_seconds() / 60
                if age_minutes > ORPHAN_TASK_TIMEOUT_MINUTES:
                    orphans.append(task)

    return orphans


def cleanup_orphan_tasks() -> int:
    """
    Reset orphaned tasks to available status.
    Implements: adv-a-014 - Orphaned task detection and cleanup

    Returns:
        Number of tasks cleaned up
    """
    orphans = detect_orphan_tasks()
    if not orphans:
        return 0

    data = load_tasks()
    cleaned = 0

    for orphan in orphans:
        for i, t in enumerate(data["tasks"]):
            if t["id"] == orphan.id:
                old_status = t["status"]
                old_worker = t.get("claimed_by")
                data["tasks"][i]["status"] = "available"
                data["tasks"][i]["claimed_by"] = None
                data["tasks"][i]["claimed_at"] = None
                # Track orphan cleanup in context
                if "context" not in data["tasks"][i] or data["tasks"][i]["context"] is None:
                    data["tasks"][i]["context"] = {}
                data["tasks"][i]["context"]["orphaned_from"] = {
                    "worker": old_worker,
                    "status": old_status,
                    "cleaned_at": now_iso()
                }
                cleaned += 1
                log_action("system", "ORPHAN_CLEANUP", f"{orphan.id} from {old_worker}")
                break

    if cleaned > 0:
        save_tasks(data)
        print(f"✓ Cleaned up {cleaned} orphaned tasks")

    return cleaned


# ============================================================================
# METRICS FEATURES (adv-a-016 to adv-a-020)
# ============================================================================

@dataclass
class MetricsData:
    """Container for all metrics data."""
    # Task throughput (adv-a-016)
    tasks_completed_total: int = 0
    tasks_failed_total: int = 0
    tasks_created_total: int = 0
    tasks_completed_per_minute: float = 0.0

    # Worker efficiency (adv-a-017)
    worker_stats: Dict[str, Dict] = field(default_factory=dict)

    # Queue metrics (adv-a-018)
    queue_depth: int = 0
    average_wait_time_seconds: float = 0.0
    max_wait_time_seconds: float = 0.0

    # Error metrics (adv-a-019)
    error_count_total: int = 0
    error_rate_per_minute: float = 0.0
    errors_by_category: Dict[str, int] = field(default_factory=dict)

    # Latency histograms (adv-a-020)
    task_duration_histogram: Dict[str, int] = field(default_factory=dict)
    claim_to_start_histogram: Dict[str, int] = field(default_factory=dict)

    # Metadata
    last_updated: str = ""
    collection_started: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MetricsData":
        return cls(**data)


def init_metrics():
    """Initialize the metrics file."""
    metrics = MetricsData(
        last_updated=now_iso(),
        collection_started=now_iso()
    )
    with file_lock(METRICS_FILE, exclusive=True):
        METRICS_FILE.write_text(json.dumps(metrics.to_dict(), indent=2))


def load_metrics() -> MetricsData:
    """Load metrics data."""
    if not METRICS_FILE.exists():
        init_metrics()
    with file_lock(METRICS_FILE, exclusive=False):
        data = json.loads(METRICS_FILE.read_text())
    return MetricsData.from_dict(data)


def save_metrics(metrics: MetricsData):
    """Save metrics data."""
    metrics.last_updated = now_iso()
    with file_lock(METRICS_FILE, exclusive=True):
        METRICS_FILE.write_text(json.dumps(metrics.to_dict(), indent=2))


def record_task_completion(worker_id: str, task_id: str, duration_seconds: float):
    """
    Record a task completion for metrics.
    Implements: adv-a-016 - Task throughput tracking
    Implements: adv-a-017 - Worker efficiency scoring
    """
    metrics = load_metrics()

    # Update throughput
    metrics.tasks_completed_total += 1

    # Update worker stats
    if worker_id not in metrics.worker_stats:
        metrics.worker_stats[worker_id] = {
            "completed": 0,
            "failed": 0,
            "total_duration_seconds": 0.0,
            "average_duration_seconds": 0.0
        }
    stats = metrics.worker_stats[worker_id]
    stats["completed"] += 1
    stats["total_duration_seconds"] += duration_seconds
    stats["average_duration_seconds"] = stats["total_duration_seconds"] / stats["completed"]

    # Update latency histogram
    bucket = get_duration_bucket(duration_seconds)
    metrics.task_duration_histogram[bucket] = metrics.task_duration_histogram.get(bucket, 0) + 1

    save_metrics(metrics)
    write_transaction_log("TASK_COMPLETED", {"task_id": task_id, "worker_id": worker_id, "duration": duration_seconds})


def record_task_failure(worker_id: str, task_id: str, error_category: str = "unknown"):
    """
    Record a task failure for metrics.
    Implements: adv-a-019 - Error rate monitoring
    """
    metrics = load_metrics()

    metrics.tasks_failed_total += 1
    metrics.error_count_total += 1
    metrics.errors_by_category[error_category] = metrics.errors_by_category.get(error_category, 0) + 1

    # Update worker stats
    if worker_id not in metrics.worker_stats:
        metrics.worker_stats[worker_id] = {
            "completed": 0,
            "failed": 0,
            "total_duration_seconds": 0.0,
            "average_duration_seconds": 0.0
        }
    metrics.worker_stats[worker_id]["failed"] += 1

    save_metrics(metrics)
    write_transaction_log("TASK_FAILED", {"task_id": task_id, "worker_id": worker_id, "category": error_category})


def record_task_created():
    """Record a task creation for metrics."""
    metrics = load_metrics()
    metrics.tasks_created_total += 1
    save_metrics(metrics)


def get_duration_bucket(seconds: float) -> str:
    """
    Get histogram bucket for a duration.
    Implements: adv-a-020 - Latency histograms

    Buckets: <10s, 10-30s, 30-60s, 1-5m, 5-15m, 15-30m, 30-60m, >60m
    """
    if seconds < 10:
        return "<10s"
    elif seconds < 30:
        return "10-30s"
    elif seconds < 60:
        return "30-60s"
    elif seconds < 300:
        return "1-5m"
    elif seconds < 900:
        return "5-15m"
    elif seconds < 1800:
        return "15-30m"
    elif seconds < 3600:
        return "30-60m"
    else:
        return ">60m"


def calculate_queue_metrics():
    """
    Calculate current queue metrics.
    Implements: adv-a-018 - Queue depth and wait time metrics
    """
    data = load_tasks()
    tasks = [Task.from_dict(t) for t in data["tasks"]]
    now = datetime.now()

    # Queue depth (available tasks)
    available = [t for t in tasks if t.status == "available"]
    queue_depth = len(available)

    # Calculate wait times
    wait_times = []
    for task in available:
        if task.created_at:
            created = datetime.fromisoformat(task.created_at)
            wait_seconds = (now - created).total_seconds()
            wait_times.append(wait_seconds)

    avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0
    max_wait = max(wait_times) if wait_times else 0.0

    metrics = load_metrics()
    metrics.queue_depth = queue_depth
    metrics.average_wait_time_seconds = avg_wait
    metrics.max_wait_time_seconds = max_wait
    save_metrics(metrics)

    return {
        "queue_depth": queue_depth,
        "average_wait_seconds": avg_wait,
        "max_wait_seconds": max_wait
    }


def calculate_throughput():
    """
    Calculate task throughput metrics.
    Implements: adv-a-016 - Task throughput tracking
    """
    # Read recent completions from transaction log
    entries = replay_transaction_log()
    now = datetime.now()
    window_start = now - timedelta(seconds=THROUGHPUT_WINDOW_SECONDS)

    completions_in_window = 0
    for entry in entries:
        if entry["action"] == "TASK_COMPLETED":
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time >= window_start:
                completions_in_window += 1

    throughput_per_minute = (completions_in_window / THROUGHPUT_WINDOW_SECONDS) * 60

    metrics = load_metrics()
    metrics.tasks_completed_per_minute = throughput_per_minute
    save_metrics(metrics)

    return throughput_per_minute


def calculate_error_rate():
    """
    Calculate error rate.
    Implements: adv-a-019 - Error rate monitoring
    """
    entries = replay_transaction_log()
    now = datetime.now()
    window_start = now - timedelta(seconds=THROUGHPUT_WINDOW_SECONDS)

    errors_in_window = 0
    for entry in entries:
        if entry["action"] == "TASK_FAILED":
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time >= window_start:
                errors_in_window += 1

    error_rate = (errors_in_window / THROUGHPUT_WINDOW_SECONDS) * 60

    metrics = load_metrics()
    metrics.error_rate_per_minute = error_rate
    save_metrics(metrics)

    return error_rate


def get_worker_utilization() -> Dict[str, Dict]:
    """
    Get worker utilization metrics.
    Implements: adv-a-017 - Worker utilization metrics
    """
    data = load_tasks()
    tasks = [Task.from_dict(t) for t in data["tasks"]]
    metrics = load_metrics()

    utilization = {}
    for worker_id, stats in metrics.worker_stats.items():
        # Count active tasks
        active_tasks = [t for t in tasks if t.claimed_by == worker_id and t.status in ["claimed", "in_progress"]]

        completed = stats.get("completed", 0)
        failed = stats.get("failed", 0)
        total_tasks = completed + failed

        utilization[worker_id] = {
            "active_tasks": len(active_tasks),
            "completed_total": completed,
            "failed_total": failed,
            "success_rate": completed / total_tasks if total_tasks > 0 else 1.0,
            "average_duration_seconds": stats.get("average_duration_seconds", 0)
        }

    return utilization


def generate_prometheus_metrics() -> str:
    """
    Generate Prometheus-compatible metrics output.
    Implements: adv-a-016 (Prometheus format), adv-a-017, adv-a-018, adv-a-019, adv-a-020

    Returns:
        Prometheus metrics format string
    """
    # Refresh calculated metrics
    calculate_queue_metrics()
    calculate_throughput()
    calculate_error_rate()

    metrics = load_metrics()
    lines = []

    # Task throughput metrics
    lines.append("# HELP coordination_tasks_completed_total Total number of completed tasks")
    lines.append("# TYPE coordination_tasks_completed_total counter")
    lines.append(f"coordination_tasks_completed_total {metrics.tasks_completed_total}")

    lines.append("# HELP coordination_tasks_failed_total Total number of failed tasks")
    lines.append("# TYPE coordination_tasks_failed_total counter")
    lines.append(f"coordination_tasks_failed_total {metrics.tasks_failed_total}")

    lines.append("# HELP coordination_tasks_created_total Total number of created tasks")
    lines.append("# TYPE coordination_tasks_created_total counter")
    lines.append(f"coordination_tasks_created_total {metrics.tasks_created_total}")

    lines.append("# HELP coordination_tasks_throughput_per_minute Tasks completed per minute")
    lines.append("# TYPE coordination_tasks_throughput_per_minute gauge")
    lines.append(f"coordination_tasks_throughput_per_minute {metrics.tasks_completed_per_minute:.4f}")

    # Queue metrics
    lines.append("# HELP coordination_queue_depth Number of available tasks in queue")
    lines.append("# TYPE coordination_queue_depth gauge")
    lines.append(f"coordination_queue_depth {metrics.queue_depth}")

    lines.append("# HELP coordination_queue_wait_seconds_avg Average wait time in queue")
    lines.append("# TYPE coordination_queue_wait_seconds_avg gauge")
    lines.append(f"coordination_queue_wait_seconds_avg {metrics.average_wait_time_seconds:.2f}")

    lines.append("# HELP coordination_queue_wait_seconds_max Maximum wait time in queue")
    lines.append("# TYPE coordination_queue_wait_seconds_max gauge")
    lines.append(f"coordination_queue_wait_seconds_max {metrics.max_wait_time_seconds:.2f}")

    # Error metrics
    lines.append("# HELP coordination_errors_total Total number of errors")
    lines.append("# TYPE coordination_errors_total counter")
    lines.append(f"coordination_errors_total {metrics.error_count_total}")

    lines.append("# HELP coordination_error_rate_per_minute Errors per minute")
    lines.append("# TYPE coordination_error_rate_per_minute gauge")
    lines.append(f"coordination_error_rate_per_minute {metrics.error_rate_per_minute:.4f}")

    # Error by category
    lines.append("# HELP coordination_errors_by_category Errors by category")
    lines.append("# TYPE coordination_errors_by_category counter")
    for category, count in metrics.errors_by_category.items():
        lines.append(f'coordination_errors_by_category{{category="{category}"}} {count}')

    # Worker metrics
    lines.append("# HELP coordination_worker_completed_total Tasks completed by worker")
    lines.append("# TYPE coordination_worker_completed_total counter")
    for worker_id, stats in metrics.worker_stats.items():
        lines.append(f'coordination_worker_completed_total{{worker="{worker_id}"}} {stats.get("completed", 0)}')

    lines.append("# HELP coordination_worker_failed_total Tasks failed by worker")
    lines.append("# TYPE coordination_worker_failed_total counter")
    for worker_id, stats in metrics.worker_stats.items():
        lines.append(f'coordination_worker_failed_total{{worker="{worker_id}"}} {stats.get("failed", 0)}')

    lines.append("# HELP coordination_worker_avg_duration_seconds Average task duration by worker")
    lines.append("# TYPE coordination_worker_avg_duration_seconds gauge")
    for worker_id, stats in metrics.worker_stats.items():
        lines.append(f'coordination_worker_avg_duration_seconds{{worker="{worker_id}"}} {stats.get("average_duration_seconds", 0):.2f}')

    # Latency histograms
    lines.append("# HELP coordination_task_duration_bucket Task duration distribution")
    lines.append("# TYPE coordination_task_duration_bucket histogram")
    for bucket, count in sorted(metrics.task_duration_histogram.items()):
        lines.append(f'coordination_task_duration_bucket{{le="{bucket}"}} {count}')

    return "\n".join(lines)


def start_metrics_server(port: int = None):
    """
    Start a simple HTTP server for Prometheus metrics.
    Implements: adv-a-016 - Prometheus metrics endpoint

    Args:
        port: Port to listen on (default: METRICS_PORT)
    """
    import http.server
    import socketserver

    port = port or METRICS_PORT

    class MetricsHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                metrics_output = generate_prometheus_metrics()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(metrics_output.encode())
            elif self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "healthy"}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress logging

    with socketserver.TCPServer(("", port), MetricsHandler) as httpd:
        print(f"✓ Metrics server started on port {port}")
        print(f"  Prometheus endpoint: http://localhost:{port}/metrics")
        print(f"  Health check: http://localhost:{port}/health")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n✓ Metrics server stopped")


def print_metrics_dashboard():
    """
    Print a terminal dashboard with metrics.
    Implements: adv-a-020 - Progress dashboard in terminal
    """
    # Refresh metrics
    calculate_queue_metrics()
    calculate_throughput()
    calculate_error_rate()

    metrics = load_metrics()
    utilization = get_worker_utilization()
    health = check_worker_health()

    print("\n" + "=" * 70)
    print("                    COORDINATION METRICS DASHBOARD")
    print("=" * 70)

    # Task throughput
    print("\n[THROUGHPUT]")
    print(f"  Tasks Completed: {metrics.tasks_completed_total}")
    print(f"  Tasks Failed:    {metrics.tasks_failed_total}")
    print(f"  Tasks Created:   {metrics.tasks_created_total}")
    print(f"  Throughput:      {metrics.tasks_completed_per_minute:.2f} tasks/min")

    # Queue status
    print("\n[QUEUE]")
    print(f"  Queue Depth:     {metrics.queue_depth}")
    print(f"  Avg Wait Time:   {metrics.average_wait_time_seconds:.1f}s")
    print(f"  Max Wait Time:   {metrics.max_wait_time_seconds:.1f}s")

    # Error rates
    print("\n[ERRORS]")
    print(f"  Total Errors:    {metrics.error_count_total}")
    print(f"  Error Rate:      {metrics.error_rate_per_minute:.4f} errors/min")
    if metrics.errors_by_category:
        print("  By Category:")
        for cat, count in sorted(metrics.errors_by_category.items(), key=lambda x: -x[1]):
            print(f"    - {cat}: {count}")

    # Worker utilization
    print("\n[WORKERS]")
    if utilization:
        print(f"  {'Worker':<20} {'Active':<8} {'Done':<8} {'Failed':<8} {'Success%':<10} {'Health':<10}")
        print("  " + "-" * 64)
        for worker_id, stats in sorted(utilization.items()):
            worker_health = health.get(worker_id, "unknown")
            health_indicator = {"healthy": "[OK]", "warning": "[!]", "dead": "[X]", "unknown": "[?]"}[worker_health]
            print(f"  {worker_id:<20} {stats['active_tasks']:<8} {stats['completed_total']:<8} "
                  f"{stats['failed_total']:<8} {stats['success_rate']*100:.1f}%{'':<5} {health_indicator}")
    else:
        print("  No workers registered yet.")

    # Latency histogram
    print("\n[LATENCY DISTRIBUTION]")
    if metrics.task_duration_histogram:
        total = sum(metrics.task_duration_histogram.values())
        for bucket in ["<10s", "10-30s", "30-60s", "1-5m", "5-15m", "15-30m", "30-60m", ">60m"]:
            count = metrics.task_duration_histogram.get(bucket, 0)
            pct = (count / total * 100) if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"  {bucket:>8}: {bar:<50} {count} ({pct:.1f}%)")
    else:
        print("  No latency data yet.")

    print("\n" + "=" * 70)
    print(f"Last updated: {metrics.last_updated}")
    print("=" * 70 + "\n")


# ============================================================================
# ADVANCED FEATURES (adv-a-001 to adv-a-010)
# ============================================================================

def leader_cancel_task(task_id: str, reason: str = "") -> bool:
    """
    Cancel a task and clean up any partial work.

    Implements feature adv-a-002: Task cancellation support with cleanup.
    Adds 'cancelled' status to task lifecycle and cleans up partial results.

    Args:
        task_id: ID of the task to cancel
        reason: Optional reason for cancellation

    Returns:
        True if task was cancelled, False otherwise
    """
    data = load_tasks()

    for i, t in enumerate(data["tasks"]):
        if t["id"] == task_id:
            old_status = t["status"]
            if old_status in ("done", "cancelled"):
                print(f"X Cannot cancel task {task_id} - status is {old_status}")
                return False

            data["tasks"][i]["status"] = "cancelled"
            data["tasks"][i]["completed_at"] = now_iso()
            data["tasks"][i]["result"] = {"cancelled": True, "reason": reason, "previous_status": old_status}

            save_tasks(data)

            # Clean up any partial result files
            result_file = RESULTS_DIR / f"{task_id}.md"
            if result_file.exists():
                # Rename to indicate cancellation
                cancelled_file = RESULTS_DIR / f"{task_id}.cancelled.md"
                result_file.rename(cancelled_file)

            log_action("leader", "CANCELLED", f"{task_id}: {reason}")
            print(f"+ Cancelled task {task_id}")
            if reason:
                print(f"  Reason: {reason}")
            return True

    print(f"X Task {task_id} not found")
    return False


def update_priority_boosts() -> int:
    """
    Update priority boosts for tasks waiting longer than threshold.

    Implements feature adv-a-001: Task priority auto-adjustment based on wait time.
    Tasks waiting more than PRIORITY_BOOST_THRESHOLD_MINUTES get their
    priority boosted by PRIORITY_BOOST_AMOUNT for each additional threshold period.

    Returns:
        Number of tasks whose priority was boosted
    """
    data = load_tasks()
    boosted_count = 0
    now = datetime.now()

    for i, t in enumerate(data["tasks"]):
        if t["status"] == "available" and t.get("created_at"):
            try:
                created = datetime.fromisoformat(t["created_at"])
                wait_minutes = (now - created).total_seconds() / 60

                if wait_minutes > PRIORITY_BOOST_THRESHOLD_MINUTES:
                    # Calculate boost: 1 for each threshold period waited
                    new_boost = int(wait_minutes // PRIORITY_BOOST_THRESHOLD_MINUTES)
                    old_boost = t.get("priority_boost", 0)

                    if new_boost > old_boost:
                        data["tasks"][i]["priority_boost"] = new_boost
                        boosted_count += 1
                        log_action("system", "PRIORITY_BOOST", f"{t['id']}: boost {old_boost} -> {new_boost}")
            except (ValueError, TypeError):
                continue

    if boosted_count > 0:
        save_tasks(data)
        print(f"+ Boosted priority for {boosted_count} tasks")

    return boosted_count


def detect_stale_workers(timeout_minutes: int = None) -> List[str]:
    """
    Detect workers that have gone stale (no activity for timeout period).

    Part of feature adv-a-003: Task reassignment when worker goes stale.

    Args:
        timeout_minutes: Override default timeout in minutes

    Returns:
        List of stale worker IDs
    """
    timeout = timeout_minutes or STALE_WORKER_TIMEOUT_MINUTES
    data = load_agents()
    now = datetime.now()
    stale_workers = []

    for agent in data.get("agents", []):
        if agent.get("last_seen"):
            try:
                last_seen = datetime.fromisoformat(agent["last_seen"])
                idle_minutes = (now - last_seen).total_seconds() / 60

                if idle_minutes > timeout:
                    stale_workers.append(agent["id"])
            except (ValueError, TypeError):
                continue

    return stale_workers


def reassign_stale_worker_tasks(timeout_minutes: int = None) -> int:
    """
    Reassign tasks from stale workers back to available status.

    Implements feature adv-a-003: Task reassignment when worker goes stale.
    Detects stale workers (no activity > timeout), auto-reassigns their tasks
    to available status, and logs reassignment events.

    Args:
        timeout_minutes: Override default timeout for stale detection

    Returns:
        Number of tasks reassigned
    """
    stale_workers = detect_stale_workers(timeout_minutes)
    if not stale_workers:
        print("No stale workers detected.")
        return 0

    data = load_tasks()
    reassigned_count = 0

    for i, t in enumerate(data["tasks"]):
        if t.get("claimed_by") in stale_workers and t["status"] in ("claimed", "in_progress"):
            old_worker = t["claimed_by"]
            data["tasks"][i]["status"] = "available"
            data["tasks"][i]["claimed_by"] = None
            data["tasks"][i]["claimed_at"] = None
            if data["tasks"][i].get("result") is None:
                data["tasks"][i]["result"] = {}
            data["tasks"][i]["result"]["reassigned_from"] = old_worker
            data["tasks"][i]["result"]["reassigned_at"] = now_iso()
            reassigned_count += 1
            log_action("system", "REASSIGNED", f"{t['id']}: from stale worker {old_worker}")

    if reassigned_count > 0:
        save_tasks(data)
        print(f"+ Reassigned {reassigned_count} tasks from stale workers: {', '.join(stale_workers)}")

    return reassigned_count


def leader_batch_add_tasks(file_path: str) -> List[str]:
    """
    Add multiple tasks from a YAML or JSON file.

    Implements feature adv-a-004: Batch task creation from YAML/JSON files.
    Parses YAML/JSON task definitions and validates against task schema.

    File format (YAML):
        tasks:
          - description: "Task 1"
            priority: 1
            tags: [backend, api]
          - description: "Task 2"
            priority: 2
            dependencies: [task-123]

    Args:
        file_path: Path to the YAML or JSON file

    Returns:
        List of created task IDs
    """
    path = Path(file_path)
    if not path.exists():
        print(f"X File not found: {file_path}")
        return []

    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        if not YAML_AVAILABLE:
            print("X PyYAML not installed. Install with: pip install pyyaml")
            return []
        batch_data = yaml.safe_load(content)
    else:
        batch_data = json.loads(content)

    if not batch_data or "tasks" not in batch_data:
        print("X Invalid file format: missing 'tasks' key")
        return []

    created_ids = []
    for task_def in batch_data["tasks"]:
        task_id = leader_add_task(
            description=task_def.get("description", "No description"),
            priority=task_def.get("priority", 5),
            dependencies=task_def.get("dependencies"),
            context_files=task_def.get("context_files", task_def.get("files")),
            hints=task_def.get("hints", ""),
            tags=task_def.get("tags"),
            parent_id=task_def.get("parent_id"),
            estimated_duration=task_def.get("estimated_duration"),
            required_capabilities=task_def.get("required_capabilities", task_def.get("capabilities")),
            max_retries=task_def.get("max_retries"),
            timeout_minutes=task_def.get("timeout_minutes", task_def.get("timeout")),
            group_id=task_def.get("group_id"),
        )
        created_ids.append(task_id)

    print(f"+ Created {len(created_ids)} tasks from {file_path}")
    return created_ids


def leader_list_tasks_by_tag(tag: str) -> List[Task]:
    """
    List all tasks with a specific tag.

    Implements feature adv-a-005: Task tagging and filtering system.

    Args:
        tag: The tag to filter by

    Returns:
        List of matching tasks
    """
    data = load_tasks()
    tasks = [Task.from_dict(t) for t in data["tasks"]]

    matching = [t for t in tasks if t.tags and tag in t.tags]

    if not matching:
        print(f"No tasks with tag '{tag}'")
        return []

    print(f"\nTasks with tag '{tag}' ({len(matching)}):")
    for t in sorted(matching, key=lambda x: (x.status, x.get_effective_priority())):
        print(f"  [{t.status}] [{t.get_effective_priority()}] {t.id}: {t.description[:40]}...")

    return matching


def worker_create_subtask(
    parent_id: str,
    terminal_id: str,
    description: str,
    priority: Optional[int] = None,
    tags: Optional[List[str]] = None,
    estimated_duration: Optional[int] = None,
) -> Optional[str]:
    """
    Create a subtask from a worker during task execution.

    Implements feature adv-a-006: Subtask support with parent-child relationships.
    The subtask inherits context from the parent and the parent
    will not be considered complete until all subtasks are done.

    Args:
        parent_id: ID of the parent task
        terminal_id: ID of the creating worker
        description: Subtask description
        priority: Override priority (defaults to parent's priority)
        tags: Additional tags (merged with parent's tags)
        estimated_duration: Estimated duration in minutes

    Returns:
        The created subtask ID, or None if parent not found
    """
    data = load_tasks()

    # Find parent task
    parent = None
    for t in data["tasks"]:
        if t["id"] == parent_id:
            parent = t
            break

    if not parent:
        print(f"X Parent task {parent_id} not found")
        return None

    # Inherit from parent
    inherited_priority = priority if priority is not None else parent.get("priority", 5)
    inherited_tags = list(set((parent.get("tags") or []) + (tags or [])))
    inherited_context = parent.get("context", {}).copy() if parent.get("context") else {}
    inherited_context["parent_id"] = parent_id
    inherited_context["created_by_worker"] = terminal_id

    task_id = leader_add_task(
        description=description,
        priority=inherited_priority,
        context_files=inherited_context.get("files"),
        hints=inherited_context.get("hints", ""),
        tags=inherited_tags if inherited_tags else None,
        parent_id=parent_id,
        estimated_duration=estimated_duration,
        required_capabilities=parent.get("required_capabilities"),
    )

    log_action(terminal_id, "CREATED_SUBTASK", f"{task_id} (parent: {parent_id})")
    print(f"+ Created subtask {task_id} under parent {parent_id}")
    return task_id


def get_subtasks(parent_id: str) -> List[Task]:
    """
    Get all subtasks of a parent task.

    Args:
        parent_id: ID of the parent task

    Returns:
        List of subtasks
    """
    data = load_tasks()
    tasks = [Task.from_dict(t) for t in data["tasks"]]
    return [t for t in tasks if t.parent_id == parent_id]


def are_subtasks_complete(parent_id: str) -> bool:
    """
    Check if all subtasks of a parent are complete.

    Args:
        parent_id: ID of the parent task

    Returns:
        True if all subtasks are done, False otherwise
    """
    subtasks = get_subtasks(parent_id)
    if not subtasks:
        return True
    return all(t.status == "done" for t in subtasks)


def calculate_actual_duration(claimed_at: str, completed_at: str) -> Optional[int]:
    """
    Calculate actual duration between claimed and completed timestamps.

    Implements feature adv-a-007: Task estimation and time tracking.

    Args:
        claimed_at: ISO timestamp when task was claimed
        completed_at: ISO timestamp when task was completed

    Returns:
        Duration in minutes, or None if calculation fails
    """
    try:
        start = datetime.fromisoformat(claimed_at)
        end = datetime.fromisoformat(completed_at)
        return int((end - start).total_seconds() / 60)
    except (ValueError, TypeError):
        return None


def get_worker_capabilities(worker_id: str) -> List[str]:
    """
    Get the capabilities of a registered worker.

    Implements feature adv-a-008: Worker capability matching.

    Args:
        worker_id: ID of the worker

    Returns:
        List of capability strings
    """
    data = load_agents()
    for agent in data.get("agents", []):
        if agent["id"] == worker_id:
            caps = agent.get("capabilities", "")
            if isinstance(caps, str):
                return [c.strip() for c in caps.split(",") if c.strip()]
            return caps or []
    return []


def task_matches_capabilities(task: Task, worker_id: str) -> bool:
    """
    Check if a worker has the required capabilities for a task.

    Implements feature adv-a-008: Worker capability matching.
    Workers declare capabilities on register, tasks can require specific
    capabilities, and claim only matches capable workers.

    Args:
        task: The task to check
        worker_id: ID of the worker

    Returns:
        True if worker has all required capabilities (or task has no requirements)
    """
    required = task.required_capabilities or []
    if not required:
        return True

    worker_caps = set(get_worker_capabilities(worker_id))
    return all(req in worker_caps for req in required)


def schedule_retry(task_dict: dict, retry_count: int) -> dict:
    """
    Schedule a retry for a failed task using exponential backoff.

    Implements feature adv-a-009: Task retry with exponential backoff.

    Args:
        task_dict: The task dictionary to modify
        retry_count: Current retry attempt number

    Returns:
        Modified task dictionary with retry scheduled
    """
    # Exponential backoff: delay = initial_delay * 2^(retry_count - 1)
    delay_seconds = INITIAL_RETRY_DELAY_SECONDS * (2 ** (retry_count - 1))
    next_retry = datetime.now() + timedelta(seconds=delay_seconds)

    task_dict["retry_count"] = retry_count
    task_dict["retry_delay"] = delay_seconds
    task_dict["next_retry_at"] = next_retry.isoformat()
    task_dict["status"] = "available"  # Make available for retry
    task_dict["claimed_by"] = None
    task_dict["claimed_at"] = None

    return task_dict


def process_retries() -> int:
    """
    Process tasks that are due for retry.

    Implements feature adv-a-009: Task retry with exponential backoff.
    Checks tasks with retry scheduled and makes them available
    if the retry time has passed.

    Returns:
        Number of tasks made available for retry
    """
    data = load_tasks()
    now = datetime.now()
    processed = 0

    for i, t in enumerate(data["tasks"]):
        if t.get("next_retry_at") and t["status"] == "available":
            try:
                retry_time = datetime.fromisoformat(t["next_retry_at"])
                if now >= retry_time:
                    # Clear retry scheduling - task is now truly available
                    data["tasks"][i]["next_retry_at"] = None
                    processed += 1
                    log_action("system", "RETRY_AVAILABLE", f"{t['id']}: retry {t.get('retry_count', 0)}")
            except (ValueError, TypeError):
                continue

    if processed > 0:
        save_tasks(data)
        print(f"+ {processed} tasks now available for retry")

    return processed


def retry_failed_task(task_id: str) -> bool:
    """
    Retry a failed task with exponential backoff.

    Implements feature adv-a-009: Task retry with exponential backoff.
    Configures max_retries per task, implements exponential backoff delay,
    and tracks retry_count in task.

    Args:
        task_id: ID of the failed task to retry

    Returns:
        True if retry was scheduled, False otherwise
    """
    data = load_tasks()

    for i, t in enumerate(data["tasks"]):
        if t["id"] == task_id:
            if t["status"] != "failed":
                print(f"X Task {task_id} is not failed (status: {t['status']})")
                return False

            max_retries = t.get("max_retries", DEFAULT_MAX_RETRIES)
            current_retry = t.get("retry_count", 0) + 1

            if current_retry > max_retries:
                print(f"X Task {task_id} has exceeded max retries ({max_retries})")
                return False

            data["tasks"][i] = schedule_retry(data["tasks"][i], current_retry)
            save_tasks(data)

            delay = data["tasks"][i]["retry_delay"]
            print(f"+ Scheduled retry {current_retry}/{max_retries} for task {task_id}")
            print(f"  Next retry in {delay} seconds")
            log_action("system", "RETRY_SCHEDULED", f"{task_id}: attempt {current_retry}/{max_retries}")
            return True

    print(f"X Task {task_id} not found")
    return False


# ============================================================================
# TASK GROUPS (adv-a-010: Parallel Execution with Barrier Sync)
# ============================================================================

def create_task_group(
    name: str,
    task_ids: List[str],
    barrier_task_id: Optional[str] = None,
) -> str:
    """
    Create a task group for parallel execution with barrier sync.

    Implements feature adv-a-010: Parallel task groups with barrier sync.
    All tasks in the group can run in parallel. If a barrier task is specified,
    it will only become available after all group tasks complete.

    Args:
        name: Human-readable group name
        task_ids: List of task IDs to include in the group
        barrier_task_id: Optional task ID that waits for all group tasks

    Returns:
        The generated group ID
    """
    ensure_coordination_structure()

    group_id = f"group-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(name.encode()).hexdigest()[:4]}"

    data = load_groups()
    group = {
        "id": group_id,
        "name": name,
        "task_ids": task_ids,
        "barrier_task_id": barrier_task_id,
        "created_at": now_iso(),
        "status": "active",
    }
    data["groups"].append(group)
    save_groups(data)

    # Update tasks with group_id
    tasks_data = load_tasks()
    for i, t in enumerate(tasks_data["tasks"]):
        if t["id"] in task_ids:
            tasks_data["tasks"][i]["group_id"] = group_id

    # If there's a barrier task, mark it as blocked
    if barrier_task_id:
        for i, t in enumerate(tasks_data["tasks"]):
            if t["id"] == barrier_task_id:
                tasks_data["tasks"][i]["status"] = "blocked"
                tasks_data["tasks"][i]["group_id"] = group_id
                break

    save_tasks(tasks_data)

    log_action("leader", "CREATE_GROUP", f"{group_id}: {name} with {len(task_ids)} tasks")
    print(f"+ Created task group {group_id}: {name}")
    print(f"  Tasks: {', '.join(task_ids)}")
    if barrier_task_id:
        print(f"  Barrier task: {barrier_task_id}")

    return group_id


def check_group_complete(group_id: str) -> bool:
    """
    Check if all tasks in a group are complete.

    Args:
        group_id: ID of the group to check

    Returns:
        True if all group tasks are done
    """
    groups_data = load_groups()
    tasks_data = load_tasks()

    group = None
    for g in groups_data.get("groups", []):
        if g["id"] == group_id:
            group = g
            break

    if not group:
        return False

    tasks = {t["id"]: t for t in tasks_data["tasks"]}
    for task_id in group["task_ids"]:
        task = tasks.get(task_id)
        if not task or task["status"] != "done":
            return False

    return True


def update_barrier_tasks():
    """
    Check all groups and make barrier tasks available if their groups are complete.

    Implements feature adv-a-010: Parallel task groups with barrier sync.
    Barrier waits for all group tasks, then continues after barrier completes.
    """
    groups_data = load_groups()
    tasks_data = load_tasks()
    updated = 0

    for group in groups_data.get("groups", []):
        if group.get("status") != "active":
            continue

        barrier_id = group.get("barrier_task_id")
        if not barrier_id:
            continue

        if check_group_complete(group["id"]):
            # Find and update barrier task
            for i, t in enumerate(tasks_data["tasks"]):
                if t["id"] == barrier_id and t["status"] == "blocked":
                    tasks_data["tasks"][i]["status"] = "available"
                    updated += 1
                    log_action("system", "BARRIER_RELEASED", f"{barrier_id}: group {group['id']} complete")

            # Mark group as complete
            for i, g in enumerate(groups_data["groups"]):
                if g["id"] == group["id"]:
                    groups_data["groups"][i]["status"] = "complete"
                    groups_data["groups"][i]["completed_at"] = now_iso()
                    break

    if updated > 0:
        save_tasks(tasks_data)
        save_groups(groups_data)
        print(f"+ Released {updated} barrier tasks")


def leader_group_status(group_id: Optional[str] = None):
    """
    Show status of task groups.

    Args:
        group_id: Optional specific group ID to show
    """
    groups_data = load_groups()
    tasks_data = load_tasks()
    tasks = {t["id"]: Task.from_dict(t) for t in tasks_data["tasks"]}

    groups = groups_data.get("groups", [])
    if group_id:
        groups = [g for g in groups if g["id"] == group_id]

    if not groups:
        print("No task groups found.")
        return

    for group in groups:
        print(f"\nGroup: {group['name']} ({group['id']})")
        print(f"  Status: {group['status']}")

        done_count = 0
        total = len(group["task_ids"])
        for tid in group["task_ids"]:
            task = tasks.get(tid)
            if task:
                status_char = "[+]" if task.status == "done" else "[ ]"
                print(f"    {status_char} {tid}: {task.description[:30]}... [{task.status}]")
                if task.status == "done":
                    done_count += 1

        print(f"  Progress: {done_count}/{total}")
        if group.get("barrier_task_id"):
            barrier = tasks.get(group["barrier_task_id"])
            if barrier:
                print(f"  Barrier: {barrier.id} [{barrier.status}]")


# ============================================================================
# ESTIMATION AND TIME TRACKING (adv-a-007)
# ============================================================================

def leader_estimation_report():
    """
    Generate a report on estimation accuracy across completed tasks.

    Implements feature adv-a-007: Task estimation and time tracking.
    Adds estimated_duration field, tracks actual_duration on completion,
    and reports estimation accuracy.
    """
    data = load_tasks()
    tasks = [Task.from_dict(t) for t in data["tasks"] if t.get("status") == "done"]

    tasks_with_estimates = [t for t in tasks if t.estimated_duration and t.actual_duration]

    if not tasks_with_estimates:
        print("No completed tasks with both estimated and actual durations.")
        return

    print("\n" + "=" * 60)
    print("ESTIMATION ACCURACY REPORT")
    print("=" * 60)

    total_estimated = 0
    total_actual = 0
    accuracies = []

    for t in tasks_with_estimates:
        accuracy = t.actual_duration / t.estimated_duration if t.estimated_duration else 0
        accuracies.append(accuracy)
        total_estimated += t.estimated_duration
        total_actual += t.actual_duration

        over_under = "on time" if 0.9 <= accuracy <= 1.1 else ("over" if accuracy > 1 else "under")
        print(f"  {t.id}: est {t.estimated_duration}m, actual {t.actual_duration}m ({over_under})")

    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"\nOverall:")
    print(f"  Total Estimated: {total_estimated} minutes")
    print(f"  Total Actual: {total_actual} minutes")
    print(f"  Average Accuracy: {avg_accuracy:.2%}")
    print(f"  Tasks Analyzed: {len(tasks_with_estimates)}")
    print("=" * 60 + "\n")


# ============================================================================
# MAINTENANCE COMMANDS
# ============================================================================

def run_maintenance():
    """
    Run all maintenance tasks: priority boosts, stale worker detection, retries, barriers.

    This should be run periodically (e.g., via cron or a watch loop) to keep
    the coordination system healthy.
    """
    print("\n" + "=" * 60)
    print("RUNNING MAINTENANCE")
    print("=" * 60)

    print("\n1. Checking priority boosts...")
    boosted = update_priority_boosts()

    print("\n2. Checking for stale workers...")
    reassigned = reassign_stale_worker_tasks()

    print("\n3. Processing retry queue...")
    retried = process_retries()

    print("\n4. Updating barrier tasks...")
    update_barrier_tasks()

    print("\n5. Cleaning up orphan tasks...")
    orphans = cleanup_orphan_tasks()

    print("\n" + "-" * 60)
    print(f"Maintenance complete: {boosted} boosted, {reassigned} reassigned, {retried} retried, {orphans} orphans cleaned")
    print("=" * 60 + "\n")


# ============================================================================
# WORKER COMMANDS
# ============================================================================

def worker_claim(terminal_id: str) -> Optional[Task]:
    """
    Claim an available task atomically.

    Implements feature adv-a-008: Worker capability matching.
    Only claims tasks that match the worker's declared capabilities.
    Also uses get_effective_priority() which includes priority boosts (adv-a-001).

    RACE CONDITION FIX: The entire claim transaction (read, check, modify, write)
    is now wrapped in a single file lock to prevent TOCTOU vulnerabilities where
    two workers could both read the task as "available" and both attempt to claim it.
    """
    # Create backup before modifying state (adv-a-011: Automatic state backup)
    if TASKS_FILE.exists():
        create_backup()

    # RACE CONDITION FIX: Hold the lock for the entire transaction
    # This prevents two workers from both reading the task as "available"
    # and both attempting to claim it simultaneously.
    with file_lock(TASKS_FILE, exclusive=True):
        data = load_tasks_no_lock()
        tasks = [Task.from_dict(t) for t in data["tasks"]]
        done_ids = {t.id for t in tasks if t.status == "done"}

        # Find available task with satisfied dependencies and matching capabilities
        available = [
            t for t in tasks
            if t.status == "available"
            and all(dep in done_ids for dep in (t.dependencies or []))
            and task_matches_capabilities(t, terminal_id)  # adv-a-008: capability matching
        ]

        if not available:
            print("No available tasks with satisfied dependencies and matching capabilities.")
            return None

        # Pick highest priority (lowest number), using effective priority with boosts (adv-a-001)
        task = min(available, key=lambda t: t.get_effective_priority())

        # Claim it - no race possible since we hold the exclusive lock
        for i, t in enumerate(data["tasks"]):
            if t["id"] == task.id:
                data["tasks"][i]["status"] = "claimed"
                data["tasks"][i]["claimed_by"] = terminal_id
                data["tasks"][i]["claimed_at"] = now_iso()
                break

        save_tasks_no_lock(data)

        # Get the claimed task for return
        for t in data["tasks"]:
            if t["id"] == task.id:
                log_action(terminal_id, "CLAIMED", task.id)
                print(f"Claimed task {task.id}")
                print(f"  Description: {task.description}")
                print(f"  Priority: {task.priority}")
                if task.context:
                    print(f"  Context: {json.dumps(task.context, indent=4)}")
                return Task.from_dict(t)

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
    """
    Mark a task as complete and write results.

    Implements feature adv-a-007: Task estimation and time tracking.
    Calculates actual_duration based on claimed_at timestamp.
    """
    data = load_tasks()
    task = None

    for i, t in enumerate(data["tasks"]):
        if t["id"] == task_id:
            if t.get("claimed_by") != terminal_id:
                print(f"X Task {task_id} not claimed by {terminal_id}")
                return

            completed_at = now_iso()
            data["tasks"][i]["status"] = "done"
            data["tasks"][i]["completed_at"] = completed_at
            data["tasks"][i]["result"] = {
                "output": output,
                "files_modified": files_modified or [],
                "files_created": files_created or []
            }

            # adv-a-007: Calculate actual duration
            if t.get("claimed_at"):
                actual_duration = calculate_actual_duration(t["claimed_at"], completed_at)
                if actual_duration is not None:
                    data["tasks"][i]["actual_duration"] = actual_duration

            task = t
            break

    if not task:
        print(f"✗ Task {task_id} not found")
        return

    # Record metrics before saving (adv-a-016, adv-a-017)
    if task.get("claimed_at"):
        try:
            start_time = datetime.fromisoformat(task["claimed_at"])
            duration_seconds = (datetime.now() - start_time).total_seconds()
            record_task_completion(terminal_id, task_id, duration_seconds)
        except (ValueError, TypeError):
            pass  # Skip metrics if timestamp parsing fails

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

    # Record metrics before saving (adv-a-019)
    record_task_failure(terminal_id, task_id, error_category="task_execution_error")

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

    # Advanced leader commands (adv-a-001 to adv-a-010)
    cancel_cmd = leader_sub.add_parser("cancel", help="Cancel a task (adv-a-002)")
    cancel_cmd.add_argument("task_id", help="Task ID to cancel")
    cancel_cmd.add_argument("--reason", "-r", default="", help="Reason for cancellation")

    batch_cmd = leader_sub.add_parser("batch-add", help="Add tasks from YAML/JSON file (adv-a-004)")
    batch_cmd.add_argument("file", help="Path to YAML or JSON file with task definitions")

    tag_cmd = leader_sub.add_parser("tag-filter", help="List tasks by tag (adv-a-005)")
    tag_cmd.add_argument("tag", help="Tag to filter by")

    retry_cmd = leader_sub.add_parser("retry", help="Retry a failed task (adv-a-009)")
    retry_cmd.add_argument("task_id", help="Failed task ID to retry")

    group_cmd = leader_sub.add_parser("create-group", help="Create parallel task group with barrier (adv-a-010)")
    group_cmd.add_argument("name", help="Group name")
    group_cmd.add_argument("--tasks", "-t", nargs="+", required=True, help="Task IDs in the group")
    group_cmd.add_argument("--barrier", "-b", help="Barrier task ID (waits for group completion)")

    leader_sub.add_parser("group-status", help="Show task group status (adv-a-010)")
    leader_sub.add_parser("estimation-report", help="Show estimation accuracy report (adv-a-007)")
    leader_sub.add_parser("maintenance", help="Run maintenance tasks (priority boost, stale cleanup, etc.)")
    leader_sub.add_parser("boost-priorities", help="Boost priority of waiting tasks (adv-a-001)")
    leader_sub.add_parser("reassign-stale", help="Reassign tasks from stale workers (adv-a-003)")

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

    # Worker subtask command (adv-a-006)
    subtask_cmd = worker_sub.add_parser("subtask", help="Create a subtask (adv-a-006)")
    subtask_cmd.add_argument("terminal_id", help="Your terminal ID")
    subtask_cmd.add_argument("parent_id", help="Parent task ID")
    subtask_cmd.add_argument("description", help="Subtask description")
    subtask_cmd.add_argument("--priority", "-p", type=int, help="Priority override")
    subtask_cmd.add_argument("--tags", "-t", nargs="*", help="Additional tags")
    subtask_cmd.add_argument("--estimate", "-e", type=int, help="Estimated duration in minutes")

    # Worker heartbeat command
    heartbeat_cmd = worker_sub.add_parser("heartbeat", help="Send worker heartbeat")
    heartbeat_cmd.add_argument("terminal_id", help="Your terminal ID")
    heartbeat_cmd.add_argument("--task", help="Currently working task ID")

    # Recovery commands
    recovery_parser = subparsers.add_parser("recovery", help="Recovery commands")
    recovery_sub = recovery_parser.add_subparsers(dest="command")

    recovery_sub.add_parser("backup", help="Create a backup of current state")
    restore_cmd = recovery_sub.add_parser("restore", help="Restore from backup")
    restore_cmd.add_argument("--file", help="Specific backup file (default: latest)")

    recovery_sub.add_parser("check", help="Check for corruption and auto-recover")

    checkpoint_cmd = recovery_sub.add_parser("checkpoint", help="Save a checkpoint")
    checkpoint_cmd.add_argument("--name", help="Checkpoint name")

    restore_checkpoint_cmd = recovery_sub.add_parser("restore-checkpoint", help="Restore from checkpoint")
    restore_checkpoint_cmd.add_argument("name", help="Checkpoint name to restore")

    recovery_sub.add_parser("list-checkpoints", help="List available checkpoints")
    recovery_sub.add_parser("cleanup-orphans", help="Clean up orphaned tasks")
    recovery_sub.add_parser("check-health", help="Check worker health status")

    compact_log_cmd = recovery_sub.add_parser("compact-log", help="Compact transaction log")
    compact_log_cmd.add_argument("--hours", type=int, default=24, help="Keep entries from last N hours")

    # Metrics commands
    metrics_parser = subparsers.add_parser("metrics", help="Metrics commands")
    metrics_sub = metrics_parser.add_subparsers(dest="command")

    metrics_sub.add_parser("show", help="Show metrics in terminal")
    metrics_sub.add_parser("prometheus", help="Print Prometheus metrics format")

    server_cmd = metrics_sub.add_parser("server", help="Start Prometheus metrics HTTP server")
    server_cmd.add_argument("--port", type=int, default=METRICS_PORT, help=f"Port (default: {METRICS_PORT})")

    metrics_sub.add_parser("dashboard", help="Show metrics dashboard")
    metrics_sub.add_parser("workers", help="Show worker utilization")
    metrics_sub.add_parser("queue", help="Show queue metrics")
    metrics_sub.add_parser("errors", help="Show error metrics")

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
        # Advanced leader commands (adv-a-001 to adv-a-010)
        elif args.command == "cancel":
            leader_cancel_task(args.task_id, args.reason)
        elif args.command == "batch-add":
            leader_batch_add_tasks(args.file)
        elif args.command == "tag-filter":
            leader_list_tasks_by_tag(args.tag)
        elif args.command == "retry":
            retry_failed_task(args.task_id)
        elif args.command == "create-group":
            create_task_group(args.name, args.tasks, args.barrier)
        elif args.command == "group-status":
            leader_group_status()
        elif args.command == "estimation-report":
            leader_estimation_report()
        elif args.command == "maintenance":
            run_maintenance()
        elif args.command == "boost-priorities":
            update_priority_boosts()
        elif args.command == "reassign-stale":
            reassign_stale_worker_tasks()
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
        elif args.command == "subtask":
            worker_create_subtask(
                args.parent_id,
                args.terminal_id,
                args.description,
                args.priority,
                args.tags,
                args.estimate
            )
        elif args.command == "heartbeat":
            worker_heartbeat(args.terminal_id, args.task)
            print(f"+ Heartbeat sent for {args.terminal_id}")
        else:
            worker_parser.print_help()

    elif args.role == "recovery":
        if args.command == "backup":
            backup_file = create_backup()
            if backup_file:
                print(f"✓ Backup created: {backup_file}")
        elif args.command == "restore":
            if args.file:
                restore_from_backup(Path(args.file))
            else:
                restore_from_backup()
        elif args.command == "check":
            if auto_recover():
                print("✓ Recovery completed successfully")
            else:
                print("✓ No recovery needed - state is healthy")
        elif args.command == "checkpoint":
            save_checkpoint(args.name)
        elif args.command == "restore-checkpoint":
            restore_checkpoint(args.name)
        elif args.command == "list-checkpoints":
            checkpoints = list_checkpoints()
            if checkpoints:
                print("\nAvailable checkpoints:")
                for cp in checkpoints:
                    print(f"  - {cp['name']} (created: {cp['created_at']})")
            else:
                print("No checkpoints found.")
        elif args.command == "cleanup-orphans":
            count = cleanup_orphan_tasks()
            if count == 0:
                print("No orphaned tasks found.")
        elif args.command == "check-health":
            health = check_worker_health()
            if health:
                print("\nWorker Health Status:")
                for worker, status in sorted(health.items()):
                    indicator = {"healthy": "[OK]", "warning": "[!]", "dead": "[X]"}[status]
                    print(f"  {indicator} {worker}: {status}")
            else:
                print("No workers registered yet.")
        elif args.command == "compact-log":
            compact_transaction_log(args.hours)
            print(f"✓ Transaction log compacted (kept last {args.hours} hours)")
        else:
            recovery_parser.print_help()

    elif args.role == "metrics":
        if args.command == "show" or args.command == "dashboard":
            print_metrics_dashboard()
        elif args.command == "prometheus":
            print(generate_prometheus_metrics())
        elif args.command == "server":
            start_metrics_server(args.port)
        elif args.command == "workers":
            utilization = get_worker_utilization()
            if utilization:
                print("\nWorker Utilization:")
                for worker_id, stats in sorted(utilization.items()):
                    print(f"\n  {worker_id}:")
                    print(f"    Active Tasks: {stats['active_tasks']}")
                    print(f"    Completed:    {stats['completed_total']}")
                    print(f"    Failed:       {stats['failed_total']}")
                    print(f"    Success Rate: {stats['success_rate']*100:.1f}%")
                    print(f"    Avg Duration: {stats['average_duration_seconds']:.1f}s")
            else:
                print("No worker metrics yet.")
        elif args.command == "queue":
            queue = calculate_queue_metrics()
            print("\nQueue Metrics:")
            print(f"  Queue Depth:    {queue['queue_depth']}")
            print(f"  Avg Wait Time:  {queue['average_wait_seconds']:.1f}s")
            print(f"  Max Wait Time:  {queue['max_wait_seconds']:.1f}s")
        elif args.command == "errors":
            calculate_error_rate()
            metrics = load_metrics()
            print("\nError Metrics:")
            print(f"  Total Errors:   {metrics.error_count_total}")
            print(f"  Error Rate:     {metrics.error_rate_per_minute:.4f} errors/min")
            if metrics.errors_by_category:
                print("  By Category:")
                for cat, count in sorted(metrics.errors_by_category.items(), key=lambda x: -x[1]):
                    print(f"    - {cat}: {count}")
        else:
            metrics_parser.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
