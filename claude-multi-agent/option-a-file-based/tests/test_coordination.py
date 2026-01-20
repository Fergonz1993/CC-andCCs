"""
Unit tests for Option A coordination.py

Run with: pytest tests/test_coordination.py -v

Coverage target: >80%
"""

import pytest
import json
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import coordination
from coordination import Task


class TestTask:
    """Tests for the Task dataclass."""

    def test_task_creation(self):
        """opt-a-task-001: Task can be created with required fields."""
        task = Task(
            id="task-001",
            description="Test task",
            status="available",
            priority=5
        )
        assert task.id == "task-001"
        assert task.description == "Test task"
        assert task.status == "available"
        assert task.priority == 5

    def test_task_to_dict(self):
        """opt-a-task-002: Task.to_dict() returns dict without None values."""
        task = Task(
            id="task-001",
            description="Test task",
            status="available",
            priority=5
        )
        d = task.to_dict()
        assert "id" in d
        assert "description" in d
        assert "claimed_by" not in d  # None values should be excluded

    def test_task_from_dict(self):
        """opt-a-task-003: Task.from_dict() creates Task from dict."""
        data = {
            "id": "task-001",
            "description": "Test task",
            "status": "available",
            "priority": 5,
            "claimed_by": "worker-1"
        }
        task = Task.from_dict(data)
        assert task.id == "task-001"
        assert task.claimed_by == "worker-1"

    def test_task_from_dict_defaults(self):
        """opt-a-task-004: Task.from_dict() applies defaults for missing fields."""
        data = {
            "id": "task-001",
            "description": "Test task",
            "status": "available"
        }
        task = Task.from_dict(data)
        assert task.priority == 5  # Default priority


class TestGenerateTaskId:
    """Tests for task ID generation."""

    def test_generate_task_id_format(self):
        """opt-a-id-001: generate_task_id() returns task- prefixed string."""
        task_id = coordination.generate_task_id()
        assert task_id.startswith("task-")

    def test_generate_task_id_unique(self):
        """opt-a-id-002: generate_task_id() generates unique IDs."""
        ids = [coordination.generate_task_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique


class TestNowIso:
    """Tests for ISO timestamp generation."""

    def test_now_iso_format(self):
        """opt-a-time-001: now_iso() returns valid ISO format string."""
        result = coordination.now_iso()
        # Should be parseable as datetime
        datetime.fromisoformat(result)


class TestFileLocking:
    """Tests for file locking mechanism."""

    def test_file_lock_creates_lock_file(self, temp_dir):
        """opt-a-lock-001: file_lock() creates .lock file."""
        test_file = temp_dir / "test.json"
        test_file.write_text("{}")

        with coordination.file_lock(test_file):
            lock_file = test_file.with_suffix(".json.lock")
            assert lock_file.exists()

    def test_file_lock_exclusive(self, temp_dir):
        """opt-a-lock-002: file_lock() acquires exclusive lock."""
        test_file = temp_dir / "test.json"
        test_file.write_text("{}")

        with coordination.file_lock(test_file, exclusive=True):
            # Lock should be held
            pass


class TestEnsureCoordinationStructure:
    """Tests for directory structure creation."""

    def test_ensure_creates_directories(self, mock_coordination_dir):
        """opt-a-struct-001: ensure_coordination_structure() creates required dirs."""
        # Remove structure to test creation
        import shutil
        shutil.rmtree(mock_coordination_dir)

        coordination.ensure_coordination_structure()

        assert coordination.COORDINATION_DIR.exists()
        assert (coordination.COORDINATION_DIR / "context").exists()
        assert coordination.LOGS_DIR.exists()
        assert coordination.RESULTS_DIR.exists()

    def test_ensure_creates_tasks_file(self, mock_coordination_dir):
        """opt-a-struct-002: ensure_coordination_structure() creates tasks.json."""
        import shutil
        shutil.rmtree(mock_coordination_dir)

        coordination.ensure_coordination_structure()

        assert coordination.TASKS_FILE.exists()
        data = json.loads(coordination.TASKS_FILE.read_text())
        assert "version" in data
        assert "tasks" in data


class TestLoadSaveTasks:
    """Tests for task loading and saving."""

    def test_load_tasks_empty(self, mock_coordination_dir):
        """opt-a-io-001: load_tasks() returns default structure when file missing."""
        result = coordination.load_tasks()
        assert "version" in result
        assert "tasks" in result
        assert isinstance(result["tasks"], list)

    def test_load_tasks_existing(self, tasks_file_with_data, sample_tasks_data):
        """opt-a-io-002: load_tasks() loads existing tasks."""
        result = coordination.load_tasks()
        assert len(result["tasks"]) == len(sample_tasks_data["tasks"])

    def test_save_tasks_updates_timestamp(self, mock_coordination_dir):
        """opt-a-io-003: save_tasks() updates last_updated timestamp."""
        data = {"version": "1.0", "tasks": []}
        coordination.save_tasks(data)

        loaded = coordination.load_tasks()
        assert "last_updated" in loaded


class TestLeaderCommands:
    """Tests for leader CLI commands."""

    def test_leader_init(self, mock_coordination_dir, capsys):
        """opt-a-leader-001: leader_init() creates master plan."""
        coordination.leader_init("Build a REST API", "Use Python Flask")

        assert coordination.MASTER_PLAN_FILE.exists()
        content = coordination.MASTER_PLAN_FILE.read_text()
        assert "Build a REST API" in content
        assert "Use Python Flask" in content

        captured = capsys.readouterr()
        assert "Initialized coordination" in captured.out

    def test_leader_add_task(self, mock_coordination_dir, capsys):
        """opt-a-leader-002: leader_add_task() adds task to queue."""
        coordination.ensure_coordination_structure()
        task_id = coordination.leader_add_task("Implement user model", priority=1)

        assert task_id.startswith("task-")

        data = coordination.load_tasks()
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["description"] == "Implement user model"
        assert data["tasks"][0]["priority"] == 1

    def test_leader_add_task_with_dependencies(self, mock_coordination_dir):
        """opt-a-leader-003: leader_add_task() supports dependencies."""
        coordination.ensure_coordination_structure()
        task1 = coordination.leader_add_task("Task 1")
        task2 = coordination.leader_add_task("Task 2", dependencies=[task1])

        data = coordination.load_tasks()
        task2_data = next(t for t in data["tasks"] if t["id"] == task2)
        assert task1 in task2_data["dependencies"]

    def test_leader_add_task_with_context(self, mock_coordination_dir):
        """opt-a-leader-004: leader_add_task() supports context files and hints."""
        coordination.ensure_coordination_structure()
        task_id = coordination.leader_add_task(
            "Implement model",
            context_files=["src/models.py"],
            hints="Use SQLAlchemy"
        )

        data = coordination.load_tasks()
        task_data = next(t for t in data["tasks"] if t["id"] == task_id)
        assert "src/models.py" in task_data["context"]["files"]
        assert "SQLAlchemy" in task_data["context"]["hints"]

    def test_leader_status(self, tasks_file_with_data, capsys):
        """opt-a-leader-005: leader_status() displays task summary."""
        coordination.leader_status()

        captured = capsys.readouterr()
        assert "COORDINATION STATUS" in captured.out
        assert "AVAILABLE" in captured.out

    def test_leader_aggregate(self, mock_coordination_dir, capsys):
        """opt-a-leader-006: leader_aggregate() aggregates results."""
        coordination.ensure_coordination_structure()

        # Create a result file
        result_file = coordination.RESULTS_DIR / "task-001.md"
        result_file.write_text("# Task 001 Results\nCompleted successfully.")

        coordination.leader_aggregate()

        summary_file = coordination.COORDINATION_DIR / "summary.md"
        assert summary_file.exists()
        assert "task-001" in summary_file.read_text()


class TestAgentRegistration:
    """Tests for agent registration."""

    def test_register_new_agent(self, mock_coordination_dir, capsys):
        """opt-a-agent-001: register_agent() registers new agent."""
        coordination.ensure_coordination_structure()
        coordination.register_agent("worker-1", "python,testing")

        data = coordination.load_agents()
        assert len(data["agents"]) == 1
        assert data["agents"][0]["id"] == "worker-1"

        captured = capsys.readouterr()
        assert "Registered agent worker-1" in captured.out

    def test_register_existing_agent_updates(self, agents_file_with_data, capsys):
        """opt-a-agent-002: register_agent() updates existing agent."""
        coordination.register_agent("worker-1", "python,testing,new-skill")

        data = coordination.load_agents()
        worker1 = next(a for a in data["agents"] if a["id"] == "worker-1")
        assert "new-skill" in worker1["capabilities"]

        captured = capsys.readouterr()
        assert "Updated agent worker-1" in captured.out


class TestWorkerCommands:
    """Tests for worker CLI commands."""

    def test_worker_claim_task(self, tasks_file_with_data, sample_tasks_data, capsys):
        """opt-a-worker-001: worker_claim() claims highest priority task."""
        task = coordination.worker_claim("terminal-2")

        assert task is not None
        assert task.id == "task-001"  # Highest priority available
        assert task.claimed_by == "terminal-2"

        captured = capsys.readouterr()
        assert "Claimed task" in captured.out

    def test_worker_claim_respects_dependencies(self, mock_coordination_dir, capsys):
        """opt-a-worker-002: worker_claim() respects dependencies."""
        coordination.ensure_coordination_structure()

        # Create tasks with dependency chain
        task1_id = coordination.leader_add_task("Task 1", priority=2)
        task2_id = coordination.leader_add_task("Task 2", priority=1, dependencies=[task1_id])

        # Should claim task1 first (task2 has unmet dependency)
        task = coordination.worker_claim("terminal-2")
        assert task.id == task1_id

    def test_worker_claim_no_available(self, mock_coordination_dir, capsys):
        """opt-a-worker-003: worker_claim() returns None when no tasks available."""
        coordination.ensure_coordination_structure()

        task = coordination.worker_claim("terminal-2")
        assert task is None

        captured = capsys.readouterr()
        assert "No available tasks" in captured.out

    def test_worker_start(self, tasks_file_with_data, capsys):
        """opt-a-worker-004: worker_start() marks task as in_progress."""
        # First claim a task
        coordination.worker_claim("terminal-2")

        # Then start it
        coordination.worker_start("terminal-2", "task-001")

        data = coordination.load_tasks()
        task_data = next(t for t in data["tasks"] if t["id"] == "task-001")
        assert task_data["status"] == "in_progress"

    def test_worker_start_wrong_owner(self, tasks_file_with_data, capsys):
        """opt-a-worker-005: worker_start() fails for non-owner."""
        # Claim with terminal-2
        coordination.worker_claim("terminal-2")

        # Try to start with terminal-3
        coordination.worker_start("terminal-3", "task-001")

        captured = capsys.readouterr()
        assert "Cannot start task" in captured.out

    def test_worker_complete(self, tasks_file_with_data, capsys):
        """opt-a-worker-006: worker_complete() marks task as done."""
        coordination.worker_claim("terminal-2")
        coordination.worker_complete(
            "terminal-2",
            "task-001",
            "Completed the implementation",
            files_modified=["src/models.py"],
            files_created=["tests/test_models.py"]
        )

        data = coordination.load_tasks()
        task_data = next(t for t in data["tasks"] if t["id"] == "task-001")
        assert task_data["status"] == "done"
        assert "Completed the implementation" in task_data["result"]["output"]

    def test_worker_complete_creates_result_file(self, tasks_file_with_data):
        """opt-a-worker-007: worker_complete() creates result file."""
        coordination.worker_claim("terminal-2")
        coordination.worker_complete("terminal-2", "task-001", "Done")

        result_file = coordination.RESULTS_DIR / "task-001.md"
        assert result_file.exists()
        content = result_file.read_text()
        assert "SUCCESS" in content

    def test_worker_fail(self, tasks_file_with_data, capsys):
        """opt-a-worker-008: worker_fail() marks task as failed."""
        coordination.worker_claim("terminal-2")
        coordination.worker_fail("terminal-2", "task-001", "Missing dependency")

        data = coordination.load_tasks()
        task_data = next(t for t in data["tasks"] if t["id"] == "task-001")
        assert task_data["status"] == "failed"
        assert "Missing dependency" in task_data["result"]["error"]

    def test_worker_fail_creates_result_file(self, tasks_file_with_data):
        """opt-a-worker-009: worker_fail() creates failure report."""
        coordination.worker_claim("terminal-2")
        coordination.worker_fail("terminal-2", "task-001", "Error occurred")

        result_file = coordination.RESULTS_DIR / "task-001.md"
        assert result_file.exists()
        content = result_file.read_text()
        assert "FAILED" in content

    def test_worker_list_available(self, tasks_file_with_data, capsys):
        """opt-a-worker-010: worker_list_available() shows available tasks."""
        coordination.worker_list_available()

        captured = capsys.readouterr()
        assert "Available tasks" in captured.out
        assert "task-001" in captured.out


class TestLogAction:
    """Tests for action logging."""

    def test_log_action_creates_file(self, mock_coordination_dir):
        """opt-a-log-001: log_action() creates log file."""
        coordination.ensure_coordination_structure()
        coordination.log_action("worker-1", "CLAIMED", "task-001")

        log_file = coordination.LOGS_DIR / "worker-1.log"
        assert log_file.exists()

    def test_log_action_appends(self, mock_coordination_dir):
        """opt-a-log-002: log_action() appends to existing log."""
        coordination.ensure_coordination_structure()
        coordination.log_action("worker-1", "ACTION1")
        coordination.log_action("worker-1", "ACTION2")

        log_file = coordination.LOGS_DIR / "worker-1.log"
        content = log_file.read_text()
        assert "ACTION1" in content
        assert "ACTION2" in content


class TestCLI:
    """Tests for the CLI interface."""

    def test_cli_leader_init(self, mock_coordination_dir, monkeypatch, capsys):
        """opt-a-cli-001: CLI leader init command works."""
        monkeypatch.setattr(
            "sys.argv",
            ["coordination.py", "leader", "init", "Build API"]
        )

        coordination.main()

        assert coordination.MASTER_PLAN_FILE.exists()

    def test_cli_worker_list(self, tasks_file_with_data, monkeypatch, capsys):
        """opt-a-cli-002: CLI worker list command works."""
        monkeypatch.setattr(
            "sys.argv",
            ["coordination.py", "worker", "list"]
        )

        coordination.main()

        captured = capsys.readouterr()
        # Should show task list or "No available" message
        assert "task" in captured.out.lower() or "no available" in captured.out.lower()
