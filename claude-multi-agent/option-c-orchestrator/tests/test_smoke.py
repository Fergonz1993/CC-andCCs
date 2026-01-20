"""
Smoke tests for quick validation.

Feature: adv-test-019 - Smoke tests for quick validation

This module provides fast-running smoke tests that verify basic
functionality works. Run these tests first to quickly catch
obvious problems before running the full test suite.

Run with: pytest tests/test_smoke.py -v --timeout=10
Marker: pytest -m smoke

These tests should:
- Complete in under 1 second each
- Test only the most critical paths
- Not require external dependencies
- Be suitable for pre-commit hooks
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from orchestrator.models import (
    Task,
    TaskStatus,
    TaskResult,
    TaskContext,
    Discovery,
    CoordinationState,
    AgentRole,
    Agent,
    AgentMetrics,
)


# Mark all tests in this module as smoke tests
pytestmark = pytest.mark.smoke


# =============================================================================
# Model Import Smoke Tests
# =============================================================================

class TestImportSmoke:
    """Smoke tests for module imports."""

    def test_smoke_001_import_models(self):
        """
        smoke-001: All model classes can be imported.

        Critical check: If this fails, nothing else will work.
        """
        # These imports should not raise
        from orchestrator.models import Task
        from orchestrator.models import TaskStatus
        from orchestrator.models import TaskResult
        from orchestrator.models import TaskContext
        from orchestrator.models import Discovery
        from orchestrator.models import CoordinationState
        from orchestrator.models import Agent
        from orchestrator.models import AgentRole
        from orchestrator.models import AgentMetrics

        assert Task is not None
        assert TaskStatus is not None
        assert TaskResult is not None

    def test_smoke_002_import_orchestrator(self):
        """
        smoke-002: Orchestrator class can be imported.

        Critical check: Main orchestrator module loads without errors.
        """
        from orchestrator.orchestrator import Orchestrator

        assert Orchestrator is not None

    def test_smoke_003_import_agent(self):
        """
        smoke-003: Agent classes can be imported.

        Critical check: Agent module loads without errors.
        """
        from orchestrator.agent import ClaudeCodeAgent, AgentPool

        assert ClaudeCodeAgent is not None
        assert AgentPool is not None

    def test_smoke_004_cli_uses_async_orchestrator(self):
        """
        smoke-004a: CLI defaults to async orchestrator implementation.
        """
        from orchestrator import cli as cli_module
        from orchestrator.async_orchestrator import Orchestrator as AsyncOrchestrator

        assert cli_module.AsyncOrchestrator is AsyncOrchestrator


# =============================================================================
# Task Model Smoke Tests
# =============================================================================

class TestTaskSmoke:
    """Smoke tests for Task model."""

    def test_smoke_004_task_creation(self):
        """
        smoke-004: Task can be created with minimal arguments.

        Basic functionality: Create a task with just a description.
        """
        task = Task(description="Test task")

        assert task is not None
        assert task.description == "Test task"
        assert task.id is not None

    def test_smoke_005_task_id_format(self):
        """
        smoke-005: Task ID has expected format.

        Verify ID follows task-XXXXXXXX pattern.
        """
        task = Task(description="Test")

        assert task.id.startswith("task-")
        assert len(task.id) > 5

    def test_smoke_006_task_default_status(self):
        """
        smoke-006: New task has AVAILABLE status.

        Verify default status is correct.
        """
        task = Task(description="Test")

        assert task.status == TaskStatus.AVAILABLE

    def test_smoke_007_task_claim(self):
        """
        smoke-007: Task can be claimed.

        Basic claim operation works.
        """
        task = Task(description="Test")
        task.claim("worker-1")

        assert task.status == TaskStatus.CLAIMED
        assert task.claimed_by == "worker-1"

    def test_smoke_008_task_complete(self):
        """
        smoke-008: Task can be completed.

        Basic completion works.
        """
        task = Task(description="Test")
        result = TaskResult(output="Done")
        task.complete(result)

        assert task.status == TaskStatus.DONE
        assert task.result is not None


# =============================================================================
# Coordination State Smoke Tests
# =============================================================================

class TestCoordinationStateSmoke:
    """Smoke tests for CoordinationState."""

    def test_smoke_009_state_creation(self):
        """
        smoke-009: CoordinationState can be created.

        Basic state initialization works.
        """
        state = CoordinationState()

        assert state is not None
        assert isinstance(state.tasks, list)

    def test_smoke_010_state_add_task(self):
        """
        smoke-010: Tasks can be added to state.

        Basic add_task works.
        """
        state = CoordinationState()
        task = Task(description="Test")
        state.add_task(task)

        assert len(state.tasks) == 1

    def test_smoke_011_state_get_task(self):
        """
        smoke-011: Tasks can be retrieved by ID.

        Basic get_task works.
        """
        state = CoordinationState()
        task = Task(description="Test")
        state.add_task(task)

        retrieved = state.get_task(task.id)

        assert retrieved is not None
        assert retrieved.id == task.id

    def test_smoke_012_state_available_tasks(self):
        """
        smoke-012: Available tasks can be queried.

        get_available_tasks returns correct tasks.
        """
        state = CoordinationState()
        state.add_task(Task(description="Available task"))

        available = state.get_available_tasks()

        assert len(available) == 1

    def test_smoke_013_state_progress(self):
        """
        smoke-013: Progress can be calculated.

        get_progress returns valid data.
        """
        state = CoordinationState()
        state.add_task(Task(description="Test"))

        progress = state.get_progress()

        assert "total_tasks" in progress
        assert "percent_complete" in progress
        assert progress["total_tasks"] == 1


# =============================================================================
# TaskResult Smoke Tests
# =============================================================================

class TestTaskResultSmoke:
    """Smoke tests for TaskResult."""

    def test_smoke_014_result_creation(self):
        """
        smoke-014: TaskResult can be created.

        Basic result creation works.
        """
        result = TaskResult(output="Completed successfully")

        assert result is not None
        assert result.output == "Completed successfully"

    def test_smoke_015_result_with_files(self):
        """
        smoke-015: TaskResult can include file lists.

        File tracking works.
        """
        result = TaskResult(
            output="Done",
            files_modified=["file.py"],
            files_created=["new.py"]
        )

        assert len(result.files_modified) == 1
        assert len(result.files_created) == 1


# =============================================================================
# Discovery Smoke Tests
# =============================================================================

class TestDiscoverySmoke:
    """Smoke tests for Discovery."""

    def test_smoke_016_discovery_creation(self):
        """
        smoke-016: Discovery can be created.

        Basic discovery creation works.
        """
        disc = Discovery(
            agent_id="worker-1",
            content="Found something important"
        )

        assert disc is not None
        assert disc.id.startswith("disc-")

    def test_smoke_017_discovery_with_tags(self):
        """
        smoke-017: Discovery can include tags.

        Tag support works.
        """
        disc = Discovery(
            agent_id="worker-1",
            content="Finding",
            tags=["important", "api"]
        )

        assert len(disc.tags) == 2


# =============================================================================
# Agent Smoke Tests
# =============================================================================

class TestAgentSmoke:
    """Smoke tests for Agent."""

    def test_smoke_018_agent_creation(self):
        """
        smoke-018: Agent can be created.

        Basic agent creation works.
        """
        agent = Agent(id="worker-1", role=AgentRole.WORKER)

        assert agent is not None
        assert agent.id == "worker-1"
        assert agent.role == AgentRole.WORKER

    def test_smoke_019_agent_leader(self):
        """
        smoke-019: Leader agent can be created.

        Leader role works.
        """
        agent = Agent(id="leader", role=AgentRole.LEADER)

        assert agent.role == AgentRole.LEADER


# =============================================================================
# Enum Smoke Tests
# =============================================================================

class TestEnumSmoke:
    """Smoke tests for enum types."""

    def test_smoke_020_task_status_values(self):
        """
        smoke-020: TaskStatus has expected values.

        All status values exist.
        """
        assert TaskStatus.AVAILABLE is not None
        assert TaskStatus.CLAIMED is not None
        assert TaskStatus.IN_PROGRESS is not None
        assert TaskStatus.DONE is not None
        assert TaskStatus.FAILED is not None

    def test_smoke_021_agent_role_values(self):
        """
        smoke-021: AgentRole has expected values.

        All role values exist.
        """
        assert AgentRole.LEADER is not None
        assert AgentRole.WORKER is not None
        assert AgentRole.SPECIALIST is not None


# =============================================================================
# Serialization Smoke Tests
# =============================================================================

class TestSerializationSmoke:
    """Smoke tests for model serialization."""

    def test_smoke_022_task_to_dict(self):
        """
        smoke-022: Task can be serialized to dict.

        model_dump works.
        """
        task = Task(description="Test")
        data = task.model_dump()

        assert isinstance(data, dict)
        assert "description" in data
        assert "id" in data

    def test_smoke_023_state_to_dict(self):
        """
        smoke-023: CoordinationState can be serialized.

        Full state serialization works.
        """
        state = CoordinationState(goal="Test goal")
        state.add_task(Task(description="Task"))
        data = state.model_dump()

        assert isinstance(data, dict)
        assert "goal" in data
        assert "tasks" in data


# =============================================================================
# Context Smoke Tests
# =============================================================================

class TestContextSmoke:
    """Smoke tests for TaskContext."""

    def test_smoke_024_context_creation(self):
        """
        smoke-024: TaskContext can be created.

        Basic context creation works.
        """
        ctx = TaskContext()

        assert ctx is not None
        assert isinstance(ctx.files, list)

    def test_smoke_025_context_with_data(self):
        """
        smoke-025: TaskContext can hold data.

        Context fields work.
        """
        ctx = TaskContext(
            files=["file.py"],
            hints="Use pattern X",
            tags=["backend"]
        )

        assert len(ctx.files) == 1
        assert ctx.hints == "Use pattern X"


# =============================================================================
# Orchestrator Smoke Tests
# =============================================================================

class TestOrchestratorSmoke:
    """Smoke tests for Orchestrator."""

    def test_smoke_026_orchestrator_creation(self):
        """
        smoke-026: Orchestrator can be instantiated.

        Basic orchestrator creation works.
        """
        from orchestrator.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp:
            orch = Orchestrator(working_directory=tmp, verbose=False)

            assert orch is not None
            assert orch.max_workers == 3  # Default

    @pytest.mark.asyncio
    async def test_smoke_027_orchestrator_initialize(self):
        """
        smoke-027: Orchestrator can be initialized.

        Basic initialization works.
        """
        from orchestrator.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp:
            orch = Orchestrator(working_directory=tmp, verbose=False)
            await orch.initialize(goal="Test goal")

            assert orch.state.goal == "Test goal"

    def test_smoke_028_orchestrator_add_task(self):
        """
        smoke-028: Orchestrator can add tasks.

        Task addition works.
        """
        from orchestrator.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp:
            orch = Orchestrator(working_directory=tmp, verbose=False)
            task = orch.add_task("Test task")

            assert task is not None
            assert len(orch.state.tasks) == 1


# =============================================================================
# Quick Validation Suite
# =============================================================================

def run_smoke_tests() -> bool:
    """
    Run all smoke tests programmatically.

    Returns True if all tests pass.

    Usage:
        from tests.test_smoke import run_smoke_tests
        if not run_smoke_tests():
            print("Smoke tests failed!")
            sys.exit(1)
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--timeout=30"],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)

    return result.returncode == 0


# =============================================================================
# Pre-commit Hook Support
# =============================================================================

PRECOMMIT_SCRIPT = '''#!/bin/bash
# Pre-commit hook to run smoke tests
# Install: cp tests/test_smoke.py .git/hooks/pre-commit

cd "$(git rev-parse --show-toplevel)/claude-multi-agent/option-c-orchestrator"

echo "Running smoke tests..."
python -m pytest tests/test_smoke.py -v --timeout=30

if [ $? -ne 0 ]; then
    echo "Smoke tests failed! Commit aborted."
    exit 1
fi

echo "Smoke tests passed."
exit 0
'''


if __name__ == "__main__":
    # Allow running directly: python -m tests.test_smoke
    import sys
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
