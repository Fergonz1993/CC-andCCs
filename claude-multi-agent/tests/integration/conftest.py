"""
Pytest configuration for integration tests across all options.

Provides fixtures for testing cross-option functionality.
"""

import pytest
import tempfile
import shutil
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Generator, Any, Dict

# Get the base project directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace for integration tests."""
    tmp = tempfile.mkdtemp(prefix="integration_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def option_a_module(temp_workspace: Path):
    """Import and configure Option A coordination module."""
    option_a_path = PROJECT_ROOT / "option-a-file-based"
    if str(option_a_path) not in sys.path:
        sys.path.insert(0, str(option_a_path))

    import coordination

    # Set up coordination in temp workspace
    os.chdir(temp_workspace)
    coordination_dir = temp_workspace / ".coordination"

    # Patch module paths
    coordination.COORDINATION_DIR = coordination_dir
    coordination.TASKS_FILE = coordination_dir / "tasks.json"
    coordination.MASTER_PLAN_FILE = coordination_dir / "master-plan.md"
    coordination.DISCOVERIES_FILE = coordination_dir / "context" / "discoveries.md"
    coordination.LOGS_DIR = coordination_dir / "logs"
    coordination.RESULTS_DIR = coordination_dir / "results"
    coordination.AGENTS_FILE = coordination_dir / "agents.json"

    return coordination


@pytest.fixture
def option_c_module(temp_workspace: Path):
    """Import and configure Option C orchestrator module."""
    option_c_path = PROJECT_ROOT / "option-c-orchestrator" / "src"
    if str(option_c_path) not in sys.path:
        sys.path.insert(0, str(option_c_path))

    from orchestrator import models
    return models


@pytest.fixture
def sample_workflow_tasks() -> list[Dict[str, Any]]:
    """Return a sample workflow with task dependencies."""
    return [
        {
            "description": "Set up project structure",
            "priority": 1,
            "dependencies": []
        },
        {
            "description": "Implement data models",
            "priority": 2,
            "dependencies": []
        },
        {
            "description": "Create API endpoints",
            "priority": 2,
            "dependencies": []
        },
        {
            "description": "Add authentication",
            "priority": 3,
            "dependencies": []  # Will be set after creation
        },
        {
            "description": "Write integration tests",
            "priority": 4,
            "dependencies": []  # Will depend on multiple tasks
        },
    ]


@pytest.fixture
def performance_config() -> Dict[str, Any]:
    """Configuration for performance testing."""
    return {
        "num_tasks": 100,
        "num_workers": 10,
        "task_timeout_seconds": 30,
        "max_concurrent_claims": 5,
    }
