"""
Pytest configuration and fixtures for Option A tests.

Provides shared fixtures for testing the file-based coordination system.
"""

import pytest
import tempfile
import shutil
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Generator, Any, Dict

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test coordination files."""
    tmp = tempfile.mkdtemp(prefix="coordination_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def mock_coordination_dir(temp_dir: Path, monkeypatch) -> Path:
    """Set up a mock coordination directory structure."""
    # Import after path is set
    import coordination

    coordination_dir = temp_dir / ".coordination"
    coordination_dir.mkdir()
    (coordination_dir / "context").mkdir()
    (coordination_dir / "logs").mkdir()
    (coordination_dir / "results").mkdir()

    # Monkeypatch the module constants
    monkeypatch.setattr(coordination, "COORDINATION_DIR", coordination_dir)
    monkeypatch.setattr(coordination, "TASKS_FILE", coordination_dir / "tasks.json")
    monkeypatch.setattr(coordination, "MASTER_PLAN_FILE", coordination_dir / "master-plan.md")
    monkeypatch.setattr(coordination, "DISCOVERIES_FILE", coordination_dir / "context" / "discoveries.md")
    monkeypatch.setattr(coordination, "LOGS_DIR", coordination_dir / "logs")
    monkeypatch.setattr(coordination, "RESULTS_DIR", coordination_dir / "results")
    monkeypatch.setattr(coordination, "AGENTS_FILE", coordination_dir / "agents.json")

    return coordination_dir


@pytest.fixture
def sample_tasks_data() -> Dict[str, Any]:
    """Return sample tasks data structure."""
    return {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "tasks": [
            {
                "id": "task-001",
                "description": "Test task 1",
                "status": "available",
                "priority": 1,
                "dependencies": [],
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "task-002",
                "description": "Test task 2",
                "status": "available",
                "priority": 2,
                "dependencies": ["task-001"],
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "task-003",
                "description": "Test task 3 - done",
                "status": "done",
                "priority": 3,
                "dependencies": [],
                "claimed_by": "worker-1",
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "result": {"output": "Completed successfully"}
            }
        ]
    }


@pytest.fixture
def tasks_file_with_data(mock_coordination_dir: Path, sample_tasks_data: Dict[str, Any]) -> Path:
    """Create a tasks.json file with sample data."""
    import coordination

    tasks_file = mock_coordination_dir / "tasks.json"
    tasks_file.write_text(json.dumps(sample_tasks_data, indent=2))
    return tasks_file


@pytest.fixture
def sample_agents_data() -> Dict[str, Any]:
    """Return sample agents data structure."""
    return {
        "agents": [
            {
                "id": "worker-1",
                "capabilities": "python,testing",
                "registered_at": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            },
            {
                "id": "worker-2",
                "capabilities": "javascript,frontend",
                "registered_at": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }
        ]
    }


@pytest.fixture
def agents_file_with_data(mock_coordination_dir: Path, sample_agents_data: Dict[str, Any]) -> Path:
    """Create an agents.json file with sample data."""
    import coordination

    agents_file = mock_coordination_dir / "agents.json"
    agents_file.write_text(json.dumps(sample_agents_data, indent=2))
    return agents_file
