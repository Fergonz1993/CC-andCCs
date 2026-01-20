"""
Pytest configuration for orchestrator tests.

Provides shared fixtures and configuration for all test modules including:
- Test environment isolation
- Test data generators
- Mock MCP server
- Snapshot testing utilities
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator

# Ensure the src directory is in the path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "smoke: mark test as a smoke test (fast, critical path)"
    )
    config.addinivalue_line(
        "markers", "regression: mark test as a regression test for a fixed bug"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running"
    )


# =============================================================================
# Basic Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    tmp = tempfile.mkdtemp(prefix="orchestrator_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def temp_working_dir(temp_dir: Path) -> Path:
    """Provide a temporary working directory."""
    return temp_dir


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def sample_task():
    """Provide a sample Task for testing."""
    from orchestrator.models import Task
    return Task(description="Sample test task")


@pytest.fixture
def sample_result():
    """Provide a sample TaskResult for testing."""
    from orchestrator.models import TaskResult
    return TaskResult(
        output="Task completed successfully",
        files_modified=["test.py"],
    )


@pytest.fixture
def sample_discovery():
    """Provide a sample Discovery for testing."""
    from orchestrator.models import Discovery
    return Discovery(
        agent_id="test-worker",
        content="Test discovery content"
    )


@pytest.fixture
def sample_state():
    """Provide a sample CoordinationState for testing."""
    from orchestrator.models import CoordinationState, Task
    state = CoordinationState(goal="Test coordination")
    state.add_task(Task(description="Task 1"))
    state.add_task(Task(description="Task 2"))
    return state


# =============================================================================
# Orchestrator Fixtures
# =============================================================================

@pytest.fixture
def orchestrator(temp_working_dir):
    """Provide an Orchestrator instance for testing."""
    from orchestrator.orchestrator import Orchestrator
    return Orchestrator(
        working_directory=str(temp_working_dir),
        max_workers=2,
        verbose=False
    )


# =============================================================================
# Import fixtures from other test modules
# =============================================================================

# Import fixtures from test_data_generators
pytest_plugins = []

try:
    from tests.test_data_generators import (
        task_factory,
        result_factory,
        discovery_factory,
        agent_factory,
        state_factory,
        scenario_generator,
        simple_scenario,
        complex_scenario,
    )
except ImportError:
    pass

try:
    from tests.test_environment import (
        isolated_fs,
        isolated_env,
        test_environment,
        coordination_environment,
    )
except ImportError:
    pass

try:
    from tests.mcp_mock_server import (
        mock_mcp_server,
        mcp_client,
    )
except ImportError:
    pass

try:
    from tests.test_snapshots import snapshot_manager
except ImportError:
    pass
