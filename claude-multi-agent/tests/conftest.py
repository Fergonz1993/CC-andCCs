"""
Root pytest configuration for the Claude Multi-Agent test suite.

This configures pytest for all tests across options.
"""

import pytest
import sys
from pathlib import Path


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "load: marks tests as load tests"
    )
    config.addinivalue_line(
        "markers", "chaos: marks tests as chaos tests"
    )
    config.addinivalue_line(
        "markers", "fuzz: marks tests as fuzzing tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark specific test types
        if "test_load" in str(item.fspath):
            item.add_marker(pytest.mark.load)
            item.add_marker(pytest.mark.slow)

        if "test_chaos" in str(item.fspath):
            item.add_marker(pytest.mark.chaos)
            item.add_marker(pytest.mark.slow)

        if "test_fuzzing" in str(item.fspath):
            item.add_marker(pytest.mark.fuzz)
            item.add_marker(pytest.mark.slow)

        if "test_benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def option_a_path(project_root: Path) -> Path:
    """Return path to Option A."""
    return project_root / "option-a-file-based"


@pytest.fixture(scope="session")
def option_b_path(project_root: Path) -> Path:
    """Return path to Option B."""
    return project_root / "option-b-mcp-broker"


@pytest.fixture(scope="session")
def option_c_path(project_root: Path) -> Path:
    """Return path to Option C."""
    return project_root / "option-c-orchestrator"
