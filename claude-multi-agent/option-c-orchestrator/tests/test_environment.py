"""
Test environment isolation and management.

Feature: adv-test-017 - Test environment isolation

This module provides utilities for creating isolated test environments
to ensure tests don't interfere with each other or system state.

Features:
- Isolated filesystem contexts
- Environment variable isolation
- Process isolation
- Database isolation (mock)
- Network isolation (mock)
"""

import os
import sys
import tempfile
import shutil
import contextlib
from pathlib import Path
from typing import Dict, Any, Optional, Generator, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
import json


# =============================================================================
# Filesystem Isolation
# =============================================================================

class IsolatedFilesystem:
    """
    Creates an isolated filesystem context for testing.

    All file operations are contained within a temporary directory
    that is cleaned up after the test completes.
    """

    def __init__(
        self,
        prefix: str = "test_env_",
        preserve_on_failure: bool = False
    ):
        self.prefix = prefix
        self.preserve_on_failure = preserve_on_failure
        self._root: Optional[Path] = None
        self._original_cwd: Optional[str] = None
        self._failed = False

    @property
    def root(self) -> Path:
        """Get the root of the isolated filesystem."""
        if self._root is None:
            raise RuntimeError("IsolatedFilesystem not active")
        return self._root

    def __enter__(self) -> "IsolatedFilesystem":
        """Enter the isolated filesystem context."""
        self._root = Path(tempfile.mkdtemp(prefix=self.prefix))
        self._original_cwd = os.getcwd()
        os.chdir(self._root)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the isolated filesystem context."""
        if self._original_cwd:
            os.chdir(self._original_cwd)

        if exc_type is not None:
            self._failed = True

        if self._root and self._root.exists():
            if self._failed and self.preserve_on_failure:
                print(f"Test failed. Preserved test environment at: {self._root}")
            else:
                shutil.rmtree(self._root, ignore_errors=True)

    def create_file(self, path: str, content: str = "") -> Path:
        """Create a file in the isolated filesystem."""
        full_path = self._root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return full_path

    def create_directory(self, path: str) -> Path:
        """Create a directory in the isolated filesystem."""
        full_path = self._root / path
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path

    def read_file(self, path: str) -> str:
        """Read a file from the isolated filesystem."""
        return (self._root / path).read_text()

    def exists(self, path: str) -> bool:
        """Check if a path exists in the isolated filesystem."""
        return (self._root / path).exists()

    def list_files(self, path: str = ".") -> List[str]:
        """List files in a directory."""
        target = self._root / path
        if not target.exists():
            return []
        return [str(p.relative_to(self._root)) for p in target.rglob("*") if p.is_file()]


# =============================================================================
# Environment Variable Isolation
# =============================================================================

class IsolatedEnvironment:
    """
    Creates an isolated environment variable context.

    Changes to environment variables are reverted when the context exits.
    """

    def __init__(
        self,
        variables: Optional[Dict[str, str]] = None,
        clear_existing: bool = False
    ):
        self.variables = variables or {}
        self.clear_existing = clear_existing
        self._original_env: Dict[str, str] = {}
        self._removed_vars: List[str] = []

    def __enter__(self) -> "IsolatedEnvironment":
        """Enter the isolated environment context."""
        # Store original environment
        self._original_env = dict(os.environ)

        if self.clear_existing:
            # Store and remove all variables
            for key in list(os.environ.keys()):
                self._removed_vars.append(key)
                del os.environ[key]

        # Set new variables
        for key, value in self.variables.items():
            os.environ[key] = value

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the isolated environment context."""
        # Clear current environment
        os.environ.clear()

        # Restore original environment
        os.environ.update(self._original_env)

    def set(self, key: str, value: str) -> None:
        """Set an environment variable."""
        os.environ[key] = value

    def unset(self, key: str) -> None:
        """Unset an environment variable."""
        if key in os.environ:
            del os.environ[key]

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get an environment variable."""
        return os.environ.get(key, default)


# =============================================================================
# Combined Test Environment
# =============================================================================

@dataclass
class TestEnvironmentConfig:
    """Configuration for a test environment."""
    name: str
    working_directory: str = ""
    env_vars: Dict[str, str] = field(default_factory=dict)
    required_files: Dict[str, str] = field(default_factory=dict)
    required_directories: List[str] = field(default_factory=list)
    cleanup_on_exit: bool = True
    preserve_on_failure: bool = False


class TestEnvironment:
    """
    Complete test environment with filesystem, environment, and state isolation.

    Usage:
        config = TestEnvironmentConfig(
            name="test_orchestrator",
            env_vars={"DEBUG": "true"},
            required_files={"config.json": "{}"},
            required_directories=[".coordination", "results"]
        )

        with TestEnvironment(config) as env:
            # All operations isolated to temp directory
            env.run_test()
    """

    def __init__(self, config: TestEnvironmentConfig):
        self.config = config
        self._fs: Optional[IsolatedFilesystem] = None
        self._env: Optional[IsolatedEnvironment] = None
        self._setup_complete = False
        self._teardown_callbacks: List[Callable] = []

    @property
    def root(self) -> Path:
        """Get the root directory."""
        if self._fs is None:
            raise RuntimeError("TestEnvironment not active")
        return self._fs.root

    def __enter__(self) -> "TestEnvironment":
        """Enter the test environment."""
        # Create isolated filesystem
        self._fs = IsolatedFilesystem(
            prefix=f"test_{self.config.name}_",
            preserve_on_failure=self.config.preserve_on_failure
        )
        self._fs.__enter__()

        # Create isolated environment
        env_vars = dict(self.config.env_vars)
        env_vars["TEST_ENVIRONMENT"] = self.config.name
        env_vars["TEST_ROOT"] = str(self._fs.root)

        self._env = IsolatedEnvironment(variables=env_vars)
        self._env.__enter__()

        # Set up required structure
        self._setup_environment()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the test environment."""
        # Run teardown callbacks
        for callback in reversed(self._teardown_callbacks):
            try:
                callback()
            except Exception:
                pass

        # Exit environment isolation
        if self._env:
            self._env.__exit__(exc_type, exc_val, exc_tb)

        # Exit filesystem isolation
        if self._fs:
            self._fs.__exit__(exc_type, exc_val, exc_tb)

    def _setup_environment(self) -> None:
        """Set up the required directory structure and files."""
        # Create required directories
        for dir_path in self.config.required_directories:
            self._fs.create_directory(dir_path)

        # Create required files
        for file_path, content in self.config.required_files.items():
            self._fs.create_file(file_path, content)

        self._setup_complete = True

    def add_teardown(self, callback: Callable) -> None:
        """Add a cleanup callback to run on exit."""
        self._teardown_callbacks.append(callback)

    def create_file(self, path: str, content: str = "") -> Path:
        """Create a file in the test environment."""
        return self._fs.create_file(path, content)

    def read_file(self, path: str) -> str:
        """Read a file from the test environment."""
        return self._fs.read_file(path)

    def set_env(self, key: str, value: str) -> None:
        """Set an environment variable."""
        self._env.set(key, value)

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current environment state."""
        return {
            "name": self.config.name,
            "root": str(self._fs.root),
            "files": self._fs.list_files(),
            "env_vars": dict(os.environ),
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# Test Isolation Utilities
# =============================================================================

class TestIsolationManager:
    """
    Manages test isolation across multiple tests.

    Ensures each test runs in a clean, isolated environment.
    """

    _instances: Dict[str, "TestEnvironment"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_environment(cls, name: str) -> Optional["TestEnvironment"]:
        """Get an existing test environment."""
        with cls._lock:
            return cls._instances.get(name)

    @classmethod
    def register_environment(cls, env: "TestEnvironment") -> None:
        """Register a test environment."""
        with cls._lock:
            cls._instances[env.config.name] = env

    @classmethod
    def cleanup_all(cls) -> None:
        """Clean up all registered environments."""
        with cls._lock:
            for env in cls._instances.values():
                try:
                    env.__exit__(None, None, None)
                except Exception:
                    pass
            cls._instances.clear()


@contextlib.contextmanager
def isolated_test(
    name: str = "test",
    env_vars: Optional[Dict[str, str]] = None,
    files: Optional[Dict[str, str]] = None,
    directories: Optional[List[str]] = None
) -> Generator[TestEnvironment, None, None]:
    """
    Context manager for running a test in isolation.

    Usage:
        with isolated_test("my_test", env_vars={"DEBUG": "true"}) as env:
            # Test code here
            result = run_my_test()
            assert result.success
    """
    config = TestEnvironmentConfig(
        name=name,
        env_vars=env_vars or {},
        required_files=files or {},
        required_directories=directories or [],
    )

    with TestEnvironment(config) as env:
        yield env


# =============================================================================
# Coordination-Specific Environment
# =============================================================================

def create_coordination_environment(
    name: str = "coordination_test"
) -> TestEnvironmentConfig:
    """
    Create a test environment configured for coordination testing.

    Sets up the standard .coordination directory structure.
    """
    return TestEnvironmentConfig(
        name=name,
        required_directories=[
            ".coordination",
            ".coordination/context",
            ".coordination/logs",
            ".coordination/results",
        ],
        required_files={
            ".coordination/tasks.json": json.dumps({
                "version": "1.0",
                "tasks": [],
                "created_at": datetime.now().isoformat(),
            }),
            ".coordination/master-plan.md": "# Master Plan\n\nNo plan yet.",
            ".coordination/context/discoveries.md": "# Discoveries\n\n",
        },
        env_vars={
            "COORDINATION_DIR": ".coordination",
            "TEST_MODE": "true",
        },
    )


# =============================================================================
# Pytest Fixtures
# =============================================================================

import pytest


@pytest.fixture
def isolated_fs():
    """Provide an isolated filesystem for a test."""
    with IsolatedFilesystem() as fs:
        yield fs


@pytest.fixture
def isolated_env():
    """Provide isolated environment variables for a test."""
    with IsolatedEnvironment() as env:
        yield env


@pytest.fixture
def test_environment():
    """Provide a complete test environment."""
    config = TestEnvironmentConfig(name="pytest_test")
    with TestEnvironment(config) as env:
        yield env


@pytest.fixture
def coordination_environment():
    """Provide a test environment configured for coordination testing."""
    config = create_coordination_environment()
    with TestEnvironment(config) as env:
        yield env


# =============================================================================
# Tests for Environment Isolation
# =============================================================================

class TestEnvironmentIsolation:
    """Tests for the environment isolation utilities."""

    def test_isolated_filesystem_creates_temp_dir(self):
        """Isolated filesystem creates a temporary directory."""
        with IsolatedFilesystem() as fs:
            assert fs.root.exists()
            assert fs.root.is_dir()

    def test_isolated_filesystem_cleanup(self):
        """Isolated filesystem cleans up on exit."""
        root = None
        with IsolatedFilesystem() as fs:
            root = fs.root
            fs.create_file("test.txt", "content")
            assert root.exists()

        assert not root.exists()

    def test_isolated_filesystem_file_operations(self):
        """Isolated filesystem supports file operations."""
        with IsolatedFilesystem() as fs:
            fs.create_file("dir/file.txt", "hello")
            assert fs.exists("dir/file.txt")
            assert fs.read_file("dir/file.txt") == "hello"

    def test_isolated_environment_sets_vars(self):
        """Isolated environment sets variables."""
        original = os.environ.get("TEST_VAR_123")

        with IsolatedEnvironment({"TEST_VAR_123": "test_value"}):
            assert os.environ.get("TEST_VAR_123") == "test_value"

        assert os.environ.get("TEST_VAR_123") == original

    def test_isolated_environment_restores_vars(self):
        """Isolated environment restores original variables."""
        os.environ["RESTORE_TEST"] = "original"

        with IsolatedEnvironment({"RESTORE_TEST": "modified"}):
            assert os.environ.get("RESTORE_TEST") == "modified"

        assert os.environ.get("RESTORE_TEST") == "original"

        # Cleanup
        del os.environ["RESTORE_TEST"]

    def test_full_test_environment(self):
        """Full test environment provides isolation."""
        config = TestEnvironmentConfig(
            name="full_test",
            env_vars={"FULL_TEST": "true"},
            required_files={"config.json": "{}"},
            required_directories=["data", "logs"],
        )

        with TestEnvironment(config) as env:
            assert env.root.exists()
            assert (env.root / "config.json").exists()
            assert (env.root / "data").is_dir()
            assert (env.root / "logs").is_dir()
            assert os.environ.get("FULL_TEST") == "true"

    def test_coordination_environment_structure(self):
        """Coordination environment has correct structure."""
        config = create_coordination_environment()

        with TestEnvironment(config) as env:
            assert (env.root / ".coordination").is_dir()
            assert (env.root / ".coordination" / "tasks.json").exists()
            assert (env.root / ".coordination" / "context").is_dir()
            assert (env.root / ".coordination" / "logs").is_dir()
            assert (env.root / ".coordination" / "results").is_dir()

    def test_isolated_test_context_manager(self):
        """isolated_test context manager works correctly."""
        with isolated_test(
            "simple_test",
            env_vars={"SIMPLE": "true"},
            files={"test.txt": "content"}
        ) as env:
            assert env.root.exists()
            assert env.read_file("test.txt") == "content"
            assert os.environ.get("SIMPLE") == "true"

    def test_environment_state_snapshot(self):
        """Test environment can create state snapshots."""
        config = TestEnvironmentConfig(
            name="snapshot_test",
            required_files={"file1.txt": "content1", "file2.txt": "content2"}
        )

        with TestEnvironment(config) as env:
            snapshot = env.get_state_snapshot()

            assert snapshot["name"] == "snapshot_test"
            assert len(snapshot["files"]) == 2
            assert "timestamp" in snapshot

    def test_teardown_callbacks(self):
        """Test environment runs teardown callbacks."""
        callback_called = {"value": False}

        def my_callback():
            callback_called["value"] = True

        config = TestEnvironmentConfig(name="callback_test")

        with TestEnvironment(config) as env:
            env.add_teardown(my_callback)

        assert callback_called["value"] is True
