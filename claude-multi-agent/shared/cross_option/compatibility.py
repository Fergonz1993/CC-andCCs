"""
Compatibility Testing Framework (adv-cross-009)

Provides a framework for testing compatibility between different
coordination options. Includes test cases for common operations
and interoperability scenarios.
"""

import asyncio
import json
import tempfile
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type


class TestStatus(Enum):
    """Status of a test execution."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: TestStatus
    duration_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
        }
        if self.message:
            d["message"] = self.message
        if self.details:
            d["details"] = self.details
        if self.error:
            d["error"] = self.error
        if self.stack_trace:
            d["stack_trace"] = self.stack_trace
        return d


@dataclass
class CompatibilityReport:
    """Report of compatibility test execution."""
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    tests_error: int = 0
    total_duration_ms: float = 0.0
    results: List[TestResult] = field(default_factory=list)
    source_option: str = ""
    target_option: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    environment: Dict[str, str] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.tests_run == 0:
            return 0.0
        return self.tests_passed / self.tests_run * 100

    @property
    def all_passed(self) -> bool:
        return self.tests_failed == 0 and self.tests_error == 0

    def add_result(self, result: TestResult) -> None:
        self.results.append(result)
        self.tests_run += 1
        self.total_duration_ms += result.duration_ms

        if result.status == TestStatus.PASSED:
            self.tests_passed += 1
        elif result.status == TestStatus.FAILED:
            self.tests_failed += 1
        elif result.status == TestStatus.SKIPPED:
            self.tests_skipped += 1
        elif result.status == TestStatus.ERROR:
            self.tests_error += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "tests_run": self.tests_run,
                "tests_passed": self.tests_passed,
                "tests_failed": self.tests_failed,
                "tests_skipped": self.tests_skipped,
                "tests_error": self.tests_error,
                "success_rate": f"{self.success_rate:.1f}%",
                "total_duration_ms": self.total_duration_ms,
            },
            "source_option": self.source_option,
            "target_option": self.target_option,
            "generated_at": self.generated_at.isoformat(),
            "environment": self.environment,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, pretty: bool = True) -> str:
        if pretty:
            return json.dumps(self.to_dict(), indent=2)
        return json.dumps(self.to_dict())

    def to_text(self) -> str:
        """Generate a human-readable text report."""
        lines = [
            "=" * 60,
            "COMPATIBILITY TEST REPORT",
            "=" * 60,
            f"Source Option: {self.source_option}",
            f"Target Option: {self.target_option}",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY",
            "-" * 40,
            f"Tests Run:    {self.tests_run}",
            f"Passed:       {self.tests_passed}",
            f"Failed:       {self.tests_failed}",
            f"Skipped:      {self.tests_skipped}",
            f"Errors:       {self.tests_error}",
            f"Success Rate: {self.success_rate:.1f}%",
            f"Duration:     {self.total_duration_ms:.2f}ms",
            "",
            "TEST RESULTS",
            "-" * 40,
        ]

        for result in self.results:
            status_symbol = {
                TestStatus.PASSED: "[PASS]",
                TestStatus.FAILED: "[FAIL]",
                TestStatus.SKIPPED: "[SKIP]",
                TestStatus.ERROR: "[ERR ]",
            }.get(result.status, "[????]")

            lines.append(f"{status_symbol} {result.name} ({result.duration_ms:.2f}ms)")

            if result.message:
                lines.append(f"        {result.message}")

            if result.error:
                lines.append(f"        Error: {result.error}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)


class CompatibilityTest(ABC):
    """
    Base class for compatibility tests.

    Subclass this to create specific compatibility tests.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        source_option: str = "",
        target_option: str = "",
    ):
        self.name = name
        self.description = description
        self.source_option = source_option
        self.target_option = target_option
        self._skip_reason: Optional[str] = None

    def skip(self, reason: str) -> None:
        """Mark the test to be skipped."""
        self._skip_reason = reason

    @abstractmethod
    def setup(self) -> None:
        """Set up test resources. Called before run()."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Run the test. Raise an exception on failure."""
        pass

    def teardown(self) -> None:
        """Clean up test resources. Called after run()."""
        pass

    def execute(self) -> TestResult:
        """Execute the test and return the result."""
        start_time = time.time()

        if self._skip_reason:
            return TestResult(
                name=self.name,
                status=TestStatus.SKIPPED,
                message=self._skip_reason,
            )

        try:
            self.setup()
            self.run()
            duration_ms = (time.time() - start_time) * 1000

            return TestResult(
                name=self.name,
                status=TestStatus.PASSED,
                duration_ms=duration_ms,
            )

        except AssertionError as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name=self.name,
                status=TestStatus.FAILED,
                duration_ms=duration_ms,
                message=str(e),
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name=self.name,
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                error=str(e),
                stack_trace=traceback.format_exc(),
            )

        finally:
            try:
                self.teardown()
            except Exception:
                pass  # Ignore teardown errors


class CompatibilityRunner:
    """
    Runs compatibility tests and generates reports.
    """

    def __init__(
        self,
        source_option: str = "",
        target_option: str = "",
    ):
        """
        Initialize the runner.

        Args:
            source_option: Source coordination option
            target_option: Target coordination option
        """
        self.source_option = source_option
        self.target_option = target_option
        self._tests: List[CompatibilityTest] = []

    def add_test(self, test: CompatibilityTest) -> None:
        """Add a test to run."""
        self._tests.append(test)

    def add_tests(self, tests: List[CompatibilityTest]) -> None:
        """Add multiple tests."""
        self._tests.extend(tests)

    def run(self, filter_pattern: Optional[str] = None) -> CompatibilityReport:
        """
        Run all tests and generate a report.

        Args:
            filter_pattern: Optional pattern to filter tests by name

        Returns:
            CompatibilityReport with all results
        """
        report = CompatibilityReport(
            source_option=self.source_option,
            target_option=self.target_option,
        )

        # Collect environment info
        import platform
        import sys
        report.environment = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
        }

        for test in self._tests:
            # Apply filter if specified
            if filter_pattern and filter_pattern not in test.name:
                continue

            result = test.execute()
            report.add_result(result)

        return report

    def run_async(
        self,
        filter_pattern: Optional[str] = None,
    ) -> CompatibilityReport:
        """Run tests that may be async."""
        return asyncio.run(self._run_async(filter_pattern))

    async def _run_async(
        self,
        filter_pattern: Optional[str] = None,
    ) -> CompatibilityReport:
        """Async implementation of run."""
        report = CompatibilityReport(
            source_option=self.source_option,
            target_option=self.target_option,
        )

        import platform
        import sys
        report.environment = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
        }

        for test in self._tests:
            if filter_pattern and filter_pattern not in test.name:
                continue

            result = test.execute()
            report.add_result(result)

        return report


# ============================================================================
# Built-in Compatibility Tests
# ============================================================================

class TaskAdapterCompatibilityTest(CompatibilityTest):
    """Test task adapter conversion between options."""

    def __init__(self, source_option: str, target_option: str):
        super().__init__(
            name=f"TaskAdapter_{source_option}_to_{target_option}",
            description=f"Test task conversion from Option {source_option} to Option {target_option}",
            source_option=source_option,
            target_option=target_option,
        )
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def setup(self) -> None:
        from .task_adapter import AdapterFactory
        self.source_adapter = AdapterFactory.get_adapter(self.source_option)
        self.target_adapter = AdapterFactory.get_adapter(self.target_option)
        self.temp_dir = tempfile.TemporaryDirectory()

    def run(self) -> None:
        from .task_adapter import UniversalTask, TaskStatus, TaskContext

        # Create a test task
        original = UniversalTask(
            id="test-task-001",
            description="Test task for compatibility",
            status=TaskStatus.AVAILABLE,
            priority=3,
            dependencies=["dep-1", "dep-2"],
            context=TaskContext(
                files=["file1.py", "file2.py"],
                hints="Some hints",
            ),
            created_at=datetime.now(),
        )

        # Convert to source format
        source_format = self.source_adapter.from_universal(original)

        # Convert back to universal
        converted = self.source_adapter.to_universal(source_format)

        # Verify core fields preserved
        assert converted.id == original.id, f"ID mismatch: {converted.id} != {original.id}"
        assert converted.description == original.description, "Description mismatch"
        assert converted.status == original.status, "Status mismatch"
        assert converted.priority == original.priority, "Priority mismatch"
        assert converted.dependencies == original.dependencies, "Dependencies mismatch"

        # Convert to target format
        target_format = self.target_adapter.from_universal(converted)

        # Convert back and verify again
        final = self.target_adapter.to_universal(target_format)
        assert final.id == original.id, "ID lost in target conversion"
        assert final.description == original.description, "Description lost in target conversion"

    def teardown(self) -> None:
        if self.temp_dir:
            self.temp_dir.cleanup()


class MigrationCompatibilityTest(CompatibilityTest):
    """Test migration between options."""

    def __init__(self, source_option: str, target_option: str):
        super().__init__(
            name=f"Migration_{source_option}_to_{target_option}",
            description=f"Test migration from Option {source_option} to Option {target_option}",
            source_option=source_option,
            target_option=target_option,
        )
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def setup(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create source data
        self._create_source_data()

    def _create_source_data(self) -> None:
        """Create test data in source format."""
        from .task_adapter import AdapterFactory, UniversalTask, TaskStatus

        adapter = AdapterFactory.get_adapter(self.source_option)

        tasks = [
            UniversalTask(
                id="task-001",
                description="First test task",
                status=TaskStatus.DONE,
                priority=1,
            ),
            UniversalTask(
                id="task-002",
                description="Second test task",
                status=TaskStatus.AVAILABLE,
                priority=2,
                dependencies=["task-001"],
            ),
        ]

        source_path = self.temp_path / "source"
        source_path.mkdir()

        if self.source_option == "A":
            adapter.save_tasks(tasks, str(source_path / "tasks.json"))
        elif self.source_option == "B":
            adapter.save_tasks(tasks, str(source_path / "mcp-state.json"))
        else:
            adapter.save_tasks(tasks, str(source_path / "state.json"))

        self.source_path = source_path

    def run(self) -> None:
        from .migration import MigrationTool, verify_migration

        tool = MigrationTool(create_backup=False)

        target_path = self.temp_path / "target"
        target_path.mkdir()

        if self.target_option == "A":
            target_file = str(target_path)
        elif self.target_option == "B":
            target_file = str(target_path / "mcp-state.json")
        else:
            target_file = str(target_path / "state.json")

        if self.source_option == "A":
            source_file = str(self.source_path)
        elif self.source_option == "B":
            source_file = str(self.source_path / "mcp-state.json")
        else:
            source_file = str(self.source_path / "state.json")

        # Perform migration
        result = tool.migrate(
            source_file,
            target_file,
            self.source_option,
            self.target_option,
        )

        assert result.success, f"Migration failed: {result.errors}"
        assert result.tasks_migrated == 2, f"Expected 2 tasks, got {result.tasks_migrated}"

    def teardown(self) -> None:
        if self.temp_dir:
            self.temp_dir.cleanup()


class SyncCompatibilityTest(CompatibilityTest):
    """Test synchronization between options."""

    def __init__(self, source_option: str, target_option: str):
        super().__init__(
            name=f"Sync_{source_option}_to_{target_option}",
            description=f"Test sync from Option {source_option} to Option {target_option}",
            source_option=source_option,
            target_option=target_option,
        )
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def setup(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def run(self) -> None:
        from .sync import TaskSynchronizer, SyncDirection
        from .task_adapter import AdapterFactory, UniversalTask, TaskStatus

        # Create source tasks
        source_adapter = AdapterFactory.get_adapter(self.source_option)
        target_adapter = AdapterFactory.get_adapter(self.target_option)

        source_tasks = [
            UniversalTask(
                id="sync-task-001",
                description="Sync test task",
                status=TaskStatus.AVAILABLE,
                priority=1,
            ),
        ]

        # Save source
        source_file = self.temp_path / "source_state.json"
        source_adapter.save_tasks(source_tasks, str(source_file))

        # Create empty target
        target_file = self.temp_path / "target_state.json"
        target_adapter.save_tasks([], str(target_file))

        # Sync
        synchronizer = TaskSynchronizer(
            self.source_option,
            self.target_option,
        )

        result = synchronizer.sync(
            str(source_file),
            str(target_file),
            SyncDirection.SOURCE_TO_TARGET,
        )

        assert result.success, f"Sync failed: {result.errors}"
        assert result.tasks_created == 1, f"Expected 1 task created, got {result.tasks_created}"

        # Verify target has the task
        target_tasks = target_adapter.load_tasks(str(target_file))
        assert len(target_tasks) == 1, f"Expected 1 task in target, got {len(target_tasks)}"

    def teardown(self) -> None:
        if self.temp_dir:
            self.temp_dir.cleanup()


class ConfigValidationCompatibilityTest(CompatibilityTest):
    """Test config validation for an option."""

    def __init__(self, option: str):
        super().__init__(
            name=f"ConfigValidation_{option}",
            description=f"Test configuration validation for Option {option}",
            source_option=option,
        )
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def setup(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def run(self) -> None:
        from .config_validation import ConfigValidator

        validator = ConfigValidator()

        # Create valid config
        if self.source_option == "A":
            config = {
                "version": "1.0",
                "tasks": [
                    {
                        "id": "task-001",
                        "description": "Test task",
                        "status": "available",
                        "priority": 5,
                    }
                ],
            }
            schema = "option_a"
        elif self.source_option == "B":
            config = {
                "goal": "Test goal",
                "tasks": [
                    {
                        "id": "task-001",
                        "description": "Test task",
                        "status": "available",
                        "priority": 5,
                    }
                ],
                "agents": {},
                "discoveries": [],
            }
            schema = "option_b"
        else:
            config = {
                "goal": "Test goal",
                "tasks": [
                    {
                        "id": "task-001",
                        "description": "Test task",
                        "status": "available",
                        "priority": 5,
                    }
                ],
                "discoveries": [],
            }
            schema = "option_c"

        # Validate
        result = validator.validate(config, schema)
        assert result.valid, f"Validation failed: {[i.message for i in result.errors]}"

        # Test invalid config
        invalid_config = {"tasks": "not_an_array"}
        result = validator.validate(invalid_config, schema)
        assert not result.valid, "Expected validation to fail for invalid config"

    def teardown(self) -> None:
        if self.temp_dir:
            self.temp_dir.cleanup()


def create_standard_tests() -> List[CompatibilityTest]:
    """Create a standard set of compatibility tests."""
    tests = []

    # Task adapter tests for all option pairs
    for source in ["A", "B", "C"]:
        for target in ["A", "B", "C"]:
            if source != target:
                tests.append(TaskAdapterCompatibilityTest(source, target))

    # Migration tests
    for source in ["A", "B", "C"]:
        for target in ["A", "B", "C"]:
            if source != target:
                tests.append(MigrationCompatibilityTest(source, target))

    # Sync tests
    for source in ["A", "B", "C"]:
        for target in ["A", "B", "C"]:
            if source != target:
                tests.append(SyncCompatibilityTest(source, target))

    # Config validation tests
    for option in ["A", "B", "C"]:
        tests.append(ConfigValidationCompatibilityTest(option))

    return tests


def run_compatibility_tests(
    filter_pattern: Optional[str] = None,
) -> CompatibilityReport:
    """
    Run standard compatibility tests.

    Args:
        filter_pattern: Optional pattern to filter tests

    Returns:
        CompatibilityReport with results
    """
    runner = CompatibilityRunner()
    runner.add_tests(create_standard_tests())
    return runner.run(filter_pattern)
