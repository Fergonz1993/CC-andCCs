"""
Data Consistency Validation (adv-rel-005)

Implements data consistency validation for task state, coordination files,
and inter-agent communication.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any, Set, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ConsistencyLevel(str, Enum):
    """Levels of consistency checking."""
    BASIC = "basic"           # Schema validation only
    STANDARD = "standard"     # Schema + referential integrity
    STRICT = "strict"         # Full validation with checksums


class ValidationResult(str, Enum):
    """Result of a validation check."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


@dataclass
class ValidationError:
    """Details of a validation error."""
    field: str
    message: str
    severity: str = "error"
    suggested_fix: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    result: ValidationResult
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.result == ValidationResult.VALID

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class ConsistencyError(Exception):
    """Raised when data consistency validation fails."""

    def __init__(self, message: str, report: Optional[ValidationReport] = None):
        super().__init__(message)
        self.report = report


class DataConsistencyValidator:
    """
    Validates data consistency in the coordination system.

    Checks:
    - Task state transitions are valid
    - Dependencies reference existing tasks
    - Claimed tasks have valid workers
    - No circular dependencies
    - Checksums match for critical data
    """

    # Valid task state transitions
    VALID_TRANSITIONS = {
        "pending": {"available", "cancelled"},
        "available": {"claimed", "cancelled"},
        "claimed": {"in_progress", "available", "failed"},
        "in_progress": {"done", "failed", "available"},
        "done": set(),  # Terminal state
        "failed": {"available"},  # Can retry
        "blocked": {"available", "cancelled"},
        "cancelled": set(),  # Terminal state
    }

    def __init__(
        self,
        level: ConsistencyLevel = ConsistencyLevel.STANDARD,
        on_error: Optional[Callable[[ValidationError], None]] = None,
        on_warning: Optional[Callable[[ValidationError], None]] = None,
    ):
        """
        Initialize the validator.

        Args:
            level: Consistency checking level
            on_error: Callback for validation errors
            on_warning: Callback for validation warnings
        """
        self.level = level
        self.on_error = on_error
        self.on_warning = on_warning
        self._validation_history: List[ValidationReport] = []

    def validate_task(self, task: Dict[str, Any]) -> ValidationReport:
        """
        Validate a single task.

        Args:
            task: Task data as dictionary

        Returns:
            ValidationReport with results
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        # Required fields
        required = ["id", "description", "status", "priority"]
        for field_name in required:
            if field_name not in task:
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Required field '{field_name}' is missing",
                    suggested_fix=f"Add '{field_name}' field to task",
                ))

        # Validate status
        status = task.get("status")
        if status and status not in self.VALID_TRANSITIONS:
            errors.append(ValidationError(
                field="status",
                message=f"Invalid status: {status}",
                suggested_fix=f"Use one of: {list(self.VALID_TRANSITIONS.keys())}",
            ))

        # Validate priority
        priority = task.get("priority")
        if priority is not None:
            if not isinstance(priority, int) or priority < 1 or priority > 10:
                errors.append(ValidationError(
                    field="priority",
                    message=f"Priority must be integer 1-10, got: {priority}",
                    suggested_fix="Set priority to integer between 1 and 10",
                ))

        # Validate claimed_by if claimed
        if status in ("claimed", "in_progress"):
            if not task.get("claimed_by"):
                errors.append(ValidationError(
                    field="claimed_by",
                    message=f"Task with status '{status}' must have claimed_by set",
                    suggested_fix="Set claimed_by to worker ID or reset status",
                ))

        # Validate timestamps
        if status == "done" and not task.get("completed_at"):
            warnings.append(ValidationError(
                field="completed_at",
                message="Completed task should have completed_at timestamp",
                severity="warning",
            ))

        # Validate dependencies format
        deps = task.get("dependencies", [])
        if deps and not isinstance(deps, list):
            errors.append(ValidationError(
                field="dependencies",
                message="Dependencies must be a list",
                suggested_fix="Convert dependencies to list format",
            ))

        result = (
            ValidationResult.VALID if not errors
            else ValidationResult.WARNING if not errors and warnings
            else ValidationResult.INVALID
        )

        report = ValidationReport(
            result=result,
            errors=errors,
            warnings=warnings,
        )

        self._handle_callbacks(report)
        return report

    def validate_state(
        self,
        state: Dict[str, Any],
        known_workers: Optional[Set[str]] = None,
    ) -> ValidationReport:
        """
        Validate complete coordination state.

        Args:
            state: Complete state data
            known_workers: Set of known valid worker IDs

        Returns:
            ValidationReport with results
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        tasks = state.get("tasks", [])
        task_ids: Set[str] = set()

        # Validate each task
        for i, task in enumerate(tasks):
            task_report = self.validate_task(task)
            errors.extend(task_report.errors)
            warnings.extend(task_report.warnings)

            task_id = task.get("id")
            if task_id:
                if task_id in task_ids:
                    errors.append(ValidationError(
                        field=f"tasks[{i}].id",
                        message=f"Duplicate task ID: {task_id}",
                        suggested_fix="Ensure all task IDs are unique",
                    ))
                task_ids.add(task_id)

        # Referential integrity: check dependencies exist
        if self.level in (ConsistencyLevel.STANDARD, ConsistencyLevel.STRICT):
            for task in tasks:
                for dep_id in task.get("dependencies", []):
                    if dep_id not in task_ids:
                        errors.append(ValidationError(
                            field=f"tasks.{task.get('id')}.dependencies",
                            message=f"Dependency references non-existent task: {dep_id}",
                            suggested_fix=f"Remove invalid dependency or create task {dep_id}",
                        ))

            # Check claimed_by references valid workers
            if known_workers:
                for task in tasks:
                    claimed_by = task.get("claimed_by")
                    if claimed_by and claimed_by not in known_workers:
                        warnings.append(ValidationError(
                            field=f"tasks.{task.get('id')}.claimed_by",
                            message=f"Task claimed by unknown worker: {claimed_by}",
                            severity="warning",
                        ))

        # Check for circular dependencies
        if self.level in (ConsistencyLevel.STANDARD, ConsistencyLevel.STRICT):
            cycles = self._detect_cycles(tasks)
            for cycle in cycles:
                errors.append(ValidationError(
                    field="dependencies",
                    message=f"Circular dependency detected: {' -> '.join(cycle)}",
                    suggested_fix="Remove one dependency to break the cycle",
                ))

        # Strict mode: compute and store checksum
        checksum = None
        if self.level == ConsistencyLevel.STRICT:
            checksum = self._compute_checksum(state)

        result = (
            ValidationResult.VALID if not errors
            else ValidationResult.INVALID
        )

        report = ValidationReport(
            result=result,
            errors=errors,
            warnings=warnings,
            checksum=checksum,
            details={
                "total_tasks": len(tasks),
                "unique_task_ids": len(task_ids),
            },
        )

        self._validation_history.append(report)
        self._handle_callbacks(report)
        return report

    def validate_transition(
        self,
        current_status: str,
        new_status: str,
    ) -> ValidationReport:
        """
        Validate a task status transition.

        Args:
            current_status: Current task status
            new_status: Proposed new status

        Returns:
            ValidationReport indicating if transition is valid
        """
        errors: List[ValidationError] = []

        if current_status not in self.VALID_TRANSITIONS:
            errors.append(ValidationError(
                field="current_status",
                message=f"Invalid current status: {current_status}",
            ))
        elif new_status not in self.VALID_TRANSITIONS.get(current_status, set()):
            errors.append(ValidationError(
                field="status",
                message=f"Invalid transition: {current_status} -> {new_status}",
                suggested_fix=f"Valid next states from '{current_status}': {self.VALID_TRANSITIONS.get(current_status, set())}",
            ))

        result = ValidationResult.VALID if not errors else ValidationResult.INVALID
        report = ValidationReport(result=result, errors=errors)
        self._handle_callbacks(report)
        return report

    def verify_checksum(self, state: Dict[str, Any], expected_checksum: str) -> bool:
        """
        Verify state data matches expected checksum.

        Args:
            state: State data to verify
            expected_checksum: Expected checksum value

        Returns:
            True if checksum matches
        """
        actual = self._compute_checksum(state)
        matches = actual == expected_checksum

        if not matches:
            logger.warning(
                f"Checksum mismatch: expected {expected_checksum}, got {actual}"
            )

        return matches

    def _detect_cycles(self, tasks: List[Dict[str, Any]]) -> List[List[str]]:
        """Detect circular dependencies in task graph."""
        cycles = []

        # Build graph
        graph: Dict[str, Set[str]] = {}
        for task in tasks:
            task_id = task.get("id", "")
            deps = set(task.get("dependencies", []))
            graph[task_id] = deps

        # DFS for cycle detection
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    result = dfs(neighbor, path)
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # Found cycle
                    idx = path.index(neighbor)
                    return path[idx:] + [neighbor]

            rec_stack.remove(node)
            path.pop()
            return None

        for task_id in graph:
            if task_id not in visited:
                cycle = dfs(task_id, [])
                if cycle:
                    cycles.append(cycle)

        return cycles

    def _compute_checksum(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 checksum of data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _handle_callbacks(self, report: ValidationReport) -> None:
        """Handle error and warning callbacks."""
        if self.on_error:
            for error in report.errors:
                try:
                    self.on_error(error)
                except Exception as e:
                    logger.error(f"Error callback failed: {e}")

        if self.on_warning:
            for warning in report.warnings:
                try:
                    self.on_warning(warning)
                except Exception as e:
                    logger.error(f"Warning callback failed: {e}")

    def get_validation_history(self) -> List[ValidationReport]:
        """Get validation history."""
        return self._validation_history.copy()

    def clear_history(self) -> None:
        """Clear validation history."""
        self._validation_history.clear()


def validate_json_file(
    filepath: str,
    validator: Optional[DataConsistencyValidator] = None,
) -> ValidationReport:
    """
    Validate a JSON coordination file.

    Args:
        filepath: Path to JSON file
        validator: Validator to use (creates default if not provided)

    Returns:
        ValidationReport
    """
    validator = validator or DataConsistencyValidator()

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return validator.validate_state(data)
    except json.JSONDecodeError as e:
        return ValidationReport(
            result=ValidationResult.INVALID,
            errors=[ValidationError(
                field="file",
                message=f"Invalid JSON: {e}",
                suggested_fix="Fix JSON syntax errors",
            )],
        )
    except FileNotFoundError:
        return ValidationReport(
            result=ValidationResult.INVALID,
            errors=[ValidationError(
                field="file",
                message=f"File not found: {filepath}",
                suggested_fix="Ensure file exists",
            )],
        )
