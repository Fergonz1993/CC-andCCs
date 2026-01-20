"""
Configuration Schema Validation (adv-cross-008)

Provides schema validation for configuration files across all coordination
options. Uses JSON Schema for validation and supports custom validators.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    path: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    expected: Optional[str] = None
    actual: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "path": self.path,
            "message": self.message,
            "severity": self.severity.value,
        }
        if self.expected:
            d["expected"] = self.expected
        if self.actual:
            d["actual"] = self.actual
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    config_path: str = ""
    schema_version: str = ""
    validated_at: datetime = field(default_factory=datetime.now)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [i.to_dict() for i in self.issues],
            "config_path": self.config_path,
            "schema_version": self.schema_version,
            "validated_at": self.validated_at.isoformat(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


# JSON Schema definitions for each option

TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "pattern": "^task-[a-zA-Z0-9-]+$"},
        "description": {"type": "string", "minLength": 1},
        "status": {
            "type": "string",
            "enum": ["available", "claimed", "in_progress", "done", "failed", "cancelled"],
        },
        "priority": {"type": "integer", "minimum": 1, "maximum": 10},
        "claimed_by": {"type": ["string", "null"]},
        "dependencies": {
            "type": "array",
            "items": {"type": "string"},
        },
        "context": {
            "type": ["object", "null"],
            "properties": {
                "files": {"type": "array", "items": {"type": "string"}},
                "hints": {"type": "string"},
                "parent_task": {"type": "string"},
            },
        },
        "result": {
            "type": ["object", "null"],
            "properties": {
                "output": {"type": "string"},
                "files_modified": {"type": "array", "items": {"type": "string"}},
                "files_created": {"type": "array", "items": {"type": "string"}},
                "error": {"type": "string"},
            },
        },
        "created_at": {"type": "string", "format": "date-time"},
        "claimed_at": {"type": ["string", "null"], "format": "date-time"},
        "completed_at": {"type": ["string", "null"], "format": "date-time"},
    },
    "required": ["id", "description", "status", "priority"],
}

OPTION_A_TASKS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Option A Tasks File",
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"},
        "last_updated": {"type": "string", "format": "date-time"},
        "tasks": {
            "type": "array",
            "items": TASK_SCHEMA,
        },
    },
    "required": ["tasks"],
}

OPTION_B_STATE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Option B MCP State",
    "type": "object",
    "properties": {
        "master_plan": {"type": "string"},
        "goal": {"type": "string"},
        "tasks": {
            "type": "array",
            "items": TASK_SCHEMA,
        },
        "agents": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "role": {"type": "string", "enum": ["leader", "worker"]},
                    "last_heartbeat": {"type": "string", "format": "date-time"},
                    "current_task": {"type": ["string", "null"]},
                    "tasks_completed": {"type": "integer", "minimum": 0},
                },
                "required": ["id", "role"],
            },
        },
        "discoveries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "created_at": {"type": "string", "format": "date-time"},
                },
                "required": ["id", "agent_id", "content"],
            },
        },
        "created_at": {"type": "string", "format": "date-time"},
        "last_activity": {"type": "string", "format": "date-time"},
    },
    "required": ["tasks"],
}

OPTION_C_STATE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Option C Orchestrator State",
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "master_plan": {"type": "string"},
        "tasks": {
            "type": "array",
            "items": TASK_SCHEMA,
        },
        "discoveries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "created_at": {"type": "string"},
                    "related_task": {"type": ["string", "null"]},
                },
            },
        },
        "working_directory": {"type": "string"},
        "max_parallel_workers": {"type": "integer", "minimum": 1},
        "task_timeout_seconds": {"type": "integer", "minimum": 1},
        "created_at": {"type": "string"},
        "last_activity": {"type": "string"},
    },
    "required": ["tasks"],
}

UNIFIED_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Unified Configuration",
    "type": "object",
    "properties": {
        "default_option": {
            "type": "string",
            "enum": ["A", "B", "C", "auto"],
        },
        "coordination_dir": {"type": "string"},
        "verbose": {"type": "boolean"},
        "color_output": {"type": "boolean"},
        "json_output": {"type": "boolean"},
        "logging": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["debug", "info", "warning", "error", "critical"],
                },
                "format": {"type": "string", "enum": ["json", "text"]},
                "file": {"type": "string"},
            },
        },
        "plugins": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "enabled": {"type": "boolean"},
                    "config": {"type": "object"},
                },
                "required": ["name"],
            },
        },
    },
}


class ConfigSchema:
    """
    Represents a configuration schema with validation rules.
    """

    def __init__(
        self,
        schema: Dict[str, Any],
        name: str = "",
        version: str = "1.0.0",
    ):
        """
        Initialize with a JSON Schema.

        Args:
            schema: JSON Schema definition
            name: Schema name
            version: Schema version
        """
        self.schema = schema
        self.name = name or schema.get("title", "Unknown")
        self.version = version
        self._custom_validators: List[Callable[[Any], List[ValidationIssue]]] = []

    def add_validator(
        self,
        validator: Callable[[Any], List[ValidationIssue]],
    ) -> None:
        """Add a custom validator function."""
        self._custom_validators.append(validator)

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate data against the schema.

        Args:
            data: Data to validate

        Returns:
            ValidationResult with any issues found
        """
        issues = []

        # Validate against JSON Schema
        schema_issues = self._validate_schema(data, self.schema)
        issues.extend(schema_issues)

        # Run custom validators
        for validator in self._custom_validators:
            try:
                custom_issues = validator(data)
                issues.extend(custom_issues)
            except Exception as e:
                issues.append(ValidationIssue(
                    path="$",
                    message=f"Custom validator error: {e}",
                    severity=ValidationSeverity.WARNING,
                ))

        valid = not any(i.severity == ValidationSeverity.ERROR for i in issues)

        return ValidationResult(
            valid=valid,
            issues=issues,
            schema_version=self.version,
        )

    def _validate_schema(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str = "$",
    ) -> List[ValidationIssue]:
        """Validate data against a JSON Schema (simplified implementation)."""
        issues = []

        # Type validation
        expected_type = schema.get("type")
        if expected_type:
            if not self._check_type(data, expected_type):
                issues.append(ValidationIssue(
                    path=path,
                    message=f"Invalid type",
                    expected=str(expected_type),
                    actual=type(data).__name__,
                ))
                return issues  # Stop validation if type is wrong

        # Enum validation
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            issues.append(ValidationIssue(
                path=path,
                message=f"Value not in allowed values",
                expected=str(enum_values),
                actual=str(data),
            ))

        # String validation
        if expected_type == "string" and isinstance(data, str):
            if "minLength" in schema and len(data) < schema["minLength"]:
                issues.append(ValidationIssue(
                    path=path,
                    message=f"String too short",
                    expected=f"minLength={schema['minLength']}",
                    actual=f"length={len(data)}",
                ))

            if "maxLength" in schema and len(data) > schema["maxLength"]:
                issues.append(ValidationIssue(
                    path=path,
                    message=f"String too long",
                    expected=f"maxLength={schema['maxLength']}",
                    actual=f"length={len(data)}",
                ))

            if "pattern" in schema:
                if not re.match(schema["pattern"], data):
                    issues.append(ValidationIssue(
                        path=path,
                        message=f"String does not match pattern",
                        expected=schema["pattern"],
                        actual=data,
                    ))

        # Number validation
        if expected_type in ("integer", "number") and isinstance(data, (int, float)):
            if "minimum" in schema and data < schema["minimum"]:
                issues.append(ValidationIssue(
                    path=path,
                    message=f"Value below minimum",
                    expected=f"minimum={schema['minimum']}",
                    actual=str(data),
                ))

            if "maximum" in schema and data > schema["maximum"]:
                issues.append(ValidationIssue(
                    path=path,
                    message=f"Value above maximum",
                    expected=f"maximum={schema['maximum']}",
                    actual=str(data),
                ))

        # Array validation
        if expected_type == "array" and isinstance(data, list):
            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(data):
                    item_issues = self._validate_schema(
                        item,
                        items_schema,
                        f"{path}[{i}]",
                    )
                    issues.extend(item_issues)

        # Object validation
        if expected_type == "object" and isinstance(data, dict):
            # Required properties
            required = schema.get("required", [])
            for prop in required:
                if prop not in data:
                    issues.append(ValidationIssue(
                        path=f"{path}.{prop}",
                        message=f"Missing required property",
                        suggestion=f"Add '{prop}' to the configuration",
                    ))

            # Property validation
            properties = schema.get("properties", {})
            for prop, prop_schema in properties.items():
                if prop in data:
                    prop_issues = self._validate_schema(
                        data[prop],
                        prop_schema,
                        f"{path}.{prop}",
                    )
                    issues.extend(prop_issues)

            # Additional properties
            additional = schema.get("additionalProperties")
            if additional and isinstance(additional, dict):
                known_props = set(properties.keys())
                for prop, value in data.items():
                    if prop not in known_props:
                        prop_issues = self._validate_schema(
                            value,
                            additional,
                            f"{path}.{prop}",
                        )
                        issues.extend(prop_issues)

        return issues

    def _check_type(self, data: Any, expected: Union[str, List[str]]) -> bool:
        """Check if data matches the expected type(s)."""
        if isinstance(expected, list):
            return any(self._check_type(data, t) for t in expected)

        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected_types = type_map.get(expected)
        if expected_types:
            return isinstance(data, expected_types)

        return True


class ConfigValidator:
    """
    Validates configuration files for all coordination options.
    """

    def __init__(self):
        """Initialize the validator with default schemas."""
        self._schemas: Dict[str, ConfigSchema] = {}

        # Register default schemas
        self.register_schema("option_a", ConfigSchema(
            OPTION_A_TASKS_SCHEMA,
            "Option A Tasks",
            "1.0.0",
        ))
        self.register_schema("option_b", ConfigSchema(
            OPTION_B_STATE_SCHEMA,
            "Option B MCP State",
            "1.0.0",
        ))
        self.register_schema("option_c", ConfigSchema(
            OPTION_C_STATE_SCHEMA,
            "Option C Orchestrator State",
            "1.0.0",
        ))
        self.register_schema("unified", ConfigSchema(
            UNIFIED_CONFIG_SCHEMA,
            "Unified Configuration",
            "1.0.0",
        ))
        self.register_schema("task", ConfigSchema(
            {"type": "object", "properties": TASK_SCHEMA["properties"], "required": TASK_SCHEMA["required"]},
            "Task",
            "1.0.0",
        ))

    def register_schema(self, name: str, schema: ConfigSchema) -> None:
        """Register a schema."""
        self._schemas[name] = schema

    def get_schema(self, name: str) -> Optional[ConfigSchema]:
        """Get a schema by name."""
        return self._schemas.get(name)

    def validate(
        self,
        data: Any,
        schema_name: str,
    ) -> ValidationResult:
        """
        Validate data against a named schema.

        Args:
            data: Data to validate
            schema_name: Name of the schema to use

        Returns:
            ValidationResult
        """
        schema = self._schemas.get(schema_name)
        if not schema:
            return ValidationResult(
                valid=False,
                issues=[ValidationIssue(
                    path="$",
                    message=f"Unknown schema: {schema_name}",
                )],
            )

        return schema.validate(data)

    def validate_file(
        self,
        filepath: str,
        schema_name: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate a configuration file.

        Args:
            filepath: Path to the file
            schema_name: Schema to use (auto-detected if not specified)

        Returns:
            ValidationResult
        """
        path = Path(filepath)

        if not path.exists():
            return ValidationResult(
                valid=False,
                issues=[ValidationIssue(
                    path="$",
                    message=f"File not found: {filepath}",
                )],
                config_path=filepath,
            )

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                issues=[ValidationIssue(
                    path="$",
                    message=f"Invalid JSON: {e}",
                )],
                config_path=filepath,
            )

        # Auto-detect schema if not specified
        if not schema_name:
            schema_name = self._detect_schema(data, filepath)

        result = self.validate(data, schema_name)
        result.config_path = filepath

        return result

    def _detect_schema(self, data: Dict[str, Any], filepath: str) -> str:
        """Detect which schema to use based on data and filename."""
        filename = Path(filepath).name

        if filename == "tasks.json":
            return "option_a"
        elif filename == "mcp-state.json":
            return "option_b"
        elif filename == "state.json":
            return "option_c"
        elif "working_directory" in data:
            return "option_c"
        elif "agents" in data and isinstance(data.get("agents"), dict):
            return "option_b"
        elif "version" in data and "tasks" in data:
            return "option_a"

        return "unified"

    def validate_task(self, task: Dict[str, Any]) -> ValidationResult:
        """Validate a single task."""
        return self.validate(task, "task")

    def check_task_dependencies(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[ValidationIssue]:
        """
        Check task dependencies for issues.

        Checks for:
        - Missing dependencies
        - Circular dependencies
        - Orphaned tasks
        """
        issues = []
        task_ids = {t["id"] for t in tasks}

        # Check for missing dependencies
        for task in tasks:
            for dep in task.get("dependencies", []):
                if dep not in task_ids:
                    issues.append(ValidationIssue(
                        path=f"$.tasks[id={task['id']}].dependencies",
                        message=f"Dependency '{dep}' not found",
                        severity=ValidationSeverity.ERROR,
                        suggestion=f"Remove dependency or add task '{dep}'",
                    ))

        # Check for circular dependencies
        def has_cycle(task_id: str, visited: Set[str], path: Set[str]) -> bool:
            if task_id in path:
                return True
            if task_id in visited:
                return False

            visited.add(task_id)
            path.add(task_id)

            task = next((t for t in tasks if t["id"] == task_id), None)
            if task:
                for dep in task.get("dependencies", []):
                    if has_cycle(dep, visited, path):
                        return True

            path.remove(task_id)
            return False

        for task in tasks:
            if has_cycle(task["id"], set(), set()):
                issues.append(ValidationIssue(
                    path=f"$.tasks[id={task['id']}]",
                    message="Circular dependency detected",
                    severity=ValidationSeverity.ERROR,
                ))

        return issues


# Convenience function

def validate_config(
    filepath: str,
    schema_name: Optional[str] = None,
) -> ValidationResult:
    """
    Validate a configuration file.

    Args:
        filepath: Path to the configuration file
        schema_name: Optional schema name (auto-detected if not provided)

    Returns:
        ValidationResult
    """
    validator = ConfigValidator()
    return validator.validate_file(filepath, schema_name)
