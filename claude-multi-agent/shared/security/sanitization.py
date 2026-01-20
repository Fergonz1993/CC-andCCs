"""
Input Sanitization Module for Claude Multi-Agent Coordination System.

Provides comprehensive input validation and sanitization to prevent:
- Prompt injection attacks
- Path traversal
- Command injection
- XSS (for any web interfaces)
- SQL injection patterns
"""

import re
import html
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Union


class SanitizationError(Exception):
    """Raised when input fails sanitization."""
    def __init__(self, message: str, field: str, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(f"{field}: {message}")


class SanitizationMode(str, Enum):
    """Sanitization strictness modes."""
    STRICT = "strict"      # Reject any suspicious input
    MODERATE = "moderate"  # Allow some patterns but sanitize
    PERMISSIVE = "permissive"  # Sanitize but rarely reject


@dataclass
class SanitizationResult:
    """Result of sanitization."""
    is_valid: bool
    sanitized_value: Any
    warnings: List[str]
    modifications_made: List[str]

    def __bool__(self) -> bool:
        return self.is_valid


class InputSanitizer:
    """
    Comprehensive input sanitization for task descriptions and other inputs.

    Features:
    - Prompt injection detection and prevention
    - Path traversal prevention
    - Command injection prevention
    - Length limits
    - Character filtering
    - Pattern-based validation
    """

    # Patterns that might indicate prompt injection
    PROMPT_INJECTION_PATTERNS = [
        # System prompt manipulation
        r'(?i)ignore\s+(all\s+)?previous\s+instructions?',
        r'(?i)forget\s+(all\s+)?previous',
        r'(?i)disregard\s+(all\s+)?previous',
        r'(?i)system\s*:\s*you\s+are',
        r'(?i)new\s+instructions?:',
        r'(?i)override\s+instructions?',
        r'(?i)instructions?\s+override',
        r'(?i)you\s+are\s+now\s+a',
        r'(?i)act\s+as\s+(if\s+you\s+are\s+)?',
        r'(?i)pretend\s+(you\s+are|to\s+be)',
        r'(?i)role\s*:\s*system',
        r'(?i)\[system\]',
        r'(?i)<\s*system\s*>',
        r'(?i)###\s*system',
        r'(?i)<\|im_start\|>',
        r'(?i)<\|endoftext\|>',
        # Jailbreak attempts
        r'(?i)jailbreak',
        r'(?i)dan\s*mode',
        r'(?i)developer\s*mode',
        r'(?i)bypass\s+(safety|content|filter)',
        # Hidden instructions
        r'(?i)hidden\s+instruction',
        r'(?i)secret\s+instruction',
    ]

    # Patterns for path traversal
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\.[/\\]',                    # ../
        r'[/\\]\.\.',                    # /..
        r'%2e%2e[%2f%5c]',              # URL encoded ../
        r'%252e%252e[%252f%255c]',      # Double URL encoded
        r'\.\.%c0%af',                   # UTF-8 overlong encoding
        r'\.\.%c1%9c',
    ]

    # Patterns for command injection
    COMMAND_INJECTION_PATTERNS = [
        r'[;&|`$]',                      # Shell metacharacters
        r'\$\([^)]+\)',                  # Command substitution $(...)
        r'`[^`]+`',                      # Backtick command substitution
        r'\|\s*\w+',                     # Pipe to command
        r'>\s*[/\w]',                    # Output redirection
        r'<\s*[/\w]',                    # Input redirection
        r'&&\s*\w+',                     # Command chaining
        r'\|\|\s*\w+',                   # Command chaining
        r'\n\s*\w+',                     # Newline command injection
    ]

    # Patterns for SQL-like injection (in case data is stored in SQL)
    SQL_INJECTION_PATTERNS = [
        r"(?i)'\s*(or|and)\s+'",
        r'(?i)"\s*(or|and)\s+"',
        r'(?i)(union\s+(all\s+)?select)',
        r"(?i)'\s*;\s*(drop|delete|update|insert)",
        r'(?i)--\s*$',
        r'(?i)/\*.*\*/',
    ]

    def __init__(
        self,
        mode: SanitizationMode = SanitizationMode.MODERATE,
        max_length: int = 10000,
        custom_patterns: Optional[List[str]] = None,
        allow_html: bool = False,
        allow_markdown: bool = True,
    ):
        """
        Initialize the sanitizer.

        Args:
            mode: Strictness mode
            max_length: Maximum allowed input length
            custom_patterns: Additional patterns to detect and block
            allow_html: Whether to allow HTML tags
            allow_markdown: Whether to allow markdown formatting
        """
        self.mode = mode
        self.max_length = max_length
        self.allow_html = allow_html
        self.allow_markdown = allow_markdown

        # Compile patterns
        self._injection_patterns = [
            re.compile(p) for p in self.PROMPT_INJECTION_PATTERNS
        ]
        self._path_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL_PATTERNS
        ]
        self._command_patterns = [
            re.compile(p) for p in self.COMMAND_INJECTION_PATTERNS
        ]
        self._sql_patterns = [
            re.compile(p) for p in self.SQL_INJECTION_PATTERNS
        ]

        if custom_patterns:
            self._custom_patterns = [re.compile(p) for p in custom_patterns]
        else:
            self._custom_patterns = []

    def sanitize_task_description(
        self,
        description: str,
        context: Optional[str] = None,
    ) -> SanitizationResult:
        """
        Sanitize a task description.

        This is the main method for sanitizing task descriptions to prevent
        prompt injection and other attacks.

        Args:
            description: The task description to sanitize
            context: Optional context about where this description is used

        Returns:
            SanitizationResult with sanitized value or error
        """
        warnings = []
        modifications = []

        # Check for None or empty
        if not description:
            return SanitizationResult(
                is_valid=True,
                sanitized_value="",
                warnings=["Empty description"],
                modifications_made=[],
            )

        value = description

        # Length check
        if len(value) > self.max_length:
            if self.mode == SanitizationMode.STRICT:
                raise SanitizationError(
                    f"Description exceeds maximum length of {self.max_length}",
                    "description",
                    value[:100] + "...",
                )
            value = value[:self.max_length]
            modifications.append(f"Truncated to {self.max_length} characters")
            warnings.append("Description was truncated")

        # Check for prompt injection patterns
        for pattern in self._injection_patterns:
            if pattern.search(value):
                if self.mode == SanitizationMode.STRICT:
                    raise SanitizationError(
                        "Potential prompt injection detected",
                        "description",
                    )
                # In moderate/permissive mode, try to neutralize
                value = pattern.sub('[REMOVED]', value)
                modifications.append(f"Removed pattern matching: {pattern.pattern[:30]}...")
                warnings.append("Potential prompt injection pattern removed")

        # Check for path traversal
        for pattern in self._path_patterns:
            if pattern.search(value):
                if self.mode == SanitizationMode.STRICT:
                    raise SanitizationError(
                        "Path traversal pattern detected",
                        "description",
                    )
                value = pattern.sub('[PATH]', value)
                modifications.append("Removed path traversal pattern")
                warnings.append("Path traversal pattern detected and removed")

        # Check for custom patterns
        for pattern in self._custom_patterns:
            if pattern.search(value):
                if self.mode == SanitizationMode.STRICT:
                    raise SanitizationError(
                        "Input matches blocked pattern",
                        "description",
                    )
                value = pattern.sub('[FILTERED]', value)
                modifications.append(f"Filtered pattern: {pattern.pattern[:20]}...")

        # HTML handling
        if not self.allow_html:
            original = value
            value = html.escape(value)
            if value != original:
                modifications.append("Escaped HTML characters")

        # Normalize whitespace
        original = value
        value = ' '.join(value.split())
        if value != original:
            modifications.append("Normalized whitespace")

        return SanitizationResult(
            is_valid=True,
            sanitized_value=value,
            warnings=warnings,
            modifications_made=modifications,
        )

    def sanitize_path(self, path: str, allow_absolute: bool = False) -> SanitizationResult:
        """
        Sanitize a file path.

        Args:
            path: The path to sanitize
            allow_absolute: Whether to allow absolute paths

        Returns:
            SanitizationResult with sanitized path
        """
        warnings = []
        modifications = []

        if not path:
            return SanitizationResult(
                is_valid=True,
                sanitized_value="",
                warnings=[],
                modifications_made=[],
            )

        value = path

        # Check for path traversal
        for pattern in self._path_patterns:
            if pattern.search(value):
                raise SanitizationError(
                    "Path traversal detected",
                    "path",
                    value,
                )

        # Check for absolute paths if not allowed
        if not allow_absolute and (value.startswith('/') or (len(value) > 1 and value[1] == ':')):
            if self.mode == SanitizationMode.STRICT:
                raise SanitizationError(
                    "Absolute paths not allowed",
                    "path",
                    value,
                )
            # Convert to relative
            value = value.lstrip('/').lstrip('\\')
            if len(value) > 1 and value[1] == ':':
                value = value[2:].lstrip('/').lstrip('\\')
            modifications.append("Converted to relative path")
            warnings.append("Absolute path converted to relative")

        # Remove null bytes
        if '\x00' in value:
            value = value.replace('\x00', '')
            modifications.append("Removed null bytes")
            warnings.append("Null bytes detected and removed")

        # Normalize separators
        value = value.replace('\\', '/')

        # Remove consecutive slashes
        while '//' in value:
            value = value.replace('//', '/')
            modifications.append("Normalized path separators")

        return SanitizationResult(
            is_valid=True,
            sanitized_value=value,
            warnings=warnings,
            modifications_made=modifications,
        )

    def sanitize_agent_id(self, agent_id: str) -> SanitizationResult:
        """
        Sanitize an agent ID.

        Agent IDs should be alphanumeric with hyphens/underscores.
        """
        if not agent_id:
            raise SanitizationError("Agent ID cannot be empty", "agent_id")

        warnings = []
        modifications = []

        # Allow only alphanumeric, hyphen, underscore
        if not re.match(r'^[a-zA-Z0-9_-]+$', agent_id):
            if self.mode == SanitizationMode.STRICT:
                raise SanitizationError(
                    "Agent ID contains invalid characters",
                    "agent_id",
                    agent_id,
                )
            # Clean it up
            cleaned = re.sub(r'[^a-zA-Z0-9_-]', '', agent_id)
            if not cleaned:
                raise SanitizationError(
                    "Agent ID has no valid characters",
                    "agent_id",
                    agent_id,
                )
            modifications.append("Removed invalid characters from agent ID")
            warnings.append("Agent ID contained invalid characters")
            agent_id = cleaned

        # Length check
        if len(agent_id) > 64:
            if self.mode == SanitizationMode.STRICT:
                raise SanitizationError(
                    "Agent ID too long (max 64 characters)",
                    "agent_id",
                )
            agent_id = agent_id[:64]
            modifications.append("Truncated agent ID to 64 characters")

        return SanitizationResult(
            is_valid=True,
            sanitized_value=agent_id,
            warnings=warnings,
            modifications_made=modifications,
        )

    def sanitize_json_input(self, data: Dict[str, Any], schema: Dict[str, Any]) -> SanitizationResult:
        """
        Sanitize JSON input according to a schema.

        The schema should define expected fields and their types.
        """
        warnings = []
        modifications = []
        sanitized = {}

        for field_name, field_spec in schema.items():
            if field_name not in data:
                if field_spec.get('required', False):
                    raise SanitizationError(
                        f"Required field '{field_name}' is missing",
                        field_name,
                    )
                continue

            value = data[field_name]
            field_type = field_spec.get('type', 'string')

            # Type validation
            if field_type == 'string':
                if not isinstance(value, str):
                    value = str(value)
                    modifications.append(f"Converted {field_name} to string")

                # Apply string sanitization based on field purpose
                if field_spec.get('sanitize_as') == 'task_description':
                    result = self.sanitize_task_description(value)
                    value = result.sanitized_value
                    warnings.extend(result.warnings)
                    modifications.extend(result.modifications_made)
                elif field_spec.get('sanitize_as') == 'path':
                    result = self.sanitize_path(value)
                    value = result.sanitized_value
                    warnings.extend(result.warnings)
                    modifications.extend(result.modifications_made)
                elif field_spec.get('sanitize_as') == 'agent_id':
                    result = self.sanitize_agent_id(value)
                    value = result.sanitized_value
                    warnings.extend(result.warnings)
                    modifications.extend(result.modifications_made)

                # Max length
                max_len = field_spec.get('max_length', self.max_length)
                if len(value) > max_len:
                    value = value[:max_len]
                    modifications.append(f"Truncated {field_name}")

            elif field_type == 'integer':
                if not isinstance(value, int):
                    try:
                        value = int(value)
                        modifications.append(f"Converted {field_name} to integer")
                    except (ValueError, TypeError):
                        raise SanitizationError(
                            f"Field '{field_name}' must be an integer",
                            field_name,
                            value,
                        )
                # Range validation
                if 'min' in field_spec and value < field_spec['min']:
                    value = field_spec['min']
                    modifications.append(f"Clamped {field_name} to minimum")
                if 'max' in field_spec and value > field_spec['max']:
                    value = field_spec['max']
                    modifications.append(f"Clamped {field_name} to maximum")

            elif field_type == 'array':
                if not isinstance(value, list):
                    raise SanitizationError(
                        f"Field '{field_name}' must be an array",
                        field_name,
                        value,
                    )
                # Recursively sanitize array items
                item_type = field_spec.get('items', {}).get('type', 'string')
                if item_type == 'string':
                    sanitized_items = []
                    for item in value:
                        if isinstance(item, str):
                            # Apply string sanitization if specified
                            if field_spec.get('items', {}).get('sanitize_as') == 'path':
                                result = self.sanitize_path(item)
                                sanitized_items.append(result.sanitized_value)
                            else:
                                sanitized_items.append(item)
                        else:
                            sanitized_items.append(str(item))
                    value = sanitized_items

            sanitized[field_name] = value

        return SanitizationResult(
            is_valid=True,
            sanitized_value=sanitized,
            warnings=warnings,
            modifications_made=modifications,
        )

    def detect_prompt_injection(self, text: str) -> tuple[bool, List[str]]:
        """
        Detect potential prompt injection patterns.

        Returns:
            (is_suspicious, list of matched patterns)
        """
        matches = []
        for pattern in self._injection_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern[:50])

        return len(matches) > 0, matches

    def sanitize_for_command(self, value: str) -> str:
        """
        Sanitize a value to be safely used in a shell command.

        This is a very strict sanitization for values that might be
        used in shell commands.
        """
        # Only allow alphanumeric, underscore, hyphen, dot, forward slash
        sanitized = re.sub(r'[^a-zA-Z0-9_\-./]', '', value)

        # Remove any path traversal
        while '..' in sanitized:
            sanitized = sanitized.replace('..', '')

        return sanitized


# Convenience function
def sanitize_task_input(
    description: str,
    priority: int = 5,
    dependencies: Optional[List[str]] = None,
    context_files: Optional[List[str]] = None,
    hints: str = "",
    mode: SanitizationMode = SanitizationMode.MODERATE,
) -> Dict[str, Any]:
    """
    Sanitize all inputs for creating a task.

    Returns sanitized task data ready for use.
    """
    sanitizer = InputSanitizer(mode=mode)

    # Sanitize description
    desc_result = sanitizer.sanitize_task_description(description)
    if not desc_result.is_valid:
        raise SanitizationError("Invalid task description", "description")

    # Sanitize hints
    hints_result = sanitizer.sanitize_task_description(hints)

    # Sanitize file paths
    sanitized_files = []
    if context_files:
        for f in context_files:
            path_result = sanitizer.sanitize_path(f, allow_absolute=True)
            sanitized_files.append(path_result.sanitized_value)

    # Sanitize dependencies (should be task IDs)
    sanitized_deps = []
    if dependencies:
        for dep in dependencies:
            dep_result = sanitizer.sanitize_agent_id(dep)  # Task IDs follow similar rules
            sanitized_deps.append(dep_result.sanitized_value)

    # Validate priority
    if not isinstance(priority, int) or priority < 1 or priority > 10:
        priority = max(1, min(10, int(priority)))

    return {
        'description': desc_result.sanitized_value,
        'priority': priority,
        'dependencies': sanitized_deps,
        'context_files': sanitized_files,
        'hints': hints_result.sanitized_value,
    }
