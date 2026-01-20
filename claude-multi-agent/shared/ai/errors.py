"""
Context-Aware Error Messages (adv-ai-007)

Generates helpful, context-aware error messages that:
- Explain what went wrong clearly
- Suggest likely causes based on context
- Recommend specific fixes
- Link to relevant resources
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re


class ErrorCategory(str, Enum):
    """Categories of errors."""

    DEPENDENCY = "dependency"
    FILE_ACCESS = "file_access"
    CONFIGURATION = "configuration"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    VALIDATION = "validation"
    NETWORK = "network"
    INTERNAL = "internal"
    USER_INPUT = "user_input"
    UNKNOWN = "unknown"


@dataclass
class EnhancedError:
    """An enhanced error message with context and suggestions."""

    original_error: str
    category: ErrorCategory
    summary: str
    detailed_explanation: str
    likely_causes: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)
    related_context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    recoverable: bool = True


class ContextAwareErrorHandler:
    """
    Generates context-aware error messages.

    Analyzes errors in context to provide helpful, actionable feedback.
    """

    # Error patterns and their categorizations
    ERROR_PATTERNS = [
        # Dependency errors
        (r"(?:import|require|module).*(?:not found|cannot find|missing)", ErrorCategory.DEPENDENCY),
        (r"no such (?:module|package|library)", ErrorCategory.DEPENDENCY),
        (r"dependency.*(?:failed|missing|unresolved)", ErrorCategory.DEPENDENCY),
        (r"cannot resolve (?:dependency|module)", ErrorCategory.DEPENDENCY),

        # File access errors
        (r"(?:file|directory).*(?:not found|does not exist|missing)", ErrorCategory.FILE_ACCESS),
        (r"no such file or directory", ErrorCategory.FILE_ACCESS),
        (r"ENOENT", ErrorCategory.FILE_ACCESS),
        (r"path.*(?:invalid|not found)", ErrorCategory.FILE_ACCESS),

        # Permission errors
        (r"permission denied", ErrorCategory.PERMISSION),
        (r"EACCES", ErrorCategory.PERMISSION),
        (r"access denied", ErrorCategory.PERMISSION),
        (r"unauthorized", ErrorCategory.PERMISSION),

        # Timeout errors
        (r"timeout", ErrorCategory.TIMEOUT),
        (r"timed out", ErrorCategory.TIMEOUT),
        (r"deadline exceeded", ErrorCategory.TIMEOUT),
        (r"operation took too long", ErrorCategory.TIMEOUT),

        # Resource errors
        (r"out of memory", ErrorCategory.RESOURCE),
        (r"memory.*(?:limit|exceeded)", ErrorCategory.RESOURCE),
        (r"disk.*(?:full|space)", ErrorCategory.RESOURCE),
        (r"resource.*(?:exhausted|unavailable)", ErrorCategory.RESOURCE),

        # Validation errors
        (r"(?:invalid|malformed).*(?:input|data|format|json|yaml)", ErrorCategory.VALIDATION),
        (r"validation.*(?:failed|error)", ErrorCategory.VALIDATION),
        (r"schema.*(?:error|invalid)", ErrorCategory.VALIDATION),
        (r"type.*(?:error|mismatch)", ErrorCategory.VALIDATION),

        # Network errors
        (r"(?:connection|network).*(?:refused|failed|error)", ErrorCategory.NETWORK),
        (r"ECONNREFUSED", ErrorCategory.NETWORK),
        (r"DNS.*(?:failed|error)", ErrorCategory.NETWORK),
        (r"host.*(?:unreachable|not found)", ErrorCategory.NETWORK),

        # Configuration errors
        (r"(?:config|configuration).*(?:error|invalid|missing)", ErrorCategory.CONFIGURATION),
        (r"(?:missing|undefined).*(?:environment|variable|setting)", ErrorCategory.CONFIGURATION),
        (r"(?:invalid|bad).*(?:setting|option|parameter)", ErrorCategory.CONFIGURATION),
    ]

    # Cause suggestions by category
    CAUSE_SUGGESTIONS = {
        ErrorCategory.DEPENDENCY: [
            "Missing package installation",
            "Incorrect import path",
            "Version mismatch between packages",
            "Package not in requirements/dependencies file"
        ],
        ErrorCategory.FILE_ACCESS: [
            "File path is incorrect or misspelled",
            "File was deleted or moved",
            "Relative vs absolute path issue",
            "Working directory is different than expected"
        ],
        ErrorCategory.PERMISSION: [
            "Insufficient user permissions",
            "File is owned by different user/process",
            "Directory permissions too restrictive",
            "Running process without required privileges"
        ],
        ErrorCategory.TIMEOUT: [
            "External service is slow or unavailable",
            "Network latency issues",
            "Operation complexity exceeds time budget",
            "Deadlock or infinite loop"
        ],
        ErrorCategory.RESOURCE: [
            "Too many concurrent operations",
            "Memory leak in application",
            "Insufficient system resources",
            "Large data set exceeding limits"
        ],
        ErrorCategory.VALIDATION: [
            "Input data doesn't match expected format",
            "Missing required fields",
            "Data type mismatch",
            "Constraint violation"
        ],
        ErrorCategory.NETWORK: [
            "Remote service is down",
            "Network connectivity issues",
            "Firewall blocking connection",
            "Incorrect host or port"
        ],
        ErrorCategory.CONFIGURATION: [
            "Missing environment variable",
            "Configuration file not found",
            "Invalid configuration value",
            "Mismatched configuration format"
        ],
    }

    # Fix suggestions by category
    FIX_SUGGESTIONS = {
        ErrorCategory.DEPENDENCY: [
            "Run package manager install (npm install, pip install, etc.)",
            "Check import statement for typos",
            "Verify package is in dependencies file",
            "Try clearing package cache and reinstalling"
        ],
        ErrorCategory.FILE_ACCESS: [
            "Verify the file path exists",
            "Check for typos in the filename",
            "Use absolute path instead of relative",
            "Confirm current working directory"
        ],
        ErrorCategory.PERMISSION: [
            "Run with elevated privileges if appropriate",
            "Change file/directory permissions",
            "Check ownership of files",
            "Verify process has required access"
        ],
        ErrorCategory.TIMEOUT: [
            "Increase timeout value if safe",
            "Check if external service is responding",
            "Optimize the operation for speed",
            "Break into smaller chunks"
        ],
        ErrorCategory.RESOURCE: [
            "Free up system resources",
            "Increase resource limits if possible",
            "Process data in smaller batches",
            "Check for memory leaks"
        ],
        ErrorCategory.VALIDATION: [
            "Check input data format against schema",
            "Verify all required fields are present",
            "Validate data types match expectations",
            "Review schema documentation"
        ],
        ErrorCategory.NETWORK: [
            "Check network connectivity",
            "Verify remote service is running",
            "Check firewall rules",
            "Verify host and port are correct"
        ],
        ErrorCategory.CONFIGURATION: [
            "Set required environment variables",
            "Check configuration file exists and is valid",
            "Verify configuration values are correct",
            "Review documentation for required settings"
        ],
    }

    def __init__(self):
        self.error_history: List[Dict[str, Any]] = []

    def enhance_error(
        self,
        error: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EnhancedError:
        """
        Enhance an error message with context and suggestions.

        Args:
            error: The original error message
            context: Optional context (task, files, worker, etc.)

        Returns:
            EnhancedError with detailed explanation and suggestions
        """
        context = context or {}

        # Categorize the error
        category = self._categorize_error(error)

        # Generate summary
        summary = self._generate_summary(error, category)

        # Generate detailed explanation
        explanation = self._generate_explanation(error, category, context)

        # Get likely causes
        causes = self._get_likely_causes(error, category, context)

        # Get suggested fixes
        fixes = self._get_suggested_fixes(error, category, context)

        # Extract related context
        related = self._extract_related_context(error, context)

        # Determine if recoverable
        recoverable = self._is_recoverable(category, error)

        # Calculate confidence
        confidence = self._calculate_confidence(error, category)

        enhanced = EnhancedError(
            original_error=error,
            category=category,
            summary=summary,
            detailed_explanation=explanation,
            likely_causes=causes,
            suggested_fixes=fixes,
            related_context=related,
            confidence=confidence,
            recoverable=recoverable
        )

        # Record for learning
        self._record_error(enhanced, context)

        return enhanced

    def _categorize_error(self, error: str) -> ErrorCategory:
        """Categorize the error based on patterns."""
        error_lower = error.lower()

        for pattern, category in self.ERROR_PATTERNS:
            if re.search(pattern, error_lower):
                return category

        return ErrorCategory.UNKNOWN

    def _generate_summary(self, error: str, category: ErrorCategory) -> str:
        """Generate a short summary of the error."""
        summaries = {
            ErrorCategory.DEPENDENCY: "Missing or incompatible dependency",
            ErrorCategory.FILE_ACCESS: "File or directory access problem",
            ErrorCategory.PERMISSION: "Permission or access denied",
            ErrorCategory.TIMEOUT: "Operation timed out",
            ErrorCategory.RESOURCE: "System resource exhausted",
            ErrorCategory.VALIDATION: "Input validation failed",
            ErrorCategory.NETWORK: "Network or connection error",
            ErrorCategory.CONFIGURATION: "Configuration problem",
            ErrorCategory.INTERNAL: "Internal error occurred",
            ErrorCategory.USER_INPUT: "Invalid user input",
            ErrorCategory.UNKNOWN: "An error occurred",
        }

        base_summary = summaries.get(category, "An error occurred")

        # Try to extract specific details
        specific = self._extract_specific_info(error)
        if specific:
            return f"{base_summary}: {specific}"

        return base_summary

    def _extract_specific_info(self, error: str) -> Optional[str]:
        """Extract specific information from error message."""
        # Try to find quoted items
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", error)
        if quoted:
            return quoted[0][:50]

        # Try to find file paths
        paths = re.findall(r"[\\/][\w\\/\-_.]+\.\w+", error)
        if paths:
            return paths[0]

        # Try to find module names
        modules = re.findall(r"(?:module|package)\s+(\S+)", error, re.IGNORECASE)
        if modules:
            return modules[0]

        return None

    def _generate_explanation(
        self,
        error: str,
        category: ErrorCategory,
        context: Dict[str, Any]
    ) -> str:
        """Generate detailed explanation of the error."""
        explanations = {
            ErrorCategory.DEPENDENCY: (
                "The system cannot find a required module or package. This typically "
                "happens when a dependency hasn't been installed, the import path is "
                "wrong, or there's a version mismatch."
            ),
            ErrorCategory.FILE_ACCESS: (
                "The system cannot access a required file or directory. This could be "
                "because the path doesn't exist, is misspelled, or the working directory "
                "is different than expected."
            ),
            ErrorCategory.PERMISSION: (
                "The operation was denied due to insufficient permissions. This often "
                "occurs when trying to access protected resources or when the process "
                "doesn't have the required privileges."
            ),
            ErrorCategory.TIMEOUT: (
                "The operation took longer than the allowed time limit. This could be "
                "due to a slow external service, network issues, or an operation that "
                "is too complex."
            ),
            ErrorCategory.RESOURCE: (
                "The system has exhausted a resource (memory, disk space, etc.). This "
                "may require freeing up resources or processing data in smaller chunks."
            ),
            ErrorCategory.VALIDATION: (
                "The input data doesn't meet the required format or constraints. Check "
                "that all required fields are present and data types are correct."
            ),
            ErrorCategory.NETWORK: (
                "A network operation failed. This could be due to connectivity issues, "
                "the remote service being unavailable, or firewall restrictions."
            ),
            ErrorCategory.CONFIGURATION: (
                "There's a problem with the configuration. This could be a missing "
                "environment variable, invalid config file, or incorrect setting."
            ),
        }

        base_explanation = explanations.get(category, "An unexpected error occurred.")

        # Add context-specific details
        if context:
            task_desc = context.get("task", {}).get("description", "")
            if task_desc:
                base_explanation += f" This occurred while: {task_desc[:100]}..."

            files = context.get("files") or context.get("task", {}).get("context", {}).get("files", [])
            if files:
                base_explanation += f" Related files: {', '.join(files[:3])}"

        return base_explanation

    def _get_likely_causes(
        self,
        error: str,
        category: ErrorCategory,
        context: Dict[str, Any]
    ) -> List[str]:
        """Get likely causes based on error and context."""
        causes = list(self.CAUSE_SUGGESTIONS.get(category, []))

        # Add context-specific causes
        error_lower = error.lower()

        # Check for specific patterns
        if "not found" in error_lower and "'" in error:
            quoted = re.findall(r"'([^']+)'", error)
            if quoted:
                causes.insert(0, f"'{quoted[0]}' may be misspelled or doesn't exist")

        if "permission" in error_lower:
            worker = context.get("worker_id")
            if worker:
                causes.insert(0, f"Worker '{worker}' may not have required permissions")

        if context.get("retry_count", 0) > 0:
            causes.insert(0, "Previous attempts also failed - may be a persistent issue")

        return causes[:5]  # Limit to top 5 causes

    def _get_suggested_fixes(
        self,
        error: str,
        category: ErrorCategory,
        context: Dict[str, Any]
    ) -> List[str]:
        """Get suggested fixes based on error and context."""
        fixes = list(self.FIX_SUGGESTIONS.get(category, []))

        # Add context-specific fixes
        task = context.get("task", {})

        if task.get("retry_count", 0) >= 2:
            fixes.insert(0, "Consider reassigning to a different worker")
            fixes.insert(1, "Manual investigation may be needed")

        if task.get("dependencies"):
            fixes.append("Verify dependent tasks completed successfully")

        files = task.get("context", {}).get("files", [])
        if files:
            fixes.append(f"Check that files exist: {', '.join(files[:2])}")

        # Add recovery suggestion
        if self._is_recoverable(category, error):
            fixes.append("This error may be recoverable - consider retrying")

        return fixes[:5]  # Limit to top 5 fixes

    def _extract_related_context(
        self,
        error: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract related context for debugging."""
        related = {}

        # Extract file paths from error
        paths = re.findall(r"[\\/\w\-_.]+\.\w{1,5}", error)
        if paths:
            related["mentioned_files"] = list(set(paths))[:5]

        # Extract module/package names
        modules = re.findall(r"(?:import|require|module)\s+['\"]?(\w+)['\"]?", error, re.IGNORECASE)
        if modules:
            related["mentioned_modules"] = list(set(modules))[:5]

        # Include task context
        task = context.get("task", {})
        if task:
            related["task_id"] = task.get("id")
            related["task_status"] = task.get("status")
            related["task_priority"] = task.get("priority")

        # Include worker context
        if context.get("worker_id"):
            related["worker_id"] = context["worker_id"]

        return related

    def _is_recoverable(self, category: ErrorCategory, error: str) -> bool:
        """Determine if error is likely recoverable."""
        # Generally non-recoverable errors
        non_recoverable = {
            ErrorCategory.PERMISSION,  # Usually requires manual intervention
        }

        if category in non_recoverable:
            return False

        # Check for specific non-recoverable patterns
        error_lower = error.lower()
        non_recoverable_patterns = [
            "fatal",
            "critical",
            "corrupt",
            "unrecoverable",
            "manual intervention",
        ]

        for pattern in non_recoverable_patterns:
            if pattern in error_lower:
                return False

        # Most other errors are potentially recoverable with retry
        return True

    def _calculate_confidence(self, error: str, category: ErrorCategory) -> float:
        """Calculate confidence in the error analysis."""
        confidence = 0.5  # Base confidence

        # Higher confidence for clearly matched categories
        if category != ErrorCategory.UNKNOWN:
            confidence += 0.3

        # Higher confidence for structured error messages
        if re.search(r"Error:\s|Exception:\s|Failed:\s", error, re.IGNORECASE):
            confidence += 0.1

        # Higher confidence for errors with specific details
        if re.search(r"['\"][^'\"]+['\"]", error):
            confidence += 0.05

        return min(confidence, 0.95)

    def _record_error(
        self,
        enhanced: EnhancedError,
        context: Dict[str, Any]
    ) -> None:
        """Record error for learning and analysis."""
        self.error_history.append({
            "error": enhanced.original_error,
            "category": enhanced.category.value,
            "task_id": context.get("task", {}).get("id"),
            "worker_id": context.get("worker_id"),
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last 100 errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]

    def get_common_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most common recent errors."""
        if not self.error_history:
            return []

        # Count errors by category
        category_counts: Dict[str, int] = {}
        for err in self.error_history:
            cat = err["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Sort by count
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {"category": cat, "count": count}
            for cat, count in sorted_cats[:limit]
        ]

    def format_for_display(self, enhanced: EnhancedError) -> str:
        """Format enhanced error for display."""
        lines = [
            f"Error: {enhanced.summary}",
            "",
            f"Category: {enhanced.category.value}",
            f"Recoverable: {'Yes' if enhanced.recoverable else 'No'}",
            "",
            "Explanation:",
            enhanced.detailed_explanation,
            "",
        ]

        if enhanced.likely_causes:
            lines.append("Likely Causes:")
            for cause in enhanced.likely_causes:
                lines.append(f"  - {cause}")
            lines.append("")

        if enhanced.suggested_fixes:
            lines.append("Suggested Fixes:")
            for fix in enhanced.suggested_fixes:
                lines.append(f"  - {fix}")
            lines.append("")

        lines.append(f"Original Error: {enhanced.original_error}")

        return "\n".join(lines)
