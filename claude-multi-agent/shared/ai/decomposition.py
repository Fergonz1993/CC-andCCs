"""
Automatic Task Decomposition (adv-ai-002)

Analyzes tasks and suggests how to break them down into smaller,
parallelizable subtasks using pattern matching and heuristics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import re


@dataclass
class DecompositionSuggestion:
    """A suggested way to decompose a task."""

    original_task: Dict[str, Any]
    subtasks: List[Dict[str, Any]]
    confidence: float  # 0.0 to 1.0
    reasoning: str
    estimated_parallelism: int  # How many subtasks can run in parallel
    estimated_total_duration: float  # Minutes


@dataclass
class TaskPattern:
    """Pattern for recognizing decomposable tasks."""

    name: str
    keywords: List[str]
    decomposition_strategy: str
    typical_subtasks: List[str]
    priority_modifier: int = 0


class TaskDecomposer:
    """
    Suggests task decomposition based on patterns and heuristics.

    Analyzes task descriptions to identify:
    - Complex multi-step operations
    - CRUD operations that can be parallelized
    - Testing/validation that can run concurrently
    - File operations across multiple files
    """

    # Common patterns that suggest decomposition
    PATTERNS = [
        TaskPattern(
            name="crud_implementation",
            keywords=["implement", "crud", "api", "endpoint", "rest"],
            decomposition_strategy="by_operation",
            typical_subtasks=[
                "Implement CREATE operation",
                "Implement READ operation",
                "Implement UPDATE operation",
                "Implement DELETE operation",
                "Add input validation",
                "Write tests"
            ]
        ),
        TaskPattern(
            name="multi_file_refactor",
            keywords=["refactor", "across", "multiple", "files", "codebase"],
            decomposition_strategy="by_file_group",
            typical_subtasks=[
                "Identify all affected files",
                "Update core implementation",
                "Update dependent modules",
                "Update tests",
                "Update documentation"
            ]
        ),
        TaskPattern(
            name="feature_implementation",
            keywords=["implement", "feature", "add", "new", "create"],
            decomposition_strategy="by_layer",
            typical_subtasks=[
                "Design data models",
                "Implement business logic",
                "Create API/interface layer",
                "Add validation and error handling",
                "Write unit tests",
                "Write integration tests"
            ]
        ),
        TaskPattern(
            name="testing_suite",
            keywords=["test", "testing", "coverage", "suite", "tests"],
            decomposition_strategy="by_test_type",
            typical_subtasks=[
                "Write unit tests for core functions",
                "Write integration tests",
                "Add edge case tests",
                "Test error handling",
                "Verify test coverage"
            ]
        ),
        TaskPattern(
            name="documentation",
            keywords=["document", "documentation", "docs", "readme", "api docs"],
            decomposition_strategy="by_section",
            typical_subtasks=[
                "Write overview/introduction",
                "Document API reference",
                "Add usage examples",
                "Document configuration options",
                "Add troubleshooting guide"
            ]
        ),
        TaskPattern(
            name="migration",
            keywords=["migrate", "migration", "upgrade", "convert", "port"],
            decomposition_strategy="by_phase",
            typical_subtasks=[
                "Create backup of current state",
                "Set up target environment",
                "Migrate data structures",
                "Update code references",
                "Run validation tests",
                "Document changes"
            ]
        ),
        TaskPattern(
            name="security_audit",
            keywords=["security", "audit", "vulnerability", "secure", "authentication"],
            decomposition_strategy="by_concern",
            typical_subtasks=[
                "Review authentication flow",
                "Check authorization logic",
                "Audit input validation",
                "Review data encryption",
                "Check dependency vulnerabilities",
                "Document security measures"
            ]
        ),
        TaskPattern(
            name="performance_optimization",
            keywords=["optimize", "performance", "speed", "cache", "efficient"],
            decomposition_strategy="by_area",
            typical_subtasks=[
                "Profile current performance",
                "Identify bottlenecks",
                "Optimize database queries",
                "Add caching layer",
                "Optimize algorithms",
                "Verify improvements"
            ]
        ),
        TaskPattern(
            name="bug_fix_complex",
            keywords=["fix", "bug", "issue", "error", "broken", "debug"],
            decomposition_strategy="by_step",
            typical_subtasks=[
                "Reproduce the issue",
                "Identify root cause",
                "Implement fix",
                "Add regression test",
                "Verify fix doesn't break other features"
            ]
        ),
        TaskPattern(
            name="setup_infrastructure",
            keywords=["setup", "infrastructure", "deploy", "ci/cd", "pipeline"],
            decomposition_strategy="by_component",
            typical_subtasks=[
                "Set up development environment",
                "Configure build system",
                "Set up testing pipeline",
                "Configure deployment",
                "Add monitoring/logging"
            ]
        )
    ]

    # Size indicators that suggest task is too large
    SIZE_INDICATORS = [
        (r"\band\b.*\band\b", "Multiple 'and' conjunctions suggest multiple tasks"),
        (r"(?:first|then|next|finally|after)", "Sequential steps suggest decomposition"),
        (r"(?:all|every|each)\s+\w+s?\b", "Operating on multiple items"),
        (r"\d+\s*(?:files?|components?|modules?|functions?)", "Multiple targets specified"),
        (r"(?:including|plus|also|as well as)", "Additional requirements suggest subtasks"),
    ]

    # Complexity thresholds
    DESCRIPTION_LENGTH_THRESHOLD = 200  # Characters
    FILE_COUNT_THRESHOLD = 3  # Number of files mentioned

    def __init__(self):
        self.historical_decompositions: List[Dict[str, Any]] = []

    def analyze(self, task: Dict[str, Any]) -> Optional[DecompositionSuggestion]:
        """
        Analyze a task and suggest decomposition if appropriate.

        Returns None if task doesn't need decomposition.
        """
        description = task.get("description", "")

        # Check if task is complex enough to decompose
        complexity_score, complexity_reasons = self._assess_complexity(task)

        if complexity_score < 0.5:
            return None

        # Find matching patterns
        matched_patterns = self._match_patterns(description)

        if not matched_patterns:
            # Try generic decomposition based on complexity indicators
            if complexity_score >= 0.7:
                return self._generic_decomposition(task, complexity_reasons)
            return None

        # Use best matching pattern
        best_pattern = matched_patterns[0]
        return self._apply_pattern(task, best_pattern, complexity_reasons)

    def suggest_decomposition(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> DecompositionSuggestion:
        """
        Force decomposition suggestion even for simpler tasks.

        Useful when you want decomposition suggestions regardless
        of complexity assessment.
        """
        description = task.get("description", "")

        # Try pattern matching first
        matched_patterns = self._match_patterns(description)

        if matched_patterns:
            return self._apply_pattern(task, matched_patterns[0], [])

        # Fall back to generic decomposition
        return self._generic_decomposition(task, ["Manual decomposition requested"])

    def _assess_complexity(self, task: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Assess task complexity to determine if decomposition is needed.

        Returns (score, reasons) where score is 0.0-1.0.
        """
        score = 0.0
        reasons = []

        description = task.get("description", "")

        # Check description length
        if len(description) > self.DESCRIPTION_LENGTH_THRESHOLD:
            score += 0.3
            reasons.append(f"Long description ({len(description)} chars)")

        # Check for size indicators
        for pattern, reason in self.SIZE_INDICATORS:
            if re.search(pattern, description, re.IGNORECASE):
                score += 0.15
                reasons.append(reason)

        # Check for multiple files
        files = task.get("context", {}).get("files") or []
        if len(files) > self.FILE_COUNT_THRESHOLD:
            score += 0.2
            reasons.append(f"Multiple files ({len(files)})")

        # Check for high priority with no estimated duration
        if task.get("priority", 5) <= 2 and not task.get("estimated_duration"):
            score += 0.1
            reasons.append("High priority without duration estimate")

        # Check for multiple tags suggesting scope
        tags = task.get("tags") or []
        if len(tags) >= 3:
            score += 0.1
            reasons.append(f"Multiple tags ({len(tags)})")

        return min(score, 1.0), reasons

    def _match_patterns(self, description: str) -> List[TaskPattern]:
        """Find patterns that match the task description."""
        matches = []
        description_lower = description.lower()

        for pattern in self.PATTERNS:
            keyword_matches = sum(
                1 for kw in pattern.keywords
                if kw in description_lower
            )
            if keyword_matches >= 2:
                matches.append((pattern, keyword_matches))
            elif keyword_matches == 1 and len(pattern.keywords) <= 3:
                matches.append((pattern, keyword_matches))

        # Sort by number of matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches]

    def _apply_pattern(
        self,
        task: Dict[str, Any],
        pattern: TaskPattern,
        complexity_reasons: List[str]
    ) -> DecompositionSuggestion:
        """Apply a decomposition pattern to create subtasks."""

        task_id = task.get("id", "parent")
        base_priority = task.get("priority", 5)

        subtasks = []
        for i, subtask_template in enumerate(pattern.typical_subtasks):
            subtask = {
                "description": f"{subtask_template} for: {task.get('description', '')[:50]}",
                "priority": min(10, base_priority + pattern.priority_modifier),
                "parent_id": task_id,
                "tags": (task.get("tags") or []) + [pattern.name],
                "estimated_duration": self._estimate_subtask_duration(
                    subtask_template, pattern
                )
            }

            # Set up dependencies (first subtask has no deps, others may)
            if i > 0 and self._needs_sequential_execution(pattern, i):
                subtasks.append(subtask)
            else:
                subtasks.append(subtask)

        # Calculate parallelism
        parallelism = self._estimate_parallelism(pattern, subtasks)

        # Estimate total duration
        total_duration = self._estimate_total_duration(subtasks, parallelism)

        return DecompositionSuggestion(
            original_task=task,
            subtasks=subtasks,
            confidence=0.8,  # Pattern match = higher confidence
            reasoning=f"Matched pattern '{pattern.name}': {pattern.decomposition_strategy}. "
                     f"{'; '.join(complexity_reasons) if complexity_reasons else ''}",
            estimated_parallelism=parallelism,
            estimated_total_duration=total_duration
        )

    def _generic_decomposition(
        self,
        task: Dict[str, Any],
        reasons: List[str]
    ) -> DecompositionSuggestion:
        """Create generic decomposition for tasks without matching patterns."""

        description = task.get("description", "")
        task_id = task.get("id", "parent")
        base_priority = task.get("priority", 5)

        # Try to split by conjunctions and sequential words
        parts = self._extract_task_parts(description)

        if len(parts) < 2:
            # Use generic phases
            parts = [
                "Analyze requirements and plan approach",
                "Implement core functionality",
                "Add error handling and validation",
                "Write tests",
                "Final review and cleanup"
            ]

        subtasks = []
        for i, part in enumerate(parts):
            subtask = {
                "description": part,
                "priority": base_priority,
                "parent_id": task_id,
                "tags": task.get("tags") or [],
                "estimated_duration": 15  # Default 15 minutes
            }
            subtasks.append(subtask)

        return DecompositionSuggestion(
            original_task=task,
            subtasks=subtasks,
            confidence=0.5,  # Lower confidence for generic decomposition
            reasoning=f"Generic decomposition based on: {'; '.join(reasons)}",
            estimated_parallelism=max(1, len(subtasks) // 2),
            estimated_total_duration=sum(s.get("estimated_duration", 15) for s in subtasks)
        )

    def _extract_task_parts(self, description: str) -> List[str]:
        """Extract distinct task parts from description."""
        parts = []

        # Split by common delimiters
        for delimiter in [" and then ", " then ", ", then ", "; ", " - "]:
            if delimiter in description.lower():
                raw_parts = re.split(re.escape(delimiter), description, flags=re.IGNORECASE)
                parts.extend(p.strip() for p in raw_parts if p.strip())

        if not parts:
            # Try splitting by numbered items
            numbered = re.findall(r"(?:\d+[.)]\s*)([^.!?]+)", description)
            parts.extend(numbered)

        if not parts:
            # Try splitting by bullet-like patterns
            bulleted = re.findall(r"(?:[-*]\s*)([^.!?\n]+)", description)
            parts.extend(bulleted)

        return parts[:6]  # Limit to 6 subtasks

    def _estimate_subtask_duration(
        self,
        subtask_template: str,
        pattern: TaskPattern
    ) -> int:
        """Estimate duration for a subtask in minutes."""
        # Base estimates by subtask type
        duration_hints = {
            "test": 20,
            "document": 15,
            "implement": 30,
            "create": 25,
            "update": 15,
            "verify": 10,
            "review": 10,
            "design": 20,
            "configure": 15,
            "migrate": 25,
        }

        template_lower = subtask_template.lower()
        for keyword, duration in duration_hints.items():
            if keyword in template_lower:
                return duration

        return 20  # Default 20 minutes

    def _needs_sequential_execution(self, pattern: TaskPattern, index: int) -> bool:
        """Determine if subtask at index needs to run after previous ones."""
        sequential_patterns = ["by_phase", "by_step"]
        if pattern.decomposition_strategy in sequential_patterns:
            return True

        # Some subtasks are typically sequential
        if index == len(pattern.typical_subtasks) - 1:
            # Last subtask often depends on others (e.g., "verify", "document")
            return True

        return False

    def _estimate_parallelism(
        self,
        pattern: TaskPattern,
        subtasks: List[Dict[str, Any]]
    ) -> int:
        """Estimate how many subtasks can run in parallel."""
        parallel_strategies = ["by_operation", "by_file_group", "by_test_type"]
        sequential_strategies = ["by_phase", "by_step"]

        if pattern.decomposition_strategy in parallel_strategies:
            return min(len(subtasks), 4)  # Cap at 4 parallel
        elif pattern.decomposition_strategy in sequential_strategies:
            return 1

        return max(1, len(subtasks) // 2)

    def _estimate_total_duration(
        self,
        subtasks: List[Dict[str, Any]],
        parallelism: int
    ) -> float:
        """Estimate total duration considering parallelism."""
        if parallelism <= 1:
            return sum(s.get("estimated_duration", 15) for s in subtasks)

        # Simple parallel estimation
        durations = sorted(
            [s.get("estimated_duration", 15) for s in subtasks],
            reverse=True
        )

        total = 0.0
        for i in range(0, len(durations), parallelism):
            batch = durations[i:i + parallelism]
            total += max(batch) if batch else 0

        return total

    def record_decomposition(
        self,
        original_task: Dict[str, Any],
        subtasks: List[Dict[str, Any]],
        success: bool
    ) -> None:
        """Record decomposition outcome for learning."""
        self.historical_decompositions.append({
            "original": original_task,
            "subtasks": subtasks,
            "success": success,
            "subtask_count": len(subtasks)
        })

        # Keep only last 100 entries
        if len(self.historical_decompositions) > 100:
            self.historical_decompositions = self.historical_decompositions[-100:]
