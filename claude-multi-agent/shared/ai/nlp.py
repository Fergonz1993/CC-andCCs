"""
Natural Language Task Creation (adv-ai-010)

Parses natural language to create structured tasks:
- Extracts task descriptions, priorities, dependencies
- Interprets time-related phrases
- Identifies file references and tags
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import re


@dataclass
class ParsedTask:
    """A task parsed from natural language input."""

    description: str
    priority: int = 5
    tags: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    estimated_duration: Optional[int] = None  # minutes
    dependencies: List[str] = field(default_factory=list)
    hints: str = ""
    confidence: float = 0.7
    parsing_notes: List[str] = field(default_factory=list)


class NaturalLanguageTaskParser:
    """
    Parses natural language input to create structured tasks.

    Understands common ways people describe tasks:
    - "Fix the bug in auth.py"
    - "Implement user login with high priority"
    - "After task-123, add validation"
    """

    # Priority keywords
    PRIORITY_KEYWORDS = {
        # High priority (1-3)
        "critical": 1,
        "urgent": 1,
        "asap": 1,
        "immediately": 1,
        "blocker": 1,
        "high priority": 2,
        "important": 2,
        "must have": 2,
        "priority": 3,  # Unqualified "priority" means high-ish

        # Low priority (7-10)
        "low priority": 7,
        "nice to have": 7,
        "when possible": 8,
        "eventually": 8,
        "backlog": 9,
        "someday": 10,
    }

    # Time estimation keywords
    TIME_KEYWORDS = {
        "quick": 10,
        "quickly": 10,
        "simple": 15,
        "small": 15,
        "brief": 10,
        "short": 15,
        "medium": 30,
        "moderate": 30,
        "long": 60,
        "complex": 60,
        "large": 60,
        "extensive": 90,
        "major": 90,
    }

    # Action verbs that indicate task type
    ACTION_VERBS = {
        "implement": ["implementation", "feature"],
        "create": ["creation", "feature"],
        "add": ["addition", "feature"],
        "fix": ["bugfix", "fix"],
        "repair": ["bugfix", "fix"],
        "resolve": ["bugfix", "fix"],
        "debug": ["bugfix", "debugging"],
        "refactor": ["refactoring", "cleanup"],
        "clean": ["cleanup", "refactoring"],
        "test": ["testing", "tests"],
        "write tests": ["testing", "tests"],
        "document": ["documentation", "docs"],
        "update": ["update", "modification"],
        "modify": ["modification", "update"],
        "remove": ["removal", "cleanup"],
        "delete": ["removal", "cleanup"],
        "optimize": ["optimization", "performance"],
        "improve": ["improvement", "enhancement"],
        "migrate": ["migration"],
        "deploy": ["deployment", "devops"],
        "configure": ["configuration", "setup"],
        "setup": ["setup", "configuration"],
        "review": ["review", "analysis"],
        "analyze": ["analysis", "review"],
        "investigate": ["investigation", "analysis"],
    }

    # File extension patterns
    FILE_PATTERNS = [
        r"[\w\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|php|c|cpp|h|hpp|cs|swift|kt)\b",
        r"[\w\-]+\.(?:json|yaml|yml|xml|toml|ini|cfg|conf)\b",
        r"[\w\-]+\.(?:md|txt|rst|html|css|scss|sass)\b",
        r"(?:[\w\-]+/)+[\w\-]+\.?\w*",  # Path-like patterns
    ]

    # Dependency keywords
    DEPENDENCY_KEYWORDS = [
        r"after\s+(task[-_]?\w+)",
        r"depends?\s+on\s+(task[-_]?\w+)",
        r"following\s+(task[-_]?\w+)",
        r"once\s+(task[-_]?\w+)\s+(?:is\s+)?(?:done|complete)",
        r"requires?\s+(task[-_]?\w+)",
        r"blocked\s+by\s+(task[-_]?\w+)",
    ]

    def __init__(self):
        self.task_history: List[Dict[str, Any]] = []

    def parse(self, text: str) -> ParsedTask:
        """
        Parse natural language text into a structured task.

        Args:
            text: Natural language task description

        Returns:
            ParsedTask with extracted information
        """
        notes = []
        text = text.strip()

        # Extract priority
        priority, priority_note = self._extract_priority(text)
        if priority_note:
            notes.append(priority_note)

        # Extract duration estimate
        duration, duration_note = self._extract_duration(text)
        if duration_note:
            notes.append(duration_note)

        # Extract tags
        tags, tags_note = self._extract_tags(text)
        if tags_note:
            notes.append(tags_note)

        # Extract file references
        files, files_note = self._extract_files(text)
        if files_note:
            notes.append(files_note)

        # Extract dependencies
        dependencies, deps_note = self._extract_dependencies(text)
        if deps_note:
            notes.append(deps_note)

        # Clean and structure description
        description = self._clean_description(text)

        # Extract hints from parenthetical content
        hints, hints_note = self._extract_hints(text)
        if hints_note:
            notes.append(hints_note)

        # Calculate confidence
        confidence = self._calculate_confidence(notes, text, description)

        return ParsedTask(
            description=description,
            priority=priority,
            tags=tags,
            files=files,
            estimated_duration=duration,
            dependencies=dependencies,
            hints=hints,
            confidence=confidence,
            parsing_notes=notes
        )

    def parse_multiple(self, text: str) -> List[ParsedTask]:
        """
        Parse text that may contain multiple tasks.

        Splits on common delimiters like newlines, semicolons,
        or numbered lists.
        """
        tasks = []

        # Try splitting by numbered list
        numbered = re.split(r"\n\s*\d+[.)]\s*", text)
        if len(numbered) > 1:
            for item in numbered:
                if item.strip():
                    tasks.append(self.parse(item.strip()))
            return tasks

        # Try splitting by newlines
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) > 1:
            # Check if each line looks like a task
            if all(self._looks_like_task(l) for l in lines):
                return [self.parse(l) for l in lines]

        # Try splitting by semicolons
        parts = [p.strip() for p in text.split(";") if p.strip()]
        if len(parts) > 1:
            return [self.parse(p) for p in parts]

        # Single task
        return [self.parse(text)]

    def suggest_completion(
        self,
        partial_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Suggest completions for partial task text.

        Args:
            partial_text: Partially typed task description
            context: Optional context (existing tasks, files, etc.)

        Returns:
            List of completion suggestions
        """
        suggestions = []
        partial_lower = partial_text.lower().strip()

        # Suggest based on starting action verbs
        for verb, tags in self.ACTION_VERBS.items():
            if verb.startswith(partial_lower):
                suggestions.append(f"{verb.capitalize()} [what]")
            elif partial_lower.startswith(verb):
                # Suggest common continuations
                suggestions.append(f"{partial_text} the [component/feature]")
                suggestions.append(f"{partial_text} error handling for [feature]")

        # Suggest priority modifiers
        if not any(p in partial_lower for p in self.PRIORITY_KEYWORDS):
            if len(partial_text) > 20:
                suggestions.append(f"{partial_text} (high priority)")
                suggestions.append(f"{partial_text} (low priority)")

        # Suggest file references if context has files
        if context and context.get("files"):
            files = context["files"][:3]
            for f in files:
                if f.lower() not in partial_lower:
                    suggestions.append(f"{partial_text} in {f}")

        # Suggest dependencies if context has tasks
        if context and context.get("existing_tasks"):
            recent_tasks = context["existing_tasks"][-3:]
            for t in recent_tasks:
                task_id = t.get("id", "")
                if task_id and task_id not in partial_lower:
                    suggestions.append(f"{partial_text} after {task_id}")

        return suggestions[:5]  # Limit suggestions

    def _extract_priority(self, text: str) -> Tuple[int, Optional[str]]:
        """Extract priority from text."""
        text_lower = text.lower()

        for keyword, priority in sorted(
            self.PRIORITY_KEYWORDS.items(),
            key=lambda x: len(x[0]),
            reverse=True
        ):
            if keyword in text_lower:
                return priority, f"Priority {priority} from '{keyword}'"

        # Check for explicit priority mentions
        match = re.search(r"(?:priority|p)[:\s]*(\d+)", text_lower)
        if match:
            p = int(match.group(1))
            p = max(1, min(10, p))  # Clamp to 1-10
            return p, f"Explicit priority {p}"

        return 5, None  # Default priority

    def _extract_duration(self, text: str) -> Tuple[Optional[int], Optional[str]]:
        """Extract duration estimate from text."""
        text_lower = text.lower()

        # Check explicit time mentions
        time_patterns = [
            (r"(\d+)\s*(?:min(?:ute)?s?)", 1),  # "30 minutes"
            (r"(\d+)\s*(?:hour|hr)s?", 60),     # "2 hours"
            (r"(\d+)\s*(?:h)", 60),              # "2h"
            (r"(\d+)\s*(?:m)", 1),               # "30m"
        ]

        for pattern, multiplier in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                duration = int(match.group(1)) * multiplier
                return duration, f"Duration {duration}min from explicit mention"

        # Check time keywords
        for keyword, duration in self.TIME_KEYWORDS.items():
            if keyword in text_lower:
                return duration, f"Duration ~{duration}min from '{keyword}'"

        return None, None

    def _extract_tags(self, text: str) -> Tuple[List[str], Optional[str]]:
        """Extract tags from text."""
        tags = set()

        # Extract from action verbs
        text_lower = text.lower()
        for verb, verb_tags in self.ACTION_VERBS.items():
            if verb in text_lower:
                tags.update(verb_tags)
                break

        # Extract hashtags
        hashtags = re.findall(r"#(\w+)", text)
        tags.update(t.lower() for t in hashtags)

        # Extract bracketed tags [tag]
        bracketed = re.findall(r"\[(\w+)\]", text)
        # Filter out common non-tag brackets like [TBD], [WIP]
        actual_tags = [t.lower() for t in bracketed if len(t) > 1 and t.lower() not in {"tbd", "wip", "todo"}]
        tags.update(actual_tags)

        # Extract from "type: xyz" patterns
        type_match = re.search(r"type:\s*(\w+)", text, re.IGNORECASE)
        if type_match:
            tags.add(type_match.group(1).lower())

        tag_list = list(tags)[:5]  # Limit to 5 tags
        note = f"Tags: {', '.join(tag_list)}" if tag_list else None
        return tag_list, note

    def _extract_files(self, text: str) -> Tuple[List[str], Optional[str]]:
        """Extract file references from text."""
        files = set()

        for pattern in self.FILE_PATTERNS:
            matches = re.findall(pattern, text)
            files.update(matches)

        # Also check for quoted paths
        quoted = re.findall(r'["\']([^"\']+\.\w+)["\']', text)
        files.update(quoted)

        file_list = list(files)[:5]  # Limit to 5 files
        note = f"Files: {', '.join(file_list)}" if file_list else None
        return file_list, note

    def _extract_dependencies(self, text: str) -> Tuple[List[str], Optional[str]]:
        """Extract task dependencies from text."""
        dependencies = set()

        for pattern in self.DEPENDENCY_KEYWORDS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dependencies.update(matches)

        # Also look for bare task IDs
        task_ids = re.findall(r"\b(task[-_][\w]+)\b", text, re.IGNORECASE)
        for tid in task_ids:
            # Check if this ID is mentioned in a dependency context
            if any(kw in text.lower() for kw in ["after", "depends", "following", "once", "requires", "blocked"]):
                dependencies.add(tid)

        dep_list = list(dependencies)[:5]  # Limit to 5 dependencies
        note = f"Dependencies: {', '.join(dep_list)}" if dep_list else None
        return dep_list, note

    def _extract_hints(self, text: str) -> Tuple[str, Optional[str]]:
        """Extract hints from parenthetical content."""
        # Find content in parentheses that looks like hints
        parens = re.findall(r"\(([^)]+)\)", text)

        hints = []
        for p in parens:
            p_lower = p.lower()
            # Skip priority/time indicators
            if any(kw in p_lower for kw in ["priority", "urgent", "hour", "minute"]):
                continue
            # Keep informative content
            if len(p) > 5 and len(p) < 200:
                hints.append(p)

        # Also extract content after "Note:" or "Hint:"
        note_match = re.search(r"(?:note|hint|tip|remember):\s*(.+?)(?:\.|$)", text, re.IGNORECASE)
        if note_match:
            hints.append(note_match.group(1).strip())

        hint_text = "; ".join(hints) if hints else ""
        note = f"Hints extracted: {len(hints)}" if hints else None
        return hint_text, note

    def _clean_description(self, text: str) -> str:
        """Clean text to create a proper task description."""
        description = text

        # Remove extracted metadata
        # Remove hashtags
        description = re.sub(r"#\w+\s*", "", description)

        # Remove explicit priority mentions
        description = re.sub(r"\b(?:high|low|medium)?\s*priority:?\s*\d*\s*", "", description, flags=re.IGNORECASE)

        # Remove time estimates
        description = re.sub(r"\b\d+\s*(?:min(?:ute)?s?|hours?|h|m)\b", "", description)

        # Remove dependency clauses (but keep the main action)
        description = re.sub(r"\s*(?:after|depends?\s+on|following|once|requires?|blocked\s+by)\s+task[-_]?\w+\s*,?\s*", " ", description, flags=re.IGNORECASE)

        # Remove [tags]
        description = re.sub(r"\s*\[\w+\]\s*", " ", description)

        # Clean up whitespace
        description = re.sub(r"\s+", " ", description).strip()

        # Ensure starts with capital letter
        if description and description[0].islower():
            description = description[0].upper() + description[1:]

        # Ensure ends with period for longer descriptions
        if len(description) > 30 and description[-1] not in ".!?":
            description += "."

        return description

    def _looks_like_task(self, text: str) -> bool:
        """Check if text looks like a task description."""
        if len(text) < 5:
            return False

        text_lower = text.lower()

        # Check for action verbs
        for verb in self.ACTION_VERBS:
            if text_lower.startswith(verb) or f" {verb} " in text_lower:
                return True

        # Check for common task patterns
        task_patterns = [
            r"^[A-Z]",  # Starts with capital
            r"\b(?:the|a|an)\s+\w+",  # Has articles
            r"\.(?:py|js|ts|json)\b",  # Has file extensions
        ]

        return any(re.search(p, text) for p in task_patterns)

    def _calculate_confidence(
        self,
        notes: List[str],
        original: str,
        description: str
    ) -> float:
        """Calculate confidence in the parsing."""
        confidence = 0.5  # Base confidence

        # More notes = more data extracted = higher confidence
        confidence += min(0.2, len(notes) * 0.05)

        # Clear action verb = higher confidence
        desc_lower = description.lower()
        for verb in self.ACTION_VERBS:
            if desc_lower.startswith(verb):
                confidence += 0.15
                break

        # Reasonable description length = higher confidence
        if 10 < len(description) < 200:
            confidence += 0.1

        # Original had structure = higher confidence
        if any(c in original for c in ["#", "[", "(", ":"]):
            confidence += 0.05

        return min(confidence, 0.95)

    def to_task_dict(self, parsed: ParsedTask) -> Dict[str, Any]:
        """Convert ParsedTask to task dictionary."""
        task = {
            "description": parsed.description,
            "priority": parsed.priority,
        }

        if parsed.tags:
            task["tags"] = parsed.tags

        if parsed.files or parsed.hints:
            task["context"] = {}
            if parsed.files:
                task["context"]["files"] = parsed.files
            if parsed.hints:
                task["context"]["hints"] = parsed.hints

        if parsed.estimated_duration:
            task["estimated_duration"] = parsed.estimated_duration

        if parsed.dependencies:
            task["dependencies"] = parsed.dependencies

        return task
