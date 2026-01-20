"""
Auto-Generated Task Descriptions (adv-ai-006)

Generates improved task descriptions based on:
- Context from files
- Common patterns
- Clear and actionable language
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import re


@dataclass
class GeneratedDescription:
    """A generated or improved task description."""

    original: str
    improved: str
    confidence: float
    suggestions: List[str] = field(default_factory=list)
    added_context: Dict[str, Any] = field(default_factory=dict)


class DescriptionGenerator:
    """
    Generates and improves task descriptions.

    Uses templates and patterns to create clear, actionable descriptions.
    """

    # Templates for common task types
    TEMPLATES = {
        "implement_feature": "Implement {feature} that {purpose}. Include {requirements}. Update relevant tests.",
        "fix_bug": "Fix bug where {symptom}. Root cause: {cause}. Verify with {verification}.",
        "refactor": "Refactor {component} to improve {quality}. Ensure no functional changes and tests pass.",
        "add_tests": "Add tests for {component} covering {scenarios}. Target {coverage}% coverage.",
        "update_docs": "Update documentation for {component}. Include {sections}.",
        "configure": "Configure {component} with {settings}. Verify configuration works correctly.",
        "integrate": "Integrate {component_a} with {component_b}. Handle {edge_cases}.",
        "optimize": "Optimize {component} for {metric}. Current: {current}, target: {target}.",
        "migrate": "Migrate {what} from {source} to {target}. Ensure backward compatibility if needed.",
        "review": "Review {component} for {concerns}. Document findings and recommendations.",
    }

    # Verb normalization (map variations to standard verbs)
    VERB_MAPPINGS = {
        "code": "implement",
        "write": "implement",
        "build": "implement",
        "make": "create",
        "set up": "configure",
        "setup": "configure",
        "repair": "fix",
        "resolve": "fix",
        "address": "fix",
        "improve": "refactor",
        "clean up": "refactor",
        "cleanup": "refactor",
        "document": "update_docs",
        "docs": "update_docs",
        "test": "add_tests",
        "spec": "add_tests",
    }

    # Action verbs that indicate clear actionable descriptions
    GOOD_ACTION_VERBS = {
        "implement", "create", "add", "build", "configure", "fix", "repair",
        "update", "modify", "refactor", "optimize", "migrate", "integrate",
        "test", "verify", "validate", "document", "review", "analyze",
        "deploy", "remove", "delete", "rename", "move", "extract"
    }

    def __init__(self):
        self.improvement_history: List[Dict[str, Any]] = []

    def improve(self, description: str, context: Optional[Dict[str, Any]] = None) -> GeneratedDescription:
        """
        Improve a task description.

        Args:
            description: Original description
            context: Optional context (files, hints, etc.)

        Returns:
            GeneratedDescription with improved version
        """
        context = context or {}
        suggestions = []
        added_context = {}

        improved = description.strip()

        # Step 1: Ensure starts with action verb
        improved, verb_suggestions = self._ensure_action_verb(improved)
        suggestions.extend(verb_suggestions)

        # Step 2: Add missing details from context
        improved, context_added = self._add_context_details(improved, context)
        added_context.update(context_added)
        if context_added:
            suggestions.append("Added context from files/hints")

        # Step 3: Ensure clear and specific
        improved, specificity_suggestions = self._improve_specificity(improved)
        suggestions.extend(specificity_suggestions)

        # Step 4: Add acceptance criteria if missing
        improved, criteria_added = self._add_acceptance_criteria(improved, context)
        if criteria_added:
            suggestions.append("Added acceptance criteria")

        # Step 5: Fix common issues
        improved, fix_suggestions = self._fix_common_issues(improved)
        suggestions.extend(fix_suggestions)

        # Calculate confidence
        confidence = self._calculate_confidence(description, improved)

        return GeneratedDescription(
            original=description,
            improved=improved,
            confidence=confidence,
            suggestions=suggestions,
            added_context=added_context
        )

    def generate_from_template(
        self,
        task_type: str,
        parameters: Dict[str, str]
    ) -> str:
        """Generate a description from a template."""
        template = self.TEMPLATES.get(task_type)
        if not template:
            # Fallback to generic format
            return f"{task_type.replace('_', ' ').title()}: {', '.join(f'{k}={v}' for k, v in parameters.items())}"

        # Fill in template
        result = template
        for key, value in parameters.items():
            placeholder = "{" + key + "}"
            result = result.replace(placeholder, value)

        # Remove unfilled placeholders
        result = re.sub(r'\{[^}]+\}', '[TBD]', result)

        return result

    def suggest_improvements(self, description: str) -> List[str]:
        """Get list of suggested improvements without modifying."""
        result = self.improve(description)
        return result.suggestions

    def _ensure_action_verb(self, description: str) -> tuple[str, List[str]]:
        """Ensure description starts with an action verb."""
        suggestions = []
        words = description.split()

        if not words:
            return description, suggestions

        first_word = words[0].lower()

        # Check if already starts with good verb
        if first_word in self.GOOD_ACTION_VERBS:
            return description, suggestions

        # Check for mapped verbs
        mapped_verb = self.VERB_MAPPINGS.get(first_word)
        if mapped_verb:
            words[0] = mapped_verb.capitalize()
            return " ".join(words), [f"Normalized verb '{first_word}' to '{mapped_verb}'"]

        # Check common non-verb starts
        if first_word in {"the", "a", "an", "this", "that", "we", "need", "to"}:
            # Try to extract action from rest of description
            action_verb = self._extract_action_verb(description)
            if action_verb:
                suggestions.append(f"Consider starting with '{action_verb}'")
                return f"{action_verb.capitalize()} {description}", suggestions
            else:
                suggestions.append("Consider starting with an action verb (implement, fix, add, etc.)")

        return description, suggestions

    def _extract_action_verb(self, description: str) -> Optional[str]:
        """Try to extract an action verb from the description."""
        description_lower = description.lower()

        for verb in self.GOOD_ACTION_VERBS:
            if f" {verb} " in f" {description_lower} ":
                return verb

        # Check for verb forms
        verb_patterns = [
            (r"need(?:s)? to (\w+)", 1),
            (r"should (\w+)", 1),
            (r"must (\w+)", 1),
            (r"(?:is|are) (\w+ing)", 1),
        ]

        for pattern, group in verb_patterns:
            match = re.search(pattern, description_lower)
            if match:
                found = match.group(group).rstrip("ing")
                if found in self.GOOD_ACTION_VERBS or found + "e" in self.GOOD_ACTION_VERBS:
                    return found if found in self.GOOD_ACTION_VERBS else found + "e"

        return None

    def _add_context_details(
        self,
        description: str,
        context: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Add relevant context details to description."""
        added = {}
        improved = description

        # Add file context if not mentioned
        files = context.get("files") or []
        if files and not any(f in description.lower() for f in ["file", "module", ".py", ".ts", ".js"]):
            if len(files) <= 3:
                file_mention = f" in {', '.join(files)}"
            else:
                file_mention = f" across {len(files)} files"

            # Find good insertion point
            sentences = description.split(". ")
            if sentences:
                sentences[0] = sentences[0] + file_mention
                improved = ". ".join(sentences)
                added["files"] = files

        # Add hints if available
        hints = context.get("hints")
        if hints and hints.lower() not in description.lower():
            improved = f"{improved}. Note: {hints}"
            added["hints"] = hints

        return improved, added

    def _improve_specificity(self, description: str) -> tuple[str, List[str]]:
        """Improve specificity of vague descriptions."""
        suggestions = []

        # Check for vague terms
        vague_terms = {
            "something": "the specific component/feature",
            "stuff": "the relevant items",
            "things": "the specific elements",
            "etc": "explicitly list items",
            "some": "specify which ones",
            "various": "list the specific items",
            "few": "specify the exact number/items",
        }

        description_lower = description.lower()
        for vague, specific in vague_terms.items():
            if vague in description_lower:
                suggestions.append(f"Replace '{vague}' with {specific}")

        # Check for missing what/where
        has_what = bool(re.search(r'(?:the|a|an)\s+\w+', description_lower))
        has_where = any(
            indicator in description_lower
            for indicator in ["in ", "at ", "on ", "for ", ".py", ".ts", ".js", "file", "module"]
        )

        if not has_what:
            suggestions.append("Specify what exactly needs to be done")
        if not has_where and len(description) < 100:
            suggestions.append("Consider specifying where (file/module/component)")

        return description, suggestions

    def _add_acceptance_criteria(
        self,
        description: str,
        context: Dict[str, Any]
    ) -> tuple[str, bool]:
        """Add acceptance criteria if missing."""
        # Check if already has criteria indicators
        has_criteria = any(
            indicator in description.lower()
            for indicator in ["verify", "ensure", "test", "should", "must", "acceptance", "criteria"]
        )

        if has_criteria:
            return description, False

        # Determine appropriate criteria based on task type
        description_lower = description.lower()

        criteria = None
        if "implement" in description_lower or "create" in description_lower or "add" in description_lower:
            criteria = "Ensure tests pass and functionality works as expected."
        elif "fix" in description_lower or "bug" in description_lower:
            criteria = "Verify the fix resolves the issue and doesn't cause regressions."
        elif "refactor" in description_lower:
            criteria = "Ensure all tests pass and no functional changes occur."
        elif "test" in description_lower:
            criteria = "Verify adequate coverage and all tests pass."
        elif "document" in description_lower:
            criteria = "Verify documentation is clear and accurate."

        if criteria and len(description) < 200:
            return f"{description} {criteria}", True

        return description, False

    def _fix_common_issues(self, description: str) -> tuple[str, List[str]]:
        """Fix common description issues."""
        suggestions = []
        improved = description

        # Fix double spaces
        if "  " in improved:
            improved = re.sub(r'\s+', ' ', improved)

        # Ensure ends with period (but not for very short descriptions)
        if len(improved) > 20 and not improved[-1] in ".!?":
            improved = improved + "."

        # Capitalize first letter
        if improved and improved[0].islower():
            improved = improved[0].upper() + improved[1:]

        # Check for overly long description
        if len(improved) > 500:
            suggestions.append("Consider breaking this into multiple smaller tasks")

        # Check for unclear acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', description)
        common_acronyms = {"API", "REST", "HTTP", "JSON", "XML", "SQL", "HTML", "CSS", "JS", "TS", "UI", "UX", "DB", "CLI"}
        unknown_acronyms = set(acronyms) - common_acronyms
        if unknown_acronyms:
            suggestions.append(f"Consider expanding acronyms: {', '.join(unknown_acronyms)}")

        return improved, suggestions

    def _calculate_confidence(self, original: str, improved: str) -> float:
        """Calculate confidence in the improvement."""
        if original == improved:
            return 0.5  # No changes means either good or we couldn't improve

        confidence = 0.7  # Base confidence for improvements

        # More changes = slightly lower confidence
        changes = abs(len(improved) - len(original))
        if changes > 100:
            confidence -= 0.1
        elif changes > 50:
            confidence -= 0.05

        # Starting with action verb = higher confidence
        first_word = improved.split()[0].lower() if improved else ""
        if first_word in self.GOOD_ACTION_VERBS:
            confidence += 0.1

        # Has acceptance criteria = higher confidence
        if any(w in improved.lower() for w in ["verify", "ensure", "test"]):
            confidence += 0.1

        return min(confidence, 0.95)

    def generate_subtask_descriptions(
        self,
        parent_task: Dict[str, Any],
        subtask_templates: List[str]
    ) -> List[str]:
        """Generate descriptions for subtasks based on parent task."""
        parent_desc = parent_task.get("description", "")
        context = parent_task.get("context", {})

        descriptions = []
        for template in subtask_templates:
            # Customize template based on parent context
            desc = template

            # If template is generic, add parent context
            if "[TBD]" in desc or len(desc) < 20:
                desc = f"{desc} for: {parent_desc[:50]}..."

            # Add file context if parent has files
            files = context.get("files") or []
            if files and "file" not in desc.lower():
                relevant_files = files[:2]  # Use first 2 files
                desc = f"{desc} (files: {', '.join(relevant_files)})"

            descriptions.append(desc)

        return descriptions

    def summarize_task(self, task: Dict[str, Any], max_length: int = 50) -> str:
        """Generate a short summary of a task."""
        description = task.get("description", "")

        if len(description) <= max_length:
            return description

        # Try to get first sentence
        first_sentence = description.split(". ")[0]
        if len(first_sentence) <= max_length:
            return first_sentence + "..."

        # Truncate intelligently
        truncated = description[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.5:
            truncated = truncated[:last_space]

        return truncated + "..."
