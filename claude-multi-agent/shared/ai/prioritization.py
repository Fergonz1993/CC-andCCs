"""
Intelligent Task Prioritization (adv-ai-001)

Uses heuristics to suggest optimal task priorities based on:
- Dependencies and impact on other tasks
- Wait time and urgency
- Historical execution patterns
- Resource requirements
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import math


@dataclass
class PriorityFactors:
    """Factors that influence task priority calculation."""

    # Base priority (1-10, lower = higher priority)
    base_priority: int = 5

    # Number of tasks that depend on this task
    dependent_task_count: int = 0

    # Depth in dependency chain (tasks closer to critical path get boosted)
    dependency_depth: int = 0

    # Time waiting since creation (in minutes)
    wait_time_minutes: float = 0.0

    # Estimated duration (shorter tasks may be prioritized)
    estimated_duration_minutes: float = 30.0

    # Historical success rate for similar tasks
    similar_task_success_rate: float = 1.0

    # Number of times this task type failed before
    failure_history_count: int = 0

    # Whether this task is on the critical path
    is_critical_path: bool = False

    # Resource contention score (higher = more contention)
    resource_contention: float = 0.0


@dataclass
class PrioritySuggestion:
    """A suggested priority with explanation."""

    suggested_priority: int
    original_priority: int
    confidence: float  # 0.0 to 1.0
    factors: PriorityFactors
    explanation: str
    boost_amount: int = 0


class TaskPrioritizer:
    """
    Intelligent task prioritization using heuristics.

    This class analyzes tasks and their relationships to suggest
    optimal priorities that maximize throughput and minimize wait times.
    """

    # Weight factors for priority calculation
    WEIGHT_DEPENDENCY_IMPACT = 2.0      # High impact for tasks many others depend on
    WEIGHT_WAIT_TIME = 0.5              # Moderate boost for long-waiting tasks
    WEIGHT_CRITICAL_PATH = 3.0          # Strong boost for critical path tasks
    WEIGHT_SUCCESS_RATE = 1.0           # Boost reliable task types
    WEIGHT_DURATION = 0.3               # Slight preference for shorter tasks
    WEIGHT_FAILURE_PENALTY = 0.5        # Penalty for historically failing tasks

    # Thresholds
    WAIT_TIME_BOOST_THRESHOLD_MINUTES = 5  # Start boosting after 5 minutes
    MAX_PRIORITY_BOOST = 4                  # Maximum priority boost

    def __init__(self):
        self.task_history: List[Dict[str, Any]] = []
        self.execution_times: Dict[str, List[float]] = {}  # task_type -> list of durations

    def calculate_priority(
        self,
        task: Dict[str, Any],
        all_tasks: List[Dict[str, Any]],
        completed_tasks: Optional[List[Dict[str, Any]]] = None
    ) -> PrioritySuggestion:
        """
        Calculate optimal priority for a task.

        Args:
            task: The task to prioritize
            all_tasks: All tasks in the system
            completed_tasks: Historical completed tasks for learning

        Returns:
            PrioritySuggestion with recommended priority
        """
        factors = self._extract_factors(task, all_tasks, completed_tasks or [])

        # Calculate priority score (lower is higher priority)
        score = factors.base_priority
        boost = 0
        explanations = []

        # Factor 1: Dependency impact (tasks many others depend on)
        if factors.dependent_task_count > 0:
            dependency_boost = min(
                factors.dependent_task_count * self.WEIGHT_DEPENDENCY_IMPACT,
                self.MAX_PRIORITY_BOOST
            )
            boost += dependency_boost
            explanations.append(
                f"+{dependency_boost:.1f} boost: {factors.dependent_task_count} tasks depend on this"
            )

        # Factor 2: Wait time escalation
        if factors.wait_time_minutes > self.WAIT_TIME_BOOST_THRESHOLD_MINUTES:
            wait_boost = min(
                (factors.wait_time_minutes - self.WAIT_TIME_BOOST_THRESHOLD_MINUTES)
                * self.WEIGHT_WAIT_TIME / 10,
                self.MAX_PRIORITY_BOOST / 2
            )
            boost += wait_boost
            explanations.append(
                f"+{wait_boost:.1f} boost: waiting {factors.wait_time_minutes:.0f} minutes"
            )

        # Factor 3: Critical path
        if factors.is_critical_path:
            boost += self.WEIGHT_CRITICAL_PATH
            explanations.append(f"+{self.WEIGHT_CRITICAL_PATH} boost: on critical path")

        # Factor 4: Historical success rate
        if factors.similar_task_success_rate < 0.8:
            # Lower success rate means we might want to prioritize for visibility
            success_boost = (1.0 - factors.similar_task_success_rate) * self.WEIGHT_SUCCESS_RATE
            boost += success_boost
            explanations.append(
                f"+{success_boost:.1f} boost: similar tasks have {factors.similar_task_success_rate*100:.0f}% success"
            )

        # Factor 5: Shorter tasks (slight preference)
        if factors.estimated_duration_minutes < 15:
            duration_boost = self.WEIGHT_DURATION
            boost += duration_boost
            explanations.append(f"+{duration_boost:.1f} boost: quick task ({factors.estimated_duration_minutes:.0f}min)")

        # Factor 6: Failure history penalty
        if factors.failure_history_count > 0:
            penalty = min(
                factors.failure_history_count * self.WEIGHT_FAILURE_PENALTY,
                2.0  # Cap the penalty
            )
            boost -= penalty
            explanations.append(
                f"-{penalty:.1f} penalty: {factors.failure_history_count} previous failures"
            )

        # Factor 7: Resource contention
        if factors.resource_contention > 0.5:
            contention_penalty = factors.resource_contention * 0.5
            boost -= contention_penalty
            explanations.append(f"-{contention_penalty:.1f}: high resource contention")

        # Calculate final priority
        suggested = max(1, min(10, round(score - boost)))

        # Calculate confidence based on available data
        confidence = self._calculate_confidence(factors, completed_tasks or [])

        return PrioritySuggestion(
            suggested_priority=suggested,
            original_priority=factors.base_priority,
            confidence=confidence,
            factors=factors,
            explanation="; ".join(explanations) if explanations else "No adjustments needed",
            boost_amount=int(boost)
        )

    def _extract_factors(
        self,
        task: Dict[str, Any],
        all_tasks: List[Dict[str, Any]],
        completed_tasks: List[Dict[str, Any]]
    ) -> PriorityFactors:
        """Extract priority factors from task and context."""

        task_id = task.get("id", "")

        # Count dependent tasks
        dependent_count = sum(
            1 for t in all_tasks
            if task_id in (t.get("dependencies") or [])
        )

        # Calculate wait time
        created_at = task.get("created_at")
        wait_time = 0.0
        if created_at:
            try:
                if isinstance(created_at, str):
                    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    created = created_at
                wait_time = (datetime.now() - created.replace(tzinfo=None)).total_seconds() / 60
            except (ValueError, TypeError):
                pass

        # Calculate dependency depth
        depth = self._calculate_dependency_depth(task, all_tasks)

        # Check critical path
        is_critical = self._is_on_critical_path(task, all_tasks)

        # Get historical success rate
        success_rate = self._get_success_rate(task, completed_tasks)

        # Count failures
        failure_count = self._count_similar_failures(task, completed_tasks)

        # Estimate resource contention
        contention = self._estimate_resource_contention(task, all_tasks)

        return PriorityFactors(
            base_priority=task.get("priority", 5),
            dependent_task_count=dependent_count,
            dependency_depth=depth,
            wait_time_minutes=wait_time,
            estimated_duration_minutes=task.get("estimated_duration", 30) or 30,
            similar_task_success_rate=success_rate,
            failure_history_count=failure_count,
            is_critical_path=is_critical,
            resource_contention=contention
        )

    def _calculate_dependency_depth(
        self,
        task: Dict[str, Any],
        all_tasks: List[Dict[str, Any]],
        visited: Optional[set] = None
    ) -> int:
        """Calculate how deep this task is in the dependency chain."""
        if visited is None:
            visited = set()

        task_id = task.get("id", "")
        if task_id in visited:
            return 0
        visited.add(task_id)

        deps = task.get("dependencies") or []
        if not deps:
            return 0

        task_map = {t.get("id"): t for t in all_tasks}
        max_depth = 0

        for dep_id in deps:
            if dep_id in task_map:
                depth = self._calculate_dependency_depth(
                    task_map[dep_id], all_tasks, visited
                )
                max_depth = max(max_depth, depth + 1)

        return max_depth

    def _is_on_critical_path(
        self,
        task: Dict[str, Any],
        all_tasks: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if task is on the critical path.

        A task is on the critical path if:
        1. It has no dependencies (entry point), or
        2. Multiple tasks depend on it and they have dependencies, or
        3. It's part of the longest dependency chain
        """
        task_id = task.get("id", "")

        # Count how many tasks depend on this one
        dependents = [
            t for t in all_tasks
            if task_id in (t.get("dependencies") or [])
        ]

        # No dependencies and has dependents = entry point on critical path
        if not task.get("dependencies") and dependents:
            return True

        # Multiple dependents suggest this is a bottleneck
        if len(dependents) >= 2:
            return True

        # Check if this task has the highest depth (longest chain)
        task_depths = []
        for t in all_tasks:
            if t.get("status") in ("available", "pending"):
                depth = self._calculate_dependency_depth(t, all_tasks, set())
                task_depths.append((t.get("id"), depth))

        if task_depths:
            max_depth = max(d for _, d in task_depths)
            current_depth = self._calculate_dependency_depth(task, all_tasks, set())
            if current_depth == max_depth and max_depth > 0:
                return True

        return False

    def _get_success_rate(
        self,
        task: Dict[str, Any],
        completed_tasks: List[Dict[str, Any]]
    ) -> float:
        """Get historical success rate for similar tasks."""
        # Use tags or description keywords to find similar tasks
        task_tags = set(task.get("tags") or [])
        keywords = self._extract_keywords(task.get("description", ""))

        similar = []
        for ct in completed_tasks:
            ct_tags = set(ct.get("tags") or [])
            ct_keywords = self._extract_keywords(ct.get("description", ""))

            # Check for tag overlap or keyword overlap
            if (task_tags & ct_tags) or (keywords & ct_keywords):
                similar.append(ct)

        if not similar:
            return 1.0  # Default to optimistic

        successful = sum(1 for t in similar if t.get("status") == "done")
        return successful / len(similar)

    def _count_similar_failures(
        self,
        task: Dict[str, Any],
        completed_tasks: List[Dict[str, Any]]
    ) -> int:
        """Count how many similar tasks have failed."""
        task_tags = set(task.get("tags") or [])
        keywords = self._extract_keywords(task.get("description", ""))

        failure_count = 0
        for ct in completed_tasks:
            if ct.get("status") != "failed":
                continue
            ct_tags = set(ct.get("tags") or [])
            ct_keywords = self._extract_keywords(ct.get("description", ""))

            if (task_tags & ct_tags) or (keywords & ct_keywords):
                failure_count += 1

        return failure_count

    def _estimate_resource_contention(
        self,
        task: Dict[str, Any],
        all_tasks: List[Dict[str, Any]]
    ) -> float:
        """Estimate resource contention for this task."""
        task_files = set(task.get("context", {}).get("files") or [])
        if not task_files:
            return 0.0

        # Count other tasks touching the same files
        contention = 0
        active_tasks = [
            t for t in all_tasks
            if t.get("status") in ("available", "claimed", "in_progress")
            and t.get("id") != task.get("id")
        ]

        for t in active_tasks:
            other_files = set(t.get("context", {}).get("files") or [])
            overlap = len(task_files & other_files)
            if overlap > 0:
                contention += overlap / len(task_files)

        return min(contention, 1.0)

    def _extract_keywords(self, text: str) -> set:
        """Extract important keywords from text for similarity matching."""
        if not text:
            return set()

        # Simple keyword extraction - remove common words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "this", "that",
            "these", "those", "it", "its", "task", "implement", "create", "add"
        }

        words = text.lower().split()
        keywords = {
            w.strip(".,!?;:'\"()[]{}")
            for w in words
            if len(w) > 2 and w.lower() not in stop_words
        }

        return keywords

    def _calculate_confidence(
        self,
        factors: PriorityFactors,
        completed_tasks: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in the priority suggestion."""
        confidence = 0.5  # Base confidence

        # More historical data = higher confidence
        if len(completed_tasks) > 10:
            confidence += 0.2
        elif len(completed_tasks) > 5:
            confidence += 0.1

        # Known dependency structure = higher confidence
        if factors.dependent_task_count > 0:
            confidence += 0.1

        # Critical path determined = higher confidence
        if factors.is_critical_path:
            confidence += 0.1

        return min(confidence, 1.0)

    def prioritize_all(
        self,
        tasks: List[Dict[str, Any]],
        completed_tasks: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest priorities for all tasks and return sorted list.

        Returns tasks sorted by suggested priority (highest priority first).
        """
        completed = completed_tasks or []
        suggestions = []

        for task in tasks:
            suggestion = self.calculate_priority(task, tasks, completed)
            suggestions.append({
                "task": task,
                "suggestion": suggestion
            })

        # Sort by suggested priority (lower = higher priority)
        suggestions.sort(key=lambda x: x["suggestion"].suggested_priority)

        return suggestions

    def record_completion(
        self,
        task: Dict[str, Any],
        duration_minutes: float,
        success: bool
    ) -> None:
        """Record task completion for future learning."""
        task_type = self._get_task_type(task)

        self.task_history.append({
            "task_type": task_type,
            "duration": duration_minutes,
            "success": success,
            "priority": task.get("priority", 5),
            "completed_at": datetime.now().isoformat()
        })

        # Track execution times by task type
        if task_type not in self.execution_times:
            self.execution_times[task_type] = []
        self.execution_times[task_type].append(duration_minutes)

        # Keep only last 100 entries per type
        if len(self.execution_times[task_type]) > 100:
            self.execution_times[task_type] = self.execution_times[task_type][-100:]

    def _get_task_type(self, task: Dict[str, Any]) -> str:
        """Derive a task type from task data for categorization."""
        tags = task.get("tags") or []
        if tags:
            return "|".join(sorted(tags[:3]))

        # Use keywords from description
        keywords = self._extract_keywords(task.get("description", ""))
        if keywords:
            return "|".join(sorted(list(keywords)[:3]))

        return "general"
