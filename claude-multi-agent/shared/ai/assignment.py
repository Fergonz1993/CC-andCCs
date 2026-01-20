"""
Smart Worker Assignment (adv-ai-003)

Intelligently assigns tasks to workers based on:
- Historical performance data
- Worker capabilities and specializations
- Current workload and availability
- Task requirements and complexity
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import math


@dataclass
class WorkerProfile:
    """Profile of a worker's capabilities and performance."""

    worker_id: str
    capabilities: List[str] = field(default_factory=list)

    # Performance metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_time_working: float = 0.0  # seconds
    avg_task_time: float = 0.0  # seconds

    # Specialization scores (tag -> success rate)
    specialization_scores: Dict[str, float] = field(default_factory=dict)

    # Recent history
    recent_task_types: List[str] = field(default_factory=list)
    last_task_completed: Optional[datetime] = None

    # Current status
    is_available: bool = True
    current_task_id: Optional[str] = None
    current_task_started: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0  # Default to optimistic
        return self.tasks_completed / total

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (0.0-1.0)."""
        if self.tasks_completed == 0:
            return 0.5  # Neutral for new workers

        # Factor in success rate and speed
        success_factor = self.success_rate
        speed_factor = 1.0

        if self.avg_task_time > 0:
            # Faster is better, but with diminishing returns
            # Assume 30 min is baseline
            baseline_time = 30 * 60  # 30 minutes in seconds
            speed_factor = min(1.0, baseline_time / self.avg_task_time)

        return (success_factor * 0.7 + speed_factor * 0.3)


@dataclass
class AssignmentScore:
    """Score for a worker-task assignment."""

    worker_id: str
    task_id: str
    score: float  # 0.0-1.0, higher is better
    confidence: float  # 0.0-1.0
    factors: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


class SmartAssigner:
    """
    Smart task assignment using historical data and heuristics.

    Learns from past assignments to improve worker-task matching.
    """

    # Weight factors for assignment scoring
    WEIGHT_SPECIALIZATION = 0.35
    WEIGHT_SUCCESS_RATE = 0.25
    WEIGHT_AVAILABILITY = 0.20
    WEIGHT_WORKLOAD_BALANCE = 0.10
    WEIGHT_RECENCY = 0.10

    def __init__(self):
        self.worker_profiles: Dict[str, WorkerProfile] = {}
        self.assignment_history: List[Dict[str, Any]] = []

    def register_worker(
        self,
        worker_id: str,
        capabilities: Optional[List[str]] = None
    ) -> WorkerProfile:
        """Register a new worker or update existing."""
        if worker_id in self.worker_profiles:
            profile = self.worker_profiles[worker_id]
            if capabilities:
                profile.capabilities = capabilities
        else:
            profile = WorkerProfile(
                worker_id=worker_id,
                capabilities=capabilities or []
            )
            self.worker_profiles[worker_id] = profile

        return profile

    def get_worker_profile(self, worker_id: str) -> Optional[WorkerProfile]:
        """Get a worker's profile."""
        return self.worker_profiles.get(worker_id)

    def suggest_assignment(
        self,
        task: Dict[str, Any],
        available_workers: List[str]
    ) -> List[AssignmentScore]:
        """
        Suggest optimal worker(s) for a task.

        Returns list of workers sorted by assignment score (best first).
        """
        if not available_workers:
            return []

        scores = []
        for worker_id in available_workers:
            score = self._calculate_assignment_score(task, worker_id)
            scores.append(score)

        # Sort by score descending
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores

    def assign_tasks_optimally(
        self,
        tasks: List[Dict[str, Any]],
        available_workers: List[str]
    ) -> Dict[str, str]:
        """
        Assign multiple tasks to workers optimally.

        Returns mapping of task_id -> worker_id.
        Uses greedy algorithm with workload balancing.
        """
        if not tasks or not available_workers:
            return {}

        assignments = {}
        worker_load = {w: 0 for w in available_workers}

        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.get("priority", 5))

        for task in sorted_tasks:
            task_id = task.get("id", "")

            # Get scores for all available workers
            scores = self.suggest_assignment(task, available_workers)

            if not scores:
                continue

            # Adjust scores based on current load
            adjusted_scores = []
            for score in scores:
                load = worker_load.get(score.worker_id, 0)
                load_penalty = load * 0.1  # 10% penalty per existing task
                adjusted = AssignmentScore(
                    worker_id=score.worker_id,
                    task_id=task_id,
                    score=max(0, score.score - load_penalty),
                    confidence=score.confidence,
                    factors=score.factors,
                    explanation=f"{score.explanation} (load adjusted: -{load_penalty:.2f})"
                )
                adjusted_scores.append(adjusted)

            # Sort by adjusted score
            adjusted_scores.sort(key=lambda s: s.score, reverse=True)

            # Assign to best worker
            best = adjusted_scores[0]
            assignments[task_id] = best.worker_id
            worker_load[best.worker_id] = worker_load.get(best.worker_id, 0) + 1

        return assignments

    def _calculate_assignment_score(
        self,
        task: Dict[str, Any],
        worker_id: str
    ) -> AssignmentScore:
        """Calculate assignment score for a worker-task pair."""
        profile = self.worker_profiles.get(worker_id)
        task_id = task.get("id", "")

        factors = {}
        explanations = []

        # Initialize profile if new worker
        if not profile:
            profile = self.register_worker(worker_id)

        # Factor 1: Specialization match
        spec_score = self._calculate_specialization_score(task, profile)
        factors["specialization"] = spec_score
        if spec_score > 0.6:
            explanations.append(f"good specialization match ({spec_score:.2f})")

        # Factor 2: Success rate
        success_score = profile.success_rate
        factors["success_rate"] = success_score
        if success_score < 0.8:
            explanations.append(f"lower success rate ({success_score:.2f})")

        # Factor 3: Availability/readiness
        availability_score = self._calculate_availability_score(profile)
        factors["availability"] = availability_score
        if not profile.is_available:
            explanations.append("currently busy")

        # Factor 4: Workload balance (prefer workers with fewer recent tasks)
        workload_score = self._calculate_workload_score(profile)
        factors["workload"] = workload_score

        # Factor 5: Recency bonus (workers who recently completed similar tasks)
        recency_score = self._calculate_recency_score(task, profile)
        factors["recency"] = recency_score
        if recency_score > 0.7:
            explanations.append("recently worked on similar tasks")

        # Calculate weighted total
        total_score = (
            spec_score * self.WEIGHT_SPECIALIZATION +
            success_score * self.WEIGHT_SUCCESS_RATE +
            availability_score * self.WEIGHT_AVAILABILITY +
            workload_score * self.WEIGHT_WORKLOAD_BALANCE +
            recency_score * self.WEIGHT_RECENCY
        )

        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(profile)

        explanation = "; ".join(explanations) if explanations else "Good general fit"

        return AssignmentScore(
            worker_id=worker_id,
            task_id=task_id,
            score=total_score,
            confidence=confidence,
            factors=factors,
            explanation=explanation
        )

    def _calculate_specialization_score(
        self,
        task: Dict[str, Any],
        profile: WorkerProfile
    ) -> float:
        """Calculate how well a worker specializes in this task type."""
        task_tags = set(task.get("tags") or [])
        task_capabilities = set(task.get("required_capabilities") or [])

        if not task_tags and not task_capabilities:
            return 0.5  # Neutral if no task requirements

        scores = []

        # Check capability match
        if task_capabilities:
            worker_caps = set(profile.capabilities)
            if worker_caps:
                cap_overlap = len(task_capabilities & worker_caps)
                cap_score = cap_overlap / len(task_capabilities)
                scores.append(cap_score)
            else:
                scores.append(0.3)  # Unknown capabilities

        # Check specialization scores for tags
        if task_tags and profile.specialization_scores:
            tag_scores = []
            for tag in task_tags:
                if tag in profile.specialization_scores:
                    tag_scores.append(profile.specialization_scores[tag])

            if tag_scores:
                scores.append(sum(tag_scores) / len(tag_scores))

        if not scores:
            return 0.5  # Neutral

        return sum(scores) / len(scores)

    def _calculate_availability_score(self, profile: WorkerProfile) -> float:
        """Calculate availability score."""
        if not profile.is_available:
            # If busy, check how long they've been working
            if profile.current_task_started:
                elapsed = (datetime.now() - profile.current_task_started).total_seconds()
                avg_time = profile.avg_task_time or 1800  # Default 30 min

                # If almost done, partial credit
                if elapsed > avg_time * 0.8:
                    return 0.3
                return 0.0

            return 0.0

        return 1.0

    def _calculate_workload_score(self, profile: WorkerProfile) -> float:
        """Calculate workload balance score."""
        # Prefer workers who have completed fewer tasks recently
        # to balance the workload

        if profile.tasks_completed == 0:
            return 0.8  # Slight preference for new workers

        # Check recent activity
        recent_tasks = len(profile.recent_task_types)

        if recent_tasks >= 5:
            return 0.4  # Busy worker
        elif recent_tasks >= 3:
            return 0.6
        else:
            return 0.9

    def _calculate_recency_score(
        self,
        task: Dict[str, Any],
        profile: WorkerProfile
    ) -> float:
        """Calculate bonus for recent similar work."""
        task_tags = set(task.get("tags") or [])
        description_keywords = self._extract_keywords(task.get("description", ""))

        if not profile.recent_task_types:
            return 0.5  # Neutral

        # Check if recent tasks overlap with current task
        recent_tags = set()
        for task_type in profile.recent_task_types[-5:]:  # Last 5 tasks
            recent_tags.update(task_type.split("|"))

        overlap = len(task_tags & recent_tags) + len(description_keywords & recent_tags)

        if overlap >= 2:
            return 0.9
        elif overlap >= 1:
            return 0.7
        else:
            return 0.5

    def _calculate_confidence(self, profile: WorkerProfile) -> float:
        """Calculate confidence in the assignment score."""
        confidence = 0.3  # Base confidence

        # More tasks completed = higher confidence
        if profile.tasks_completed >= 10:
            confidence += 0.3
        elif profile.tasks_completed >= 5:
            confidence += 0.2
        elif profile.tasks_completed >= 1:
            confidence += 0.1

        # Specialization data available
        if profile.specialization_scores:
            confidence += 0.2

        # Recent activity
        if profile.last_task_completed:
            days_since = (datetime.now() - profile.last_task_completed).days
            if days_since < 1:
                confidence += 0.2
            elif days_since < 7:
                confidence += 0.1

        return min(confidence, 1.0)

    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text."""
        if not text:
            return set()

        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "task", "implement"
        }

        words = text.lower().split()
        return {
            w.strip(".,!?;:'\"")
            for w in words
            if len(w) > 2 and w not in stop_words
        }

    def record_completion(
        self,
        worker_id: str,
        task: Dict[str, Any],
        duration_seconds: float,
        success: bool
    ) -> None:
        """Record task completion to update worker profile."""
        profile = self.worker_profiles.get(worker_id)
        if not profile:
            profile = self.register_worker(worker_id)

        # Update metrics
        if success:
            profile.tasks_completed += 1
        else:
            profile.tasks_failed += 1

        profile.total_time_working += duration_seconds

        # Update average task time
        total_tasks = profile.tasks_completed + profile.tasks_failed
        profile.avg_task_time = profile.total_time_working / total_tasks

        profile.last_task_completed = datetime.now()
        profile.is_available = True
        profile.current_task_id = None
        profile.current_task_started = None

        # Update specialization scores
        task_tags = task.get("tags") or []
        for tag in task_tags:
            if tag not in profile.specialization_scores:
                profile.specialization_scores[tag] = 1.0 if success else 0.0
            else:
                # Exponential moving average
                alpha = 0.3
                current = profile.specialization_scores[tag]
                new_value = 1.0 if success else 0.0
                profile.specialization_scores[tag] = alpha * new_value + (1 - alpha) * current

        # Update recent task types
        task_type = "|".join(sorted(task_tags[:3])) if task_tags else "general"
        profile.recent_task_types.append(task_type)
        if len(profile.recent_task_types) > 10:
            profile.recent_task_types = profile.recent_task_types[-10:]

        # Record assignment
        self.assignment_history.append({
            "worker_id": worker_id,
            "task_id": task.get("id"),
            "task_tags": task_tags,
            "duration": duration_seconds,
            "success": success,
            "completed_at": datetime.now().isoformat()
        })

        # Keep history bounded
        if len(self.assignment_history) > 500:
            self.assignment_history = self.assignment_history[-500:]

    def mark_worker_busy(
        self,
        worker_id: str,
        task_id: str
    ) -> None:
        """Mark a worker as busy with a task."""
        profile = self.worker_profiles.get(worker_id)
        if not profile:
            profile = self.register_worker(worker_id)

        profile.is_available = False
        profile.current_task_id = task_id
        profile.current_task_started = datetime.now()

    def mark_worker_available(self, worker_id: str) -> None:
        """Mark a worker as available."""
        profile = self.worker_profiles.get(worker_id)
        if profile:
            profile.is_available = True
            profile.current_task_id = None
            profile.current_task_started = None

    def get_worker_statistics(self) -> Dict[str, Any]:
        """Get statistics about all workers."""
        if not self.worker_profiles:
            return {"workers": 0, "total_completed": 0, "total_failed": 0}

        total_completed = sum(p.tasks_completed for p in self.worker_profiles.values())
        total_failed = sum(p.tasks_failed for p in self.worker_profiles.values())
        available = sum(1 for p in self.worker_profiles.values() if p.is_available)

        best_worker = max(
            self.worker_profiles.values(),
            key=lambda p: p.efficiency_score
        )

        return {
            "workers": len(self.worker_profiles),
            "available": available,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "overall_success_rate": total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 1.0,
            "best_worker": best_worker.worker_id,
            "best_efficiency": best_worker.efficiency_score
        }
