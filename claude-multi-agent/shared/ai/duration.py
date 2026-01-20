"""
Predictive Task Duration Estimation (adv-ai-005)

Predicts how long tasks will take based on:
- Historical execution data
- Task complexity indicators
- Worker performance history
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import math


@dataclass
class DurationEstimate:
    """A duration estimate with confidence interval."""

    estimated_minutes: float
    confidence: float  # 0.0-1.0
    min_minutes: float  # Lower bound
    max_minutes: float  # Upper bound
    factors: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


class DurationPredictor:
    """
    Predicts task duration using historical data and heuristics.

    Uses simple statistical methods to estimate completion times.
    """

    # Default duration estimates by task type keywords
    KEYWORD_DURATIONS = {
        # Implementation tasks
        "implement": 45,
        "create": 30,
        "add": 25,
        "build": 40,
        "develop": 50,

        # Modification tasks
        "update": 20,
        "modify": 20,
        "change": 15,
        "refactor": 35,
        "fix": 25,

        # Testing tasks
        "test": 20,
        "write tests": 25,
        "unit test": 15,
        "integration test": 30,

        # Documentation
        "document": 15,
        "docs": 15,
        "readme": 10,
        "api docs": 20,

        # Investigation
        "investigate": 25,
        "analyze": 20,
        "review": 15,
        "debug": 30,

        # Setup/Config
        "setup": 20,
        "configure": 15,
        "install": 10,
        "deploy": 25,
    }

    # Complexity multipliers based on description indicators
    COMPLEXITY_MULTIPLIERS = {
        "complex": 1.5,
        "simple": 0.7,
        "trivial": 0.5,
        "large": 1.4,
        "small": 0.6,
        "comprehensive": 1.5,
        "full": 1.3,
        "basic": 0.7,
        "complete": 1.2,
        "partial": 0.8,
        "quick": 0.6,
        "thorough": 1.4,
    }

    # Default duration when no hints available
    DEFAULT_DURATION_MINUTES = 30

    def __init__(self):
        # Historical data: task_type -> list of (estimated, actual)
        self.historical_data: Dict[str, List[Tuple[float, float]]] = {}

        # Worker-specific adjustments
        self.worker_factors: Dict[str, float] = {}

        # Estimation accuracy tracking
        self.predictions: List[Dict[str, Any]] = []

    def estimate(
        self,
        task: Dict[str, Any],
        worker_id: Optional[str] = None
    ) -> DurationEstimate:
        """
        Estimate duration for a task.

        Args:
            task: The task to estimate
            worker_id: Optional worker for worker-specific adjustment

        Returns:
            DurationEstimate with predicted duration and confidence
        """
        factors = {}

        # Start with explicit estimate if available
        explicit_estimate = task.get("estimated_duration")
        if explicit_estimate:
            base_estimate = float(explicit_estimate)
            factors["explicit"] = base_estimate
        else:
            # Estimate from description
            base_estimate = self._estimate_from_description(task)
            factors["description_based"] = base_estimate

        # Adjust based on historical data
        historical_adjustment = self._get_historical_adjustment(task)
        if historical_adjustment != 1.0:
            base_estimate *= historical_adjustment
            factors["historical_adjustment"] = historical_adjustment

        # Adjust for complexity indicators
        complexity_factor = self._calculate_complexity_factor(task)
        if complexity_factor != 1.0:
            base_estimate *= complexity_factor
            factors["complexity_factor"] = complexity_factor

        # Adjust for files count
        file_factor = self._calculate_file_factor(task)
        if file_factor != 1.0:
            base_estimate *= file_factor
            factors["file_factor"] = file_factor

        # Adjust for dependencies (more deps = more context switching)
        dep_factor = self._calculate_dependency_factor(task)
        if dep_factor != 1.0:
            base_estimate *= dep_factor
            factors["dependency_factor"] = dep_factor

        # Worker-specific adjustment
        if worker_id:
            worker_factor = self.worker_factors.get(worker_id, 1.0)
            if worker_factor != 1.0:
                base_estimate *= worker_factor
                factors["worker_factor"] = worker_factor

        # Calculate confidence and bounds
        confidence = self._calculate_confidence(task)
        min_minutes, max_minutes = self._calculate_bounds(
            base_estimate, confidence
        )

        explanation = self._generate_explanation(factors)

        return DurationEstimate(
            estimated_minutes=round(base_estimate, 1),
            confidence=confidence,
            min_minutes=round(min_minutes, 1),
            max_minutes=round(max_minutes, 1),
            factors=factors,
            explanation=explanation
        )

    def _estimate_from_description(self, task: Dict[str, Any]) -> float:
        """Estimate duration based on task description."""
        description = task.get("description", "").lower()

        # Check for keyword matches
        best_match_duration = None
        best_match_length = 0

        for keyword, duration in self.KEYWORD_DURATIONS.items():
            if keyword in description:
                if len(keyword) > best_match_length:
                    best_match_duration = duration
                    best_match_length = len(keyword)

        if best_match_duration:
            return best_match_duration

        # Fallback: estimate based on description length
        # Longer descriptions often mean more complex tasks
        desc_length = len(description)
        if desc_length < 50:
            return 15
        elif desc_length < 100:
            return 25
        elif desc_length < 200:
            return 35
        else:
            return 45

    def _get_historical_adjustment(self, task: Dict[str, Any]) -> float:
        """Get adjustment factor based on historical accuracy."""
        task_type = self._get_task_type(task)
        history = self.historical_data.get(task_type, [])

        if len(history) < 3:
            return 1.0

        # Calculate average ratio of actual to estimated
        ratios = [actual / estimated for estimated, actual in history if estimated > 0]
        if not ratios:
            return 1.0

        avg_ratio = sum(ratios) / len(ratios)

        # Cap the adjustment to prevent extreme values
        return max(0.5, min(2.0, avg_ratio))

    def _calculate_complexity_factor(self, task: Dict[str, Any]) -> float:
        """Calculate complexity multiplier from description."""
        description = task.get("description", "").lower()
        factor = 1.0

        for keyword, multiplier in self.COMPLEXITY_MULTIPLIERS.items():
            if keyword in description:
                # Take the most significant modifier
                if abs(multiplier - 1.0) > abs(factor - 1.0):
                    factor = multiplier

        return factor

    def _calculate_file_factor(self, task: Dict[str, Any]) -> float:
        """Adjust estimate based on number of files involved."""
        files = task.get("context", {}).get("files") or []
        file_count = len(files)

        if file_count == 0:
            return 1.0
        elif file_count <= 2:
            return 1.0
        elif file_count <= 5:
            return 1.2
        elif file_count <= 10:
            return 1.4
        else:
            return 1.6

    def _calculate_dependency_factor(self, task: Dict[str, Any]) -> float:
        """Adjust estimate based on dependencies."""
        deps = task.get("dependencies") or []
        dep_count = len(deps)

        if dep_count == 0:
            return 1.0
        elif dep_count <= 2:
            return 1.1
        else:
            return 1.2

    def _calculate_confidence(self, task: Dict[str, Any]) -> float:
        """Calculate confidence in the estimate."""
        confidence = 0.4  # Base confidence

        task_type = self._get_task_type(task)
        history = self.historical_data.get(task_type, [])

        # More historical data = higher confidence
        if len(history) >= 10:
            confidence += 0.25
        elif len(history) >= 5:
            confidence += 0.15
        elif len(history) >= 1:
            confidence += 0.05

        # Explicit estimate given = higher confidence
        if task.get("estimated_duration"):
            confidence += 0.15

        # Task has clear keywords = higher confidence
        description = task.get("description", "").lower()
        for keyword in self.KEYWORD_DURATIONS:
            if keyword in description:
                confidence += 0.1
                break

        return min(confidence, 0.95)

    def _calculate_bounds(
        self,
        estimate: float,
        confidence: float
    ) -> Tuple[float, float]:
        """Calculate min/max bounds based on confidence."""
        # Higher confidence = tighter bounds
        spread = 1.0 - confidence

        min_factor = max(0.3, 1.0 - spread)
        max_factor = 1.0 + spread * 2

        return (estimate * min_factor, estimate * max_factor)

    def _generate_explanation(self, factors: Dict[str, float]) -> str:
        """Generate human-readable explanation of estimate."""
        parts = []

        if "explicit" in factors:
            parts.append(f"Base: {factors['explicit']:.0f}min (explicit)")
        elif "description_based" in factors:
            parts.append(f"Base: {factors['description_based']:.0f}min (from description)")

        if factors.get("historical_adjustment", 1.0) != 1.0:
            adj = factors["historical_adjustment"]
            direction = "longer" if adj > 1 else "shorter"
            parts.append(f"Historical: usually {adj:.1f}x {direction}")

        if factors.get("complexity_factor", 1.0) != 1.0:
            parts.append(f"Complexity: {factors['complexity_factor']:.1f}x")

        if factors.get("file_factor", 1.0) != 1.0:
            parts.append(f"Files: {factors['file_factor']:.1f}x")

        if factors.get("worker_factor", 1.0) != 1.0:
            parts.append(f"Worker speed: {factors['worker_factor']:.1f}x")

        return "; ".join(parts) if parts else "Default estimate"

    def _get_task_type(self, task: Dict[str, Any]) -> str:
        """Get task type for categorization."""
        tags = task.get("tags") or []
        if tags:
            return "|".join(sorted(tags[:3]))

        # Extract key verb from description
        description = task.get("description", "").lower()
        for keyword in self.KEYWORD_DURATIONS:
            if keyword in description:
                return keyword

        return "general"

    def record_actual_duration(
        self,
        task: Dict[str, Any],
        actual_minutes: float,
        worker_id: Optional[str] = None
    ) -> None:
        """Record actual duration to improve future estimates."""
        task_type = self._get_task_type(task)
        estimated = task.get("estimated_duration", self.DEFAULT_DURATION_MINUTES)

        # Record for task type
        if task_type not in self.historical_data:
            self.historical_data[task_type] = []

        self.historical_data[task_type].append((estimated, actual_minutes))

        # Keep only last 50 entries per type
        if len(self.historical_data[task_type]) > 50:
            self.historical_data[task_type] = self.historical_data[task_type][-50:]

        # Update worker factor
        if worker_id and estimated > 0:
            ratio = actual_minutes / estimated
            current_factor = self.worker_factors.get(worker_id, 1.0)
            # Exponential moving average
            alpha = 0.2
            self.worker_factors[worker_id] = alpha * ratio + (1 - alpha) * current_factor

        # Track prediction accuracy
        self.predictions.append({
            "task_id": task.get("id"),
            "task_type": task_type,
            "estimated": estimated,
            "actual": actual_minutes,
            "error": abs(actual_minutes - estimated),
            "recorded_at": datetime.now().isoformat()
        })

        # Keep only last 100 predictions
        if len(self.predictions) > 100:
            self.predictions = self.predictions[-100:]

    def estimate_batch_duration(
        self,
        tasks: List[Dict[str, Any]],
        parallelism: int = 1
    ) -> DurationEstimate:
        """
        Estimate total duration for a batch of tasks.

        Args:
            tasks: List of tasks
            parallelism: Number of parallel workers

        Returns:
            Total duration estimate
        """
        if not tasks:
            return DurationEstimate(
                estimated_minutes=0,
                confidence=1.0,
                min_minutes=0,
                max_minutes=0,
                explanation="No tasks"
            )

        # Get individual estimates
        estimates = [self.estimate(task) for task in tasks]

        if parallelism <= 1:
            # Sequential execution
            total = sum(e.estimated_minutes for e in estimates)
            min_total = sum(e.min_minutes for e in estimates)
            max_total = sum(e.max_minutes for e in estimates)
        else:
            # Parallel execution (simplified)
            durations = sorted([e.estimated_minutes for e in estimates], reverse=True)
            total = 0
            for i in range(0, len(durations), parallelism):
                batch = durations[i:i + parallelism]
                total += max(batch)

            # Bounds with parallelism consideration
            min_durations = sorted([e.min_minutes for e in estimates], reverse=True)
            max_durations = sorted([e.max_minutes for e in estimates], reverse=True)

            min_total = 0
            max_total = 0
            for i in range(0, len(min_durations), parallelism):
                min_batch = min_durations[i:i + parallelism]
                max_batch = max_durations[i:i + parallelism]
                min_total += max(min_batch) if min_batch else 0
                max_total += max(max_batch) if max_batch else 0

        avg_confidence = sum(e.confidence for e in estimates) / len(estimates)

        return DurationEstimate(
            estimated_minutes=round(total, 1),
            confidence=avg_confidence * 0.9,  # Slight reduction for batch
            min_minutes=round(min_total, 1),
            max_minutes=round(max_total, 1),
            explanation=f"Batch of {len(tasks)} tasks with parallelism={parallelism}"
        )

    def get_accuracy_stats(self) -> Dict[str, Any]:
        """Get statistics on prediction accuracy."""
        if not self.predictions:
            return {"predictions": 0, "accuracy": "unknown"}

        errors = [p["error"] for p in self.predictions]
        avg_error = sum(errors) / len(errors)

        # Calculate percentage error
        pct_errors = []
        for p in self.predictions:
            if p["estimated"] > 0:
                pct = abs(p["actual"] - p["estimated"]) / p["estimated"] * 100
                pct_errors.append(pct)

        avg_pct_error = sum(pct_errors) / len(pct_errors) if pct_errors else 0

        return {
            "predictions": len(self.predictions),
            "avg_error_minutes": round(avg_error, 1),
            "avg_percentage_error": round(avg_pct_error, 1),
            "accuracy_rating": self._get_accuracy_rating(avg_pct_error)
        }

    def _get_accuracy_rating(self, avg_pct_error: float) -> str:
        """Convert percentage error to rating."""
        if avg_pct_error < 15:
            return "excellent"
        elif avg_pct_error < 30:
            return "good"
        elif avg_pct_error < 50:
            return "fair"
        else:
            return "needs improvement"
