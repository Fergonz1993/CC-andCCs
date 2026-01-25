"""
Adaptive Prioritization Learning Loop (adv-ai-012)

Learns optimal priority weights from execution outcomes using gradient descent.
Continuously improves task prioritization based on observed results.

This module provides:
- ExecutionOutcome: Records actual execution performance
- AdaptiveWeights: Learned weights that adapt over time
- AdaptivePrioritizer: Main class that calculates and learns priorities

Usage Example:
    ```python
    from shared.ai.adaptive_prioritization import (
        AdaptivePrioritizer,
        create_adaptive_prioritizer
    )

    # Create or load a prioritizer
    prioritizer = create_adaptive_prioritizer("./weights.json")

    # Calculate priority for a task
    suggestion = prioritizer.calculate_priority(
        task={"id": "task-1", "description": "Implement feature X", "priority": 5},
        all_tasks=all_tasks,
        completed_tasks=completed_tasks
    )
    print(f"Suggested priority: {suggestion.suggested_priority}")

    # After task completes, record outcome for learning
    prioritizer.record_outcome(
        task=task,
        wait_time=120.5,          # seconds waiting
        execution_time=450.0,     # actual execution seconds
        estimated_time=300.0,     # what we estimated
        success=True,
        blocked_downstream=0.0    # seconds other tasks were blocked
    )

    # Periodically save learned weights
    prioritizer.save_state("./weights.json")

    # Get insights about learned weights
    insights = prioritizer.get_weight_insights()
    print(insights)
    ```

No external ML libraries required - uses pure Python with math/statistics modules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import math
import statistics
from pathlib import Path

from .prioritization import PrioritySuggestion, PriorityFactors, TaskPrioritizer


@dataclass
class ExecutionOutcome:
    """
    Records the actual execution outcome of a task for learning.

    Attributes:
        task_id: Unique identifier of the executed task
        task_type: Derived type/category of the task
        priority_factors: The PriorityFactors used when calculating priority
        wait_time_seconds: Time the task spent waiting before execution started
        execution_time_seconds: Actual time spent executing the task
        estimated_time_seconds: What was estimated for task duration
        success: Whether the task completed successfully
        blocked_downstream_seconds: Total time downstream tasks were blocked
        recorded_at: When this outcome was recorded
    """

    task_id: str
    task_type: str
    priority_factors: Dict[str, Any]
    wait_time_seconds: float
    execution_time_seconds: float
    estimated_time_seconds: float
    success: bool
    blocked_downstream_seconds: float = 0.0
    recorded_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority_factors": self.priority_factors,
            "wait_time_seconds": self.wait_time_seconds,
            "execution_time_seconds": self.execution_time_seconds,
            "estimated_time_seconds": self.estimated_time_seconds,
            "success": self.success,
            "blocked_downstream_seconds": self.blocked_downstream_seconds,
            "recorded_at": self.recorded_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionOutcome":
        """Create from dictionary."""
        recorded_at = data.get("recorded_at")
        if isinstance(recorded_at, str):
            recorded_at = datetime.fromisoformat(recorded_at)
        else:
            recorded_at = datetime.now()

        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            priority_factors=data.get("priority_factors", {}),
            wait_time_seconds=data.get("wait_time_seconds", 0.0),
            execution_time_seconds=data.get("execution_time_seconds", 0.0),
            estimated_time_seconds=data.get("estimated_time_seconds", 0.0),
            success=data.get("success", True),
            blocked_downstream_seconds=data.get("blocked_downstream_seconds", 0.0),
            recorded_at=recorded_at
        )


@dataclass
class AdaptiveWeights:
    """
    Learned weights for priority calculation that adapt over time.

    These weights are updated through gradient descent based on execution outcomes.
    Higher weights mean more influence on the final priority score.

    Attributes:
        dependency_impact: Weight for how many tasks depend on this one (default 2.0)
        wait_time: Weight for how long a task has been waiting (default 0.5)
        critical_path: Weight for tasks on the critical execution path (default 3.0)
        success_rate: Weight for historical success rate of similar tasks (default 1.0)
        duration: Weight for preferring shorter tasks (default 0.3)
        failure_penalty: Penalty weight for tasks with failure history (default 0.5)
        contention: Weight for resource contention between tasks (default 0.5)
        learning_rate: How fast weights adapt to new data (default 0.01)
        update_count: Number of weight updates performed
    """

    dependency_impact: float = 2.0
    wait_time: float = 0.5
    critical_path: float = 3.0
    success_rate: float = 1.0
    duration: float = 0.3
    failure_penalty: float = 0.5
    contention: float = 0.5
    learning_rate: float = 0.01
    update_count: int = 0

    # Weight bounds to prevent extreme values
    _MIN_WEIGHT: float = 0.1
    _MAX_WEIGHT: float = 10.0
    _MIN_LEARNING_RATE: float = 0.001
    _MAX_LEARNING_RATE: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dependency_impact": self.dependency_impact,
            "wait_time": self.wait_time,
            "critical_path": self.critical_path,
            "success_rate": self.success_rate,
            "duration": self.duration,
            "failure_penalty": self.failure_penalty,
            "contention": self.contention,
            "learning_rate": self.learning_rate,
            "update_count": self.update_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveWeights":
        """Create from dictionary."""
        return cls(
            dependency_impact=data.get("dependency_impact", 2.0),
            wait_time=data.get("wait_time", 0.5),
            critical_path=data.get("critical_path", 3.0),
            success_rate=data.get("success_rate", 1.0),
            duration=data.get("duration", 0.3),
            failure_penalty=data.get("failure_penalty", 0.5),
            contention=data.get("contention", 0.5),
            learning_rate=data.get("learning_rate", 0.01),
            update_count=data.get("update_count", 0)
        )

    def clip_weights(self) -> None:
        """Clip all weights to reasonable ranges."""
        self.dependency_impact = max(self._MIN_WEIGHT, min(self._MAX_WEIGHT, self.dependency_impact))
        self.wait_time = max(self._MIN_WEIGHT, min(self._MAX_WEIGHT, self.wait_time))
        self.critical_path = max(self._MIN_WEIGHT, min(self._MAX_WEIGHT, self.critical_path))
        self.success_rate = max(self._MIN_WEIGHT, min(self._MAX_WEIGHT, self.success_rate))
        self.duration = max(self._MIN_WEIGHT, min(self._MAX_WEIGHT, self.duration))
        self.failure_penalty = max(self._MIN_WEIGHT, min(self._MAX_WEIGHT, self.failure_penalty))
        self.contention = max(self._MIN_WEIGHT, min(self._MAX_WEIGHT, self.contention))
        self.learning_rate = max(self._MIN_LEARNING_RATE, min(self._MAX_LEARNING_RATE, self.learning_rate))

    def get_weight_vector(self) -> List[float]:
        """Get weights as a vector for gradient operations."""
        return [
            self.dependency_impact,
            self.wait_time,
            self.critical_path,
            self.success_rate,
            self.duration,
            self.failure_penalty,
            self.contention
        ]

    def set_from_vector(self, vector: List[float]) -> None:
        """Set weights from a vector."""
        if len(vector) >= 7:
            self.dependency_impact = vector[0]
            self.wait_time = vector[1]
            self.critical_path = vector[2]
            self.success_rate = vector[3]
            self.duration = vector[4]
            self.failure_penalty = vector[5]
            self.contention = vector[6]
            self.clip_weights()


@dataclass
class WeightHistoryEntry:
    """Records weight values at a point in time for analysis."""

    weights: Dict[str, float]
    reward: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "weights": self.weights,
            "reward": self.reward,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightHistoryEntry":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        else:
            timestamp = datetime.now()

        return cls(
            weights=data.get("weights", {}),
            reward=data.get("reward", 0.0),
            timestamp=timestamp
        )


class AdaptivePrioritizer:
    """
    Adaptive task prioritizer that learns optimal weights from execution outcomes.

    This class extends the basic TaskPrioritizer with learning capabilities.
    It uses gradient descent to adjust priority weights based on observed
    execution results, continuously improving prioritization accuracy.

    The learning algorithm:
    1. Records execution outcomes (wait time, execution time, success, blocking)
    2. Computes a reward signal from outcomes
    3. Updates weights via gradient descent toward better outcomes
    4. Clips weights to reasonable ranges
    5. Tracks weight history for analysis

    Attributes:
        weights: Current adaptive weights
        outcomes: List of recorded execution outcomes
        weight_history: History of weight changes for analysis
        base_prioritizer: The underlying TaskPrioritizer for factor extraction
    """

    # Reward computation constants
    EFFICIENCY_WEIGHT = 0.4       # How much efficiency matters
    THROUGHPUT_WEIGHT = 0.3       # How much throughput matters
    BLOCKING_PENALTY_WEIGHT = 0.3 # How much blocking is penalized

    # Learning constants
    MIN_OUTCOMES_FOR_UPDATE = 5   # Minimum outcomes before updating weights
    MAX_OUTCOMES_TO_KEEP = 500    # Maximum outcomes to retain
    MAX_HISTORY_TO_KEEP = 100     # Maximum weight history entries

    def __init__(
        self,
        weights: Optional[AdaptiveWeights] = None,
        outcomes: Optional[List[ExecutionOutcome]] = None,
        weight_history: Optional[List[WeightHistoryEntry]] = None
    ):
        """
        Initialize the adaptive prioritizer.

        Args:
            weights: Initial adaptive weights (uses defaults if None)
            outcomes: Pre-existing outcomes for warm start
            weight_history: Pre-existing weight history
        """
        self.weights = weights or AdaptiveWeights()
        self.outcomes: List[ExecutionOutcome] = outcomes or []
        self.weight_history: List[WeightHistoryEntry] = weight_history or []
        self.base_prioritizer = TaskPrioritizer()

        # Apply current weights to base prioritizer
        self._apply_weights_to_base()

    def _apply_weights_to_base(self) -> None:
        """Apply current adaptive weights to the base prioritizer."""
        self.base_prioritizer.WEIGHT_DEPENDENCY_IMPACT = self.weights.dependency_impact
        self.base_prioritizer.WEIGHT_WAIT_TIME = self.weights.wait_time
        self.base_prioritizer.WEIGHT_CRITICAL_PATH = self.weights.critical_path
        self.base_prioritizer.WEIGHT_SUCCESS_RATE = self.weights.success_rate
        self.base_prioritizer.WEIGHT_DURATION = self.weights.duration
        self.base_prioritizer.WEIGHT_FAILURE_PENALTY = self.weights.failure_penalty

    def calculate_priority(
        self,
        task: Dict[str, Any],
        all_tasks: List[Dict[str, Any]],
        completed_tasks: Optional[List[Dict[str, Any]]] = None
    ) -> PrioritySuggestion:
        """
        Calculate optimal priority for a task using adaptive weights.

        Uses the learned weights to compute a priority suggestion. The weights
        are adjusted over time based on execution outcomes.

        Args:
            task: The task to prioritize
            all_tasks: All tasks in the system for context
            completed_tasks: Historical completed tasks for pattern matching

        Returns:
            PrioritySuggestion with recommended priority and explanation
        """
        # Ensure weights are applied
        self._apply_weights_to_base()

        # Get suggestion from base prioritizer
        suggestion = self.base_prioritizer.calculate_priority(
            task, all_tasks, completed_tasks
        )

        # Add adaptive weight information to explanation
        adaptive_note = f" [Adaptive: {self.weights.update_count} updates]"
        suggestion.explanation += adaptive_note

        return suggestion

    def record_outcome(
        self,
        task: Dict[str, Any],
        wait_time: float,
        execution_time: float,
        estimated_time: Optional[float] = None,
        success: bool = True,
        blocked_downstream: float = 0.0,
        priority_factors: Optional[Dict[str, Any]] = None
    ) -> ExecutionOutcome:
        """
        Record an execution outcome for learning.

        Call this after a task completes to record its actual performance.
        The prioritizer will use this data to adjust weights.

        Args:
            task: The completed task
            wait_time: Seconds the task waited before starting
            execution_time: Seconds the task took to execute
            estimated_time: What was estimated (defaults to task estimate or 1800)
            success: Whether the task succeeded
            blocked_downstream: Seconds downstream tasks were blocked
            priority_factors: The factors used when prioritizing (optional)

        Returns:
            The recorded ExecutionOutcome
        """
        task_type = self._get_task_type(task)
        est_time = estimated_time or task.get("estimated_duration", 30) * 60 or 1800

        # Extract or default priority factors
        factors_dict = priority_factors
        if factors_dict is None:
            factors = self.base_prioritizer._extract_factors(task, [], [])
            factors_dict = {
                "base_priority": factors.base_priority,
                "dependent_task_count": factors.dependent_task_count,
                "dependency_depth": factors.dependency_depth,
                "wait_time_minutes": factors.wait_time_minutes,
                "estimated_duration_minutes": factors.estimated_duration_minutes,
                "similar_task_success_rate": factors.similar_task_success_rate,
                "failure_history_count": factors.failure_history_count,
                "is_critical_path": factors.is_critical_path,
                "resource_contention": factors.resource_contention
            }

        outcome = ExecutionOutcome(
            task_id=task.get("id", "unknown"),
            task_type=task_type,
            priority_factors=factors_dict,
            wait_time_seconds=wait_time,
            execution_time_seconds=execution_time,
            estimated_time_seconds=est_time,
            success=success,
            blocked_downstream_seconds=blocked_downstream
        )

        self.outcomes.append(outcome)

        # Prune old outcomes
        if len(self.outcomes) > self.MAX_OUTCOMES_TO_KEEP:
            self.outcomes = self.outcomes[-self.MAX_OUTCOMES_TO_KEEP:]

        # Trigger weight update if we have enough data
        if len(self.outcomes) >= self.MIN_OUTCOMES_FOR_UPDATE:
            self._update_weights()

        return outcome

    def _update_weights(self) -> None:
        """
        Update weights using gradient descent based on recorded outcomes.

        The algorithm:
        1. Compute reward for recent outcomes
        2. Estimate gradient by perturbing weights
        3. Update weights in direction that improves reward
        4. Clip to valid ranges
        5. Record history
        """
        if len(self.outcomes) < self.MIN_OUTCOMES_FOR_UPDATE:
            return

        # Use recent outcomes for update
        recent_outcomes = self.outcomes[-20:]

        # Compute current reward
        current_reward = self._compute_total_reward(recent_outcomes)

        # Estimate gradients using finite differences
        gradients = self._estimate_gradients(recent_outcomes, current_reward)

        # Update weights using gradient ascent (we want to maximize reward)
        weight_vector = self.weights.get_weight_vector()
        for i, grad in enumerate(gradients):
            weight_vector[i] += self.weights.learning_rate * grad

        self.weights.set_from_vector(weight_vector)
        self.weights.update_count += 1

        # Adaptive learning rate: decrease slightly over time for stability
        if self.weights.update_count % 10 == 0:
            self.weights.learning_rate *= 0.99
            self.weights.clip_weights()

        # Apply updated weights
        self._apply_weights_to_base()

        # Record history
        self._record_history(current_reward)

    def _compute_total_reward(self, outcomes: List[ExecutionOutcome]) -> float:
        """
        Compute a total reward signal from execution outcomes.

        The reward combines:
        - Efficiency: How close execution time was to estimate
        - Throughput: Inverse of wait time (less waiting = better)
        - Blocking penalty: How much downstream work was blocked

        Args:
            outcomes: List of outcomes to evaluate

        Returns:
            Total reward value (higher is better)
        """
        if not outcomes:
            return 0.0

        total_reward = 0.0

        for outcome in outcomes:
            # Efficiency reward: ratio of estimated to actual (capped at 1.5)
            if outcome.execution_time_seconds > 0:
                efficiency_ratio = min(
                    outcome.estimated_time_seconds / outcome.execution_time_seconds,
                    1.5
                )
            else:
                efficiency_ratio = 1.0

            efficiency_reward = efficiency_ratio * (1.0 if outcome.success else 0.5)

            # Throughput reward: inverse of normalized wait time
            # Lower wait time = higher reward
            normalized_wait = outcome.wait_time_seconds / 3600.0  # Normalize to hours
            throughput_reward = 1.0 / (1.0 + normalized_wait)

            # Blocking penalty: how much downstream was blocked
            normalized_blocking = outcome.blocked_downstream_seconds / 3600.0
            blocking_penalty = normalized_blocking

            # Combine rewards
            outcome_reward = (
                self.EFFICIENCY_WEIGHT * efficiency_reward +
                self.THROUGHPUT_WEIGHT * throughput_reward -
                self.BLOCKING_PENALTY_WEIGHT * blocking_penalty
            )

            total_reward += outcome_reward

        return total_reward / len(outcomes)

    def _estimate_gradients(
        self,
        outcomes: List[ExecutionOutcome],
        current_reward: float
    ) -> List[float]:
        """
        Estimate gradients using finite differences.

        Perturbs each weight slightly and measures the change in reward
        to approximate the gradient.

        Args:
            outcomes: Outcomes to evaluate reward with
            current_reward: Current reward value

        Returns:
            List of gradient values for each weight
        """
        epsilon = 0.01  # Perturbation size
        gradients = []
        original_weights = self.weights.get_weight_vector()

        for i in range(len(original_weights)):
            # Perturb weight positively
            perturbed = original_weights.copy()
            perturbed[i] += epsilon
            self.weights.set_from_vector(perturbed)
            self._apply_weights_to_base()

            # Compute reward with perturbed weight
            # We simulate by recomputing priority impact
            perturbed_reward = self._simulate_reward_with_weights(outcomes)

            # Gradient approximation
            gradient = (perturbed_reward - current_reward) / epsilon
            gradients.append(gradient)

        # Restore original weights
        self.weights.set_from_vector(original_weights)
        self._apply_weights_to_base()

        return gradients

    def _simulate_reward_with_weights(
        self,
        outcomes: List[ExecutionOutcome]
    ) -> float:
        """
        Simulate reward computation with current weights.

        This estimates how well the current weights would have prioritized
        the tasks based on their outcomes.

        Args:
            outcomes: Outcomes to evaluate

        Returns:
            Simulated reward value
        """
        if not outcomes:
            return 0.0

        # Compute simulated reward based on how well weights match outcomes
        # Better weights should prioritize tasks that:
        # 1. Have high dependency impact (weights.dependency_impact)
        # 2. Don't wait too long (weights.wait_time)
        # 3. Are on critical path (weights.critical_path)
        # 4. Are likely to succeed (weights.success_rate)
        # 5. Are shorter (weights.duration)
        # 6. Don't have failure history (weights.failure_penalty)

        total_reward = 0.0

        for outcome in outcomes:
            factors = outcome.priority_factors

            # Compute how well the current weights would score this task
            score = 0.0

            # Dependency impact contribution
            dep_count = factors.get("dependent_task_count", 0)
            if dep_count > 0:
                score += self.weights.dependency_impact * min(dep_count, 4) / 4

            # Wait time contribution (less wait is better, so subtract)
            wait_mins = factors.get("wait_time_minutes", 0)
            if wait_mins > 5:
                score += self.weights.wait_time * min((wait_mins - 5) / 30, 1)

            # Critical path contribution
            if factors.get("is_critical_path", False):
                score += self.weights.critical_path

            # Success rate contribution
            success_rate = factors.get("similar_task_success_rate", 1.0)
            if success_rate < 1.0:
                score += self.weights.success_rate * (1 - success_rate)

            # Duration contribution (shorter tasks scored higher)
            duration = factors.get("estimated_duration_minutes", 30)
            if duration < 15:
                score += self.weights.duration

            # Failure penalty
            failures = factors.get("failure_history_count", 0)
            if failures > 0:
                score -= self.weights.failure_penalty * min(failures, 3) / 3

            # Contention penalty
            contention = factors.get("resource_contention", 0)
            if contention > 0.5:
                score -= self.weights.contention * contention

            # Reward is based on alignment between score and actual outcome
            # If high score and good outcome, reward is positive
            # If high score but bad outcome, reward is lower
            outcome_quality = (
                (1.0 if outcome.success else 0.3) *
                (outcome.estimated_time_seconds / max(outcome.execution_time_seconds, 1)) *
                (1.0 / (1.0 + outcome.blocked_downstream_seconds / 3600))
            )

            # Reward alignment
            total_reward += score * min(outcome_quality, 1.5)

        return total_reward / len(outcomes)

    def _record_history(self, reward: float) -> None:
        """Record current weights and reward to history."""
        entry = WeightHistoryEntry(
            weights=self.weights.to_dict(),
            reward=reward
        )
        self.weight_history.append(entry)

        # Prune old history
        if len(self.weight_history) > self.MAX_HISTORY_TO_KEEP:
            self.weight_history = self.weight_history[-self.MAX_HISTORY_TO_KEEP:]

    def get_weight_insights(self) -> Dict[str, Any]:
        """
        Get human-readable interpretation of learned weights.

        Provides insights about:
        - What the current weights emphasize
        - How weights have changed over time
        - Recommendations based on learning

        Returns:
            Dictionary with weight insights and interpretations
        """
        insights = {
            "current_weights": self.weights.to_dict(),
            "total_updates": self.weights.update_count,
            "total_outcomes_recorded": len(self.outcomes),
            "interpretations": [],
            "recommendations": [],
            "weight_changes": {}
        }

        # Interpret current weights
        if self.weights.dependency_impact > 2.5:
            insights["interpretations"].append(
                "High dependency_impact: The system prioritizes tasks that unblock others"
            )
        elif self.weights.dependency_impact < 1.5:
            insights["interpretations"].append(
                "Low dependency_impact: Blocking tasks are not strongly prioritized"
            )

        if self.weights.wait_time > 0.7:
            insights["interpretations"].append(
                "High wait_time weight: System avoids letting tasks wait too long"
            )
        elif self.weights.wait_time < 0.3:
            insights["interpretations"].append(
                "Low wait_time weight: Task age has little effect on priority"
            )

        if self.weights.critical_path > 4.0:
            insights["interpretations"].append(
                "High critical_path weight: Critical path tasks get strong priority"
            )

        if self.weights.failure_penalty > 0.8:
            insights["interpretations"].append(
                "High failure_penalty: Tasks with failure history are deprioritized"
            )

        if self.weights.duration > 0.5:
            insights["interpretations"].append(
                "Preference for shorter tasks to maximize throughput"
            )

        # Analyze weight history if available
        if len(self.weight_history) >= 5:
            recent = self.weight_history[-5:]
            old = self.weight_history[:5] if len(self.weight_history) >= 10 else [self.weight_history[0]]

            recent_avg = {}
            old_avg = {}

            weight_names = [
                "dependency_impact", "wait_time", "critical_path",
                "success_rate", "duration", "failure_penalty", "contention"
            ]

            for name in weight_names:
                recent_vals = [h.weights.get(name, 0) for h in recent]
                old_vals = [h.weights.get(name, 0) for h in old]

                recent_avg[name] = statistics.mean(recent_vals) if recent_vals else 0
                old_avg[name] = statistics.mean(old_vals) if old_vals else 0

                change = recent_avg[name] - old_avg[name]
                if abs(change) > 0.1:
                    direction = "increased" if change > 0 else "decreased"
                    insights["weight_changes"][name] = {
                        "direction": direction,
                        "magnitude": abs(change)
                    }

            # Reward trend
            recent_rewards = [h.reward for h in recent]
            old_rewards = [h.reward for h in old]

            if recent_rewards and old_rewards:
                reward_change = statistics.mean(recent_rewards) - statistics.mean(old_rewards)
                if reward_change > 0.1:
                    insights["recommendations"].append(
                        "Learning is improving: reward has increased"
                    )
                elif reward_change < -0.1:
                    insights["recommendations"].append(
                        "Consider resetting weights or adjusting learning rate"
                    )

        # General recommendations
        if self.weights.learning_rate < 0.005:
            insights["recommendations"].append(
                "Learning rate is very low. Consider resetting if weights seem stuck."
            )

        if len(self.outcomes) < 20:
            insights["recommendations"].append(
                "More outcome data needed for reliable weight adaptation"
            )

        return insights

    def save_state(self, path: str) -> None:
        """
        Save the prioritizer state to a JSON file.

        Saves weights, outcomes, and history for later loading.

        Args:
            path: File path to save to
        """
        state = {
            "version": "1.0.0",
            "weights": self.weights.to_dict(),
            "outcomes": [o.to_dict() for o in self.outcomes[-100:]],  # Keep last 100
            "weight_history": [h.to_dict() for h in self.weight_history[-50:]],  # Keep last 50
            "saved_at": datetime.now().isoformat()
        }

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str) -> bool:
        """
        Load prioritizer state from a JSON file.

        Args:
            path: File path to load from

        Returns:
            True if loaded successfully, False otherwise
        """
        filepath = Path(path)
        if not filepath.exists():
            return False

        try:
            with open(filepath) as f:
                state = json.load(f)

            self.weights = AdaptiveWeights.from_dict(state.get("weights", {}))
            self.outcomes = [
                ExecutionOutcome.from_dict(o)
                for o in state.get("outcomes", [])
            ]
            self.weight_history = [
                WeightHistoryEntry.from_dict(h)
                for h in state.get("weight_history", [])
            ]

            # Apply loaded weights
            self._apply_weights_to_base()

            return True

        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def reset_weights(self) -> None:
        """Reset weights to defaults while preserving history."""
        self.weight_history.append(WeightHistoryEntry(
            weights=self.weights.to_dict(),
            reward=0.0
        ))

        self.weights = AdaptiveWeights()
        self._apply_weights_to_base()

    def _get_task_type(self, task: Dict[str, Any]) -> str:
        """Derive a task type from task data for categorization."""
        tags = task.get("tags") or []
        if tags:
            return "|".join(sorted(tags[:3]))

        description = task.get("description", "")
        if description:
            # Simple keyword extraction
            words = description.lower().split()[:5]
            return "|".join(sorted(words[:3]))

        return "general"


def create_adaptive_prioritizer(
    state_path: Optional[str] = None
) -> AdaptivePrioritizer:
    """
    Factory function to create an AdaptivePrioritizer.

    If a state path is provided and exists, loads the saved state.
    Otherwise, creates a new prioritizer with default weights.

    Args:
        state_path: Optional path to load/save state from

    Returns:
        Configured AdaptivePrioritizer instance

    Example:
        ```python
        # Create new or load existing
        prioritizer = create_adaptive_prioritizer("./my_weights.json")

        # Use it
        suggestion = prioritizer.calculate_priority(task, all_tasks)

        # Record outcomes
        prioritizer.record_outcome(task, wait_time=60, execution_time=300)

        # Save periodically
        prioritizer.save_state("./my_weights.json")
        ```
    """
    prioritizer = AdaptivePrioritizer()

    if state_path:
        filepath = Path(state_path)
        if filepath.exists():
            prioritizer.load_state(state_path)

    return prioritizer
