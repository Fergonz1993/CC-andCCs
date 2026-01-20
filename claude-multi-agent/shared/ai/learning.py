"""
Learning from Task Failures (adv-ai-008)

Analyzes task failures to:
- Identify patterns in failures
- Suggest preventive measures
- Improve future task planning
- Track failure trends over time
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import re


@dataclass
class FailurePattern:
    """A detected pattern in task failures."""

    pattern_id: str
    description: str
    frequency: int
    affected_task_types: List[str]
    common_causes: List[str]
    suggested_prevention: List[str]
    confidence: float
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FailureInsight:
    """Insight derived from failure analysis."""

    insight_type: str  # trend, correlation, anomaly, recommendation
    title: str
    description: str
    affected_areas: List[str]
    severity: str  # low, medium, high
    action_items: List[str]


class FailureLearner:
    """
    Learns from task failures to improve future execution.

    Uses pattern recognition and statistical analysis to:
    - Identify common failure patterns
    - Correlate failures with task characteristics
    - Generate actionable recommendations
    """

    # Minimum failures to establish a pattern
    MIN_PATTERN_FREQUENCY = 3

    # Common failure signatures
    FAILURE_SIGNATURES = {
        "timeout": {
            "patterns": [r"timeout", r"timed out", r"deadline"],
            "causes": ["Operation too slow", "Resource contention", "External dependency slow"],
            "prevention": ["Increase timeout", "Optimize operation", "Add caching"]
        },
        "dependency_missing": {
            "patterns": [r"module not found", r"import error", r"cannot resolve"],
            "causes": ["Package not installed", "Wrong import path", "Version mismatch"],
            "prevention": ["Verify dependencies", "Lock versions", "Test imports first"]
        },
        "file_not_found": {
            "patterns": [r"file not found", r"no such file", r"ENOENT"],
            "causes": ["Wrong path", "File deleted", "Working directory mismatch"],
            "prevention": ["Validate paths", "Use absolute paths", "Check file exists first"]
        },
        "permission_denied": {
            "patterns": [r"permission denied", r"access denied", r"EACCES"],
            "causes": ["Insufficient privileges", "File locked", "Protected resource"],
            "prevention": ["Check permissions upfront", "Run with correct user", "Request access"]
        },
        "validation_error": {
            "patterns": [r"validation", r"invalid", r"schema", r"type error"],
            "causes": ["Bad input data", "Schema mismatch", "Missing fields"],
            "prevention": ["Validate input early", "Add data sanitization", "Clear documentation"]
        },
        "resource_exhausted": {
            "patterns": [r"out of memory", r"disk full", r"resource", r"limit exceeded"],
            "causes": ["Too much data", "Memory leak", "Insufficient resources"],
            "prevention": ["Process in batches", "Increase limits", "Monitor resources"]
        },
        "network_error": {
            "patterns": [r"connection refused", r"network error", r"ECONNREFUSED"],
            "causes": ["Service unavailable", "Network issue", "Firewall blocking"],
            "prevention": ["Add retry logic", "Check connectivity first", "Use circuit breaker"]
        },
        "syntax_error": {
            "patterns": [r"syntax error", r"parse error", r"unexpected token"],
            "causes": ["Code error", "Invalid format", "Encoding issue"],
            "prevention": ["Lint before execution", "Validate syntax", "Use consistent encoding"]
        }
    }

    def __init__(self):
        self.failures: List[Dict[str, Any]] = []
        self.patterns: Dict[str, FailurePattern] = {}
        self.task_failure_counts: Dict[str, int] = {}  # task_type -> failure count
        self.worker_failure_counts: Dict[str, int] = {}  # worker -> failure count

    def record_failure(
        self,
        task: Dict[str, Any],
        error: str,
        worker_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[FailurePattern]:
        """
        Record a task failure and check for patterns.

        Returns detected pattern if this failure matches one.
        """
        failure = {
            "task_id": task.get("id"),
            "task_type": self._get_task_type(task),
            "description": task.get("description", "")[:200],
            "error": error,
            "worker_id": worker_id,
            "tags": task.get("tags") or [],
            "files": task.get("context", {}).get("files") or [],
            "timestamp": datetime.now(),
            "context": context or {}
        }

        self.failures.append(failure)

        # Keep only last 500 failures
        if len(self.failures) > 500:
            self.failures = self.failures[-500:]

        # Update counts
        task_type = failure["task_type"]
        self.task_failure_counts[task_type] = self.task_failure_counts.get(task_type, 0) + 1

        if worker_id:
            self.worker_failure_counts[worker_id] = self.worker_failure_counts.get(worker_id, 0) + 1

        # Check for pattern match
        pattern = self._match_failure_pattern(failure)
        if pattern:
            self._update_pattern(pattern, failure)
            return self.patterns.get(pattern)

        # Try to identify new patterns
        self._analyze_for_new_patterns()

        return None

    def get_patterns(self, min_frequency: int = 3) -> List[FailurePattern]:
        """Get all detected failure patterns."""
        return [
            p for p in self.patterns.values()
            if p.frequency >= min_frequency
        ]

    def get_insights(self) -> List[FailureInsight]:
        """Generate insights from failure data."""
        insights = []

        # Trend: Increasing failure rate
        trend_insight = self._analyze_trends()
        if trend_insight:
            insights.append(trend_insight)

        # Correlation: Task types with high failure rate
        task_insight = self._analyze_task_correlations()
        if task_insight:
            insights.append(task_insight)

        # Correlation: Workers with high failure rate
        worker_insight = self._analyze_worker_correlations()
        if worker_insight:
            insights.append(worker_insight)

        # Anomaly: Sudden failure spike
        anomaly_insight = self._detect_failure_anomaly()
        if anomaly_insight:
            insights.append(anomaly_insight)

        # Recommendations based on patterns
        recommendation = self._generate_recommendations()
        if recommendation:
            insights.append(recommendation)

        return insights

    def get_prevention_suggestions(
        self,
        task: Dict[str, Any]
    ) -> List[str]:
        """
        Get suggestions to prevent failures for a task.

        Based on historical failures of similar tasks.
        """
        suggestions = []
        task_type = self._get_task_type(task)
        task_tags = set(task.get("tags") or [])

        # Find similar failures
        similar_failures = [
            f for f in self.failures
            if f["task_type"] == task_type or (
                task_tags and task_tags & set(f.get("tags") or [])
            )
        ]

        if not similar_failures:
            return ["No similar failures recorded - proceed normally"]

        # Analyze common issues
        error_counts: Dict[str, int] = defaultdict(int)
        for f in similar_failures:
            signature = self._match_failure_pattern(f)
            if signature:
                error_counts[signature] += 1

        # Generate suggestions based on common issues
        for signature, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            sig_info = self.FAILURE_SIGNATURES.get(signature, {})
            prevention = sig_info.get("prevention", [])
            if prevention:
                suggestions.append(f"Common issue: {signature}. Prevention: {prevention[0]}")

        # Add general suggestions
        failure_rate = len(similar_failures) / max(1, self.task_failure_counts.get(task_type, 1))
        if failure_rate > 0.3:
            suggestions.append(f"High failure rate ({failure_rate*100:.0f}%) for similar tasks - extra validation recommended")

        return suggestions if suggestions else ["No specific warnings based on history"]

    def _match_failure_pattern(self, failure: Dict[str, Any]) -> Optional[str]:
        """Match failure to a known pattern signature."""
        error = failure.get("error", "").lower()

        for signature, info in self.FAILURE_SIGNATURES.items():
            for pattern in info["patterns"]:
                if re.search(pattern, error):
                    return signature

        return None

    def _update_pattern(self, signature: str, failure: Dict[str, Any]) -> None:
        """Update pattern with new failure data."""
        if signature not in self.patterns:
            sig_info = self.FAILURE_SIGNATURES.get(signature, {})
            self.patterns[signature] = FailurePattern(
                pattern_id=signature,
                description=f"Failures matching '{signature}' pattern",
                frequency=0,
                affected_task_types=[],
                common_causes=sig_info.get("causes", []),
                suggested_prevention=sig_info.get("prevention", []),
                confidence=0.7,
                examples=[]
            )

        pattern = self.patterns[signature]
        pattern.frequency += 1

        # Track affected task types
        task_type = failure["task_type"]
        if task_type not in pattern.affected_task_types:
            pattern.affected_task_types.append(task_type)

        # Keep recent examples
        pattern.examples.append({
            "task_id": failure["task_id"],
            "error": failure["error"][:100],
            "timestamp": failure["timestamp"].isoformat()
        })
        if len(pattern.examples) > 5:
            pattern.examples = pattern.examples[-5:]

        # Update confidence based on frequency
        pattern.confidence = min(0.95, 0.5 + pattern.frequency * 0.05)

    def _analyze_for_new_patterns(self) -> None:
        """Analyze recent failures for new patterns."""
        # Look for repeated error messages
        recent = self.failures[-50:] if len(self.failures) >= 50 else self.failures

        error_groups: Dict[str, List[Dict]] = defaultdict(list)
        for f in recent:
            # Normalize error for grouping
            normalized = self._normalize_error(f["error"])
            error_groups[normalized].append(f)

        # Create patterns for frequent errors
        for normalized_error, failures in error_groups.items():
            if len(failures) >= self.MIN_PATTERN_FREQUENCY:
                # Check if already matches known pattern
                if not self._match_failure_pattern(failures[0]):
                    # Create new pattern
                    pattern_id = f"custom_{len(self.patterns)}"
                    if pattern_id not in self.patterns:
                        self.patterns[pattern_id] = FailurePattern(
                            pattern_id=pattern_id,
                            description=f"Custom pattern: {normalized_error[:50]}...",
                            frequency=len(failures),
                            affected_task_types=list(set(f["task_type"] for f in failures)),
                            common_causes=["Pattern detected from recurring failures"],
                            suggested_prevention=["Investigate root cause of this recurring error"],
                            confidence=0.6,
                            examples=[{
                                "task_id": f["task_id"],
                                "error": f["error"][:100]
                            } for f in failures[-3:]]
                        )

    def _normalize_error(self, error: str) -> str:
        """Normalize error for grouping similar errors."""
        # Remove specific values like numbers, paths, IDs
        normalized = error.lower()
        normalized = re.sub(r"[\\/][\w\\/\-_.]+", "[PATH]", normalized)
        normalized = re.sub(r"0x[0-9a-f]+", "[HEX]", normalized)
        normalized = re.sub(r"\b\d+\b", "[NUM]", normalized)
        normalized = re.sub(r"['\"][^'\"]+['\"]", "[STR]", normalized)
        return normalized[:100]  # Truncate for grouping

    def _analyze_trends(self) -> Optional[FailureInsight]:
        """Analyze failure trends over time."""
        if len(self.failures) < 10:
            return None

        now = datetime.now()

        # Count failures in recent periods
        last_hour = sum(
            1 for f in self.failures
            if now - f["timestamp"] < timedelta(hours=1)
        )
        previous_hour = sum(
            1 for f in self.failures
            if timedelta(hours=1) <= now - f["timestamp"] < timedelta(hours=2)
        )

        if previous_hour > 0 and last_hour > previous_hour * 2:
            return FailureInsight(
                insight_type="trend",
                title="Increasing Failure Rate",
                description=f"Failure rate has doubled: {previous_hour} failures last hour vs {last_hour} this hour",
                affected_areas=["all"],
                severity="high",
                action_items=[
                    "Review recent changes that may have caused this",
                    "Check system health and resources",
                    "Consider pausing new task assignments"
                ]
            )

        return None

    def _analyze_task_correlations(self) -> Optional[FailureInsight]:
        """Find task types with unusually high failure rates."""
        if len(self.task_failure_counts) < 3:
            return None

        total_failures = sum(self.task_failure_counts.values())
        avg_failures = total_failures / len(self.task_failure_counts)

        problematic = [
            (task_type, count)
            for task_type, count in self.task_failure_counts.items()
            if count > avg_failures * 2 and count >= 3
        ]

        if problematic:
            worst = max(problematic, key=lambda x: x[1])
            return FailureInsight(
                insight_type="correlation",
                title="High-Failure Task Type Identified",
                description=f"Task type '{worst[0]}' has {worst[1]} failures (avg: {avg_failures:.1f})",
                affected_areas=[worst[0]],
                severity="medium",
                action_items=[
                    f"Review all '{worst[0]}' tasks for common issues",
                    "Consider decomposing complex tasks",
                    "Add extra validation for this task type"
                ]
            )

        return None

    def _analyze_worker_correlations(self) -> Optional[FailureInsight]:
        """Find workers with unusually high failure rates."""
        if len(self.worker_failure_counts) < 2:
            return None

        total_failures = sum(self.worker_failure_counts.values())
        avg_failures = total_failures / len(self.worker_failure_counts)

        problematic_workers = [
            (worker, count)
            for worker, count in self.worker_failure_counts.items()
            if count > avg_failures * 2 and count >= 3
        ]

        if problematic_workers:
            worst = max(problematic_workers, key=lambda x: x[1])
            return FailureInsight(
                insight_type="correlation",
                title="Worker Performance Issue",
                description=f"Worker '{worst[0]}' has {worst[1]} failures (avg: {avg_failures:.1f})",
                affected_areas=[worst[0]],
                severity="medium",
                action_items=[
                    f"Check worker '{worst[0]}' environment",
                    "Review task assignment for this worker",
                    "Consider restarting or replacing worker"
                ]
            )

        return None

    def _detect_failure_anomaly(self) -> Optional[FailureInsight]:
        """Detect sudden failure spikes."""
        if len(self.failures) < 20:
            return None

        now = datetime.now()

        # Count failures in 10-minute windows
        windows: Dict[int, int] = defaultdict(int)
        for f in self.failures:
            minutes_ago = int((now - f["timestamp"]).total_seconds() / 60)
            window = minutes_ago // 10  # 10-minute windows
            windows[window] += 1

        # Check for spike
        if len(windows) >= 3:
            recent = windows.get(0, 0)
            historical_avg = sum(windows.get(i, 0) for i in range(1, 6)) / 5

            if historical_avg > 0 and recent > historical_avg * 3:
                return FailureInsight(
                    insight_type="anomaly",
                    title="Sudden Failure Spike",
                    description=f"Spike detected: {recent} failures in last 10 min vs avg {historical_avg:.1f}",
                    affected_areas=["system"],
                    severity="high",
                    action_items=[
                        "Investigate immediate cause",
                        "Check for environmental changes",
                        "Consider emergency procedures"
                    ]
                )

        return None

    def _generate_recommendations(self) -> Optional[FailureInsight]:
        """Generate recommendations based on patterns."""
        if not self.patterns:
            return None

        # Find most impactful patterns
        top_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.frequency,
            reverse=True
        )[:3]

        if top_patterns[0].frequency < 3:
            return None

        actions = []
        for pattern in top_patterns:
            if pattern.suggested_prevention:
                actions.append(f"For {pattern.pattern_id}: {pattern.suggested_prevention[0]}")

        return FailureInsight(
            insight_type="recommendation",
            title="Failure Prevention Recommendations",
            description=f"Based on {sum(p.frequency for p in top_patterns)} failures from top patterns",
            affected_areas=[p.pattern_id for p in top_patterns],
            severity="medium",
            action_items=actions
        )

    def _get_task_type(self, task: Dict[str, Any]) -> str:
        """Get task type for categorization."""
        tags = task.get("tags") or []
        if tags:
            return "|".join(sorted(tags[:3]))
        return "general"

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get overall failure statistics."""
        if not self.failures:
            return {"total_failures": 0}

        now = datetime.now()

        return {
            "total_failures": len(self.failures),
            "failures_last_hour": sum(
                1 for f in self.failures
                if now - f["timestamp"] < timedelta(hours=1)
            ),
            "failures_last_24h": sum(
                1 for f in self.failures
                if now - f["timestamp"] < timedelta(hours=24)
            ),
            "patterns_detected": len(self.patterns),
            "most_common_pattern": max(
                self.patterns.values(),
                key=lambda p: p.frequency
            ).pattern_id if self.patterns else None,
            "task_types_affected": len(self.task_failure_counts),
            "workers_with_failures": len(self.worker_failure_counts)
        }
