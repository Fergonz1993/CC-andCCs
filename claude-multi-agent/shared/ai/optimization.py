"""
Workflow Optimization Suggestions (adv-ai-009)

Analyzes workflows and provides optimization suggestions for:
- Task ordering and parallelization
- Resource utilization
- Bottleneck identification
- Queue management
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict


@dataclass
class OptimizationSuggestion:
    """A suggested optimization for the workflow."""

    category: str  # parallelization, ordering, resources, queue, bottleneck
    title: str
    description: str
    impact: str  # low, medium, high
    effort: str  # low, medium, high
    estimated_improvement: str
    action_items: List[str] = field(default_factory=list)
    affected_tasks: List[str] = field(default_factory=list)
    confidence: float = 0.7


@dataclass
class WorkflowMetrics:
    """Metrics about the current workflow."""

    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_duration: float
    parallelism_potential: float  # 0.0-1.0
    bottleneck_score: float  # 0.0-1.0
    efficiency_score: float  # 0.0-1.0


class WorkflowOptimizer:
    """
    Analyzes workflows and suggests optimizations.

    Uses graph analysis and statistical methods to identify
    opportunities for improvement.
    """

    def __init__(self):
        self.execution_history: List[Dict[str, Any]] = []
        self.queue_history: List[Tuple[datetime, int]] = []

    def analyze(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]]
    ) -> List[OptimizationSuggestion]:
        """
        Analyze workflow and generate optimization suggestions.

        Args:
            tasks: All tasks in the workflow
            workers: Available workers

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Record queue state
        available_count = len([t for t in tasks if t.get("status") == "available"])
        self.queue_history.append((datetime.now(), available_count))
        if len(self.queue_history) > 100:
            self.queue_history = self.queue_history[-100:]

        # Analyze parallelization opportunities
        parallel_suggestions = self._analyze_parallelization(tasks, workers)
        suggestions.extend(parallel_suggestions)

        # Analyze task ordering
        ordering_suggestions = self._analyze_ordering(tasks)
        suggestions.extend(ordering_suggestions)

        # Analyze resource utilization
        resource_suggestions = self._analyze_resources(tasks, workers)
        suggestions.extend(resource_suggestions)

        # Analyze queue management
        queue_suggestions = self._analyze_queue(tasks, workers)
        suggestions.extend(queue_suggestions)

        # Analyze bottlenecks
        bottleneck_suggestions = self._analyze_bottlenecks(tasks)
        suggestions.extend(bottleneck_suggestions)

        # Sort by impact
        impact_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: impact_order.get(s.impact, 3))

        return suggestions

    def get_metrics(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]]
    ) -> WorkflowMetrics:
        """Calculate workflow metrics."""
        total = len(tasks)
        completed = len([t for t in tasks if t.get("status") == "done"])
        failed = len([t for t in tasks if t.get("status") == "failed"])

        # Calculate average duration from history
        durations = [
            e["duration"] for e in self.execution_history
            if e.get("duration")
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Calculate parallelism potential
        parallel_potential = self._calculate_parallelism_potential(tasks)

        # Calculate bottleneck score
        bottleneck_score = self._calculate_bottleneck_score(tasks)

        # Calculate efficiency
        efficiency = self._calculate_efficiency(tasks, workers)

        return WorkflowMetrics(
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            average_duration=avg_duration,
            parallelism_potential=parallel_potential,
            bottleneck_score=bottleneck_score,
            efficiency_score=efficiency
        )

    def _analyze_parallelization(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]]
    ) -> List[OptimizationSuggestion]:
        """Find opportunities to run more tasks in parallel."""
        suggestions = []

        available = [t for t in tasks if t.get("status") == "available"]
        active_workers = len([w for w in workers if w.get("is_active")])

        # Check for underutilized parallelism
        if len(available) > active_workers * 2 and active_workers > 0:
            suggestions.append(OptimizationSuggestion(
                category="parallelization",
                title="Add More Workers",
                description=f"Queue has {len(available)} available tasks but only {active_workers} workers",
                impact="high",
                effort="low",
                estimated_improvement=f"Could reduce completion time by ~{min(50, (len(available)//active_workers - 1) * 20)}%",
                action_items=[
                    f"Consider adding {min(len(available)//2, 5)} more workers",
                    "Scale based on task complexity",
                    "Monitor resource usage"
                ]
            ))

        # Find independent tasks that could be parallelized
        task_map = {t.get("id"): t for t in tasks}
        independent_groups = self._find_independent_groups(tasks)

        if len(independent_groups) > 1:
            largest_group = max(independent_groups, key=len)
            if len(largest_group) > active_workers:
                suggestions.append(OptimizationSuggestion(
                    category="parallelization",
                    title="Parallel Task Group Found",
                    description=f"Found {len(largest_group)} tasks with no dependencies that can run in parallel",
                    impact="medium",
                    effort="low",
                    estimated_improvement="Potential 2-3x speedup for this batch",
                    action_items=[
                        "Ensure workers claim from this group",
                        "These tasks can all start immediately"
                    ],
                    affected_tasks=list(largest_group)[:5]
                ))

        return suggestions

    def _analyze_ordering(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[OptimizationSuggestion]:
        """Analyze task ordering for optimization opportunities."""
        suggestions = []

        # Check for long dependency chains
        max_chain = self._find_longest_chain(tasks)
        if max_chain > 5:
            suggestions.append(OptimizationSuggestion(
                category="ordering",
                title="Long Dependency Chain",
                description=f"Found dependency chain of {max_chain} tasks - this limits parallelism",
                impact="medium",
                effort="medium",
                estimated_improvement="Could improve if some dependencies are removed",
                action_items=[
                    "Review if all dependencies are necessary",
                    "Consider breaking into parallel sub-chains",
                    "Prioritize tasks on the critical path"
                ]
            ))

        # Check priority distribution
        priorities = [t.get("priority", 5) for t in tasks if t.get("status") == "available"]
        if priorities:
            avg_priority = sum(priorities) / len(priorities)
            if avg_priority > 6:
                suggestions.append(OptimizationSuggestion(
                    category="ordering",
                    title="Low Priority Tasks Dominating Queue",
                    description=f"Average priority of available tasks is {avg_priority:.1f} (lower is higher priority)",
                    impact="low",
                    effort="low",
                    estimated_improvement="Better prioritization may improve important task completion",
                    action_items=[
                        "Review task priorities",
                        "Ensure critical tasks have priority 1-3",
                        "Consider auto-boosting stale tasks"
                    ]
                ))

        return suggestions

    def _analyze_resources(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]]
    ) -> List[OptimizationSuggestion]:
        """Analyze resource utilization."""
        suggestions = []

        # Check for idle workers
        active_workers = [w for w in workers if w.get("is_active")]
        busy_workers = [w for w in active_workers if w.get("current_task")]
        available_tasks = [t for t in tasks if t.get("status") == "available"]

        if len(busy_workers) < len(active_workers) and available_tasks:
            idle_count = len(active_workers) - len(busy_workers)
            suggestions.append(OptimizationSuggestion(
                category="resources",
                title="Idle Workers Detected",
                description=f"{idle_count} workers are idle while {len(available_tasks)} tasks are available",
                impact="high",
                effort="low",
                estimated_improvement="Immediate throughput increase possible",
                action_items=[
                    "Check if tasks have unsatisfied dependencies",
                    "Review task claiming logic",
                    "Verify workers are not stuck"
                ]
            ))

        # Check for resource contention
        file_usage = self._analyze_file_usage(tasks)
        if file_usage["conflicts"] > 0:
            suggestions.append(OptimizationSuggestion(
                category="resources",
                title="File Contention Detected",
                description=f"{file_usage['conflicts']} files are being accessed by multiple active tasks",
                impact="medium",
                effort="medium",
                estimated_improvement="Reducing conflicts could prevent race conditions",
                action_items=[
                    "Serialize tasks that access same files",
                    "Use file locking if not already",
                    "Consider task dependencies for shared files"
                ],
                affected_tasks=file_usage.get("conflicting_tasks", [])[:5]
            ))

        return suggestions

    def _analyze_queue(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]]
    ) -> List[OptimizationSuggestion]:
        """Analyze queue management."""
        suggestions = []

        available = [t for t in tasks if t.get("status") == "available"]
        in_progress = [t for t in tasks if t.get("status") == "in_progress"]

        # Check for queue buildup
        if len(self.queue_history) >= 5:
            recent_depths = [d for _, d in self.queue_history[-5:]]
            if all(recent_depths[i] <= recent_depths[i+1] for i in range(len(recent_depths)-1)):
                if recent_depths[-1] > recent_depths[0] * 1.5:
                    suggestions.append(OptimizationSuggestion(
                        category="queue",
                        title="Growing Queue Backlog",
                        description=f"Queue has grown from {recent_depths[0]} to {recent_depths[-1]} tasks",
                        impact="high",
                        effort="medium",
                        estimated_improvement="Prevent further backlog growth",
                        action_items=[
                            "Add more workers if possible",
                            "Pause adding new tasks until backlog clears",
                            "Review task completion rates"
                        ]
                    ))

        # Check for blocked tasks
        done_ids = {t.get("id") for t in tasks if t.get("status") == "done"}
        blocked = []
        for t in tasks:
            if t.get("status") == "available":
                deps = t.get("dependencies") or []
                unmet = [d for d in deps if d not in done_ids]
                if unmet:
                    blocked.append((t.get("id"), unmet))

        if len(blocked) > len(available) * 0.3 and len(blocked) > 2:
            suggestions.append(OptimizationSuggestion(
                category="queue",
                title="Many Blocked Tasks",
                description=f"{len(blocked)} tasks are waiting on dependencies",
                impact="medium",
                effort="low",
                estimated_improvement="Prioritizing blockers could unblock these tasks",
                action_items=[
                    "Prioritize tasks that unblock others",
                    "Check for circular dependencies",
                    "Consider removing non-essential dependencies"
                ]
            ))

        return suggestions

    def _analyze_bottlenecks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[OptimizationSuggestion]:
        """Identify workflow bottlenecks."""
        suggestions = []

        # Find tasks that block many others
        blocker_counts = defaultdict(int)
        task_map = {t.get("id"): t for t in tasks}

        for t in tasks:
            for dep in (t.get("dependencies") or []):
                blocker_counts[dep] += 1

        # Find biggest blockers
        major_blockers = [
            (task_id, count)
            for task_id, count in blocker_counts.items()
            if count >= 3 and task_id in task_map
        ]

        if major_blockers:
            worst = max(major_blockers, key=lambda x: x[1])
            blocker_task = task_map[worst[0]]
            suggestions.append(OptimizationSuggestion(
                category="bottleneck",
                title="Bottleneck Task Identified",
                description=f"Task '{worst[0]}' blocks {worst[1]} other tasks",
                impact="high",
                effort="low",
                estimated_improvement=f"Completing this task will unblock {worst[1]} tasks",
                action_items=[
                    f"Prioritize task {worst[0]} (current: P{blocker_task.get('priority', 5)})",
                    "Assign to most capable worker",
                    "Consider decomposing if complex"
                ],
                affected_tasks=[worst[0]]
            ))

        # Check for slow task types
        if self.execution_history:
            type_durations: Dict[str, List[float]] = defaultdict(list)
            for e in self.execution_history:
                if e.get("duration"):
                    type_durations[e.get("task_type", "general")].append(e["duration"])

            slow_types = [
                (task_type, sum(durs)/len(durs))
                for task_type, durs in type_durations.items()
                if len(durs) >= 3
            ]

            if slow_types:
                slowest = max(slow_types, key=lambda x: x[1])
                avg_overall = sum(
                    sum(durs)/len(durs) for durs in type_durations.values()
                ) / len(type_durations)

                if slowest[1] > avg_overall * 2:
                    suggestions.append(OptimizationSuggestion(
                        category="bottleneck",
                        title="Slow Task Type Identified",
                        description=f"Task type '{slowest[0]}' averages {slowest[1]/60:.1f}min (overall avg: {avg_overall/60:.1f}min)",
                        impact="medium",
                        effort="medium",
                        estimated_improvement="Optimizing these tasks could significantly reduce total time",
                        action_items=[
                            f"Review '{slowest[0]}' tasks for optimization",
                            "Consider breaking into smaller tasks",
                            "Check for inefficient patterns"
                        ]
                    ))

        return suggestions

    def _find_independent_groups(self, tasks: List[Dict[str, Any]]) -> List[Set[str]]:
        """Find groups of tasks that can run independently."""
        # Build dependency graph
        task_ids = {t.get("id") for t in tasks}
        task_deps: Dict[str, Set[str]] = {}

        for t in tasks:
            task_deps[t.get("id", "")] = set(t.get("dependencies") or [])

        # Find tasks with no dependencies
        independent = {
            tid for tid, deps in task_deps.items()
            if not deps or not (deps & task_ids)
        }

        # Group remaining by shared dependencies
        groups = []
        if independent:
            groups.append(independent)

        # Remaining tasks form other groups based on dependency patterns
        remaining = task_ids - independent
        while remaining:
            # Start new group with first remaining task
            first = next(iter(remaining))
            group = {first}
            remaining.remove(first)

            # Add tasks with similar dependencies
            for tid in list(remaining):
                if task_deps.get(tid, set()) == task_deps.get(first, set()):
                    group.add(tid)
                    remaining.remove(tid)

            groups.append(group)

        return groups

    def _find_longest_chain(self, tasks: List[Dict[str, Any]]) -> int:
        """Find the longest dependency chain."""
        task_map = {t.get("id"): t for t in tasks}

        def chain_length(task_id: str, visited: Set[str]) -> int:
            if task_id in visited or task_id not in task_map:
                return 0
            visited.add(task_id)

            task = task_map[task_id]
            deps = task.get("dependencies") or []
            if not deps:
                return 1

            max_dep_length = max(
                chain_length(dep, visited.copy())
                for dep in deps
            )
            return 1 + max_dep_length

        max_chain = 0
        for t in tasks:
            chain = chain_length(t.get("id", ""), set())
            max_chain = max(max_chain, chain)

        return max_chain

    def _analyze_file_usage(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze file usage across tasks."""
        in_progress = [t for t in tasks if t.get("status") == "in_progress"]

        file_usage: Dict[str, List[str]] = defaultdict(list)
        for t in in_progress:
            files = t.get("context", {}).get("files") or []
            for f in files:
                file_usage[f].append(t.get("id", "unknown"))

        conflicts = {f: task_ids for f, task_ids in file_usage.items() if len(task_ids) > 1}
        conflicting_tasks = set()
        for task_ids in conflicts.values():
            conflicting_tasks.update(task_ids)

        return {
            "conflicts": len(conflicts),
            "conflicting_files": list(conflicts.keys())[:5],
            "conflicting_tasks": list(conflicting_tasks)[:5]
        }

    def _calculate_parallelism_potential(self, tasks: List[Dict[str, Any]]) -> float:
        """Calculate how parallelizable the workflow is."""
        if not tasks:
            return 0.0

        groups = self._find_independent_groups(tasks)
        if not groups:
            return 0.0

        # Ratio of largest independent group to total tasks
        largest = max(len(g) for g in groups)
        return min(1.0, largest / len(tasks))

    def _calculate_bottleneck_score(self, tasks: List[Dict[str, Any]]) -> float:
        """Calculate how bottlenecked the workflow is (0 = not, 1 = severely)."""
        if not tasks:
            return 0.0

        # Count tasks that block multiple others
        blocker_counts = defaultdict(int)
        for t in tasks:
            for dep in (t.get("dependencies") or []):
                blocker_counts[dep] += 1

        if not blocker_counts:
            return 0.0

        # High bottleneck if few tasks block many
        max_blocked = max(blocker_counts.values())
        return min(1.0, max_blocked / (len(tasks) / 2))

    def _calculate_efficiency(
        self,
        tasks: List[Dict[str, Any]],
        workers: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall efficiency (0-1)."""
        if not workers or not tasks:
            return 0.5

        active = [w for w in workers if w.get("is_active")]
        if not active:
            return 0.0

        busy = [w for w in active if w.get("current_task")]
        available = [t for t in tasks if t.get("status") == "available"]

        # Worker utilization
        utilization = len(busy) / len(active) if active else 0

        # Task throughput (tasks not blocked)
        done_ids = {t.get("id") for t in tasks if t.get("status") == "done"}
        unblocked = [
            t for t in available
            if all(d in done_ids for d in (t.get("dependencies") or []))
        ]
        throughput = len(unblocked) / max(1, len(available)) if available else 1.0

        return (utilization + throughput) / 2

    def record_execution(
        self,
        task: Dict[str, Any],
        duration_seconds: float,
        success: bool
    ) -> None:
        """Record task execution for analysis."""
        self.execution_history.append({
            "task_id": task.get("id"),
            "task_type": "|".join(sorted((task.get("tags") or [])[:3])) or "general",
            "duration": duration_seconds,
            "success": success,
            "timestamp": datetime.now()
        })

        # Keep only last 200 executions
        if len(self.execution_history) > 200:
            self.execution_history = self.execution_history[-200:]
