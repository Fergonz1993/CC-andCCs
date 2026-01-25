"""
Task Planner - DAG-based task planning and execution optimization.

This module provides advanced planning capabilities including:
- DAG-based task execution with cycle detection (adv-c-plan-001)
- Critical path analysis (adv-c-plan-002)
- Resource constraint solving (adv-c-plan-003)
- Parallel execution optimization (adv-c-plan-004)
- Task grouping by affinity (adv-c-plan-005)
- Milestone tracking (adv-c-plan-006)
- Plan versioning (adv-c-plan-007)
- What-if scenario analysis (adv-c-plan-008)
- Automated plan adjustment (adv-c-plan-009)
- Plan export/import (adv-c-plan-010)
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pathlib import Path

from pydantic import BaseModel, Field

from .models import Task


# =============================================================================
# Core DAG Data Structures
# =============================================================================


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the task dependency graph."""

    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        super().__init__(f"Dependency cycle detected: {' -> '.join(cycle)}")


class ResourceConflictError(Exception):
    """Raised when resource constraints cannot be satisfied."""

    def __init__(self, resource: str, tasks: list[str]):
        self.resource = resource
        self.tasks = tasks
        super().__init__(
            f"Resource conflict for '{resource}' between tasks: {', '.join(tasks)}"
        )


@dataclass
class DAGNode:
    """A node in the task dependency DAG."""

    task_id: str
    dependencies: set[str] = field(default_factory=set)
    dependents: set[str] = field(default_factory=set)
    estimated_duration: float = 0.0  # seconds
    earliest_start: float = 0.0
    latest_start: float = float("inf")
    earliest_finish: float = 0.0
    latest_finish: float = float("inf")
    slack: float = float("inf")
    is_critical: bool = False
    resources: set[str] = field(default_factory=set)
    affinity_group: Optional[str] = None


class TaskDAG:
    """
    Directed Acyclic Graph for task dependencies.

    Provides cycle detection, topological sorting, and critical path analysis.
    Implements adv-c-plan-001: DAG-based task execution with cycle detection.
    """

    def __init__(self):
        self.nodes: dict[str, DAGNode] = {}
        self._topo_order: Optional[list[str]] = None
        self._topo_dirty = True

    def add_task(
        self,
        task_id: str,
        dependencies: Optional[list[str]] = None,
        estimated_duration: float = 0.0,
        resources: Optional[list[str]] = None,
        affinity_group: Optional[str] = None,
    ) -> None:
        """Add a task to the DAG."""
        if task_id not in self.nodes:
            self.nodes[task_id] = DAGNode(task_id=task_id)

        node = self.nodes[task_id]
        node.estimated_duration = estimated_duration
        node.resources = set(resources or [])
        node.affinity_group = affinity_group

        if dependencies:
            for dep_id in dependencies:
                if dep_id not in self.nodes:
                    self.nodes[dep_id] = DAGNode(task_id=dep_id)
                node.dependencies.add(dep_id)
                self.nodes[dep_id].dependents.add(task_id)

        self._topo_dirty = True

    def remove_task(self, task_id: str) -> None:
        """Remove a task from the DAG."""
        if task_id not in self.nodes:
            return

        node = self.nodes[task_id]

        # Remove from dependents' dependency lists
        for dep_id in node.dependencies:
            if dep_id in self.nodes:
                self.nodes[dep_id].dependents.discard(task_id)

        # Remove from dependencies' dependent lists
        for dep_id in node.dependents:
            if dep_id in self.nodes:
                self.nodes[dep_id].dependencies.discard(task_id)

        del self.nodes[task_id]
        self._topo_dirty = True

    def detect_cycles(self) -> Optional[list[str]]:
        """
        Detect cycles in the DAG using DFS.

        Returns the cycle path if found, None otherwise.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node_id: WHITE for node_id in self.nodes}
        parent = {}

        def dfs(node_id: str) -> Optional[list[str]]:
            color[node_id] = GRAY

            for dep_id in self.nodes[node_id].dependencies:
                if dep_id not in self.nodes:
                    continue

                if color[dep_id] == GRAY:
                    # Cycle found - reconstruct it
                    cycle = [dep_id]
                    current = node_id
                    while current != dep_id:
                        cycle.append(current)
                        current = parent.get(current, dep_id)
                    cycle.append(dep_id)
                    return cycle[::-1]

                if color[dep_id] == WHITE:
                    parent[dep_id] = node_id
                    result = dfs(dep_id)
                    if result:
                        return result

            color[node_id] = BLACK
            return None

        for node_id in self.nodes:
            if color[node_id] == WHITE:
                cycle = dfs(node_id)
                if cycle:
                    return cycle

        return None

    def validate(self) -> None:
        """Validate the DAG has no cycles."""
        cycle = self.detect_cycles()
        if cycle:
            raise CycleDetectedError(cycle)

    def topological_sort(self) -> list[str]:
        """
        Return tasks in topological order (dependencies first).

        Uses Kahn's algorithm for deterministic ordering.
        """
        if not self._topo_dirty and self._topo_order is not None:
            return self._topo_order

        self.validate()

        # Calculate in-degrees
        in_degree = {node_id: len(node.dependencies) for node_id, node in self.nodes.items()}

        # Start with nodes that have no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        queue.sort()  # Deterministic ordering

        result = []
        while queue:
            # Pop the first (alphabetically) task with no remaining dependencies
            node_id = queue.pop(0)
            result.append(node_id)

            # Reduce in-degree for all dependents
            for dep_id in self.nodes[node_id].dependents:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    # Insert in sorted order for determinism
                    queue.append(dep_id)
                    queue.sort()

        self._topo_order = result
        self._topo_dirty = False
        return result

    def get_roots(self) -> list[str]:
        """Get tasks with no dependencies (entry points)."""
        return [node_id for node_id, node in self.nodes.items() if not node.dependencies]

    def get_leaves(self) -> list[str]:
        """Get tasks with no dependents (exit points)."""
        return [node_id for node_id, node in self.nodes.items() if not node.dependents]

    def get_ready_tasks(self, completed: set[str]) -> list[str]:
        """Get tasks that are ready to execute given completed tasks."""
        ready = []
        for node_id, node in self.nodes.items():
            if node_id in completed:
                continue
            if node.dependencies.issubset(completed):
                ready.append(node_id)
        return ready


# =============================================================================
# Critical Path Analysis (adv-c-plan-002)
# =============================================================================


class CriticalPathAnalyzer:
    """
    Analyzes the critical path through a task DAG.

    The critical path is the longest sequence of dependent tasks,
    determining the minimum project completion time.
    """

    def __init__(self, dag: TaskDAG):
        self.dag = dag
        self._analyzed = False

    def analyze(self) -> None:
        """
        Perform critical path analysis using forward and backward passes.

        This computes earliest/latest start/finish times and slack for each task.
        """
        self.dag.validate()

        # Forward pass - compute earliest start/finish times
        topo_order = self.dag.topological_sort()

        for node_id in topo_order:
            node = self.dag.nodes[node_id]

            # Earliest start is max of all dependency finish times
            if node.dependencies:
                node.earliest_start = max(
                    self.dag.nodes[dep_id].earliest_finish
                    for dep_id in node.dependencies
                    if dep_id in self.dag.nodes
                )
            else:
                node.earliest_start = 0.0

            node.earliest_finish = node.earliest_start + node.estimated_duration

        # Find project completion time
        if topo_order:
            project_end = max(
                self.dag.nodes[node_id].earliest_finish for node_id in topo_order
            )
        else:
            project_end = 0.0

        # Backward pass - compute latest start/finish times
        for node_id in reversed(topo_order):
            node = self.dag.nodes[node_id]

            # Latest finish is min of all dependent start times
            if node.dependents:
                node.latest_finish = min(
                    self.dag.nodes[dep_id].latest_start
                    for dep_id in node.dependents
                    if dep_id in self.dag.nodes
                )
            else:
                node.latest_finish = project_end

            node.latest_start = node.latest_finish - node.estimated_duration

        # Calculate slack and identify critical path
        for node_id in topo_order:
            node = self.dag.nodes[node_id]
            node.slack = node.latest_start - node.earliest_start
            node.is_critical = abs(node.slack) < 0.0001  # Float comparison

        self._analyzed = True

    def get_critical_path(self) -> list[str]:
        """Get the critical path (tasks with zero slack)."""
        if not self._analyzed:
            self.analyze()

        critical_tasks = [
            node_id for node_id, node in self.dag.nodes.items() if node.is_critical
        ]

        # Sort by earliest start time
        return sorted(critical_tasks, key=lambda x: self.dag.nodes[x].earliest_start)

    def get_project_duration(self) -> float:
        """Get the minimum project completion time."""
        if not self._analyzed:
            self.analyze()

        if not self.dag.nodes:
            return 0.0

        return max(node.earliest_finish for node in self.dag.nodes.values())

    def get_bottlenecks(self) -> list[str]:
        """
        Identify bottleneck tasks that have many dependents.

        These are tasks that, if delayed, would impact many downstream tasks.
        """
        bottlenecks = []
        for node_id, node in self.dag.nodes.items():
            # A bottleneck has multiple dependents and is on critical path
            if len(node.dependents) > 1 and node.is_critical:
                bottlenecks.append(node_id)

        return sorted(bottlenecks, key=lambda x: len(self.dag.nodes[x].dependents), reverse=True)

    def get_schedule(self) -> dict[str, dict[str, float]]:
        """Get the computed schedule for all tasks."""
        if not self._analyzed:
            self.analyze()

        return {
            node_id: {
                "earliest_start": node.earliest_start,
                "earliest_finish": node.earliest_finish,
                "latest_start": node.latest_start,
                "latest_finish": node.latest_finish,
                "slack": node.slack,
                "is_critical": node.is_critical,
            }
            for node_id, node in self.dag.nodes.items()
        }


# =============================================================================
# Resource Constraint Solving (adv-c-plan-003)
# =============================================================================


@dataclass
class Resource:
    """A constrained resource that can be used by tasks."""

    name: str
    capacity: int = 1  # How many concurrent uses allowed
    current_usage: int = 0


class ResourceConstraintSolver:
    """
    Solves resource-constrained scheduling problems.

    Ensures tasks don't exceed resource capacity constraints.
    Implements adv-c-plan-003.
    """

    def __init__(self, dag: TaskDAG):
        self.dag = dag
        self.resources: dict[str, Resource] = {}
        self._schedule: dict[str, tuple[float, float]] = {}  # task_id -> (start, end)

    def add_resource(self, name: str, capacity: int = 1) -> None:
        """Register a resource with capacity."""
        self.resources[name] = Resource(name=name, capacity=capacity)

    def detect_conflicts(self) -> list[tuple[str, list[str]]]:
        """
        Detect potential resource conflicts.

        Returns list of (resource_name, conflicting_task_ids) tuples.
        """
        conflicts = []

        # Group tasks by resource
        resource_tasks: dict[str, list[str]] = defaultdict(list)
        for node_id, node in self.dag.nodes.items():
            for res in node.resources:
                resource_tasks[res].append(node_id)

        # Check for conflicts (tasks that could run in parallel but share resources)
        for res_name, task_ids in resource_tasks.items():
            if len(task_ids) <= 1:
                continue

            # Tasks can conflict if they're not dependent on each other
            potential_conflicts = []
            for i, task1 in enumerate(task_ids):
                for task2 in task_ids[i + 1 :]:
                    node1 = self.dag.nodes[task1]
                    node2 = self.dag.nodes[task2]

                    # Check if they could run in parallel (no dependency relationship)
                    if task2 not in node1.dependencies and task1 not in node2.dependencies:
                        if task2 not in node1.dependents and task1 not in node2.dependents:
                            potential_conflicts.extend([task1, task2])

            if potential_conflicts:
                conflicts.append((res_name, list(set(potential_conflicts))))

        return conflicts

    def solve(self) -> dict[str, tuple[float, float]]:
        """
        Generate a resource-constrained schedule.

        Uses list scheduling algorithm with priority based on critical path.
        Returns dict mapping task_id to (start_time, end_time).
        """
        self.dag.validate()

        # Run critical path analysis for priorities
        analyzer = CriticalPathAnalyzer(self.dag)
        analyzer.analyze()

        # Initialize
        self._schedule = {}
        resource_free_at: dict[str, float] = {name: 0.0 for name in self.resources}

        # Sort tasks by slack (critical tasks first), then by earliest start
        ready = self.dag.get_roots()
        ready.sort(key=lambda x: (self.dag.nodes[x].slack, self.dag.nodes[x].earliest_start))

        scheduled = set()
        current_time = 0.0

        while len(scheduled) < len(self.dag.nodes):
            # Find next schedulable task
            best_task = None
            best_start = float("inf")

            for task_id in ready:
                if task_id in scheduled:
                    continue

                node = self.dag.nodes[task_id]

                # Calculate earliest possible start
                start = current_time

                # Must wait for dependencies
                for dep_id in node.dependencies:
                    if dep_id in self._schedule:
                        start = max(start, self._schedule[dep_id][1])

                # Must wait for resources
                for res_name in node.resources:
                    if res_name in resource_free_at:
                        start = max(start, resource_free_at[res_name])

                if start < best_start:
                    best_start = start
                    best_task = task_id

            if best_task is None:
                # Move time forward
                if self._schedule:
                    current_time = min(end for start, end in self._schedule.values())
                else:
                    break
                continue

            # Schedule the task
            node = self.dag.nodes[best_task]
            end_time = best_start + node.estimated_duration

            self._schedule[best_task] = (best_start, end_time)
            scheduled.add(best_task)

            # Update resource availability
            for res_name in node.resources:
                if res_name in resource_free_at:
                    resource_free_at[res_name] = end_time

            # Add newly ready tasks
            for dep_id in node.dependents:
                dep_node = self.dag.nodes[dep_id]
                if all(d in scheduled for d in dep_node.dependencies):
                    ready.append(dep_id)
                    ready.sort(
                        key=lambda x: (
                            self.dag.nodes[x].slack,
                            self.dag.nodes[x].earliest_start,
                        )
                    )

            current_time = best_start

        return self._schedule

    def get_resource_timeline(self) -> dict[str, list[tuple[float, float, str]]]:
        """
        Get resource usage timeline.

        Returns dict mapping resource_name to list of (start, end, task_id) tuples.
        """
        if not self._schedule:
            self.solve()

        timeline: dict[str, list[tuple[float, float, str]]] = defaultdict(list)

        for task_id, (start, end) in self._schedule.items():
            node = self.dag.nodes[task_id]
            for res_name in node.resources:
                timeline[res_name].append((start, end, task_id))

        # Sort by start time
        for res_name in timeline:
            timeline[res_name].sort(key=lambda x: x[0])

        return dict(timeline)


# =============================================================================
# Parallel Execution Optimization (adv-c-plan-004)
# =============================================================================


class ParallelExecutionOptimizer:
    """
    Optimizes parallel task execution.

    Implements adv-c-plan-004.
    """

    def __init__(self, dag: TaskDAG, max_parallelism: int = 4):
        self.dag = dag
        self.max_parallelism = max_parallelism

    def get_parallel_stages(self) -> list[list[str]]:
        """
        Organize tasks into parallel execution stages.

        Each stage contains tasks that can run concurrently.
        """
        self.dag.validate()

        completed: set[str] = set()
        stages: list[list[str]] = []

        while len(completed) < len(self.dag.nodes):
            ready = self.dag.get_ready_tasks(completed)
            if not ready:
                break

            # Limit parallelism
            stage = ready[: self.max_parallelism]
            stages.append(stage)
            completed.update(stage)

        return stages

    def calculate_speedup(self) -> dict[str, float]:
        """
        Calculate theoretical speedup from parallelization.

        Returns dict with sequential time, parallel time, and speedup factor.
        """
        # Sequential time: sum of all durations
        sequential_time = sum(node.estimated_duration for node in self.dag.nodes.values())

        # Parallel time: sum of stage max durations
        stages = self.get_parallel_stages()
        parallel_time = 0.0

        for stage in stages:
            if stage:
                stage_duration = max(
                    self.dag.nodes[task_id].estimated_duration for task_id in stage
                )
                parallel_time += stage_duration

        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0

        return {
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "speedup": speedup,
            "efficiency": speedup / self.max_parallelism if self.max_parallelism > 0 else 1.0,
        }

    def suggest_rebalancing(self) -> list[dict[str, Any]]:
        """
        Suggest ways to improve parallel execution.

        Returns list of suggestions.
        """
        suggestions = []
        stages = self.get_parallel_stages()

        # Find stages with low utilization
        for i, stage in enumerate(stages):
            utilization = len(stage) / self.max_parallelism

            if utilization < 0.5 and len(stage) > 0:
                suggestions.append(
                    {
                        "stage": i,
                        "utilization": utilization,
                        "suggestion": f"Stage {i} only uses {len(stage)}/{self.max_parallelism} workers. "
                        f"Consider breaking down tasks to increase parallelism.",
                        "affected_tasks": stage,
                    }
                )

        # Find long critical path tasks that could be split
        analyzer = CriticalPathAnalyzer(self.dag)
        analyzer.analyze()

        avg_duration = (
            sum(node.estimated_duration for node in self.dag.nodes.values())
            / len(self.dag.nodes)
            if self.dag.nodes
            else 0
        )

        for node_id, node in self.dag.nodes.items():
            if node.is_critical and node.estimated_duration > 2 * avg_duration:
                suggestions.append(
                    {
                        "task": node_id,
                        "duration": node.estimated_duration,
                        "suggestion": f"Task {node_id} is on critical path and takes "
                        f"{node.estimated_duration:.1f}s (2x average). Consider splitting it.",
                    }
                )

        return suggestions


# =============================================================================
# Task Grouping by Affinity (adv-c-plan-005)
# =============================================================================


class AffinityGrouper:
    """
    Groups tasks by affinity for efficient execution.

    Tasks in the same affinity group should run on the same worker
    to benefit from shared context (e.g., cached files, warm state).

    Implements adv-c-plan-005.
    """

    def __init__(self, dag: TaskDAG):
        self.dag = dag

    def detect_affinity_groups(self) -> dict[str, list[str]]:
        """
        Auto-detect affinity groups based on shared resources.

        Returns dict mapping group_name to list of task_ids.
        """
        groups: dict[str, list[str]] = defaultdict(list)

        # Group by explicit affinity
        for node_id, node in self.dag.nodes.items():
            if node.affinity_group:
                groups[node.affinity_group].append(node_id)

        # Group by shared resources
        resource_tasks: dict[str, set[str]] = defaultdict(set)
        for node_id, node in self.dag.nodes.items():
            for res in node.resources:
                resource_tasks[res].add(node_id)

        # Create implicit groups for tasks sharing resources
        for res_name, task_ids in resource_tasks.items():
            if len(task_ids) > 1:
                group_name = f"resource-{res_name}"
                if group_name not in groups:
                    groups[group_name] = list(task_ids)

        # Remove duplicates (tasks may be in multiple groups)
        return dict(groups)

    def optimize_worker_assignment(
        self, num_workers: int
    ) -> dict[str, int]:
        """
        Optimize task-to-worker assignment based on affinity.

        Returns dict mapping task_id to worker_index.
        """
        groups = self.detect_affinity_groups()
        assignment: dict[str, int] = {}

        # Assign groups to workers round-robin
        group_worker: dict[str, int] = {}
        next_worker = 0

        for group_name, task_ids in groups.items():
            if group_name not in group_worker:
                group_worker[group_name] = next_worker
                next_worker = (next_worker + 1) % num_workers

            worker = group_worker[group_name]
            for task_id in task_ids:
                assignment[task_id] = worker

        # Assign ungrouped tasks
        for node_id in self.dag.nodes:
            if node_id not in assignment:
                # Load balance: find worker with fewest tasks
                worker_loads = [0] * num_workers
                for w in assignment.values():
                    worker_loads[w] += 1
                assignment[node_id] = worker_loads.index(min(worker_loads))

        return assignment


# =============================================================================
# Milestone Tracking (adv-c-plan-006)
# =============================================================================


class MilestoneStatus(str, Enum):
    """Status of a milestone."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    AT_RISK = "at_risk"
    MISSED = "missed"


@dataclass
class Milestone:
    """A project milestone."""

    id: str
    name: str
    description: str
    required_tasks: list[str]
    target_time: Optional[float] = None  # Seconds from start
    status: MilestoneStatus = MilestoneStatus.PENDING
    completed_at: Optional[datetime] = None


class MilestoneTracker:
    """
    Tracks project milestones based on task completion.

    Implements adv-c-plan-006.
    """

    def __init__(self, dag: TaskDAG):
        self.dag = dag
        self.milestones: dict[str, Milestone] = {}
        self._start_time: Optional[datetime] = None

    def add_milestone(
        self,
        name: str,
        required_tasks: list[str],
        target_time: Optional[float] = None,
        description: str = "",
    ) -> Milestone:
        """Add a new milestone."""
        milestone_id = f"milestone-{uuid.uuid4().hex[:8]}"
        milestone = Milestone(
            id=milestone_id,
            name=name,
            description=description,
            required_tasks=required_tasks,
            target_time=target_time,
        )
        self.milestones[milestone_id] = milestone
        return milestone

    def start_tracking(self) -> None:
        """Start milestone tracking."""
        self._start_time = datetime.now()

    def update(self, completed_tasks: set[str]) -> list[Milestone]:
        """
        Update milestone statuses based on completed tasks.

        Returns list of milestones that just completed.
        """
        if not self._start_time:
            self.start_tracking()

        elapsed = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0

        newly_completed = []

        for milestone in self.milestones.values():
            if milestone.status == MilestoneStatus.COMPLETED:
                continue

            # Check if all required tasks are complete
            all_complete = all(t in completed_tasks for t in milestone.required_tasks)
            some_complete = any(t in completed_tasks for t in milestone.required_tasks)

            if all_complete:
                milestone.status = MilestoneStatus.COMPLETED
                milestone.completed_at = datetime.now()
                newly_completed.append(milestone)
            elif some_complete:
                milestone.status = MilestoneStatus.IN_PROGRESS
                # Check if at risk
                if milestone.target_time and elapsed > milestone.target_time * 0.8:
                    milestone.status = MilestoneStatus.AT_RISK
            elif milestone.target_time and elapsed > milestone.target_time:
                milestone.status = MilestoneStatus.MISSED

        return newly_completed

    def get_progress(self) -> dict[str, Any]:
        """Get milestone progress summary."""
        total = len(self.milestones)
        by_status = defaultdict(int)

        for milestone in self.milestones.values():
            by_status[milestone.status.value] += 1

        return {
            "total": total,
            "by_status": dict(by_status),
            "percent_complete": (by_status[MilestoneStatus.COMPLETED.value] / total * 100)
            if total > 0
            else 0,
        }

    def get_next_milestone(self) -> Optional[Milestone]:
        """Get the next incomplete milestone."""
        for milestone in self.milestones.values():
            if milestone.status not in (MilestoneStatus.COMPLETED, MilestoneStatus.MISSED):
                return milestone
        return None


# =============================================================================
# Plan Versioning (adv-c-plan-007)
# =============================================================================


class PlanVersion(BaseModel):
    """A versioned snapshot of the plan."""

    version: int
    timestamp: datetime = Field(default_factory=datetime.now)
    description: str = ""
    task_ids: list[str]
    dependencies: dict[str, list[str]]  # task_id -> dependency_ids
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanVersionControl:
    """
    Version control for execution plans.

    Implements adv-c-plan-007.
    """

    def __init__(self, dag: TaskDAG):
        self.dag = dag
        self.versions: list[PlanVersion] = []
        self.current_version = 0

    def commit(self, description: str = "") -> PlanVersion:
        """Create a new version of the current plan."""
        self.current_version += 1

        version = PlanVersion(
            version=self.current_version,
            description=description,
            task_ids=list(self.dag.nodes.keys()),
            dependencies={
                node_id: list(node.dependencies) for node_id, node in self.dag.nodes.items()
            },
            metadata={
                node_id: {
                    "estimated_duration": node.estimated_duration,
                    "resources": list(node.resources),
                    "affinity_group": node.affinity_group,
                }
                for node_id, node in self.dag.nodes.items()
            },
        )

        self.versions.append(version)
        return version

    def revert(self, version_num: int) -> bool:
        """Revert to a previous version."""
        for version in self.versions:
            if version.version == version_num:
                # Clear current DAG
                self.dag.nodes.clear()

                # Restore from version
                for task_id in version.task_ids:
                    meta = version.metadata.get(task_id, {})
                    self.dag.add_task(
                        task_id=task_id,
                        dependencies=version.dependencies.get(task_id, []),
                        estimated_duration=meta.get("estimated_duration", 0.0),
                        resources=meta.get("resources", []),
                        affinity_group=meta.get("affinity_group"),
                    )

                self.current_version = version_num
                return True

        return False

    def diff(self, version1: int, version2: int) -> dict[str, Any]:
        """Compare two versions."""
        v1 = next((v for v in self.versions if v.version == version1), None)
        v2 = next((v for v in self.versions if v.version == version2), None)

        if not v1 or not v2:
            return {"error": "Version not found"}

        tasks_added = set(v2.task_ids) - set(v1.task_ids)
        tasks_removed = set(v1.task_ids) - set(v2.task_ids)
        tasks_common = set(v1.task_ids) & set(v2.task_ids)

        deps_changed = []
        for task_id in tasks_common:
            if set(v1.dependencies.get(task_id, [])) != set(v2.dependencies.get(task_id, [])):
                deps_changed.append(task_id)

        return {
            "version1": version1,
            "version2": version2,
            "tasks_added": list(tasks_added),
            "tasks_removed": list(tasks_removed),
            "dependencies_changed": deps_changed,
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Get version history."""
        return [
            {
                "version": v.version,
                "timestamp": v.timestamp.isoformat(),
                "description": v.description,
                "task_count": len(v.task_ids),
            }
            for v in self.versions
        ]


# =============================================================================
# What-If Scenario Analysis (adv-c-plan-008)
# =============================================================================


@dataclass
class Scenario:
    """A what-if scenario configuration."""

    id: str
    name: str
    num_workers: int = 1
    worker_speed_factor: float = 1.0  # Multiplier for task durations
    removed_tasks: list[str] = field(default_factory=list)
    added_dependencies: list[tuple[str, str]] = field(default_factory=list)


class ScenarioAnalyzer:
    """
    Analyzes what-if scenarios for planning.

    Implements adv-c-plan-008.
    """

    def __init__(self, dag: TaskDAG):
        self.dag = dag
        self.scenarios: dict[str, Scenario] = {}

    def create_scenario(
        self,
        name: str,
        num_workers: int = 1,
        worker_speed_factor: float = 1.0,
        removed_tasks: Optional[list[str]] = None,
        added_dependencies: Optional[list[tuple[str, str]]] = None,
    ) -> Scenario:
        """Create a new scenario."""
        scenario_id = f"scenario-{uuid.uuid4().hex[:8]}"
        scenario = Scenario(
            id=scenario_id,
            name=name,
            num_workers=num_workers,
            worker_speed_factor=worker_speed_factor,
            removed_tasks=removed_tasks or [],
            added_dependencies=added_dependencies or [],
        )
        self.scenarios[scenario_id] = scenario
        return scenario

    def _apply_scenario(self, scenario: Scenario) -> TaskDAG:
        """Create a modified DAG based on scenario."""
        modified = TaskDAG()

        # Copy nodes with modified durations
        for node_id, node in self.dag.nodes.items():
            if node_id in scenario.removed_tasks:
                continue

            modified.add_task(
                task_id=node_id,
                dependencies=[d for d in node.dependencies if d not in scenario.removed_tasks],
                estimated_duration=node.estimated_duration * scenario.worker_speed_factor,
                resources=list(node.resources),
                affinity_group=node.affinity_group,
            )

        # Add new dependencies
        for task_id, dep_id in scenario.added_dependencies:
            if task_id in modified.nodes and dep_id in modified.nodes:
                modified.nodes[task_id].dependencies.add(dep_id)
                modified.nodes[dep_id].dependents.add(task_id)

        return modified

    def analyze(self, scenario_id: str) -> dict[str, Any]:
        """Analyze a scenario."""
        scenario = self.scenarios.get(scenario_id)
        if not scenario:
            return {"error": "Scenario not found"}

        # Create modified DAG
        modified_dag = self._apply_scenario(scenario)

        # Run analyses
        try:
            modified_dag.validate()
        except CycleDetectedError as e:
            return {"error": f"Scenario creates cycle: {e.cycle}"}

        analyzer = CriticalPathAnalyzer(modified_dag)
        analyzer.analyze()

        optimizer = ParallelExecutionOptimizer(modified_dag, scenario.num_workers)
        speedup = optimizer.calculate_speedup()
        stages = optimizer.get_parallel_stages()

        return {
            "scenario": scenario.name,
            "num_workers": scenario.num_workers,
            "speed_factor": scenario.worker_speed_factor,
            "project_duration": analyzer.get_project_duration(),
            "critical_path": analyzer.get_critical_path(),
            "parallel_stages": len(stages),
            "speedup": speedup,
        }

    def compare_scenarios(self, scenario_ids: list[str]) -> dict[str, Any]:
        """Compare multiple scenarios."""
        results = []

        for scenario_id in scenario_ids:
            result = self.analyze(scenario_id)
            if "error" not in result:
                results.append(result)

        if not results:
            return {"error": "No valid scenarios to compare"}

        # Find best scenario
        best = min(results, key=lambda x: x["project_duration"])

        return {
            "scenarios": results,
            "best_scenario": best["scenario"],
            "best_duration": best["project_duration"],
        }


# =============================================================================
# Automated Plan Adjustment (adv-c-plan-009)
# =============================================================================


@dataclass
class AdjustmentRule:
    """A rule for automated plan adjustment."""

    name: str
    condition: str  # Description of when to apply
    action: str  # Description of what to do
    priority: int = 5


class PlanAdjuster:
    """
    Automatically adjusts plans based on execution feedback.

    Implements adv-c-plan-009.
    """

    def __init__(self, dag: TaskDAG):
        self.dag = dag
        self.rules: list[AdjustmentRule] = []
        self.adjustment_history: list[dict[str, Any]] = []

    def add_rule(
        self,
        name: str,
        condition: str,
        action: str,
        priority: int = 5,
    ) -> None:
        """Add an adjustment rule."""
        rule = AdjustmentRule(name=name, condition=condition, action=action, priority=priority)
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)

    def adjust_for_slow_task(
        self,
        task_id: str,
        expected_duration: float,
        actual_duration: float,
    ) -> list[str]:
        """
        Adjust plan when a task takes longer than expected.

        Returns list of adjustment descriptions.
        """
        adjustments = []
        ratio = actual_duration / expected_duration if expected_duration > 0 else 1.0

        if ratio > 1.5:
            # Task took 50% longer than expected
            # Increase estimates for similar tasks
            node = self.dag.nodes.get(task_id)
            if node:
                for other_id, other_node in self.dag.nodes.items():
                    if other_id != task_id and other_node.affinity_group == node.affinity_group:
                        old_duration = other_node.estimated_duration
                        other_node.estimated_duration *= min(ratio, 2.0)
                        adjustments.append(
                            f"Increased estimate for {other_id} from {old_duration:.1f}s to "
                            f"{other_node.estimated_duration:.1f}s"
                        )

            self.adjustment_history.append(
                {
                    "type": "slow_task",
                    "task_id": task_id,
                    "ratio": ratio,
                    "adjustments": adjustments,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return adjustments

    def adjust_for_failed_task(self, task_id: str, error: str) -> list[str]:
        """
        Adjust plan when a task fails.

        Returns list of adjustment descriptions.
        """
        adjustments = []

        node = self.dag.nodes.get(task_id)
        if not node:
            return adjustments

        # Check if dependent tasks can proceed without this task
        for dep_id in node.dependents:
            dep_node = self.dag.nodes.get(dep_id)
            if dep_node and len(dep_node.dependencies) > 1:
                # Could potentially proceed if this was optional
                adjustments.append(
                    f"Consider making {task_id} optional for {dep_id} if appropriate"
                )

        self.adjustment_history.append(
            {
                "type": "failed_task",
                "task_id": task_id,
                "error": error,
                "adjustments": adjustments,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return adjustments

    def suggest_replan(
        self,
        completed_tasks: set[str],
        failed_tasks: set[str],
    ) -> dict[str, Any]:
        """
        Suggest replanning based on current execution state.

        Returns suggestions for how to proceed.
        """
        remaining = set(self.dag.nodes.keys()) - completed_tasks - failed_tasks

        # Build a DAG with only remaining tasks
        remaining_dag = TaskDAG()
        for task_id in remaining:
            node = self.dag.nodes[task_id]
            deps = [d for d in node.dependencies if d in remaining]
            remaining_dag.add_task(
                task_id=task_id,
                dependencies=deps,
                estimated_duration=node.estimated_duration,
            )

        analyzer = CriticalPathAnalyzer(remaining_dag)
        analyzer.analyze()

        return {
            "remaining_tasks": len(remaining),
            "remaining_duration": analyzer.get_project_duration(),
            "critical_path": analyzer.get_critical_path(),
            "suggested_parallelism": min(len(remaining_dag.get_roots()), 4),
            "blocked_by_failures": [
                task_id
                for task_id in remaining
                if any(f in self.dag.nodes[task_id].dependencies for f in failed_tasks)
            ],
        }


# =============================================================================
# Plan Export/Import (adv-c-plan-010)
# =============================================================================


class PlanFormat(str, Enum):
    """Supported plan export formats."""

    JSON = "json"
    YAML = "yaml"
    DOT = "dot"  # GraphViz DOT format


class PlanExporter:
    """
    Exports and imports execution plans.

    Implements adv-c-plan-010.
    """

    def __init__(self, dag: TaskDAG):
        self.dag = dag

    def export_json(self) -> str:
        """Export plan to JSON format."""
        data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "tasks": [
                {
                    "id": node_id,
                    "dependencies": list(node.dependencies),
                    "estimated_duration": node.estimated_duration,
                    "resources": list(node.resources),
                    "affinity_group": node.affinity_group,
                }
                for node_id, node in self.dag.nodes.items()
            ],
        }
        return json.dumps(data, indent=2)

    def import_json(self, json_str: str) -> None:
        """Import plan from JSON format."""
        data = json.loads(json_str)

        self.dag.nodes.clear()

        for task_data in data.get("tasks", []):
            self.dag.add_task(
                task_id=task_data["id"],
                dependencies=task_data.get("dependencies", []),
                estimated_duration=task_data.get("estimated_duration", 0.0),
                resources=task_data.get("resources", []),
                affinity_group=task_data.get("affinity_group"),
            )

    def export_yaml(self) -> str:
        """Export plan to YAML format."""
        # Simple YAML generation without external dependency
        lines = [
            "version: '1.0'",
            f"exported_at: '{datetime.now().isoformat()}'",
            "tasks:",
        ]

        for node_id, node in self.dag.nodes.items():
            lines.append(f"  - id: '{node_id}'")

            if node.dependencies:
                lines.append("    dependencies:")
                for dep in node.dependencies:
                    lines.append(f"      - '{dep}'")
            else:
                lines.append("    dependencies: []")

            lines.append(f"    estimated_duration: {node.estimated_duration}")

            if node.resources:
                lines.append("    resources:")
                for res in node.resources:
                    lines.append(f"      - '{res}'")
            else:
                lines.append("    resources: []")

            if node.affinity_group:
                lines.append(f"    affinity_group: '{node.affinity_group}'")

        return "\n".join(lines)

    def export_dot(self) -> str:
        """Export plan to GraphViz DOT format for visualization."""
        lines = [
            "digraph TaskPlan {",
            "  rankdir=LR;",
            "  node [shape=box, style=rounded];",
        ]

        # Run critical path analysis for coloring
        analyzer = CriticalPathAnalyzer(self.dag)
        try:
            analyzer.analyze()
        except CycleDetectedError:
            pass

        # Add nodes
        for node_id, node in self.dag.nodes.items():
            color = "red" if node.is_critical else "black"
            label = f"{node_id}\\n({node.estimated_duration:.1f}s)"
            lines.append(f'  "{node_id}" [label="{label}", color={color}];')

        # Add edges
        for node_id, node in self.dag.nodes.items():
            for dep_id in node.dependencies:
                lines.append(f'  "{dep_id}" -> "{node_id}";')

        lines.append("}")
        return "\n".join(lines)

    def save_to_file(self, filepath: str, format: PlanFormat = PlanFormat.JSON) -> None:
        """Save plan to a file."""
        path = Path(filepath)

        if format == PlanFormat.JSON:
            content = self.export_json()
        elif format == PlanFormat.YAML:
            content = self.export_yaml()
        elif format == PlanFormat.DOT:
            content = self.export_dot()
        else:
            raise ValueError(f"Unknown format: {format}")

        path.write_text(content)

    def load_from_file(self, filepath: str) -> None:
        """Load plan from a file (JSON only)."""
        path = Path(filepath)
        content = path.read_text()
        self.import_json(content)


# =============================================================================
# High-Level Planner Interface
# =============================================================================


class TaskPlanner:
    """
    High-level interface for all planning features.

    Combines DAG management, critical path analysis, resource solving,
    and all other planning capabilities.
    """

    def __init__(self):
        self.dag = TaskDAG()
        self._critical_path_analyzer: Optional[CriticalPathAnalyzer] = None
        self._resource_solver: Optional[ResourceConstraintSolver] = None
        self._parallel_optimizer: Optional[ParallelExecutionOptimizer] = None
        self._affinity_grouper: Optional[AffinityGrouper] = None
        self._milestone_tracker: Optional[MilestoneTracker] = None
        self._version_control: Optional[PlanVersionControl] = None
        self._scenario_analyzer: Optional[ScenarioAnalyzer] = None
        self._plan_adjuster: Optional[PlanAdjuster] = None
        self._exporter: Optional[PlanExporter] = None

    def add_task(
        self,
        task_id: str,
        dependencies: Optional[list[str]] = None,
        estimated_duration: float = 0.0,
        resources: Optional[list[str]] = None,
        affinity_group: Optional[str] = None,
    ) -> None:
        """Add a task to the plan."""
        self.dag.add_task(
            task_id=task_id,
            dependencies=dependencies,
            estimated_duration=estimated_duration,
            resources=resources,
            affinity_group=affinity_group,
        )

    def add_tasks_from_list(self, tasks: list[Task]) -> None:
        """Add tasks from Task model objects."""
        for task in tasks:
            duration = task.context.metadata.get("estimated_duration", 60.0)
            resources = task.context.metadata.get("resources", [])
            affinity = task.context.metadata.get("affinity_group")

            self.dag.add_task(
                task_id=task.id,
                dependencies=task.dependencies,
                estimated_duration=duration,
                resources=resources,
                affinity_group=affinity,
            )

    def validate(self) -> bool:
        """Validate the plan has no cycles."""
        try:
            self.dag.validate()
            return True
        except CycleDetectedError:
            return False

    def get_execution_order(self) -> list[str]:
        """Get tasks in execution order (topological sort)."""
        return self.dag.topological_sort()

    def get_ready_tasks(self, completed: set[str]) -> list[str]:
        """Get tasks ready for execution."""
        return self.dag.get_ready_tasks(completed)

    # Critical Path Analysis
    def analyze_critical_path(self) -> CriticalPathAnalyzer:
        """Get or create critical path analyzer."""
        if self._critical_path_analyzer is None:
            self._critical_path_analyzer = CriticalPathAnalyzer(self.dag)
        return self._critical_path_analyzer

    def get_critical_path(self) -> list[str]:
        """Get the critical path."""
        return self.analyze_critical_path().get_critical_path()

    def get_project_duration(self) -> float:
        """Get minimum project completion time."""
        return self.analyze_critical_path().get_project_duration()

    # Resource Management
    def add_resource(self, name: str, capacity: int = 1) -> None:
        """Add a resource constraint."""
        if self._resource_solver is None:
            self._resource_solver = ResourceConstraintSolver(self.dag)
        self._resource_solver.add_resource(name, capacity)

    def get_resource_schedule(self) -> dict[str, tuple[float, float]]:
        """Get resource-constrained schedule."""
        if self._resource_solver is None:
            self._resource_solver = ResourceConstraintSolver(self.dag)
        return self._resource_solver.solve()

    # Parallel Optimization
    def optimize_parallel(self, max_parallelism: int = 4) -> ParallelExecutionOptimizer:
        """Get or create parallel optimizer."""
        self._parallel_optimizer = ParallelExecutionOptimizer(self.dag, max_parallelism)
        return self._parallel_optimizer

    def get_parallel_stages(self, max_parallelism: int = 4) -> list[list[str]]:
        """Get parallel execution stages."""
        return self.optimize_parallel(max_parallelism).get_parallel_stages()

    # Affinity Grouping
    def get_affinity_groups(self) -> dict[str, list[str]]:
        """Get task affinity groups."""
        if self._affinity_grouper is None:
            self._affinity_grouper = AffinityGrouper(self.dag)
        return self._affinity_grouper.detect_affinity_groups()

    def get_worker_assignment(self, num_workers: int) -> dict[str, int]:
        """Get optimized task-to-worker assignment."""
        if self._affinity_grouper is None:
            self._affinity_grouper = AffinityGrouper(self.dag)
        return self._affinity_grouper.optimize_worker_assignment(num_workers)

    # Milestone Tracking
    def add_milestone(
        self,
        name: str,
        required_tasks: list[str],
        target_time: Optional[float] = None,
    ) -> Milestone:
        """Add a project milestone."""
        if self._milestone_tracker is None:
            self._milestone_tracker = MilestoneTracker(self.dag)
        return self._milestone_tracker.add_milestone(name, required_tasks, target_time)

    def update_milestones(self, completed_tasks: set[str]) -> list[Milestone]:
        """Update milestone statuses."""
        if self._milestone_tracker is None:
            return []
        return self._milestone_tracker.update(completed_tasks)

    # Version Control
    def commit_plan(self, description: str = "") -> PlanVersion:
        """Create a new version of the plan."""
        if self._version_control is None:
            self._version_control = PlanVersionControl(self.dag)
        return self._version_control.commit(description)

    def revert_plan(self, version: int) -> bool:
        """Revert to a previous plan version."""
        if self._version_control is None:
            return False
        return self._version_control.revert(version)

    # Scenario Analysis
    def create_scenario(
        self,
        name: str,
        num_workers: int = 1,
        **kwargs: Any,
    ) -> Scenario:
        """Create a what-if scenario."""
        if self._scenario_analyzer is None:
            self._scenario_analyzer = ScenarioAnalyzer(self.dag)
        return self._scenario_analyzer.create_scenario(name, num_workers, **kwargs)

    def analyze_scenario(self, scenario_id: str) -> dict[str, Any]:
        """Analyze a scenario."""
        if self._scenario_analyzer is None:
            return {"error": "No scenarios created"}
        return self._scenario_analyzer.analyze(scenario_id)

    # Plan Adjustment
    def adjust_for_slow_task(
        self,
        task_id: str,
        expected: float,
        actual: float,
    ) -> list[str]:
        """Adjust plan for a slow task."""
        if self._plan_adjuster is None:
            self._plan_adjuster = PlanAdjuster(self.dag)
        return self._plan_adjuster.adjust_for_slow_task(task_id, expected, actual)

    def suggest_replan(
        self,
        completed: set[str],
        failed: set[str],
    ) -> dict[str, Any]:
        """Get replanning suggestions."""
        if self._plan_adjuster is None:
            self._plan_adjuster = PlanAdjuster(self.dag)
        return self._plan_adjuster.suggest_replan(completed, failed)

    # Export/Import
    def export(self, format: PlanFormat = PlanFormat.JSON) -> str:
        """Export the plan."""
        if self._exporter is None:
            self._exporter = PlanExporter(self.dag)

        if format == PlanFormat.JSON:
            return self._exporter.export_json()
        elif format == PlanFormat.YAML:
            return self._exporter.export_yaml()
        elif format == PlanFormat.DOT:
            return self._exporter.export_dot()
        else:
            raise ValueError(f"Unknown format: {format}")

    def import_plan(self, json_str: str) -> None:
        """Import a plan from JSON."""
        if self._exporter is None:
            self._exporter = PlanExporter(self.dag)
        self._exporter.import_json(json_str)

    def save(self, filepath: str, format: PlanFormat = PlanFormat.JSON) -> None:
        """Save plan to file."""
        if self._exporter is None:
            self._exporter = PlanExporter(self.dag)
        self._exporter.save_to_file(filepath, format)

    def load(self, filepath: str) -> None:
        """Load plan from file."""
        if self._exporter is None:
            self._exporter = PlanExporter(self.dag)
        self._exporter.load_from_file(filepath)
