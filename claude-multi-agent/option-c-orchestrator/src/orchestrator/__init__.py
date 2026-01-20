"""
Claude Multi-Agent Orchestrator

A comprehensive system for coordinating multiple Claude Code instances
to work together on complex tasks.

Architecture:
    Orchestrator (this module)
         │
         ├── Leader Agent (Terminal 1)
         │     └── Plans work, creates tasks, aggregates results
         │
         ├── Worker Agent (Terminal 2)
         │     └── Claims and executes tasks
         │
         └── Worker Agent (Terminal 3)
               └── Claims and executes tasks

Communication flows through the orchestrator, which:
1. Spawns Claude Code processes
2. Sends prompts via stdin
3. Receives responses via stdout
4. Coordinates task distribution
5. Aggregates results

Advanced Features (v2.0):
- Adaptive worker pool scaling (adv-c-001)
- Worker health monitoring with auto-restart (adv-c-002)
- Load balancing strategies (adv-c-003)
- Parallel execution with max concurrency (adv-c-004)
- Task timeout with graceful cancellation (adv-c-005)
- Inter-worker communication channel (adv-c-006)
- Shared memory for large data (adv-c-007)
- Worker specialization and routing (adv-c-008)
- Task execution sandboxing (adv-c-009)
- Execution replay for debugging (adv-c-010)

Planning Features (v2.1):
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

Monitoring Features (v2.1):
- Real-time status dashboard data (adv-c-mon-001)
- Worker health checks (adv-c-mon-002)
- Anomaly detection alerts (adv-c-mon-003)
- Resource usage tracking (adv-c-mon-004)
- Execution timeline visualization (adv-c-mon-005)
"""

from .models import Task, TaskStatus, Agent, AgentRole, CoordinationState
from .agent import ClaudeCodeAgent
from .orchestrator import Orchestrator
from .async_orchestrator import Orchestrator as AsyncOrchestrator
from .cli import app

# Advanced features
from .advanced import (
    # adv-c-001: Adaptive worker scaling
    ScalingConfig,
    AdaptiveScaler,
    # adv-c-002: Worker health monitoring
    HealthConfig,
    WorkerHealthStatus,
    WorkerHealthRecord,
    HealthMonitor,
    # adv-c-003: Load balancing strategies
    LoadBalancingStrategy,
    WorkerLoad,
    LoadBalancer,
    # adv-c-004: Parallel execution with max concurrency
    ConcurrencyLimiter,
    # adv-c-005: Task timeout with graceful cancellation
    TimeoutConfig,
    TaskTimeoutError,
    TimeoutManager,
    # adv-c-006: Inter-worker communication
    Message,
    MessageChannel,
    # adv-c-007: Shared memory for large data
    SharedArtifact,
    SharedMemoryManager,
    # adv-c-008: Worker specialization and routing
    WorkerSpecialization,
    SpecializationRouter,
    # adv-c-009: Task execution sandboxing
    SandboxConfig,
    ResourceMonitor,
    TaskSandbox,
    # adv-c-010: Execution replay for debugging
    ExecutionEvent,
    ExecutionRecording,
    ExecutionRecorder,
    # Mixin for easy integration
    AdvancedOrchestratorMixin,
)

# Planning features (adv-c-plan-001 through adv-c-plan-010)
from .planner import (
    TaskPlanner,
    TaskDAG,
    CriticalPathAnalyzer,
    ResourceConstraintSolver,
    ParallelExecutionOptimizer,
    AffinityGrouper,
    MilestoneTracker,
    PlanVersionControl,
    ScenarioAnalyzer,
    PlanAdjuster,
    PlanExporter,
    PlanFormat,
    CycleDetectedError,
    ResourceConflictError,
    Milestone,
    MilestoneStatus,
)

# Monitoring features (adv-c-mon-001 through adv-c-mon-005)
from .monitor import (
    TaskMonitor,
    StatusDashboard,
    WorkerHealthChecker,
    AnomalyDetector,
    ResourceTracker,
    TimelineVisualization,
    Alert,
    AlertSeverity,
    AlertType,
    HealthCheckConfig,
)

__version__ = "2.1.0"
__all__ = [
    # Core
    "Task",
    "TaskStatus",
    "Agent",
    "AgentRole",
    "CoordinationState",
    "ClaudeCodeAgent",
    "Orchestrator",
    "AsyncOrchestrator",
    "app",
    # adv-c-001: Adaptive worker scaling
    "ScalingConfig",
    "AdaptiveScaler",
    # adv-c-002: Worker health monitoring
    "HealthConfig",
    "WorkerHealthStatus",
    "WorkerHealthRecord",
    "HealthMonitor",
    # adv-c-003: Load balancing strategies
    "LoadBalancingStrategy",
    "WorkerLoad",
    "LoadBalancer",
    # adv-c-004: Parallel execution with max concurrency
    "ConcurrencyLimiter",
    # adv-c-005: Task timeout with graceful cancellation
    "TimeoutConfig",
    "TaskTimeoutError",
    "TimeoutManager",
    # adv-c-006: Inter-worker communication
    "Message",
    "MessageChannel",
    # adv-c-007: Shared memory for large data
    "SharedArtifact",
    "SharedMemoryManager",
    # adv-c-008: Worker specialization and routing
    "WorkerSpecialization",
    "SpecializationRouter",
    # adv-c-009: Task execution sandboxing
    "SandboxConfig",
    "ResourceMonitor",
    "TaskSandbox",
    # adv-c-010: Execution replay for debugging
    "ExecutionEvent",
    "ExecutionRecording",
    "ExecutionRecorder",
    # Mixin
    "AdvancedOrchestratorMixin",
    # Planning features (adv-c-plan-001 through adv-c-plan-010)
    "TaskPlanner",
    "TaskDAG",
    "CriticalPathAnalyzer",
    "ResourceConstraintSolver",
    "ParallelExecutionOptimizer",
    "AffinityGrouper",
    "MilestoneTracker",
    "PlanVersionControl",
    "ScenarioAnalyzer",
    "PlanAdjuster",
    "PlanExporter",
    "PlanFormat",
    "CycleDetectedError",
    "ResourceConflictError",
    "Milestone",
    "MilestoneStatus",
    # Monitoring features (adv-c-mon-001 through adv-c-mon-005)
    "TaskMonitor",
    "StatusDashboard",
    "WorkerHealthChecker",
    "AnomalyDetector",
    "ResourceTracker",
    "TimelineVisualization",
    "Alert",
    "AlertSeverity",
    "AlertType",
    "HealthCheckConfig",
]
