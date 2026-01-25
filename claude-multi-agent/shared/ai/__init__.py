"""
AI-powered features for the Claude Multi-Agent Coordination System.

This module provides intelligent algorithms and heuristics for:
- Task prioritization
- Task decomposition
- Worker assignment
- Anomaly detection
- Duration prediction
- Auto-generated descriptions
- Context-aware errors
- Learning from failures
- Workflow optimization
- Natural language task creation

No external ML libraries required - uses statistical methods and heuristics.
"""

from .prioritization import TaskPrioritizer, PriorityFactors, PrioritySuggestion
from .adaptive_prioritization import (
    AdaptivePrioritizer,
    AdaptiveWeights,
    ExecutionOutcome,
    create_adaptive_prioritizer,
)
from .decomposition import TaskDecomposer, DecompositionSuggestion
from .assignment import SmartAssigner, WorkerProfile
from .anomaly import AnomalyDetector, AnomalyAlert
from .duration import DurationPredictor, DurationEstimate
from .descriptions import DescriptionGenerator
from .errors import ContextAwareErrorHandler, EnhancedError
from .learning import FailureLearner, FailurePattern
from .optimization import WorkflowOptimizer, OptimizationSuggestion
from .nlp import NaturalLanguageTaskParser, ParsedTask

__all__ = [
    # Prioritization
    "TaskPrioritizer",
    "PriorityFactors",
    "PrioritySuggestion",
    # Adaptive Prioritization
    "AdaptivePrioritizer",
    "AdaptiveWeights",
    "ExecutionOutcome",
    "create_adaptive_prioritizer",
    # Decomposition
    "TaskDecomposer",
    "DecompositionSuggestion",
    # Assignment
    "SmartAssigner",
    "WorkerProfile",
    # Anomaly Detection
    "AnomalyDetector",
    "AnomalyAlert",
    # Duration Prediction
    "DurationPredictor",
    "DurationEstimate",
    # Description Generation
    "DescriptionGenerator",
    # Error Handling
    "ContextAwareErrorHandler",
    "EnhancedError",
    # Failure Learning
    "FailureLearner",
    "FailurePattern",
    # Optimization
    "WorkflowOptimizer",
    "OptimizationSuggestion",
    # NLP
    "NaturalLanguageTaskParser",
    "ParsedTask",
]

__version__ = "1.0.0"
