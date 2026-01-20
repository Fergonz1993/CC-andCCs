"""
Reliability patterns for Multi-Agent Coordination System.

This module provides shared reliability patterns that can be used across all options:
- Circuit breaker for external calls
- Automatic retry with jitter
- Fallback strategies
- Deadlock detection and recovery
- Data consistency validation
- Backup and restore functionality
- Leader election for high availability
- Split-brain prevention
- Graceful degradation modes
- Self-healing mechanisms
"""

from .circuit_breaker import CircuitBreaker, CircuitState
from .retry import RetryWithJitter, RetryConfig
from .fallback import FallbackStrategy, FallbackChain
from .deadlock import DeadlockDetector, DeadlockRecovery
from .consistency import DataConsistencyValidator, ConsistencyError
from .backup import BackupManager, BackupConfig
from .leader_election import LeaderElection, LeaderElectionConfig
from .split_brain import SplitBrainPrevention, SplitBrainConfig
from .degradation import GracefulDegradation, DegradationLevel
from .self_healing import SelfHealingManager, HealthCheck

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    # Retry
    "RetryWithJitter",
    "RetryConfig",
    # Fallback
    "FallbackStrategy",
    "FallbackChain",
    # Deadlock
    "DeadlockDetector",
    "DeadlockRecovery",
    # Consistency
    "DataConsistencyValidator",
    "ConsistencyError",
    # Backup
    "BackupManager",
    "BackupConfig",
    # Leader Election
    "LeaderElection",
    "LeaderElectionConfig",
    # Split Brain
    "SplitBrainPrevention",
    "SplitBrainConfig",
    # Degradation
    "GracefulDegradation",
    "DegradationLevel",
    # Self Healing
    "SelfHealingManager",
    "HealthCheck",
]
