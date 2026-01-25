"""
Centralized configuration for the Claude Multi-Agent Orchestrator.

This module provides a single source of truth for default values,
allowing easy updates and environment variable overrides.
"""

import os
from typing import Optional


# =============================================================================
# Model Configuration
# =============================================================================

DEFAULT_MODEL = os.environ.get(
    "CLAUDE_MODEL",
    "claude-sonnet-4-20250514"
)

# Alternative models that can be used
AVAILABLE_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-haiku-3-5-20250114",
]


# =============================================================================
# Orchestrator Defaults
# =============================================================================

DEFAULT_MAX_WORKERS = int(os.environ.get("ORCHESTRATOR_MAX_WORKERS", "3"))
DEFAULT_TASK_TIMEOUT = int(os.environ.get("ORCHESTRATOR_TASK_TIMEOUT", "600"))  # seconds


# =============================================================================
# Retry Configuration
# =============================================================================

DEFAULT_MAX_TASK_ATTEMPTS = int(os.environ.get("ORCHESTRATOR_MAX_ATTEMPTS", "3"))


# =============================================================================
# File Paths
# =============================================================================

DEFAULT_COORDINATION_DIR = os.environ.get(
    "COORDINATION_DIR",
    ".coordination"
)


# =============================================================================
# Reliability Configuration
# =============================================================================

# Circuit Breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.environ.get("CB_FAILURE_THRESHOLD", "3"))
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = int(os.environ.get("CB_SUCCESS_THRESHOLD", "2"))
CIRCUIT_BREAKER_TIMEOUT = float(os.environ.get("CB_TIMEOUT_SECONDS", "30.0"))
CIRCUIT_BREAKER_WINDOW = float(os.environ.get("CB_WINDOW_SECONDS", "60.0"))

# Main orchestrator circuit breaker (more tolerant)
MAIN_CB_FAILURE_THRESHOLD = int(os.environ.get("MAIN_CB_FAILURE_THRESHOLD", "5"))
MAIN_CB_TIMEOUT = float(os.environ.get("MAIN_CB_TIMEOUT_SECONDS", "60.0"))

# Retry Configuration
RETRY_MAX_ATTEMPTS = int(os.environ.get("RETRY_MAX_ATTEMPTS", "3"))
RETRY_INITIAL_DELAY = float(os.environ.get("RETRY_INITIAL_DELAY", "1.0"))
RETRY_MAX_DELAY = float(os.environ.get("RETRY_MAX_DELAY", "30.0"))
RETRY_JITTER_FACTOR = float(os.environ.get("RETRY_JITTER_FACTOR", "0.5"))

# Critical retry (more aggressive)
CRITICAL_RETRY_MAX_ATTEMPTS = int(os.environ.get("CRITICAL_RETRY_MAX_ATTEMPTS", "5"))
CRITICAL_RETRY_INITIAL_DELAY = float(os.environ.get("CRITICAL_RETRY_INITIAL_DELAY", "0.5"))
CRITICAL_RETRY_MAX_DELAY = float(os.environ.get("CRITICAL_RETRY_MAX_DELAY", "10.0"))
CRITICAL_RETRY_JITTER_FACTOR = float(os.environ.get("CRITICAL_RETRY_JITTER_FACTOR", "0.3"))

# Fallback
CACHE_FALLBACK_MAX_AGE = float(os.environ.get("CACHE_FALLBACK_MAX_AGE", "300.0"))

# Deadlock Detection
DEADLOCK_STALE_THRESHOLD = float(os.environ.get("DEADLOCK_STALE_THRESHOLD", "300.0"))
DEADLOCK_CHECK_INTERVAL = float(os.environ.get("DEADLOCK_CHECK_INTERVAL", "30.0"))

# Backup Configuration
BACKUP_MAX_COUNT = int(os.environ.get("BACKUP_MAX_COUNT", "10"))
BACKUP_AUTO_INTERVAL = float(os.environ.get("BACKUP_AUTO_INTERVAL", "300.0"))

# Leader Election
LEADER_HEARTBEAT_INTERVAL = float(os.environ.get("LEADER_HEARTBEAT_INTERVAL", "5.0"))
LEADER_ELECTION_TIMEOUT = float(os.environ.get("LEADER_ELECTION_TIMEOUT", "15.0"))


def get_model(override: Optional[str] = None) -> str:
    """
    Get the model to use, with optional override.

    Priority: override > environment variable > default
    """
    if override:
        return override
    return DEFAULT_MODEL
