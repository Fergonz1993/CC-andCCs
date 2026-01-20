"""
Cross-Option Integration Module for Claude Multi-Agent Coordination System.

This module provides utilities for integrating and interoperating between
the three coordination options (A: File-based, B: MCP Server, C: Orchestrator).

Features:
- Universal task adapter interface (adv-cross-001)
- Option migration tools (adv-cross-002)
- Cross-option task synchronization (adv-cross-003)
- Unified CLI wrapper (adv-cross-004)
- Common logging format (adv-cross-005)
- Shared metrics interface (adv-cross-006)
- Plugin system for extensibility (adv-cross-007)
- Configuration schema validation (adv-cross-008)
- Compatibility testing framework (adv-cross-009)
- Feature parity checker (adv-cross-010)
"""

from .task_adapter import (
    UniversalTask,
    TaskAdapter,
    OptionAAdapter,
    OptionBAdapter,
    OptionCAdapter,
    AdapterFactory,
)

from .migration import (
    MigrationTool,
    MigrationResult,
    migrate_a_to_b,
    migrate_b_to_c,
    migrate_a_to_c,
    migrate_c_to_a,
    migrate_c_to_b,
    migrate_b_to_a,
)

from .sync import (
    TaskSynchronizer,
    SyncDirection,
    SyncResult,
    ConflictResolution,
)

from .unified_cli import (
    UnifiedCLI,
    CLIConfig,
)

from .logging_format import (
    CommonLogger,
    LogLevel,
    LogEntry,
    JSONLogFormatter,
    create_logger,
)

from .metrics import (
    MetricsCollector,
    MetricType,
    Metric,
    MetricsExporter,
)

from .plugins import (
    Plugin,
    PluginManager,
    PluginHook,
    PluginConfig,
)

from .config_validation import (
    ConfigValidator,
    ConfigSchema,
    ValidationResult,
    validate_config,
)

from .compatibility import (
    CompatibilityTest,
    CompatibilityRunner,
    CompatibilityReport,
)

from .feature_parity import (
    FeatureParityChecker,
    FeatureDefinition,
    ParityReport,
    FeatureStatus,
)

__all__ = [
    # Task Adapter (adv-cross-001)
    "UniversalTask",
    "TaskAdapter",
    "OptionAAdapter",
    "OptionBAdapter",
    "OptionCAdapter",
    "AdapterFactory",
    # Migration (adv-cross-002)
    "MigrationTool",
    "MigrationResult",
    "migrate_a_to_b",
    "migrate_b_to_c",
    "migrate_a_to_c",
    "migrate_c_to_a",
    "migrate_c_to_b",
    "migrate_b_to_a",
    # Sync (adv-cross-003)
    "TaskSynchronizer",
    "SyncDirection",
    "SyncResult",
    "ConflictResolution",
    # Unified CLI (adv-cross-004)
    "UnifiedCLI",
    "CLIConfig",
    # Logging (adv-cross-005)
    "CommonLogger",
    "LogLevel",
    "LogEntry",
    "JSONLogFormatter",
    "create_logger",
    # Metrics (adv-cross-006)
    "MetricsCollector",
    "MetricType",
    "Metric",
    "MetricsExporter",
    # Plugins (adv-cross-007)
    "Plugin",
    "PluginManager",
    "PluginHook",
    "PluginConfig",
    # Config Validation (adv-cross-008)
    "ConfigValidator",
    "ConfigSchema",
    "ValidationResult",
    "validate_config",
    # Compatibility Testing (adv-cross-009)
    "CompatibilityTest",
    "CompatibilityRunner",
    "CompatibilityReport",
    # Feature Parity (adv-cross-010)
    "FeatureParityChecker",
    "FeatureDefinition",
    "ParityReport",
    "FeatureStatus",
]

__version__ = "1.0.0"
