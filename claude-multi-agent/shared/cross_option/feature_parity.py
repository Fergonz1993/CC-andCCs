"""
Feature Parity Checker (adv-cross-010)

Provides tools to check and report on feature parity between the
different coordination options. Helps identify missing features
and differences in implementation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


class FeatureStatus(Enum):
    """Status of a feature implementation."""
    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    MISSING = "missing"
    DEPRECATED = "deprecated"
    PLANNED = "planned"


class FeatureCategory(Enum):
    """Categories of features."""
    CORE = "core"
    TASK_MANAGEMENT = "task_management"
    AGENT_MANAGEMENT = "agent_management"
    DISCOVERY = "discovery"
    METRICS = "metrics"
    PERSISTENCE = "persistence"
    SECURITY = "security"
    PERFORMANCE = "performance"
    EXTENSIBILITY = "extensibility"


@dataclass
class FeatureDefinition:
    """Definition of a coordination feature."""
    id: str
    name: str
    description: str
    category: FeatureCategory
    required: bool = True
    version_added: str = "1.0.0"
    depends_on: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "required": self.required,
            "version_added": self.version_added,
            "depends_on": self.depends_on,
            "notes": self.notes,
        }


@dataclass
class FeatureImplementation:
    """Implementation status of a feature for a specific option."""
    feature_id: str
    option: str
    status: FeatureStatus
    version: str = ""
    notes: str = ""
    limitations: List[str] = field(default_factory=list)
    api_differences: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "feature_id": self.feature_id,
            "option": self.option,
            "status": self.status.value,
        }
        if self.version:
            d["version"] = self.version
        if self.notes:
            d["notes"] = self.notes
        if self.limitations:
            d["limitations"] = self.limitations
        if self.api_differences:
            d["api_differences"] = self.api_differences
        return d


@dataclass
class ParityResult:
    """Result of a parity check for a single feature."""
    feature: FeatureDefinition
    implementations: Dict[str, FeatureImplementation]
    has_parity: bool
    missing_options: List[str] = field(default_factory=list)
    partial_options: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature.to_dict(),
            "implementations": {
                opt: impl.to_dict()
                for opt, impl in self.implementations.items()
            },
            "has_parity": self.has_parity,
            "missing_options": self.missing_options,
            "partial_options": self.partial_options,
            "differences": self.differences,
        }


@dataclass
class ParityReport:
    """Complete feature parity report."""
    total_features: int = 0
    features_with_parity: int = 0
    features_partial_parity: int = 0
    features_missing_parity: int = 0
    option_coverage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    results: List[ParityResult] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def parity_percentage(self) -> float:
        if self.total_features == 0:
            return 0.0
        return self.features_with_parity / self.total_features * 100

    def get_option_coverage(self, option: str) -> float:
        """Get coverage percentage for an option."""
        coverage = self.option_coverage.get(option, {})
        total = coverage.get("total", 0)
        implemented = coverage.get("implemented", 0)
        if total == 0:
            return 0.0
        return implemented / total * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_features": self.total_features,
                "features_with_parity": self.features_with_parity,
                "features_partial_parity": self.features_partial_parity,
                "features_missing_parity": self.features_missing_parity,
                "parity_percentage": f"{self.parity_percentage:.1f}%",
            },
            "option_coverage": self.option_coverage,
            "generated_at": self.generated_at.isoformat(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, pretty: bool = True) -> str:
        if pretty:
            return json.dumps(self.to_dict(), indent=2)
        return json.dumps(self.to_dict())

    def to_markdown(self) -> str:
        """Generate a markdown report."""
        lines = [
            "# Feature Parity Report",
            "",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- Total Features: {self.total_features}",
            f"- Full Parity: {self.features_with_parity} ({self.parity_percentage:.1f}%)",
            f"- Partial Parity: {self.features_partial_parity}",
            f"- Missing Parity: {self.features_missing_parity}",
            "",
            "## Option Coverage",
            "",
            "| Option | Implemented | Partial | Missing | Coverage |",
            "|--------|-------------|---------|---------|----------|",
        ]

        for option in ["A", "B", "C"]:
            coverage = self.option_coverage.get(option, {})
            impl = coverage.get("implemented", 0)
            partial = coverage.get("partial", 0)
            missing = coverage.get("missing", 0)
            pct = self.get_option_coverage(option)
            lines.append(f"| {option} | {impl} | {partial} | {missing} | {pct:.1f}% |")

        lines.extend(["", "## Feature Details", ""])

        # Group by category
        by_category: Dict[str, List[ParityResult]] = {}
        for result in self.results:
            cat = result.feature.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)

        for category, results in sorted(by_category.items()):
            lines.extend([
                f"### {category.replace('_', ' ').title()}",
                "",
                "| Feature | Option A | Option B | Option C | Parity |",
                "|---------|----------|----------|----------|--------|",
            ])

            for result in results:
                status_a = self._format_status(result.implementations.get("A"))
                status_b = self._format_status(result.implementations.get("B"))
                status_c = self._format_status(result.implementations.get("C"))
                parity = "Yes" if result.has_parity else "No"
                lines.append(f"| {result.feature.name} | {status_a} | {status_b} | {status_c} | {parity} |")

            lines.append("")

        # Missing features section
        missing_results = [r for r in self.results if r.missing_options]
        if missing_results:
            lines.extend([
                "## Missing Features",
                "",
                "Features that are not implemented in all options:",
                "",
            ])
            for result in missing_results:
                lines.append(f"- **{result.feature.name}**: Missing in {', '.join(result.missing_options)}")
                if result.differences:
                    for diff in result.differences:
                        lines.append(f"  - {diff}")
            lines.append("")

        return "\n".join(lines)

    def _format_status(self, impl: Optional[FeatureImplementation]) -> str:
        """Format a feature status for the table."""
        if impl is None:
            return "N/A"
        symbols = {
            FeatureStatus.IMPLEMENTED: "Yes",
            FeatureStatus.PARTIAL: "Partial",
            FeatureStatus.MISSING: "No",
            FeatureStatus.DEPRECATED: "Deprecated",
            FeatureStatus.PLANNED: "Planned",
        }
        return symbols.get(impl.status, "?")


class FeatureParityChecker:
    """
    Checks feature parity between coordination options.

    Features are defined in a registry and implementations are
    tracked for each option.
    """

    def __init__(self):
        """Initialize the checker."""
        self._features: Dict[str, FeatureDefinition] = {}
        self._implementations: Dict[str, Dict[str, FeatureImplementation]] = {}

        # Register standard features
        self._register_standard_features()

    def register_feature(self, feature: FeatureDefinition) -> None:
        """Register a feature definition."""
        self._features[feature.id] = feature

    def register_implementation(self, impl: FeatureImplementation) -> None:
        """Register a feature implementation."""
        if impl.feature_id not in self._implementations:
            self._implementations[impl.feature_id] = {}
        self._implementations[impl.feature_id][impl.option] = impl

    def get_feature(self, feature_id: str) -> Optional[FeatureDefinition]:
        """Get a feature by ID."""
        return self._features.get(feature_id)

    def get_features_by_category(
        self,
        category: FeatureCategory,
    ) -> List[FeatureDefinition]:
        """Get all features in a category."""
        return [f for f in self._features.values() if f.category == category]

    def check_parity(
        self,
        feature_ids: Optional[List[str]] = None,
        options: Optional[List[str]] = None,
    ) -> ParityReport:
        """
        Check feature parity.

        Args:
            feature_ids: Specific features to check (all if None)
            options: Options to compare (default: A, B, C)

        Returns:
            ParityReport with results
        """
        options = options or ["A", "B", "C"]
        features_to_check = (
            [self._features[fid] for fid in feature_ids if fid in self._features]
            if feature_ids
            else list(self._features.values())
        )

        report = ParityReport()
        report.total_features = len(features_to_check)

        # Initialize option coverage
        for opt in options:
            report.option_coverage[opt] = {
                "total": len(features_to_check),
                "implemented": 0,
                "partial": 0,
                "missing": 0,
            }

        for feature in features_to_check:
            implementations = {}
            missing_options = []
            partial_options = []
            differences = []

            for opt in options:
                impl = self._implementations.get(feature.id, {}).get(opt)
                if impl:
                    implementations[opt] = impl

                    if impl.status == FeatureStatus.IMPLEMENTED:
                        report.option_coverage[opt]["implemented"] += 1
                    elif impl.status == FeatureStatus.PARTIAL:
                        report.option_coverage[opt]["partial"] += 1
                        partial_options.append(opt)
                        if impl.limitations:
                            for lim in impl.limitations:
                                differences.append(f"Option {opt}: {lim}")
                    elif impl.status == FeatureStatus.MISSING:
                        report.option_coverage[opt]["missing"] += 1
                        missing_options.append(opt)
                else:
                    # No implementation record = missing
                    missing_options.append(opt)
                    report.option_coverage[opt]["missing"] += 1

            # Determine parity
            has_parity = len(missing_options) == 0 and len(partial_options) == 0

            if has_parity:
                report.features_with_parity += 1
            elif missing_options:
                report.features_missing_parity += 1
            else:
                report.features_partial_parity += 1

            result = ParityResult(
                feature=feature,
                implementations=implementations,
                has_parity=has_parity,
                missing_options=missing_options,
                partial_options=partial_options,
                differences=differences,
            )
            report.results.append(result)

        return report

    def check_option_parity(
        self,
        option1: str,
        option2: str,
    ) -> List[ParityResult]:
        """Check parity between two specific options."""
        results = []

        for feature in self._features.values():
            impl1 = self._implementations.get(feature.id, {}).get(option1)
            impl2 = self._implementations.get(feature.id, {}).get(option2)

            missing = []
            partial = []
            differences = []

            if impl1 and not impl2:
                missing.append(option2)
            elif impl2 and not impl1:
                missing.append(option1)
            elif impl1 and impl2:
                # Both have implementations - check status
                if impl1.status != impl2.status:
                    if impl1.status == FeatureStatus.PARTIAL:
                        partial.append(option1)
                    if impl2.status == FeatureStatus.PARTIAL:
                        partial.append(option2)
                    if impl1.status == FeatureStatus.MISSING:
                        missing.append(option1)
                    if impl2.status == FeatureStatus.MISSING:
                        missing.append(option2)

                # Check API differences
                if impl1.api_differences or impl2.api_differences:
                    all_apis = set(impl1.api_differences.keys()) | set(impl2.api_differences.keys())
                    for api in all_apis:
                        api1 = impl1.api_differences.get(api)
                        api2 = impl2.api_differences.get(api)
                        if api1 != api2:
                            differences.append(f"API '{api}': {option1}={api1}, {option2}={api2}")

            has_parity = not missing and not partial and not differences

            results.append(ParityResult(
                feature=feature,
                implementations={
                    option1: impl1,
                    option2: impl2,
                } if impl1 or impl2 else {},
                has_parity=has_parity,
                missing_options=missing,
                partial_options=partial,
                differences=differences,
            ))

        return results

    def _register_standard_features(self) -> None:
        """Register the standard set of coordination features."""
        # Core features
        core_features = [
            FeatureDefinition(
                id="init_coordination",
                name="Initialize Coordination",
                description="Initialize a new coordination session with goal and plan",
                category=FeatureCategory.CORE,
            ),
            FeatureDefinition(
                id="task_create",
                name="Create Task",
                description="Create a new task in the queue",
                category=FeatureCategory.TASK_MANAGEMENT,
            ),
            FeatureDefinition(
                id="task_claim",
                name="Claim Task",
                description="Claim an available task for execution",
                category=FeatureCategory.TASK_MANAGEMENT,
            ),
            FeatureDefinition(
                id="task_complete",
                name="Complete Task",
                description="Mark a task as completed with results",
                category=FeatureCategory.TASK_MANAGEMENT,
            ),
            FeatureDefinition(
                id="task_fail",
                name="Fail Task",
                description="Mark a task as failed with error",
                category=FeatureCategory.TASK_MANAGEMENT,
            ),
            FeatureDefinition(
                id="task_dependencies",
                name="Task Dependencies",
                description="Support for task dependencies",
                category=FeatureCategory.TASK_MANAGEMENT,
            ),
            FeatureDefinition(
                id="task_priority",
                name="Task Priority",
                description="Task priority-based ordering",
                category=FeatureCategory.TASK_MANAGEMENT,
            ),
            FeatureDefinition(
                id="agent_register",
                name="Agent Registration",
                description="Register agents (leader/worker)",
                category=FeatureCategory.AGENT_MANAGEMENT,
            ),
            FeatureDefinition(
                id="agent_heartbeat",
                name="Agent Heartbeat",
                description="Agent health check heartbeats",
                category=FeatureCategory.AGENT_MANAGEMENT,
            ),
            FeatureDefinition(
                id="discovery_add",
                name="Add Discovery",
                description="Share discoveries between agents",
                category=FeatureCategory.DISCOVERY,
            ),
            FeatureDefinition(
                id="discovery_list",
                name="List Discoveries",
                description="List shared discoveries",
                category=FeatureCategory.DISCOVERY,
            ),
            FeatureDefinition(
                id="state_persistence",
                name="State Persistence",
                description="Persist state to disk",
                category=FeatureCategory.PERSISTENCE,
            ),
            FeatureDefinition(
                id="status_report",
                name="Status Report",
                description="Get coordination status and progress",
                category=FeatureCategory.CORE,
            ),
            FeatureDefinition(
                id="batch_task_create",
                name="Batch Task Creation",
                description="Create multiple tasks at once",
                category=FeatureCategory.TASK_MANAGEMENT,
                required=False,
            ),
            FeatureDefinition(
                id="task_context",
                name="Task Context",
                description="Attach context (files, hints) to tasks",
                category=FeatureCategory.TASK_MANAGEMENT,
            ),
            FeatureDefinition(
                id="result_aggregation",
                name="Result Aggregation",
                description="Aggregate results from completed tasks",
                category=FeatureCategory.CORE,
            ),
            FeatureDefinition(
                id="file_locking",
                name="File Locking",
                description="Thread-safe file operations",
                category=FeatureCategory.PERSISTENCE,
            ),
            FeatureDefinition(
                id="race_condition_handling",
                name="Race Condition Handling",
                description="Handle concurrent task claims",
                category=FeatureCategory.PERSISTENCE,
            ),
        ]

        for feature in core_features:
            self.register_feature(feature)

        # Register implementations for all options
        self._register_option_a_implementations()
        self._register_option_b_implementations()
        self._register_option_c_implementations()

    def _register_option_a_implementations(self) -> None:
        """Register Option A implementations."""
        implementations = [
            ("init_coordination", FeatureStatus.IMPLEMENTED),
            ("task_create", FeatureStatus.IMPLEMENTED),
            ("task_claim", FeatureStatus.IMPLEMENTED),
            ("task_complete", FeatureStatus.IMPLEMENTED),
            ("task_fail", FeatureStatus.IMPLEMENTED),
            ("task_dependencies", FeatureStatus.IMPLEMENTED),
            ("task_priority", FeatureStatus.IMPLEMENTED),
            ("agent_register", FeatureStatus.IMPLEMENTED),
            ("agent_heartbeat", FeatureStatus.PARTIAL, ["No automatic timeout detection"]),
            ("discovery_add", FeatureStatus.IMPLEMENTED),
            ("discovery_list", FeatureStatus.IMPLEMENTED),
            ("state_persistence", FeatureStatus.IMPLEMENTED),
            ("status_report", FeatureStatus.IMPLEMENTED),
            ("batch_task_create", FeatureStatus.MISSING),
            ("task_context", FeatureStatus.IMPLEMENTED),
            ("result_aggregation", FeatureStatus.IMPLEMENTED),
            ("file_locking", FeatureStatus.IMPLEMENTED),
            ("race_condition_handling", FeatureStatus.IMPLEMENTED),
        ]

        for item in implementations:
            feature_id = item[0]
            status = item[1]
            limitations = item[2] if len(item) > 2 else []

            self.register_implementation(FeatureImplementation(
                feature_id=feature_id,
                option="A",
                status=status,
                limitations=limitations,
            ))

    def _register_option_b_implementations(self) -> None:
        """Register Option B implementations."""
        implementations = [
            ("init_coordination", FeatureStatus.IMPLEMENTED),
            ("task_create", FeatureStatus.IMPLEMENTED),
            ("task_claim", FeatureStatus.IMPLEMENTED),
            ("task_complete", FeatureStatus.IMPLEMENTED),
            ("task_fail", FeatureStatus.IMPLEMENTED),
            ("task_dependencies", FeatureStatus.IMPLEMENTED),
            ("task_priority", FeatureStatus.IMPLEMENTED),
            ("agent_register", FeatureStatus.IMPLEMENTED),
            ("agent_heartbeat", FeatureStatus.IMPLEMENTED),
            ("discovery_add", FeatureStatus.IMPLEMENTED),
            ("discovery_list", FeatureStatus.IMPLEMENTED),
            ("state_persistence", FeatureStatus.IMPLEMENTED),
            ("status_report", FeatureStatus.IMPLEMENTED),
            ("batch_task_create", FeatureStatus.IMPLEMENTED),
            ("task_context", FeatureStatus.IMPLEMENTED),
            ("result_aggregation", FeatureStatus.IMPLEMENTED),
            ("file_locking", FeatureStatus.PARTIAL, ["Single-process only via MCP"]),
            ("race_condition_handling", FeatureStatus.IMPLEMENTED),
        ]

        for item in implementations:
            feature_id = item[0]
            status = item[1]
            limitations = item[2] if len(item) > 2 else []

            self.register_implementation(FeatureImplementation(
                feature_id=feature_id,
                option="B",
                status=status,
                limitations=limitations,
            ))

    def _register_option_c_implementations(self) -> None:
        """Register Option C implementations."""
        implementations = [
            ("init_coordination", FeatureStatus.IMPLEMENTED),
            ("task_create", FeatureStatus.IMPLEMENTED),
            ("task_claim", FeatureStatus.IMPLEMENTED),
            ("task_complete", FeatureStatus.IMPLEMENTED),
            ("task_fail", FeatureStatus.IMPLEMENTED),
            ("task_dependencies", FeatureStatus.IMPLEMENTED),
            ("task_priority", FeatureStatus.IMPLEMENTED),
            ("agent_register", FeatureStatus.IMPLEMENTED),
            ("agent_heartbeat", FeatureStatus.IMPLEMENTED),
            ("discovery_add", FeatureStatus.IMPLEMENTED),
            ("discovery_list", FeatureStatus.IMPLEMENTED),
            ("state_persistence", FeatureStatus.IMPLEMENTED),
            ("status_report", FeatureStatus.IMPLEMENTED),
            ("batch_task_create", FeatureStatus.IMPLEMENTED),
            ("task_context", FeatureStatus.IMPLEMENTED),
            ("result_aggregation", FeatureStatus.IMPLEMENTED),
            ("file_locking", FeatureStatus.IMPLEMENTED),
            ("race_condition_handling", FeatureStatus.IMPLEMENTED),
        ]

        for item in implementations:
            feature_id = item[0]
            status = item[1]
            limitations = item[2] if len(item) > 2 else []

            self.register_implementation(FeatureImplementation(
                feature_id=feature_id,
                option="C",
                status=status,
                limitations=limitations,
            ))


def check_feature_parity() -> ParityReport:
    """
    Check feature parity across all options.

    Returns:
        ParityReport with complete parity analysis
    """
    checker = FeatureParityChecker()
    return checker.check_parity()


def get_missing_features(option: str) -> List[FeatureDefinition]:
    """
    Get list of features missing in a specific option.

    Args:
        option: Option to check ('A', 'B', or 'C')

    Returns:
        List of missing features
    """
    checker = FeatureParityChecker()
    report = checker.check_parity()

    missing = []
    for result in report.results:
        if option in result.missing_options:
            missing.append(result.feature)

    return missing
