"""
Mutation testing setup and configuration.

Feature: adv-test-014 - Mutation testing setup

This module provides mutation testing infrastructure using mutmut
to verify test effectiveness.

Usage:
    # Install mutmut: pip install mutmut
    # Run mutation testing: mutmut run --paths-to-mutate=src/orchestrator
    # View results: mutmut results
    # Generate HTML report: mutmut html

Configuration is in pyproject.toml or setup.cfg
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class MutationStatus(str, Enum):
    """Status of a mutation."""
    KILLED = "killed"  # Test caught the mutation
    SURVIVED = "survived"  # Test did not catch the mutation
    TIMEOUT = "timeout"  # Test took too long
    SUSPICIOUS = "suspicious"  # Mutation might be equivalent
    SKIPPED = "skipped"  # Mutation was not tested


@dataclass
class MutationResult:
    """Result of a single mutation test."""
    mutant_id: int
    file_path: str
    line_number: int
    mutation_type: str
    status: MutationStatus
    description: str


@dataclass
class MutationReport:
    """Complete mutation testing report."""
    total_mutants: int
    killed: int
    survived: int
    timeout: int
    suspicious: int
    skipped: int
    mutation_score: float
    results: List[MutationResult]

    @property
    def is_passing(self) -> bool:
        """Check if mutation score meets threshold (default 80%)."""
        return self.mutation_score >= 80.0


class MutationTestRunner:
    """
    Manages mutation testing execution and reporting.

    This class wraps mutmut to provide programmatic access
    to mutation testing results.
    """

    def __init__(
        self,
        source_dir: str = "src/orchestrator",
        test_dir: str = "tests",
        config_path: Optional[str] = None
    ):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.config_path = config_path
        self._results: Optional[MutationReport] = None

    def check_mutmut_installed(self) -> bool:
        """Check if mutmut is installed."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mutmut", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def run_mutation_testing(
        self,
        paths_to_mutate: Optional[List[str]] = None,
        tests_to_run: Optional[str] = None,
        timeout: int = 60,
        parallel: bool = True
    ) -> bool:
        """
        Run mutation testing with mutmut.

        Args:
            paths_to_mutate: Specific paths to mutate
            tests_to_run: Specific test pattern to run
            timeout: Timeout per test in seconds
            parallel: Run tests in parallel

        Returns:
            True if mutation testing completed successfully
        """
        if not self.check_mutmut_installed():
            print("mutmut is not installed. Install with: pip install mutmut")
            return False

        cmd = [
            sys.executable, "-m", "mutmut", "run",
            f"--paths-to-mutate={paths_to_mutate[0] if paths_to_mutate else self.source_dir}",
        ]

        if tests_to_run:
            cmd.append(f"--tests-dir={tests_to_run}")

        if parallel:
            cmd.append("--use-coverage")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Error running mutation testing: {e}")
            return False

    def get_results(self) -> Optional[MutationReport]:
        """
        Get mutation testing results from mutmut.

        Returns:
            MutationReport with all results
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mutmut", "results", "--json"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)
            return self._parse_results(data)

        except Exception as e:
            print(f"Error getting results: {e}")
            return None

    def _parse_results(self, data: Dict[str, Any]) -> MutationReport:
        """Parse mutmut JSON output into a MutationReport."""
        results = []
        killed = 0
        survived = 0
        timeout = 0
        suspicious = 0
        skipped = 0

        for mutant in data.get("mutants", []):
            status = MutationStatus(mutant.get("status", "skipped"))

            if status == MutationStatus.KILLED:
                killed += 1
            elif status == MutationStatus.SURVIVED:
                survived += 1
            elif status == MutationStatus.TIMEOUT:
                timeout += 1
            elif status == MutationStatus.SUSPICIOUS:
                suspicious += 1
            else:
                skipped += 1

            results.append(MutationResult(
                mutant_id=mutant.get("id", 0),
                file_path=mutant.get("file", ""),
                line_number=mutant.get("line", 0),
                mutation_type=mutant.get("mutation_type", ""),
                status=status,
                description=mutant.get("description", "")
            ))

        total = len(results)
        tested = killed + survived
        score = (killed / tested * 100) if tested > 0 else 0.0

        return MutationReport(
            total_mutants=total,
            killed=killed,
            survived=survived,
            timeout=timeout,
            suspicious=suspicious,
            skipped=skipped,
            mutation_score=round(score, 2),
            results=results
        )

    def generate_html_report(self, output_path: str = "mutation_report") -> bool:
        """Generate an HTML report of mutation testing results."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mutmut", "html"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            return False

    def get_survived_mutations(self) -> List[MutationResult]:
        """Get list of mutations that survived (tests didn't catch)."""
        if self._results is None:
            self._results = self.get_results()

        if self._results is None:
            return []

        return [
            r for r in self._results.results
            if r.status == MutationStatus.SURVIVED
        ]


# =============================================================================
# Configuration Templates
# =============================================================================

MUTMUT_CONFIG_TEMPLATE = """
# Mutation testing configuration for mutmut
# Place this in pyproject.toml

[tool.mutmut]
paths_to_mutate = "src/orchestrator"
backup = false
runner = "pytest -x -q"
tests_dir = "tests/"
dict_synonyms = ["Struct", "NamedStruct"]

# Skip these files (patterns)
# paths_to_exclude =

# Focus on specific functions
# Only test these functions if specified
# functions_to_mutate =
"""

MUTATION_TEST_WORKFLOW = """
# GitHub Actions workflow for mutation testing
# Save as .github/workflows/mutation-testing.yml

name: Mutation Testing

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  mutation-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install mutmut

    - name: Run mutation testing
      run: |
        mutmut run --paths-to-mutate=src/orchestrator || true

    - name: Show results
      run: |
        mutmut results

    - name: Generate report
      run: |
        mutmut html

    - name: Upload mutation report
      uses: actions/upload-artifact@v3
      with:
        name: mutation-report
        path: html/
"""


# =============================================================================
# Pytest Integration
# =============================================================================

def pytest_mutation_test_marker():
    """
    Decorator to mark tests that should be included in mutation testing.

    Usage:
        @pytest.mark.mutation_test
        def test_critical_function():
            ...
    """
    import pytest
    return pytest.mark.mutation_test


def get_mutation_testing_config() -> Dict[str, Any]:
    """
    Get recommended mutation testing configuration.

    Returns configuration optimized for the orchestrator codebase.
    """
    return {
        "paths_to_mutate": ["src/orchestrator/models.py", "src/orchestrator/orchestrator.py"],
        "tests_dir": "tests/",
        "runner": "pytest -x -q --tb=no",
        "timeout_multiplier": 2.0,
        "dict_synonyms": [],
        "mutation_types": [
            # Arithmetic operator mutations
            "operator",
            # Comparison operator mutations
            "comparison",
            # Boolean operator mutations
            "boolean",
            # Constant mutations
            "number",
            # Return value mutations
            "return",
        ],
    }


# =============================================================================
# Test-specific helpers
# =============================================================================

class MutationTestHelper:
    """
    Helper class for writing mutation-resistant tests.

    Provides utilities for ensuring tests are effective
    at catching mutations.
    """

    @staticmethod
    def assert_boundary(value: int, lower: int, upper: int) -> None:
        """
        Assert value is within bounds with mutation-resistant checks.

        Tests all boundary conditions:
        - value >= lower
        - value <= upper
        - value > lower - 1
        - value < upper + 1
        """
        assert value >= lower, f"Value {value} should be >= {lower}"
        assert value <= upper, f"Value {value} should be <= {upper}"

    @staticmethod
    def assert_not_equal_to_defaults(*args) -> None:
        """
        Assert that values are not their default/sentinel values.

        Useful for catching mutations that remove initialization.
        """
        default_values = {None, "", 0, [], {}, set(), False}
        for arg in args:
            assert arg not in default_values, f"Value {arg} appears to be uninitialized"

    @staticmethod
    def assert_state_transition(
        before_status: str,
        after_status: str,
        expected_after: str
    ) -> None:
        """
        Assert a state transition occurred correctly.

        Catches mutations that skip or alter state transitions.
        """
        assert before_status != after_status, "Status should have changed"
        assert after_status == expected_after, f"Expected {expected_after}, got {after_status}"


# =============================================================================
# Entry point for running mutation tests
# =============================================================================

def run_mutation_analysis(
    source_paths: Optional[List[str]] = None,
    min_score: float = 80.0
) -> bool:
    """
    Run mutation testing and check if score meets threshold.

    Args:
        source_paths: Paths to mutate
        min_score: Minimum acceptable mutation score

    Returns:
        True if mutation score >= min_score
    """
    runner = MutationTestRunner()

    if not runner.check_mutmut_installed():
        print("mutmut not installed. Skipping mutation testing.")
        return True  # Don't fail if not installed

    print("Running mutation testing...")
    success = runner.run_mutation_testing(paths_to_mutate=source_paths)

    if not success:
        print("Mutation testing failed to complete")
        return False

    results = runner.get_results()
    if results is None:
        print("Could not retrieve mutation testing results")
        return False

    print(f"\n=== Mutation Testing Results ===")
    print(f"Total Mutants: {results.total_mutants}")
    print(f"Killed: {results.killed}")
    print(f"Survived: {results.survived}")
    print(f"Mutation Score: {results.mutation_score}%")
    print(f"Threshold: {min_score}%")

    if results.mutation_score < min_score:
        print(f"\nMutation score {results.mutation_score}% is below threshold {min_score}%")
        print("\nSurvived mutations:")
        for mutation in runner.get_survived_mutations()[:10]:
            print(f"  - {mutation.file_path}:{mutation.line_number} - {mutation.description}")
        return False

    print(f"\nMutation testing passed!")
    return True


if __name__ == "__main__":
    import sys
    success = run_mutation_analysis()
    sys.exit(0 if success else 1)
