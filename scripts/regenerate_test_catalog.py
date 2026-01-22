#!/usr/bin/env python3
"""
Regenerate test catalog markdown from test files (ATOM-208).

Usage:
    ./scripts/regenerate_test_catalog.py [--format md|json|both] [--output PATH]

This script scans all test files in the project and generates documentation.
"""

import argparse
import sys
from pathlib import Path

# Add option-c-orchestrator to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
OPTION_C_TESTS = REPO_ROOT / "claude-multi-agent" / "option-c-orchestrator" / "tests"
sys.path.insert(0, str(OPTION_C_TESTS))

try:
    from test_doc_generator import TestDocGenerator
except ImportError:
    print("Error: Could not import TestDocGenerator from option-c-orchestrator/tests")
    print("Make sure the Option C environment is set up:")
    print("  cd claude-multi-agent/option-c-orchestrator && pip install -e .[dev]")
    sys.exit(1)


def scan_all_test_dirs() -> list[Path]:
    """Find all test directories in the project."""
    test_dirs = []

    # Option A tests
    option_a = REPO_ROOT / "claude-multi-agent" / "option-a-file-based" / "tests"
    if option_a.exists():
        test_dirs.append(option_a)

    # Option C tests
    if OPTION_C_TESTS.exists():
        test_dirs.append(OPTION_C_TESTS)

    # Integration tests
    integration = REPO_ROOT / "claude-multi-agent" / "tests" / "integration"
    if integration.exists():
        test_dirs.append(integration)

    return test_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate test catalog documentation"
    )
    parser.add_argument(
        "--format",
        choices=["md", "json", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "docs" / "TEST_CATALOG"),
        help="Output file path (without extension)"
    )
    parser.add_argument(
        "--test-dir",
        help="Specific test directory to scan (default: scan all)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.test_dir:
        test_dirs = [Path(args.test_dir)]
    else:
        test_dirs = scan_all_test_dirs()

    if not test_dirs:
        print("No test directories found")
        sys.exit(1)

    total_tests = 0
    total_modules = 0
    all_modules: list = []

    for test_dir in test_dirs:
        print(f"Scanning: {test_dir}")
        generator = TestDocGenerator(str(test_dir))
        catalog = generator.scan_tests()
        total_tests += catalog.total_tests
        total_modules += len(catalog.modules)
        all_modules.extend(catalog.modules)

    # Use the last generator for output (Option C tests if available)
    # but update its catalog with accumulated totals
    if OPTION_C_TESTS.exists() and OPTION_C_TESTS in test_dirs:
        generator = TestDocGenerator(str(OPTION_C_TESTS))
        generator.scan_tests()
        # Update the catalog with combined totals for summary reporting
        generator.catalog.total_tests = total_tests

    if args.format in ("md", "both"):
        md_path = f"{args.output}.md"
        generator.generate_markdown(md_path)
        print(f"Generated: {md_path}")

    if args.format in ("json", "both"):
        json_path = f"{args.output}.json"
        generator.generate_json(json_path)
        print(f"Generated: {json_path}")

    print(f"\nSummary:")
    print(f"  Test directories: {len(test_dirs)}")
    print(f"  Total tests found: {generator.catalog.total_tests}")
    print(f"  Total modules: {len(generator.catalog.modules)}")


if __name__ == "__main__":
    main()
