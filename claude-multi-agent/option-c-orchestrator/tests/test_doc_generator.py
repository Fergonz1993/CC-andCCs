"""
Test documentation generator.

Feature: adv-test-020 - Test documentation generator

This module provides utilities for generating documentation from
test files, including:
- Test case catalogs
- Coverage reports
- Test matrix generation
- API documentation from tests

Usage:
    from tests.test_doc_generator import TestDocGenerator

    generator = TestDocGenerator("tests/")
    generator.generate_catalog()
    generator.generate_markdown("TEST_CATALOG.md")
"""

import ast
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class TestCase:
    """Represents a single test case."""
    name: str
    module: str
    class_name: Optional[str]
    docstring: str
    markers: List[str]
    file_path: str
    line_number: int
    test_id: Optional[str] = None
    category: str = "general"
    is_async: bool = False

    @property
    def full_name(self) -> str:
        """Get the full test name including class."""
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name


@dataclass
class TestModule:
    """Represents a test module/file."""
    name: str
    file_path: str
    docstring: str
    test_cases: List[TestCase] = field(default_factory=list)
    test_classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    feature_id: Optional[str] = None


@dataclass
class TestCatalog:
    """Complete catalog of all tests."""
    modules: List[TestModule] = field(default_factory=list)
    total_tests: int = 0
    by_category: Dict[str, List[TestCase]] = field(default_factory=dict)
    by_marker: Dict[str, List[TestCase]] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class TestDocGenerator:
    """
    Generates documentation from test files.

    Parses Python test files using AST to extract:
    - Test function names and docstrings
    - Test class organization
    - Pytest markers
    - Module-level documentation
    """

    def __init__(self, test_dir: str = "tests"):
        self.test_dir = Path(test_dir)
        self.catalog = TestCatalog()
        self._parsed_files: Set[str] = set()

    def scan_tests(self) -> TestCatalog:
        """
        Scan all test files and build a catalog.

        Returns:
            TestCatalog with all discovered tests
        """
        for test_file in self.test_dir.glob("test_*.py"):
            if str(test_file) not in self._parsed_files:
                module = self._parse_test_file(test_file)
                if module:
                    self.catalog.modules.append(module)
                    self._parsed_files.add(str(test_file))

        # Build category and marker indexes
        self._build_indexes()

        return self.catalog

    def _parse_test_file(self, file_path: Path) -> Optional[TestModule]:
        """Parse a single test file."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Error parsing {file_path}: {e}")
            return None

        module = TestModule(
            name=file_path.stem,
            file_path=str(file_path),
            docstring=ast.get_docstring(tree) or "",
        )

        # Extract feature ID from docstring
        feature_match = re.search(r'Feature:\s*(\w+-\d+)', module.docstring)
        if feature_match:
            module.feature_id = feature_match.group(1)

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module.imports.append(f"{node.module}")

        # Extract test classes and functions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if node.name.startswith("Test"):
                    module.test_classes.append(node.name)
                    self._extract_tests_from_class(node, module, file_path)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if node.name.startswith("test_"):
                    test_case = self._extract_test_case(node, module, None, file_path)
                    module.test_cases.append(test_case)
                    self.catalog.total_tests += 1

        return module

    def _extract_tests_from_class(
        self,
        class_node: ast.ClassDef,
        module: TestModule,
        file_path: Path
    ) -> None:
        """Extract test methods from a class."""
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    test_case = self._extract_test_case(
                        node, module, class_node.name, file_path
                    )
                    module.test_cases.append(test_case)
                    self.catalog.total_tests += 1

    def _extract_test_case(
        self,
        node: ast.FunctionDef,
        module: TestModule,
        class_name: Optional[str],
        file_path: Path
    ) -> TestCase:
        """Extract a TestCase from a function node."""
        docstring = ast.get_docstring(node) or ""

        # Extract test ID from name or docstring
        test_id = None
        id_match = re.search(r'(\w+-\d+)', node.name)
        if id_match:
            test_id = id_match.group(1)
        else:
            id_match = re.search(r'^(\w+-\d+):', docstring)
            if id_match:
                test_id = id_match.group(1)

        # Extract category from class name or test name
        category = "general"
        if class_name:
            # TestTaskModel -> task_model
            category = re.sub(r'Test', '', class_name)
            category = re.sub(r'([A-Z])', r'_\1', category).lower().strip('_')
        elif '_' in node.name:
            # test_task_creation -> task
            parts = node.name.split('_')
            if len(parts) > 2:
                category = parts[1]

        # Extract markers from decorators
        markers = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute):
                if decorator.attr:
                    markers.append(decorator.attr)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    markers.append(decorator.func.attr)

        return TestCase(
            name=node.name,
            module=module.name,
            class_name=class_name,
            docstring=docstring,
            markers=markers,
            file_path=str(file_path),
            line_number=node.lineno,
            test_id=test_id,
            category=category,
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )

    def _build_indexes(self) -> None:
        """Build category and marker indexes."""
        self.catalog.by_category = defaultdict(list)
        self.catalog.by_marker = defaultdict(list)

        for module in self.catalog.modules:
            for test in module.test_cases:
                self.catalog.by_category[test.category].append(test)
                for marker in test.markers:
                    self.catalog.by_marker[marker].append(test)

    def generate_markdown(self, output_path: str) -> None:
        """
        Generate markdown documentation from the catalog.

        Args:
            output_path: Path to write the markdown file
        """
        if not self.catalog.modules:
            self.scan_tests()

        lines = [
            "# Test Documentation",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total Tests: {self.catalog.total_tests}",
            "",
            "## Table of Contents",
            "",
        ]

        # TOC
        for module in self.catalog.modules:
            lines.append(f"- [{module.name}](#{module.name.lower().replace('_', '-')})")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Module sections
        for module in self.catalog.modules:
            lines.extend(self._generate_module_section(module))

        # Category summary
        lines.append("## Tests by Category")
        lines.append("")
        for category, tests in sorted(self.catalog.by_category.items()):
            lines.append(f"### {category.replace('_', ' ').title()}")
            lines.append(f"*{len(tests)} tests*")
            lines.append("")
            for test in tests:
                lines.append(f"- `{test.full_name}`")
            lines.append("")

        Path(output_path).write_text("\n".join(lines))

    def _generate_module_section(self, module: TestModule) -> List[str]:
        """Generate markdown section for a module."""
        lines = [
            f"## {module.name}",
            "",
        ]

        if module.feature_id:
            lines.append(f"**Feature ID:** {module.feature_id}")
            lines.append("")

        if module.docstring:
            # First paragraph of docstring
            first_para = module.docstring.split("\n\n")[0]
            lines.append(first_para)
            lines.append("")

        lines.append(f"**Tests:** {len(module.test_cases)}")
        lines.append("")

        # Group tests by class
        by_class: Dict[str, List[TestCase]] = defaultdict(list)
        for test in module.test_cases:
            by_class[test.class_name or "Module Level"].append(test)

        for class_name, tests in by_class.items():
            if class_name != "Module Level":
                lines.append(f"### {class_name}")
                lines.append("")

            for test in tests:
                lines.extend(self._generate_test_entry(test))

        lines.append("---")
        lines.append("")

        return lines

    def _generate_test_entry(self, test: TestCase) -> List[str]:
        """Generate markdown entry for a test."""
        lines = [f"#### `{test.name}`"]

        if test.test_id:
            lines[0] += f" ({test.test_id})"

        lines.append("")

        if test.docstring:
            # Clean up docstring
            doc_lines = test.docstring.strip().split("\n")
            for line in doc_lines[:5]:  # First 5 lines
                lines.append(line.strip())
            lines.append("")

        if test.markers:
            lines.append(f"*Markers:* {', '.join(test.markers)}")
            lines.append("")

        lines.append(f"*Location:* `{test.file_path}:{test.line_number}`")
        lines.append("")

        return lines

    def generate_json(self, output_path: str) -> None:
        """
        Generate JSON documentation from the catalog.

        Args:
            output_path: Path to write the JSON file
        """
        if not self.catalog.modules:
            self.scan_tests()

        data = {
            "generated_at": self.catalog.generated_at,
            "total_tests": self.catalog.total_tests,
            "modules": [],
            "by_category": {},
            "by_marker": {},
        }

        for module in self.catalog.modules:
            module_data = {
                "name": module.name,
                "file_path": module.file_path,
                "docstring": module.docstring[:500] if module.docstring else "",
                "feature_id": module.feature_id,
                "test_count": len(module.test_cases),
                "tests": [
                    {
                        "name": t.name,
                        "full_name": t.full_name,
                        "class": t.class_name,
                        "test_id": t.test_id,
                        "category": t.category,
                        "markers": t.markers,
                        "is_async": t.is_async,
                        "line": t.line_number,
                        "docstring": t.docstring[:200] if t.docstring else "",
                    }
                    for t in module.test_cases
                ],
            }
            data["modules"].append(module_data)

        for category, tests in self.catalog.by_category.items():
            data["by_category"][category] = [t.full_name for t in tests]

        for marker, tests in self.catalog.by_marker.items():
            data["by_marker"][marker] = [t.full_name for t in tests]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_test_matrix(self) -> Dict[str, Any]:
        """
        Generate a test matrix showing coverage across features.

        Returns:
            Dictionary with test matrix data
        """
        if not self.catalog.modules:
            self.scan_tests()

        matrix = {
            "features": [],
            "categories": list(self.catalog.by_category.keys()),
            "matrix": [],
        }

        for module in self.catalog.modules:
            if module.feature_id:
                matrix["features"].append(module.feature_id)
                row = {
                    "feature": module.feature_id,
                    "module": module.name,
                    "tests": len(module.test_cases),
                    "by_category": {},
                }
                for test in module.test_cases:
                    row["by_category"][test.category] = (
                        row["by_category"].get(test.category, 0) + 1
                    )
                matrix["matrix"].append(row)

        return matrix


# =============================================================================
# CLI Support
# =============================================================================

def main():
    """Command-line interface for test documentation generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate documentation from test files"
    )
    parser.add_argument(
        "--test-dir",
        default="tests",
        help="Directory containing test files"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "--output",
        default="TEST_CATALOG",
        help="Output file name (without extension)"
    )

    args = parser.parse_args()

    generator = TestDocGenerator(args.test_dir)
    generator.scan_tests()

    print(f"Found {generator.catalog.total_tests} tests in "
          f"{len(generator.catalog.modules)} modules")

    if args.format in ("markdown", "both"):
        md_path = f"{args.output}.md"
        generator.generate_markdown(md_path)
        print(f"Generated: {md_path}")

    if args.format in ("json", "both"):
        json_path = f"{args.output}.json"
        generator.generate_json(json_path)
        print(f"Generated: {json_path}")


# =============================================================================
# Pytest Plugin for Documentation
# =============================================================================

class TestDocPlugin:
    """
    Pytest plugin for generating documentation during test runs.

    Install by adding to conftest.py:
        from tests.test_doc_generator import TestDocPlugin
        pytest_plugins = [TestDocPlugin]
    """

    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []

    def pytest_runtest_logreport(self, report):
        """Collect test results."""
        if report.when == "call":
            self.test_results.append({
                "name": report.nodeid,
                "outcome": report.outcome,
                "duration": report.duration,
            })

    def pytest_sessionfinish(self, session, exitstatus):
        """Generate documentation after tests complete."""
        # Could generate test report here
        pass


# =============================================================================
# Tests for the Documentation Generator
# =============================================================================

class TestTestDocGenerator:
    """Tests for the documentation generator itself."""

    def test_doc_gen_001_scan_tests(self, tmp_path):
        """doc-gen-001: Generator can scan test files."""
        # Create a minimal test file
        test_file = tmp_path / "test_example.py"
        test_file.write_text('''
"""Example test module."""

def test_example():
    """Example test case."""
    assert True
''')

        generator = TestDocGenerator(str(tmp_path))
        catalog = generator.scan_tests()

        assert catalog.total_tests == 1
        assert len(catalog.modules) == 1

    def test_doc_gen_002_extract_docstrings(self, tmp_path):
        """doc-gen-002: Generator extracts docstrings."""
        test_file = tmp_path / "test_docs.py"
        test_file.write_text('''
"""Module docstring."""

class TestClass:
    """Class docstring."""

    def test_method(self):
        """Method docstring."""
        pass
''')

        generator = TestDocGenerator(str(tmp_path))
        catalog = generator.scan_tests()

        assert "Module docstring" in catalog.modules[0].docstring
        assert "Method docstring" in catalog.modules[0].test_cases[0].docstring

    def test_doc_gen_003_extract_markers(self, tmp_path):
        """doc-gen-003: Generator extracts pytest markers."""
        test_file = tmp_path / "test_markers.py"
        test_file.write_text('''
import pytest

@pytest.mark.smoke
def test_marked():
    pass
''')

        generator = TestDocGenerator(str(tmp_path))
        catalog = generator.scan_tests()

        assert "smoke" in catalog.modules[0].test_cases[0].markers

    def test_doc_gen_004_generate_markdown(self, tmp_path):
        """doc-gen-004: Generator creates markdown output."""
        test_file = tmp_path / "test_md.py"
        test_file.write_text('''
def test_example():
    """Example test."""
    pass
''')

        generator = TestDocGenerator(str(tmp_path))
        generator.scan_tests()

        output_path = tmp_path / "catalog.md"
        generator.generate_markdown(str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "test_example" in content

    def test_doc_gen_005_generate_json(self, tmp_path):
        """doc-gen-005: Generator creates JSON output."""
        test_file = tmp_path / "test_json.py"
        test_file.write_text('''
def test_example():
    pass
''')

        generator = TestDocGenerator(str(tmp_path))
        generator.scan_tests()

        output_path = tmp_path / "catalog.json"
        generator.generate_json(str(output_path))

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "modules" in data
        assert data["total_tests"] == 1

    def test_doc_gen_006_category_indexing(self, tmp_path):
        """doc-gen-006: Generator indexes tests by category."""
        test_file = tmp_path / "test_categories.py"
        test_file.write_text('''
class TestModel:
    def test_create(self):
        pass

class TestAPI:
    def test_endpoint(self):
        pass
''')

        generator = TestDocGenerator(str(tmp_path))
        catalog = generator.scan_tests()

        assert "model" in catalog.by_category
        assert "api" in catalog.by_category

    def test_doc_gen_007_async_detection(self, tmp_path):
        """doc-gen-007: Generator detects async tests."""
        test_file = tmp_path / "test_async.py"
        test_file.write_text('''
async def test_async_example():
    pass

def test_sync_example():
    pass
''')

        generator = TestDocGenerator(str(tmp_path))
        catalog = generator.scan_tests()

        async_test = [t for t in catalog.modules[0].test_cases if "async" in t.name][0]
        sync_test = [t for t in catalog.modules[0].test_cases if "sync" in t.name][0]

        assert async_test.is_async is True
        assert sync_test.is_async is False


if __name__ == "__main__":
    main()
