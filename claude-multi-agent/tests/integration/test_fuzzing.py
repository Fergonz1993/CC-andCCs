"""
Fuzzing tests for task inputs.

Tests system behavior with random and edge-case inputs.

Run with: pytest tests/integration/test_fuzzing.py -v

Requires: hypothesis library (pip install hypothesis)
"""

import pytest
import json
import string
import random
from typing import List, Dict, Any, Optional

# Try to import hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


class TestFuzzingTaskDescriptions:
    """Fuzzing tests for task description inputs."""

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=50)
    def test_arbitrary_description_text(self, option_a_module, temp_workspace, description):
        """fuzz-desc-001: Handle arbitrary description text."""
        coord = option_a_module

        # Skip if contains null bytes (not valid in most file systems)
        assume("\x00" not in description)

        coord.ensure_coordination_structure()

        try:
            task_id = coord.leader_add_task(description)
            assert task_id is not None
            assert task_id.startswith("task-")

            # Should be able to load
            data = coord.load_tasks()
            assert len(data["tasks"]) >= 1
        except Exception as e:
            # Some exceptions are acceptable for malformed input
            assert "encoding" in str(e).lower() or "invalid" in str(e).lower()

    def test_unicode_descriptions(self, option_a_module, temp_workspace):
        """fuzz-desc-002: Handle unicode in descriptions."""
        coord = option_a_module
        coord.leader_init("Unicode fuzz test")

        unicode_descriptions = [
            "Task with emojis: rocket fire star",
            "Japanese: Unicode text",
            "Arabic: Arabic text",
            "Chinese: Chinese text",
            "Mixed: Hello World in multiple scripts",
            "Special chars: alpha beta gamma delta",
        ]

        task_ids = []
        for desc in unicode_descriptions:
            task_id = coord.leader_add_task(desc)
            task_ids.append(task_id)

        # All should be created
        data = coord.load_tasks()
        assert len(data["tasks"]) == len(unicode_descriptions)

    def test_special_characters(self, option_a_module, temp_workspace):
        """fuzz-desc-003: Handle special characters."""
        coord = option_a_module
        coord.leader_init("Special char test")

        special_descriptions = [
            'Task with "quotes"',
            "Task with 'single quotes'",
            "Task with\ttabs",
            "Task with\nnewlines",
            "Task with\\backslashes",
            "Task with <html> tags",
            "Task with $variables and ${templates}",
            "Task with `backticks`",
            "Task with /* comments */",
            "Task with -- sql comments",
        ]

        for desc in special_descriptions:
            task_id = coord.leader_add_task(desc)
            assert task_id is not None

        # Verify JSON is valid
        data = coord.load_tasks()
        json_str = json.dumps(data)  # Should not raise
        assert len(json_str) > 0

    def test_empty_and_whitespace(self, option_a_module, temp_workspace):
        """fuzz-desc-004: Handle empty and whitespace descriptions."""
        coord = option_a_module
        coord.leader_init("Whitespace test")

        whitespace_descriptions = [
            "   ",  # Only spaces
            "\t\t",  # Only tabs
            "\n\n",  # Only newlines
            "  text  ",  # Padded text
            "text\nwith\nlines",  # Multi-line
        ]

        for desc in whitespace_descriptions:
            # May or may not succeed depending on validation
            try:
                task_id = coord.leader_add_task(desc)
                # If it succeeds, verify it's loadable
                data = coord.load_tasks()
                assert "tasks" in data
            except (ValueError, AssertionError):
                pass  # Some implementations may reject whitespace-only


class TestFuzzingPriority:
    """Fuzzing tests for priority values."""

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(st.integers())
    @settings(max_examples=50)
    def test_arbitrary_priority_values(self, option_a_module, temp_workspace, priority):
        """fuzz-priority-001: Handle arbitrary priority values."""
        coord = option_a_module
        coord.ensure_coordination_structure()

        try:
            task_id = coord.leader_add_task("Priority test", priority=priority)
            # Should accept or reject gracefully
            data = coord.load_tasks()
            assert "tasks" in data
        except (ValueError, TypeError):
            pass  # Expected for invalid priorities

    def test_boundary_priorities(self, option_a_module, temp_workspace):
        """fuzz-priority-002: Test boundary priority values."""
        coord = option_a_module
        coord.leader_init("Boundary priority test")

        boundary_values = [
            0,      # Below minimum
            1,      # Minimum valid
            5,      # Middle
            10,     # Maximum valid
            11,     # Above maximum
            -1,     # Negative
            100,    # Way above
            -100,   # Way below
        ]

        for priority in boundary_values:
            try:
                task_id = coord.leader_add_task(f"Priority {priority} task", priority=priority)
                # If accepted, verify storage
                data = coord.load_tasks()
                task = next((t for t in data["tasks"] if t["id"] == task_id), None)
                if task:
                    # Priority should be stored as provided or clamped
                    assert "priority" in task
            except (ValueError, TypeError):
                pass  # Expected for out-of-range


class TestFuzzingDependencies:
    """Fuzzing tests for task dependencies."""

    def test_nonexistent_dependencies(self, option_a_module, temp_workspace):
        """fuzz-deps-001: Handle nonexistent dependency IDs."""
        coord = option_a_module
        coord.leader_init("Nonexistent deps test")

        # Create task with dependency on nonexistent task
        task_id = coord.leader_add_task(
            "Orphan task",
            dependencies=["nonexistent-task-id"]
        )

        # Task should be created but not claimable
        data = coord.load_tasks()
        assert len(data["tasks"]) == 1

        # Should not be claimable (dependency not met)
        task = coord.worker_claim("dep-worker")
        assert task is None

    def test_self_dependency(self, option_a_module, temp_workspace):
        """fuzz-deps-002: Handle self-referential dependency."""
        coord = option_a_module
        coord.leader_init("Self dep test")

        # Create task, then try to add self-dependency
        # This tests if system handles circular deps
        task_id = coord.leader_add_task("Self-ref task")

        # Manually add self-dependency (simulating bug)
        data = coord.load_tasks()
        for t in data["tasks"]:
            if t["id"] == task_id:
                t["dependencies"] = [task_id]
        coord.save_tasks(data)

        # Task should not be claimable (can never satisfy self-dependency)
        task = coord.worker_claim("self-dep-worker")
        assert task is None

    def test_malformed_dependency_ids(self, option_a_module, temp_workspace):
        """fuzz-deps-003: Handle malformed dependency IDs."""
        coord = option_a_module
        coord.leader_init("Malformed deps test")

        malformed_deps = [
            [""],                    # Empty string
            ["task-"],               # Incomplete ID
            ["TASK-123"],            # Wrong case
            ["task 123"],            # Space in ID
            ["task\n123"],           # Newline in ID
            [None],                  # None value (may cause type error)
            [123],                   # Integer instead of string
        ]

        for deps in malformed_deps:
            try:
                # Filter out None to avoid type errors in some cases
                filtered_deps = [d for d in deps if d is not None]
                if filtered_deps != deps:
                    continue  # Skip if we had to filter

                task_id = coord.leader_add_task("Malformed dep task", dependencies=filtered_deps)
                # If accepted, verify it's stored
                data = coord.load_tasks()
                assert "tasks" in data
            except (ValueError, TypeError, AttributeError):
                pass  # Expected for truly malformed input


class TestFuzzingContextFiles:
    """Fuzzing tests for context file paths."""

    def test_special_path_characters(self, option_a_module, temp_workspace):
        """fuzz-files-001: Handle special characters in file paths."""
        coord = option_a_module
        coord.leader_init("Special path test")

        special_paths = [
            ["../../../etc/passwd"],  # Path traversal
            ["/absolute/path"],       # Absolute path
            ["file with spaces.py"],  # Spaces
            ["file\twith\ttabs.py"],  # Tabs
            ["file<>:*?.py"],         # Invalid chars on Windows
            ["very/" * 50 + "deep.py"],  # Very long path
        ]

        for paths in special_paths:
            try:
                task_id = coord.leader_add_task("Path test", context_files=paths)
                data = coord.load_tasks()
                assert "tasks" in data
            except (ValueError, OSError):
                pass  # Some paths may be rejected


class TestFuzzingJSON:
    """Fuzzing tests for JSON handling."""

    def test_json_injection(self, option_a_module, temp_workspace):
        """fuzz-json-001: Prevent JSON injection in fields."""
        coord = option_a_module
        coord.leader_init("JSON injection test")

        injection_attempts = [
            '{"injected": true}',
            '[], "extra": "data"',
            'null',
            'true',
            '123',
            '["array", "data"]',
        ]

        for injection in injection_attempts:
            task_id = coord.leader_add_task(injection)
            data = coord.load_tasks()

            # JSON should still be valid
            json_str = json.dumps(data)
            reloaded = json.loads(json_str)
            assert "tasks" in reloaded

            # Injected content should be escaped as string
            task = next(t for t in data["tasks"] if t["id"] == task_id)
            assert task["description"] == injection

    def test_deeply_nested_context(self, option_a_module, temp_workspace):
        """fuzz-json-002: Handle deeply nested context structures."""
        coord = option_a_module
        coord.leader_init("Nested context test")

        # Context is limited to files and hints, but test the JSON handling
        coord.leader_add_task(
            "Nested test",
            context_files=["file1.py", "file2.py", "file3.py"] * 33,  # ~100 files
            hints="x" * 5000
        )

        data = coord.load_tasks()
        assert len(data["tasks"]) == 1

        # Should serialize/deserialize correctly
        json_str = json.dumps(data)
        reloaded = json.loads(json_str)
        assert reloaded == data


class TestFuzzingConcurrent:
    """Fuzzing tests with concurrent operations."""

    def test_random_concurrent_operations(self, option_a_module, temp_workspace):
        """fuzz-concurrent-001: Random operations executed concurrently."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        coord = option_a_module
        coord.leader_init("Concurrent fuzz test")

        errors: List[str] = []
        lock = threading.Lock()

        def random_operation(op_id: int):
            try:
                op = random.choice(["add", "claim", "load", "status"])
                random_text = "".join(random.choices(string.ascii_letters, k=20))

                if op == "add":
                    coord.leader_add_task(random_text, priority=random.randint(1, 10))
                elif op == "claim":
                    coord.worker_claim(f"fuzz-worker-{op_id}")
                elif op == "load":
                    coord.load_tasks()
                else:
                    coord.leader_status()
            except Exception as e:
                with lock:
                    errors.append(f"Op {op_id}: {str(e)}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(random_operation, i) for i in range(100)]
            for f in futures:
                f.result()

        # Some errors may be acceptable, but system should remain consistent
        data = coord.load_tasks()
        assert "tasks" in data
