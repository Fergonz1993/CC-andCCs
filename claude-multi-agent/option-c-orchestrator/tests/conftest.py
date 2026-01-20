"""
Pytest configuration for orchestrator tests.
"""

import pytest
import sys
from pathlib import Path

# Ensure the src directory is in the path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
