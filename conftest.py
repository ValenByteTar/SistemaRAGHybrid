"""
Root conftest.py — centralizes sys.path configuration for the test suite.

This eliminates the need for sys.path hacks in individual test files.
The project currently uses a pragmatic transitional layout (root modules + src/).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"

for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)
