"""Pytest configuration for the tracker test suite.

The application modules live at the repository root (e.g. ``ai_parallax_correction``),
so make sure that directory is importable when tests are collected from ``tests/``.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
