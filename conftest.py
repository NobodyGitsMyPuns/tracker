import os
import sys

# Ensure the repository root (where the modules under test live) is importable
# regardless of pytest's import mode or the directory tests are collected from.
sys.path.insert(0, os.path.dirname(__file__))
