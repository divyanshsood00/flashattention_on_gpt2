import os
import sys

# Ensure repository root is importable for `from test_utils import ...` imports.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
