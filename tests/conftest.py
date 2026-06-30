"""Make the project root importable so tests can `import config`, `import db`, etc.

Lets the suite run with a bare `pytest` from anywhere, not just `python -m pytest`
from the repo root.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
