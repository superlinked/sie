"""Make the public source-only tools namespace importable under pytest's importlib mode."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
