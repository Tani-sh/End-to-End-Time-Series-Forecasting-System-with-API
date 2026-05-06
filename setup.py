"""
setup.py
========
One-shot training script — trains all 4 models for all 43 states and saves
artifacts. This is the first thing you run after installing requirements.

Usage
-----
    # Full training (all states, all models) — takes ~30-60 min on CPU
    python setup.py

    # Quick test: one state, no LSTM, no auto-search
    python setup.py --state Alabama --skip-lstm --fast

    # Skip LSTM only (saves ~20 min)
    python setup.py --skip-lstm
"""

import subprocess
import sys
from pathlib import Path

# Delegate to the trainer module
if __name__ == "__main__":
    args = sys.argv[1:]   # pass-through all arguments
    result = subprocess.run(
        [sys.executable, "-m", "src.trainer"] + args,
        cwd=Path(__file__).parent,
    )
    sys.exit(result.returncode)
