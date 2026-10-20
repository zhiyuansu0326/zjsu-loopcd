#!/usr/bin/env python3
"""Compatibility entry point: ARC-Easy is the unified evaluator's default task."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import main, summarize

if __name__ == "__main__":
    main()
