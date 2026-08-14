#!/usr/bin/env python3
"""Compatibility wrapper for Conan HuBERT preprocessing.

Use ``scripts/binarize_conan.py`` for the config-driven batched ONNX path.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.binarize_conan import main


if __name__ == "__main__":
    main()
