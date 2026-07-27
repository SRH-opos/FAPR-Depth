#!/usr/bin/env python3
"""Small launcher for `inference.py`.

Edit the three paths below, or run inference.py directly with command-line
arguments.
"""
from pathlib import Path
import subprocess
import sys

CACHE_SHARD = Path("sample_data/example.pt")
CHECKPOINT = Path("weights/best_candidate.pth")
BASE_SOURCE_ROOT = Path("third_party/FDCT-main")

if __name__ == "__main__":
    command = [
        sys.executable,
        "inference.py",
        "--cache-shard", str(CACHE_SHARD),
        "--checkpoint", str(CHECKPOINT),
        "--base-source-root", str(BASE_SOURCE_ROOT),
        "--output-dir", "outputs/sample_inference",
    ]
    raise SystemExit(subprocess.call(command))
