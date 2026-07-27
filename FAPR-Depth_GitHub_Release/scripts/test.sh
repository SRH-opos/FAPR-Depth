#!/usr/bin/env bash
set -euo pipefail
export FAPR_PROJECT_ROOT="${FAPR_PROJECT_ROOT:-$(pwd)}"
python test.py --checkpoint-set main
