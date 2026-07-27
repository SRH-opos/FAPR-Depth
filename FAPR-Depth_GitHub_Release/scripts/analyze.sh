#!/usr/bin/env bash
set -euo pipefail
python analysis/analyze_failure_posterior_fixed.py
python analysis/visualize_expert_routing_paper.py
python analysis/analyze_risk_calibration_fixed.py
