#!/usr/bin/env bash
# Experiment U: sample dispersion as an uncertainty signal.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f configs/paths.env ] && source configs/paths.env
KEY="${MODEL_KEY:-qwen3vl_8b}"
python sampling/exp_uncertainty.py --model "$KEY" "$@"
