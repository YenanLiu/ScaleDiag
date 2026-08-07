#!/usr/bin/env bash
# Experiment E: multi-sample (greedy / medoid / oracle@K).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f configs/paths.env ] && source configs/paths.env
KEY="${MODEL_KEY:-qwen3vl_8b}"
python sampling/exp_sample.py --model "$KEY" "$@"
