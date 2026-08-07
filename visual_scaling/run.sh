#!/usr/bin/env bash
# Experiment A: aspect-preserving crop / visual scaling.
#   MODEL_KEY=qwen3vl_8b MODEL_PATH=/path/to/ckpt bash visual_scaling/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f configs/paths.env ] && source configs/paths.env
KEY="${MODEL_KEY:-qwen3vl_8b}"
python visual_scaling/exp_crop.py --model "$KEY" "$@"
# optional offline size analysis (needs crop json already written):
# python visual_scaling/analyze_target_size.py --model "$KEY"
