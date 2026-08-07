#!/usr/bin/env bash
# Full-benchmark baseline accuracy.
#   MODEL_KEY=qwen3vl_8b MODEL_PATH=/path/to/ckpt bash baseline/run.sh
# Or use a registry key under MODELS_ROOT:
#   bash baseline/run.sh --model qwen3vl_8b
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f configs/paths.env ] && source configs/paths.env
KEY="${MODEL_KEY:-qwen3vl_8b}"
python baseline/exp_baseline.py --model "$KEY" "$@"
