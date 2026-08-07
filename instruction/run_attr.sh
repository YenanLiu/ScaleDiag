#!/usr/bin/env bash
# Experiment B: inject position/color/size/shape into the instruction.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f configs/paths.env ] && source configs/paths.env
KEY="${MODEL_KEY:-qwen3vl_8b}"
python instruction/exp_instruction.py --model "$KEY" "$@"
