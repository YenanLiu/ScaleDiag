#!/usr/bin/env bash
# Experiment F: evaluate the rewritten multi-perspective instructions.
# Requires instruction/rewrites_v2/all_instructions.json from run_rewrite_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f configs/paths.env ] && source configs/paths.env
KEY="${MODEL_KEY:-qwen3vl_8b}"
python instruction/exp_instruction_v2.py --model "$KEY" "$@"
