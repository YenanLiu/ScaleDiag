#!/usr/bin/env bash
# Experiment D: translate / language-ladder conditions.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f configs/paths.env ] && source configs/paths.env
KEY="${MODEL_KEY:-qwen3vl_8b}"
python instruction/exp_language.py --model "$KEY" "$@"
