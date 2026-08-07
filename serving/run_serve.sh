#!/usr/bin/env bash
# Serve a grounding model with a multi-GPU vLLM pool.
# Usage:
#   MODEL=/path/to/ckpt GPUS="0 1 2 3" bash serving/run_serve.sh
# Optional: NAME, TP, BASE_PORT, MAXLEN, GPUFRAC, VLLM, LOGDIR
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f configs/paths.env ] && source configs/paths.env
: "${MODEL:?set MODEL=/path/to/checkpoint}"
export MODEL NAME="${NAME:-grounder}" GPUS="${GPUS:-0 1 2 3}" TP="${TP:-1}"
export BASE_PORT="${BASE_PORT:-8000}" VLLM="${VLLM:-vllm}"
bash serving/serve_pool.sh
echo "[ok] pool up; stop with: bash serving/stop_pool.sh"
