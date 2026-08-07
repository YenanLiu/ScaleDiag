#!/usr/bin/env bash
# Data prep for Experiment F: GPT image-grounded instruction rewrites.
# Needs OPENAI_API_KEY (and optionally OPENAI_BASE_URL).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f configs/paths.env ] && source configs/paths.env
: "${OPENAI_API_KEY:?export OPENAI_API_KEY}"
python instruction/rewrite_instructions.py \
  --datasets screenspot_v2 mmbench_gui osworld_g screenspot_pro ui_vision \
  --limit "${LIMIT:--1}" \
  --out instruction/rewrites_v2 \
  --concurrency "${CONCURRENCY:-16}" \
  "$@"
