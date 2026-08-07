#!/usr/bin/env bash
# Stop all vLLM serve workers started by serve_pool.sh (leaves other processes).
set -u
LOGDIR="${LOGDIR:-$(dirname "$0")/logs}"
rm -f "$LOGDIR/pool.map" 2>/dev/null
pids=$(ps -eo pid,cmd | grep -E "vllm serve" | grep -v grep | awk '{print $1}')
if [ -z "$pids" ]; then echo "[stop] no vllm serve found"; exit 0; fi
echo "[stop] TERM $pids"; kill -TERM $pids 2>/dev/null
for _ in $(seq 1 20); do
  sleep 2
  left=$(ps -eo pid,cmd | grep -E "vllm serve|VLLM::" | grep -v grep | grep -v defunct | awk '{print $1}')
  [ -z "$left" ] && { echo "[stop] all down"; exit 0; }
done
left=$(ps -eo pid,cmd | grep -E "vllm serve|VLLM::" | grep -v grep | grep -v defunct | awk '{print $1}')
[ -n "$left" ] && { echo "[stop] KILL $left"; kill -9 $left 2>/dev/null; }
echo "[stop] done"
