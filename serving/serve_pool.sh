#!/usr/bin/env bash
# Launch a data-parallel pool of vLLM servers for a grounding model so the
# evaluation client can load-balance across them. With TP=1 we start one worker
# per GPU in GPUS; with TP=2 we pair GPUs (0,1)(2,3)... into fewer workers.
# Ports are BASE_PORT+1, BASE_PORT+2, ... served under --served-model-name grounder.
set -u

MODEL="${MODEL:?set MODEL to the checkpoint path}"
NAME="${NAME:-grounder}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
TP="${TP:-1}"
BASE_PORT="${BASE_PORT:-8000}"
MAXLEN="${MAXLEN:-16384}"
GPUFRAC="${GPUFRAC:-0.85}"
LOGDIR="${LOGDIR:-$(dirname "$0")/logs}"
VLLM="${VLLM:-vllm}"
# Prefer the vLLM env's libs when VLLM is an absolute path into a conda/venv.
if [[ "$VLLM" == /* ]]; then
  ENV_PREFIX="$(dirname "$(dirname "$VLLM")")"
  export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
mkdir -p "$LOGDIR"

# Launch one server. Usage: launch_one <gpu_group> <port>
launch_one() {
  local group="$1" port="$2"
  echo "[launch] GPUs $group -> port $port (TP=$TP)"
  CUDA_VISIBLE_DEVICES=$group "$VLLM" serve "$MODEL" \
    --host 0.0.0.0 --port "$port" --served-model-name "$NAME" \
    --tensor-parallel-size "$TP" --trust-remote-code --dtype bfloat16 \
    --max-model-len "$MAXLEN" --limit-mm-per-prompt.video 0 \
    --enable-prefix-caching \
    --gpu-memory-utilization "$GPUFRAC" \
    > "$LOGDIR/serve_${port}.log" 2>&1 &
}

read -ra GARR <<< "$GPUS"
n=${#GARR[@]}
# pool.map = "port\tgpu_group" (used by the start-up health wait / stop_pool).
: > "$LOGDIR/pool.map"

i=0; port_i=0
while [ $i -lt $n ]; do
  group=$(IFS=,; echo "${GARR[*]:$i:$TP}")
  port_i=$((port_i+1))
  port=$((BASE_PORT + port_i))
  printf '%s\t%s\n' "$port" "$group" >> "$LOGDIR/pool.map"
  launch_one "$group" "$port"
  i=$((i+TP))
done
echo "[launched] $port_i servers on ports $((BASE_PORT+1))..$((BASE_PORT+port_i)); logs in $LOGDIR"

# Initial heal: wait up to ~10min for every port to answer /health, relaunching
# any that die during warm-up, so the pool starts complete.
deadline=$(( $(date +%s) + 600 ))
while :; do
  missing=0
  while IFS=$'\t' read -r port group; do
    [ -z "$port" ] && continue
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://127.0.0.1:$port/health")
    if [ "$code" != "200" ]; then
      missing=$((missing+1))
      # relaunch only if no live process already owns this port
      if ! pgrep -f "vllm serve .*--port $port( |$)" >/dev/null 2>&1; then
        echo "[heal-init] port $port (GPUs $group) not up -> relaunch"
        launch_one "$group" "$port"
      fi
    fi
  done < "$LOGDIR/pool.map"
  [ "$missing" = 0 ] && { echo "[heal-init] all servers healthy"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[heal-init] timeout with $missing missing"; break; }
  sleep 10
done
