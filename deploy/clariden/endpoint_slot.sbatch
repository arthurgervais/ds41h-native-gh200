#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.log   # (relative to the submit directory; #SBATCH lines do not expand variables)
# One serving slot: ds41h on one GH200 node behind the named Cloudflare tunnel ds41h-node$SLOT -> https://node$SLOT.<DOMAIN>
#   sbatch --job-name=ds41h-slot1 --export=ALL,SLOT=1 clariden/endpoint_slot.sbatch
# --requeue: on preemption Slurm puts the job back in the queue; it restarts on a new node with the same slot/hostname.
# The tunnel connects only once vLLM is healthy, so Cloudflare never routes to a loading node.
# API key on every route except /health (VLLM_API_KEY from $DS41H_HOME/secrets/endpoint.key).
set -uo pipefail
: "${SLOT:?set SLOT=1..4}"
REPO=${DS41H_REPO:?set DS41H_REPO to this repository}
source $REPO/clariden/activate.sh
cd $REPO
RUN=$DS41H_HOME/runs/${SLURM_JOB_ID}-slot${SLOT}-r${SLURM_RESTART_COUNT:-0}; mkdir -p "$RUN"
VLLM_API_KEY=$(cat "$DS41H_HOME/secrets/endpoint.key"); export VLLM_API_KEY
# NO_API_KEY=1 (test jobs only, together with NO_TUNNEL=1): no key on localhost, for tools that send none (BRRR bench,
# scripts/bench_concurrency.py). Ignored on a slot with tunnels.
if [ "${NO_API_KEY:-0}" = 1 ] && [ "${NO_TUNNEL:-0}" = 1 ]; then unset VLLM_API_KEY; fi
TID=$(awk -v n="ds41h-node$SLOT" '$2==n{print $1}' < <("$DS41H_HOME/bin/cloudflared" tunnel list 2>/dev/null))
CRED=$HOME/.cloudflared/$TID.json
# shared tunnel ds41h-api -> https://api.<DOMAIN>: every healthy slot runs a replica; Cloudflare spreads requests over them
ATID=$(awk '$2=="ds41h-api"{print $1}' < <("$DS41H_HOME/bin/cloudflared" tunnel list 2>/dev/null)); ACRED=$HOME/.cloudflared/$ATID.json
# ENGINE=ds41h: vLLM's model runner with the ds41h kernels (scripts/serve_ds41h.sh, TP4).
# ENGINE=fpr: FPRv1 (docs/FPR.md, scripts/serve_fpr.sh): each sequence on one GPU, no KV replicated across the four, the
#   pools sized from what is free after the weights. Needs clariden/fpr_prebuild.sbatch once (extensions + weight cache).
# Runtime settings, read when the job STARTS (and at every requeue start): $DS41H_HOME/slot.env, lines KEY=VALUE, e.g.
# ENGINE=fpr, FPR_RESERVE_GB=6. Slurm keeps the script a job was submitted with, so this file is how queued slots
# follow a later switch. Variables given with --export win over it.
# Per-slot overrides (main, 14 Nov): $DS41H_HOME/slot.env.slot$SLOT, same KEY=VALUE format, read BEFORE slot.env so its
# values win (first set wins). For canaries / per-slot A/B. Test jobs (NO_TUNNEL=1) ignore it.
if [ "${NO_TUNNEL:-0}" != 1 ] && [ -f "$DS41H_HOME/slot.env.slot$SLOT" ]; then
  while IFS='=' read -r k v; do
    [[ $k =~ ^[A-Z_][A-Z0-9_]*$ ]] && [ -z "${!k:-}" ] && export "$k=$v" && echo "slot.env.slot$SLOT: $k"
  done < "$DS41H_HOME/slot.env.slot$SLOT"
fi
# Production slots write no CPU core dumps (main, 14 Nov): each was 20-116 GB in the release tree and delayed the in-job
# restart; the backtraces are known. Test jobs keep the default.
[ "${NO_TUNNEL:-0}" != 1 ] && ulimit -c 0
if [ -f "$DS41H_HOME/slot.env" ]; then
  while IFS='=' read -r k v; do
    # FPR_RELEASE (14 Nov): production slots only. Test jobs (NO_TUNNEL=1: fpr_c1.sh, soaks, gates) start from this
    # checkout unless their caller sets FPR_RELEASE itself (release gates do); from 23:54 to this fix every runner test
    # server took slot.env's release/A instead of the repo
    [ "$k" = FPR_RELEASE ] && [ "${NO_TUNNEL:-0}" = 1 ] && continue
    [[ $k =~ ^[A-Z_][A-Z0-9_]*$ ]] && [ -z "${!k:-}" ] && export "$k=$v"
  done < "$DS41H_HOME/slot.env"
fi
ENGINE=${ENGINE:-ds41h}
# NO_TUNNEL=1: serve on localhost:8100 only (test jobs: clariden/fpr_test.sbatch), no Cloudflare tunnels
NO_TUNNEL=${NO_TUNNEL:-0}
if [ "$NO_TUNNEL" != 1 ]; then
  [ -n "$TID" ] && [ -f "$CRED" ] || { echo "no tunnel credentials for ds41h-node$SLOT"; exit 1; }
fi
# 64 concurrent requests per node (the router on the user's Mac caps at 64 too and queues the rest across nodes);
# at <= 64 streams DSpark k=5 beats k=3
SPEC_K=${SPEC_K:-5}; MAX_SEQS=${MAX_SEQS:-64}; CAPTURE=$(( MAX_SEQS * (SPEC_K + 1) ))
# 512K context since 7 Nov 2026 (agent conversations hit 262K; test job 3883261: 512K prefill 14.8 s, recall OK, no OOM)
# GPU_UTIL 0.93: at 0.91 a 512K slot does not start (0.54 GiB KV per GPU left, one 512K request needs 1.34 GiB)
MAXLEN=${MAXLEN:-524288}; GPU_UTIL=${GPU_UTIL:-0.93}; NAME=${MODEL_NAME:-deepseek-v4.1-flash}
export EXTRA="--max-num-batched-tokens 8192 --served-model-name $NAME"
# IMAGES=1 (default): load the vision encoder (ds41h serves --language-model-only otherwise; costs ~15% of the KV cache)
[ "${IMAGES:-1}" = "1" ] && EXTRA="$EXTRA --no-language-model-only"
# Responses API store, so previous_response_id works (in memory per node: lost when the slot restarts)
export VLLM_ENABLE_RESPONSES_API_STORE=${VLLM_ENABLE_RESPONSES_API_STORE:-1}
if [ "$ENGINE" = fpr ]; then
  # FPRv1's certified configuration (fpr/config/fprv1.env: DSpark k=3, 224 seats, CP prefill, 17,664 batched tokens);
  # our name and context appended to its vllm arguments (later flags win). Its own warm-up (adapter/warm.py) asks for
  # the model "deepseek-v41-flash" without our key: off, clariden/slot_warmup.py below warms it instead.
  # VENV_LOCAL=0: the extensions were built against the Lustre venv's torch; another torch path would rebuild them.
  # Engram stays in host RAM (engram=phys); numactl keeps host allocations off the GPUs' HBM NUMA nodes.
  MAX_SEQS=${FPR_SEATS:-224}
  export EXTRA="--served-model-name $NAME --max-model-len $MAXLEN" FPR_WARM=0 IMAGES=${IMAGES:-1} RESPONSES_STORE=1
  export DS41H_BULK_EXPERTS=0 MAX_SEQS
  # KV pools = what is free on the tightest GPU after the weights minus FPR_RESERVE_GB, which must hold everything
  # allocated after the pools: the vision towers (~0.9 GiB, loaded after sizing), graph pools, mixed-step and long-
  # prompt workspaces, lazily loaded kernels. FPR's defaults keep 18 GB (engine 10 + worker 8); here all of it
  # goes through engine_reserve_bytes (the worker's share 0), sized by clariden/fpr_test.sbatch (measured peak + margin).
  FPR_RESERVE_GB=${FPR_RESERVE_GB:-8}
  # FPR_CONFIG: another configuration file with fprv1.env's variables (clariden/fpr_gh200.env: the GH200 memory trims)
  [ -n "${FPR_CONFIG:-}" ] && export FPR_CONFIG
  FPR_ENGINE_CFG=$(source "${FPR_CONFIG:-fpr/config/fprv1.env}" && echo "$FPR_ENGINE")   # the engine pieces, kept as they are
  if [ -n "${FPR_ENGINE:-}" ] && [ "$FPR_ENGINE" != "$FPR_ENGINE_CFG" ]; then   # a caller's FPR_ENGINE wins (warned)
    echo "endpoint_slot: FPR_ENGINE from the environment overrides the configuration file's"
    FPR_ENGINE_CFG=$FPR_ENGINE
  fi
  export PHYS_ENGINE="$FPR_ENGINE_CFG,engine_reserve_bytes=$(python3 -c "print(int($FPR_RESERVE_GB * 1e9))")"
  export PHYS_WORKER_RESERVE=0
  VENV_LOCAL=${VENV_LOCAL:-0}
fi
echo "slot $SLOT job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} engine $ENGINE on $(hostname) $(date -Is) run=$RUN"

PFPID=""
if [ "${PREFETCH:-0}" = "1" ] && [ -f $REPO/clariden/prefetch_ckpt.py ]; then
  numactl --interleave=0-3 env -u PYTHONPATH /usr/bin/python3 $REPO/clariden/prefetch_ckpt.py --model "$MODEL" --server-log "$RUN/server.log" \
    --warm "$DS41H_HOME/venv/lib/python3.12/site-packages" > "$RUN/prefetch.log" 2>&1 &
  PFPID=$!
fi
# Python packages unpacked to node-local /tmp (validated by the optimization session: ready 220-239 s vs 303 s).
# VIRTUAL_ENV/CUDA_HOME stay on Lustre on purpose: changing CUDA_HOME would trigger a ~15 min ds41h JIT rebuild.
if [ "${VENV_LOCAL:-1}" = "1" ] && [ -f "$DS41H_HOME/images/site-packages.tar" ]; then
  mkdir -p /tmp/ds41h-venv && tar -xf "$DS41H_HOME/images/site-packages.tar" -C /tmp/ds41h-venv \
    && export PYTHONPATH=/tmp/ds41h-venv/site-packages${PYTHONPATH:+:$PYTHONPATH}
fi
# bulk raw-expert fill (ds41h/bulk_experts.py, bit-exact vs stock): ~5 min start instead of ~10
export DS41H_BULK_EXPERTS=${DS41H_BULK_EXPERTS:-1}
if [ "$ENGINE" = fpr ]; then
  # FPR opens production's Engram tables in /dev/shm and never builds them: fill them first (~195 GiB, host NUMA only)
  T_ENG=$(date +%s)
  # FPR_ENGRAM_NUMA=interleave: the two ~102 GB tables spread over Grace nodes 0-3 (~51 GB each) instead of filling
  # nodes 2 and 3, which pushed GPUs 2 and 3's host KV pools (FPR_KV_HOST_GB) and host-mapped weights onto the remote
  # nodes 0 and 1 (numa_maps, 9 Nov). Placement only: the same bytes.
  ENG_NUMA="--membind=0-3"; [ "${FPR_ENGRAM_NUMA:-}" = interleave ] && ENG_NUMA="--interleave=0-3"
  if ! (source scripts/env.sh >/dev/null 2>&1; numactl $ENG_NUMA python clariden/engram_shm.py) > "$RUN/engram_shm.log" 2>&1; then
    echo "Engram tables failed $(date -Is)"; tail -20 "$RUN/engram_shm.log"; exit 1
  fi
  echo "Engram tables in /dev/shm after $(( $(date +%s) - T_ENG )) s: $(tail -1 "$RUN/engram_shm.log")"
fi
launch_server() {   # appends: an in-job restart (below) continues the same server.log
  if [ "$ENGINE" = fpr ]; then
    numactl --membind=0-3 bash scripts/serve_fpr.sh >> "$RUN/server.log" 2>&1 &
  else
    VHEAD=0 THROUGHPUT=1 SPEC_K=$SPEC_K ENGRAM=host MAXLEN=$MAXLEN MAX_SEQS=$MAX_SEQS CAPTURE=$CAPTURE GPU_UTIL=$GPU_UTIL \
      numactl --membind=0-3 bash scripts/serve_ds41h.sh >> "$RUN/server.log" 2>&1 &
  fi
  SPID=$!
}
launch_server
CFPID=""; APID=""
cleanup() {
  for p in $APID $CFPID $PFPID; do kill "$p" 2>/dev/null; done
  kill -INT $SPID 2>/dev/null; for _ in $(seq 1 20); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
  pkill -KILL -u "$USER" -f "bin/vllm serve" 2>/dev/null
  (source scripts/env.sh >/dev/null 2>&1; python -m ds41h.engram_host drop) >> "$RUN/server.log" 2>&1 || true
  echo "stopped $(date -Is)"
}
# SIGTERM (Slurm: time limit, preemption, scancel) ends the job: cleanup runs once, from the EXIT trap. A TERM trap that
# only cleaned up let the monitor loop below take the dead server for a crash and start an in-job restart (14 Nov 03:29).
trap cleanup EXIT
trap 'echo "SIGTERM $(date -Is): ending the slot"; exit 143' TERM

wait_ready() {
T0=$(date +%s)
until curl -sf localhost:8100/health >/dev/null; do
  kill -0 $SPID 2>/dev/null || { echo "server exited during start"; tail -30 "$RUN/server.log"; exit 1; }
  sleep 3
done
echo "ready after $(( $(date +%s) - T0 )) s $(date -Is)"
# warm-up before the tunnel opens (clariden/slot_warmup.py): kernels load lazily and need GPU memory outside vLLM's
# pool; slot 4 (job 3883340) died loading one during its first burst of long prompts. Fail here, not under traffic.
MODEL_ID=$(curl -s -H "Authorization: Bearer $VLLM_API_KEY" localhost:8100/v1/models | env -u PYTHONPATH /usr/bin/python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
WIMG=""; [ "${IMAGES:-1}" = "1" ] && WIMG="$MODEL/inference/examples/images/carrots.jpeg"
# The helpers are stdlib-only and run on the system python 3.6: without the runtime PYTHONPATH (release runtime/
# sitecustomize.py needs 3.7+; 3.6 printed a SyntaxError at exit that "warm-up ok" quoted, 15 Nov).
if env -u PYTHONPATH /usr/bin/python3 "$REPO/clariden/slot_warmup.py" "$MODEL_ID" $WIMG > "$RUN/warmup.log" 2>&1 && curl -sf localhost:8100/health >/dev/null; then
  echo "warm-up ok $(date -Is): $(tail -1 "$RUN/warmup.log")"
else
  echo "warm-up failed $(date -Is)"; tail -20 "$RUN/warmup.log"; tail -30 "$RUN/server.log"; exit 1
fi
}
wait_ready
start_tunnel() {
  "$DS41H_HOME/bin/cloudflared" tunnel --no-autoupdate --protocol http2 run --credentials-file "$CRED" \
    --url http://127.0.0.1:8100 "ds41h-node$SLOT" >> "$RUN/cloudflared.log" 2>&1 &
  CFPID=$!; echo "tunnel ds41h-node$SLOT -> https://node$SLOT.<DOMAIN> started $(date -Is)"
}
start_api() {
  [ -f "$ACRED" ] || return 0
  "$DS41H_HOME/bin/cloudflared" tunnel --no-autoupdate --protocol http2 run --credentials-file "$ACRED" \
    --url http://127.0.0.1:8100 ds41h-api >> "$RUN/cloudflared-api.log" 2>&1 &
  APID=$!; echo "replica of ds41h-api -> https://api.<DOMAIN> started $(date -Is)"
}
if [ "$NO_TUNNEL" != 1 ]; then
  start_api
  start_tunnel
fi
# In-job server restarts (main, 14 Nov): a crashed server (13-14 Nov: mixed-step OOMs, an hsd assertion, an illegal
# address) used to end the job, and the slot then waited hours in the queue for a new node. A production slot (tunnels
# on) now restarts the server on the same node, at most MAX_SRV_RESTARTS times per job (~3 min each: Engram tables stay
# in /dev/shm, the builds are cached): tunnels down first (Cloudflare and the router stop sending here), old processes
# killed, GPUs empty, then the same start, ready check and warm-up as the first start, tunnels back up. Test jobs
# (NO_TUNNEL=1) still end on the first crash, so soaks and gates see it.
MAX_SRV_RESTARTS=${MAX_SRV_RESTARTS:-4}; NSR=0
restart_server() {
  for p in $APID $CFPID; do kill "$p" 2>/dev/null; done; APID=""; CFPID=""
  kill -INT $SPID 2>/dev/null; for _ in $(seq 1 20); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
  pkill -KILL -u "$USER" -f "bin/vllm serve" 2>/dev/null; pkill -KILL -u "$USER" -f "VLLM::EngineCore" 2>/dev/null
  for _ in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && break; sleep 2; done
  echo "=== slot $SLOT in-job restart $NSR of $MAX_SRV_RESTARTS $(date -Is) ===" >> "$RUN/server.log"
  launch_server
  wait_ready   # exits the job if the new server does not come up or fails its warm-up
  start_api
  start_tunnel
}
BAD=0
while true; do
  WHY=""
  if kill -0 $SPID 2>/dev/null; then
    # dead engine (EngineDeadError leaves the API server up, /health 503)
    if curl -sf -m 10 localhost:8100/health >/dev/null; then BAD=0; else BAD=$((BAD + 1)); fi
    [ $BAD -ge 3 ] && WHY="vLLM unhealthy for 3 checks (~1 min)"
  else
    WHY="server exited"
  fi
  if [ -n "$WHY" ]; then
    if [ "$NO_TUNNEL" != 1 ] && [ $NSR -lt $MAX_SRV_RESTARTS ]; then
      NSR=$((NSR + 1)); echo "$WHY $(date -Is): restarting the server on this node ($NSR of $MAX_SRV_RESTARTS)"
      tail -30 "$RUN/server.log"; restart_server; BAD=0; continue
    fi
    echo "$WHY, ending the slot $(date -Is)"; tail -30 "$RUN/server.log"; exit 1
  fi
  if [ "$NO_TUNNEL" != 1 ]; then
    kill -0 "$CFPID" 2>/dev/null || { echo "cloudflared exited, restarting $(date -Is)"; start_tunnel; }
    [ -z "$APID" ] || kill -0 "$APID" 2>/dev/null || { echo "api replica exited, restarting $(date -Is)"; start_api; }
  fi
  sleep 20
done
