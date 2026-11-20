#!/bin/bash
#SBATCH --job-name=ds41h-prebuild
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log   # (relative to the submit directory; #SBATCH lines do not expand variables)
# JIT-build the ds41h CUDA extensions on a GH200 node (aarch64 host code, sm_90a device code), without using the GPUs.
# Uses the same flag set as clariden/serve_bench.sbatch so the server finds the builds in the cache.
#   sbatch --export=ALL,SPEC_K=5 clariden/prebuild.sbatch
set -uo pipefail
REPO=${DS41H_REPO:?set DS41H_REPO to this repository}
source $REPO/clariden/activate.sh
cd $REPO
mapfile -t ENVS < <(THROUGHPUT=1 SPEC_K=${SPEC_K:-5} ENGRAM=host DSML_RT=0 DRYRUN=1 bash scripts/serve_ds41h.sh)
printf 'prebuild env: %s\n' "${ENVS[@]}"
echo "start $(date -Is) on $(hostname)"
(source scripts/env.sh && python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0), 'cc', torch.cuda.get_device_capability(0), 'p2p01', torch.cuda.can_device_access_peer(0, 1)); print('vllm', vllm.__version__)")
MAX_JOBS_PB=${MAX_JOBS_PB:-48} bash scripts/prebuild.sh "${ENVS[@]}"
echo "end $(date -Is)"
