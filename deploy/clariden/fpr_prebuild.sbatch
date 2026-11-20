#!/bin/bash
#SBATCH --job-name=ds41h-fpr-prebuild
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.log   # (relative to the submit directory; #SBATCH lines do not expand variables)
# FPRv1 (fpr/, docs/FPR.md) prebuild on a GH200 node without the GPUs: the ~30 CUDA extensions (sm_90a, aarch64 host
# code) and the converted-weight cache ($DS41H_HOME/cache/wcache, ~280 GiB: attention per layer + raw experts per
# layer and rank). Both are reused by every later start (serve_fpr.sh reads fpr/build/prebuilt.env). Resumable: built
# extensions and cached entries are skipped, so a run cut by the time limit continues where it stopped.
#   sbatch clariden/fpr_prebuild.sbatch                 (PARTS=ext or PARTS=weights for one half)
set -uo pipefail
REPO=${DS41H_REPO:?set DS41H_REPO to this repository}
source $REPO/clariden/activate.sh
cd $REPO
echo "start $(date -Is) on $(hostname), repo $(git log --oneline -1)"
export MAX_JOBS=${MAX_JOBS:-96} DS41H_BULK_EXPERTS=0
(source scripts/env.sh && CUDA_VISIBLE_DEVICES= python -c "import torch, vllm, triton, numpy, safetensors, pynvml, yaml; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__, 'triton', triton.__version__); import shutil; print('nvcc', shutil.which('nvcc'))")
T0=$(date +%s)
PARTS=${PARTS:-ext,weights} JOBS=${JOBS:-4} numactl --membind=0-3 bash scripts/prebuild_fpr.sh
rc=$?
echo "prebuild_fpr exit $rc after $(( $(date +%s) - T0 )) s, end $(date -Is)"
du -sh "$DS41H_HOME/cache/wcache" 2>/dev/null
exit $rc
