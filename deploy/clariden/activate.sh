# ds41h on CSCS Clariden (Alps): one node = 4x GH200 (96 GB HBM3 each), aarch64, Grace host RAM.
# Source this before any ds41h script (setup/, scripts/, clariden/). Everything lives on iopsstor (flash Lustre).
export DS41H_HOME=${DS41H_HOME:?set DS41H_HOME (scratch directory: venv, bin, caches, logs)}
export DS41H_REPO_CL=${DS41H_REPO_CL:-${DS41H_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
# uv + gcc/g++/cc/c++ -> gcc-13: the system gcc is 7.5, too old for CUDA 13 / torch 2.13 extension builds
export PATH=$DS41H_HOME/bin:$PATH
# the venv's python3 first: scripts/serve_ds41h.sh runs plugins/kvram_budget.py with `python3`, and the system python3
# on Clariden is 3.6 (no `from __future__ import annotations`) -> KVRAM silently disabled otherwise
export PATH=$DS41H_HOME/venv/bin:$PATH
export CC=$DS41H_HOME/bin/gcc CXX=$DS41H_HOME/bin/g++
# Served checkpoint (same tensor layout as deepseek-ai/DeepSeek-V4.1-Flash-1017)
export MODEL=${MODEL:-${MODEL_DIR:?set MODEL_DIR to the checkpoint directory}}
export DS41H_MODEL_DIR=${DS41H_MODEL_DIR:-$MODEL}
