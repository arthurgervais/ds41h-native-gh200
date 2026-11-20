# ds41h native

Native C/C++ serving of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on a
single node of **4x GH200 96 GB** (CSCS Clariden, aarch64).

- **Engine.** Each sequence lives on one GPU. The experts are split across the four GPUs, and long prompts are
  prefilled across all four. The Engram n-gram tables (~190 GiB) live in host RAM, not in GPU memory.
- **Native loop.** The scheduler, KV manager, block tables, step builders and engine loop are C++. Decode runs on CUDA
  graphs launched by a C++ driver. A C++ frontend replaces vLLM's Python API server.
- **Kernels.** Custom sm_90a kernels: a persistent decoder kernel, the sparse-attention indexer, routed-expert GEMMs
  and NVLink all-gather / reduce-scatter.
- **Precision.** No quantization beyond the checkpoint's own formats (MXFP4 experts, FP8 dense weights). The sparse-attention
  top-k is exact at every context length (checked up to 400K).

vLLM 0.32.0 still launches the engine process. Python remains for start-up, weight loading, and steps the native loop
does not take, such as image steps.

## Performance

8K prompts, 2,048 output tokens, thinking off, aggregate decode tok/s at *n* concurrent streams (Cn).

### Against stock vLLM (estimate)

We have not run stock vLLM 0.32.0 on Clariden, so this is an estimate from memory sizes.

- **Stock vLLM does not fit on one node.** It keeps the whole ~480 GB checkpoint, including the ~190 GiB of Engram
  tables, in GPU memory, and one node has 384 GB of HBM. It needs two or more nodes.
- **ds41h native serves it on one node.** The Engram tables sit in Grace memory, which leaves room for KV cache up to
  256 concurrent streams.
- **The table below is a floor.** ds41h-on-vLLM is stock vLLM 0.32.0 plus our kernels at stock precision, and it is
  the fastest vLLM baseline we have that fits on one node. So per node we expect the gain over stock vLLM to be at
  least the gain over ds41h-on-vLLM.

### Against ds41h-on-vLLM (measured on Clariden)

| 4x GH200 | ds41h-on-vLLM | ds41h native | Gain |
|---|---|---|---|
| C1 | ~575 | 676 | x1.18 |
| C64 | 6,566 | 7,104 | x1.08 |
| C128 | out of memory | 10,295 | - |
| C256 | doesn't fit | 12,977 | - |
| Prefill 6K / 49K | 29.9K / 44.0K | 35.9K / 49.9K | x1.20 / x1.13 |
| Decode at 287-290K context, 1 stream | 398 | 537 | x1.35 |
| KV cache, tokens | ~0.73M (vLLM's count, 262K max context) | 45.9M (11.47M per GPU, 655K max context) | x20+ |

**KV cache.** Each GPU holds 179,237 pages of 64 tokens: 8,192 pages in HBM, the rest in its Grace memory over
NVLink-C2C. vLLM's start-up count understates what ds41h-on-vLLM holds at once by up to ~3x, so x20 is a floor.

## Requirements

- One node with 4x GH200 (`sm_90a`). The kernels are built for exactly four GPUs.
- CUDA 13 driver, gcc/g++ 13, CMake >= 3.20, [uv](https://docs.astral.sh/uv/).
- The checkpoint (~480 GB), plus ~195 GiB of host RAM for the Engram tables in `/dev/shm`.

## Build and run

```bash
export DS41H_HOME=/path/to/scratch DS41H_REPO=$PWD MODEL_DIR=/path/to/DeepSeek-V4.1-Flash-1017
bash setup/build-venv.sh                                   # vLLM 0.32.0 + patches/
source deploy/clariden/activate.sh                         # GH200: gcc-13 + venv (adapt to your host)
export FPR_CONFIG=$DS41H_REPO/fpr/config/r2_gh200.env
MODEL=$MODEL_DIR scripts/prebuild_fpr.sh                   # CUDA extensions + weight cache, on the CPU
```

1. **Native modules.** Build `fpr/core` (makefiles in `fpr/core/tests/`), `packages/core2b`
   (`packages/core2b/tests/core.mk`) and the C++ frontend (`frontend_cxx/`, CMake). The makefile headers list the
   flags.
2. **Serve:**

   ```bash
   source $DS41H_REPO/fpr/config/native_r2.env
   export VLLM_USE_RUST_FRONTEND=1 VLLM_RUST_FRONTEND_PATH=$DS41H_REPO/frontend_cxx/build/frontend_cxx
   MODEL=$MODEL_DIR scripts/serve_fpr.sh                      # OpenAI-compatible API on 127.0.0.1:8100
   ```

   Wait for "warm-up done" before sending load.
3. **Slurm.** `deploy/clariden/endpoint_slot.sbatch` runs one serving slot per node, with a health check and restart
   on crash. Fill in `<SLURM_ACCOUNT>` before submitting it.

## Layout

| Path | Contents |
|---|---|
| `fpr/engine/`, `fpr/kernels/` | Model step, attention, indexer; CUDA kernels and loaders |
| `fpr/core/`, `packages/core2b/`, `packages/nd/` | C++ scheduler, KV manager, block tables, engine process, native decode |
| `frontend_cxx/` | C++ API frontend |
| `ds41h/`, `plugins/`, `patches/` | Kernels and hooks shared with ds41h-on-vLLM; vLLM plugins and patches |
| `fpr/config/`, `scripts/`, `deploy/` | Serving configurations, launchers, Slurm jobs |
| `docs/` | Design, exactness method and Engram notes (from ds41h-on-vLLM) |

## Known limits

- Tied to vLLM 0.32.0, CUDA 13 and `sm_90a`, with exactly four GPUs per node.
- Tested on Clariden, one node per server.
- Image steps run on the Python path.
- Flags that are off in the served configuration may refer to kernels that are not included here.

## License

Apache 2.0, see [LICENSE](LICENSE). Third-party notices are in [NOTICE](NOTICE) and [LICENSES/](LICENSES). The model
weights have their own license.
