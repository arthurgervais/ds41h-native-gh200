// head_dense_cabi.h -- C ABI of kernels/head/vendor/dense.cu, the HSD drafter's head_dense (built with -DDENSE_PDL=0) (FPR
// session, 14 Nov 2026), per fpr/engine/mixed_cxx/fpr_cabi.h. vendor/dense.cu was byte-identical to attn_dense/dense.cu
// before its C ABI (b85def80) and carries c128-attn's conversion (attn_dense/dense.cu 4e5f47e0), so the Args structs are
// attn_dense/dense_cabi.h's: FprAttnDenseNormRope2Args, FprAttnDenseNormRopeArgs, FprAttnDenseRopeQInsertArgs,
// FprAttnDenseInvRopeArgs (field `call`, not `c`). One cudaLaunchKernelEx per function on call.stream, its own return code
// (no cudaGetLastError); no scratch, no allocation, no host sync, no per-device state (no init entry).
// Table: fpr_head_dense_table() -> ext "head_dense"; "head_dense.norm_rope2", "head_dense.norm_rope",
// "head_dense.rope_q_insert", "head_dense.inv_rope".
#pragma once
#include "../../attn_dense/dense_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

const FprExtTable* fpr_head_dense_table(void);

#ifdef __cplusplus
}
#endif
