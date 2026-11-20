// srv8x (c1-mega, 15 Nov; main's one-copy lane, phase 1a): srv7m with wo_a / wo_b read from the engine's OWN W8A16 packed fp8
// (AW["w8"]["wo_a" / "wo_b"], this rank's views; C1M_B16 woa8 / wob8 = bits 32 / 64 instead of wo_a / wo_b 2 / 4): the bf16 A
// fragments rebuilt bit-exactly in registers (skw8t's e4m3x4_to_w) into srv7m's bf16 HMMAs in srv7m's k order -> BITWISE vs
// srv7m, no bf16 quarters (wo_aT_c1 / wo_b_c1: -1.51 GB per GPU once the engine skips them, ENG_WO_ONECOPY). Fragments
// c1dense_f1_l8.inc / c1dense_f2_u8.inc; deploy config C1M_B16=wqkv,woa8,wob8,sh,qb8
// srv7m (c1-mega + c1-sync, 15 Nov): srv7b + c1-sync's og multicast on a cluster-2 launch. C1S_CLS 2: 66 clusters x 2 CTAs
// via cudaLaunchKernelEx, per-device co-residency TORCH_CHECK (max active clusters >= 66); C1S_OGMC: the pair's rank-0 CTA
// waits C_O and bulk-multicasts the og rows into both CTAs' image rows. c1mega.cu 7874d306 = srv7b's 3da4f730 + ONLY the
// C1S_CLS / C1S_OGMC hunks (flags off: SASS = srv7b; on: 168 regs, SMEM +16 B for the og mbarrier). Bitwise expected;
// cl-02 (64K, owners 0 / 2, on srv7): graph L -0.064 ms per step. (srv7c is the 14 Nov C-ABI package: a different line.)
#define C1S_CLS 2
#define C1S_OGMC
// srv7b (C1 session, 13 Nov): the BITWISE srv7 stack = srv6s + c1-overlap's L3 (C1IDX_X1IMG + C1IDX_ARRM; c1idx.cuh 4a870d67,
// c1hooks_ix.inc 24faa22c; served -0.03..-0.045 ms, c1x-06 / 06b) + c1-sync's H-chain a2r (C1S_RTS2 RT1T GPF T6W SVC; harness
// -0.14..-0.15 ms, served c1x-08). c1mega.cu = srv6s's 2d987e53 + a2r + the inactive C1M_A8C hunks. Expected BITWISE vs srv6.
#define C1M_FN_FRAG2
#define C1M_DENSE_A5 "c1dense_a5_e8.inc"
#define C1M_DEP_IX
#define C1M_HOOKS "c1hooks_ix.inc"
#define C1M_ONLY_FULL
#define C1M_DEP_B16 121   // (srv8x: 1 wqkv | 8 sh | 16 qb8 | 32 woa8 | 64 wob8)
#define C1M_XN 2
#define C1M_XFRAG
#define C1M_W13KB 5
#define X2Y_B 2
#define C1M_MOE_PRODUCE "c1moe_x2n_produce.inc"
#define C1M_MOE_CONSUME "c1moe_x2n_consume.inc"
#define C1M_RTL
#define C1M_RTF
#define C1M_FONE_H
#define C1M_DU
#define C1M_FIMG
#define C1M_RNP
#define C1M_NWP
#define C1M_OWNSTG
#define C1M_STF
#define C1M_LBM
#define C1IDX_NO_DS
#define C1IDX_X1IMG
#define C1IDX_ARRM
#define C1M_DENSE_A3 "c1dense_a3_l.inc"
#define C1M_DENSE_F1 "c1dense_f1_l8.inc"
#define C1M_DENSE_F2 "c1dense_f2_u8.inc"
#define C1S_RTS2
#define C1S_RT1T
#define C1S_GPF
#define C1S_T6W
#define C1S_SVC
#include "c1mega_depn.cu"
