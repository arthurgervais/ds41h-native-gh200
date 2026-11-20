// c1d_base.cu (C1 tail session, 11 Nov 2026): the drafter's c1mega instance (fpr/kernels/hsdtail/c1d.py, kind "base") =
// fpr/kernels/c1mega/c1mega.cu with the drafter's shape knobs and the expert fragments PINNED to c1moe_produce /
// consume.inc (c1mega's default moves to c128-moe's x2b fragments, which are bitwise equal but +3.6 us per drafter block at
// NE 128 / 3 rows, t-35). C1M_WIN 131 = the ring's keys (132 until c1mega's STG_P alignment fix, ee1302dd, 11 Nov 20:50).
// Source: c1d_src/ = a frozen copy of c1mega.cu ee1302dd + its hook / pre-x2b fragment files (c1d_src/SOURCE.md); quoted
// includes inside it resolve in c1d_src.
#define C1M_NE 128
#define C1M_TOPK 3
#define C1M_WIN 131
#define C1M_TKA 0
#define C1M_MOE_PRODUCE "c1moe_produce.inc"
#define C1M_MOE_CONSUME "c1moe_consume.inc"
#include "c1d_src/c1mega.cu"
