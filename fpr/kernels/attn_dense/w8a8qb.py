#!/usr/bin/env python3
"""ds41h-phys: W8A8 q_b (w8a8qb.cu) for the C128 decode buckets -- build, call wrapper, reference.

MATH-CHANGING (e4m3 activations, the official act_quant numerics); default off in the engine (ENG_C128_QB8).
Weights: the q_b W8A16 Gemm's packed bytes (w8a16.py repack), shared, no copy.

  QB8(ext, g)(qr, q2, pos, cs, kv=..., slot=..., cache=..., page_stride=..., shift=...)
      = quant(qr) -> x8 (k-permuted e4m3) + xe (ue8m0 per row per 32 k) -> gemm (+ RoPE of q + window KV insert),
  the same arguments as w8a16.Gemm / attn_half.w8_qb.
"""
from __future__ import annotations

import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KC = 128


def build(verbose: bool = False, extra_defs: tuple = ()):
    from torch.utils.cpp_extension import load

    defs = [d for d in os.environ.get("W8A8QB_DEFS", "").split(",") if d] + list(extra_defs)
    sfx = "".join("_" + d.replace("=", "").lower() for d in defs)
    bdir = os.environ.get("W8A8QB_BUILD", os.path.join(HERE, "build_w8a8qb")) + sfx
    os.makedirs(bdir, exist_ok=True)
    flags = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-Xptxas=-v"] + [f"-D{d}" for d in defs]
    return load(name="w8a8qb_ext" + sfx, sources=[os.path.join(HERE, "w8a8qb.cu")], extra_cuda_cflags=flags,
                extra_ldflags=["-lcuda"], extra_include_paths=[HERE], build_directory=bdir, verbose=verbose)


def perm32(device=None) -> torch.Tensor:
    """LOGICAL position L (0..31) of a k32 block -> ACTUAL k: 16 (L >> 4) + 2 t + (j & 1) + 8 (j >> 1), t = (L & 15) >> 2,
    j = L & 3 (the packed fragments' k order seen by the fp8 wgmma)."""
    L = torch.arange(32, device=device)
    t, j = (L & 15) >> 2, L & 3
    return 16 * (L >> 4) + 2 * t + (j & 1) + 8 * (j >> 1)


def unpermute_x8(x8: torch.Tensor) -> torch.Tensor:
    """x8 [R, K] uint8 in the kernel's order -> actual k order (for checks)."""
    R, K = x8.shape
    p = perm32(x8.device)
    out = torch.empty_like(x8)
    out.view(R, K // 32, 32)[:, :, p] = x8.view(R, K // 32, 32)
    return out


class QB8:
    """W8A8 q_b over the q_b W8A16 Gemm g (g.packed). Buffers: one per device, grown outside graph capture only
    (reserve() before capturing)."""

    _BUF: dict = {}

    def __init__(self, ext, g):
        self.ext, self.g = ext, g
        self.K = g.packed.shape[0] * KC

    def reserve(self, rows: int, device) -> None:
        self.reserve_rows(rows, device, self.K)

    @classmethod
    def reserve_rows(cls, rows: int, device, K: int = 1280):
        """flat byte buffers (x8: Rp x K, xe: K/128 x Rp x 4) for up to rows rows, shared by every QB8 of the device and K
        (calls run in stream order); grown only outside graph capture."""
        Rp = (rows + 127) // 128 * 128
        key = (str(torch.device(device)), K)
        b = cls._BUF.get(key)
        if b is None or b[0] < Rp:
            assert not torch.cuda.is_current_stream_capturing(), "w8a8qb: reserve the row count before graph capture"
            b = (Rp, torch.empty(Rp * K, dtype=torch.uint8, device=device),
                 torch.zeros(Rp * K // KC * 4, dtype=torch.uint8, device=device))
            cls._BUF[key] = b
        return b

    def _bufs(self, R: int, device):
        return self.reserve_rows(R, device, self.K)

    def quant(self, qr: torch.Tensor):
        """qr [R, K] bf16 -> (x8 [R, K], xe [K/128, Rp, 4]) contiguous views of the shared buffers (xe rows >= R zeroed)."""
        R = qr.shape[0]
        Rp = (R + 127) // 128 * 128
        _, x8b, xeb = self._bufs(R, qr.device)
        x8 = x8b[:R * self.K].view(R, self.K)
        xe = xeb[:Rp * self.K // KC * 4].view(self.K // KC, Rp, 4)
        self.ext.quant(qr, x8, xe)
        return x8, xe

    def __call__(self, qr, q2, pos=None, cs=None, kv=None, slot=None, cache=None, page_stride=0, shift=0):
        x8, xe = self.quant(qr)
        self.ext.gemm(x8, xe, self.g.packed, q2, pos, cs, kv, slot, cache, int(page_stride), int(shift))
        return q2


def reference(qr: torch.Tensor, w8: torch.Tensor, s8: torch.Tensor) -> torch.Tensor:
    """fp64 reference of the kernel's math (no RoPE): the official act_quant(32, ue8m0) of qr x the exact fp8 weights,
    every product exact, one sum. Returns fp64 [R, N]."""
    import sys

    sys.path.insert(0, os.path.join(HERE, "..", "w8a8"))
    import act_quant_ref as AQ  # noqa: E402

    q, e8 = AQ.quant(qr)  # q uint8 [R, K] e4m3 bits, e8 uint8 [R, K/32] ue8m0 bytes
    R = qr.shape[0]
    xd = q.view(torch.float8_e4m3fn).double().view(R, -1, 32) * torch.pow(2.0, e8.double() - 127).unsqueeze(-1)
    N, K = w8.shape
    wd = w8.contiguous().view(torch.float8_e4m3fn).double().view(N // 32, 32, K // 32, 32) * torch.pow(
        2.0, s8.contiguous().view(torch.uint8).double() - 127).view(N // 32, 1, K // 32, 1)
    return xd.view(R, K) @ wd.view(N, K).t()


def reference_quant(qr: torch.Tensor):
    """(q uint8 [R, K], e8 uint8 [R, K/32]) of the shared reference quantization (to check quant() bitwise)."""
    import sys

    sys.path.insert(0, os.path.join(HERE, "..", "w8a8"))
    import act_quant_ref as AQ  # noqa: E402

    return AQ.quant(qr)
