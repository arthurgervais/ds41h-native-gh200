"""ctypes mirrors of kernels/hsdtail's C ABI headers (xg_cabi.h, dtail_cabi.h, ctx3_cabi.h, rtag_cabi.h, c1d_cabi.h) and
packers that build each Args struct from the same tensors the pybind wrapper takes (C1 tail session, 14 Nov 2026). For gate
(c) of fpr_cabi.h (fpr/engine/mixed_cxx/cabi_test: the C path through the table vs the pybind path) and for the head's
static structs. Field order / types = the headers; sizes are checked against the tables' args_size by cabi_test.
"""
from __future__ import annotations

import ctypes as C

VP, I32, I64, F32, U64 = C.c_void_p, C.c_int32, C.c_int64, C.c_float, C.c_uint64


class FprCall(C.Structure):
    _fields_ = [("stream", VP), ("device", C.c_int), ("rank", C.c_int)]


class XgGather(C.Structure):
    _fields_ = [("c", FprCall), ("src", VP), ("out", VP), ("n", I32), ("nmax", I32), ("box", VP), ("peers", VP * 4),
                ("ctr", VP)]


class Dtail(C.Structure):
    _fields_ = [("c", FprCall), ("hd", VP), ("R", I32), ("wq", VP), ("ws", VP), ("loc", I32), ("vstart", I32), ("w2", VP),
                ("w1", VP), ("bias0", VP), ("bias0_rs", I64), ("noise", VP), ("nz_s0", I64), ("nz_s1", I64), ("idx", VP),
                ("idx_s0", I64), ("idx_s1", I64), ("temp", VP), ("cache", VP), ("c_s0", I64), ("c_s1", I64), ("tokens", VP),
                ("tok_s0", I64), ("tlogit", VP), ("lse", VP), ("vp_s0", I64), ("sink", I32), ("base_out", VP), ("part", VP),
                ("bar", VP), ("xloc", VP), ("xpeers", VP * 4), ("n", I32), ("k", I32), ("dbg", VP), ("markov", I32),
                ("xq", VP), ("xsg", VP)]


class Ctx3NormInsert3(C.Structure):
    _fields_ = [("c", FprCall), ("kv3", VP), ("ld", I64), ("R", I32), ("shift", I32), ("w", VP * 3), ("out", VP * 3),
                ("pos", VP), ("cos_sin", VP), ("slot", VP), ("cache", VP * 3), ("page_stride", I64), ("eps", F32)]


class Rtag(C.Structure):
    _fields_ = [("c", FprCall), ("f", VP), ("R", I32), ("gate", VP), ("gbias", VP), ("lg", VP), ("cnt", VP), ("rows", VP),
                ("dslot", VP), ("r0", I32), ("kq", I32), ("nc", I32), ("fbufs", VP * 4), ("rbufs", VP * 4), ("flags", VP * 4),
                ("epoch", VP)]


class C1dRun(C.Structure):
    _fields_ = [("c", FprCall), ("packed", VP), ("nly", I32), ("full", I32)]


def _call(t, rank=0):
    import torch
    st = torch.cuda.current_stream(t.device)
    return FprCall(st.cuda_stream, t.device.index, rank)


def _p(t):
    return VP(t.data_ptr()) if t is not None and t.numel() else VP(0)


def pack(st) -> bytes:
    return C.string_at(C.addressof(st), C.sizeof(st))


# ---- packers (= each pybind wrapper's Args fill)
def xg_args(src, out, box, peers, rank, ctr) -> bytes:
    a = XgGather(_call(src, rank), _p(src), _p(out), src.numel(), box.numel() // (2 * 4), _p(box), (VP * 4)(*peers), _p(ctr))
    return pack(a)


def dtail_args(hd, wq, ws, w2, w1, bias0, noise, idx, temp, cache, tokens, tlogit, lse, sink, vstart, base_out, part, bar, xloc,
               xpeers, rank, n, k, dbg, markov, xq, xsg) -> bytes:
    a = Dtail()
    a.c = _call(hd, rank)
    a.hd, a.R = _p(hd), hd.shape[0]
    a.wq, a.ws, a.loc, a.vstart = _p(wq), _p(ws), wq.shape[0], vstart
    a.w2, a.w1 = _p(w2), _p(w1)
    a.bias0, a.bias0_rs = _p(bias0), bias0.stride(0)
    a.noise, a.nz_s0, a.nz_s1 = _p(noise), noise.stride(0), noise.stride(1)
    a.idx, a.idx_s0, a.idx_s1 = _p(idx), idx.stride(0), idx.stride(1)
    a.temp = _p(temp)
    a.cache, a.c_s0, a.c_s1 = _p(cache), cache.stride(0), cache.stride(1)
    a.tokens, a.tok_s0 = _p(tokens), tokens.stride(0)
    a.tlogit, a.lse, a.vp_s0, a.sink = _p(tlogit), _p(lse), lse.stride(0), sink
    a.base_out = _p(base_out)
    a.part, a.bar, a.xloc = _p(part), _p(bar), _p(xloc)
    a.xpeers = (VP * 4)(*xpeers)
    a.n, a.k = n, k
    a.dbg = _p(dbg) if dbg.numel() >= 132 * 16 else VP(0)
    a.markov = 1 if markov else 0
    a.xq, a.xsg = (_p(xq), _p(xsg)) if xq.numel() else (VP(0), VP(0))
    return pack(a)


def ctx3_args(kv3, w, out, pos, cos_sin, slot, cache, page_stride, shift, eps) -> bytes:
    a = Ctx3NormInsert3()
    a.c = _call(kv3)
    a.kv3, a.ld, a.R, a.shift = _p(kv3), kv3.stride(0), kv3.shape[0], shift
    a.w = (VP * 3)(*[x.data_ptr() for x in w])
    a.out = (VP * 3)(*[x.data_ptr() for x in out])
    a.pos, a.cos_sin, a.slot = _p(pos), _p(cos_sin), _p(slot)
    a.cache = (VP * 3)(*[x.data_ptr() for x in cache])
    a.page_stride, a.eps = page_stride, eps
    return pack(a)


def rtag_args(f, gate, gbias, lg, cnt, rows, dslot, r0, kq, rank, nc, fbufs, rbufs, flags, epoch) -> bytes:
    a = Rtag()
    a.c = _call(f, rank)
    a.f, a.R, a.gate, a.gbias, a.lg, a.cnt, a.rows = _p(f), f.shape[0], _p(gate), _p(gbias), _p(lg), _p(cnt), _p(rows)
    a.dslot = _p(dslot)
    a.r0, a.kq, a.nc = r0, kq, nc
    a.fbufs = (VP * 4)(*[x.data_ptr() for x in fbufs])
    a.rbufs = (VP * 4)(*[x.data_ptr() for x in rbufs])
    a.flags = (VP * 4)(*[x.data_ptr() for x in flags])
    a.epoch = _p(epoch)
    return pack(a)


def c1d_args(packed, nly, full) -> bytes:
    return pack(C1dRun(_call(packed), _p(packed), nly, 1 if full else 0))


SIZES = dict(xg=C.sizeof(XgGather), dtail=C.sizeof(Dtail), ctx3=C.sizeof(Ctx3NormInsert3), rtag=C.sizeof(Rtag),
             c1d=C.sizeof(C1dRun))   # = the tables' args_size: 88 / 320 / 152 / 200 / 32
