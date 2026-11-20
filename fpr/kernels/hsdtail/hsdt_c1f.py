"""hsdt_c1f: hsdt_c1e's HSDT + switch "nsplit" + c1d kinds s4 / s4t / s4p / s4b (c1d2.py) (C1 tail session, 12 Nov 2026; TEST, not in any served config).

nsplit: graph H issues the draft noise + slot 0's bias (the side-stream block of hsdt.HSDT._steps) right after _post_glue,
before _x0_glue, instead of after it. The noise needs only post_glue's dmi / dmp; c1d (one persistent launch that needs
every SM) then no longer waits for the noise kernel's SMs after x0_glue (t-70 server trace, owner GPU 0: x0_glue ends
169.5 us, the noise 170.1 -> 174.0, the context chain 173.7, c1d starts 174.2). The bias gather / GEMM stay behind c1d in
practice (c1d is ready first and holds every SM; they run in its tail, as today). Same kernels, same inputs, same values.
Pair it with ctx3f (the context chain's 3 norm_rope + insert pairs as one launch), else the context chain still gates c1d.

hsdt.py is frozen: _steps here is hsdt.HSDT._steps' own source with ONE edit (moved block), 
compiled with hsdt's live module globals; the module refuses to load if that source changed (exact-text checks).
"""
from __future__ import annotations

import inspect
import textwrap
import types

import hsdt as _H
import hsdt_c1e as _E

HAS_CTX3 = True
HAS_NSPLIT = True                  # the bench passes nsplit=False per variant when the module has it
_C1D2_KINDS = ("s4t", "s4", "s4p", "s4b")   # c1d kinds served by c1d2.DrafterC1M (c1-mega's srv4 layer code)
parse_switches_c1e = _E.parse_switches


def parse_switches(spec):
    items = [x.strip() for x in spec.replace("+", ",").split(",")]
    own = ("nsplit", "sfuse", "pfuse", "mfuse", "dhef", "vwin", "vchunk", "mpwin", "mkwin")
    out = parse_switches_c1e(",".join(x for x in items if x and x not in own))
    for k_ in own:
        if k_ in items:
            out[k_] = True
    return out


_OLD_X0 = '''        HS._x0_glue[(Dp, k, triton.cdiv(HS.H, 1024))](d.b["X0"], self.emb[r], g["anc"], r * Dp, DR.NOISE_ID, K=k, HC_=DR.HC,
                                                      D_=HS.H, BLOCK=1024)
        anchors_all = g["anc"][:n]
        dm = S.DraftMeta(idx=g["dmi"][:n], sample_pos=g["dmp"][:n], num_reqs=n)
        ev = torch.cuda.Event()
        ev.record(cur)
        side = self.side[r] if HS.NOISE_SIDE else cur
        side.wait_event(ev)
        with torch.cuda.stream(side):
            S.draft_noise(st, self.noise[r][:, :n], dm)
            torch.index_select(st.w1, 0, anchors_all, out=st.emb[0, :n])
            torch.mm(st.emb[0, :n], st.w2.t(), out=st.bias[:n])
            ev2 = torch.cuda.Event()
            ev2.record(side)
        self._mark(r, "noise+bias0")
'''
_NEW_X0 = '''        anchors_all = g["anc"][:n]
        dm = S.DraftMeta(idx=g["dmi"][:n], sample_pos=g["dmp"][:n], num_reqs=n)

        def _noise_bias():
            ev = torch.cuda.Event()
            ev.record(cur)
            side = self.side[r] if HS.NOISE_SIDE else cur
            side.wait_event(ev)
            with torch.cuda.stream(side):
                S.draft_noise(st, self.noise[r][:, :n], dm)
                torch.index_select(st.w1, 0, anchors_all, out=st.emb[0, :n])
                if self.sw.get("mkwin"):   # mkwin (hsdt_c1f): the Markov head w2 under an L2 Streaming window
                    __import__("hsdt_c1f").vwin(side, st.w2)
                torch.mm(st.emb[0, :n], st.w2.t(), out=st.bias[:n])
                if self.sw.get("mkwin"):
                    __import__("hsdt_c1f").vwin(side, None)
                ev2_ = torch.cuda.Event()
                ev2_.record(side)
            self._mark(r, "noise+bias0")
            return ev2_

        _ns_ = bool(self.sw.get("nsplit"))
        if _ns_:   # nsplit: the side-stream block before x0_glue (needs only post_glue's dmi / dmp / anc)
            ev2 = _noise_bias()
        HS._x0_glue[(Dp, k, triton.cdiv(HS.H, 1024))](d.b["X0"], self.emb[r], g["anc"], r * Dp, DR.NOISE_ID, K=k, HC_=DR.HC,
                                                      D_=HS.H, BLOCK=1024)
        if not _ns_:
            ev2 = _noise_bias()
'''
_OLD_DR = """        S.verify2_decide(st, st.g1(T), st.g2(T), m, T)
        self._mark(r, "decide")
        S.verify2_resample(st, lg, st.g1(T), m)
"""
_NEW_DR = """        if self.sw.get("sfuse"):   # sfuse (hsdt_c1f): decide + resample as ONE kernel (sampfuse.py; the same outputs)
            __import__("sampfuse").decide_resample(st, lg, st.g1(T), st.g2(T), m, T)
            self._mark(r, "decide")
        else:
            S.verify2_decide(st, st.g1(T), st.g2(T), m, T)
            self._mark(r, "decide")
            S.verify2_resample(st, lg, st.g1(T), m)
"""
_OLD_FP = """        S.verify2_finish(st, st.g3(n), m)
        self._mark(r, "finish")
        # ---- drafter metadata from the result (GPU only)
        g = self._glue_bufs(r, n)
        HS._post_glue[(n,)](st.num_sampled, st.last, tb["g_dslot"], tb["g_p0"], n, g["dmi"], g["dmp"], g["anc"], r * Dp, Dp,
                            tb["wbase"], tb["wpages"], tb["rstart"], tb["lslot"], d.b["q_pos"], d.b["q_slot"], d.b["idx"],
                            d.b["idx"].stride(0), g["gns"], K=k, KP=triton.next_power_of_2(k), NBW_=HS.NBW, WIN=DR.WINDOW,
                            WBLK=triton.next_power_of_2(DR.WINDOW + k), POOL=self.pool, RING_=DR.RING, MAXS=self.max_seqs,
                            num_warps=4)
"""
_NEW_FP = """        if self.sw.get("pfuse"):   # pfuse (hsdt_c1f): finish + post_glue as ONE kernel (sampfuse.py; the same outputs)
            g = self._glue_bufs(r, n)
            __import__("sampfuse").finish_post(st, st.g3(n), m, tb, g, r * Dp, Dp, d, k, HS.NBW, DR.WINDOW, self.pool,
                                               DR.RING, self.max_seqs)
            self._mark(r, "finish")
        else:
            S.verify2_finish(st, st.g3(n), m)
            self._mark(r, "finish")
            # ---- drafter metadata from the result (GPU only)
            g = self._glue_bufs(r, n)
            HS._post_glue[(n,)](st.num_sampled, st.last, tb["g_dslot"], tb["g_p0"], n, g["dmi"], g["dmp"], g["anc"], r * Dp,
                                Dp, tb["wbase"], tb["wpages"], tb["rstart"], tb["lslot"], d.b["q_pos"], d.b["q_slot"],
                                d.b["idx"], d.b["idx"].stride(0), g["gns"], K=k, KP=triton.next_power_of_2(k), NBW_=HS.NBW,
                                WIN=DR.WINDOW, WBLK=triton.next_power_of_2(DR.WINDOW + k), POOL=self.pool, RING_=DR.RING,
                                MAXS=self.max_seqs, num_warps=4)
"""
_OLD_MK = """            for i in range(k):
                S.draft3_local(st, b3[:, i], st.emb[i % 2, :n], dm.idx[:, i], self.noise[r][i, :n], i, 2048,
                               bias_ready=i == 0)
                yield ("ag", f"d{i}", st.d_out[:n], [x.d_gath(n, i) for x in sts])
                S.draft_combine(st, st.d_gath(n, i), n, i, st.emb[(i + 1) % 2, :n], dm.idx[:, i])
"""
_NEW_MK = """            _mf_ = bool(self.sw.get("mfuse"))
            if self.sw.get("mkwin"):   # mkwin: the draws' w2 reads (cuBLAS bias GEMMs or mfuse's kernel) evict-first
                __import__("hsdt_c1f").vwin(cur, st.w2)
            for i in range(k):
                if _mf_ and i > 0:   # mfuse (hsdt_c1f): combine(i - 1) + slot i's bias + draft3(i) as ONE kernel (sampfuse)
                    __import__("sampfuse").draft_step(st, b3[:, i], st.d_gath(n, i - 1), n, i, dm.idx[:, i], dm.idx[:, i - 1],
                                                      self.noise[r][i, :n], st.emb[i % 2, :n])
                else:
                    S.draft3_local(st, b3[:, i], st.emb[i % 2, :n], dm.idx[:, i], self.noise[r][i, :n], i, 2048,
                                   bias_ready=i == 0)
                yield ("ag", f"d{i}", st.d_out[:n], [x.d_gath(n, i) for x in sts])
                if not _mf_ or i == k - 1:
                    S.draft_combine(st, st.d_gath(n, i), n, i, st.emb[(i + 1) % 2, :n], dm.idx[:, i])
            if self.sw.get("mkwin"):
                __import__("hsdt_c1f").vwin(cur, None)
"""
_OLD_DH = "                _DTAIL.dtail(self.hd_all[r][: n * k], dh.wq,"
_NEW_DH = ("                (__import__(\"hsdt_c1f\").dtail2_ext() if self.sw.get(\"dhef\") else _DTAIL).dtail("
           "self.hd_all[r][: n * k], dh.wq,")
_OLD_VH = """        lg = self.vhead[r](self.h_all[r][:T])
"""
_NEW_VH = """        if self.sw.get("vchunk"):   # vchunk (hsdt_c1f): the verify head as <= 128 MiB GEMM chunks, each under its window
            lg = __import__("hsdt_c1f").vhead_chunked(cur, self.vhead[r], self.h_all[r][:T])
        elif self.sw.get("vwin"):   # vwin (hsdt_c1f): the verify head's weight stream under an L2 Streaming access window
            _vw_ = __import__("hsdt_c1f")
            _vw_.vwin(cur, self.vhead[r].w)
            lg = self.vhead[r](self.h_all[r][:T])
            _vw_.vwin(cur, None)
        else:
            lg = self.vhead[r](self.h_all[r][:T])
"""
_OLD_MP = """            with torch.cuda.stream(sctx):
                d.ctp_mid(self, r, NT)
"""
_NEW_MP = """            with torch.cuda.stream(sctx):
                if self.sw.get("mpwin"):   # mpwin (hsdt_c1f): ctxtp's main_proj slice (39 MB) under an L2 Streaming window
                    __import__("hsdt_c1f").vwin(sctx, d.main_proj[r * CTP_COLS:(r + 1) * CTP_COLS])
                d.ctp_mid(self, r, NT)
                if self.sw.get("mpwin"):
                    __import__("hsdt_c1f").vwin(sctx, None)
"""
_OLD_SUPER = "yield from super()._steps(r, h_rows, aux_rows)"
_NEW_SUPER = "yield from HS.HSD._steps(self, r, h_rows, aux_rows)"   # hsdt.HSDT's base (zero-arg super needs a class)


def _patched_steps():
    src = inspect.getsource(_H.HSDT._steps)
    for old in (_OLD_X0, _OLD_DR, _OLD_FP, _OLD_MK, _OLD_DH, _OLD_VH, _OLD_MP, _OLD_SUPER):
        if src.count(old) != 1:
            raise RuntimeError("hsdt_c1f: hsdt.HSDT._steps changed (patch anchor not found exactly once); refusing to load")
    src = (src.replace(_OLD_X0, _NEW_X0).replace(_OLD_DR, _NEW_DR).replace(_OLD_FP, _NEW_FP).replace(_OLD_MK, _NEW_MK)
           .replace(_OLD_DH, _NEW_DH).replace(_OLD_VH, _NEW_VH).replace(_OLD_MP, _NEW_MP).replace(_OLD_SUPER, _NEW_SUPER))
    assert _H.HSDT.__mro__[1] is _H.HS.HSD
    code = compile(textwrap.dedent(src), "<hsdt_c1f._steps>", "exec")
    fco = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == "_steps")
    return types.FunctionType(fco, vars(_H), "_steps")   # hsdt's LIVE globals (_MHCF / _DTAIL load lazily)


_DT2 = None


def dtail2_ext():
    """dtail2.cu = dtail.cu with the weight stream under an L2 evict_first policy (switch "dhef"; build dir build/dtail2)."""
    global _DT2
    if _DT2 is None:
        import os
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(_H.__file__))
        bd = os.path.join(here, "build", "dtail2")
        os.makedirs(bd, exist_ok=True)
        _DT2 = load(name="hsdt_dtail2", sources=[os.path.join(here, "dtail2.cu")], build_directory=bd, verbose=False,
                    extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])
    return _DT2


_VW = {}


def vwin(stream, w):
    """Switch "vwin": cudaStreamSetAttribute(AccessPolicyWindow) on `stream` = w's bytes (capped at the device's max window
    size) with hitProp / missProp Streaming (L2 evicts these lines first), or w = None: the window cleared. Kernels launched
    (or captured) on the stream in between carry it."""
    import ctypes as C
    import sys
    import torch
    if "lib" not in _VW:
        names = sorted({ln.split()[-1] for ln in open("/proc/self/maps") if "/libcudart.so" in ln})
        names += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
        lib = None
        for nm in names:
            try:
                lib = C.CDLL(nm)
                break
            except OSError:
                continue
        assert lib is not None, "vwin: no libcudart"
        _VW["lib"] = lib
    lib = _VW["lib"]

    class APW(C.Structure):
        _fields_ = [("base_ptr", C.c_void_p), ("num_bytes", C.c_size_t), ("hitRatio", C.c_float), ("hitProp", C.c_int),
                    ("missProp", C.c_int)]

    buf = (C.c_ubyte * 64)()
    win = APW.from_buffer(buf)
    if w is not None:
        dev = w.device.index
        if ("max", dev) not in _VW:
            v = C.c_int(0)
            lib.cudaDeviceGetAttribute(C.byref(v), 109, dev)   # cudaDevAttrMaxAccessPolicyWindowSize
            _VW[("max", dev)] = v.value
            print(f"hsdt_c1f vwin: device {dev} max access-policy window {v.value / 2**20:.1f} MiB, verify head weight "
                  f"{w.numel() * w.element_size() / 2**20:.1f} MiB", file=sys.stderr, flush=True)
        win.base_ptr = w.data_ptr()
        win.num_bytes = min(w.numel() * w.element_size(), _VW[("max", dev)])
        win.hitRatio, win.hitProp, win.missProp = 1.0, 1, 1      # cudaAccessPropertyStreaming
    rc = lib.cudaStreamSetAttribute(C.c_void_p(stream.cuda_stream), 1, buf)   # cudaStreamAttributeAccessPolicyWindow
    assert rc == 0, f"vwin: cudaStreamSetAttribute rc {rc}"


def vhead_chunked(stream, vh, h):
    """Switch "vchunk": = head.VerifyHead.__call__ (o = h @ w^T into vh.out) as row chunks of w of at most the device's max
    access-policy window (128 MiB on H100 = 13,107 rows of 5,120 bf16 -> 3 chunks for 32,320 vocab rows), each launched
    under a Streaming window over exactly its rows (vwin), so the whole 316 MiB stream is evict-first and L2's other content
    (c1mega / c1d code, m-114) stays resident. Each chunk writes its columns of vh.out in place: out = o^T[a:b] (strides
    1 x V, cuBLAS's transposed C), the same per-element K-sum as the one-call GEMM unless cuBLAS picks another kernel."""
    import torch
    w = vh.w
    o = vh.out[: h.shape[0]]
    V, K = w.shape
    dev = w.device.index
    if ("max", dev) not in _VW:
        vwin(stream, w[:1])                                    # queries + logs the device's max window once
    rows = max(256, (_VW[("max", dev)] // (K * w.element_size())) // 256 * 256)
    ot = o.t()
    for a in range(0, V, rows):
        b = min(V, a + rows)
        vwin(stream, w[a:b])
        torch.mm(w[a:b], h.t(), out=ot[a:b])
    vwin(stream, None)
    return o


class HSDT(_E.HSDT):
    """hsdt_c1e.HSDT + nsplit (module docstring)."""

    _steps = _patched_steps()

    def configure(self, nsplit=None, sfuse=None, pfuse=None, mfuse=None, dhef=None, vwin=None, vchunk=None, mpwin=None,
                  mkwin=None, c1d=None, **kw):
        if isinstance(c1d, str) and c1d in _C1D2_KINDS:   # c1d on srv4's layer code (c1d2.py): seed hsdt's instance cache
            import c1d2 as _D2
            insts_ = self.__dict__.setdefault("_c1d_insts", {})
            if c1d not in insts_:
                from types import SimpleNamespace as NS_
                d0 = self.ranks0[0]
                _H.mhcf_ext()
                insts_[c1d] = _D2.DrafterC1M(self, self.rd, NS_(norm_eps=d0.eps, hc_eps=d0.hc_eps, hc_iters=d0.iters),
                                             kind=c1d)
        super().configure(c1d=c1d, **kw)
        if nsplit is not None:
            self.sw["nsplit"] = bool(nsplit)
        self.sw.setdefault("nsplit", False)
        if sfuse is not None:
            self.sw["sfuse"] = bool(sfuse)
        self.sw.setdefault("sfuse", False)
        if pfuse is not None:
            self.sw["pfuse"] = bool(pfuse)
        self.sw.setdefault("pfuse", False)
        if mfuse is not None:
            self.sw["mfuse"] = bool(mfuse)
        self.sw.setdefault("mfuse", False)
        for k_, v_ in (("dhef", dhef), ("vwin", vwin), ("vchunk", vchunk), ("mpwin", mpwin), ("mkwin", mkwin)):   # L2 evict-first
            if v_ is not None:
                self.sw[k_] = bool(v_)
            self.sw.setdefault(k_, False)
        if self.sw.get("dhef"):
            dtail2_ext()
