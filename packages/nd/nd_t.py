"""nd_t.py -- ENG_ND_T=1 (c1-tail, 15 Nov 2026; 2b-beta): bucket T's step-input upload of a mixed step done by nd.

FPR's seam in c1-overlap's step_mixed split (engine.py 6a739938): Engine._mixed_upload_T(r, stT_r, tab, tb_r), called in
_mixed_body's T loop right after tab = self.tb_tail.build(...), inside the rank's device / stream context. Default body
(two statements): self._upload(r, stT_r, tab); stT_r["tab"], stT_r["batch"] = tab, tb_r. _img_rows_T stays at the call
site (Python; this override never touches mm_emb / mm_rt). The override replaces exactly the default:

    i = s["in_i"]; wait the set's holder event          (Engine._upload_v2, dev_pos False: T is host-positioned)
    nd2.upload_inputs(...)   C++ (fpr_nd2 Engine2b, nd_mixed.cpp Engine::upload_inputs): p0::build_inputs (_upload_v2's
                             blob + Hooks._device_tokens_v2's host half; goldens: core/tests/p0_inputs_check.py and
                             c1-overlap's replay_mixed on the real step_mixed) -> every check (sizes, Hooks._dt_d capacity,
                             pair / copy ranges) -> the pinned set -> its H2D into in_d on streams[r] (explicit) -> the
                             device-token ops (migrated copies, the pairs' H2D, index_copy) on the same stream
    the set's event on streams[r], in_i flip, s["n"] / s["n_sample"], Hooks._dt_i[r] flip when pairs were issued (as
    _device_tokens_v2: Python's own staging parity unchanged), tab / batch

Host-side state = _upload_v2's, field for field (its source): in_h[i][1] (the holder event), in_i, n, n_sample, the stub
indexer's ti / tl copies (none: the real indexer is required), Hooks._dt_i (+ _dt[r][i]'s event: nd's own staging is used, so
Python's set keeps its older event, which only makes a later wait weaker-or-equal, never wrong); in_h / in_hv are created
by the default path (the first upload of a bucket falls back).
Falls back to the default (today's code) unless ALL hold (FPR 15 Nov): cfg.upload_v2, ENG_DEVTOK=1, the real indexer
(hooks.idx), no image rows in the step (StepContext mm), the bucket's sets exist; and when nd refuses (ValueError: nothing
was issued). The first fallback of each reason is logged once. stats nd_t_native / nd_t_fallback / nd_t_why_<reason>.
ENG_ND_T_TRACE=1 (gates, both arms): after every T upload, one "nd_t trace" line per (mixed step, rank) with the host-side
fields and md5s of the used set's bytes and of in_d (D2H after a device sync); the ENG_ND_T 0 vs 1 arms must print the
same lines.

    import nd_t; nd_t.install(eng, nd2)     # after nd2 (fpr_nd2 Engine2b) is initialized on eng's handover
"""
from __future__ import annotations

import hashlib
import os
import sys
import types


def _why(eng, E) -> str:
    """the first condition of FPR's list that fails ("" = native)"""
    if not eng.cfg.upload_v2:
        return "upload_v1"
    if not E.DEVTOK_V2:
        return "devtok_v1"
    if eng.hooks.idx is None:
        return "stub_indexer"
    mm = getattr(eng, "_mm", None)
    if mm and any(m is not None and m[0] is not None and m[0].numel() for m in mm):
        return "image_rows"
    return ""


def install(eng, nd2) -> bool:
    """Override eng._mixed_upload_T (instance attribute; the class keeps the default). ENG_ND_T_TRACE=1 wraps the
    default too (the ENG_ND_T=0 arm of a gate). Returns True when the native override is installed."""
    native = os.environ.get("ENG_ND_T", "0") == "1"
    trace = os.environ.get("ENG_ND_T_TRACE", "0") == "1"
    if not native and not trace:
        return False
    E = sys.modules[type(eng).__module__]
    if not hasattr(type(eng), "_mixed_upload_T"):
        raise RuntimeError("ENG_ND_T: the engine has no _mixed_upload_T seam (c1-overlap's step_mixed split needed)")
    import torch
    default = types.MethodType(type(eng)._mixed_upload_T, eng)
    wpt = int(E.WINDOW_PAGE_TOKENS)
    nt = {"step": 0, "last_serial": None}

    def _fallback(self, why, r, s, tab, tb_r):
        st = self.stats
        st["nd_t_fallback"] = st.get("nd_t_fallback", 0) + 1
        k = "nd_t_why_" + why.split(":")[0].replace(" ", "_")
        if not st.get(k):
            print(f"nd_t: native T upload not taken ({why}); Engine._upload runs (logged once per reason)", flush=True)
        st[k] = st.get(k, 0) + 1
        return default(r, s, tab, tb_r)

    def _native(self, r, s, tab, tb_r):
        why = _why(self, E)
        if not why and (s.get("in_h") is None or s.get("in_d") is None):
            why = "no_sets_yet"
        if why:
            return _fallback(self, why, r, s, tab, tb_r)
        i = s["in_i"]
        hb, ev = s["in_h"][i]
        if ev is not None:
            ev.synchronize()                     # the set's copy from two steps ago (as _upload_v2)
        prev = []
        hsd_on = self.hooks.hsd is not None
        if hsd_on and tab.n:
            for rid, row0, nr, kind, last in tab.seq_rows:
                if kind == "decode":
                    p = self.prev_of(rid)
                    if p is not None:
                        prev.append((rid, int(p[0]), int(p[1])))
        try:
            n, ns, npairs = nd2.upload_inputs(r, int(s["R"]), int(self.pools[r].scratch_window_page * wpt), hsd_on,
                                              s["in_d"].data_ptr(), s["in_d"].numel(), hb.data_ptr(), hb.numel(), tab,
                                              prev)
        except ValueError as e:                  # refused before any device op: today's code
            return _fallback(self, f"refused: {str(e)[:120]}", r, s, tab, tb_r)
        ev = torch.cuda.Event()
        ev.record(self.streams[r])               # explicit (FPR): the stream nd issued on
        s["in_h"][i][1] = ev
        s["in_i"] = i ^ 1
        s["n"], s["n_sample"] = int(n), int(ns)
        if npairs:                               # _device_tokens_v2's set flip (Python's staging parity)
            self.hooks._dt_i[r] ^= 1
        s["tab"], s["batch"] = tab, tb_r
        self.stats["nd_t_native"] = self.stats.get("nd_t_native", 0) + 1

    def _traced(self, r, s, tab, tb_r):
        i0 = s.get("in_i")
        dti0 = list(getattr(self.hooks, "_dt_i", []) or [])
        (_native if native else default)(r, s, tab, tb_r)
        ser = getattr(self, "step_serial", None)
        if ser != nt["last_serial"]:
            nt["last_serial"] = ser
            nt["step"] += 1
        torch.cuda.synchronize(s["dev"])
        used = i0 if i0 is not None else 0
        hset = s["in_h"][used][0]
        f = dict(step=nt["step"], r=r, R=int(s["R"]), in_i0=i0, in_i=s["in_i"], n=s["n"], n_sample=s["n_sample"],
                 tab=s["tab"] is tab, batch=s["batch"] is tb_r, ev=[x[1] is not None for x in s["in_h"]],
                 dt_i=[a ^ b for a, b in zip(dti0, list(getattr(self.hooks, "_dt_i", []) or []))],
                 set_md5=hashlib.md5(hset.numpy().tobytes()).hexdigest()[:12],
                 in_d_md5=hashlib.md5(s["in_d"].cpu().numpy().tobytes()).hexdigest()[:12])
        print("nd_t trace " + " ".join(f"{k}={v}" for k, v in f.items()), flush=True)

    fn = _traced if trace else _native
    eng._mixed_upload_T = types.MethodType(fn, eng)
    return native
