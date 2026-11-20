"""c1d2: c1d's DrafterC1M on c1-mega's srv4 layer code (C1 tail session, 12 Nov 2026; TEST. c1d.py is on the live serving
path and stays untouched: this module subclasses it).

Kinds (sources: c1d_src2/ = a frozen copy of fpr/kernels/c1mega/srv4 + c1d_src's pre-x2b expert fragments; wrappers):
  s4t  c1d_s4t.cu: srv4's deployment-style build (attention mode 3 fixed, dense A5 e8, whole-layer kernel only, no index /
       Engram phases) with the drafter knobs and every matrix from c1d's fp8 copies (C1M_DEP_B16 0); timeline kept
  s4   c1d_s4.cu: the same without the timeline (c1mega_depn.cu, the smallest code per launch)
  s4p  c1d_s4p.cu: s4 with the pre-x2b expert fragments (c1d's today)
  s4b  c1d_s4b.cu: s4 with srv4's bf16 set (bits 15): q_a | wkv, wo_a, wo_b and the shared expert streamed from the
       DraftRank's own bf16 tensors (wqkv_a [1792, 5120], wo_aT [8, 4096, 1024], wo_b [5120, 8192], sw13 / sw2 = this
       rank's shared-expert slices: the shapes c1mega's maps3 checks), no copies; c1d's fp8 copies of those are dropped
The host interface is c1d's: pack_layer is byte-identical in c1d_src/c1mega.cu (ee1302dd) and srv4/c1mega.cu; with bf16 bits
0 the newer Args fields (bf16 / shared-expert / q_b W8A16 pointers) stay null = the fp8 paths. Differences handled here:
the build's fixed_config() is checked (amode 3, bf16 bits = the kind's), and the expert-score scratch is sized like srv4's adapter
(3 x NRM x 384 floats). Builds go where c1d.ext puts them (fpr/kernels/hsdtail/build/c1d_<wrapper>; the bench redirects
every JIT build into its private tree).
"""
from __future__ import annotations

import os

import torch

import c1d as _C

HERE = _C.HERE
KINDS = {"s4t": (os.path.join(HERE, "c1d_s4t.cu"), ()), "s4": (os.path.join(HERE, "c1d_s4.cu"), ()),
         "s4p": (os.path.join(HERE, "c1d_s4p.cu"), ()), "s4b": (os.path.join(HERE, "c1d_s4b.cu"), ())}
# + C1M_INS_OLD (t-83b: srv4's new insert / mode-3 push path assumes WIN <= 128 and hangs at the drafter's WIN 131; the
# "o" wrappers select the old, WIN-general path; route them through hsdt_c1g, hsdt_c1f's kind list is frozen)
KINDS.update({k + "o": (os.path.join(HERE, f"c1d_{k}o.cu"), ()) for k in ("s4", "s4b", "s4t", "s4p")})
B16 = {"s4b": 15, "s4bo": 15}   # compile-time bf16 bits (1 q_a | wkv, 2 wo_a, 4 wo_b, 8 shared expert): the DraftRank's own bf16 tensors
for k_, v_ in KINDS.items():         # runtime registration: c1d.DrafterC1M.__init__ looks kinds up in c1d.KINDS
    _C.KINDS.setdefault(k_, v_)


class DrafterC1M(_C.DrafterC1M):
    """c1d.DrafterC1M for the srv4-based kinds (module docstring)."""

    def __init__(self, hsdt, rd, cfg, log=print, kind="s4"):
        assert kind in KINDS, kind
        super().__init__(hsdt, rd, cfg, log=log, kind=kind)
        fc = list(self.X.fixed_config())
        am, bits = fc[0], fc[1]
        assert am in (-1, 3), f"c1d2 {kind}: build fixes amode {am}; c1d packs amode 3"
        self.has_tim = bool(fc[2]) if len(fc) > 2 else True
        if self.tim and not self.has_tim:
            log(f"c1d2 ({kind}): C1D_TIM set but this build has no timeline (no stamps)")
        self.b16 = B16.get(kind, 0)
        assert bits in (-1, self.b16), f"c1d2 {kind}: build bf16 bits {bits} != {self.b16}"
        self.b16t = {}
        if self.b16:
            self._b16_weights(hsdt)
        log(f"c1d2 ({kind}): fixed_config {fc}, bf16 bits {self.b16}")

    def _b16_weights(self, hsdt):
        """bits 15: wt / fw with empty tensors where the bf16 ones replace c1d's fp8 copies (as srv4's adapter), maps3."""
        for r, dev in enumerate(self.devs):
            d0 = hsdt.ranks0[r]
            with torch.cuda.device(dev):
                z8 = torch.zeros(0, dtype=torch.uint8, device=dev)
                for i, B in enumerate(d0.blocks):
                    wt, fw, _ = self.w[(r, i)]
                    wt, fw = list(wt), list(fw)
                    wt[0] = wt[1] = wt[2] = wt[3] = z8            # wo_a / wo_b fp8 + scales (bits 2, 4)
                    wt[10] = wt[11] = wt[12] = wt[13] = z8        # shared expert w1 | w3, w2 fp8 + scales (bit 8)
                    fw[4] = fw[5] = z8                            # q_a | wkv fp8 + scales (bit 1)
                    b16t = (B["wqkv_a"].contiguous(), B["wo_aT"].contiguous(), B["wo_b"].contiguous())
                    sht = (B["sw13"].contiguous(), B["sw2"].contiguous())
                    tm = self.X.maps3(wt[0], wt[2], fw[4], fw[8], wt[10], wt[14], wt[15], wt[4], fw[0], *b16t, *sht).to(dev)
                    self.w[(r, i)] = (wt, fw, tm)
                    self.b16t[(r, i)] = (b16t, sht)
            torch.cuda.synchronize(dev)

    def packs(self, owner, dranks):
        """= c1d.DrafterC1M.packs with the bf16 / shared-expert pointers patched (patch_bf16 / patch_sh16) for b16 kinds."""
        if not self.b16:
            return super().packs(owner, dranks)
        key = (owner, id(dranks[owner]))
        if key in self._packs:
            return self._packs[key]
        X, Bs, mc = self.X, self.B, self.cfg
        do = dranks[owner]
        b = do.b
        Rd = do.Rd
        ar1, ar2 = [x["ar1"] for x in Bs], [x["ar2"] for x in Bs]
        qall, ogl = [x["qall"] for x in Bs], [x["ogl"] for x in Bs]
        rdy = [x["rdy"].data_ptr() for x in Bs]
        stg = [x["stg"].data_ptr() for x in Bs]
        si = b["idx"].view(Rd, -1)
        out = []
        for r, dev in enumerate(self.devs):
            B = Bs[r]
            ch = B["chain"]
            blobs = []
            for i in range(len(do.blocks)):
                wt, fw, tm = self.w[(r, i)]
                b16t, sht = self.b16t[(r, i)]
                if i == 0:
                    fin = [b["zero_y"], b["X0"], b["zero_post"], b["eye"].view(Rd, 16), b["onehot"]]
                else:
                    fin = [ch[10], ch[1], ch[5], ch[6], ch[4]]
                fin += [b["q_pos"], b["q_slot"], si, b["lens"], B["ti0"], B["tl0"]]
                fo = [B["Xo"], B["ain"], B["apre"], B["apost"], B["acomb"], B["qakv"], B["qrout"], B["kvout"], B["q"], B["o"]]
                ins = [B["o"], B["Xo"], B["apost"], B["acomb"], B["apre"]]
                stamped = self.tim and i == (0 if self.tim == "first" else len(do.blocks) - 1)
                with torch.cuda.device(dev):
                    p_ = X.pack_layer(r, Rd, ins, wt, ch, B["scr"] + [tm], ar1, ar2, B["tim"] if stamped else B["tim"][:0],
                                      mc.norm_eps, mc.hc_eps, mc.hc_iters,
                                      1.5, fin, fw, fo, B["fscr"], do.cache[i], do.page_stride, 32, B["comp0"], 1, 1,
                                      mc.norm_eps, 512 ** -0.5, 1 if r == owner else 0, 3, owner, qall, ogl)
                    p_ = X.patch_integ(p_, rdy, si.stride(0), B["ti0"].stride(0))
                    p_ = X.patch_bf16(p_, *[t.data_ptr() for t in b16t])
                    p_ = X.patch_sh16(p_, *[t.data_ptr() for t in sht])
                    blobs.append(X.patch_stg(p_, stg))
            out.append(torch.cat(blobs).to(dev))
        for d in self.devs:
            torch.cuda.synchronize(d)
        self._packs[key] = out
        return out

    # ------------------------------------------------------------------ C1D_HANGDUMP (tests, t-83b: s4 hung in its first run)
    CN = {0: "Z", 1: "SS", 2: "RT", 3: "SH", 4: "SSA", 5: "MIXA", 6: "QA", 7: "QB", 8: "SHT", 44: "X2G", 47: "MXG", 72: "O",
          73: "AT0", 74: "AT1", 75: "AT2", 76: "AT3", 77: "MIXF", 78: "INS", 79: "MXA"}

    def launch(self, r, owner, dranks):
        """c1d.DrafterC1M.launch + C1D_HANGDUMP=<s> (with C1D_SYNC=1): if the eager run has not completed <s> seconds after
        the last rank's launch, print every GPU's c1mega counters (scr[9]), per-CTA layer ticks (scr[10]), ready flags and
        the stamped block's per-stamp CTA counts, read by copies on a side stream (the hung kernel holds the SMs, not the
        copy engines). Pinned buffers and streams are made here (main thread) before the launch."""
        hd = float(os.environ.get("C1D_HANGDUMP", "0") or 0)
        tmr = None
        if (hd > 0 and r == _C.W - 1 and not getattr(self, "_dumped", False)
                and not torch.cuda.is_current_stream_capturing()):
            import threading
            if not hasattr(self, "_hd"):
                self._hd = []
                for rr, dev in enumerate(self.devs):
                    B = self.B[rr]
                    src = [B["scr"][9], B["scr"][10], B["rdy"], B["tim"]]
                    self._hd.append((dev, torch.cuda.Stream(device=dev), src,
                                     [torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in src]))
            tmr = threading.Timer(hd, self._hang_dump, args=(owner, hd))
            tmr.daemon = True
            tmr.start()
        try:
            super().launch(r, owner, dranks)
        finally:
            if tmr is not None:
                tmr.cancel()

    def _hang_dump(self, owner, hd):
        self._dumped = True
        print(f"c1d2 HANG DUMP ({self.kind}, owner {owner}): the eager run has not completed after {hd:.0f} s", flush=True)
        for rr, (dev, s, src, dst) in enumerate(self._hd):
            try:
                with torch.cuda.device(dev), torch.cuda.stream(s):
                    for d_, t in zip(dst, src):
                        if t.numel():
                            d_.copy_(t, non_blocking=True)
                s.synchronize()
            except Exception as e:  # noqa: BLE001
                print(f"  GPU {rr}: read failed: {e}", flush=True)
                continue
            cnt, tick, rdy, tim = dst
            cs = " ".join(f"{n}={int(cnt[i])}" for i, n in sorted(self.CN.items()))
            hs = " ".join(f"{int(v)}" for v in cnt[48:72])
            print(f"  GPU {rr}: {cs}\n    H[0..23]={hs}\n    tick min {int(tick.min())} max {int(tick.max())}; "
                  f"rdy[0..131] min {int(rdy[:132].min())} max {int(rdy[:132].max())}, rdy[132..]={rdy[132:].tolist()}",
                  flush=True)
            if tim.numel():
                ts = tim.view(132, 192)[:, :48]
                seen = [(i, int((ts[:, i] > 0).sum())) for i in range(48)]
                print("    stamps (id: CTAs that reached it): " + " ".join(f"{i}:{n}" for i, n in seen if n), flush=True)

    def _buffers(self):
        super()._buffers()
        for r, dev in enumerate(self.devs):          # srv4's adapter: scr[4] = 3 x NRM x 384 floats (c1d: NRM x 384)
            self.B[r]["scr"][4] = torch.zeros(3 * _C.NRM * 384, dtype=torch.float32, device=dev)
