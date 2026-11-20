"""hsdt_c1h: hsdt_c1g (hsdt_c1f + the c1d2 kinds + "c1d=<kind>") with the graph-H bundle's switches GATED to C1 steps
(C1 tail session, 13 Nov 2026; TEST, not in any served config).

Why: every 128-seat soak with the ungated bundle (uwcm9xh / uwcm9xhd: ctx3f, nsplit, dhef, vchunk, mpwin, mkwin) lost
5-13 % out tok/s vs live uwcm9x on the same seed (t-84 / t-85 / t-85b / t-87). The evict-first windows' purpose is C1 only
(keep c1mega's code in L2 between graph L launches); at large batches the head GEMMs reuse every weight tile across row
blocks and an evict-first hint throws that reuse away. Main's rule (13 Nov ~21:00): changes on a path that runs at every
batch size are measured at C128 before deploy; gate the bundle to S <= 4.

How: the same per-step pattern as hsdt_c1d's c1d gate (HSDT_C1D_SMAX): after HSD's prepare, each switch named in
HSDT_GATE_SW (default "nsplit,sfuse,pfuse,dhef,vwin,vchunk,mpwin,mkwin") keeps its configured value only on PURE-DECODE
steps with S_pad <= HSDT_GATE_SMAX (default 4 = the C1 bucket at k = 3: S_pad counts sample rows) and NT == S_pad (no
prefill chunk tail); otherwise it is off for the step. _steps reads self.sw while it runs (eager step and graph capture);
graph H's key carries S_pad / NT, so every captured graph has one fixed switch set and replays never re-read sw.
ctx3 / ctx3f is NOT gated here (DraftRankT3 state at configure; measured on its own, t-88).

Engine side: ENG_HSDT_MOD=hsdt_c1h (engine.py's hook imports HSDT and parse_switches from this module).
"""
from __future__ import annotations

import os

import hsdt_c1g as _G

GATE = tuple(x for x in os.environ.get("HSDT_GATE_SW", "nsplit,sfuse,pfuse,dhef,vwin,vchunk,mpwin,mkwin").split(",") if x)
GSMAX = int(os.environ.get("HSDT_GATE_SMAX", "4"))
parse_switches = _G.parse_switches
HAS_CTX3 = _G.HAS_CTX3
HAS_NSPLIT = _G.HAS_NSPLIT


class HSDT(_G.HSDT):
    """hsdt_c1g.HSDT with the bundle's switches on C1 steps only (module docstring)."""

    def configure(self, **kw):
        super().configure(**kw)
        self._gate_cfg = {k: bool(self.sw.get(k)) for k in GATE}   # the configured values (prepare gates them per step)

    def prepare(self, batches, states, bitmasks=None):
        out = super().prepare(batches, states, bitmasks)
        c = getattr(self, "cur", None)
        cfg = getattr(self, "_gate_cfg", None)
        if c is not None and cfg:
            nt = getattr(c, "NT", None)
            small = c.S_pad <= GSMAX and (nt is None or nt == c.S_pad)
            for k, v in cfg.items():
                self.sw[k] = v and small
        return out


def __getattr__(name):
    return getattr(_G, name)
