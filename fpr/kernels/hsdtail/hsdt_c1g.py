"""hsdt_c1g: hsdt_c1f + the c1d2 kinds built with C1M_INS_OLD (s4o / s4bo / s4to / s4po) (C1 tail session, 13 Nov 2026;
TEST, bench only: HSDT_BENCH_MOD=hsdt_c1g). hsdt_c1f.py is frozen (the uwcm9xh candidate), so its kind list is extended
here; everything else is hsdt_c1f's (module attributes forwarded)."""
from __future__ import annotations

import hsdt_c1f as _F

_F._C1D2_KINDS = tuple(_F._C1D2_KINDS) + tuple(k for k in ("s4o", "s4bo", "s4to", "s4po") if k not in _F._C1D2_KINDS)
HSDT = _F.HSDT
HAS_CTX3 = _F.HAS_CTX3
HAS_NSPLIT = _F.HAS_NSPLIT


def parse_switches(spec):
    """hsdt_c1f.parse_switches + "c1d=<kind>" (engine side: ENG_HSDT=...,c1d=s4po,... with ENG_HSDT_MOD=hsdt_c1g): the
    c1d2 kind as a string (hsdt.parse_switches turns any name=value into a bool); plain "c1d" stays True (the live c1d)."""
    items = [x.strip() for x in spec.replace("+", ",").split(",")]
    kind = None
    rest = []
    for x in items:
        if x.startswith("c1d="):
            kind = x.split("=", 1)[1]
        elif x:
            rest.append(x)
    out = _F.parse_switches(",".join(rest))
    if kind is not None:
        assert kind in _F._C1D2_KINDS, f"ENG_HSDT c1d={kind}: not a c1d2 kind {_F._C1D2_KINDS}"
        out["c1d"] = kind
    return out


def __getattr__(name):
    return getattr(_F, name)
