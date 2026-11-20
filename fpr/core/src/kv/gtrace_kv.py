"""Golden-trace recorder of serve/kv_manager.py OwnedKVManager (c1-sync, 14 Nov 2026; fpr/core/GTRACE.md, stage 2).

install() -- called once before the serve modules are used (the corpus runner, or a default-off PHYS_GTRACE hook) --
wraps every public OwnedKVManager method at the CLASS level. The wrappers pass arguments and return values through
unchanged; only the OUTERMOST kv call of a thread is recorded (admit's own lookup / free_pages calls are not).

One JSONL line per call through fpr/core/gtrace.py (shared seq) to $PHYS_GTRACE/kv.jsonl:
  {"seq", "comp": "kv", "call": <method> | "__init__" | "_heads", "obj": k (manager instance), "in": {...},
   "out": <return value>, "post": {key: view | null}   the touched sequences after the call (null = gone),
   "via": "tables"   (only on calls made from inside a recorded tables call: build_step_batch's take_poison)}
Forms (the C++ replay reads the same):
  key      str -> str; int -> {"i": n}; tuple -> {"t": [...]} (CP shadows ("cp", rid, rank) -> fpr::shadow_key)
  hash     interned per process: 1, 2, ... in order of first appearance (Python == classes; collision-free)
  CopyOp   [pool name, src_rank, src_page, dst_rank, dst_page, step]
  view     {"rank", "pages": list form, "nreg", "window": {"W": id, "items": [[blk, page], ...]}, "slot" (-1 = None),
            "reserved", "cached", "rs", "wver", "shadow", "inc"}   (cp_have is the scheduler's field: not recorded)
  list     page lists by identity: {"L": id, "full": [...]} first sight, {"L": id, "n0": k, "add": [...]} after appends,
           {"L": id} unchanged (Python only appends to a page list; migrate makes a new one)
  "_heads" every 256 recorded calls per manager and after every check(): per pool and rank the free list's first 8 in
           pop order ("free"; TierFree: HBM heap "free" + host FIFO "host"), the LRU's first 8, and the counts.
"""
from __future__ import annotations

import functools
import heapq
import inspect
import itertools
import os
import sys
import threading
import weakref

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)
from gtrace import tr  # noqa: E402

COMP = "kv"
HEADS_EVERY = int(os.environ.get("PHYS_GTRACE_SNAP", "256") or 256)
LOCK = threading.RLock()
_TL = threading.local()


# ---------------------------------------------------------------------------------------------- value forms
class Store:
    """Per-component identity store (each component's trace replays standalone)."""

    def __init__(self):
        self.hashes = {}
        self.lists = {}           # id(list) -> [list, lid, recorded length]   (strong refs: no id reuse)
        self.dicts = {}           # id(dict) -> [dict, wid]
        self.n = 0

    def H(self, h) -> int:
        v = self.hashes.get(h)
        if v is None:
            v = self.hashes[h] = len(self.hashes) + 1
        return v

    def L(self, lst) -> dict:
        e = self.lists.get(id(lst))
        n = len(lst)
        if e is None or e[0] is not lst or n < e[2]:
            self.n += 1
            self.lists[id(lst)] = [lst, self.n, n]
            return {"L": self.n, "full": [int(x) for x in lst]}
        if n > e[2]:
            n0, e[2] = e[2], n
            return {"L": e[1], "n0": n0, "add": [int(x) for x in lst[n0:]]}
        return {"L": e[1]}

    def W(self, d) -> dict:
        e = self.dicts.get(id(d))
        if e is None or e[0] is not d:
            self.n += 1
            e = self.dicts[id(d)] = [d, self.n]
        return {"W": e[1], "items": [[int(b), int(p)] for b, p in d.items()]}


def K(x):
    if isinstance(x, str):
        return x
    if isinstance(x, bool) or not isinstance(x, (int, tuple)):
        return {"r": repr(x)}
    if isinstance(x, int):
        return {"i": x}
    return {"t": [K(e) for e in x]}


def rle(a) -> list:
    """Runs of +1 or constant steps: [v, n, d, v, n, d, ...] (n values v, v + d, ...)."""
    out = []
    it = iter(a)
    try:
        v0 = prev = int(next(it))
    except StopIteration:
        return out
    n, d = 1, None
    for x in it:
        x = int(x)
        if d is None and x - prev in (0, 1):
            d, n = x - prev, n + 1
        elif d is not None and x - prev == d:
            n += 1
        else:
            out += [v0, n, d or 0]
            v0, n, d = x, 1, None
        prev = x
    out += [v0, n, d or 0]
    return out


def copy_op(c) -> list:
    return [c.pool, int(c.src_rank), int(c.src_page), int(c.dst_rank), int(c.dst_page), int(c.step)]


class nest:
    """Per-thread nesting depth of a component (only the outermost call is recorded)."""

    def __init__(self, comp: str):
        self.comp = comp

    def __enter__(self):
        d = getattr(_TL, "d", None)
        if d is None:
            d = _TL.d = {}
        d[self.comp] = d.get(self.comp, 0) + 1
        return d[self.comp]

    def __exit__(self, *a):
        _TL.d[self.comp] -= 1
        return False


def active(comp: str) -> bool:
    d = getattr(_TL, "d", None)
    return bool(d and d.get(comp, 0))


# ---------------------------------------------------------------------------------------------- the kv recorder
ST = Store()
_OBJ: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_NOBJ = itertools.count(1)
_NCALL: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_PLEN: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()   # kv -> poison list lengths after its last call


def _plens(kv) -> tuple:
    return tuple(len(p.poison) for pools in (kv.paged, kv.window, kv.state) for p in pools)


def _ext(kv) -> None:
    """State changed OUTSIDE the kv calls since the last recorded one (tests re-queue poison entries by hand:
    kv.<pool>[r].poison.append): one "_ext" line with the new poison lists, which the replay installs as they are."""
    prev = _PLEN.get(kv)
    if prev is not None and prev != _plens(kv):
        tr.write(COMP, "_ext", {}, {"poison": {name: [[int(x) for x in p.poison] for p in pools] for name, pools in
                                               (("paged", kv.paged), ("window", kv.window), ("state", kv.state))}},
                 obj=obj_id(kv))


def real(kv):
    """The OwnedKVManager behind a recording proxy (c1-mega's gtrace_sched._KVProxy keeps it in _gt_real)."""
    for _ in range(8):
        r = getattr(kv, "_gt_real", None)
        if r is None:
            return kv
        kv = r
    return kv


def obj_id(kv) -> int:
    kv = real(kv)
    with LOCK:
        k = _OBJ.get(kv)
        if k is None:
            k = _OBJ[kv] = next(_NOBJ)
        return k


def view(kv, key):
    sk = kv.seqs.get(key)
    if sk is None:
        return None
    return {"rank": sk.rank, "pages": ST.L(sk.pages), "nreg": sk.num_registered, "window": ST.W(sk.window),
            "slot": -1 if sk.state_slot is None else int(sk.state_slot), "reserved": sk.reserved,
            "cached": sk.cached_tokens, "rs": sk.replay_start, "wver": sk.wver, "shadow": bool(sk.shadow), "inc": sk.inc}


def heads(kv) -> dict:
    TierFree = sys.modules[type(kv).__module__].TierFree
    out = {}
    for name, pools in (("paged", kv.paged), ("window", kv.window), ("state", kv.state)):
        rows = []
        for pool in pools:
            fu = pool.free_uncached
            h = {}
            if isinstance(fu, TierFree):
                h["free"] = heapq.nsmallest(8, fu.hbm)
                h["host"] = list(itertools.islice(fu.host, 8))
            elif pool.low_first:
                h["free"] = heapq.nsmallest(8, fu)
            else:
                h["free"] = list(itertools.islice(fu, 8))
            h.update(lru=list(itertools.islice(pool.lru, 8)), nfree=len(fu), nlru=len(pool.lru), ncached=len(pool.cached),
                     reserved=pool.reserved, demote=len(pool.demote_q), npoison=len(pool.poison))
            rows.append(h)
        out[name] = rows
    return out


def geom_in(kv) -> dict:
    g = kv.geom
    pools = [[p.spec.name, p.spec.kind, p.spec.tokens_per_page, p.spec.bytes_per_page, p.spec.num_pages]
             for p in (kv.paged[0], kv.window[0], kv.state[0])]
    return {"n_ranks": g.n_ranks, "page_tokens": g.page_tokens, "window": g.window,
            "window_page_tokens": g.window_page_tokens, "replay_tokens": g.replay_tokens,
            "reserve_watermark": g.reserve_watermark, "pools": pools, "hbm_pages": kv.paged[0].n_hbm,
            "poison": {"paged": kv.paged[0].poison_on, "window": kv.window[0].poison_on, "state": kv.state[0].poison_on},
            "geom_class": type(g).__name__}


def _admit_out(r):
    if r is None:
        return None
    return {"rank": r.rank, "hit_blocks": r.hit_blocks, "cached_tokens": r.cached_tokens, "replay_start": r.replay_start,
            "copied_blocks": r.copied_blocks, "copies": [copy_op(c) for c in r.copies]}


# method -> (input serializer(args dict), output serializer, touched key arg or None)
def _ids(a):
    return {"id": K(a["seq_id"])}


SPEC = {
    "lookup": (lambda a: {"h": [ST.H(h) for h in a["block_hashes"]]}, list, None),
    "free_pages": (lambda a: {"rank": a["rank"]}, int, None),
    "usage": (lambda a: {"rank": a["rank"]}, dict, None),
    "block_table": (_ids, lambda r: ST.L(r), None),
    "window_table": (_ids, lambda r: ST.W(r), None),
    "pages_for": (lambda a: {"tokens": a["tokens"]}, int, None),
    "hit_after_cap": (lambda a: {"hit_blocks": a["hit_blocks"], "prompt_len": a["prompt_len"]}, list, None),
    "admit": (lambda a: {"id": K(a["seq_id"]), "rank": a["rank"], "h": [ST.H(h) for h in a["block_hashes"]],
                         "prompt_len": a["prompt_len"], "step": a["step"], "copy_from": a["copy_from"]}, _admit_out, "seq_id"),
    "can_allocate": (lambda a: {"id": K(a["seq_id"]), "start": a["start"], "end": a["end"], "window_end": a["window_end"]},
                     bool, None),
    "allocate": (lambda a: {"id": K(a["seq_id"]), "start": a["start"], "end": a["end"], "window_end": a["window_end"]},
                 bool, "seq_id"),
    "commit": (None, None, "seq_id"),                       # special-cased: the block_hash callable
    "shrink_reservation": (_ids, None, "seq_id"),
    "open_shadow": (lambda a: {"id": K(a["key"]), "rank": a["rank"], "replay_start": a["replay_start"]}, bool, "key"),
    "trim_window": (lambda a: {"id": K(a["seq_id"]), "lo_pos": a["lo_pos"]}, None, "seq_id"),
    "free": (lambda a: {"id": K(a["seq_id"]), "defer_until": a["defer_until"]}, None, "seq_id"),
    "flush_deferred": (lambda a: {"upto_step": a["upto_step"]}, None, None),
    "migrate": (lambda a: {"id": K(a["seq_id"]), "dst": a["dst"], "step": a["step"], "num_computed": a["num_computed"]},
                lambda r: None if r is None else [copy_op(c) for c in r], "seq_id"),
    "tier_copies": (lambda a: {"step": a["step"], "max_pages": a["max_pages"]}, lambda r: [copy_op(c) for c in r], None),
    "release_copies": (lambda a: {"step": a["step"]}, None, None),
    "take_poison": (lambda a: {"rank": a["rank"]}, lambda r: [[p, int(q)] for p, q in r], None),
    "check": (lambda a: {}, None, None),
}


def _stats(kv) -> dict:
    return {k: kv.stats.get(k, 0) for k in ("copies", "copy_bytes", "evictions", "promoted", "demoted")}


def _wrap(name, orig):
    sig = inspect.signature(orig)
    fin, fout, touched = SPEC[name]

    @functools.wraps(orig)
    def w(self, *a, **k):
        if not tr.on or active(COMP):
            return orig(self, *a, **k)
        with LOCK, nest(COMP):
            _ext(self)
            ba = sig.bind(self, *a, **k)
            ba.apply_defaults()
            args = ba.arguments
            asked = None
            if name == "commit":
                fn = args["block_hash"]
                asked = []

                def rec_fn(i, _fn=fn):
                    h = _fn(i)
                    asked.append([i, ST.H(h)])
                    return h
                args["block_hash"] = rec_fn
                inp = None
            else:
                inp = fin(args)
            try:
                ret = orig(*ba.args, **ba.kwargs) if name != "commit" else orig(self, args["seq_id"], args["num_computed"],
                                                                                    args["block_hash"])
            except BaseException as e:   # noqa: BLE001 -- recorded, then re-raised unchanged
                if name == "commit":
                    inp = {"id": K(args["seq_id"]), "num_computed": args["num_computed"], "asked": asked}
                tr.write(COMP, name, inp, None, obj=obj_id(self), exc=f"{type(e).__name__}: {e}",
                         **({"via": "tables"} if active("tables") else {}))
                _PLEN[self] = _plens(self)
                raise
            if name == "commit":
                inp = {"id": K(args["seq_id"]), "num_computed": args["num_computed"], "asked": asked}
            out = None if fout is None else fout(ret)
            extra = {}
            if touched is not None:
                key = args[touched]
                extra["post"] = {"k": K(key), "v": view(self, key)}
            if name in ("admit", "allocate", "migrate", "tier_copies"):
                extra["stats"] = _stats(self)
            if active("tables"):
                extra["via"] = "tables"
            tr.write(COMP, name, inp, out, obj=obj_id(self), **extra)
            _PLEN[self] = _plens(self)
            n = _NCALL.get(self, 0) + 1
            _NCALL[self] = n
            if name == "check" or n % HEADS_EVERY == 0:
                tr.write(COMP, "_heads", {}, heads(self), obj=obj_id(self), **({"via": "tables"} if active("tables") else {}))
            return ret
    w.__gtrace__ = True
    return w


def install_kv(cls) -> bool:
    """Wrap the class's public methods (once): serve/kv_manager.py's own hook passes OwnedKVManager at the end of its
    import; the corpus runner calls install()."""
    if not tr.on:
        return False
    if getattr(cls, "__cxx__", False):   # PHYS_CXX_KV: the C++ manager (its own replay is the check; nothing to record)
        print("gtrace_kv: kv recorder not installed on the C++ KV manager (PHYS_CXX_KV)", flush=True)
        return False
    if getattr(cls, "__gtrace_kv__", False):
        return True
    for name in SPEC:
        setattr(cls, name, _wrap(name, getattr(cls, name)))
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def init(self, *a, **k):
        orig_init(self, *a, **k)
        if tr.on:
            with LOCK:
                tr.write(COMP, "__init__", geom_in(self), None, obj=obj_id(self))
                _PLEN[self] = _plens(self)
    cls.__init__ = init
    cls.__gtrace_kv__ = True
    return True


def install() -> bool:
    """False when PHYS_GTRACE is off (nothing changes)."""
    if not tr.on:
        return False
    import kv_manager as KM
    return install_kv(KM.OwnedKVManager)
