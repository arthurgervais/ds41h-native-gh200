"""Golden-trace recorder of D13's host side (engine/d13.py) and HSD's host prepare (kernels/head/hsd.py HSD._build)
(C1 tail session, 14 Nov 2026; fpr/core/GTRACE.md, stage 2; C++: fpr/core/src/d13, replays fpr/core/tests/replay_d13.cpp
and replay_hsd.cpp).

    install(d13)        wraps, on that D13 instance only: _async_ok, _owner_sig, _deltas, _positions, _fallback
    install_hsd(hsd)    wraps that HSD instance's _build
Called by d13.py (D13.__init__) when $PHYS_GTRACE is set (default off), or by the CPU harness
(fpr/core/tests/record_d13_cpu.py). Every wrapper passes arguments and return values through unchanged and never raises
into the engine (a recorder error is printed once and recording stops). Writes through fpr/core/gtrace.py (one global
seq) to $PHYS_GTRACE/d13.jsonl and $PHYS_GTRACE/hsd.jsonl.

d13.jsonl calls:
  config    D13's constants (Config in include/d13.h)
  dispatch  in {b: [StepBatch x 4], ct: [CompactTables x 4], graphs}  out {why}            (_async_ok)
  pre       in {r, prev: [[rid, has, rank, index]], sidx}             out {empty, sig, nb, desc}
            (_owner_sig's call in _pre; empty = _empty_eligible, nb = _guard_host's block counts, desc =
             RowExp.desc_of_compact of the same ct: the descriptor _pre uploads)
  deltas    in {r}                                                     out {any, nu, direct, rows, cols, pages, mirror}
  positions in {pos: [[...] x 4], own: [[...] x 4]}                    out {pos, ws} (the StepBatches' new lists)
  fallback  out {rb: [{seqs: [[start, num_computed, n_blocks]], pos, pg, ws}], tabs: [RankTables], acc: [[...]],
                 accepted: [[[rid, n]]]}
  The ct / b of deltas, pre, positions and fallback are those of the latest dispatch line.
hsd.jsonl calls:
  init      the blob layout, HSD's config, both host blob sets (hex) and the per-slot temperature / seed / state ids
  build     in {b: [per rank: n_pos, samp, ss (seq_slot at the sample rows), ps (positions at the sample rows), tail,
                pt / wt (positions / window slots at the tail rows), seqs: [[rid, kind, state_slot, replay_start, window
                items or null]], states: {seq index: [temperature, seed, drafts_valid, na, id]}], bm: null | [...]}
            out {ret: null | [hi, S_pad, D_pad, NT, T, has_bm], cur: {...}, diff: [per rank [[byte offset, hex], ...]],
                 bm: h64 of _bm_h[:T] | null, err: null | "Type: message"}
Privacy: token ids (CompactTables tokens / targets, RankTables tokens / targets) are masked to their sign unless
PHYS_GTRACE_TOKENS=1 (decisions read the sign only); no token text, no sampled ids. Still: test streams only.
Encodings: rids as str(rid) (CP shadow tuples -> 'cp|rid|rank', as gtrace_sched.enc_rid); block tables incremental per
(rank, rid, uid, inc): ["=", pages] (full) or ["+", base, pages appended]; window tables as [[block, page], ...] in dict
order; si of RankTables and the bitmask staging as h64 (see h64()). Recorded sizes: C1 ~1-2 KB per step (d13 + hsd), a
mixed 8K-row step ~1-2 MB (the rows' lists): record mixed-heavy streams for ~1-2 minutes.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import threading
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)
from gtrace import tr  # noqa: E402

HP = 0x100000001B3
_ERR = {"n": 0}
_LOCK = threading.Lock()


def h64(arr) -> int:
    """Hash of an int32 array (si, bitmasks): sum(uint32(v[i]) * HP ** i) mod 2^64, xor n. Vectorized (numpy uint64
    arithmetic wraps); the replays compute the same loop in C++."""
    import numpy as np
    v = np.ascontiguousarray(arr).reshape(-1).astype(np.int32, copy=False).view(np.uint32).astype(np.uint64)
    n = int(v.shape[0])
    if not n:
        return 0
    pw = np.full(n, HP, dtype=np.uint64)
    pw[0] = 1
    pw = np.cumprod(pw, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return int((v * pw).sum(dtype=np.uint64)) ^ n


def enc_rid(x) -> str:
    if isinstance(x, tuple) and len(x) == 3 and x[0] == "cp":
        return f"cp|{enc_rid(x[1])}|{x[2]}"
    return x if isinstance(x, str) else str(x)


def _ints(x) -> list:
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return [int(v) for v in x]


TOKENS = os.environ.get("PHYS_GTRACE_TOKENS", "0") == "1"   # default: token ids masked to their sign (t >= 0 -> 0).
# Every D13 decision reads only the sign (owner signature: all device rows; the fallback tables copy tokens and derive
# targets from them), so masked traces replay identically; prompt / output ids are customer data on served streams.


def _tok(x) -> list:
    v = _ints(x)
    return v if TOKENS else [t if t < 0 else 0 for t in v]


def _fail(where: str) -> None:
    _ERR["n"] += 1
    if _ERR["n"] <= 3:
        print(f"gtrace_d13: recorder error in {where} (recording continues; this line is lost):\n"
              f"{traceback.format_exc()[-1500:]}", flush=True)


class _BT:
    """Incremental block-table encoding per (rank, rid, uid, inc)."""

    def __init__(self):
        self.last = {}

    def enc(self, rank, se) -> list:
        bt = se.block_table
        key = (rank, enc_rid(se.rid), int(getattr(se, "uid", -1)), int(getattr(se, "inc", 0)))
        prev = self.last.get(key)
        cur = list(bt) if bt is not None else []
        self.last[key] = cur
        if prev is not None and len(cur) >= len(prev) and cur[:len(prev)] == prev:
            return ["+", len(prev), [int(p) for p in cur[len(prev):]]]
        return ["=", [int(p) for p in cur]]


def enc_seq(rank, se, bts: _BT) -> list:
    wt = se.window_table
    return [enc_rid(se.rid), se.kind, int(se.row0), int(se.n_rows), int(se.start), int(se.num_computed),
            int(se.cached_upto), int(se.replay_start), bool(se.last_chunk), int(se.n_blocks or 0), bool(se.score),
            bool(se.forced), int(se.uid), int(getattr(se, "known", -1)), int(getattr(se, "wver", -1)),
            int(getattr(se, "inc", 0)), int(se.state_slot), int(getattr(se, "tail_row0", 0)), bts.enc(rank, se),
            [[int(b), int(p)] for b, p in wt.items()] if wt is not None else []]


def enc_sb(sb, bts: _BT) -> dict:
    return dict(rank=int(sb.rank), step=int(sb.step), n_rows=int(sb.n_rows), pos=_ints(sb.positions),
                ss=_ints(sb.seq_slot), pg=_ints(sb.paged_slot), ws=_ints(sb.window_slot), tail=_ints(sb.tail_rows),
                samp=_ints(sb.sample_rows), ncopies=len(sb.copies or ()),
                seqs=[enc_seq(sb.rank, se, bts) for se in sb.seqs])


CT_FIELDS = ("slot", "row0", "n_rows", "kind", "start", "replay_start", "cached_upto", "uid", "last_chunk", "score",
             "forced", "win_slot", "win_blk", "win_page", "bt_upd_slot", "bt_upd_start", "bt_upd_len", "bt_upd_pages",
             "tokens", "tail_rows", "sample_idx", "targets")


def enc_ct(ct) -> dict:
    d = {f: (_tok if f in ("tokens", "targets") else _ints)(getattr(ct, f, None)) for f in CT_FIELDS}
    d["rank"], d["step"], d["k1"] = int(ct.rank), int(ct.step), int(ct.k1)
    d["rids"] = [enc_rid(x) for x in (ct.rids or [])]
    return d


def enc_desc(d) -> dict:
    return dict(S=int(d.S), rows=int(d.rows), slot=_ints(d.slot), row0=_ints(d.row0), nrows=_ints(d.nrows),
                kind=_ints(d.kind), start=_ints(d.start), replay=_ints(d.replay), cached=_ints(d.cached),
                ops=_ints(d.ops.reshape(-1)))


def enc_tables(t) -> dict:
    import numpy as np
    sq = t.seq
    si = np.ascontiguousarray(t.si.numpy().astype(np.int32, copy=False)).reshape(-1)
    return dict(n=int(t.n), positions=_ints(t.positions), tokens=_tok(t.tokens), own_slot=_ints(t.own_slot),
                si=h64(si), sl=_ints(t.sl), sample_rows=_ints(t.sample_rows),
                seq_rows=[[enc_rid(a), int(b), int(c), k, bool(e)] for a, b, c, k, e in t.seq_rows],
                seq_slot=_ints(t.seq_slot), paged_slot=_ints(t.paged_slot), tail_rows=_ints(t.tail_rows),
                sample_idx=_ints(t.sample_idx), targets=_tok(t.targets),
                seq={k: _ints(v) for k, v in sq.items()},
                dtypes={k: str(v.dtype).replace("torch.", "") for k, v in dict(sq, positions=t.positions, tokens=t.tokens,
                        own_slot=t.own_slot, si=t.si, sl=t.sl, seq_slot=t.seq_slot, paged_slot=t.paged_slot,
                        tail_rows=t.tail_rows, sample_idx=t.sample_idx, targets=t.targets).items()})


# ------------------------------------------------------------------------------------------ D13
class _D13Rec:
    def __init__(self, d13):
        self.d13 = d13
        self.bts = _BT()
        self.last_b = None
        self.last_ct = None
        self._cfg_done = False

    def config(self):
        d = self.d13
        eng = d.eng
        M = sys.modules[type(d).__module__]
        tr.write("d13", "config", {}, dict(
            graphs=bool(eng.cfg.graphs), largest_bucket=int(eng.cfg.buckets[-1]), graph_max_ctx=int(d.E.GRAPH_MAX_CTX),
            k1=int(M.K1), delta_cap=int(M.DELTA_CAP), fb_reuse=bool(M.FB_REUSE),
            scratch=[int(x.scratch) for x in d.rx], flags=dict(OWNER=bool(M.OWNER), EMPTY=bool(M.EMPTY), GT=bool(M.GT),
                                                              CHECK=bool(M.CHECK), ADV_SLICE=bool(M.ADV_SLICE))))
        self._cfg_done = True


def install(d13) -> None:
    """Wrap the D13 instance's host decisions (instance attributes shadow the class methods; nothing else changes)."""
    if not tr.on or getattr(d13, "_gtrace", None) is not None:
        return
    R = d13._gtrace = _D13Rec(d13)
    M = sys.modules[type(d13).__module__]
    o_async, o_sig, o_deltas, o_pos, o_fb = d13._async_ok, d13._owner_sig, d13._deltas, d13._positions, d13._fallback

    def _async_ok(batches, cts):
        why = o_async(batches, cts)
        try:
            with _LOCK:
                if not R._cfg_done:
                    R.config()
                R.last_b, R.last_ct = list(batches), list(cts)
                tr.write("d13", "dispatch", dict(b=[enc_sb(sb, R.bts) for sb in batches], ct=[enc_ct(c) for c in cts],
                                                 graphs=bool(d13.eng.cfg.graphs)), dict(why=why))
        except Exception:  # noqa: BLE001
            _fail("dispatch")
        return why

    def _owner_sig(r, sb, ct):
        sig = o_sig(r, sb, ct)
        try:
            eng = d13.eng
            prev = []
            for se in sb.seqs:
                p = eng.prev_of(se.rid)
                prev.append([enc_rid(se.rid), p is not None, int(p[0]) if p else -1, int(p[1]) if p else -1])
            el = type(d13)._empty_eligible(sb, ct)
            nb = [int(se.n_blocks or len(se.block_table)) for se in sb.seqs]
            desc = d13.RX.RowExp.desc_of_compact(ct)
            s_ = None if sig is None else [int(sig[0]), [[enc_rid(a), int(b), int(c), int(e), int(f), int(g)]
                                                          for a, b, c, e, f, g in sig[1]], [int(x) for x in sig[2]]]
            tr.write("d13", "pre", dict(r=int(r), prev=prev, sidx=_ints(ct.sample_idx)),
                     dict(empty=bool(el), sig=s_, nb=nb, desc=enc_desc(desc)))
        except Exception:  # noqa: BLE001
            _fail("pre")
        return sig

    def _deltas(r, ct):
        out = o_deltas(r, ct)
        try:
            import numpy as np
            rec = dict(any=False, nu=0, direct=False, rows=[], cols=[], pages=[], mirror=False)
            if ct.bt_upd_slot is not None and len(ct.bt_upd_slot):
                nu = int(np.asarray(ct.bt_upd_len, np.int64).sum())
                hx_ = d13.eng.hooks.idx
                rec.update(any=True, nu=nu, direct=nu > M.DELTA_CAP,
                           mirror=bool(hx_ is not None and getattr(hx_, "mir", None)
                                       and getattr(hx_, "mirror_deltas", None) is not None))
                if 0 < nu <= M.DELTA_CAP:   # the pinned staging _deltas just filled (read back: what Python computed)
                    hr, hc, hp = d13.dl_h[d13.dl_i][r]
                    rec.update(rows=hr[:nu].tolist(), cols=hc[:nu].tolist(), pages=hp[:nu].tolist())
                elif nu:
                    rec.update(rows=None, cols=None, pages=None)
            tr.write("d13", "deltas", dict(r=int(r)), rec)
        except Exception:  # noqa: BLE001
            _fail("deltas")
        return out

    def _positions(st, batches):
        out = o_pos(st, batches)
        try:
            pos, own = [], []
            for r in range(M.W):
                n = batches[r].n_rows
                ph, oh = d13.d2h[d13.d2h_i][r]
                pos.append(ph[:n].tolist())
                own.append(oh[:n].tolist())
            tr.write("d13", "positions", dict(pos=pos, own=own), dict(pos=[_ints(x.positions) for x in out],
                                                                       ws=[_ints(x.window_slot) for x in out]))
        except Exception:  # noqa: BLE001
            _fail("positions")
        return out

    def _fallback(batches, cts, ctx, why):
        eng = d13.eng
        cap = {}
        o_step = eng.step

        def step(rb, tables, ctx=None):
            cap["rb"], cap["tables"] = rb, tables
            return o_step(rb, tables, ctx=ctx)
        had = "step" in getattr(eng, "__dict__", {})
        eng.step = step
        try:
            h = o_fb(batches, cts, ctx, why)
        finally:
            if had:
                eng.step = o_step
            else:
                del eng.step                         # back to the class method
        try:
            rb, tables = cap.get("rb"), cap.get("tables")
            res = h._res
            acc, accepted = [], []
            for r in range(M.W):
                S = len(rb[r].seqs)
                acc.append(d13.acc_h[r].numpy()[:S].tolist() if S else [])
                accepted.append([[enc_rid(k), int(v)] for k, v in res[r].accepted.items()])
            tr.write("d13", "fallback", {}, dict(
                rb=[dict(seqs=[[int(se.start), int(se.num_computed), int(se.n_blocks or 0)] for se in sb.seqs],
                         pos=_ints(sb.positions), pg=_ints(sb.paged_slot), ws=_ints(sb.window_slot)) for sb in rb],
                tabs=[enc_tables(t) for t in tables], acc=acc, accepted=accepted))
        except Exception:  # noqa: BLE001
            _fail("fallback")
        return h

    d13._async_ok, d13._owner_sig, d13._deltas, d13._positions, d13._fallback = (_async_ok, _owner_sig, _deltas,
                                                                                 _positions, _fallback)


# ------------------------------------------------------------------------------------------ HSD
def _diff(old, new, gap: int = 16) -> list:
    """Byte runs where new != old: [[offset, hex], ...] (runs closer than `gap` merged)."""
    import numpy as np
    d = np.flatnonzero(old != new)
    if not len(d):
        return []
    out = []
    s = e = int(d[0])
    for x in d[1:].tolist():
        if x - e > gap:
            out.append([s, new[s:e + 1].tobytes().hex()])
            s = x
        e = x
    out.append([s, new[s:e + 1].tobytes().hex()])
    return out


def install_hsd(hsd) -> None:
    if not tr.on or getattr(hsd, "_gtrace", None) is not None or getattr(hsd, "prep", "1") == "0":
        return
    import numpy as np
    M = sys.modules[type(hsd).__module__]
    st_ = dict(shadow=None)
    hsd._gtrace = st_
    o_build = hsd._build

    def init():
        lay = [[nm, off, np.dtype(dt).name, dim, cols, hsd._dims[dim] * cols] for nm, (off, dt, dim, cols)
               in hsd._lay.items()]
        st_["shadow"] = [[hsd._hblob[i][r].numpy().copy() for r in range(M.W)] for i in range(2)]
        drk = hsd.ranks0[0].b["aux"].shape[0]
        tr.write("hsd", "init", {}, dict(
            layout=lay, nbytes=int(hsd._nbytes), shared_end=int(hsd._shared_end), hi=int(hsd._hi),
            cfg=dict(k=int(hsd.k), max_seqs=int(hsd.max_seqs), seq_buckets=[int(x) for x in hsd.seq_buckets],
                     row_buckets=[int(x) for x in hsd.row_buckets], nt_max=int(drk), pool=bool(hsd.pool),
                     window=int(M.DR.WINDOW), ring=int(M.DR.RING)),
            blobs=[[b.tobytes().hex() for b in s] for s in st_["shadow"]],
            temp=[float(x) for x in hsd._temp_np.tolist()], seeds=[int(x) for x in hsd._seed_np.tolist()],
            keys=[0 if x is None else id(x) for x in hsd._slot_key]))

    def enc_b(b, states):
        srl = list(b.sample_rows)
        n = len(srl)
        pos, ss = b.positions, b.seq_slot
        g = (lambda lst, idx: [int(lst[i]) for i in idx])
        drafting = sorted({int(ss[i]) for i in srl}) if n else []
        seqs = []
        sts = {}
        for k, se in enumerate(b.seqs):
            dr = k in drafting
            seqs.append([enc_rid(se.rid), se.kind, int(se.state_slot), int(se.replay_start),
                         [[int(x), int(y)] for x, y in se.window_table.items()] if dr else None])
            if dr:
                s = states.get(se.rid)
                na = getattr(s, "num_acceptable_drafts", None)
                sts[str(k)] = [float(getattr(s, "temperature", 1.0)), int(getattr(s, "seed", 0)),
                               bool(getattr(s, "drafts_valid", True)), None if na is None else int(na), id(s)]
        tl = list(b.tail_rows)
        return dict(n_pos=len(pos), samp=[int(x) for x in srl], ss=g(ss, srl), ps=g(pos, srl), tail=[int(x) for x in tl],
                    pt=g(pos, tl), wt=g(b.window_slot, tl) if hsd.pool else [], seqs=seqs, states=sts,
                    sst=[int(x) for x in ss] if not hsd.pool else [])

    def _build(batches, states, bitmasks):
        if st_["shadow"] is None:
            try:
                init()
            except Exception:  # noqa: BLE001
                _fail("hsd init")
        err = None
        try:
            hb = o_build(batches, states, bitmasks)
        except Exception as e:  # noqa: BLE001
            err, hb = f"{type(e).__name__}: {e}", None
            exc = e
        try:
            inp = dict(b=[enc_b(b, states) for b in batches], bm=None)
            if bitmasks is not None and any(x is not None for x in bitmasks):
                inp["bm"] = [None if x is None else [int(x.shape[0]), int(x.shape[1]),
                                                     np.ascontiguousarray(x.numpy() if hasattr(x, "numpy") else x,
                                                                          dtype=np.int32).tobytes().hex()]
                             for x in bitmasks]
            out = dict(ret=None, cur=None, diff=None, bm=None, err=err)
            if hb is not None:
                hi, key, T, has_bm = hb
                c = hsd.cur
                out["ret"] = [int(hi), int(key[0]), int(key[1]), int(key[2]), int(T), bool(has_bm)]
                out["cur"] = dict(n_s=[int(x) for x in c.n_s], S_pad=int(c.S_pad), D_pad=int(c.D_pad),
                                  n_req=int(c.n_req), T=int(c.T), NT=int(c.NT),
                                  dseqs=[[[int(a), int(b)] for a, b in x] for x in c.dseqs],
                                  rids=[None if x is None else enc_rid(x) for x in c.rids],
                                  n_seq=[int(x) for x in c.n_seq], n_t=[int(x) for x in c.n_t],
                                  step=[int(x) for x in c.step])
                diffs = []
                for r in range(M.W):
                    new = hsd._hblob[hi][r].numpy()
                    old = st_["shadow"][hi][r]
                    diffs.append(_diff(old, new))
                    st_["shadow"][hi][r] = new.copy()
                out["diff"] = diffs
                if has_bm:
                    out["bm"] = h64(hsd._bm_h.numpy()[:T])
            tr.write("hsd", "build", inp, out)
        except Exception:  # noqa: BLE001
            _fail("hsd build")
        if err is not None:
            raise exc
        return hb

    hsd._build = _build
