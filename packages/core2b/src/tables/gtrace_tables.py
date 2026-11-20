"""Golden-trace recorder of the worker's tables (c1-sync, 14 Nov 2026; fpr/core/GTRACE.md, stage 2):
interfaces.build_step_batch (StepPlan + KV -> StepBatch), worker.TableBuilder.build (StepBatch -> RankTables),
worker.build_compact (StepBatch -> CompactTables) and Worker.copies' grouping (FPR_PAGECOPY groups + copy_runs).

install() wraps them (module functions: in `interfaces` / `worker` AND in every already-imported module that bound
them by name; TableBuilder.build / Worker.copies at the class). Wrappers pass everything through unchanged.

One JSONL line per call through fpr/core/gtrace.py to $PHYS_GTRACE/tables.jsonl:
  {"seq", "comp": "tables", "call": "build_step_batch" | "build" | "build_compact" | "copies", "obj": k, "in", "out"}
  obj: build_step_batch -> the KV manager's id (as in kv.jsonl); build / build_compact -> the TableBuilder's id
  (its `sent` / `cstate` caches persist per instance); copies -> the Worker's id.
Forms: as gtrace_kv (keys, lists by identity, rle arrays), with this component's own identity store, so tables.jsonl
replays standalone. The KV views build_step_batch read are in "views" (key -> {pages, window, slot, wver, inc}).
build / build_compact without row_tokens: the TableBuilder.tokens_of calls made are recorded in order in "tok"
([[key, pos, token], ...]; the default tokens_of is Python's per-process string hash: replay feeds the values).
Token ids are masked to their sign unless PHYS_GTRACE_TOKENS=raw (see tok()).
Worker.copies: the grouping is inline Python (not a function); "out" is computed here by the same lines (verbatim) on
the same deduplicated ops, plus interfaces.copy_runs on them.
"""
from __future__ import annotations

import functools
import itertools
import os
import sys
import weakref

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_CORE, os.path.join(_CORE, "src", "kv")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from gtrace import tr  # noqa: E402
import gtrace_kv as GK  # noqa: E402

COMP = "tables"
ST = GK.Store()
# Token ids are request content: PHYS_GTRACE_TOKENS=sign (default) records each token id as its sign (-1 / 0 / 1); =raw
# keeps them (CPU corpus only). The table builders only copy token values or insert -1, so masking commutes with them:
# the replays compare the masked outputs against C++ fed the masked inputs, exactly.
RAW_TOKENS = os.environ.get("PHYS_GTRACE_TOKENS", "sign") == "raw"


def tok(v) -> int:
    v = int(v)
    return v if RAW_TOKENS else (v > 0) - (v < 0)


def toks(a) -> list:
    return [tok(x) for x in a]
K, rle, copy_op = GK.K, GK.rle, GK.copy_op
_OBJ: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_NOBJ = itertools.count(1)


def obj_id(o) -> int:
    with GK.LOCK:
        k = _OBJ.get(o)
        if k is None:
            k = _OBJ[o] = next(_NOBJ)
        return k


def _cp(cp):
    if cp is None:
        return None
    return {"group": int(cp.group), "rid": K(cp.rid), "owner": int(cp.owner),
            "subs": [[int(r), int(s), int(n)] for r, s, n in cp.subs], "me": int(cp.me)}


def _images(ims):
    return [[int(o), int(n), K(k)] for o, n, k in (ims or ())]


def work_json(w) -> dict:
    return {"rid": K(w.rid), "rank": w.rank, "kind": w.kind, "start": w.start, "n": w.n,
            "cached_upto": getattr(w, "cached_upto", 0), "replay_start": getattr(w, "replay_start", 0),
            "last_chunk": bool(getattr(w, "last_chunk", False)), "score": bool(getattr(w, "score", False)),
            "forced": bool(getattr(w, "forced", False)), "uid": int(getattr(w, "uid", -1)),
            "known": int(getattr(w, "known", -1)), "cp": _cp(getattr(w, "cp", None)),
            "images": _images(getattr(w, "images", ()))}


def se_json(se) -> dict:
    return {"rid": K(se.rid), "kind": se.kind, "row0": se.row0, "n_rows": se.n_rows, "start": se.start,
            "num_computed": se.num_computed, "cached_upto": se.cached_upto, "replay_start": se.replay_start,
            "last_chunk": bool(se.last_chunk), "bt": ST.L(se.block_table), "wt": ST.W(se.window_table),
            "slot": -1 if se.state_slot is None else int(se.state_slot), "tail_row0": se.tail_row0,
            "n_blocks": se.n_blocks, "score": bool(se.score), "forced": bool(se.forced), "uid": se.uid,
            "known": se.known, "wver": se.wver, "cp": _cp(se.cp), "images": _images(se.images), "inc": se.inc}


def sb_json(sb) -> dict:
    return {"rank": sb.rank, "step": sb.step, "n_rows": sb.n_rows, "positions": rle(sb.positions),
            "seq_slot": rle(sb.seq_slot), "paged_slot": rle(sb.paged_slot), "window_slot": rle(sb.window_slot),
            "tail_rows": rle(sb.tail_rows), "sample_rows": rle(sb.sample_rows), "seqs": [se_json(se) for se in sb.seqs],
            "copies": [copy_op(c) for c in sb.copies], "poison": [[p, int(q)] for p, q in getattr(sb, "poison", [])]}


def _flat(t) -> list:
    return t.reshape(-1).tolist() if hasattr(t, "reshape") else list(t)


def rt_json(t) -> dict:
    seq = {k: {"shape": list(v.shape), "v": rle(_flat(v))} for k, v in t.seq.items()}
    return {"n": t.n, "positions": rle(_flat(t.positions)), "tokens": rle(toks(_flat(t.tokens))), "own_slot": rle(_flat(t.own_slot)),
            "si": {"shape": list(t.si.shape), "v": rle(_flat(t.si))}, "sl": rle(_flat(t.sl)),
            "ti": {str(s): {"shape": list(v.shape), "v": rle(_flat(v))} for s, v in t.ti.items()},
            "tl": {str(s): rle(_flat(v)) for s, v in t.tl.items()}, "sample_rows": rle(t.sample_rows),
            "seq_rows": [[K(r), int(a), int(b), k, bool(c)] for r, a, b, k, c in t.seq_rows],
            "seq_slot": rle(_flat(t.seq_slot)), "paged_slot": rle(_flat(t.paged_slot)), "tail_rows": rle(_flat(t.tail_rows)),
            "sample_idx": rle(_flat(t.sample_idx)), "seq": seq, "targets": rle(toks(_flat(t.targets)))}


CT_FIELDS = ("slot", "row0", "n_rows", "kind", "start", "replay_start", "cached_upto", "win_slot", "win_blk", "win_page",
             "bt_upd_slot", "bt_upd_start", "bt_upd_len", "bt_upd_pages", "tokens", "tail_rows", "sample_idx", "targets",
             "uid", "last_chunk", "score", "forced")


def ct_json(ct) -> dict:
    out = {"rank": ct.rank, "step": ct.step, "k1": ct.k1, "n": ct.n, "rids": [K(r) for r in ct.rids]}
    for f in CT_FIELDS:
        v = _flat(getattr(ct, f))
        out[f] = rle(toks(v) if f in ("tokens", "targets") else v)
    return out


def _tok_recorder(tb):
    calls = []
    fn = tb.tokens_of

    def rec(rid, pos, _fn=fn):
        v = _fn(rid, pos)
        calls.append([K(rid), int(pos), tok(v)])
        return v
    return calls, fn, rec


def _rebind(mod_names: tuple, attr: str, orig, new) -> None:
    for name, m in list(sys.modules.items()):
        try:
            if m is not None and getattr(m, attr, None) is orig:
                setattr(m, attr, new)
        except Exception:  # noqa: BLE001 -- a module with an odd __getattr__: leave it alone
            pass


def install_interfaces(interfaces) -> bool:
    """Wrap interfaces.build_step_batch (serve/interfaces.py's own hook passes its module at the end of its import, so
    every later `from interfaces import build_step_batch` binds the wrapper; modules that bound it earlier are rebound)."""
    if not tr.on or getattr(interfaces, "__gtrace_tables__", False):
        return tr.on
    o_bsb = interfaces.build_step_batch

    @functools.wraps(o_bsb)
    def build_step_batch(plan, kv, geom, rank):
        if GK.active(COMP):
            return o_bsb(plan, kv, geom, rank)
        with GK.LOCK, GK.nest(COMP):
            views = {}
            for w in plan.work:
                if w.rank != rank:
                    continue
                sk = kv.seqs.get(w.rid)
                if sk is not None:
                    views[repr(K(w.rid))] = {"k": K(w.rid), "pages": ST.L(sk.pages), "window": ST.W(sk.window),
                                             "slot": -1 if sk.state_slot is None else int(sk.state_slot),
                                             "wver": sk.wver, "inc": sk.inc}
            inp = {"rank": rank, "geom": {"page_tokens": geom.page_tokens, "window_page_tokens": geom.window_page_tokens},
                   "plan": {"step": plan.step, "work": [work_json(w) for w in plan.work],
                            "copies": [copy_op(c) for c in plan.copies]},
                   "views": list(views.values())}
            sb = o_bsb(plan, kv, geom, rank)
            tr.write(COMP, "build_step_batch", inp, sb_json(sb), obj=GK.obj_id(kv))
            return sb

    interfaces.build_step_batch = build_step_batch
    _rebind((), "build_step_batch", o_bsb, build_step_batch)
    interfaces.__gtrace_tables__ = True
    return True


def install_worker(worker) -> bool:
    """Wrap worker.TableBuilder.build, worker.build_compact and Worker.copies (serve/worker.py's own hook passes its
    module at the end of its import)."""
    if not tr.on or getattr(worker, "__gtrace_tables__", False):
        return tr.on
    import interfaces as _if   # copy_runs (the non-PAGECOPY grouping)

    o_build = worker.TableBuilder.build

    @functools.wraps(o_build)
    def build(self, sb, row_tokens=None):
        if GK.active(COMP):
            return o_build(self, sb, row_tokens)
        with GK.LOCK, GK.nest(COMP):
            inp = {"cfg": {"sources": list(self.sources), "stub_indexer": bool(self.stub_indexer),
                           "full_block_tables": bool(self.full_block_tables)},
                   "sb": sb_json(sb), "row_tokens": None if row_tokens is None else rle(toks(row_tokens))}
            calls, fn, rec = _tok_recorder(self)
            self.tokens_of = rec
            try:
                t = o_build(self, sb, row_tokens)
            finally:
                self.tokens_of = fn
            inp["tok"] = calls
            tr.write(COMP, "build", inp, rt_json(t), obj=obj_id(self))
            return t

    o_bc = worker.build_compact

    @functools.wraps(o_bc)
    def build_compact(tb, sb, k1, row_tokens=None):
        if GK.active(COMP):
            return o_bc(tb, sb, k1, row_tokens)
        with GK.LOCK, GK.nest(COMP):
            inp = {"sb": sb_json(sb), "k1": k1, "row_tokens": None if row_tokens is None else rle(toks(row_tokens))}
            calls, fn, rec = _tok_recorder(tb)
            tb.tokens_of = rec
            try:
                ct = o_bc(tb, sb, k1, row_tokens)
            finally:
                tb.tokens_of = fn
            inp["tok"] = calls
            tr.write(COMP, "build_compact", inp, ct_json(ct), obj=obj_id(tb))
            return ct

    o_copies = worker.Worker.copies

    @functools.wraps(o_copies)
    def copies(self, batches):
        if GK.active(COMP):
            return o_copies(self, batches)
        with GK.LOCK, GK.nest(COMP):
            inp = {"batches": [[copy_op(c) for c in sb.copies] for sb in batches]}
            # Worker.copies' grouping, the same lines (worker.py 95ca70da): dedup, FPR_PAGECOPY groups, copy_runs
            ops = {}
            for sb in batches:
                for c in sb.copies:
                    ops[(c.pool, c.src_rank, c.src_page, c.dst_rank, c.dst_page)] = c
            groups = {}
            for c in ops.values():
                g_ = groups.setdefault((c.pool, c.src_rank, c.dst_rank), ([], []))
                g_[0].append(c.src_page)
                g_[1].append(c.dst_page)
            size = dict(paged=worker.PAGE_BYTES, window=worker.WINDOW_PAGE_BYTES, state=worker.STATE_SLOT_BYTES)
            out = {"ops": [copy_op(c) for c in ops.values()],
                   "groups": [[p, sr, dr, list(sp), list(dp), size[p]] for (p, sr, dr), (sp, dp) in groups.items()],
                   "runs": [list(r) for r in _if.copy_runs(list(ops.values()))],
                   "state_ops": [[c.src_rank, c.src_page, c.dst_rank, c.dst_page] for c in ops.values() if c.pool == "state"]}
            n = o_copies(self, batches)
            out["n"] = n
            tr.write(COMP, "copies", inp, out, obj=obj_id(self))
            return n

    worker.build_compact = build_compact
    _rebind((), "build_compact", o_bc, build_compact)
    worker.TableBuilder.build = build
    worker.Worker.copies = copies
    worker.__gtrace_tables__ = True
    return True


def install() -> bool:
    """The corpus runner: kv + tables recorders on the bare-name modules (False when PHYS_GTRACE is off); honours
    PHYS_GTRACE_SKIP=kv / tables as serve/'s hooks do."""
    if not tr.on:
        return False
    skip = os.environ.get("PHYS_GTRACE_SKIP", "").split(",")
    if "kv" not in skip:
        GK.install()
    if "tables" in skip:
        return True
    import interfaces
    import worker
    return install_interfaces(interfaces) and install_worker(worker)
