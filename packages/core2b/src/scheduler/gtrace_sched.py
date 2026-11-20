"""Golden-trace recorder of serve/scheduler.py RankScheduler (c1-mega, 14 Nov 2026; fpr/core/GTRACE.md, stage 2).

install(sched) -- called by RankScheduler.__init__ when $PHYS_GTRACE is set (default off), or by a test runner -- wraps,
on that instance only, the four boundary calls (add_request, abort, schedule, update) and the KV manager's public methods.
Recording changes no decision: every wrapper passes its arguments and return values through unchanged.

One JSONL line per boundary call, written through fpr/core/gtrace.py (shared seq counter) to $PHYS_GTRACE/scheduler.jsonl:
  {"seq", "comp": "scheduler", "inst": k, "call": add_request | abort | schedule | update | state | config,
   "in": {...}, "out": {...},
   "ext":  [...]   state changed OUTSIDE the four calls since the last line (tests / the vLLM adapter set fields, hide the
                   waiting queue, seed requests; KV calls made outside: their post-state for the replay's KV mirror),
   "hash": [[hid, i, h], ...]   base block hashes evaluated (hid = request incarnation, h = interned id),
   "kv":   [[method, args, ret, post], ...]   the KV calls the scheduler made during this call, in order}
Block hashes are interned per scheduler instance: h = 1, 2, 3, ... in order of first appearance, equal Python values ->
the same id (the KV manager only ever compares them). post = the touched sequences' table changes after the call:
{key: null (freed) | {"pf": pages} | {"pa": appended pages}, "w": [[block, page], ...], "s": state slot}.
Serialization of the types follows fpr_core.h (SeqWork / CopyOp / AdmitResult / RankLoad / StepPlan field order).
"""
from __future__ import annotations

import dataclasses
import inspect
import itertools
import json
import os
import sys
import traceback
import weakref

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)
from gtrace import tr  # noqa: E402

COMP = "scheduler"
_INST = itertools.count()
_LIVE: "weakref.WeakSet" = weakref.WeakSet()
SNAP_EVERY = int(os.environ.get("PHYS_GTRACE_SNAP", "256") or 256)

REQ_FIELDS = ("phase", "rank", "num_computed", "num_output", "prefill_end", "cached_tokens", "replay_start",
              "skip_until_step", "inflight", "first_pending", "max_new_tokens", "priority", "score", "forced",
              "arrival", "prompt_len", "admit_time", "pf_progress_time", "first_chunk_time", "first_token_time",
              "finish_time", "last_pf_step", "cp_forced", "uid", "home_rank", "n_preempt", "n_migrate",
              "admit_steps_waited", "hit_tokens_total", "copied_tokens_total")
SCHED_FIELDS = ("step_id", "rr", "imbalance_steps", "pf_credit", "dec_since_mixed", "lam")
LOAD_FIELDS = ("dec_rows", "dec_idx_pos", "dec_cand_pos", "n_dec_seqs", "max_dec_ctx", "pf_rows", "pf_tail_rows",
               "pf_idx_pos", "pf_cand_pos")
KV_SEQ = ("lookup", "free_pages", "hit_after_cap", "admit", "can_allocate", "allocate", "commit", "shrink_reservation",
          "open_shadow", "trim_window", "free", "flush_deferred", "migrate", "tier_copies", "release_copies")
KV_TOUCH = {"admit", "allocate", "commit", "open_shadow", "trim_window", "free", "migrate"}   # arg 0 = the sequence key
POOL = {"paged": 0, "window": 1, "state": 2}


def enc_rid(x) -> str:
    """Rid as the C++ side holds it: str(rid) (the scheduler sorts by str(rid)); CP shadow keys ("cp", rid, rank) ->
    kv.h shadow_key(): 'cp\x1f<rid>\x1f<rank>'."""
    if isinstance(x, tuple) and len(x) == 3 and x[0] == "cp":
        return f"cp\x1f{enc_rid(x[1])}\x1f{x[2]}"
    return x if isinstance(x, str) else str(x)


def enc_ident(x) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else "#" + repr(x)


def enc_images(ims) -> list:
    return [[im[0], im[1], enc_ident(im[2])] if len(im) == 3 else ["bad", repr(im)] for im in ims]


def enc_copy(o) -> list:
    return [POOL.get(o.pool, o.pool), o.src_rank, o.src_page, o.dst_rank, o.dst_page, o.step]


def enc_load(ld) -> list:
    return [getattr(ld, f) for f in LOAD_FIELDS]


def enc_cp(cp):
    if cp is None:
        return None
    return [cp.group, enc_rid(cp.rid), cp.owner, [list(s) for s in cp.subs], cp.me]


def enc_work(w) -> list:
    # fpr_core.h SeqWork order: rid rank kind start n cached_upto replay_start last_chunk score forced uid known cp images
    return [enc_rid(w.rid), w.rank, w.kind, w.start, w.n, w.cached_upto, w.replay_start, bool(w.last_chunk),
            bool(w.score), bool(w.forced), w.uid, w.known, enc_cp(w.cp), enc_images(w.images)]


def enc_plan(p) -> dict:
    return {"step": p.step, "work": [enc_work(w) for w in p.work], "copies": [enc_copy(o) for o in p.copies],
            "preempted": [enc_rid(r) for r in p.preempted], "migrated": [enc_rid(r) for r in p.migrated],
            "loads": [enc_load(ld) for ld in p.loads], "est_us": p.est_us, "idle_frac": p.idle_frac}


def jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return [[jsonable(k), jsonable(x)] for k, x in v.items()]
    return repr(v)


_REQ_REC: dict = {}           # id(Request) -> (Request, Recorder, hid): the recorder that logs its base hashes


def _patch_block_hashes() -> None:
    """Request.block_hashes (class level, once): log the base hashes it evaluated for a recorded request. The return
    value and every argument pass through; hash_fn objects are never replaced (tests check them by identity)."""
    import scheduler as _S
    if getattr(_S.Request.block_hashes, "_gt_wrapped", False):
        return
    orig = _S.Request.block_hashes

    def block_hashes(self, n_tokens):
        r = orig(self, n_tokens)
        e = _REQ_REC.get(id(self))
        if e is not None and e[0] is self:
            e[1].log_bases(self, e[2], enumerate(r))
        return r
    block_hashes._gt_wrapped = True
    _S.Request.block_hashes = block_hashes


class _KVProxy:
    """scheduler.kv while recording: every attribute read / write goes to the real OwnedKVManager; the scheduler's
    calls of its public methods are logged by the OUTERMOST wrapper (so a test that interposes its own function on
    kv.<method> -- e.g. a deliberately broken migrate -- is logged with what the scheduler actually received)."""
    __slots__ = ("_gt_real", "_gt_rec")

    def __init__(self, real, rec):
        object.__setattr__(self, "_gt_real", real)
        object.__setattr__(self, "_gt_rec", rec)

    def __getattr__(self, name):
        v = getattr(self._gt_real, name)
        if name in KV_SEQ and callable(v):
            return self._gt_rec.kv_call(name, v)
        return v

    def __setattr__(self, name, v):
        setattr(self._gt_real, name, v)

    def __delattr__(self, name):
        delattr(self._gt_real, name)


class Recorder:
    def __init__(self, s):
        self.s = s
        self.inst = next(_INST)
        self.ids: dict = {}                 # hash value -> id
        self.hid = itertools.count(1)
        self.hids: dict = {}                # id(Request) -> (Request, hid)
        self.depth = 0                      # inside a boundary call
        self.kv_depth = 0
        self.cur_kv: list | None = None
        self.ext_kv: list = []              # KV calls outside the boundary calls (their post-state)
        self.hash_log: list = []
        self.mirror: dict = {}              # seq key -> (id(pages), len(pages), window list, slot)
        self.snap = None                    # externally visible state at the last line's exit
        self.plans: dict = {}               # id(plan) -> (plan, encoded) for update()
        self.n_sched = 0
        self.kv_obj = None                  # the _KVProxy installed as s.kv
        self.seen: dict = {}                # hid -> blocks whose base hash is logged
        self.sigs: dict = {}                # KV method -> its signature (the class method's, without self)

    # ------------------------------------------------------------------ hashes
    def intern(self, h) -> int:
        i = self.ids.get(h)
        if i is None:
            i = self.ids[h] = len(self.ids) + 1
        return i

    def hid_of(self, q) -> int:
        e = self.hids.get(id(q))
        if e is None or e[0] is not q:
            e = self.hids[id(q)] = (q, next(self.hid))
            _REQ_REC[id(q)] = (q, self, e[1])
            self.seen[e[1]] = set()
        return e[1]

    def log_bases(self, q, hid, items) -> None:
        """Base block hashes (hid, i, id): the key itself, or for an ImageBlockKeys key ("mm", base, prefix) its base."""
        from scheduler import ImageBlockKeys
        img = isinstance(q.hash_fn, ImageBlockKeys)
        seen = self.seen[hid]
        for i, h in items:
            if i in seen:
                continue
            seen.add(i)
            b = h[1] if img and type(h) is tuple and len(h) == 3 and h[0] == "mm" else h
            self.hash_log.append([hid, i, self.intern(b)])

    def hash_kind(self, q) -> str:
        from scheduler import ImageBlockKeys
        return "image" if isinstance(q.hash_fn, ImageBlockKeys) else "base"

    # ------------------------------------------------------------------ KV wrappers
    def wrap_kv(self) -> None:
        """Install the KV proxy (again after the KV manager object was replaced, e.g. the adapter's reset)."""
        kv = self.s.kv
        if kv is self.kv_obj:
            return
        if isinstance(kv, _KVProxy):
            kv = kv._gt_real
        self.mirror = {}
        self.kv_obj = _KVProxy(kv, self)
        self.s.kv = self.kv_obj

    def kv_call(self, m, f):
        rec = self
        sig = self.sigs.get(m)
        if sig is None:
            try:
                sig = inspect.signature(getattr(type(self.kv_obj._gt_real), m))
                sig = sig.replace(parameters=list(sig.parameters.values())[1:])     # drop self
            except (TypeError, ValueError, AttributeError):
                sig = False
            self.sigs[m] = sig

        def wrapper(*a, **k):
            if rec.kv_depth:                         # a test's own wrapper calling the captured method: pass through
                return f(*a, **k)
            args = sig.bind(*a, **k) if sig else None
            if args is not None:
                args.apply_defaults()
                av = dict(args.arguments)
            else:
                av = {"a": list(a), **k}
            asked = None
            if m == "commit" and args is not None:
                bh = av.get("block_hash")
                asked = []
                q = rec.s.reqs.get(av.get("seq_id"))
                e = rec.hids.get(id(q)) if q is not None else None

                def logged(i, _bh=bh):
                    h = _bh(i)
                    asked.append([i, rec.intern(h)])
                    if e is not None and e[0] is q:
                        rec.log_bases(q, e[1], ((i, h),))
                    return h
                args.arguments["block_hash"] = logged
                a, k = args.args, args.kwargs
            rec.kv_depth += 1
            try:
                r = f(*a, **k)
            finally:
                rec.kv_depth -= 1
            ent = [m, rec.enc_kv_args(m, av, asked), rec.enc_kv_ret(m, r)]
            if m in KV_TOUCH:
                key = av.get("seq_id", av.get("key"))
                post = rec.post_state(key)
                if post is not None:
                    ent.append(post)
            if rec.depth:
                rec.cur_kv.append(ent)
            else:
                rec.ext_kv.append(ent)
            return r
        return wrapper

    def enc_kv_args(self, m, av, asked):
        out = {}
        for k, v in av.items():
            if k in ("self",):
                continue
            if k in ("seq_id", "key"):
                out["id"] = enc_rid(v)
            elif k == "block_hashes":
                out["h"] = [self.intern(h) for h in v]
            elif k == "block_hash":
                out["asked"] = asked
            else:
                out[k] = v
        return out

    def enc_kv_ret(self, m, r):
        if r is None or isinstance(r, (bool, int)):
            return r
        if m == "admit":
            return [r.rank, r.hit_blocks, r.cached_tokens, r.replay_start, r.copied_blocks, [enc_copy(o) for o in r.copies]]
        if m in ("migrate", "tier_copies"):
            return [enc_copy(o) for o in r]
        if isinstance(r, (list, tuple)):
            return list(r)
        return repr(r)

    def post_state(self, key):
        """Table changes of sequence `key` since the last logged state (None: unchanged)."""
        kv = self.s.kv
        ek = enc_rid(key)
        sk = kv.seqs.get(key)
        last = self.mirror.get(ek)
        if sk is None:
            if last is None:
                return None
            del self.mirror[ek]
            return {ek: None}
        win = [[b, p] for b, p in sk.window.items()]
        d = {}
        if last is None or last[0] != id(sk.pages) or last[1] > len(sk.pages):
            d["pf"] = list(sk.pages)
        elif last[1] < len(sk.pages):
            d["pa"] = sk.pages[last[1]:]
        if last is None or last[2] != win:
            d["w"] = win
        if last is None or last[3] != sk.state_slot:
            d["s"] = sk.state_slot
        self.mirror[ek] = (id(sk.pages), len(sk.pages), win, sk.state_slot)
        return {ek: d} if d else None

    # ------------------------------------------------------------------ external-state reconciliation
    def req_state(self, q) -> list:
        return [jsonable(getattr(q, f, None)) for f in REQ_FIELDS] + [
            [[ri, enc_rid(v[0]), v[1]] for ri, v in q.cp_shadows.items()], jsonable(q.cp_prev_subs),
            enc_images(q.images), self.hash_kind(q)]

    def state(self) -> dict:
        s = self.s
        return {"reqs": {enc_rid(r): self.req_state(q) for r, q in s.reqs.items()},
                "waiting": [enc_rid(q.rid) for q in s.waiting], "running": [enc_rid(q.rid) for q in s.running],
                "prefilling": sorted(enc_rid(q.rid) for q in s.prefilling),
                "sched": [jsonable(getattr(s, f)) for f in SCHED_FIELDS],
                "free_uids": list(s.free_uids), "free_shadow_uids": list(s.free_shadow_uids),
                "cfg": dataclasses.asdict(s.cfg), "cost": dict(vars(s.cost.p)) if hasattr(s.cost, "p") else {},
                "rejected": {enc_rid(k): v for k, v in s.rejected.items()}}

    def ext(self) -> list:
        """What changed outside the boundary calls since the last line (applied by the replay before this call)."""
        s = self.s
        out = []
        if s.kv is not self.kv_obj:
            self.wrap_kv()
            out.append(["kv_reset"])
        cur = self.state()
        old = self.snap
        if old is None:
            return out
        for r, st in cur["reqs"].items():
            o = old["reqs"].get(r)
            q = s.reqs[r if r in s.reqs else next(k for k in s.reqs if enc_rid(k) == r)]
            if o is None:
                out.append(["new", r, self.hid_of(q), st])
                continue
            for i, f in enumerate(REQ_FIELDS + ("cp_shadows", "cp_prev_subs", "images", "hash_kind")):
                if st[i] != o[i]:
                    out.append(["req", r, f, st[i]])
        for r in old["reqs"]:
            if r not in cur["reqs"]:
                out.append(["del", r])
        for k in ("waiting", "running", "prefilling", "free_uids", "free_shadow_uids"):
            if cur[k] != old[k]:
                out.append([k, cur[k]])
        for i, f in enumerate(SCHED_FIELDS):
            if cur["sched"][i] != old["sched"][i]:
                out.append(["sched", f, cur["sched"][i]])
        for k in ("cfg", "cost"):
            for f, v in cur[k].items():
                if old[k].get(f) != v:
                    out.append([k, f, jsonable(v)])
        for r, v in cur["rejected"].items():
            if old["rejected"].get(r) != v:
                out.append(["rejected", r, v])
        return out

    # ------------------------------------------------------------------ lines
    def begin(self) -> list:
        ext = self.ext()
        if self.ext_kv:
            ext.append(["kv", self.ext_kv])
            self.ext_kv = []
        self.depth += 1
        self.cur_kv = []
        return ext

    def end(self, call, inp, out, ext) -> None:
        self.depth -= 1
        hashes, self.hash_log = self.hash_log, []
        kv, self.cur_kv = self.cur_kv, None
        extra = {"inst": self.inst}
        if ext:
            extra["ext"] = ext
        if hashes:
            extra["hash"] = hashes
        if kv:
            extra["kv"] = kv
        tr.write(COMP, call, inp, out, **extra)
        self.snap = self.state()

    def snapshot(self, why: str) -> None:
        st = self.state()
        st["stats"] = jsonable(self.s.stats)
        st["uids"] = {enc_rid(r): q.uid for r, q in self.s.reqs.items()}
        tr.write(COMP, "state", {"why": why}, st, inst=self.inst)

    def config(self) -> None:
        s = self.s
        g = s.g
        test = None
        for fr in inspect.stack()[2:12]:
            o = fr.frame.f_locals.get("self")
            if o is not None and hasattr(o, "id") and type(o).__module__ != __name__ and "TestCase" in \
                    [c.__name__ for c in type(o).__mro__]:
                test = o.id()
                break
        cost = type(s.cost).__name__
        tr.write(COMP, "config", {"tag": os.environ.get("PHYS_GTRACE_TAG", ""), "test": test, "pid": os.getpid()},
                 {"geometry": dataclasses.asdict(g), "geometry_cls": type(g).__name__,
                  "pools": [[p.name, p.kind, p.tokens_per_page, p.bytes_per_page, p.num_pages] for p in g.pools()],
                  "rows_per_decode_seq": g.rows_per_decode_seq, "rows_cap": g.rows_cap,
                  "sched": dataclasses.asdict(s.cfg), "cost_cls": cost,
                  "cost": dict(vars(s.cost.p)) if hasattr(s.cost, "p") else {},
                  "kv_hbm_pages": int(os.environ.get("FPR_KV_HBM_PAGES", "0") or 0)}, inst=self.inst)


def install(s) -> Recorder:
    """Wrap scheduler instance `s` (idempotent)."""
    if getattr(s, "_gtrace", None) is not None or not tr.on:
        return getattr(s, "_gtrace", None)
    rec = Recorder(s)
    s._gtrace = rec
    _LIVE.add(s)
    _patch_block_hashes()
    rec.config()
    rec.wrap_kv()
    add, abort, schedule, update = s.add_request, s.abort, s.schedule, s.update

    def _exc(e) -> dict:
        return {"exc": type(e).__name__, "msg": str(e)[:300]}

    def add_request(req):
        ext = rec.begin()
        hid = rec.hid_of(req)
        inp = {"rid": enc_rid(req.rid), "hid": hid, "prompt_len": req.prompt_len, "max_new_tokens": req.max_new_tokens,
               "arrival": req.arrival, "priority": req.priority, "score": bool(req.score), "forced": bool(req.forced),
               "images": enc_images(req.images), "hash_kind": rec.hash_kind(req)}
        try:
            ok = add(req)
        except Exception as e:  # noqa: BLE001
            rec.end("add_request", inp, _exc(e), ext)
            raise
        rec.end("add_request", inp, {"ok": ok, "reason": s.rejected.get(req.rid) if not ok else None,
                                     "max_new_tokens": req.max_new_tokens, "hash_kind": rec.hash_kind(req)}, ext)
        return ok

    def abort_(rid):
        ext = rec.begin()
        inp = {"rid": enc_rid(rid)}
        try:
            r = abort(rid)
        except Exception as e:  # noqa: BLE001
            rec.end("abort", inp, _exc(e), ext)
            raise
        rec.end("abort", inp, {}, ext)
        return r

    def schedule_(now=0.0):
        ext = rec.begin()
        inp = {"now": now, "waiting_len": len(s.waiting)}
        try:
            p = schedule(now)
        except Exception as e:  # noqa: BLE001
            rec.end("schedule", inp, _exc(e), ext)
            raise
        enc = enc_plan(p)
        rec.plans[id(p)] = (p, enc)
        if len(rec.plans) > 8:
            for k in list(rec.plans)[:-8]:
                del rec.plans[k]
        rec.end("schedule", inp, enc, ext)
        rec.n_sched += 1
        if p.step % SNAP_EVERY == 0:
            rec.snapshot(f"step {p.step}")
        return p

    def update_(plan, accepted, now=0.0):
        ext = rec.begin()
        known = rec.plans.get(id(plan))
        inp = {"step": plan.step, "accepted": [[enc_rid(r), a] for r, a in accepted.items()], "now": now}
        if known is None or known[0] is not plan or enc_plan(plan) != known[1]:
            inp["plan"] = enc_plan(plan)                 # not the plan schedule() returned (or changed since)
        try:
            fin = update(plan, accepted, now)
        except Exception as e:  # noqa: BLE001
            rec.end("update", inp, _exc(e), ext)
            raise
        rec.end("update", inp, {"finished": [enc_rid(q.rid) for q in fin]}, ext)
        return fin

    s.add_request, s.abort, s.schedule, s.update = add_request, abort_, schedule_, update_
    rec.snap = rec.state()
    return rec


def _final():
    for s in list(_LIVE):
        rec = getattr(s, "_gtrace", None)
        if rec is not None and rec.n_sched:
            try:
                rec.snapshot("exit")
            except Exception:  # noqa: BLE001
                traceback.print_exc()


import atexit  # noqa: E402

atexit.register(_final)
