"""c1m_precapture (c1-mega, 15 Nov 2026; FPR's Z5 on main's zero-Python list): the C1 mega graph L of every owner key captured
BEFORE serving, so native steps (nd) never meet an unpublished "R|LA|M|<owner>" and fall back to Python on a C1 step.

CAPTURE ONLY. Nothing executes: no synthetic row, no dummy sequence. No KV, Engram (ring / gathers), HSD or D13 state is
touched. (A dummy C1 row would need a sequence slot, KV pages and an Engram state slot of its own. c1mega's A4 insert and the
index layers' compressor / key writes go wherever the bucket's tables point.) A capture only records launches on pointers:
the R=4 bucket state's persistent buffers / tables, which every replay of that bucket reads anyway.

What the D13 first step (d13.D13._first) does for a new mega key, minus its eager step:
  1. tag the bucket state with the owner (Engine._tag_mega: the capture's layer() runs C1Mega.run on it);
  2. the adapter's launch blobs EAGERLY, outside the capture (C1Mega.run builds them on its first call, with .to(dev) H2D
     copies that a stream capture refuses), keyed exactly as run() keys them;
  3. Engine._capture_split(R, st, True, None, lkey=(R, kind, "M", owner) + _ab_key()). It captures graph L (p_embed with
     e0_prepare on estream, the c1mega launch(es) with their Engram event waits and aux means, p_final, the joins) on every
     rank, stores and keeps it, and syncs. Graph H is the bucket's (hkey None: unchanged).
The first replay of such a graph runs exactly what a graph captured after an eager step runs: same code path, same tensors.
CUDA lazy loading is no hazard here: the c1mega aux-mean deadlock, c1-tail's a-t-125c, needs EAGER execution, and graph
instantiation loads every kernel before the first launch.

Preconditions (else skipped / logged, never raised):
  - ENG_C1_MEGA on and the active adapter built (Engine.c1m());
  - bucket R=4 created and populated by at least one decode step: the blobs bake its tables (caches[L] ti / tl, s["idx"],
    pos / own / si / sl, moe_out ...). The D13 path keeps those buffers across steps; c1mega_eng's pack cache relies on it;
  - an idle point: no step in flight on any GPU and no other thread issuing CUDA work (torch's graph capture is in global
    error mode). ENG_DECODE_PRECREATE's done point qualifies.
A bucket state that is later evicted / re-created drops its graphs as today: the next C1 step then takes the Python first
step again.

  from c1m_precapture import precapture
  res = precapture(eng)   # {"captured": [keys], "skipped": [(key, why)], "failed": [(key, error)], "s": seconds}

(fix A, c1-mega 19 Nov) levels=(0, 1): with the adapter's C1M_LVL1 on, also the lvl-1 C1 bucket Rk = R + LVL_TAG (rows
262,144 < ctx <= 524,288 on c1mega): CREATED here when absent (Engine.bucket(Rk): its state, hooks_idx.alloc incl. the shared
long-row logits scratch on the first long variant), then its graph L (Rk, kind, "M", owner) per owner, exactly as lvl 0. The
pack key follows the adapter's run() (+ s["lvl"] when the adapter has C1M_LVL1). Per key: allocated / reserved MiB per GPU.
Graph H of Rk (the decode head) is decode_head_precap's (c1-tail), not captured here.
"""
from __future__ import annotations

import sys
import time

import torch


def precapture(eng, owners=None, kinds=("LA",), log=print, levels=(0,)) -> dict:
    """Capture graph L of the C1 mega keys (R, kind, "M", owner) + Engine._ab_key() for every owner in `owners` (default:
    all ranks) and every kind (D13: "LA"; the synchronous path's "L"). Keys already in eng.graphs are skipped.
    Never raises: failures are logged and returned. A failed owner keeps today's behaviour (the Python first step)."""
    t0 = time.perf_counter()
    res = dict(captured=[], skipped=[], failed=[], s=0.0)
    E_ = sys.modules.get(type(eng).__module__)
    if not getattr(E_, "C1_MEGA", False):
        res["skipped"].append((None, "ENG_C1_MEGA off"))
        return res
    try:
        cm = eng.c1m()
    except Exception as e:  # noqa: BLE001
        res["failed"].append((None, f"adapter: {type(e).__name__}: {e}"))
        log(f"c1m_precapture: no adapter ({type(e).__name__}: {e})")
        return res
    cmod = sys.modules.get(type(cm).__module__)
    R0 = int(getattr(cmod, "NRM", 4))   # the C1 bucket (owner_of: R == NRM)
    lvkey = hasattr(cmod, "C1M_LVL1")   # (fix A adapter: run()'s pack key carries s["lvl"])
    mem0 = [(torch.cuda.memory_allocated(d), torch.cuda.memory_reserved(d)) for d in eng.devs]
    res["per_key_mb"] = {}
    res["create_mb"] = {}
    for lvl in levels:
        R = R0 + int(lvl) * int(getattr(E_, "LVL_TAG", 1 << 20))
        if lvl and not getattr(cmod, "C1M_LVL1", False):
            res["skipped"].append((R, "lvl 1 needs the adapter's C1M_LVL1=1"))
            continue
        if R not in eng.bk:
            if not lvl:
                res["skipped"].append((None, f"bucket {R} not created yet"))
                log(f"c1m_precapture: bucket {R} not created yet: nothing captured")
                continue
            m_ = [(torch.cuda.memory_allocated(d), torch.cuda.memory_reserved(d)) for d in eng.devs]
            try:
                eng.bucket(R)            # (the lvl-1 C1 state; serving would create it at its first long C1 step)
                for d in eng.devs:
                    torch.cuda.synchronize(d)
            except Exception as e:  # noqa: BLE001
                res["failed"].append((R, f"create: {type(e).__name__}: {e}"))
                log(f"c1m_precapture: creating bucket {R} FAILED ({type(e).__name__}: {e}); lvl-{lvl} C1 stays first-step")
                continue
            res["create_mb"][R] = [((torch.cuda.memory_allocated(d) - a) / 2**20, (torch.cuda.memory_reserved(d) - b) / 2**20)
                                   for d, (a, b) in zip(eng.devs, m_)]
            log(f"c1m_precapture: created bucket {R} (lvl {lvl}); allocated / reserved MiB per GPU: "
                + " ".join(f"{a:+.1f}/{b:+.1f}" for a, b in res["create_mb"][R]))
        st = eng.bucket(R)
        W = len(st)
        X1 = getattr(cm, "X1", None)
        if lvl and X1 is not None and X1 is not getattr(cm, "X", None) and hasattr(X1, "prime"):
            for d in eng.devs:   # (fix A) the lvl-1 module's load + smem attribute OUTSIDE the capture (launch() would do it inside)
                X1.prime(int(torch.device(d).index if not isinstance(d, int) else d))
            for d in eng.devs:
                torch.cuda.synchronize(d)
        for o in (range(W) if owners is None else owners):
            for kind in kinds:
                lk = (R, kind, "M", int(o)) + tuple(eng._ab_key())
                if lk in eng.graphs:
                    res["skipped"].append((lk, "already captured"))
                    continue
                m_ = [(torch.cuda.memory_allocated(d), torch.cuda.memory_reserved(d)) for d in eng.devs]
                try:
                    eng._tag_mega(st, int(o))
                    cm._weights()
                    cm._buffers()
                    for k in range(len(cm.runs)):   # C1Mega.run's pack key (s["R"] is the bucket's R on every rank)
                        key = (st[0]["R"], int(o), k, st[int(o)]["moe_out"].data_ptr())
                        if lvkey:
                            key = key + (st[0].get("lvl", 0),)
                        if key not in cm._packs:
                            cm._packs[key] = cm._pack(st, int(o), k)
                    for d in eng.devs:
                        torch.cuda.synchronize(d)
                    eng._capture_split(R, st, True, None, lkey=lk)
                    res["captured"].append(lk)
                    res["per_key_mb"][lk] = [((torch.cuda.memory_allocated(d) - a) / 2**20, (torch.cuda.memory_reserved(d) - b) / 2**20)
                                             for d, (a, b) in zip(eng.devs, m_)]
                except Exception as e:  # noqa: BLE001
                    res["failed"].append((lk, f"{type(e).__name__}: {e}"))
                    log(f"c1m_precapture: {lk} FAILED ({type(e).__name__}: {e}); its first C1 step stays Python")
                finally:
                    eng._tag_mega(st, None)
    res["s"] = time.perf_counter() - t0
    mem1 = [(torch.cuda.memory_allocated(d), torch.cuda.memory_reserved(d)) for d in eng.devs]
    res["mem_mb"] = [((a1 - a0) / 2**20, (r1 - r0) / 2**20) for (a0, r0), (a1, r1) in zip(mem0, mem1)]   # (allocated, reserved)
    log(f"c1m_precapture: captured {len(res['captured'])} ({res['captured']}), skipped {len(res['skipped'])}, failed "
        f"{len(res['failed'])} in {res['s']:.1f} s; memory delta per GPU (allocated / reserved MiB): "
        + " ".join(f"{a:+.1f}/{r:+.1f}" for a, r in res["mem_mb"]))
    for lk, mb in res["per_key_mb"].items():
        log(f"c1m_precapture: key {lk}: allocated / reserved MiB per GPU " + " ".join(f"{a:+.1f}/{r:+.1f}" for a, r in mb))
    return res
