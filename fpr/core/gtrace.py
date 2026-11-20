"""Shared golden-trace writer for the stage-2 recorders (FPR session, 14 Nov 2026; fpr/core/GTRACE.md).
    from gtrace import tr          (fpr/core on sys.path)
    if tr.on: tr.write("scheduler", call, inp, out)
One process-wide sequence counter (seq) across components keeps their relative order; one JSONL file per component under
$PHYS_GTRACE; thread-safe; values must already be JSON-ready (the recorders serialize their own types as in fpr_core.h).
"""
import itertools
import json
import os
import threading


class _Trace:
    def __init__(self):
        self.dir = os.environ.get("PHYS_GTRACE", "")
        self.on = bool(self.dir)
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._files = {}
        if self.on:
            os.makedirs(self.dir, exist_ok=True)

    def next_seq(self) -> int:
        with self._lock:
            return next(self._seq)

    def write(self, comp: str, call: str, inp, out, **extra) -> int:
        if not self.on:
            return -1
        with self._lock:
            seq = next(self._seq)
            f = self._files.get(comp)
            if f is None:
                f = self._files[comp] = open(os.path.join(self.dir, f"{comp}.jsonl"), "a", buffering=1)
            rec = {"seq": seq, "comp": comp, "call": call, "in": inp, "out": out}
            rec.update(extra)
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            return seq


tr = _Trace()
