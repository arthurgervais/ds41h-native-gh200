// fpr_core.h -- stage 2: the engine core in C++ (FPR session, architect; 14 Nov 2026). Design note:
// side-session-notes/<main>/cxx-engine-design-14nov.md. Golden-trace method: fpr/core/GTRACE.md.
//
// Every type mirrors today's Python 1:1 (same field names, same semantics, same units) so golden traces compare field by
// field: serve/scheduler.py (Request, SchedConfig, SeqWork, StepPlan), serve/kv_manager.py (CopyOp, SeqKV, AdmitResult,
// OwnedKVManager), serve/interfaces.py (SeqEntry, StepBatch, StepContext, StepResult), serve/rowexp_ref.py
// (CompactTables), engine/d13.py (the D13 host side). A field Python has and C++ does not = a spec bug: report it.
//
// Components and owners (implementations under fpr/core/src/<component>/, each with a golden-trace replay test):
//   IScheduler   = RankScheduler                         c1-mega
//   IKVManager   = OwnedKVManager                        c1-sync
//   ITables      = worker TableBuilder / build_compact / page-copy batching (FPR_PAGECOPY descriptors)   c1-sync
//   ID13Host     = D13 dispatch host side (_async_ok, deltas, descriptors, fallback tables) + HSD prepare host part   c1-tail
//   ICore        = request lifecycle, stop checks, grammar (xgrammar C++), output assembly, the loop thread   c1-overlap
//   IEngine      = the device engine: stage 2a binds today's Python Engine (GIL released around device work); later C++
//   (the stage-1 mixed driver, mixed_plan.h, plugs in under IEngine)                                  FPR (integration)
// v2 (14 Nov ~10:30): c1-mega's 9 spec gaps fixed (block hashes, forced, SeqWork.cached_upto + CPRole, StepPlan.loads /
// idle_frac, CopyOp.step, AdmitResult names, KV admit copy_from / free_pages / hit_after_cap, Geometry, the adapter's
// scheduler controls). Build rule for EVERY component: -O2 -std=c++17 -ffp-contract=off (no FMA contraction: float parity
// with Python's est_us / loads in the last bit). Shared trace writer: fpr/core/gtrace.py (next_seq + per-component files).
// Rules: C++17, no exceptions across component boundaries (Status return), no Python, deterministic (no unordered
// iteration where Python iterates in insertion order: use insertion-ordered maps), no wall-clock reads inside components
// (time comes in as `now` arguments, as in Python), all randomness seeded from the request (sampling is on the device).
// v3 (14 Nov ~10:45): SPLIT into per-component headers, each owned by its porting session (the owner edits it; changes
// that cross components are agreed between the owners and FPR): common.h (FPR), kv.h + tables.h (c1-sync), scheduler.h
// + geometry.h + cost_model.h (c1-mega), d13.h (c1-tail), core.h (c1-overlap), engine.h (FPR). Hash is 128-bit (common.h).
#pragma once
#include "common.h"
#include "geometry.h"
#include "kv.h"
#include "scheduler.h"
#include "tables.h"
#include "engine.h"
#include "d13.h"
#include "core.h"
