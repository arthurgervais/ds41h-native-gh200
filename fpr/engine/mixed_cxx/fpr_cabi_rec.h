// fpr_cabi_rec.h -- fpr_cabi.h RULE 9 (FPR session, 14 Nov 2026; c1-tail's proposal, approved 11:45): an optional
// per-extension RECORD hook, for recorded per-key op programs (c1-tail's H2 head recorder, mixed_cxx_head2.py).
// Kept in its own header (not fpr_cabi.h) so that adding it rebuilds only the extensions that use it.
//
//  9. A pybind wrapper may report the exact Args bytes it is about to launch: FPR_CABI_RECORD("<ext>.<fn>", args) right
//     before its core call, after its own checks, where <ext>.<fn> is the table entry that takes those Args (the same
//     struct type: the recorder replays the bytes through the table). The C entries (the table's functions, called by
//     the driver) never record. One FPR_CABI_RECORD_DECL(<ext>) per extension (file scope, once per .so) defines the hook
//     and exports `void fpr_<ext>_set_record(FprRecordHook hook, void* user)`; hook NULL = off (the default: one pointer
//     test per wrapper call). Set it ONLY from the Python thread while no driver thread runs (no locking, rule 4).
//     The hook gets (user, name, args, sizeof(args)) synchronously, before the launch; it must not launch, synchronize
//     or keep the args pointer (copy the bytes).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef void (*FprRecordHook)(void* user, const char* name, const void* args, int64_t args_size);
#ifdef __cplusplus
}
#endif

#define FPR_CABI_REC_VERSION 1

#define FPR_CABI_RECORD_DECL(ext)                                                                              \
  static FprRecordHook fpr_rec_hook_##ext = nullptr;                                                         \
  static void* fpr_rec_user_##ext = nullptr;                                                                 \
  static FprRecordHook* const fpr_rec_hook_p_ = &fpr_rec_hook_##ext;                                         \
  static void** const fpr_rec_user_p_ = &fpr_rec_user_##ext;                                                 \
  extern "C" __attribute__((visibility("default"))) void fpr_##ext##_set_record(FprRecordHook hook, void* user) { \
    fpr_rec_user_##ext = user;                                                                               \
    fpr_rec_hook_##ext = hook;                                                                               \
  }

#define FPR_CABI_RECORD(name, args)                                                                            \
  do {                                                                                                         \
    if (*fpr_rec_hook_p_ != nullptr)                                                                           \
      (*fpr_rec_hook_p_)(*fpr_rec_user_p_, (name), static_cast<const void*>(&(args)),                         \
                         static_cast<int64_t>(sizeof(args)));                                                  \
  } while (0)
