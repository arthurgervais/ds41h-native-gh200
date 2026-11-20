"""sampfuse: graph H's verify sampler, decide + resample as ONE kernel (C1 tail session, 12 Nov 2026; TEST, used by
hsdt_c1f.py's switch "sfuse"; kernels/head/sampler.py is frozen and unchanged: its device helpers are imported).

sampler.verify2_decide (_decide_kernel, grid (n,)) writes each request's accepted count, sampled tokens and tau = the row
the residual is drawn at; verify2_resample (_resample_kernel, grid (n, vblocks)) then draws at row tau. Here every
(request, vocab block) program recomputes its request's decide (S <= 4 lanes, TP = 4 ranks: a few small loads) to get
tau itself, program (r, 0) writes decide's outputs, and the resample half follows unchanged. The same expressions in the
same order: bitwise the same outputs. One launch and one kernel boundary less on graph H's critical path.
"""
from __future__ import annotations

import sampler as SM
from sampler import (_allowed, _block_stats, _div_t, _last_program, _logsumexp, _merge, _nan_to_neg_inf,  # noqa: F401
                     _pick, _ratio_terms, gumbel_noised_argmax, tl, tl_rand32, tldevice, triton)


@triton.jit
def _decide_resample_kernel(sampled_ptr, sampled_stride, num_sampled_ptr, tau_ptr, g1_ptr, g2_ptr, vp_logit_ptr,
                            vp_lse_ptr, vp_stride, draft_ptr, num_logits, cs_ptr, ce_ptr, idx_ptr, pos_ptr, temp_ptr,
                            seeds_ptr, out_ptr, part_ptr, counts_ptr, logits_ptr, logits_stride, cache_ptr,
                            cache_stride_0, cache_stride_1, row_req_ptr, row_local_ptr, bm_ptr, bm_stride, local_vocab,
                            vocab_start, BLOCK: tl.constexpr, NB: tl.constexpr, TP: tl.constexpr, S: tl.constexpr,
                            WS: tl.constexpr, HAS_MASK: tl.constexpr):
    r = tl.program_id(0)
    b = tl.program_id(1)
    # ---- = sampler._decide_kernel for request r (every program; only program (r, 0) stores)
    req = tl.load(idx_ptr + r).to(tl.int64)
    start = tl.load(cs_ptr + r).to(tl.int64)
    end = tl.load(ce_ptr + r).to(tl.int64)
    num_draft = end - start - 1
    seed = tl.load(seeds_ptr + req)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    greedy = temp == 0.0
    lanes = tl.arange(0, S)
    ranks = tl.arange(0, TP)
    live = lanes < num_draft
    drafted = tl.load(draft_ptr + start + lanes + 1, mask=live, other=-1).to(tl.int64)
    following = tl.load(draft_ptr + start + lanes + 2, mask=lanes < num_draft - 1, other=-1)
    u = tl_rand32(seed, tl.load(pos_ptr + start + lanes, mask=live, other=0), includes_zero=False)
    terms, _m0, _s0 = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, num_draft, temp,
                                   TP, S, WS)
    bq = g1_ptr + (ranks[:, None] * num_logits + start + lanes[None, :]) * WS
    blive = (lanes < num_draft + 1)[None, :]
    bm = _nan_to_neg_inf(tl.load(bq, mask=blive, other=float("-inf")))
    best = _pick(tl.load(bq + 1, mask=blive, other=0.0), ranks[:, None], tl.argmax(bm, axis=0)[None, :]).to(tl.int64)
    mass = tl.sum(tl.load(g2_ptr + (ranks[:, None] * num_logits + start + lanes[None, :] + 1) * WS,
                          mask=(lanes < num_draft - 1)[None, :], other=0.0), axis=0)
    accepted = tl.zeros((), tl.int64)
    verifying = num_draft >= 0
    log_p = tl.zeros((), tl.float32)
    own = b == 0
    for i in tl.static_range(S):
        if i < num_draft:
            raw = _pick(drafted, lanes, i)
            valid = raw >= 0
            token = tl.maximum(raw, 0)
            if greedy:
                if verifying:
                    top = _pick(best, lanes, i)
                    okk = (top == token) & valid
                    verifying = okk
                    accepted += okk
                    if own:
                        tl.store(sampled_ptr + r * sampled_stride + i, tl.where(okk, token, top))
            else:
                if valid:
                    log_p = tl.minimum(log_p + _pick(terms, lanes, i), 0.0)
                verifying &= valid
                if verifying:
                    prefix = tl.exp(log_p)
                    if _pick(following, lanes, i) >= 0:
                        weight = _pick(mass, lanes, i)
                        denom = weight + 1.0 - prefix
                        h = tl.where(denom > 0.0, weight / denom, 1.0)
                    else:
                        h = prefix
                    accepted = tl.where(_pick(u, lanes, i) <= h, i + 1, accepted)
                    if own:
                        tl.store(sampled_ptr + r * sampled_stride + i, token)
    if own:
        tl.store(num_sampled_ptr + r, (accepted + 1).to(tl.int32))
        if greedy & (start + accepted == end - 1):
            tl.store(sampled_ptr + r * sampled_stride + accepted, _pick(best, lanes, accepted))
        tl.store(tau_ptr + r, (start + accepted).to(tl.int32))
    # ---- = sampler._resample_kernel at row tau (computed above, not reloaded); its names carry a "2" (no reuse of the
    #      decide half's variables with other types)
    r2 = r.to(tl.int64)
    row = (start + accepted).to(tl.int32).to(tl.int64)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < local_vocab
    req2 = tl.load(row_req_ptr + row).to(tl.int64)
    temp2 = tl.load(temp_ptr + req2).to(tl.float32)
    value2 = float("-inf")
    token2 = vocab_start + b * BLOCK
    if temp2 != 0.0:
        ok = _allowed(bm_ptr, bm_stride, row, vocab_start + offs, mask, HAS_MASK)
        seed2 = tl.load(seeds_ptr + req2)
        step = tl.load(row_local_ptr + row).to(tl.int64)
        start2 = row - step
        has_next = row + 1 < num_logits
        next_pos = tl.load(row_local_ptr + row + 1, mask=has_next, other=0)
        not_bonus = has_next & (next_pos != 0)
        rejected = tl.load(draft_ptr + row + 1, mask=has_next, other=-1)
        use_draft = not_bonus & (rejected >= 0)
        x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
        x = _div_t(tl.where(ok, x, float("-inf")), temp2)
        if use_draft:
            lanes2 = tl.arange(0, S)
            terms2, _, _ = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req2, start2, step, temp2,
                                        TP, S, WS)
            drafted2 = tl.load(draft_ptr + start2 + lanes2 + 1, mask=lanes2 < step, other=-1)
            log_p2 = 0.0
            for s2 in tl.static_range(S):
                take = (s2 < step) & (_pick(drafted2, lanes2, s2) >= 0)
                log_p2 = tl.where(take, tl.minimum(log_p2 + _pick(terms2, lanes2, s2), 0.0), log_p2)
            stats = g1_ptr + (tl.arange(0, TP) * num_logits + row) * WS
            lse = _logsumexp(tl.load(stats), tl.load(stats + 1))
            t_lp = x - lse
            d_lp = _div_t(tl.load(cache_ptr + req2 * cache_stride_0 + step * cache_stride_1 + offs, mask=mask,
                                  other=float("-inf")).to(tl.float32), temp2) - tl.load(vp_lse_ptr + req2 * vp_stride + step)
            shifted = t_lp + log_p2
            ratio = tl.exp(d_lp - shifted)
            residual = tl.where(mask & (ratio < 1.0), shifted + tldevice.log1p(-ratio), float("-inf")).to(tl.float32)
        else:
            residual = x
        value2, i2 = gumbel_noised_argmax(residual, vocab_start + offs, mask, seed2, tl.load(pos_ptr + row), temp2,
                                          IS_DRAFTING=False, USE_FP64=False, APPLY_TEMPERATURE=False)
        token2 = vocab_start + b * BLOCK + i2
    p = part_ptr + (r2 * NB + b) * 3
    tl.store(p + 1, value2)
    tl.store(p + 2, token2.to(tl.float32))
    if _last_program(counts_ptr, r2, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (r2 * NB + blocks) * 3
        vals = _nan_to_neg_inf(tl.load(q + 1, volatile=True))
        o = out_ptr + r2 * WS
        tl.store(o + 1, tl.max(vals, axis=0))
        tl.store(o + 2, _pick(tl.load(q + 2, volatile=True), blocks, tl.argmax(vals, axis=0)))


def decide_resample(st, logits, g1, g2, m, T: int):
    """= sampler.verify2_decide(st, g1, g2, m, T) then verify2_resample(st, logits, g1, m), as one launch; returns out3."""
    ce = m.cu_end if m.cu_end is not None else m.cu[1:]
    n = m.num_reqs
    bm = m.bitmask
    _decide_resample_kernel[(n, st.vblocks)](
        st.sampled, st.sampled.stride(0), st.num_sampled, st.tau, g1, g2, st.token_logit, st.lse, st.lse.stride(0),
        m.draft_sampled, T, m.cu, ce, m.req_idx, m.pos, st.temp, st.seeds,
        st.out3, st.v_part, st.v_counts, logits, logits.stride(0), st.cache, st.cache.stride(0), st.cache.stride(1),
        m.row_req, m.row_local, bm if bm is not None else st.stats, bm.stride(0) if bm is not None else 0, st.local,
        st.start, BLOCK=SM.VERIFY_BLOCK, NB=st.vblocks, TP=st.ranks, S=triton.next_power_of_2(st.n_spec + 1), WS=SM.WS_V,
        HAS_MASK=bm is not None, num_warps=8)
    return st.out3[:n]


# ---------------------------------------------------------------------------------------------------------------------
# pfuse: sampler.verify2_finish (_finish_kernel, grid (n,)) + hsd._post_glue (grid (n,)) as ONE kernel. Program q does the
# finish of request q (the residual winner over the TP ranks -> last; sampled[q, ns - 1] for temp > 0) and then post_glue's
# body for drafting row q with that `last` / `ns` in registers (post_glue loads them from st.last / st.num_sampled, which
# finish of the same q wrote). Same expressions, same outputs (st.sampled, st.last, gns, anc, dmi, dmp, idx, q_pos, q_slot).
@triton.jit
def _finish_post_kernel(sampled_ptr, sampled_stride, num_sampled_ptr, last_ptr, g3_ptr, nreq, ridx_ptr, temp_ptr,
                        g_dslot_ptr, g_p0_ptr, dmi_ptr, dmp_ptr, anc_ptr, r0, Dp, wbase_ptr, wpages_ptr, rstart_ptr,
                        lslot_ptr, q_pos_ptr, q_slot_ptr, idx_ptr, idx_stride, gns_ptr,
                        TP: tl.constexpr, S: tl.constexpr, WS: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                        NBW_: tl.constexpr, WIN: tl.constexpr, WBLK: tl.constexpr, POOL: tl.constexpr,
                        RING_: tl.constexpr, MAXS: tl.constexpr):
    q = tl.program_id(0)
    # ---- = sampler._finish_kernel, r = q
    r = q.to(tl.int64)
    req = tl.load(ridx_ptr + r).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    nsf = tl.load(num_sampled_ptr + r).to(tl.int64)
    ranks = tl.arange(0, TP)
    last = tl.load(sampled_ptr + r * sampled_stride + nsf - 1)
    if temp != 0.0:
        gq = g3_ptr + (ranks * nreq + r) * WS
        vals = _nan_to_neg_inf(tl.load(gq + 1))
        last = _pick(tl.load(gq + 2), ranks, tl.argmax(vals, axis=0)).to(tl.int64)
        tl.store(sampled_ptr + r * sampled_stride + nsf - 1, last)
    tl.store(last_ptr + r, last)
    # ---- = hsd._post_glue for drafting row q, with ns / last from above
    dslot = tl.load(g_dslot_ptr + q)
    ok = dslot >= 0
    ns = tl.where(ok, nsf, 1)
    p0 = tl.load(g_p0_ptr + q)
    tl.store(gns_ptr + q, ns)
    tl.store(anc_ptr + q, tl.where(ok, last, 0))
    kk = tl.arange(0, KP)
    tl.store(dmi_ptr + q * K + kk, tl.where(ok, dslot, -1).to(tl.int32) + kk * 0, mask=kk < K)
    tl.store(dmp_ptr + q * K + kk, p0 + ns + 1 + kk, mask=kk < K)
    j = q - r0
    if (j >= 0) & (j < Dp):
        Q = p0 + ns
        w = tl.arange(0, WBLK)
        pos = Q - WIN + w
        live_w = w < WIN + K
        if POOL:
            blk = tl.minimum(tl.maximum(pos // 32 - tl.load(wbase_ptr + j), 0), NBW_ - 1)
            page = tl.load(wpages_ptr + j * NBW_ + blk, mask=live_w, other=-1)
            good = (page >= 0) & (pos >= tl.load(rstart_ptr + j)) & (pos >= 0)
            slot = tl.where(good, page * 32 + pos % 32, -1)
        else:
            seq = tl.load(lslot_ptr + j)
            good = (pos >= tl.load(rstart_ptr + j)) & (pos >= 0) & (seq < MAXS)
            slot = tl.where(good, seq * RING_ + pos % RING_, -1)
        for t in tl.static_range(K):
            tl.store(idx_ptr + (j * K + t) * idx_stride + w, slot.to(tl.int32), mask=live_w)
        tl.store(q_pos_ptr + j * K + (w - WIN), pos, mask=(w >= WIN) & live_w)
        tl.store(q_slot_ptr + j * K + (w - WIN), slot, mask=(w >= WIN) & live_w)


def finish_post(st, g3, m, tb, g, r0, Dp, d, k, NBW, WINDOW, pool, RING, max_seqs):
    """= sampler.verify2_finish(st, g3, m) then hsd._post_glue[(n,)](...) as hsdt.HSDT._steps launches it, one launch."""
    n = m.num_reqs
    _finish_post_kernel[(n,)](
        st.sampled, st.sampled.stride(0), st.num_sampled, st.last, g3, n, m.req_idx, st.temp,
        tb["g_dslot"], tb["g_p0"], g["dmi"], g["dmp"], g["anc"], r0, Dp, tb["wbase"], tb["wpages"], tb["rstart"],
        tb["lslot"], d.b["q_pos"], d.b["q_slot"], d.b["idx"], d.b["idx"].stride(0), g["gns"],
        TP=st.ranks, S=triton.next_power_of_2(st.n_spec + 1), WS=SM.WS_V, K=k, KP=triton.next_power_of_2(k), NBW_=NBW,
        WIN=WINDOW, WBLK=triton.next_power_of_2(WINDOW + k), POOL=pool, RING_=RING, MAXS=max_seqs, num_warps=4)


# ---------------------------------------------------------------------------------------------------------------------
# mfuse: Markov draw i >= 1 as ONE kernel = sampler._draft_combine_kernel of draw i - 1 (the winner over the TP ranks:
# token, its logit / lse, the next Markov embedding) + draw i's bias rows bf16(w1[token] . w2[v]) + sampler._draft3_kernel's
# body. HSD's path: combine (1 program per row), torch.mm(emb, w2^T) -> bf16 bias (cuBLAS), draft3 (16 blocks of 2048).
# Here every (row, block of 256 vocab rows) program redoes the combine (TP = 4 words) and its 256 bias dot products
# (RANK = 256, fp32 sums: the same products as cuBLAS in another order, then bf16 rtne as torch.mm's output), so draws
# 1..k-1 cost 1 kernel + the exchange instead of 3 kernels + the exchange. Drafter math only (the bias sums' order may
# differ from cuBLAS by an ulp); the target side is untouched. Program (row, 0) stores draw i - 1's combine outputs.
@triton.jit
def _draft_step_kernel(out_ptr, part_ptr, counts_ptr, base_ptr, base_stride, noise_ptr, noise_stride, cache_ptr,
                       cache_stride_0, cache_stride_1, col, idx_ptr, idx_stride, temp_ptr, local_vocab, vocab_start,
                       gath_ptr, rows, tokens_ptr, tokens_stride, emb_ptr, emb_stride, w1_ptr, w2_ptr, pidx_ptr,
                       pidx_stride, token_logit_ptr, lse_ptr, vp_stride, sink,
                       BLOCK: tl.constexpr, NB: tl.constexpr, WS: tl.constexpr, TP: tl.constexpr, RANK: tl.constexpr,
                       KC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    # ---- draw col - 1's combine (= _draft_combine_kernel)
    ranks = tl.arange(0, TP)
    gq = gath_ptr + (ranks * rows + row) * WS
    gwin = tl.argmax(tl.load(gq), axis=0)
    ptok = _pick(tl.load(gq + 1), ranks, gwin).to(tl.int64)
    rr = tl.arange(0, RANK)
    if b == 0:
        preq = tl.load(pidx_ptr + row * pidx_stride).to(tl.int64)
        pslot = tl.where(preq >= 0, preq, sink)
        tl.store(tokens_ptr + row * tokens_stride + col - 1, ptok)
        tl.store(token_logit_ptr + pslot * vp_stride + col - 1, _pick(tl.load(gq + 2), ranks, gwin))
        tl.store(lse_ptr + pslot * vp_stride + col - 1, _logsumexp(tl.load(gq + 3), tl.load(gq + 4)))
        tl.store(emb_ptr + row * emb_stride + rr, tl.load(w1_ptr + ptok * RANK + rr))
    # ---- draw col's bias rows of this block
    lanes = tl.arange(0, BLOCK)
    offs = b * BLOCK + lanes
    mask = offs < local_vocab
    acc = tl.zeros([BLOCK], tl.float32)
    for k0 in tl.static_range(0, RANK, KC):
        kc = k0 + tl.arange(0, KC)
        e = tl.load(w1_ptr + ptok * RANK + kc).to(tl.float32)
        w = tl.load(w2_ptr + offs[:, None] * RANK + kc[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(w * e[None, :], axis=1)
    bias = acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    # ---- = _draft3_kernel's body (BLOCK-row blocks)
    base = tl.load(base_ptr + row * base_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
    logits = (base + bias).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    logits = tl.where(mask, logits, float("-inf"))
    req = tl.load(idx_ptr + row * idx_stride).to(tl.int64)
    valid = req >= 0
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    tl.store(cache_ptr + req * cache_stride_0 + col * cache_stride_1 + offs, logits.to(cache_ptr.dtype.element_ty),
             mask=mask & valid)
    x = logits
    if temp != 0.0:
        x = _div_t(x, temp)
        nz = tl.load(noise_ptr + row * noise_stride + offs, mask=mask, other=0.0)
        x = tl.where(mask, x + nz, float("-inf"))
    value, idx = tl.max(x, axis=0, return_indices=True)
    t = tl.where(temp > 0.0, temp, 1.0)
    m, s = _block_stats(tl.where(mask, _div_t(logits, t), float("-inf")))
    p = part_ptr + (row * NB + b) * 5
    tl.store(p, value)
    tl.store(p + 1, (vocab_start + b * BLOCK + idx).to(tl.float32))
    tl.store(p + 2, _pick(logits, lanes, idx))
    tl.store(p + 3, m)
    tl.store(p + 4, s)
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (row * NB + blocks) * 5
        vals = tl.load(q, volatile=True)
        win = tl.argmax(vals, axis=0)
        o = out_ptr + row * WS
        tl.store(o, tl.max(vals, axis=0))
        tl.store(o + 1, _pick(tl.load(q + 1, volatile=True), blocks, win))
        tl.store(o + 2, _pick(tl.load(q + 2, volatile=True), blocks, win))
        top, total = _merge(tl.load(q + 3, volatile=True), tl.load(q + 4, volatile=True))
        tl.store(o + 3, top)
        tl.store(o + 4, total)


_PART = {}   # id(RankSampler) -> its [max_reqs, NB, 5] partials for the fused draw (draft3's own d_part has 2048-row blocks)


def draft_step(st, base, gath_prev, rows: int, col: int, idx, pidx, noise, emb_out, block: int = 256):
    """Draw `col` (>= 1): draft_combine(st, gath_prev, rows, col - 1, emb_out, pidx) + torch.mm(emb, w2^T) bias +
    draft3_local(st, base, ..., idx, noise, col) as one launch; returns st.d_out[:rows] (exchanged next, as draft3's)."""
    import torch
    nb = triton.next_power_of_2(triton.cdiv(st.local, block))
    key = (id(st), nb)
    part = _PART.get(key)
    if part is None:
        part = _PART[key] = torch.zeros(st.max_reqs, nb, 5, dtype=torch.float32, device=st.w2.device)
    _draft_step_kernel[(rows, nb)](
        st.d_out, part, st.d_counts, base, base.stride(0), noise, noise.stride(0), st.cache, st.cache.stride(0),
        st.cache.stride(1), col, idx, idx.stride(0), st.temp, st.local, st.start, gath_prev, rows, st.draft_tokens,
        st.draft_tokens.stride(0), emb_out, emb_out.stride(0), st.w1, st.w2, pidx, pidx.stride(0), st.token_logit, st.lse,
        st.lse.stride(0), st.max_reqs, BLOCK=block, NB=nb, WS=SM.WS_D, TP=st.ranks, RANK=st.w1.shape[1], KC=64,
        num_warps=4)
    return st.d_out[:rows]
