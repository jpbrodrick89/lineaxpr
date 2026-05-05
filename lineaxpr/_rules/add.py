"""add_p rule and its structural helper functions."""

from __future__ import annotations

import functools
import operator
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from .._linops import (
    BEllpack,
    ColArr,
    ConstantDiagonal,
    Diagonal,
    LinOpProtocol,
    _bcoo_concat,
    _ellpack_to_bcoo_batched,
    replace_slots,
)

# ---------------------------------------------------------------------------
# Tuneable constants (configurable at runtime via env or direct assignment)
# ---------------------------------------------------------------------------

# Cap on `K_total = sum(v.k for v in operands)` for `_add_rule`'s
# partial-match band-dedup scan. Above the cap the scan is skipped and
# the rule falls through to naive band-widening.
#
# Configurable via the `LINEAXPR_BELLPACK_DEDUP_LIMIT` env var, or by
# assigning to `lineaxpr._transform.BELLPACK_DEDUP_LIMIT` at runtime.
#
# Default 200: empirically captures all observed wins across FREUROTH /
# CRAGGLVY / WOODS / CHAINWOO / SROSENBR (K_total ≤ 20), DRCAV1LQ/2LQ
# (K_total up to 39), and NONMSQRT (K_total up to 141). Since the
# dedup uses a hash-based grouping (O(K) at trace time via
# `col.tobytes()` keys), compile-time cost scales linearly in K even
# at large limits — no reason to go lower for a tight budget.
BELLPACK_DEDUP_LIMIT = int(
    os.environ.get("LINEAXPR_BELLPACK_DEDUP_LIMIT", "200")
)

# Minimum K_total to use the 2-operand two-gather reorder path instead of the
# per-band accumulation loop.  For small K the per-band loop emits K cheap
# lax.slice ops that XLA fuses efficiently; the two-gather approach emits 2
# real gather kernels + concat whose overhead dominates at K < threshold.
# Empirical breakeven: regressions confirmed at K≤10 (NONCVXU2/UN K=6,
# BROYDN3DLS K=2-4, EDENSCH K=3-4, LUKSAN12LS K=3-8, BDQRTIC K=6-10);
# wins at K≥15 (DRCAV1LQ/2LQ), K≥72 (NONMSQRT).  Gap between BDQRTIC's
# max K=10 and DRCAV's min K=15 gives clean margin; 12 sits in the middle.
BELLPACK_DEDUP_VECTORISED_MIN = int(
    os.environ.get("LINEAXPR_BELLPACK_DEDUP_VECTORISED_MIN", "12")
)


# ---------------------------------------------------------------------------
# Shared structural helpers (also used by _scatter_add_rule and
# _reduce_sum_rule in materialize.py)
# ---------------------------------------------------------------------------

def _cols_equal(a, b) -> bool:
    """Structural equality test for BEllpack ColArr (np.ndarray only).

    Conservative: returns False for traced jnp arrays (can't compare at
    trace time). The caller falls back to band concat, which is correct
    just wider than necessary.
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b)
    return False


def _broadcast_common_batch(batch_shapes):
    """Return the broadcast shape of a list of batch_shape tuples, or
    None if they're not broadcast-compatible.

    Standard numpy-style right-aligned broadcast: dims of 1 expand to
    match larger; mismatched non-1 dims fail. Mirrors the semantics of
    the `_mul_rule` batch-expand path (ac3c7a6).
    """
    if not batch_shapes:
        return ()
    max_ndim = max(len(s) for s in batch_shapes)
    padded = [(1,) * (max_ndim - len(s)) + s for s in batch_shapes]
    result = []
    for dims in zip(*padded):
        non_one = [d for d in dims if d != 1]
        if not non_one:
            result.append(1)
        elif all(d == non_one[0] for d in non_one):
            result.append(non_one[0])
        else:
            return None
    return tuple(result)


def _broadcast_be_to_batch(be, target_batch_shape):
    """Broadcast a BEllpack's batch_shape to `target_batch_shape`.

    Returns None if the shapes aren't broadcast-compatible. Only
    handles the simple case where every dim of `be.batch_shape` is
    either equal to the corresponding target dim or is 1.
    """
    if be.batch_shape == target_batch_shape:
        return be
    nb = len(target_batch_shape)
    if len(be.batch_shape) != nb:
        return None
    if not all(s == t or s == 1 for s, t in
               zip(be.batch_shape, target_batch_shape)):
        return None
    # Broadcast values.
    if be.k == 1:
        new_values = jnp.broadcast_to(be.data, target_batch_shape + (be.nrows,))
    else:
        new_values = jnp.broadcast_to(
            be.data, target_batch_shape + (be.nrows, be.k))
    # Broadcast cols.
    new_in_cols: list[ColArr] = []
    for c in be.in_cols:
        if not hasattr(c, "ndim") or c.ndim == 1:
            new_in_cols.append(c)
        elif isinstance(c, np.ndarray):
            new_in_cols.append(
                # pyrefly: ignore [bad-argument-type]
                np.broadcast_to(c, target_batch_shape + (be.nrows,))
            )
        else:
            new_in_cols.append(
                jnp.broadcast_to(c, target_batch_shape + (be.nrows,))
            )
    return BEllpack(
        start_row=be.start_row, end_row=be.end_row,
        in_cols=tuple(new_in_cols), data=new_values,
        out_size=be.out_size, in_size=be.in_size,
        batch_shape=target_batch_shape,
    )


def _bcoo_move_v_to_front(bc):
    """BCOO with V at axis -1 → BCOO with V at axis 0, n_batch=0.

    BCOO.transpose forbids permuting batch axes with non-batch axes,
    so the natural `(ndim-1, 0, ..., ndim-2)` permutation we'd need to
    bring V from the trailing axis to the leading axis is unsupported
    for batched BCOOs. We bypass the restriction by flattening the
    batch axes into the index columns and emitting an unbatched
    (n_batch=0) BCOO at the V-at-0 layout.

    Only handles n_dense=0 BCOOs. Returns None on other forms so the
    caller can fall back to densify-and-transpose.
    """
    if not isinstance(bc, sparse.BCOO):
        return None
    if bc.n_dense != 0:
        return None
    n_batch = bc.n_batch
    sh = tuple(bc.shape)
    V = sh[-1]
    sparse_shape = sh[n_batch:]            # includes V at -1
    n_sparse = len(sparse_shape)
    batch_shape = sh[:n_batch]
    B = int(np.prod(batch_shape)) if batch_shape else 1
    nse_per = bc.data.shape[-1]
    # Flatten batch and per-batch nse together.
    flat_data = bc.data.reshape(B * nse_per)
    flat_indices = bc.indices.reshape(B * nse_per, n_sparse)
    if batch_shape:
        # Build per-row batch index columns from a meshgrid.
        grids = np.indices(batch_shape).reshape(n_batch, B)
        batch_cols = np.repeat(grids, nse_per, axis=1).T  # (B*nse, n_batch)
        batch_cols = jnp.asarray(batch_cols, dtype=flat_indices.dtype)
    else:
        batch_cols = jnp.zeros((B * nse_per, 0), dtype=flat_indices.dtype)
    v_col = flat_indices[:, -1:]                # (rows, 1)
    sparse_other = flat_indices[:, :-1]         # (rows, n_sparse - 1)
    new_indices = jnp.concatenate(
        [v_col, batch_cols, sparse_other], axis=1)
    new_shape = (V,) + batch_shape + sparse_shape[:-1]
    return sparse.BCOO((flat_data, new_indices), shape=new_shape)


def _densify_if_wider_than_dense(op, n):
    """Densify a BEllpack whose k >= in_size — no storage win over dense.

    When k >= in_size, the BE stores one band per input column (or more),
    and `values.shape = (*batch, out, k)` already equals dense storage
    `(*batch, out, in_size)`. Any downstream rule that needs to compute
    on it pays BE's bookkeeping overhead for no sparsity benefit. This
    helper is called at emission points in rules that can grow k
    unboundedly (reduce_sum out-axis, add band-widening) to prevent BE
    from carrying effectively-dense state through the rest of the walk.
    """
    if isinstance(op, BEllpack) and op.k >= op.in_size:
        return op.todense()
    return op


def _tile_1row_bellpack(ep, target_rows):
    """Tile a 1-row BEllpack (out_size=1) to have `target_rows` output rows.

    All rows share the same column pattern (ep.in_cols). Values broadcast
    by repeating ep.data `target_rows` times along the row axis.
    Returns a BEllpack with start_row=0, end_row=target_rows.
    """
    assert ep.out_size == 1 and ep.n_batch == 0
    # k=1: values shape (1,) → (target_rows,) via broadcast.
    # k>=2: values shape (1, k) → (target_rows, k).
    if ep.k == 1:
        new_values = jnp.broadcast_to(ep.data, (target_rows,))
    else:
        new_values = jnp.broadcast_to(ep.data, (target_rows, ep.k))
    # Cols: 1D (nrows=1,) → broadcast to (target_rows,).
    new_in_cols: list[ColArr] = []
    for c in ep.in_cols:
        if isinstance(c, np.ndarray):
            # pyrefly: ignore [bad-argument-type]
            new_in_cols.append(np.broadcast_to(c, (target_rows,)).copy())
        else:
            # pyrefly: ignore [bad-argument-type]
            new_in_cols.append(jnp.broadcast_to(jnp.asarray(c), (target_rows,)))
    return BEllpack(
        start_row=0, end_row=target_rows,
        in_cols=tuple(new_in_cols), data=new_values,
        out_size=target_rows, in_size=ep.in_size,
    )


def _slice_be_rows(ep, lo, hi):
    """Slice a BEllpack to absolute row range [lo, hi) within its out_size."""
    rel_lo = lo - ep.start_row
    rel_hi = hi - ep.start_row
    new_in_cols = tuple(
        c[rel_lo:rel_hi] if c.ndim == 1 else c[..., rel_lo:rel_hi]
        for c in ep.in_cols
    )
    if ep.n_batch == 0:
        new_values = ep.data[rel_lo:rel_hi] if ep.k == 1 else ep.data[rel_lo:rel_hi, :]
    else:
        sl = (slice(None),) * ep.n_batch + (slice(rel_lo, rel_hi),)
        new_values = ep.data[sl]
    return BEllpack(lo, hi, new_in_cols, new_values,
                   ep.out_size, ep.in_size, batch_shape=ep.batch_shape)


# ---------------------------------------------------------------------------
# _add_rule sub-path helpers
# ---------------------------------------------------------------------------

def _add_be_dedup(vals, first, n):
    """Partial-match band dedup for all-BEllpack same-range operands.

    Groups bands across operands by column equality; sums values within
    each group to emit one band per unique column pattern. For the
    2-operand case with K_total >= BELLPACK_DEDUP_VECTORISED_MIN, uses
    a two-gather reorder (2 HLO gathers vs K_total lax.slice ops).

    Returns a BEllpack if dedup reduces K, otherwise None (caller falls
    through to band-widen).
    """
    K_total = sum(v.k for v in vals)

    def _col_key(c):
        if isinstance(c, np.ndarray):
            return ("np", c.shape, c.tobytes())
        if isinstance(c, slice):
            return ("slc", c.start, c.stop, c.step)
        return ("id", id(c))  # traced — won't group

    group_cols: list = []
    key_to_group: dict = {}
    inverse = np.empty(K_total, dtype=np.intp)
    band_idx = 0
    for v in vals:
        for b in range(v.k):
            c = v.in_cols[b]
            k_ = _col_key(c)
            g = key_to_group.get(k_)
            if g is None:
                g = len(group_cols)
                key_to_group[k_] = g
                group_cols.append(c)
            inverse[band_idx] = g
            band_idx += 1

    new_k = len(group_cols)
    if new_k >= K_total:
        return None  # no dedup possible

    if len(vals) == 2 and K_total >= BELLPACK_DEDUP_VECTORISED_MIN:
        # Two-gather reorder: gather each operand in [dups | uniq] band
        # order, add the dup slices, concat. dups2 is sorted to match
        # dups1's group order (static np.argsort, zero runtime cost) so
        # v1r[..., i] and v2r[..., i] always correspond to the same group.
        v1, v2 = vals
        inv1 = inverse[:v1.k]
        inv2 = inverse[v1.k:]
        dup_set = set(inv1.tolist()) & set(inv2.tolist())
        dups1 = np.where(np.isin(inv1, list(dup_set)))[0]
        dups2 = np.where(np.isin(inv2, list(dup_set)))[0]
        uniq1 = np.where(~np.isin(inv1, list(dup_set)))[0]
        uniq2 = np.where(~np.isin(inv2, list(dup_set)))[0]
        n_d = len(dups1)
        if n_d == len(dups2):
            dups1_groups = inv1[dups1]
            g_to_pos = {int(g): i for i, g in enumerate(dups1_groups.tolist())}
            dups2_aligned = dups2[np.argsort(
                [g_to_pos[int(inv2[d])] for d in dups2])]
            order1 = np.concatenate([dups1, uniq1])
            order2 = np.concatenate([dups2_aligned, uniq2])
            s1 = bool(np.all(np.diff(order1) >= 0)) if len(order1) > 1 else True
            s2 = bool(np.all(np.diff(order2) >= 0)) if len(order2) > 1 else True
            def _ov(v):
                return v.data_2d
            v1r = _ov(v1).at[..., order1].get(
                unique_indices=True, indices_are_sorted=s1)
            v2r = _ov(v2).at[..., order2].get(
                unique_indices=True, indices_are_sorted=s2)
            combined = jnp.concatenate(
                [v1r[..., :n_d] + v2r[..., :n_d],
                 v1r[..., n_d:], v2r[..., n_d:]], axis=-1)
            new_group_cols = [group_cols[int(gi)] for gi in
                             np.concatenate([dups1_groups,
                                             inv1[uniq1],
                                             inv2[uniq2]])]
            new_values = combined[..., 0] if new_k == 1 else combined
            return _densify_if_wider_than_dense(BEllpack(
                first.start_row, first.end_row,
                tuple(new_group_cols), new_values,
                first.out_size, first.in_size,
                batch_shape=first.batch_shape,
            ), n)
        # n_d != len(dups2): a dup group appears multiple times in one
        # operand; fall through to per-band accumulation loop.

    # N-operand (or n_d mismatch) fallback: per-band accumulation.
    group_values: list = [None] * new_k  # type: ignore[assignment]
    band_idx = 0
    for v in vals:
        for b in range(v.k):
            vals_b = v.data_2d[..., b]
            g = int(inverse[band_idx])
            if group_values[g] is None:
                group_values[g] = vals_b
            else:
                group_values[g] = group_values[g] + vals_b
            band_idx += 1
    new_values = (group_values[0] if new_k == 1
                  else jnp.stack(group_values, axis=-1))
    return _densify_if_wider_than_dense(BEllpack(
        first.start_row, first.end_row,
        tuple(group_cols), new_values,
        first.out_size, first.in_size,
        batch_shape=first.batch_shape,
    ), n)


def _add_be_overlap_merge(ep1, ep2, n):
    """Overlap-merge for exactly two unbatched BEllpacks with partially
    overlapping row ranges. Slices both to the overlap, recurses to
    apply dedup, then row-disjoint-unions head/overlap/tail after
    k-padding with -1 sentinel columns.

    Returns a BEllpack if dedup reduces k in the overlap; otherwise None.
    """
    lo = max(ep1.start_row, ep2.start_row)
    hi = min(ep1.end_row, ep2.end_row)
    if lo >= hi:
        return None  # no overlap
    ov1 = _slice_be_rows(ep1, lo, hi)
    ov2 = _slice_be_rows(ep2, lo, hi)
    merged_ov = _add_rule([ov1, ov2], [True, True], n)
    if not (isinstance(merged_ov, BEllpack) and merged_ov.k < ov1.k + ov2.k):
        return None
    pieces = []
    if ep1.start_row < lo:
        pieces.append(_slice_be_rows(ep1, ep1.start_row, lo))
    if ep2.start_row < lo:
        pieces.append(_slice_be_rows(ep2, ep2.start_row, lo))
    pieces.append(merged_ov)
    if ep1.end_row > hi:
        pieces.append(_slice_be_rows(ep1, hi, ep1.end_row))
    if ep2.end_row > hi:
        pieces.append(_slice_be_rows(ep2, hi, ep2.end_row))
    k_max = max(p.k for p in pieces)
    padded = []
    for p in pieces:
        if p.k < k_max:
            extra_k = k_max - p.k
            extra_cols = tuple(
                np.full(p.nrows, -1, dtype=np.intp) for _ in range(extra_k)
            )
            extra_vals = jnp.zeros((p.nrows, extra_k), dtype=p.data.dtype)
            base = p.data_2d
            padded.append(BEllpack(
                p.start_row, p.end_row,
                p.in_cols + extra_cols,
                jnp.concatenate([base, extra_vals], axis=1),
                p.out_size, p.in_size))
        else:
            padded.append(p)
    return _add_rule(padded, [True] * len(padded), n)


# ---------------------------------------------------------------------------
# _add_rule
# ---------------------------------------------------------------------------

def _add_rule(invals, traced, n, **params):
    """Handle `lax.add_p` / `add_any_p`: sum compatible LinOps, promoting to
    the least-specific form needed. Dispatch is on the set of input kinds.

    Phase B transposed-flag handling: if all traced BEllpack operands
    have `transposed=True`, flip them to canonical (free flag flip),
    run the rest of the rule, flip the result back. If flags are
    mixed, densify any transposed=True BE first (lossy but correct;
    structural sparsity could be recovered later by adding native
    BE row/col-swap real-motion).
    """
    del params

    # ---- Phase B: pre-process transposed flags -----
    # Flip-and-flip-back only safe when ALL traced operands are BE with
    # the same transposed flag. If any traced operand is non-BE (BCOO,
    # dense, CD/Diagonal which are symmetric), or flags are mixed,
    # densify the transposed=True BEs so the rest of the rule sees
    # logical-view dense or canonical BEs.
    all_traced = [v for v, t in zip(invals, traced) if t]
    traced_be = [v for v in all_traced if isinstance(v, BEllpack)]

    # V-axis-middle fast path: all traced are BE with same v_axis (set,
    # i.e. middle), same shape, same in_cols, same row range. Merge by
    # summing data directly. Used by EIGEN-class chains where two
    # parallel BEs converge at v_axis=middle before a final transpose.
    if (traced_be
            and len(traced_be) == len(all_traced)
            and all(b.v_axis is not None for b in traced_be)
            and len({b.v_axis for b in traced_be}) == 1
            and len({b.transposed for b in traced_be}) == 1
            and len({b.shape for b in traced_be}) == 1
            and len({(b.start_row, b.end_row) for b in traced_be}) == 1
            and len({tuple(c.tobytes() if hasattr(c, 'tobytes') else id(c)
                           for c in b.in_cols) for b in traced_be}) == 1):
        first = traced_be[0]
        summed = sum(b.data for b in traced_be[1:]) + traced_be[0].data
        return replace_slots(first, data=summed)

    if traced_be:
        be_flags = {v.transposed for v in traced_be}
        all_be = len(traced_be) == len(all_traced)
        if all_be and be_flags == {True}:
            # All transposed=True BEs: flip all to canonical (free),
            # recurse, flip result back. If all operands share the same
            # v_axis (set or None), restore it on the result so band-
            # widening preserves the V-at-middle invariant.
            common_v_axis = (traced_be[0].v_axis
                             if len({b.v_axis for b in traced_be}) == 1
                             else None)
            flipped = [
                replace_slots(v, transposed=False)
                if t and isinstance(v, BEllpack) else v
                for v, t in zip(invals, traced)
            ]
            res = _add_rule_canonical(flipped, traced, n)
            if isinstance(res, BEllpack):
                return replace_slots(res, transposed=True,
                                     v_axis=common_v_axis)
            # The recurse computed in canonical V-at-(-1) frame; we need
            # V back at axis 0. For ndim==2 that's a swap; for ndim>2
            # it's "move last axis to front".
            if isinstance(res, sparse.BCOO):
                if res.ndim == 2:
                    return res.transpose(axes=(1, 0))
                if res.ndim >= 3:
                    moved = _bcoo_move_v_to_front(res)
                    if moved is not None:
                        return moved
                    perm = (res.ndim - 1,) + tuple(range(res.ndim - 1))
                    return jnp.transpose(res.todense(), perm)
                return res
            if hasattr(res, "ndim") and res.ndim >= 2:
                perm = (res.ndim - 1,) + tuple(range(res.ndim - 1))
                return jnp.transpose(res, perm)
            return res
        if True in be_flags:
            # `(n,)` linear-form + no-op-squeezed BE_T `(n, 1)` mix
            # remains a special case: extract the BE as `(n,)` so the
            # broadcast lines up with the dense vector. The BE_T's
            # `(n, 1)` `.todense()` would broadcast to `(n, n)` here.
            other_is_linear_form = any(
                t and (
                    (isinstance(v, jax.Array) and v.ndim == 1
                     and v.shape[0] == n)
                    or (isinstance(v, sparse.BCOO) and v.ndim == 1
                        and v.shape[0] == n)
                )
                for v, t in zip(invals, traced)
            )
            if other_is_linear_form:
                # Extract BE_T `(n, 1)` no-op-squeezed col-vector to a
                # `(n,)` ndarray — and densify a 1D BCOO too so all
                # operands sum at logical 1D shape (the eqn's expected
                # output aval is 1D in this V-augmented chain).
                invals = tuple(
                    (v.todense()[:, 0]
                     if (t and isinstance(v, BEllpack) and v.transposed
                         and v.n_batch == 0 and v.out_size == 1
                         and v.start_row == 0 and v.end_row == 1)
                     else v.todense()
                     if (t and isinstance(v, sparse.BCOO) and v.ndim == 1)
                     else v)
                    for v, t in zip(invals, traced)
                )
            else:
                # Pre-process: BE_T `(n, 1)` no-op-squeezed col-vector
                # mixed with a square `(n, n)` LinOp (Diagonal /
                # ConstantDiagonal / BEllpack) — broadcast the BE_T to
                # `(n, n)` by tiling its single column n times. Reuses
                # `_tile_1row_bellpack` via the user's transpose
                # pattern: flag-flip BE_T → BE_F (now `(1, n)` row),
                # tile to `(n, n)` BE_F, flag-flip back to BE_T.
                _shapes = [tuple(v.shape) for v, t in zip(invals, traced)
                           if t and hasattr(v, "shape")]
                if _shapes and any(s == (n, n) for s in _shapes):
                    invals = tuple(
                        (replace_slots(
                            _tile_1row_bellpack(
                                replace_slots(v, transposed=False), n,
                            ),
                            transposed=True,
                        )
                         if (t and isinstance(v, BEllpack) and v.transposed
                             and v.n_batch == 0 and v.out_size == 1
                             and v.start_row == 0 and v.end_row == 1
                             and v.in_size == n
                             and tuple(v.shape) != (n, n))
                         else v)
                        for v, t in zip(invals, traced)
                    )
                traced_vals = [v for v, t in zip(invals, traced) if t]
                # Mixed `BE_T + (Diagonal/CD)` (no BCOO/ndarray): flip
                # all BE_T to canonical (free flag flip), recurse via
                # canonical body. If it returns a BE result, flip the
                # flag back to T=True. If it returns a BCOO (e.g. via
                # the canonical `kinds <= {CD, D, BE, BCOO}` branch
                # when full-rows isn't met), keep it at canonical
                # `(out, in)` layout — downstream BCOO rules
                # (`_select_n_rule`, `_scatter_add_rule`) assume that
                # convention.
                # Only fire when the BE_T occupies its full row range:
                # the canonical body's `kinds <= {CD, D, BE}` branch
                # requires `full_rows_ok` and would otherwise fall
                # through to BCOO concat — emitting a BCOO at canonical
                # layout would clash with downstream rules expecting a
                # BE here. Partial-row BE_T's keep the densify path.
                _all_be_full = (
                    all(isinstance(v, (BEllpack, ConstantDiagonal, Diagonal))
                        for v in traced_vals)
                    and all(hasattr(v, 'shape') for v in traced_vals)
                    and len({tuple(v.shape) for v in traced_vals}) == 1
                    and all(getattr(v, 'n_batch', 0) == 0
                            for v in traced_vals
                            if isinstance(v, BEllpack))
                    and all(len(v.shape) == 2 and v.shape[0] == v.shape[1]
                            for v in traced_vals)
                    and all(v.start_row == 0 and v.end_row == v.shape[0]
                            for v in traced_vals
                            if isinstance(v, BEllpack))
                )
                if _all_be_full:
                    flipped = [
                        replace_slots(v, transposed=False)
                        if t and isinstance(v, BEllpack) else v
                        for v, t in zip(invals, traced)
                    ]
                    res = _add_rule_canonical(flipped, traced, n)
                    if isinstance(res, BEllpack):
                        return replace_slots(res, transposed=True)
                    # The recurse computed in canonical V-at-(-1) frame;
                    # we need V back at axis 0.
                    if isinstance(res, sparse.BCOO):
                        if res.ndim == 2:
                            return res.transpose(axes=(1, 0))
                        if res.ndim >= 3:
                            moved = _bcoo_move_v_to_front(res)
                            if moved is not None:
                                return moved
                            perm = ((res.ndim - 1,)
                                    + tuple(range(res.ndim - 1)))
                            return jnp.transpose(res.todense(), perm)
                        return res
                    if hasattr(res, "ndim") and res.ndim >= 2:
                        perm = ((res.ndim - 1,)
                                + tuple(range(res.ndim - 1)))
                        return jnp.transpose(res, perm)
                    return res
                # If a BCOO operand is present and all operands have a
                # `to_bcoo` route at matching shape, promote each to
                # BCOO and concat (preserves sparsity).
                if (any(isinstance(v, sparse.BCOO) for v in traced_vals)
                        and all(hasattr(v, 'shape') for v in traced_vals)
                        and len({tuple(v.shape) for v in traced_vals}) == 1
                        and all(isinstance(v, (BEllpack, sparse.BCOO,
                                                ConstantDiagonal, Diagonal))
                                for v in traced_vals)):
                    bcoo_vals = [
                        v.to_bcoo() if hasattr(v, 'to_bcoo') else v
                        for v in traced_vals
                    ]
                    return _bcoo_concat(bcoo_vals,
                                        shape=tuple(traced_vals[0].shape))
                # Mixed transposed-flag BEllpack operands (BE_T + BE_F)
                # — same shape OR different shape — is a code smell
                # under the supported vmap(in_axes=-1, out_axes=-1)
                # convention. An upstream rule must have dropped the
                # `transposed` flag (or otherwise put V at the wrong
                # axis) for two parallel sub-trees of the same `add`
                # to disagree on the layout. Raise loudly rather than
                # densify or BCOO-promote — patching the symptom in
                # `_add_rule` hides the upstream bug. Audit the
                # producers and fix them.
                _be_only = [v for v in traced_vals
                            if isinstance(v, BEllpack)]
                _non_be_traced = [v for v in traced_vals
                                  if not isinstance(v, BEllpack)]
                if (len(_be_only) >= 2
                        and not _non_be_traced
                        and len({v.transposed for v in _be_only}) > 1):
                    raise AssertionError(
                        "_add_rule: mixed transposed-flag BEllpack "
                        "operands (BE_T + BE_F) — this should not "
                        "happen under the supported vmap convention "
                        "(in_axes=-1, out_axes=-1). An upstream rule "
                        "dropped the `transposed` flag (or put V at "
                        "the wrong axis); please audit the producers "
                        "below and file a lineaxpr issue with the "
                        "minimal `f(y)` that triggers this:\n"
                        + "\n".join(
                            f"  BE shape={v.shape} T={v.transposed} "
                            f"k={v.k} out_size={v.out_size} "
                            f"in_size={v.in_size} batch={v.batch_shape}"
                            for v in _be_only)
                    )
                # BE (any flag) + Diagonal/CD at same logical shape:
                # promote to BCOO and concat. Sparsity is preserved
                # because Diagonal/CD has a fixed n nonzeros and the
                # BE adds its own structural support — the result is
                # well-bounded. NOT applied to BE+BE: that case
                # almost always comes from an upstream `transposed`
                # flag-drop and should be fixed at the producer.
                _has_diag = any(isinstance(v, (ConstantDiagonal, Diagonal))
                                for v in traced_vals)
                if (_has_diag
                        and all(isinstance(v, (BEllpack, ConstantDiagonal,
                                               Diagonal))
                                for v in traced_vals)
                        and all(hasattr(v, 'shape') for v in traced_vals)
                        and len({tuple(v.shape) for v in traced_vals}) == 1
                        and all(getattr(v, 'n_batch', 0) == 0
                                for v in traced_vals
                                if isinstance(v, BEllpack))):
                    bcoo_vals = [
                        v.to_bcoo() if hasattr(v, 'to_bcoo') else v
                        for v in traced_vals
                    ]
                    return _bcoo_concat(bcoo_vals,
                                        shape=tuple(traced_vals[0].shape))
                # Last resort: densify T=True BEs so canonical body
                # sees consistent forms. The mixed-flag BE+BE case is
                # already raised above; this only fires on
                # `BE_T + dense / BCOO / Diagonal` mixes that didn't
                # match the structural BCOO-promote path.
                invals = tuple(
                    (v.todense()
                     if (t and isinstance(v, BEllpack) and v.transposed)
                     else v)
                    for v, t in zip(invals, traced)
                )
    return _add_rule_canonical(invals, traced, n)


def _add_rule_canonical(invals, traced, n):
    """Body of _add_rule, expecting all traced BE operands to be at
    `transposed=False` canonical layout."""
    vals = [v for v, t in zip(invals, traced) if t]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]

    # Broadcast-add (linear-form + matrix): one operand is a BEllpack
    # row-vector carrying a sparse linear form (aval () → matrix shape
    # (1, n)), produced by `_squeeze_rule` on a 1-row slice/gather. Tile
    # its single sparse row `c` to the other operands' `m` rows —
    # yielding a k=len(c.in_cols) column-constant BEllpack of shape
    # (m, n) whose column pattern is c's nonzero columns, row-broadcast.
    # Then recurse through the same-shape structural paths to widen
    # bands with the matrix operand. Avoids the O(m·n) dense fallback
    # on `linear_form + vector` broadcasts that used to bottleneck
    # LIARWHD-class problems.
    if len(vals) >= 2:
        # Only consider LinOp/BCOO shapes — plain arrays (dense fallbacks) have
        # no well-defined "output size" and must not trigger the tiling path.
        non_scalar_out = [v.shape[0] for v in vals
                          if isinstance(v, LinOpProtocol) and v.shape[0] != 1]
        if non_scalar_out and all(s == non_scalar_out[0] for s in non_scalar_out):
            target = non_scalar_out[0]
            tiled_any = False
            new_vals = []
            for v in vals:
                if (v.shape[0] == 1
                        and isinstance(v, BEllpack)
                        and v.n_batch == 0 and v.start_row == 0
                        and v.end_row == 1):
                    new_vals.append(_tile_1row_bellpack(v, target))
                    tiled_any = True
                else:
                    new_vals.append(v)
            if tiled_any:
                return _add_rule(new_vals, [True] * len(new_vals), n)

    kinds = {type(v) for v in vals}

    # All-ConstantDiagonal with matching n: sum the scalar values.
    if kinds == {ConstantDiagonal} and all(v.n == vals[0].n for v in vals):
        return ConstantDiagonal(vals[0].n, sum(v.data for v in vals))

    # Subset of {ConstantDiagonal, Diagonal} with matching n: emit Diagonal.
    if kinds <= {ConstantDiagonal, Diagonal} and all(v.n == vals[0].n for v in vals):
        dtype = next((v.data.dtype for v in vals if isinstance(v, Diagonal)),
                     jnp.result_type(float))
        total = jnp.zeros(vals[0].n, dtype=dtype)
        for v in vals:
            total = total + (
                jnp.broadcast_to(v.data, (v.n,))
                if isinstance(v, ConstantDiagonal) else v.data
            )
        return Diagonal(total)

    # All-BEllpack with matching (start_row, end_row, out_size, in_size,
    # batch_shape): extend bands (tuple concat + values stack on the band
    # axis). O(1) bookkeeping, no per-row value copy. Mismatched ranges
    # or batch_shapes promote to BCOO below.
    if kinds == {BEllpack}:
        # Batch-broadcast path: when BE operands share everything except
        # batch_shape AND batch shapes are numpy-broadcast-compatible
        # (e.g. `(1,)` vs `(4643,)`), expand size-1 axes to the common
        # shape and recurse. Mirrors `_mul_rule`'s batch-expand — same
        # sparsity-preservation rationale. Unblocks DMN15103LS etc.
        # where upstream produces BE with batch=(1,) broadcasting
        # against BE with batch=(4643,).
        first = vals[0]
        same_non_batch = all(
            v.start_row == first.start_row
            and v.end_row == first.end_row
            and v.out_size == first.out_size
            and v.in_size == first.in_size
            for v in vals[1:]
        )
        if same_non_batch and not all(v.batch_shape == first.batch_shape
                                      for v in vals[1:]):
            target_batch = _broadcast_common_batch([v.batch_shape for v in vals])
            if target_batch is not None:
                broadcast_vals = [_broadcast_be_to_batch(v, target_batch)
                                  for v in vals]
                if all(bv is not None for bv in broadcast_vals):
                    return _add_rule(broadcast_vals, traced, n)
        same_range = all(
            v.start_row == first.start_row
            and v.end_row == first.end_row
            and v.out_size == first.out_size
            and v.in_size == first.in_size
            and v.batch_shape == first.batch_shape
            for v in vals[1:]
        )
        if same_range:
            n_batch = first.n_batch
            band_axis = n_batch + 1
            # Same-cols fast path: if every BEllpack has identical in_cols
            # tuples (band for band), sum the values tensors directly
            # (works uniformly for 1D k=1 and 2D k>=2 layouts).
            same_cols = all(
                len(v.in_cols) == len(first.in_cols)
                and all(_cols_equal(c1, c2)
                        for c1, c2 in zip(v.in_cols, first.in_cols))
                for v in vals[1:]
            )
            if same_cols:
                summed_values = vals[0].data
                for v in vals[1:]:
                    summed_values = summed_values + v.data
                return BEllpack(first.start_row, first.end_row,
                               first.in_cols, summed_values,
                               first.out_size, first.in_size,
                               batch_shape=first.batch_shape)
            # Partial-match dedup: generalizes same_cols to group bands
            # across operands by cols-equality. Bands in the same group
            # sum their values and emit ONE band instead of len(group).
            # Strict superset of the same_cols fast path (when every
            # band matches band-for-band, identical behavior). Caps K
            # at `K_total ≤ BELLPACK_DEDUP_LIMIT` to bound the O(K²)
            # cols-equality scan at trace time — beyond that, fall
            # through to the naive widen below.
            K_total = sum(v.k for v in vals)
            if K_total <= BELLPACK_DEDUP_LIMIT:
                result = _add_be_dedup(vals, first, n)
                if result is not None:
                    return result
            # Different cols (or dedup found no reduction): widen bands.
            # Values shape for n_batch=0 is (nrows,) for k=1 or (nrows, k)
            # for k>=2; concat along the band axis (axis=1 i.e. the k dim).
            # For batched values (n_batch>0) the band axis is n_batch+1.
            parts = [v.data_2d for v in vals]
            new_values = jnp.concatenate(parts, axis=band_axis) if len(parts) > 1 else parts[0]
            return _densify_if_wider_than_dense(BEllpack(
                first.start_row, first.end_row,
                tuple(c for v in vals for c in v.in_cols), new_values,
                first.out_size, first.in_size,
                batch_shape=first.batch_shape), n)

        # Row-disjoint union: all-BE with same k / out_size / in_size /
        # batch_shape=(), row ranges pairwise non-overlapping and union
        # contiguous. Produced by `reduce_sum → pad` chains (COATING-
        # class) — each operand is a sparse row-vector at a different
        # position. Concatenate per-row values and cols along the row
        # axis to emit one BE spanning the union range without a BCOO
        # promote per operand.
        if (first.batch_shape == ()
                and all(v.batch_shape == () and v.out_size == first.out_size
                        and v.in_size == first.in_size and v.k == first.k
                        for v in vals)):
            sorted_vals = sorted(vals, key=lambda v: v.start_row)
            abut = all(
                sorted_vals[i].end_row == sorted_vals[i + 1].start_row
                for i in range(len(sorted_vals) - 1)
            )
            if abut:
                k_new = first.k
                if k_new == 1:
                    parts = [v.data_2d[..., 0] for v in sorted_vals]
                    new_values = jnp.concatenate(parts, axis=0)
                else:
                    parts = [v.data for v in sorted_vals]
                    new_values = jnp.concatenate(parts, axis=0)
                new_in_cols: list[ColArr] = []
                for b in range(k_new):
                    band_rows = [v.in_cols[b]
                                 for v in sorted_vals]
                    if all(isinstance(c, np.ndarray) for c in band_rows):
                        new_in_cols.append(np.concatenate(band_rows, axis=0))
                    else:
                        # pyrefly: ignore [bad-argument-type]
                        new_in_cols.append(jnp.concatenate(
                            [jnp.asarray(c) for c in band_rows], axis=0))
                return _densify_if_wider_than_dense(BEllpack(
                    sorted_vals[0].start_row, sorted_vals[-1].end_row,
                    tuple(new_in_cols), new_values,
                    first.out_size, first.in_size,
                ), n)

        # Overlap merge for exactly two unbatched BEllpacks with non-trivial
        # overlapping (non-same, non-disjoint) row ranges.
        # Targets BROYDN3DLS / LEVYMONT class.
        if len(vals) == 2 and all(v.batch_shape == () for v in vals):
            ep1, ep2 = vals[0], vals[1]
            if ep1.out_size == ep2.out_size and ep1.in_size == ep2.in_size:
                result = _add_be_overlap_merge(ep1, ep2, n)
                if result is not None:
                    return result

    # Mix of {ConstantDiagonal, Diagonal, BEllpack} at matching (n, n) shape:
    # promote diagonals to BEllpack bands over the full row range and
    # widen, avoiding BCOO promote. BEllpack's `(start_row, end_row)`
    # range must be `(0, n)` for this to work — a diagonal always spans
    # the full range.
    if (kinds <= {ConstantDiagonal, Diagonal, BEllpack}
            and all(v.shape == vals[0].shape for v in vals)):
        shape = vals[0].shape
        # Only unbatched 2D shapes can be square; skip this
        # diagonal-promote path for batched BEllpacks.
        if len(shape) == 2 and shape[0] == shape[1]:  # square — diagonals fit
            full_rows_ok = all(
                not isinstance(v, BEllpack)
                or (v.start_row == 0 and v.end_row == shape[0])
                for v in vals
            )
            if full_rows_ok:
                # Convert each operand to an BEllpack over [0, n), then add.
                n_sq = shape[0]
                arange_n = np.arange(n_sq)
                ep_vals = []
                for v in vals:
                    if isinstance(v, ConstantDiagonal):
                        ep_vals.append(BEllpack(
                            0, n_sq, (arange_n,),
                            jnp.broadcast_to(jnp.asarray(v.data), (n_sq,)),
                            n_sq, n_sq,
                        ))
                    elif isinstance(v, Diagonal):
                        ep_vals.append(BEllpack(
                            0, n_sq, (arange_n,), v.data, n_sq, n_sq,
                        ))
                    else:
                        ep_vals.append(v)
                return _add_rule(ep_vals, [True] * len(ep_vals), n)

    # Batched-rank-preserving BCOO concat for operands that share a
    # common non-empty batch structure. Fires when either (a) all
    # operands are BEllpacks with the same non-empty `batch_shape` but
    # are otherwise incompatible for the structural same-range widen
    # (e.g. different row ranges from asymmetric pads), or (b) operands
    # are a mix of batched BEllpacks and already-batched BCOOs
    # (produced by a previous pass through this same path) with
    # matching `(*batch, out, in)` shape.
    #
    # Preserves the operand rank `(*batch, out, in)` so downstream
    # rules (notably `_transpose_rule`'s dense fallback) still see the
    # right number of output axes. The flat-BCOO path below would
    # collapse to `(prod_batch * out, in)` and break a later transpose
    # with the original per-output-axis permutation. Regression repro:
    # CLPLATE / TORSION classes under jit (surfaced by the 0c
    # structural transpose keeping a batched BE alive where the dense
    # fallback previously flattened the rank away). Without the mixed
    # {BE, batched-BCOO} branch, DRCAV1LQ/2LQ regressed from BCOO
    # (nse=670761) to dense after the first batched-BCOO emission
    # collided with the next BE add_any.
    if kinds <= {BEllpack, sparse.BCOO}:
        be_operands = [v for v in vals if isinstance(v, BEllpack)]
        bcoo_operands = [v for v in vals if isinstance(v, sparse.BCOO)]
        be_batch_shapes = {v.batch_shape for v in be_operands}
        if be_operands and len(be_batch_shapes) == 1:
            batch = next(iter(be_batch_shapes))
            if batch:
                out_size = be_operands[0].out_size
                in_size = be_operands[0].in_size
                expected_shape = batch + (out_size, in_size)
                be_match = all(v.out_size == out_size
                               and v.in_size == in_size for v in be_operands)
                bcoo_match = all(
                    b.n_batch == len(batch) and b.shape == expected_shape
                    for b in bcoo_operands
                )
                if be_match and bcoo_match:
                    converted = [
                        _ellpack_to_bcoo_batched(v)
                        if isinstance(v, BEllpack) else v
                        for v in vals
                    ]
                    return _bcoo_concat(converted, shape=expected_shape)

    # Any combination of {ConstantDiagonal, Diagonal, BEllpack, BCOO} at
    # compatible matrix shape: promote each to BCOO and concat.
    # CHARDIS0 disambiguation: two batched BEllpacks can flat-collide to
    # the same `.shape` while representing different
    # semantic tensors — e.g. `batch=(n,), out=1, in=m` (col-broadcast,
    # true shape `(n, 1, m)`) vs `batch=(1,), out=n, in=m` (row-broadcast,
    # true shape `(1, n, m)`), both flatten to `(n, m)`. Concatenating
    # their flat BCOOs mis-aligns rows. Guard by requiring all BEllpack
    # operands to share the same `batch_shape`; the fallback dense
    # `reduce(add, …)` then broadcasts correctly via numpy-style rules.
    if kinds <= {ConstantDiagonal, Diagonal, BEllpack, sparse.BCOO}:
        if all(v.shape == vals[0].shape for v in vals):
            ep_batch_shapes = {v.batch_shape for v in vals
                               if isinstance(v, BEllpack)}
            if len(ep_batch_shapes) <= 1:
                bcoo_vals = [v.to_bcoo() if hasattr(v, 'to_bcoo') else v for v in vals]
                return _bcoo_concat(bcoo_vals, shape=vals[0].shape)

    # Linear-form adds: a vector-aval-(k,) LinOp is normally stored as a
    # (k, n) matrix, but an aval-() linear form emerges either as a (n,)
    # ndarray (canonical after `_reduce_sum_rule`) or a BEllpack
    # row-vector/BCOO (after `_squeeze_rule`). When the fallback would mix
    # these forms it'd broadcast-sum to a (1, n) 2D ndarray that
    # downstream rules mis-handle. Normalise all linear-form operands to
    # (n,) ndarrays and sum. Loses row-sparsity info; that's fine —
    # this branch only fires for the rare mixed-forms case after the
    # structural matrix paths above already got a chance.
    # 1D linear-form sparse path: when all operands are 1D-linear-form
    # representations (BE row/col vectors, 1D BCOOs, or actual 1D
    # arrays), each can be converted to a 1D BCOO of shape `(n,)` and
    # concatenated. Preserves sparsity through the canonical body.
    # Only fires when all operands have static `np.ndarray` in_cols
    # (so the 1D-BCOO indices are buildable without traced gathers).
    # Triggered by BROYDN3DLS-class chains where post-vmap forks
    # produce both BE_T(n, 1) (col-vector, V-at-0) and BE_F(1, n)
    # (row-vector, V-at-(-1)) representations of the same 1D linear
    # form (`add_any` aval is `float64[n]` 1D).
    def _be_to_1d_bcoo(v):
        """Convert a 1D-linear-form BE (out_size=1, end_row=1) to a
        1D BCOO of shape (n,). Returns None if cols aren't static."""
        if not all(isinstance(c, np.ndarray) and c.ndim == 1
                   and c.shape[0] == 1 for c in v.in_cols):
            return None
        cols_concat = np.concatenate([c.reshape(1) for c in v.in_cols])
        valid_idx = np.where(cols_concat >= 0)[0]
        valid_cols = cols_concat[valid_idx]
        # data is (1,) for k=1, (1, k) for k>=2 — flatten to (k,).
        vals_flat = jnp.asarray(v.data).reshape(-1)
        valid_vals = vals_flat[jnp.asarray(valid_idx)]
        return sparse.BCOO(
            (valid_vals, jnp.asarray(valid_cols[:, None])),
            shape=(n,),
        )

    def _as_1d_bcoo(v):
        if isinstance(v, sparse.BCOO) and v.shape == (n,):
            return v
        if (isinstance(v, BEllpack) and v.n_batch == 0
                and v.out_size == 1 and v.start_row == 0
                and v.end_row == 1):
            return _be_to_1d_bcoo(v)
        return None

    if (len(vals) >= 2
            and all((isinstance(v, jax.Array) and v.ndim == 1
                     and v.shape[0] == n)
                    or (isinstance(v, sparse.BCOO) and v.shape == (n,))
                    or (isinstance(v, BEllpack) and v.n_batch == 0
                        and v.out_size == 1 and v.start_row == 0
                        and v.end_row == 1)
                    for v in vals)):
        as_bcoo = [_as_1d_bcoo(v) for v in vals]
        if all(b is not None for b in as_bcoo):
            return _bcoo_concat(as_bcoo, shape=(n,))

    def _as_linear_form_row(v):
        if isinstance(v, jax.Array) and v.ndim == 1 and v.shape[0] == n:
            return v
        if (isinstance(v, BEllpack) and v.n_batch == 0
                and v.out_size == 1 and v.start_row == 0
                and v.end_row == 1):
            # transposed=False: dense (1, n); transposed=True: dense (n, 1).
            d = v.todense()
            return d[0] if not v.transposed else d[:, 0]
        if isinstance(v, sparse.BCOO) and v.shape == (1, n):
            return v.todense()[0]
        if isinstance(v, sparse.BCOO) and v.shape == (n, 1):
            return v.todense()[:, 0]
        return None
    linear_form_rows = [_as_linear_form_row(v) for v in vals]
    if all(r is not None for r in linear_form_rows):
        return functools.reduce(operator.add, linear_form_rows)

    # Dense fallback: densify everything and sum. Align V positions
    # if any operand has V at axis 0 (densified-from-transposed-BE
    # chain) — transpose the V-at-(-1) ones to match.
    dense_vals = [v.todense() if isinstance(v, LinOpProtocol) else v for v in vals]
    v_at_zero = any(d.ndim >= 2 and d.shape[0] == n and d.shape[-1] != n
                    for d in dense_vals)
    if v_at_zero:
        aligned = []
        for d in dense_vals:
            if d.ndim >= 2 and d.shape[-1] == n and d.shape[0] != n:
                # V at -1; move to 0.
                perm = (d.ndim - 1,) + tuple(range(d.ndim - 1))
                aligned.append(jnp.transpose(d, perm))
            else:
                aligned.append(d)
        dense_vals = aligned
    return functools.reduce(operator.add, dense_vals)
