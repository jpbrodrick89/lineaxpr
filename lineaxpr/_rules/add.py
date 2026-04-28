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
    ConstantDiagonal,
    Diagonal,
    _bcoo_concat,
    _ellpack_to_bcoo_batched,
    _to_bcoo,
    _to_dense,
)

# ---------------------------------------------------------------------------
# Tuneable constants (configurable at runtime via env or direct assignment)
# ---------------------------------------------------------------------------

# Cap on `K_total = sum(v.k for v in operands)` for `_add_rule`'s
# partial-match band-dedup scan. Above the cap the scan is skipped and
# the rule falls through to naive band-widening.
#
# Configurable via the `LINEAXPR_BELLPACK_DEDUP_LIMIT` env var, or by
# assigning to `lineaxpr.materialize.BELLPACK_DEDUP_LIMIT` at runtime.
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

def _linop_matrix_shape(v):
    """Return the structural shape used to check operand compatibility
    in `_add_rule`'s mix-path. Unbatched LinOps → 2D `(out, in)`;
    batched BEllpack → `(*batch_shape, out_size, in_size)` (matching
    what `_ellpack_to_bcoo_batched` now emits — a batched BCOO, since
    the old flat-collapse was removed in commit TODO-0j). BCOO passes
    through its own shape.

    Returning structural (rather than flattened) shape keeps operand
    compatibility honest: a batched BE only matches another batched
    LinOp with the same batch structure, and batched-vs-unbatched
    mixes fall through to the dense fallback rather than attempting
    a `_bcoo_concat` across inconsistent `n_batch` values."""
    if isinstance(v, (ConstantDiagonal, Diagonal)):
        return (v.n, v.n)
    if isinstance(v, BEllpack):
        return (*v.batch_shape, v.out_size, v.in_size)
    if isinstance(v, sparse.BCOO):
        return v.shape
    return None


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
        new_values = jnp.broadcast_to(be.values, target_batch_shape + (be.nrows,))
    else:
        new_values = jnp.broadcast_to(
            be.values, target_batch_shape + (be.nrows, be.k))
    # Broadcast cols.
    new_in_cols = []
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
        in_cols=tuple(new_in_cols), values=new_values,
        out_size=be.out_size, in_size=be.in_size,
        batch_shape=target_batch_shape,
    )


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
        return _to_dense(op, n)
    return op


def _tile_1row_bellpack(ep, target_rows):
    """Tile a 1-row BEllpack (out_size=1) to have `target_rows` output rows.

    All rows share the same column pattern (ep.in_cols). Values broadcast
    by repeating ep.values `target_rows` times along the row axis.
    Returns a BEllpack with start_row=0, end_row=target_rows.
    """
    assert ep.out_size == 1 and ep.n_batch == 0
    # k=1: values shape (1,) → (target_rows,) via broadcast.
    # k>=2: values shape (1, k) → (target_rows, k).
    if ep.k == 1:
        new_values = jnp.broadcast_to(ep.values, (target_rows,))
    else:
        new_values = jnp.broadcast_to(ep.values, (target_rows, ep.k))
    # Cols: 1D (nrows=1,) → broadcast to (target_rows,).
    new_in_cols = []
    for c in ep.in_cols:
        if isinstance(c, np.ndarray):
            # pyrefly: ignore [bad-argument-type]
            new_in_cols.append(np.broadcast_to(c, (target_rows,)).copy())
        else:
            # pyrefly: ignore [bad-argument-type]
            new_in_cols.append(jnp.broadcast_to(jnp.asarray(c), (target_rows,)))
    return BEllpack(
        start_row=0, end_row=target_rows,
        in_cols=tuple(new_in_cols), values=new_values,
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
        new_values = ep.values[rel_lo:rel_hi] if ep.k == 1 else ep.values[rel_lo:rel_hi, :]
    else:
        sl = (slice(None),) * ep.n_batch + (slice(rel_lo, rel_hi),)
        new_values = ep.values[sl]
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
                return v.values_2d
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
            vals_b = v.values_2d[..., b]
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
            extra_vals = jnp.zeros((p.nrows, extra_k), dtype=p.values.dtype)
            base = p.values_2d
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
    the least-specific form needed. Dispatch is on the set of input kinds."""
    del params
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
        shapes = [_linop_matrix_shape(v) for v in vals]
        non_scalar_out = [s[0] for s in shapes
                          if s is not None and s[0] != 1]
        if non_scalar_out and all(s == non_scalar_out[0] for s in non_scalar_out):
            target = non_scalar_out[0]
            tiled_any = False
            new_vals = []
            for v, s in zip(vals, shapes):
                if (s is not None and s[0] == 1
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
        return ConstantDiagonal(vals[0].n, sum(v.value for v in vals))

    # Subset of {ConstantDiagonal, Diagonal} with matching n: emit Diagonal.
    if kinds <= {ConstantDiagonal, Diagonal} and all(v.n == vals[0].n for v in vals):
        dtype = next((v.values.dtype for v in vals if isinstance(v, Diagonal)),
                     jnp.result_type(float))
        total = jnp.zeros(vals[0].n, dtype=dtype)
        for v in vals:
            total = total + (
                jnp.broadcast_to(v.value, (v.n,))
                if isinstance(v, ConstantDiagonal) else v.values
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
                summed_values = vals[0].values
                for v in vals[1:]:
                    summed_values = summed_values + v.values
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
            new_in_cols = tuple(c for v in vals for c in v.in_cols)
            parts = [v.values_2d for v in vals]
            new_values = jnp.concatenate(parts, axis=band_axis) if len(parts) > 1 else parts[0]
            return _densify_if_wider_than_dense(BEllpack(
                first.start_row, first.end_row,
                new_in_cols, new_values,
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
                    parts = [v.values_2d[..., 0] for v in sorted_vals]
                    new_values = jnp.concatenate(parts, axis=0)
                else:
                    parts = [v.values for v in sorted_vals]
                    new_values = jnp.concatenate(parts, axis=0)
                new_in_cols = []
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
            and all(_linop_matrix_shape(v) == _linop_matrix_shape(vals[0])
                    for v in vals)):
        shape = _linop_matrix_shape(vals[0])
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
                            jnp.broadcast_to(jnp.asarray(v.value), (n_sq,)),
                            n_sq, n_sq,
                        ))
                    elif isinstance(v, Diagonal):
                        ep_vals.append(BEllpack(
                            0, n_sq, (arange_n,), v.values, n_sq, n_sq,
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
    # the same `_linop_matrix_shape` while representing different
    # semantic tensors — e.g. `batch=(n,), out=1, in=m` (col-broadcast,
    # true shape `(n, 1, m)`) vs `batch=(1,), out=n, in=m` (row-broadcast,
    # true shape `(1, n, m)`), both flatten to `(n, m)`. Concatenating
    # their flat BCOOs mis-aligns rows. Guard by requiring all BEllpack
    # operands to share the same `batch_shape`; the fallback dense
    # `reduce(add, …)` then broadcasts correctly via numpy-style rules.
    if kinds <= {ConstantDiagonal, Diagonal, BEllpack, sparse.BCOO}:
        shapes = [_linop_matrix_shape(v) for v in vals]
        if all(s == shapes[0] for s in shapes):
            ep_batch_shapes = {v.batch_shape for v in vals
                               if isinstance(v, BEllpack)}
            if len(ep_batch_shapes) <= 1:
                bcoo_vals = [_to_bcoo(v, n) for v in vals]
                return _bcoo_concat(bcoo_vals, shape=shapes[0])

    # Linear-form adds: a vector-aval-(k,) LinOp is normally stored as a
    # (k, n) matrix, but an aval-() linear form emerges either as a (n,)
    # ndarray (canonical after `_reduce_sum_rule`) or a BEllpack
    # row-vector/BCOO (after `_squeeze_rule`). When the fallback would mix
    # these forms it'd broadcast-sum to a (1, n) 2D ndarray that
    # downstream rules mis-handle. Normalise all linear-form operands to
    # (n,) ndarrays and sum. Loses row-sparsity info; that's fine —
    # this branch only fires for the rare mixed-forms case after the
    # structural matrix paths above already got a chance.
    def _as_linear_form_row(v):
        if isinstance(v, jax.Array) and v.ndim == 1 and v.shape[0] == n:
            return v
        if (isinstance(v, BEllpack) and v.n_batch == 0
                and v.out_size == 1 and v.start_row == 0
                and v.end_row == 1):
            return _to_dense(v, n)[0]
        if isinstance(v, sparse.BCOO) and v.shape == (1, n):
            return v.todense()[0]
        return None
    linear_form_rows = [_as_linear_form_row(v) for v in vals]
    if all(r is not None for r in linear_form_rows):
        return functools.reduce(operator.add, linear_form_rows)

    # Dense fallback: densify everything and sum.
    dense_vals = [_to_dense(v, n) for v in vals]
    return functools.reduce(operator.add, dense_vals)
