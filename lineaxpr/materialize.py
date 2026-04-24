"""Coloring-free Jacobian extraction for linear callables.

Public API (see `lineaxpr/__init__.py`):

* `jacfwd(f)(y)` / `bcoo_jacfwd(f)(y)` — forward-mode Jacobian.
* `jacrev(f)(y)` / `bcoo_jacrev(f)(y)` — reverse-mode Jacobian.
* `hessian(f)(y)` / `bcoo_hessian(f)(y)` — full Hessian.
* `materialize(linear_fn, primal, format='dense'|'bcoo')` — core helper,
  when you already have a linearized callable.
* `sparsify(linear_fn)(seed_linop)` — primitive transform returning a
  LinOp (before format conversion).

All of the above trace `linear_fn` to a jaxpr and walk its equations
with per-primitive rules that propagate structural per-var operators.
The LinOp classes (`ConstantDiagonal`, `Diagonal`, `BEllpack`; see
`_base.py`) let common patterns (scalar · I, vector-scaled I, sparse
banded blocks) avoid materialising intermediate identity matrices; they
are converted to BCOO or dense at the boundary.

## Known gap: non-finite closures in structural paths

Our structural rules assume `0 * x = 0` for any `x`. This is correct
when `x` is finite but wrong for `x ∈ {inf, nan}` (where `0 * inf = nan`).
When a mul/div/add structural rule emits a BEllpack/BCOO that skips
zero positions, it silently drops positions where the closure operand
has `inf`/`nan`. CUTEst objectives don't produce non-finite intermediate
values in practice, so this is a latent correctness gap rather than an
observed issue. A fully-general fix would require reading the closure
for non-finite entries (essentially densifying), losing the structural
optimisation — not worth it unless the gap bites.
"""

from __future__ import annotations

import functools
import operator
import os
import string
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.lax import slicing as _slicing
from jax.experimental import sparse
from jax.extend import core

from ._base import (
    ConstantDiagonal,
    Diagonal,
    BEllpack,
    Identity,
    _ellpack_to_bcoo_batched,
    _resolve_col,
    _to_bcoo,
    _to_dense,
    _traced_shape,
)


# -------------------------- rule registry --------------------------


materialize_rules: dict[core.Primitive, Callable] = {}


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
BELLPACK_DEDUP_LIMIT = int(os.environ.get("LINEAXPR_BELLPACK_DEDUP_LIMIT", "200"))


def _dedup_band_tuple(in_cols, values_per_band, nrows, limit=None):
    """Group bands with identical cols, summing their values.

    Inputs:
      in_cols:          tuple/list of k cols (slice or np.ndarray; traced
                        jnp cols fall back to no-dedup).
      values_per_band:  list of k values arrays (one per band). Each is
                        `(*batch_shape, nrows)` for the k=1 band-slice
                        convention. No Python loop over these arrays —
                        each `values_per_band[i]` was obtained by a
                        single trace-time slice.
      nrows:            resolved nrows for `_resolve_col`.
      limit:            `K ≤ limit` gates the O(K²) scan; defaults to
                        `BELLPACK_DEDUP_LIMIT`.

    Returns `(group_cols: list, group_values: list)`. If no dedup
    happens (every band unique), returns inputs as-is (same length).
    """
    k = len(in_cols)
    if limit is None:
        limit = BELLPACK_DEDUP_LIMIT
    if k <= 1 or k > limit:
        return list(in_cols), list(values_per_band)
    resolved = [_resolve_col(c, nrows) for c in in_cols]
    assigned = [-1] * k
    group_cols: list = []
    group_values: list = []
    for i in range(k):
        if assigned[i] != -1:
            continue
        g = len(group_cols)
        assigned[i] = g
        group_cols.append(resolved[i])
        current_sum = values_per_band[i]
        for j in range(i + 1, k):
            if assigned[j] != -1:
                continue
            if _cols_equal(resolved[i], resolved[j]):
                assigned[j] = g
                current_sum = current_sum + values_per_band[j]
        group_values.append(current_sum)
    return group_cols, group_values


def _input_size(invals, traced):
    """Derive n (walk input dimension) from any traced input.

    Rules that need n should prefer this helper over the `n` arg so they
    can eventually drop it from their signature. See docs/TODO.md — the
    `n`-parameter threading is being phased out as rules are touched.
    """
    for v, t in zip(invals, traced):
        if not t:
            continue
        if isinstance(v, (ConstantDiagonal, Diagonal)):
            return v.n
        if isinstance(v, BEllpack):
            return v.in_size
        if isinstance(v, sparse.BCOO):
            return v.shape[-1]
        # Traced ndarray fallback: last axis is the input coordinate.
        return v.shape[-1]
    raise ValueError("_input_size: no traced input among invals")


# -------------------------- rules --------------------------


def _bcoo_scale_scalar(b: sparse.BCOO, s) -> sparse.BCOO:
    return sparse.BCOO((s * b.data, b.indices), shape=b.shape)


def _bcoo_scale_per_out_row(b: sparse.BCOO, v) -> sparse.BCOO:
    row_idx = b.indices[:, 0]
    v_arr = jnp.asarray(v)
    return sparse.BCOO((b.data * jnp.take(v_arr, row_idx), b.indices), shape=b.shape)


def _bcoo_negate(b: sparse.BCOO) -> sparse.BCOO:
    return sparse.BCOO((-b.data, b.indices), shape=b.shape)


def _mul_rule(invals, traced, n, **params):
    del params
    x, y = invals
    tx, ty = traced
    if not tx and not ty:
        return None
    if not tx:
        scale, traced_op = x, y
    elif not ty:
        scale, traced_op = y, x
    else:
        raise NotImplementedError("mul of two traced operands — not linear")

    scalar_like = not hasattr(scale, "shape") or scale.shape in ((), (1,))
    if scalar_like:
        s = jnp.asarray(scale).reshape(())
        if isinstance(traced_op, (ConstantDiagonal, Diagonal, BEllpack)):
            return traced_op.scale_scalar(s)
        if isinstance(traced_op, sparse.BCOO):
            return _bcoo_scale_scalar(traced_op, s)
        return s * traced_op
    # scale_per_out_row assumes scale has shape that broadcasts cleanly
    # against the op's var_shape (batch_shape + (out_size,)). If scale has
    # extra dims (jaxpr outer-product-like broadcasts), fall back to dense.
    traced_var_shape = _traced_shape(traced_op)
    scale_ok = (
        hasattr(scale, "shape")
        and len(scale.shape) <= len(traced_var_shape)
        and all(
            s in (1, t)
            for s, t in zip(scale.shape[::-1], traced_var_shape[::-1])
        )
    )
    if scale_ok and isinstance(traced_op, (ConstantDiagonal, Diagonal, BEllpack)):
        return traced_op.scale_per_out_row(scale)
    if scale_ok and isinstance(traced_op, sparse.BCOO):
        return _bcoo_scale_per_out_row(traced_op, scale)
    # Batch-expand path: scale broadcasts same-ndim as traced_var_shape
    # but expands one or more size-1 batch axes of the BE (dims where BE
    # has 1 and scale has K > 1). Structurally: new BEllpack with
    # enlarged batch_shape, values broadcast-mul'd. Cols pattern
    # preserved. Motivation: DMN15102LS's `BE(batch=(1,), out=33, k=2) /
    # closure(4643, 33)` should stay structural (nse per batch preserved
    # by mul-sparsity-preservation).
    if (isinstance(traced_op, BEllpack)
            and hasattr(scale, "shape")
            and len(scale.shape) == len(traced_var_shape)
            and all(s == t or t == 1 for s, t in
                    zip(scale.shape, traced_var_shape))
            and scale.shape[-1] == traced_var_shape[-1]):
        new_batch = scale.shape[:-1]
        scale_arr = jnp.asarray(scale)
        if traced_op.k == 1:
            new_values = scale_arr * traced_op.values
        else:
            new_values = scale_arr[..., None] * traced_op.values
        new_in_cols = []
        can_emit = True
        for c in traced_op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif isinstance(c, np.ndarray):
                if c.ndim == 1:
                    new_in_cols.append(c)
                elif c.shape[:len(traced_op.batch_shape)] == traced_op.batch_shape \
                        and all(t == 1 for t in traced_op.batch_shape):
                    new_in_cols.append(
                        np.broadcast_to(c, new_batch + c.shape[-1:]).copy()
                    )
                else:
                    can_emit = False
                    break
            else:
                can_emit = False
                break
        if can_emit:
            return BEllpack(
                start_row=traced_op.start_row, end_row=traced_op.end_row,
                in_cols=tuple(new_in_cols), values=new_values,
                out_size=traced_op.out_size, in_size=traced_op.in_size,
                batch_shape=new_batch,
            )
    # Out-size-broadcast path: scale expands a size-1 out axis to
    # `scale.shape[-1]`. Triggered by the NONMSQRT / EIGENALS-class
    # pattern where an aval-(B, 1) BEllpack (from `bid`-trailing-
    # singleton + slice / reduce_sum chain) multiplies by a (B, S)
    # closure — the primal broadcasts to (B, S) and each of the S new
    # rows is a scaled copy of the traced op's single row. Stays
    # structural: new BEllpack `batch_shape=batch, out_size=S`, k
    # unchanged; for k=1 the value mul is a direct scale*values with
    # no axis insertion; for k>=2 we insert `[..., None]` to broadcast
    # over the band axis; cols broadcast statically across new out.
    #
    # Previously gated on `traced.k >= 2` after an EIGENALS regression
    # measurement (~30ms, 97→126ms). That gate was set before
    # 0c/0d/0l — downstream rules densified a k=1 broadcast-expand BE
    # almost immediately via `add_any(..., dense_closure)`. With 0d's
    # structural select_n / gather paths now consuming the output
    # without densifying, the earlier regression no longer applies.
    if (isinstance(traced_op, BEllpack)
            and traced_op.k >= 1
            and traced_op.out_size == 1
            and traced_op.start_row == 0 and traced_op.end_row == 1
            and hasattr(scale, "shape")
            and len(scale.shape) == traced_op.n_batch + 1
            and all(s in (1, t)
                    for s, t in zip(scale.shape[:-1], traced_op.batch_shape))
            and int(scale.shape[-1]) >= 1):
        new_out = int(scale.shape[-1])
        scale_arr = jnp.asarray(scale)
        if traced_op.k == 1:
            # traced values (*batch, 1). scale (*batch, new_out).
            # Result (*batch, new_out).
            new_values = scale_arr * traced_op.values
        else:
            # traced values (*batch, 1, k). Insert new_out axis then mul.
            new_values = scale_arr[..., None] * traced_op.values
        new_in_cols = []
        for c in traced_op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif isinstance(c, np.ndarray):
                if c.ndim == traced_op.n_batch + 1:  # (*batch, 1) per-batch
                    new_in_cols.append(
                        np.broadcast_to(
                            c, traced_op.batch_shape + (new_out,)
                        ).copy()
                    )
                else:
                    new_in_cols.append(c)
            else:
                # Traced cols — would need jnp broadcast; keep simple by
                # falling through to dense. Rare.
                new_in_cols = None
                break
        if new_in_cols is not None:
            return BEllpack(
                start_row=0, end_row=new_out,
                in_cols=tuple(new_in_cols), values=new_values,
                out_size=new_out, in_size=traced_op.in_size,
                batch_shape=traced_op.batch_shape,
            )
    dense = _to_dense(traced_op, n)
    return scale[..., None] * dense

materialize_rules[lax.mul_p] = _mul_rule

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
    """Structural equality test for BEllpack ColArr (slice / np.ndarray).

    Conservative: returns False for traced jnp arrays (can't compare at
    trace time) and for heterogeneous pairs (slice vs array). That's
    fine — the caller falls back to band concat, which is correct just
    wider than necessary.
    """
    if isinstance(a, slice) and isinstance(b, slice):
        return a.start == b.start and a.stop == b.stop and a.step == b.step
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b)
    return False


def _col_batch_slice(col, batch_idx):
    """Select one element from a batched ColArr along its leading axis.

    - `slice` (shared across batches): pass through unchanged.
    - `np.ndarray` / `jnp.ndarray`: if ndim > 1, index the leading axis;
      else treat as shared (1D cols broadcast across batches).
    """
    if isinstance(col, slice):
        return col
    if col.ndim >= 2:
        return col[batch_idx]
    return col


def _bellpack_unbatch(bep):
    """Split a BEllpack with n_batch == 1 into a tuple of unbatched Ellpacks.

    Each slice shares `(start_row, end_row, out_size, in_size)` and differs
    in per-batch `in_cols` and `values` rows.
    """
    assert bep.n_batch >= 1, "use only when n_batch > 0"
    # Flatten batch_shape to a single leading axis of size B.
    B = bep.batch_shape[0]
    # For n_batch > 1 we'd need to iterate over the product; leave as TODO.
    assert bep.n_batch == 1, (
        f"_bellpack_unbatch only supports n_batch=1 currently, got {bep.n_batch}"
    )
    result = []
    for b in range(B):
        in_cols_b = tuple(_col_batch_slice(c, b) for c in bep.in_cols)
        values_b = bep.values[b]
        result.append(BEllpack(
            start_row=bep.start_row, end_row=bep.end_row,
            in_cols=in_cols_b, values=values_b,
            out_size=bep.out_size, in_size=bep.in_size,
            batch_shape=(),
        ))
    return tuple(result)


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
    out = []
    for axis in range(max_ndim):
        dims = [p[axis] for p in padded]
        non1 = {d for d in dims if d != 1}
        if len(non1) > 1:
            return None  # conflicting non-1 dims
        out.append(max(dims))
    return tuple(out)


def _broadcast_be_to_batch(be, target_batch_shape):
    """Expand a BEllpack to a larger broadcast-compatible batch_shape.

    Preserves cols/values sparsity pattern — just broadcasts along
    size-1 batch axes. Same mechanic as `_mul_rule`'s batch-expand
    (ac3c7a6): `mul(BE, dense)` is sparsity-preserving, so replicating
    the pattern over a new batch axis stays structural.

    Returns None if broadcast isn't cleanly expressible (e.g. batched
    cols with non-broadcast-compatible shape) — caller falls back.
    """
    if be.batch_shape == target_batch_shape:
        return be
    # Pad BE's batch_shape with leading 1s to match target's ndim.
    src_ndim = len(be.batch_shape)
    tgt_ndim = len(target_batch_shape)
    if src_ndim > tgt_ndim:
        return None
    pad_ndim = tgt_ndim - src_ndim
    # Values: broadcast leading (batch) axes to target. Trailing axes
    # (nrows, [k]) unchanged.
    val_pad_shape = (1,) * pad_ndim + be.values.shape
    val_target_shape = target_batch_shape + be.values.shape[src_ndim:]
    new_values = jnp.broadcast_to(
        be.values.reshape(val_pad_shape), val_target_shape
    )
    # Cols: 1D shared cols pass through unchanged. N-D cols need
    # broadcasting — supported only when all batch axes were size-1.
    new_in_cols = []
    for c in be.in_cols:
        if isinstance(c, slice):
            new_in_cols.append(c)
        elif isinstance(c, np.ndarray):
            if c.ndim == 1:
                new_in_cols.append(c)  # shared across batch
            elif c.ndim == src_ndim + 1:
                # Per-batch cols: broadcast along new batch axes.
                pad = (1,) * pad_ndim + c.shape
                new_in_cols.append(
                    np.broadcast_to(c.reshape(pad),
                                    target_batch_shape + c.shape[-1:]).copy()
                )
            else:
                return None
        else:
            # Traced cols — skip for now to keep the change small.
            return None
    return BEllpack(
        start_row=be.start_row, end_row=be.end_row,
        in_cols=tuple(new_in_cols), values=new_values,
        out_size=be.out_size, in_size=be.in_size,
        batch_shape=target_batch_shape,
    )


def _densify_if_wider_than_dense(op, n):
    """Densify a BEllpack whose k ≥ in_size — no storage win over dense.

    When k ≥ in_size, the BE stores one band per input column (or more),
    and `values.shape = (*batch, out, k)` already equals dense storage
    `(*batch, out, in_size)`. Any downstream rule that needs to compute
    on it pays BE's bookkeeping overhead for no sparsity benefit. This
    helper is called at emission points in rules that can grow k
    unboundedly (reduce_sum out-axis, add band-widening) to prevent BE
    from carrying effectively-dense state through the rest of the walk.

    Motivation: DMN15102LS regressed 1.25× (2026-04-23 sweep) because
    `reduce_sum` out-axis emitted `k=in_size=66` BE that then flowed
    through 9 downstream adds and a pad, where a dense path would have
    been a single scatter-add.
    """
    if isinstance(op, BEllpack) and op.k >= op.in_size:
        return _to_dense(op, n)
    return op


def _bcoo_concat(bcoo_vals, shape):
    """Concatenate a list of BCOOs (matching shape) entry-wise.

    For unbatched BCOOs the nse axis is 0 in both `data` and `indices`.
    For batched BCOOs (`n_batch > 0`), the nse axis follows the batch
    dims — it's `n_batch` in `data` and `indices`. We concat along that
    axis so batches stay aligned and each batch's entries are
    accumulated.

    Structural duplicates (same index appearing in multiple operands) are
    resolved at densification via scatter-add, matching the semantics of
    `lax.add_any` on summed entries.
    """
    n_batch = bcoo_vals[0].n_batch
    # Indices path: if every BCOO's indices are concretely np-backed
    # (the common case — `_ellpack_to_bcoo` builds them via pure np),
    # concatenate in numpy so the result stays a single compile-time
    # constant. Using `jnp.concatenate` on K constant DeviceArrays
    # emits a traced concat that XLA "compresses" into a runtime
    # `iota + gather` pattern when it detects tile-like regularity
    # — paying per-call decompression cost that dominates on NONCVX /
    # LUKSAN-class problems. Fall back to `jnp.concatenate` if any
    # operand is a tracer.
    try:
        indices = np.concatenate(
            [np.asarray(v.indices) for v in bcoo_vals], axis=n_batch
        )
    except (jax.errors.TracerArrayConversionError, TypeError):
        indices = jnp.concatenate(
            [v.indices for v in bcoo_vals], axis=n_batch
        )
    data = jnp.concatenate([v.data for v in bcoo_vals], axis=n_batch)
    return sparse.BCOO((data, indices), shape=shape)


def _tile_1row_bellpack(ep, target_rows):
    """Broadcast a BEllpack row-vector (shape (1, n), holding a sparse
    linear form `c`) to (target_rows, n) by tiling its single row.
    Each band's in_cols / values broadcast from length-1 to
    length-target_rows. Storage stays O(target_rows · k) where k is
    the original BEllpack's band count — so as long as the linear
    form has few nonzeros (k small), we avoid n² blow-up. Dense rows
    (k ≈ n) should go through BCOO instead; BEllpack row-vector at large k
    wastes per-band np.ndarray overhead.

    Used by `_add_rule` to fold a linear form (BEllpack row-vector from
    `_squeeze_rule`) into a broadcast-add with a (target_rows, n)
    matrix LinOp — the structural analogue of `numpy` broadcasting
    `(n,) + (m, n)`."""
    assert ep.n_batch == 0 and ep.out_size == 1
    assert ep.start_row == 0 and ep.end_row == 1
    k = ep.k
    new_in_cols = []
    for col in ep.in_cols:
        if isinstance(col, slice):
            new_in_cols.append(col)
        elif isinstance(col, np.ndarray):
            # col has shape (1,); broadcast to (target_rows,).
            new_in_cols.append(np.broadcast_to(col, (target_rows,)))
        else:
            new_in_cols.append(jnp.broadcast_to(col, (target_rows,)))
    if k == 1:
        new_values = jnp.broadcast_to(ep.values, (target_rows,))
    else:
        new_values = jnp.broadcast_to(ep.values, (target_rows, k))
    return BEllpack(
        start_row=0, end_row=target_rows,
        in_cols=tuple(new_in_cols), values=new_values,
        out_size=target_rows, in_size=ep.in_size,
    )


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
            # through to the naive widen below. Targets FREUROTH /
            # CRAGGLVY / WOODS / CHAINWOO / SROSENBR where the
            # chain-rule walk derives the same (row, col) structure
            # via multiple paths; measured nse reductions of 1.5–5.5×.
            K_total = sum(v.k for v in vals)
            n_batch = first.n_batch
            band_axis = n_batch + 1
            if K_total <= BELLPACK_DEDUP_LIMIT:
                # Hash-based grouping via `col.tobytes()` keys. Same
                # algorithm as `_reduce_sum_rule`'s dedup — consistent
                # approach, O(K) at trace time. Empirically saves ~17%
                # compile time on NONMSQRT vs O(K²) `np.array_equal`
                # (285ms → 235ms); other problems flat. Values slice:
                # `v.values` for k=1, `v.values[..., b]` for k>=2 —
                # same convention as the widen path below; tuple
                # iteration over in_cols, values passed whole (per
                # CLAUDE.md "never loop over arrays").
                def _col_key(c):
                    if isinstance(c, np.ndarray):
                        return ("np", c.shape, c.tobytes())
                    if isinstance(c, slice):
                        return ("slc", c.start, c.stop, c.step)
                    return ("id", id(c))  # traced — won't group
                group_cols: list = []
                group_values: list = []
                key_to_group: dict = {}
                for v in vals:
                    for b in range(v.k):
                        c = _resolve_col(v.in_cols[b], v.nrows)
                        vals_b = v.values if v.k == 1 else v.values[..., b]
                        k_ = _col_key(c)
                        g = key_to_group.get(k_)
                        if g is None:
                            key_to_group[k_] = len(group_cols)
                            group_cols.append(c)
                            group_values.append(vals_b)
                        else:
                            group_values[g] = group_values[g] + vals_b
                new_k = len(group_cols)
                if new_k < K_total:
                    if new_k == 1:
                        new_values = group_values[0]
                    else:
                        new_values = jnp.stack(group_values, axis=-1)
                    return _densify_if_wider_than_dense(BEllpack(
                        first.start_row, first.end_row,
                        tuple(group_cols), new_values,
                        first.out_size, first.in_size,
                        batch_shape=first.batch_shape,
                    ), n)
            # Different cols: widen bands. Values shape for n_batch=0 is
            # (nrows,) for k=1 or (nrows, k) for k>=2; concat along the
            # band axis (axis=1, i.e. the k dim). For batched values
            # (n_batch>0) the band axis is n_batch+1.
            new_in_cols = tuple(c for v in vals for c in v.in_cols)
            if n_batch == 0:
                # Preserve the exact unbatched HLO — previously measured
                # to differ from jnp.expand_dims form on small problems.
                parts = [v.values if v.values.ndim == 2 else v.values[:, None]
                         for v in vals]
                new_values = jnp.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
            else:
                parts = [v.values if v.values.ndim >= band_axis + 1
                         else jnp.expand_dims(v.values, band_axis)
                         for v in vals]
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
                    parts = [v.values if v.values.ndim == 1 else v.values[:, 0]
                             for v in sorted_vals]
                    new_values = jnp.concatenate(parts, axis=0)
                else:
                    parts = [v.values for v in sorted_vals]
                    new_values = jnp.concatenate(parts, axis=0)
                new_in_cols = []
                for b in range(k_new):
                    band_rows = [_resolve_col(v.in_cols[b], v.nrows)
                                 for v in sorted_vals]
                    if all(isinstance(c, np.ndarray) for c in band_rows):
                        new_in_cols.append(np.concatenate(band_rows, axis=0))
                    else:
                        new_in_cols.append(jnp.concatenate(
                            [jnp.asarray(c) for c in band_rows], axis=0))
                return _densify_if_wider_than_dense(BEllpack(
                    sorted_vals[0].start_row, sorted_vals[-1].end_row,
                    tuple(new_in_cols), new_values,
                    first.out_size, first.in_size,
                ), n)

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


materialize_rules[lax.add_p] = _add_rule
try:
    from jax._src.ad_util import add_jaxvals_p

    materialize_rules[add_jaxvals_p] = _add_rule
except ImportError:
    pass


def _identity_rule(invals, traced, n, **params):
    """For primitives that don't change value (convert_element_type, copy)."""
    del params
    (op,) = invals
    (t,) = traced
    return op if t else None


materialize_rules[lax.convert_element_type_p] = _identity_rule
materialize_rules[lax.copy_p] = _identity_rule


def _neg_rule(invals, traced, n, **params):
    del params, n
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    if isinstance(op, (ConstantDiagonal, Diagonal, BEllpack)):
        return op.negate()
    if isinstance(op, sparse.BCOO):
        return _bcoo_negate(op)
    return -op

materialize_rules[lax.neg_p] = _neg_rule

def _sub_rule(invals, traced, n, **params):
    """a - b = a + (-b). Reuse add via negating the second operand if traced."""
    a, b = invals
    ta, tb = traced
    if not ta and not tb:
        return None
    if not tb:
        # a is traced, b is closure. a - b is still linear with same A as a.
        return a if ta else None
    if not ta:
        # -b only. Negate via _neg_rule equivalent.
        return _neg_rule([b], [True], n)
    # both traced
    neg_b = _neg_rule([b], [True], n)
    return _add_rule([a, neg_b], [True, True], n)

materialize_rules[lax.sub_p] = _sub_rule

def _be_dot_closure_matrix(be: BEllpack, M, c_be: int, c_M: int,
                           traced_is_first: bool):
    """Structural `BEllpack ⊗ closure_matrix` contract. Returns `None`
    when dense would be no larger (gate `k_old * A >= in_size`), or
    when the contract is on an earlier-batch axis (unsupported — would
    require reshuffling every per-band col tensor).

    Band enumeration (β, a) flattens to `β * A + a`; the einsum output
    order `(remaining, J, [K], contract)` is chosen so a trailing
    reshape produces that layout directly.
    """
    n_batch = be.n_batch
    aval_shape = (*be.batch_shape, be.out_size)
    k_old, in_size = be.k, be.in_size
    if not (n_batch - 1 <= c_be <= n_batch):
        return None
    A = aval_shape[c_be]
    if A != M.shape[c_M] or k_old * A >= in_size:
        return None

    B = M.shape[1 - c_M]
    M_AB = M if c_M == 0 else M.T
    new_aval = aval_shape[:c_be] + aval_shape[c_be + 1:] + (B,)
    new_batch, new_out = new_aval[:-1], new_aval[-1]

    new_in_cols = tuple(
        _bcast(c_full.take(a, axis=c_be)[..., None], new_aval)
        for c_full in (_resolve_full(c, be.nrows, be.batch_shape)
                       for c in be.in_cols)
        for a in range(A)
    )

    # einsum: one letter per aval axis, K for optional k, J for M's
    # free axis. Trailing (K, contract) reshape gives the β*A+a layout.
    letters = string.ascii_lowercase[:len(aval_shape)]
    assert len(letters) == len(aval_shape), "aval rank exceeds letter pool"
    ctr = letters[c_be]
    remaining = letters[:c_be] + letters[c_be + 1:]
    k_let = "K" if k_old > 1 else ""
    eq = f"{letters}{k_let},{ctr}J->{remaining}J{k_let}{ctr}"
    new_vals = jnp.einsum(eq, be.values, M_AB)
    if k_old > 1:
        new_vals = new_vals.reshape(new_aval + (k_old * A,))

    out_be = BEllpack(
        start_row=0, end_row=new_out,
        in_cols=new_in_cols, values=new_vals,
        out_size=new_out, in_size=in_size, batch_shape=new_batch,
    )
    if not traced_is_first:
        # dot_general(closure, BE) aval is (*remaining_M, *remaining_BE);
        # BE's out axis is structurally last so we permute batch↔out.
        # Cheap: reorders the in_cols tuple + one values transpose.
        out_be = out_be.transpose((n_batch,) + tuple(range(n_batch)))
    return out_be


def _bcast(arr, shape):
    return (np if isinstance(arr, np.ndarray) else jnp).broadcast_to(arr, shape)


def _resolve_full(c, nrows, batch_shape):
    """Resolve a ColArr (slice | 1D | N-D) to shape `(*batch, nrows)`."""
    if isinstance(c, slice):
        c = _resolve_col(c, nrows)
    if c.ndim == 1:
        return _bcast(c, batch_shape + (nrows,))
    return c


def _dot_general_rule(invals, traced, n, **params):
    x, y = invals
    tx, ty = traced
    (contract, batch) = params["dimension_numbers"]
    (cx, cy) = contract
    if batch != ((), ()):
        raise NotImplementedError("dot_general with batch dims not yet handled")

    if tx and not ty:
        traced_op, c_tr, M, c_M = x, list(cx), y, list(cy)
        traced_is_first = True
    elif ty and not tx:
        traced_op, c_tr, M, c_M = y, list(cy), x, list(cx)
        traced_is_first = False
    else:
        raise NotImplementedError("dot_general of two traced operands")
    traced_shape = _traced_shape(traced_op)

    if len(c_tr) == 0 and len(c_M) == 0 and M.shape == ():
        if isinstance(traced_op, ConstantDiagonal):
            return ConstantDiagonal(traced_op.n, M * traced_op.value)
        return M * traced_op
    if len(c_tr) == 0 and len(c_M) == 0:
        # Outer product. BE's trailing `n` axis stays last.
        dense = _to_dense(traced_op, n)
        t_rank, m_rank = len(traced_shape), M.ndim
        if traced_is_first:
            # (*t, n) × (*m,) → (*t, *m, n)
            d = dense.reshape(traced_shape + (1,) * m_rank + dense.shape[-1:])
            return d * M[..., None]
        # (*m,) × (*t, n) → (*m, *t, n)
        return M.reshape(M.shape + (1,) * (t_rank + 1)) * dense

    if isinstance(traced_op, ConstantDiagonal):
        remaining = [a for a in range(M.ndim) if a not in c_M]
        tensor = lax.transpose(M, remaining + c_M)
        return traced_op.value * tensor

    # Structural BEllpack × closure-matrix path (see
    # `_be_dot_closure_matrix` for the gate: `k_new >= in_size`
    # falls through to dense).
    if (isinstance(traced_op, BEllpack)
            and M.ndim == 2
            and len(c_tr) == 1 and len(c_M) == 1
            and traced_op.start_row == 0
            and traced_op.end_row == traced_op.out_size):
        be_result = _be_dot_closure_matrix(
            traced_op, M, c_tr[0], c_M[0], traced_is_first,
        )
        if be_result is not None:
            return be_result

    dense = _to_dense(traced_op, n)
    if traced_is_first:
        out = lax.dot_general(
            dense, M, ((tuple(c_tr), tuple(c_M)), ((), ()))
        )
        # dense's trailing `n` axis is never contracted; dot_general's
        # output places it at `len(traced_shape) - len(c_tr)`. Move to end.
        return jnp.moveaxis(out, len(traced_shape) - len(c_tr), -1)
    return lax.dot_general(M, dense, ((tuple(c_M), tuple(c_tr)), ((), ())))

materialize_rules[lax.dot_general_p] = _dot_general_rule

def _slice_rule(invals, traced, n, **params):
    (operand,) = invals
    (to,) = traced
    if not to:
        return None
    starts = tuple(int(s) for s in params["start_indices"])
    limits = tuple(int(l) for l in params["limit_indices"])
    strides_p = params.get("strides")
    strides = tuple(int(s) for s in strides_p) if strides_p else (1,) * len(starts)

    # Structural fast path — unit-stride 1D (hot path, unchanged HLO).
    if (len(starts) == 1 and strides == (1,)
            and isinstance(operand, (ConstantDiagonal, Diagonal, BEllpack, sparse.BCOO))):
        s, e = starts[0], limits[0]
        k = e - s
        if isinstance(operand, ConstantDiagonal):
            values_b = jnp.broadcast_to(jnp.asarray(operand.value), (k,))
            return BEllpack(
                start_row=0, end_row=k,
                in_cols=(np.arange(s, e),),
                values=values_b,
                out_size=k, in_size=operand.n,
            )
        if isinstance(operand, Diagonal):
            return BEllpack(
                start_row=0, end_row=k,
                in_cols=(np.arange(s, e),),
                values=operand.values[s:e],
                out_size=k, in_size=operand.n,
            )
        if isinstance(operand, BEllpack):
            return operand.pad_rows(-s, -(operand.out_size - e))
        rows = operand.indices[:, 0]
        mask = (rows >= s) & (rows < e)
        new_data = operand.data * mask
        new_rows = rows - s
        new_indices = jnp.stack([new_rows, operand.indices[:, 1]], axis=1)
        return sparse.BCOO((new_data, new_indices), shape=(k, operand.shape[1]))

    # Strided 1D slice on ConstantDiagonal/Diagonal — emit a BEllpack
    # whose in_cols carry the strided index pattern. Used by RAYBENDL's
    # `y[::2]` pattern. BEllpack/BCOO with stride > 1 fall through to
    # dense (pad_rows can't express strided row selection).
    if (len(starts) == 1
            and isinstance(operand, (ConstantDiagonal, Diagonal))):
        s, e = starts[0], limits[0]
        stride = strides[0]
        cols = np.arange(s, e, stride)
        k_out = len(cols)
        if isinstance(operand, ConstantDiagonal):
            values_b = jnp.broadcast_to(jnp.asarray(operand.value), (k_out,))
            return BEllpack(
                start_row=0, end_row=k_out,
                in_cols=(cols,),
                values=values_b,
                out_size=k_out, in_size=operand.n,
            )
        return BEllpack(
            start_row=0, end_row=k_out,
            in_cols=(cols,),
            values=operand.values[s:e:stride],
            out_size=k_out, in_size=operand.n,
        )

    # Structural path: n-D unit-stride slice on a batched BEllpack where
    # the slice axes cover `batch_shape + (out_size,)`. Slice the batch
    # dims on `values` and per-batch `in_cols` via basic indexing, then
    # slice the out_size axis via `pad_rows`. Triggers on problems that
    # reshape(Identity) to n-D then slice (e.g. MSQRT, SPARSINE-like,
    # DRCAV1LQ's 2D stencil).
    if (isinstance(operand, BEllpack) and operand.n_batch > 0
            and len(starts) == operand.n_batch + 1
            and all(st == 1 for st in strides)):
        batch_slicer = tuple(slice(int(s), int(e))
                             for s, e in zip(starts[:-1], limits[:-1]))
        out_start, out_limit = int(starts[-1]), int(limits[-1])
        # values shape (*batch, nrows) for k=1 or (*batch, nrows, k) for k>=2.
        tail = (slice(None),) * (operand.values.ndim - operand.n_batch)
        new_values = operand.values[batch_slicer + tail]
        new_in_cols = []
        for c in operand.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif hasattr(c, "ndim") and c.ndim > 1:
                new_in_cols.append(c[batch_slicer + (slice(None),)])
            else:
                new_in_cols.append(c)  # 1D shared cols
        new_batch = tuple(b.stop - b.start for b in batch_slicer)
        sliced = BEllpack(
            operand.start_row, operand.end_row,
            tuple(new_in_cols), new_values,
            operand.out_size, operand.in_size,
            batch_shape=new_batch,
        )
        return sliced.pad_rows(-out_start, -(operand.out_size - out_limit))

    # Fallback: densify and slice along output (non-input) axes; preserve the
    # trailing input-coordinate axis with start=0, limit=n, stride=1.
    dense = _to_dense(operand, n)
    s_full = starts + (0,)
    l_full = limits + (n,)
    str_full = strides + (1,)
    return lax.slice(dense, s_full, l_full, str_full)

materialize_rules[lax.slice_p] = _slice_rule

def _pad_rule(invals, traced, n, **params):
    operand, padding_value = invals
    to, tp = traced
    if tp:
        raise NotImplementedError("pad with traced padding_value")
    if not to:
        return None
    if hasattr(padding_value, "shape") and padding_value.shape != ():
        raise NotImplementedError("pad with non-scalar padding_value")
    config = params["padding_config"]
    before, after, interior = config[0] if len(config) >= 1 else (0, 0, 0)
    before, after = int(before), int(after)
    interior = int(interior)
    if isinstance(operand, BEllpack) and len(config) == 1 and interior == 0:
        return operand.pad_rows(before, after)
    if isinstance(operand, sparse.BCOO) and len(config) == 1 and interior == 0:
        out_size = operand.shape[0] + before + after
        new_rows = operand.indices[:, 0] + before
        new_indices = jnp.stack([new_rows, operand.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (operand.data, new_indices), shape=(out_size, operand.shape[1])
        )
    # Structural n-D zero-interior pad on a batched BEllpack where pad
    # axes cover `batch_shape + (out_size,)`. Inner (out_size) axis:
    # `pad_rows(before, after)` shifts start_row / end_row. Outer (batch)
    # axes: zero-pad `values` on the batch axes and extend `batch_shape`.
    # Per-batch `in_cols` get padded with `-1` sentinels at new batch
    # slots (values are 0 there so col doesn't matter for correctness;
    # BCOO conversion filters). Used by DRCAV1LQ/DRCAV2LQ to pad each
    # 13-point stencil window back to the full 2D grid.
    if (isinstance(operand, BEllpack) and operand.n_batch > 0
            and len(config) == operand.n_batch + 1
            and all(int(c[2]) == 0 for c in config)):
        batch_pads = tuple((int(c[0]), int(c[1])) for c in config[:-1])
        out_before, out_after = int(config[-1][0]), int(config[-1][1])
        new_batch_shape = tuple(
            b + s + a for (b, a), s in zip(batch_pads, operand.batch_shape)
        )
        tail_pad = ((0, 0),) * (operand.values.ndim - operand.n_batch)
        new_values = jnp.pad(operand.values, batch_pads + tail_pad)
        new_in_cols = []
        for c in operand.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif hasattr(c, "ndim") and c.ndim > 1:
                pad_cfg = batch_pads + ((0, 0),)
                if isinstance(c, np.ndarray):
                    new_in_cols.append(np.pad(c, pad_cfg, constant_values=-1))
                else:
                    new_in_cols.append(jnp.pad(c, pad_cfg, constant_values=-1))
            else:
                new_in_cols.append(c)  # 1D shared cols — broadcast OK
        padded_batch = BEllpack(
            operand.start_row, operand.end_row,
            tuple(new_in_cols), new_values,
            operand.out_size, operand.in_size,
            batch_shape=new_batch_shape,
        )
        return padded_batch.pad_rows(out_before, out_after)

    if len(config) == 1 and interior > 0 and isinstance(operand, (sparse.BCOO, BEllpack)):
        # Interior padding inserts `interior` zeros between each original
        # entry — the adjoint of a strided slice. Promote BEllpack to
        # BCOO, then `new_row = old_row * (interior + 1) + before`.
        if isinstance(operand, BEllpack):
            from ._base import _to_bcoo
            operand = _to_bcoo(operand, n)
        step = interior + 1
        old_size = operand.shape[0]
        out_size = old_size + before + after + interior * max(old_size - 1, 0)
        new_rows = operand.indices[:, 0] * step + before
        new_indices = jnp.stack([new_rows, operand.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (operand.data, new_indices), shape=(out_size, operand.shape[1])
        )
    # Dense fallback: pad along output axes (input axis untouched).
    dense = _to_dense(operand, n)
    full_config = tuple((int(b), int(a), int(i)) for (b, a, i) in config) + ((0, 0, 0),)
    return lax.pad(dense, jnp.asarray(0.0, dtype=dense.dtype), full_config)

materialize_rules[lax.pad_p] = _pad_rule

def _squeeze_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    dimensions = params["dimensions"]
    # For 1D structural forms, the var is shape (n,) so there's nothing to
    # squeeze. Only fail if some other rule produced a higher-dim form.
    if isinstance(op, (ConstantDiagonal, Diagonal)):
        if not dimensions:
            return op
        raise NotImplementedError(f"squeeze on diag with dims {dimensions}")
    if isinstance(op, BEllpack) and op.n_batch == 0 and dimensions == (0,) \
            and op.out_size == 1 and op.start_row == 0 and op.end_row == 1:
        # BEllpack row-vector squeezed along its row axis: the result has
        # aval () — a *linear form* (1×n row vector, the Jacobian of a
        # scalar-aval variable w.r.t. the n-dim input). Keep it as a
        # BEllpack row-vector (shape (1, in_size)) so downstream broadcast-
        # add in `_add_rule` can tile the sparse row cheaply — instead
        # of the old densify-to-(n,)-ndarray path which forced
        # subsequent linear_form + vector adds to materialise (n, n)
        # dense. Only valid when the row is sparse (few bands); dense
        # rows should go via BCOO, not a k-many-bands BEllpack.
        return op
    # Batched BEllpack with out_size=1 squeezing the out axis: flatten
    # batch axes into a new unbatched `out_size = prod(batch)`. Each
    # original batch's single output row becomes a row of the new
    # unbatched BEllpack. Attacks WOODS (`reshape → slice → squeeze` on
    # the `(1000, 4)` variable grid, 123× vs asdex).
    if (isinstance(op, BEllpack) and op.n_batch >= 1
            and op.out_size == 1
            and op.start_row == 0 and op.end_row == 1
            and dimensions == (op.n_batch,)):
        B = int(np.prod(op.batch_shape))
        # Values: k=1 is (*batch, 1) → (B,); k>=2 is (*batch, 1, k) → (B, k).
        if op.k == 1:
            new_values = op.values.reshape(B)
        else:
            new_values = op.values.reshape(B, op.k)
        new_in_cols = []
        ok = True
        for c in op.in_cols:
            if isinstance(c, slice):
                rs = np.arange(c.start or 0, c.stop or 1, c.step or 1)
                # nrows=1 so slice yields 1 col; tile to (B,).
                if len(rs) == 1:
                    new_in_cols.append(np.broadcast_to(rs, (B,)).copy())
                else:
                    ok = False; break
            elif isinstance(c, np.ndarray):
                if c.ndim == op.n_batch + 1:  # (*batch, 1) per-batch
                    new_in_cols.append(c.reshape(B))
                elif c.ndim == 1 and c.shape[0] == 1:  # shared (1,)
                    new_in_cols.append(np.broadcast_to(c, (B,)).copy())
                elif c.ndim == 1 and c.shape[0] == B:
                    new_in_cols.append(c)
                else:
                    ok = False; break
            else:  # traced cols
                ok = False; break
        if ok:
            return BEllpack(
                start_row=0, end_row=B,
                in_cols=tuple(new_in_cols), values=new_values,
                out_size=B, in_size=op.in_size,
            )
    if isinstance(op, (BEllpack, sparse.BCOO)):
        # Densify (sparse → (out_size, in_size)) then squeeze leading axes.
        return lax.squeeze(_to_dense(op, n), dimensions)
    # Dense: squeeze the specified axes (always output axes, never the last
    # input-coordinate axis).
    return lax.squeeze(op, dimensions)

materialize_rules[lax.squeeze_p] = _squeeze_rule

def _rev_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    dimensions = params["dimensions"]
    if isinstance(op, ConstantDiagonal):
        return op  # constant under axis-reversal
    if isinstance(op, Diagonal):
        # Reverse the diagonal values along axis 0.
        if dimensions == (0,):
            return Diagonal(op.values[::-1])
        return op
    # BCOO / dense: densify and reverse.
    dense = _to_dense(op, n)
    return lax.rev(dense, dimensions)

materialize_rules[lax.rev_p] = _rev_rule

# TODO(structural): reshape/broadcast_in_dim/reduce_sum/cumsum/split/transpose
# all densify unconditionally. Structural alternatives exist (e.g. transpose on
# BCOO swaps index columns; reduce_sum on a sparse axis drops it). Deferred —
# see docs/RESEARCH_NOTES.md §10 "Densifying vs structure-preserving" audit.
def _reshape_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    new_sizes = tuple(int(s) for s in params["new_sizes"])
    # Structural path: splitting the output axis of a (Constant)Diagonal
    # into batch-shape + nrows encodes the reshape permutation into a
    # batched BEllpack with k=1. `in_cols[*batch_idx, r] =
    # flat_index(*batch_idx, r)` so that densifying recovers
    # `lax.reshape(diag.todense(), (*new_sizes, n))` bit-exactly. Storage
    # is O(n) — no (67, 67, 4489)-class dense intermediate. Triggered by
    # 2D-stencil problems (DRCAV1LQ etc.) that start with
    # `reshape(Identity, (sqrt_n, sqrt_n))`. Downstream rules that don't
    # yet support batched BEllpack will still densify at their own site;
    # this keeps the walk structural at least through reshape.
    if (isinstance(op, (ConstantDiagonal, Diagonal))
            and len(new_sizes) >= 2
            and int(np.prod(new_sizes)) == op.n):
        batch_shape = new_sizes[:-1]
        nrows = new_sizes[-1]
        flat_idx = np.arange(op.n).reshape(new_sizes)
        if isinstance(op, ConstantDiagonal):
            values = jnp.broadcast_to(jnp.asarray(op.value), new_sizes)
        else:
            values = op.values.reshape(new_sizes)
        return BEllpack(
            start_row=0, end_row=nrows,
            in_cols=(flat_idx,), values=values,
            out_size=nrows, in_size=op.n,
            batch_shape=batch_shape,
        )
    # Pass-through: unbatched BCOO whose shape already equals the
    # target `(*new_sizes, n)`. Rare post-0j (batched BEllpacks now
    # convert to batched BCOO, not unbatched-flat), but still fires
    # when an already-unbatched BCOO reaches a final aval-flatten
    # reshape that's structurally a no-op.
    if (isinstance(op, sparse.BCOO) and op.n_batch == 0
            and len(new_sizes) == 1
            and op.shape == (int(new_sizes[0]), op.shape[-1])):
        return op

    # Structural path: batched BEllpack → unbatched BEllpack when the
    # reshape fully flattens the leading (batch + out) axes into one
    # aval dimension. Stays in BE form so downstream ops (`mul`,
    # band-widen `add`, `pad`) can carry BE-specific fast paths.
    # Values reshape `(*batch, nrows) → (B*O,)` for k=1 or `(*batch,
    # nrows, k) → (B*O, k)` for k>=2. Per-band cols broadcast to
    # `(*batch, nrows)` if 1D, then reshape to `(B*O,)`. Final BCOO
    # conversion (if needed) happens at the public-API boundary via
    # the now-vectorized `_ellpack_to_bcoo`. Target must be rank 1
    # and the total equal `prod(batch) * out_size`; otherwise fall
    # through.
    if (isinstance(op, BEllpack) and op.n_batch >= 1
            and len(new_sizes) == 1
            and int(np.prod(op.batch_shape)) * op.out_size
                == int(new_sizes[0])):
        prod_b = int(np.prod(op.batch_shape))
        total = prod_b * op.out_size
        if op.k == 1:
            new_values = op.values.reshape(total)
        else:
            new_values = op.values.reshape(total, op.k)
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                rs = np.arange(c.start or 0, c.stop or op.nrows,
                               c.step or 1)
                c_full = np.broadcast_to(
                    rs, op.batch_shape + (op.nrows,)
                )
                new_in_cols.append(c_full.reshape(total))
            elif isinstance(c, np.ndarray):
                if c.ndim == 1:
                    c_full = np.broadcast_to(
                        c, op.batch_shape + (op.nrows,)
                    )
                    new_in_cols.append(c_full.reshape(total))
                else:
                    new_in_cols.append(c.reshape(total))
            else:
                ca = jnp.asarray(c)
                if ca.ndim == 1:
                    ca = jnp.broadcast_to(
                        ca, op.batch_shape + (op.nrows,)
                    )
                new_in_cols.append(ca.reshape(total))
        return BEllpack(
            start_row=0, end_row=total,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=total, in_size=op.in_size,
        )
    # Unflatten on unbatched BEllpack: target `(A, B)` from aval `(N,)`
    # where `N == op.out_size == A * B` and B > 1. Inverse of the
    # batch+out flatten emit above. Each flat row i becomes (batch_idx,
    # local_row) = (i // B, i % B); `values` and each band's cols
    # reshape directly from `(N,)` to `(A, B)` (or `(A, B, k)` for
    # values at k>=2). Closes LUKSAN11-15LS's `reshape → mul → reshape`
    # chain where the intermediate (198,) flat BE was previously
    # densified at the unflatten step.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and len(new_sizes) == 2
            and int(new_sizes[0]) * int(new_sizes[1]) == op.out_size
            and int(new_sizes[1]) > 1
            and op.start_row == 0 and op.end_row == op.out_size):
        A = int(new_sizes[0])
        B_out = int(new_sizes[1])
        new_batch = (A,)
        # Values: (N,) → (A, B) for k=1, (N, k) → (A, B, k) for k>=2.
        if op.k == 1:
            new_values = op.values.reshape(A, B_out)
        else:
            new_values = op.values.reshape(A, B_out, op.k)
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                # Slice covers [0, N); reshape to per-(batch, row) by
                # resolving then reshaping.
                rs = np.arange(c.start or 0, c.stop or op.nrows, c.step or 1)
                new_in_cols.append(rs.reshape(A, B_out))
            elif isinstance(c, np.ndarray):
                new_in_cols.append(c.reshape(A, B_out))
            else:
                new_in_cols.append(jnp.asarray(c).reshape(A, B_out))
        return BEllpack(
            start_row=0, end_row=B_out,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=B_out, in_size=op.in_size,
            batch_shape=new_batch,
        )
    # Singleton-axis-insert on unbatched BEllpack: target
    # `(N, 1, ..., 1)` from aval `(N,)` where `N == op.out_size`. The
    # original rows become separate batches and the trailing size-1
    # axes become the new out axis plus singleton batch axes. Mirrors
    # the Change 3 `bid` trailing-singleton path (commit 0123250);
    # reshape can produce the same aval shift via different primitives.
    # Triggered by NONMSQRT's `reshape(BE(out=N), (N, 1))` step that
    # previously densified after the second reduce_sum.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and len(new_sizes) >= 2
            and new_sizes[0] == op.out_size
            and all(s == 1 for s in new_sizes[1:])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(new_sizes[:-1])
        if op.k == 1:
            new_values = op.values.reshape(new_batch + (1,))
        else:
            new_values = op.values.reshape(new_batch + (1, op.k))
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif isinstance(c, np.ndarray):
                new_in_cols.append(c.reshape(new_batch + (1,) + c.shape[1:]))
            else:
                new_in_cols.append(jnp.asarray(c).reshape(
                    new_batch + (1,) + c.shape[1:]))
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=1, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Structural path: flatten a batched BCOO's leading (batch + out)
    # axes into a single flat out axis. Handles the final reshape in
    # DRCAV1LQ/2LQ (`(67, 67) → (4489,)` as aval, LinOp `(67, 67, n) →
    # (4489, n)`). Remaps `new_row = batch_flat * old_out + old_row`
    # with `batch_flat = ravel_multi_index(batch_idx, batch_shape)`,
    # cols unchanged. Only supports fully-flattening the leading dims
    # (target rank 1); partial flattens fall through to dense.
    if (isinstance(op, sparse.BCOO) and op.n_batch >= 1
            and len(new_sizes) == 1
            and int(np.prod(op.shape[:-1])) == int(new_sizes[0])):
        nb = op.n_batch
        old_out = op.shape[nb]  # out axis (after batch, before in)
        in_size = op.shape[-1]
        batch_total = int(np.prod(op.shape[:nb]))
        nse_per_batch = op.data.shape[nb]
        flat_data = op.data.reshape(batch_total, nse_per_batch)
        flat_indices = op.indices.reshape(batch_total, nse_per_batch, 2)
        # Flat batch index b contributes row offset b * old_out.
        offsets = jnp.arange(batch_total, dtype=flat_indices.dtype) * old_out
        new_rows = flat_indices[..., 0] + offsets[:, None]
        new_cols = flat_indices[..., 1]
        new_indices = jnp.stack(
            [new_rows.reshape(-1), new_cols.reshape(-1)], axis=1,
        )
        return sparse.BCOO(
            (flat_data.reshape(-1), new_indices),
            shape=(int(new_sizes[0]), in_size),
        )
    dense = _to_dense(op, n)
    # Reshape applies to output axes only; preserve the trailing input axis.
    return lax.reshape(dense, tuple(new_sizes) + (n,))

materialize_rules[lax.reshape_p] = _reshape_rule

def _broadcast_in_dim_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    shape = params["shape"]
    broadcast_dimensions = params["broadcast_dimensions"]
    # Structural fast path: adding new leading axes to an unbatched BEllpack
    # (the pattern emitted by e.g. `jnp.sum(a[perm_indices], axis=0)`'s
    # backwards-linearize). Produces a BEllpack with the new dims in
    # batch_shape — values broadcast-tiled, in_cols shared across batches.
    # Linear form (aval ()) broadcast to shape (1,): target matrix shape
    # is (1, n) — already what a BEllpack row-vector carries, so pass through.
    # For a (n,)-ndarray linear form, promote to BCOO(1, n) so the
    # subsequent `pad` stays structural (its BCOO path just shifts row
    # indices; the dense fallback would zero-fill an (out, n) block).
    # Triggered by the `reduce_sum → neg → broadcast_in_dim → pad`
    # chain in LIARWHD-class problems.
    if broadcast_dimensions == () and tuple(shape) == (1,):
        if (isinstance(op, BEllpack) and op.n_batch == 0
                and op.out_size == 1 and op.start_row == 0
                and op.end_row == 1):
            return op
        if isinstance(op, jax.Array) and op.ndim == 1 and op.shape[0] == n:
            zeros_row = jnp.zeros((n,), dtype=jnp.int32)
            cols = jnp.arange(n, dtype=jnp.int32)
            indices = jnp.stack([zeros_row, cols], axis=1)
            return sparse.BCOO((op, indices), shape=(1, n))
    # Fallback normalisation: a BEllpack row-vector represents an aval-()
    # linear form. For other broadcast patterns the dense fallback
    # below expects the canonical (n,)-ndarray linear-form shape, so
    # squeeze the BEllpack row first.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and op.out_size == 1 and op.start_row == 0 and op.end_row == 1
            and broadcast_dimensions == ()):
        op = _to_dense(op, n)[0]
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and len(broadcast_dimensions) == 1
            and broadcast_dimensions[0] == len(shape) - 1
            and shape[-1] == op.out_size):
        new_batch = tuple(shape[:-1])
        new_values_shape = new_batch + op.values.shape
        new_values = jnp.broadcast_to(op.values, new_values_shape)
        return BEllpack(
            start_row=op.start_row, end_row=op.end_row,
            in_cols=op.in_cols, values=new_values,
            out_size=op.out_size, in_size=op.in_size,
            batch_shape=new_batch,
        )
    # Trailing-singleton (the `jnp.stack` pattern in LUKSAN11–16):
    # unbatched BEllpack aval-(n,) broadcast to aval-(n, 1, ..., 1). The
    # original rows become separate batches; the trailing size-1 axis
    # becomes the new out_size=1, and any extra middle-1 axes become
    # additional singleton batch axes. Triggered by `bid(shape=(n,
    # 1, ..., 1), broadcast_dimensions=(0,))`.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and len(broadcast_dimensions) == 1
            and broadcast_dimensions[0] == 0
            and len(shape) >= 2
            and shape[0] == op.out_size
            and all(s == 1 for s in shape[1:])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(shape[:-1])  # (n,) or (n, 1, ...)
        # Values: (nrows,) → (*new_batch, 1) for k=1; (nrows, k) →
        # (*new_batch, 1, k) for k>=2. All added axes are size-1 so a
        # plain reshape suffices (no broadcast_to needed).
        if op.k == 1:
            new_values = op.values.reshape(new_batch + (1,))
        else:
            new_values = op.values.reshape(new_batch + (1, op.k))
        # Cols: slice stays shared; ndarray (nrows,) or (nrows, k_band)
        # reshapes to (*new_batch, 1, ...k_band).
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            else:
                new_in_cols.append(c.reshape(new_batch + (1,) + c.shape[1:]))
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=1, in_size=op.in_size,
            batch_shape=new_batch,
        )
    # Leading-dim row-broadcast: `bid(unbatched BE out=N, shape=(N,
    # M_1, ..., M_{r-1}), bd=(0,))`. Output aval adds trailing
    # broadcast axes — each of the N original rows is replicated
    # across the new axes. Represent as batched BE `bs=(N, M_1, ...,
    # M_{r-2}), out=M_{r-1}`, values and cols broadcast-tiled over the
    # new axes. Chains with the reshape singleton-insert
    # (`_reshape_rule`) and dedup-in-reduce_sum to unblock NONMSQRT's
    # final `bid` step. An earlier version of this rule (reverted)
    # regressed because the upstream dedup was missing — K_intermediate
    # blew up past the dense alternative. With dedup, K at this step
    # is ≤ 70 for NONMSQRT-class (vs 5000+ without), and the bid
    # produces BE with nse close to the true matrix nnz.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and len(broadcast_dimensions) == 1
            and broadcast_dimensions[0] == 0
            and len(shape) >= 2
            and shape[0] == op.out_size
            and any(s > 1 for s in shape[1:])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(shape[:-1])           # (N, M_1, ..., M_{r-2})
        new_out = int(shape[-1])                # M_{r-1}
        N = op.out_size
        if op.k == 1:
            reshape_shape = (N,) + (1,) * (len(new_batch) - 1) + (1,)
            new_values = jnp.broadcast_to(
                op.values.reshape(reshape_shape),
                new_batch + (new_out,),
            )
        else:
            reshape_shape = (N,) + (1,) * (len(new_batch) - 1) + (1, op.k)
            new_values = jnp.broadcast_to(
                op.values.reshape(reshape_shape),
                new_batch + (new_out, op.k),
            )
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif isinstance(c, np.ndarray):
                reshape_c = (N,) + (1,) * (len(new_batch) - 1) + (1,) + c.shape[1:]
                new_in_cols.append(
                    np.broadcast_to(c.reshape(reshape_c),
                                    new_batch + (new_out,) + c.shape[1:])
                )
            else:
                ca = jnp.asarray(c)
                reshape_c = (N,) + (1,) * (len(new_batch) - 1) + (1,) + c.shape[1:]
                new_in_cols.append(
                    jnp.broadcast_to(ca.reshape(reshape_c),
                                     new_batch + (new_out,) + c.shape[1:])
                )
        return BEllpack(
            start_row=0, end_row=new_out,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=new_out, in_size=op.in_size,
            batch_shape=new_batch,
        )
    dense = _to_dense(op, n)
    # Map each output axis to the corresponding input axis (or broadcast it).
    out_dims = tuple(broadcast_dimensions) + (len(shape),)  # add input axis
    return lax.broadcast_in_dim(dense, tuple(shape) + (n,), out_dims)

materialize_rules[lax.broadcast_in_dim_p] = _broadcast_in_dim_rule

def _bellpack_row_sum(ep):
    """Sum the rows of an unbatched BEllpack. When static cols let us
    compute the set of touched columns at trace time AND the result is
    structurally sparse (distinct cols < in_size), emit a BEllpack
    row-vector `(1, in_size)` whose bands hold the per-col sums —
    preserving sparsity through downstream broadcast_in_dim / pad /
    add_any chains. Otherwise fall back to a dense `(in_size,)` array.

    Used as `_reduce_sum_rule(ep, axes=(0,))`'s structural path.
    """
    assert ep.n_batch == 0
    nrows = ep.nrows
    k = ep.k
    in_size = ep.in_size
    per_band_cols = [_resolve_col(c, nrows) for c in ep.in_cols]
    if all(isinstance(c, np.ndarray) for c in per_band_cols):
        cols_flat = np.concatenate(per_band_cols)
        valid = cols_flat >= 0
        cols_valid = cols_flat[valid]
        uniq_cols, inverse = np.unique(cols_valid, return_inverse=True)
        n_groups = uniq_cols.shape[0]
        if 0 < n_groups < in_size:
            vals_flat = ep.values if k == 1 else ep.values.T.reshape(-1)
            keep = np.nonzero(valid)[0]
            if keep.shape[0] < cols_flat.shape[0]:
                vals_keep = jnp.take(vals_flat, jnp.asarray(keep))
            else:
                vals_keep = vals_flat
            summed = jnp.zeros((n_groups,), ep.dtype).at[
                jnp.asarray(inverse)].add(vals_keep)
            if n_groups == 1:
                return BEllpack(
                    start_row=0, end_row=1,
                    in_cols=(np.asarray([uniq_cols[0]], dtype=uniq_cols.dtype),),
                    values=summed.reshape(1),
                    out_size=1, in_size=in_size,
                )
            return BEllpack(
                start_row=0, end_row=1,
                in_cols=tuple(np.asarray([c], dtype=uniq_cols.dtype)
                              for c in uniq_cols),
                values=summed.reshape(1, n_groups),
                out_size=1, in_size=in_size,
            )
    result = jnp.zeros((in_size,), ep.dtype)
    for b in range(k):
        cols_b = per_band_cols[b]
        vals_b = ep.values if k == 1 else ep.values[..., b]
        cols_j = jnp.asarray(cols_b)
        mask = cols_j >= 0
        safe_cols = jnp.where(mask, cols_j, 0)
        safe_vals = jnp.where(mask, vals_b, jnp.zeros((), ep.dtype))
        result = result.at[safe_cols].add(safe_vals)
    return result


def _reduce_sum_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    axes = params["axes"]
    # BEllpack with leading batch dims: if `axes` cover the entire
    # batch_shape, split into per-batch Ellpack slices and sum them via
    # `_add_rule` (which handles same-cols dedup / band widening).
    if isinstance(op, BEllpack) and op.n_batch > 0:
        axes_t = tuple(sorted(axes))
        if axes_t == tuple(range(op.n_batch)):
            slices = _bellpack_unbatch(op)
            if len(slices) == 1:
                return slices[0]
            return _add_rule(list(slices), [True] * len(slices), n)
        # Out-axis-only reduction on a single-batch-axis BEllpack:
        # `axes == (n_batch,)` sums the out_size rows within each batch,
        # yielding an unbatched `(batch, in_size)` operator whose row-b
        # is the sum of batch b's out rows. Structural: unbatched
        # BEllpack `out_size=B, in_size=in_size, k=O*K_orig` with
        # (batch, orig_row, orig_band) → (new_band). Values reshape;
        # cols stack per (orig_row, orig_band). Downstream densify
        # goes through the single-scatter fused `.todense()` path, so
        # wide-K does not emit per-band loops.
        # Attacks NONMSQRT (42× vs asdex) — operator is 1/sqrt(n) dense
        # per row (K=70 at n=4900), a natural sparsity width.
        if (axes_t == (op.n_batch,) and op.n_batch == 1
                and op.start_row == 0 and op.end_row == op.out_size):
            B = op.batch_shape[0]
            O = op.out_size
            K = op.k
            # New unbatched BE has k_new = O * K. The iteration order
            # below MUST match the flattening of values: outer loop over
            # original rows r, inner loop over original bands b, so
            # new_band_idx = r * K + b — which matches the default
            # C-order flatten of `values.shape = (*batch, nrows=O, k=K)`
            # to `(B, O*K)`.
            new_in_cols = []
            for r in range(O):
                for b in range(K):
                    c = op.in_cols[b]
                    if isinstance(c, slice):
                        rs = np.arange(c.start or 0, c.stop or O, c.step or 1)
                        new_in_cols.append(np.broadcast_to(
                            np.asarray(rs[r]), (B,)).copy())
                    elif isinstance(c, np.ndarray) and c.ndim == 1:
                        new_in_cols.append(np.broadcast_to(
                            c[r:r+1], (B,)).copy())
                    elif isinstance(c, np.ndarray) and c.ndim == 2:
                        new_in_cols.append(c[:, r])
                    else:
                        # jnp tracer cols — 1D shared or (*batch, nrows).
                        c_full = c if c.ndim >= 2 else jnp.broadcast_to(
                            c, op.batch_shape + (op.nrows,))
                        new_in_cols.append(c_full[:, r])
            if True:
                if K == 1:
                    # (B, O) already in natural r-major order, k=O.
                    new_values = op.values
                else:
                    # (B, O, K) → flatten to (B, O*K) in r-major, b-minor order.
                    new_values = op.values.reshape(B, O * K)
                # Dedup the O*K emitted bands. After out-axis reduction
                # many (r, b) pairs produce cols-identical bands — e.g.
                # NONMSQRT at n=4900 has O*K=5040 bands but only 70
                # unique cols (72× savings). Hash-group by cols bytes
                # (O(K) at trace time) and sum per-group values via one
                # scatter-add HLO op. Not gated on BELLPACK_DEDUP_LIMIT:
                # the dedup savings on wide-K justify the linear-time
                # cost (and the values scatter-add is one HLO op
                # regardless of K).
                def _col_key(c):
                    if isinstance(c, np.ndarray):
                        return ("np", c.shape, c.tobytes())
                    if isinstance(c, slice):
                        return ("slc", c.start, c.stop, c.step)
                    return ("id", id(c))  # traced — won't group
                assigned = np.empty(len(new_in_cols), dtype=np.int64)
                group_cols: list = []
                key_to_group: dict = {}
                for i, c in enumerate(new_in_cols):
                    k_ = _col_key(c)
                    g = key_to_group.get(k_)
                    if g is None:
                        g = len(group_cols)
                        key_to_group[k_] = g
                        group_cols.append(c)
                    assigned[i] = g
                n_groups = len(group_cols)
                if n_groups < len(new_in_cols):
                    # Scatter-add: new_values shape (B, O*K),
                    # assigned shape (O*K,), output shape (B, n_groups).
                    # `.at[..., assigned].add(new_values)` accumulates
                    # along the last axis with repeated indices.
                    dedup_values = jnp.zeros(
                        (B, n_groups), dtype=new_values.dtype
                    ).at[:, assigned].add(new_values)
                    return _densify_if_wider_than_dense(BEllpack(
                        start_row=0, end_row=B,
                        in_cols=tuple(group_cols), values=dedup_values,
                        out_size=B, in_size=op.in_size,
                    ), n)
                return _densify_if_wider_than_dense(BEllpack(
                    start_row=0, end_row=B,
                    in_cols=tuple(new_in_cols), values=new_values,
                    out_size=B, in_size=op.in_size,
                ), n)
    # BEllpack row-sum: accumulate per-column values via scatter-add.
    # Returns a 1D (in_size,) ndarray linear form — the Jacobian
    # coefficients of the resulting scalar-aval variable. Avoids the
    # (out_size, in_size) dense materialisation. Used by the LIARWHD-
    # class walk where the walker arrives at reduce_sum with a
    # structural BEllpack still carrying the sparsity (squeeze kept the
    # BEllpack row-vector, _add_rule tiled it into a banded matrix).
    if tuple(axes) == (0,) and isinstance(op, BEllpack) and op.n_batch == 0:
        return _bellpack_row_sum(op)
    # Fallback: densify + sum. Intentionally no shortcut for Diagonal /
    # ConstantDiagonal — those always densify to a (n,) linear form
    # anyway (1ᵀ diag(v) = v, dense) and returning `op.values` directly
    # breaks XLA fusion (measured 2.25× regression on ARGTRIGLS's dense
    # Hessian where the walk aggregates Diagonals through reduce_sum
    # before the inevitable dense-add downstream). Letting
    # `_to_dense + jnp.sum` stay in the jaxpr lets XLA reduce the
    # combined graph; the (n, n) intermediate is DCE'd.
    dense = _to_dense(op, n)
    return jnp.sum(dense, axis=tuple(axes))

materialize_rules[lax.reduce_sum_p] = _reduce_sum_rule

def _concatenate_rule(invals, traced, n, **params):
    if not any(traced):
        return None
    dimension = params["dimension"]
    traced_idxs = [i for i, t in enumerate(traced) if t]

    # Structural path: all-traced BEllpack concat, each operand spans
    # its full (0, out_size) range, same batch_shape and same in_size.
    # `dimension` in the aval is either a batch axis (dim < n_batch) or
    # the out_size axis (dim == n_batch). For unbatched operands, dim
    # must be 0 (the out_size axis). Produces a single BEllpack:
    #   * dim < n_batch: extend `batch_shape` at that axis.
    #   * dim == n_batch: extend `out_size`, widen to max_k bands.
    # Closure operands in the mix fall through to the pad-based paths.
    if (len(traced_idxs) == len(invals)
            and all(isinstance(v, BEllpack) for v in invals)
            and all(v.batch_shape == invals[0].batch_shape for v in invals)
            and all(v.in_size == invals[0].in_size for v in invals)
            and all(v.start_row == 0 and v.end_row == v.out_size for v in invals)
            and all(v.out_size == invals[0].out_size for v in invals[1:]
                    if dimension < invals[0].n_batch)):
        nb = invals[0].n_batch
        in_size = invals[0].in_size
        if dimension < nb:
            # Batch-axis concat: same out_size, same k assumed, just
            # concatenate values + per-batch in_cols along that axis
            # and grow batch_shape[dim].
            if not all(v.k == invals[0].k for v in invals[1:]):
                pass  # fall through to dense fallback (k mismatch rare
                      # on batch-axis concat; avoid complexity)
            else:
                new_values = jnp.concatenate([v.values for v in invals], axis=dimension)
                new_in_cols = []
                for b in range(invals[0].k):
                    parts = []
                    has_per_batch = False
                    for v in invals:
                        c = v.in_cols[b]
                        if isinstance(c, slice):
                            c = np.arange(c.start or 0, c.stop or v.nrows, c.step or 1)
                        if hasattr(c, "ndim") and c.ndim > 1:
                            has_per_batch = True
                        parts.append(c)
                    if has_per_batch:
                        norm = []
                        for v, c in zip(invals, parts):
                            if hasattr(c, "ndim") and c.ndim == 1:
                                shape = v.batch_shape + (v.nrows,)
                                if isinstance(c, np.ndarray):
                                    c = np.broadcast_to(c, shape)
                                else:
                                    c = jnp.broadcast_to(c, shape)
                            norm.append(c)
                        parts = norm
                        if all(isinstance(c, np.ndarray) for c in parts):
                            new_in_cols.append(np.concatenate(parts, axis=dimension))
                        else:
                            new_in_cols.append(jnp.concatenate(
                                [jnp.asarray(c) for c in parts], axis=dimension))
                    else:
                        # All 1D cols. Two sub-cases:
                        #   - All identical across operands: keep as 1D
                        #     (most efficient — broadcasts across batches).
                        #   - Differ: broadcast each to `(batch_shape[dim],
                        #     nrows)` and concatenate along `dim`, giving
                        #     per-batch 2D cols of shape `(sum_batch,
                        #     nrows)`. Closes LUKSAN17LS's
                        #     `concat(4 × BE(bs=(1,), out=49), dim=0)`
                        #     where each operand has different strided
                        #     1D cols (0,2,4,…; 1,3,5,…; etc.) — this
                        #     previously fell through to dense fallback.
                        if all(np.array_equal(np.asarray(c), np.asarray(parts[0])) for c in parts[1:]):
                            new_in_cols.append(parts[0])
                        else:
                            norm = []
                            for v, c in zip(invals, parts):
                                shape = v.batch_shape + (v.nrows,)
                                if isinstance(c, np.ndarray):
                                    norm.append(np.broadcast_to(c, shape))
                                else:
                                    norm.append(jnp.broadcast_to(c, shape))
                            if all(isinstance(c, np.ndarray) for c in norm):
                                new_in_cols.append(
                                    np.concatenate(norm, axis=dimension))
                            else:
                                new_in_cols.append(jnp.concatenate(
                                    [jnp.asarray(c) for c in norm],
                                    axis=dimension))
                if new_in_cols is not None:
                    new_batch = list(invals[0].batch_shape)
                    new_batch[dimension] = sum(v.batch_shape[dimension] for v in invals)
                    return BEllpack(
                        0, invals[0].out_size, tuple(new_in_cols), new_values,
                        invals[0].out_size, in_size,
                        batch_shape=tuple(new_batch),
                    )
        elif dimension == nb:
            # Out-axis concat: extend out_size, widen bands to max_k
            # (shorter operands pad with -1 sentinels + 0 values).
            max_k = max(v.k for v in invals)
            def _widen_values(v):
                if max_k == 1:
                    return v.values
                vals = v.values if v.values.ndim == nb + 2 else v.values[..., None]
                if v.k < max_k:
                    pad = [(0, 0)] * vals.ndim
                    pad[-1] = (0, max_k - v.k)
                    vals = jnp.pad(vals, pad)
                return vals
            new_values = jnp.concatenate([_widen_values(v) for v in invals], axis=nb)
            new_in_cols = []
            for b in range(max_k):
                band_parts = []
                has_per_batch = False
                for v in invals:
                    if b < v.k:
                        c = v.in_cols[b]
                        if isinstance(c, slice):
                            c = np.arange(c.start or 0, c.stop or v.nrows, c.step or 1)
                    else:
                        c = np.full((v.nrows,), -1, dtype=np.int64)
                    if hasattr(c, "ndim") and c.ndim > 1:
                        has_per_batch = True
                    band_parts.append(c)
                if has_per_batch:
                    normalized = []
                    for v, c in zip(invals, band_parts):
                        if hasattr(c, "ndim") and c.ndim == 1:
                            shape = v.batch_shape + (v.nrows,)
                            if isinstance(c, np.ndarray):
                                c = np.broadcast_to(c, shape)
                            else:
                                c = jnp.broadcast_to(c, shape)
                        normalized.append(c)
                    axis = nb
                    band_parts = normalized
                else:
                    axis = 0
                if all(isinstance(c, np.ndarray) for c in band_parts):
                    new_in_cols.append(np.concatenate(band_parts, axis=axis))
                else:
                    new_in_cols.append(jnp.concatenate(
                        [jnp.asarray(c) for c in band_parts], axis=axis))
            total_out = sum(v.out_size for v in invals)
            return BEllpack(
                0, total_out, tuple(new_in_cols), new_values,
                total_out, in_size,
                batch_shape=invals[0].batch_shape,
            )

    # Structural fast path: `concatenate([C, ..., traced_op, ..., C], axis=0)`
    # — exactly one traced operand sandwiched by closures. Closures have no
    # dependency on the traced input, so their Jacobian rows are zero and the
    # result is structurally `op.pad_rows(left_total, right_total)`. Promote
    # (Constant)Diagonal to BEllpack first so pad_rows is available.
    if dimension == 0 and len(traced_idxs) == 1:
        idx = traced_idxs[0]
        op = invals[idx]
        left_total = sum(int(invals[i].shape[0]) for i in range(idx))
        right_total = sum(int(invals[i].shape[0])
                          for i in range(idx + 1, len(invals)))
        if isinstance(op, ConstantDiagonal):
            op = BEllpack(
                0, op.n, (np.arange(op.n),),
                jnp.broadcast_to(jnp.asarray(op.value), (op.n,)),
                op.n, op.n,
            )
        elif isinstance(op, Diagonal):
            op = BEllpack(
                0, op.n, (np.arange(op.n),), op.values, op.n, op.n,
            )
        if isinstance(op, BEllpack) and op.n_batch == 0:
            return op.pad_rows(left_total, right_total)
        if isinstance(op, sparse.BCOO):
            out_size = op.shape[0] + left_total + right_total
            new_rows = op.indices[:, 0] + left_total
            new_indices = jnp.stack([new_rows, op.indices[:, 1]], axis=1)
            return sparse.BCOO(
                (op.data, new_indices), shape=(out_size, op.shape[1])
            )
    # Fallback: densify everything.
    parts = []
    for v, t in zip(invals, traced):
        if t:
            parts.append(_to_dense(v, n))
        else:
            # Closure constant: extend with a zero "input axis" of size n.
            parts.append(jnp.broadcast_to(v[..., None] * 0, v.shape + (n,)))
    return lax.concatenate(parts, dimension)

materialize_rules[lax.concatenate_p] = _concatenate_rule

def _split_rule(invals, traced, n, **params):
    (operand,) = invals
    (t,) = traced
    if not t:
        return None
    sizes = params["sizes"]
    axis = params["axis"]
    # Structural path: batched BE split along the out-axis (== n_batch).
    # Slice values and each band's cols along the out axis; keep batch_shape.
    # Requires full out coverage so each chunk's rows map to [0, sz) cleanly.
    if (isinstance(operand, BEllpack)
            and operand.n_batch >= 1
            and axis == operand.n_batch
            and operand.start_row == 0
            and operand.end_row == operand.out_size):
        nb = operand.n_batch
        out = []
        start = 0
        for sz in sizes:
            sz_i = int(sz)
            end = start + sz_i
            val_slc = [slice(None)] * operand.values.ndim
            val_slc[nb] = slice(start, end)
            new_values = operand.values[tuple(val_slc)]
            new_in_cols = []
            for c in operand.in_cols:
                arr = c
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 1:
                        new_in_cols.append(arr[start:end])
                    else:
                        slc = [slice(None)] * arr.ndim
                        slc[nb] = slice(start, end)
                        new_in_cols.append(arr[tuple(slc)])
                else:
                    arr_j = jnp.asarray(arr)
                    if arr_j.ndim == 1:
                        new_in_cols.append(arr_j[start:end])
                    else:
                        slc = [slice(None)] * arr_j.ndim
                        slc[nb] = slice(start, end)
                        new_in_cols.append(arr_j[tuple(slc)])
            out.append(BEllpack(
                0, sz_i, tuple(new_in_cols), new_values,
                sz_i, operand.in_size, batch_shape=operand.batch_shape,
            ))
            start = end
        return out
    # Structural path: split along output axis 0 (the "out_size" dim).
    # For an unbatched BEllpack with static cols we slice the BE
    # per-chunk (row range + per-band-col row-slice) and emit one
    # proper BCOO per chunk. Going through `_to_bcoo` on the full BE
    # and then masking out-of-range rows to `(row=0, value=0)` would
    # leave zero-valued entries clogging row 0 of every chunk — those
    # count as BCOO nse and manufacture "duplicates" at row 0 that
    # propagate through every downstream add/concat (observed as
    # COATING's 4.5× final nse bloat).
    if (axis == 0 and isinstance(operand, BEllpack)
            and operand.n_batch == 0
            and all(isinstance(c, np.ndarray) or isinstance(c, slice)
                    for c in operand.in_cols)):
        out = []
        start = 0
        for sz in sizes:
            sz_i = int(sz)
            end = start + sz_i
            # Row range [start, end) intersected with BE's own
            # [start_row, end_row). Slice cols/values along the row axis.
            be_start = max(operand.start_row - start, 0)
            be_end = min(operand.end_row - start, sz_i)
            if be_end <= be_start:
                out.append(sparse.BCOO(
                    (jnp.zeros((0,), operand.values.dtype),
                     np.zeros((0, 2), np.int32)),
                    shape=(sz_i, operand.in_size),
                ))
                start = end
                continue
            row_lo = max(start, operand.start_row) - operand.start_row
            row_hi = min(end, operand.end_row) - operand.start_row
            new_in_cols = []
            for c in operand.in_cols:
                if isinstance(c, slice):
                    c = _resolve_col(c, operand.nrows)
                new_in_cols.append(c[row_lo:row_hi])
            if operand.k == 1:
                new_values = operand.values[row_lo:row_hi]
            else:
                new_values = operand.values[row_lo:row_hi, :]
            chunk_be = BEllpack(
                be_start, be_end, tuple(new_in_cols), new_values,
                sz_i, operand.in_size,
            )
            out.append(chunk_be)
            start = end
        return out
    if axis == 0 and isinstance(operand, (ConstantDiagonal, Diagonal,
                                          BEllpack, sparse.BCOO)):
        bcoo = _to_bcoo(operand, n)
        rows = bcoo.indices[:, 0]
        out = []
        start = 0
        for sz in sizes:
            end = start + int(sz)
            in_range = (rows >= start) & (rows < end)
            new_rows = jnp.where(in_range, rows - start, 0)
            new_data = jnp.where(in_range, bcoo.data,
                                 jnp.zeros((), bcoo.data.dtype))
            new_indices = jnp.stack(
                [new_rows, bcoo.indices[:, 1]], axis=1
            )
            out.append(sparse.BCOO(
                (new_data, new_indices), shape=(int(sz), bcoo.shape[1])
            ))
            start = end
        return out
    dense = _to_dense(operand, n)
    out = []
    start = 0
    for sz in sizes:
        slc = [slice(None)] * dense.ndim
        slc[axis] = slice(int(start), int(start) + int(sz))
        out.append(dense[tuple(slc)])
        start += int(sz)
    return out

materialize_rules[lax.split_p] = _split_rule

def _cond_rule(invals, traced, n, **params):
    """`lax.cond` with a compile-time-decidable branch choice.

    Two senses of "traced" coexist here and are orthogonal:
      * `traced[0]` (walker-level): does the index depend on the
        walker's seed input? `True` means the branch choice is part
        of the structural chain — genuinely data-dependent control
        flow we don't support (non-linear).
      * `isinstance(invals[0], Tracer)` (JAX-level): is the VALUE a
        `DynamicJaxprTracer`? Under outer `jax.jit`, every closure
        `jnp.ndarray` gets lifted into the traced graph, so even
        walker-static closures can be tracers at the value level.

    Two structural (no-densify) patterns we handle:

    1. **Closure-concrete index**: un-jitted walks, or jitted walks
       under `EAGER_CONSTANT_FOLDING=TRUE`. `int(invals[0])` succeeds
       (0-d int arrays support `__index__`); we pick the branch.

    2. **`lax.platform_dependent`**: emits a `cond` whose index is
       `platform_index_p`, which stays an abstract tracer under outer
       jit without ECF. By the `platform_dependent` contract all
       branches are semantically equivalent — the actual platform is
       decided at lowering time. The eqn carries `branches_platforms`;
       we detect that and pick the `None` (default) branch. Covers
       the `jnp.diagonal` mosaic-vs-default dispatch HADAMALS hits
       under un-ECF jit.

    Neither case requires densification.
    """
    if traced[0]:
        raise NotImplementedError(
            "cond with walker-traced index (genuine control flow)"
        )
    branches = params["branches"]
    try:
        # Tuple-indexing uses `__index__`; 0-d int arrays / np scalars
        # / Python ints all support it. Under concrete conditions
        # (un-jit, or jit + ECF) this picks the right branch directly —
        # including the `platform_dependent` case, where
        # `platform_index_p` evaluates eagerly to the current
        # platform's branch index.
        chosen = branches[invals[0]]
    except jax.errors.TracerIntegerConversionError as e:
        # Tracer index (outer jit, no ECF). If this is a
        # `platform_dependent` cond, all branches are semantically
        # equivalent per its contract — pick the default (`None`
        # platform) branch. Otherwise we can't decide without
        # densifying.
        bp = params.get("branches_platforms")
        if bp is None:
            raise NotImplementedError(
                f"cond with tracer index ({type(invals[0]).__name__}) "
                f"and no `branches_platforms` hint — can't pick a branch "
                f"without densifying both"
            ) from e
        chosen = branches[next((i for i, pl in enumerate(bp) if pl is None), 0)]
    inner = chosen.jaxpr
    operand_invals = invals[1:]
    operand_traced = traced[1:]
    inner_env: dict = {v: (False, c) for v, c in zip(inner.constvars, chosen.consts)}
    for inner_invar, outer_val, was_traced in zip(
        inner.invars, operand_invals, operand_traced
    ):
        inner_env[inner_invar] = (was_traced, outer_val)
    _walk_jaxpr(inner, inner_env, n)
    return [inner_env[outvar][1] for outvar in inner.outvars]


try:
    from jax._src.lax.control_flow.conditionals import cond_p
    materialize_rules[cond_p] = _cond_rule
except ImportError:
    pass


def _jit_rule(invals, traced, n, **params):
    """Recurse into the inner jaxpr of a `jit` (pjit) call."""
    inner_cj = params["jaxpr"]  # ClosedJaxpr
    inner = inner_cj.jaxpr
    inner_consts = inner_cj.consts

    inner_env: dict = {v: (False, c) for v, c in zip(inner.constvars, inner_consts)}
    for inner_invar, outer_val, was_traced in zip(inner.invars, invals, traced):
        inner_env[inner_invar] = (was_traced, outer_val)
    _walk_jaxpr(inner, inner_env, n)

    # jit_p is always multiple_results; walker sets all outputs to traced
    # since _jit_rule is only called when any input is traced.
    return [inner_env[outvar][1] for outvar in inner.outvars]


# pjit_p is the modern name for jit's primitive.
try:
    from jax._src.pjit import jit_p

    materialize_rules[jit_p] = _jit_rule
except ImportError:
    pass


def _select_n_rule(invals, traced, n, **params):
    """`select_n(pred, *cases)` for constant `pred`. Predicates derived from
    traced inputs would imply a data-dependent branch, which is non-linear and
    not supported.
    """
    del params
    pred = invals[0]
    cases = invals[1:]
    pred_traced = traced[0]
    case_traced = traced[1:]
    if pred_traced:
        raise NotImplementedError("select_n with traced predicate")
    if not any(case_traced):
        return None  # caller's constant-prop path handles concrete eval

    # Structural fast path (single traced op, rest closures): output row i
    # is zero wherever pred picks a closure case, else matches the traced
    # op's row i. Promote (Constant)Diagonal → BEllpack first, then mask.
    if sum(case_traced) == 1:
        t_idx = case_traced.index(True)
        t_case = cases[t_idx]
        if isinstance(t_case, ConstantDiagonal):
            t_case = BEllpack(
                0, t_case.n, (np.arange(t_case.n),),
                jnp.broadcast_to(jnp.asarray(t_case.value), (t_case.n,)),
                t_case.n, t_case.n,
            )
        elif isinstance(t_case, Diagonal):
            t_case = BEllpack(
                0, t_case.n, (np.arange(t_case.n),), t_case.values,
                t_case.n, t_case.n,
            )
        # BCOO: mask data entries by row-predicate.
        if isinstance(t_case, sparse.BCOO):
            pred_arr = jnp.asarray(pred)
            entry_rows = t_case.indices[:, 0]
            entry_mask = (pred_arr[entry_rows] == t_idx)
            new_data = jnp.where(entry_mask, t_case.data,
                                 jnp.zeros((), t_case.data.dtype))
            return sparse.BCOO(
                (new_data, t_case.indices), shape=t_case.shape
            )
    if sum(case_traced) == 1 and isinstance(t_case, BEllpack):
        # pred has the BE's aval shape `(*batch_shape, out_size)`;
        # slice the last axis to the active row range. mask is
        # `(*batch_shape, nrows)`, broadcasting over the trailing k
        # axis for k>=2 values.
        pred_arr = jnp.asarray(pred)
        pred_slice = pred_arr[..., t_case.start_row:t_case.end_row]
        mask = (pred_slice == t_idx)
        if t_case.k >= 2:
            mask = mask[..., None]
        new_values = jnp.where(mask, t_case.values,
                               jnp.zeros((), t_case.dtype))
        return BEllpack(
            t_case.start_row, t_case.end_row, t_case.in_cols,
            new_values, t_case.out_size, t_case.in_size,
            batch_shape=t_case.batch_shape,
        )

    # Structural fast path: all cases are BEllpack with matching
    # (start_row, end_row, out_size, in_size) and identical in_cols tuples.
    # Then select_n is a per-row choice among their values — emit one
    # BEllpack with `values = where(pred_slice, case_0.values, case_1.values, ...)`.
    if all(t and isinstance(c, BEllpack) and c.n_batch == 0
           for c, t in zip(cases, case_traced)):
        first = cases[0]
        same_shape = all(
            c.start_row == first.start_row and c.end_row == first.end_row
            and c.out_size == first.out_size and c.in_size == first.in_size
            for c in cases[1:]
        )
        same_cols = same_shape and all(
            len(c.in_cols) == len(first.in_cols)
            and all(_cols_equal(a, b)
                    for a, b in zip(c.in_cols, first.in_cols))
            for c in cases[1:]
        )
        if same_cols:
            pred_arr = jnp.asarray(pred)
            pred_slice = pred_arr[first.start_row:first.end_row]
            if first.values.ndim > 1:
                pred_b = pred_slice[:, None]
            else:
                pred_b = pred_slice
            # select_n with bool pred: cases[0] when pred is False, cases[1] when True
            # (matching lax.select_n semantics for 2-case).
            if len(cases) == 2:
                new_values = jnp.where(pred_b, cases[1].values, cases[0].values)
            else:
                # N-way: use lax.select_n on stacked values.
                stacked = jnp.stack([c.values for c in cases], axis=0)
                new_values = lax.select_n(pred_b, *[stacked[i] for i in range(len(cases))])
            return BEllpack(
                first.start_row, first.end_row, first.in_cols,
                new_values, first.out_size, first.in_size,
            )

    # Multi-traced BCOO / BE (mismatched cols): mask each traced case by
    # its own `pred[row] == case_idx` predicate and concat the results as
    # a BCOO. Non-traced cases contribute zero to the linear-in-input
    # part, so we drop them. BE operands are promoted via `_to_bcoo` +
    # row-mask. Used by BROYDN7D (2×BCOO select_n over 5000-row state —
    # the dense fallback would emit a 25M-element matrix).
    # Gate: only fire when at least one operand is already BCOO AND
    # the aval has rank 1 (simple row-select). Pure CD/Diag cases
    # should go through the existing dense fallback to preserve the
    # bit-exact contract the `select_n(pred, d0, d1)` HLO provides.
    any_bcoo = any(isinstance(c, sparse.BCOO)
                   for c, t in zip(cases, case_traced) if t)
    if (any_bcoo and pred.ndim == 1
            and all(isinstance(c, (sparse.BCOO, BEllpack, Diagonal,
                                   ConstantDiagonal))
                    for c, t in zip(cases, case_traced) if t)):
        pred_arr = jnp.asarray(pred)
        masked_bcoos = []
        for c_idx, (c, t) in enumerate(zip(cases, case_traced)):
            if not t:
                continue
            bc = _to_bcoo(c, n)
            if bc.n_batch != 0:
                masked_bcoos = None
                break
            entry_rows = bc.indices[:, 0]
            mask = pred_arr[entry_rows] == c_idx
            new_data = jnp.where(mask, bc.data,
                                 jnp.zeros((), bc.data.dtype))
            masked_bcoos.append(sparse.BCOO(
                (new_data, bc.indices), shape=bc.shape,
            ))
        if masked_bcoos and len(masked_bcoos) == 1:
            return masked_bcoos[0]
        if masked_bcoos:
            return _bcoo_concat(masked_bcoos, shape=masked_bcoos[0].shape)

    # Densify each case to shape (*var_shape, n). Non-traced cases contribute
    # zero to the linear-in-input part (their dependence on the traced input
    # is zero), so we represent them as a zero tensor of the right shape.
    case_dense = []
    for c, t in zip(cases, case_traced):
        if t:
            case_dense.append(_to_dense(c, n))
        else:
            arr = jnp.asarray(c)
            zero_shape = arr.shape + (n,)
            case_dense.append(jnp.zeros(zero_shape, dtype=arr.dtype))

    pred_arr = jnp.asarray(pred)
    # pred has shape (*var_shape,); broadcast it to (*var_shape, n) so each
    # row across the input-coord axis is selected the same way.
    target_shape = case_dense[0].shape
    pred_b = jnp.broadcast_to(pred_arr[..., None], target_shape)
    return lax.select_n(pred_b, *case_dense)

materialize_rules[lax.select_n_p] = _select_n_rule

def _cumsum_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    axis = params["axis"]
    reverse = params.get("reverse", False)
    dense = _to_dense(op, n)
    return lax.cumsum(dense, axis=axis, reverse=reverse)

materialize_rules[lax.cumsum_p] = _cumsum_rule

def _div_rule(invals, traced, n, **params):
    a, b = invals
    ta, tb = traced
    if not ta and not tb:
        return None
    if tb:
        raise NotImplementedError("div with traced denominator")
    # a / b where b is closure: equivalent to mul(a, 1/b).
    inv_b = jnp.reciprocal(jnp.asarray(b))
    return _mul_rule([a, inv_b], [True, False], n)

materialize_rules[lax.div_p] = _div_rule

def _transpose_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    permutation = tuple(params["permutation"])
    if isinstance(op, BEllpack):
        return op.transpose(permutation)
    dense = _to_dense(op, n)
    # Permutation applies to output axes only; preserve trailing input axis.
    return lax.transpose(dense, permutation + (len(permutation),))

materialize_rules[lax.transpose_p] = _transpose_rule

def _gather_rule(invals, traced, n, **params):
    operand, start_indices = invals
    to, ti = traced
    if ti:
        raise NotImplementedError("gather with traced indices")
    if not to:
        return None
    dnums = params["dimension_numbers"]
    # 2D point-gather fallback (HADAMALS-class): `M[b[i,0], b[i,1]]`
    # with dnums `collapsed_slice_dims=(0, 1), start_index_map=(0, 1),
    # slice_sizes=(1, 1)`. Materialize dense and gather. Correct but
    # not structural — leaves optimization for follow-up.
    if (
        dnums.offset_dims == ()
        and dnums.collapsed_slice_dims == (0, 1)
        and dnums.start_index_map == (0, 1)
        and params["slice_sizes"] == (1, 1)
    ):
        # Structural 2D point-gather on batched BEllpack (0d): each
        # gather index `(r, c)` picks the k entries stored at
        # `operand[r, c]` across all bands. Shapes: operand aval
        # `(R, C)` with nrows = C, per-band cols resolved to `(R, C)`,
        # values `(R, C[, k])`. Output LinOp batch_shape = leading
        # axes of `start_indices`, out_size = last leading axis,
        # `k = k_old`. Leaves the walk sparse where the dense fallback
        # would densify at `k_old * N` entries.
        if (isinstance(operand, BEllpack)
                and operand.n_batch == 1
                and operand.start_row == 0
                and operand.end_row == operand.out_size
                and start_indices.ndim >= 2):
            leading = start_indices.shape[:-1]
            new_batch = leading[:-1]
            new_out = leading[-1]
            # Fancy indexing with `row_flat`/`col_flat` needs jnp on
            # both sides when indices are outer-jit tracers. Cols that
            # were static np.ndarrays become jnp tracers after this —
            # downstream just treats them as traced cols.
            idx_static = isinstance(start_indices, np.ndarray)
            if idx_static:
                row_flat = start_indices[..., 0].reshape(-1)
                col_flat = start_indices[..., 1].reshape(-1)
            else:
                row_flat = jnp.asarray(start_indices[..., 0]).reshape(-1)
                col_flat = jnp.asarray(start_indices[..., 1]).reshape(-1)
            new_in_cols = []
            for c in operand.in_cols:
                c_full = _resolve_full(c, operand.nrows, operand.batch_shape)
                if not idx_static and isinstance(c_full, np.ndarray):
                    c_full = jnp.asarray(c_full)
                new_in_cols.append(c_full[row_flat, col_flat].reshape(leading))
            vals = operand.values[row_flat, col_flat]
            if operand.k == 1:
                vals = vals.reshape(leading)
            else:
                vals = vals.reshape(leading + (operand.k,))
            return BEllpack(
                start_row=0, end_row=new_out,
                in_cols=tuple(new_in_cols), values=vals,
                out_size=new_out, in_size=operand.in_size,
                batch_shape=new_batch,
            )
        dense = _to_dense(operand, n)
        # dense has shape `(*operand_primal_shape, n)` where the last
        # axis is the input axis; gather collapses the first two.
        row_idx = start_indices[..., 0]
        col_idx = start_indices[..., 1]
        return dense[row_idx, col_idx]
    if (
        dnums.offset_dims != ()
        or dnums.collapsed_slice_dims != (0,)
        or dnums.start_index_map != (0,)
        or params["slice_sizes"] != (1,)
    ):
        raise NotImplementedError(f"gather with unhandled dnums: {dnums}")
    row_idx = start_indices[..., 0]
    # Build a BEllpack for (Constant)Diagonal operand. row_idx has shape
    # (*batch_shape, N) — 1D for the standard gather case, multi-dim for
    # batched gathers like SPARSINE's `sine_values[perm_indices]` with
    # perm_indices shape (B, N).
    if isinstance(operand, (ConstantDiagonal, Diagonal)):
        batch_shape = tuple(row_idx.shape[:-1])
        N = row_idx.shape[-1]
        if isinstance(operand, ConstantDiagonal):
            vals = jnp.broadcast_to(
                jnp.asarray(operand.value), batch_shape + (N,)
            )
        else:
            # Diagonal(v) — value at col c is v[c]. Gather v[row_idx].
            vals = jnp.take(operand.values, row_idx)
        return BEllpack(
            start_row=0, end_row=N,
            in_cols=(row_idx,),
            values=vals,
            out_size=N, in_size=operand.n,
            batch_shape=batch_shape,
        )
    if isinstance(operand, sparse.BCOO):
        raise NotImplementedError("gather on BCOO operand")
    # Dense fallback: gather rows of the dense linop.
    dense = _to_dense(operand, n)
    return dense[row_idx]

materialize_rules[lax.gather_p] = _gather_rule

def _scatter_add_rule(invals, traced, n, **params):
    operand, scatter_indices, updates = invals
    to, ti, tu = traced
    if ti:
        raise NotImplementedError("scatter-add with traced indices")
    if to:
        raise NotImplementedError("scatter-add with traced operand")
    if not tu:
        return None
    dnums = params["dimension_numbers"]
    # 2D point-scatter fallback (HADAMALS-class): inserts `updates[k]`
    # at `operand[scatter_indices[k, 0], scatter_indices[k, 1]]`. Emit
    # as a BCOO with the static (i, j) indices and the traced values.
    if (
        dnums.update_window_dims == ()
        and dnums.inserted_window_dims == (0, 1)
        and dnums.scatter_dims_to_operand_dims == (0, 1)
        and operand.ndim == 2
    ):
        operand_dense = jnp.asarray(operand)
        out_shape_2d = operand_dense.shape  # (R, C)
        # updates shape: scatter_indices' leading dims (N,) + input_axis (n,)
        updates_dense = _to_dense(updates, n)  # (N, n) or broader
        # Flatten any leading batch shape of scatter_indices to N.
        si_flat = scatter_indices.reshape(-1, 2)
        updates_flat = updates_dense.reshape(-1, n)
        # Emit to a flat 2D BCOO matching operand shape: the output has
        # shape `(R*C, n)` — flatten the (R, C) output axis so it lives
        # in LinOp's (out_size, in_size) layout.
        flat_rows = (
            si_flat[:, 0].astype(jnp.int64) * out_shape_2d[1]
            + si_flat[:, 1].astype(jnp.int64)
        )
        # Dense approach: scatter into flattened zeros.
        out_size_flat = out_shape_2d[0] * out_shape_2d[1]
        return (
            jnp.zeros((out_size_flat, n), updates_flat.dtype)
            .at[flat_rows]
            .add(updates_flat)
            .reshape(out_shape_2d + (n,))
        )
    if (
        dnums.update_window_dims != ()
        or dnums.inserted_window_dims != (0,)
        or dnums.scatter_dims_to_operand_dims != (0,)
    ):
        raise NotImplementedError(f"scatter-add with unhandled dnums: {dnums}")
    out_idx = scatter_indices[..., 0]
    out_size = operand.shape[0]
    # BEllpack updates: batched case handled per-slice (each batch's
    # Ellpack rows get remapped via scatter_indices[b]), then concat as
    # BCOO. Unbatched case falls through to the 1D-BCOO path below.
    if isinstance(updates, BEllpack):
        if updates.n_batch == 0:
            updates = _to_bcoo(updates, n)
        else:
            # Batched: unbatch, remap each slice's rows, concat.
            slices = _bellpack_unbatch(updates)
            bcoo_pieces = []
            for b_idx, ep in enumerate(slices):
                bc = _to_bcoo(ep, n)
                old_rows = bc.indices[:, 0]
                new_rows = out_idx[b_idx][old_rows]
                new_indices = jnp.stack(
                    [new_rows, bc.indices[:, 1]], axis=1
                )
                bcoo_pieces.append(sparse.BCOO(
                    (bc.data, new_indices),
                    shape=(out_size, updates.in_size),
                ))
            return _bcoo_concat(
                bcoo_pieces, shape=(out_size, updates.in_size)
            )
    out_idx_flat = out_idx.reshape(-1)
    if isinstance(updates, sparse.BCOO):
        new_rows = out_idx_flat[updates.indices[:, 0]]
        new_indices = jnp.stack([new_rows, updates.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (updates.data, new_indices), shape=(out_size, n)
        )
    # Dense fallback: updates has shape (*batch, n). Reshape to (prod_batch, n)
    # and scatter-add row i of flat into row out_idx_flat[i] of a zero (out_size, n).
    updates_dense = _to_dense(updates, n)
    flat_updates = updates_dense.reshape(-1, n)
    return (jnp.zeros((out_size, n), flat_updates.dtype)
            .at[out_idx_flat].add(flat_updates))

materialize_rules[_slicing.scatter_add_p] = _scatter_add_rule

# -------------------------- driver --------------------------


def _walk_jaxpr(jaxpr, env, n):
    """Walk a jaxpr, mutating env.

    Env is `dict[Var, tuple[bool, Any]]` where the bool is `traced`:
      * (True, LinOp) — this var depends on the walk's input; value is a LinOp.
      * (False, concrete_array) — this var is pure closure data.
    Literals are read directly from `.val`; traced status comes from the
    invars the caller seeded.
    """

    def read(atom):
        if isinstance(atom, core.Literal):
            return (False, atom.val)
        return env[atom]

    for eqn in jaxpr.eqns:
        entries = [read(v) for v in eqn.invars]
        invals = [e[1] for e in entries]
        traced = [e[0] for e in entries]
        if not any(traced):
            # Constant propagation: no traced inputs → evaluate concretely
            # and stash as closure data. Important for constant-H problems
            # (DUAL, CMPC) — lets the whole walk fold to a trace-time BCOO
            # literal. See docs/RESEARCH_NOTES.md §10.
            concrete_outs = eqn.primitive.bind(*invals, **eqn.params)
            if eqn.primitive.multiple_results:
                for v, o in zip(eqn.outvars, concrete_outs):
                    env[v] = (False, o)
            else:
                (outvar,) = eqn.outvars
                env[outvar] = (False, concrete_outs)
            continue
        rule = materialize_rules.get(eqn.primitive)
        if rule is None:
            forms = ", ".join(
                type(v).__name__ if t else f"closure:{type(v).__name__}"
                for v, t in zip(invals, traced)
            )
            raise NotImplementedError(
                f"No lineaxpr rule for primitive '{eqn.primitive}'.\n"
                f"  Input forms: [{forms}]\n"
                f"  To add a rule: register at lineaxpr.materialize_rules[{eqn.primitive}] = your_rule\n"
                f"  Or file an issue at https://github.com/jpbrodrick89/lineaxpr/issues "
                f"with the minimal f(y) that triggers this."
            )
        outs = rule(invals, traced, n, **eqn.params)
        if eqn.primitive.multiple_results:
            for v, o in zip(eqn.outvars, outs):
                env[v] = (True, o)
        else:
            (outvar,) = eqn.outvars
            env[outvar] = (True, outs)


def _walk_with_seed(linear_fn, seed_linop):
    """Trace `linear_fn` with the aval implied by `seed_linop`, walk the
    jaxpr, return the output LinOp."""
    aval = seed_linop.primal_aval()
    placeholder = jax.ShapeDtypeStruct(aval.shape, aval.dtype)
    cj = jax.make_jaxpr(linear_fn)(placeholder)
    jaxpr = cj.jaxpr

    if len(jaxpr.invars) != 1:
        raise NotImplementedError("multi-input linear_fn not yet handled")
    (invar,) = jaxpr.invars
    if invar.aval.ndim != 1:
        raise NotImplementedError("non-1D input not yet handled")
    n = invar.aval.size

    env: dict = {v: (False, c) for v, c in zip(jaxpr.constvars, cj.consts)}
    env[invar] = (True, seed_linop)
    _walk_jaxpr(jaxpr, env, n)

    if len(jaxpr.outvars) != 1:
        raise NotImplementedError("multi-output linear_fn not yet handled")
    (outvar,) = jaxpr.outvars
    return env[outvar][1]


def sparsify(linear_fn):
    """Transform a linear function into one that operates on LinOps.

    `sparsify(linear_fn)(seed_linop)` traces `linear_fn` against the aval
    implied by `seed_linop.primal_aval()`, walks the resulting jaxpr with
    per-primitive structural rules, and returns a LinOp representing the
    linear function's matrix.

    Seeds are explicit — no automatic Identity cast. For the common case
    of extracting the full Jacobian, the public wrappers `materialize` /
    `jacfwd` / `jacrev` / `hessian` build
    `Identity(primal.size, dtype=primal.dtype)` and pass it through.
    """
    def inner(seed_linop):
        return _walk_with_seed(linear_fn, seed_linop)

    return inner


_SMALL_N_VMAP_THRESHOLD = 16
"""Below this n, vmap(linear_fn)(eye) emits less HLO than the structural walk
on most problems. Above it the walk's structure exploitation dominates."""


def to_dense(op):
    """Densify a LinOp returned by `sparsify` to a jnp.ndarray.

    Uniform across all possible return types:
    - Our LinOp classes (ConstantDiagonal, Diagonal, BEllpack) → `.todense()`.
    - `jax.experimental.sparse.BCOO` → `.todense()`.
    - Plain ndarray → passthrough.
    """
    if isinstance(op, (ConstantDiagonal, Diagonal, BEllpack)):
        return op.todense()
    if isinstance(op, sparse.BCOO):
        return op.todense()
    return op


def to_bcoo(op):
    """Convert a LinOp returned by `sparsify` to a BCOO (or ndarray if
    the walk produced a dense fallback that can't be usefully sparsified).

    - Our LinOp classes → `.to_bcoo()`.
    - `BCOO` passthrough.
    - Plain ndarray passthrough (caller decides what to do).
    """
    if isinstance(op, sparse.BCOO):
        return op
    if isinstance(op, (ConstantDiagonal, Diagonal, BEllpack)):
        return op.to_bcoo()
    return op


_VALID_FORMATS = ("dense", "bcoo")


def materialize(linear_fn, primal, format: str = "dense"):
    """Materialize the Jacobian matrix of a linear callable.

    Args:
      linear_fn: a linear callable `R^n -> R^m` (typically the output of
        `jax.linearize(...)[1]` or `jax.linear_transpose(...)`).
      primal: a shape/dtype witness for the input to `linear_fn`. Only
        `primal.size` and `primal.dtype` are read, so this can be any of:
        a concrete array, a `jax.Array` / `jnp.ndarray`, or a
        `jax.ShapeDtypeStruct` (matching the convention used by
        `jax.linear_transpose` / `jax.eval_shape`). Passing a
        ShapeDtypeStruct is the preferred option when you don't already
        have a concrete primal on hand.
      format: one of `"dense"` or `"bcoo"`.
        - `"dense"` returns a `jnp.ndarray`.
        - `"bcoo"` returns a `jax.experimental.sparse.BCOO` when the
          walk preserves structural sparsity, otherwise a dense ndarray
          (dense fallbacks surface to the caller unchanged).

    For tiny inputs (n < `_SMALL_N_VMAP_THRESHOLD`) the structural walk
    emits more HLO than `vmap(linear_fn)(eye)` — the short-circuit is
    always dense; users asking for `"bcoo"` at tiny n still get dense
    output (by design; densification at small n is the right call).
    """
    if format not in _VALID_FORMATS:
        raise ValueError(f"format must be one of {_VALID_FORMATS}, got {format!r}")
    n = primal.size if hasattr(primal, "size") else int(jnp.size(primal))
    if n < _SMALL_N_VMAP_THRESHOLD:
        return jax.vmap(linear_fn)(jnp.eye(n, dtype=primal.dtype)).T
    seed = Identity(n, dtype=primal.dtype)
    linop = sparsify(linear_fn)(seed)
    if format == "dense":
        return to_dense(linop)
    return to_bcoo(linop)


# -------------------------- jax-like public API --------------------------


def _jacfwd_impl(f, y, format: str):
    """materialize ∘ jax.linearize — forward-mode Jacobian."""
    y_out, lin = jax.linearize(f, y)
    del y_out  # only needed for shape in jacrev
    return materialize(lin, y, format=format)


def _jacrev_impl(f, y, format: str):
    """materialize ∘ jax.linear_transpose ∘ jax.linearize — reverse-mode
    Jacobian. linear_transpose of the JVP is the VJP; materializing gives
    Jᵀ, so we transpose the result to match `jax.jacrev`'s shape."""
    y_out, lin = jax.linearize(f, y)
    vjp = jax.linear_transpose(lin, y)
    # vjp: R^m -> R^n, where m = y_out.shape.
    # jax.linear_transpose wraps the result in a tuple (multi-output),
    # so we unpack.
    def vjp_unpacked(w):
        (out,) = vjp(w)
        return out
    # Primal for the VJP is a shape/dtype witness of y_out.
    jt = materialize(vjp_unpacked, y_out, format=format)
    # jt has shape (n, m); we want (m, n) to match jax.jacrev.
    if format == "dense":
        return jt.T
    # BCOO supports .T via transpose().
    return jt.T


def jacfwd(f, *, format: str = "dense"):
    """Forward-mode Jacobian, matching `jax.jacfwd`'s output shape.

    Equivalent to `materialize(jax.linearize(f, y)[1], y, format=format)`.

    Returns a function `(y) -> Jacobian`. `format='dense'` (default)
    returns a `jnp.ndarray`; `format='bcoo'` returns a BCOO when
    structural sparsity survives, else a dense ndarray.

    Only single-input / single-output `f` with 1D `y` is currently
    supported — see `docs/TODO.md` for the multi-input / multi-output
    roadmap.
    """
    def wrapped(y):
        return _jacfwd_impl(f, y, format)
    return wrapped


def bcoo_jacfwd(f):
    """Forward-mode Jacobian returned as BCOO. Alias for
    `jacfwd(f, format='bcoo')`."""
    return jacfwd(f, format="bcoo")


def jacrev(f, *, format: str = "dense"):
    """Reverse-mode Jacobian, matching `jax.jacrev`'s output shape.

    Equivalent to `materialize(linear_transpose(linearize(f, y)[1], y),
    y_out, format=format).T`.

    Returns a function `(y) -> Jacobian`.
    """
    def wrapped(y):
        return _jacrev_impl(f, y, format)
    return wrapped


def bcoo_jacrev(f):
    """Reverse-mode Jacobian returned as BCOO. Alias for
    `jacrev(f, format='bcoo')`."""
    return jacrev(f, format="bcoo")


def hessian(f, *, format: str = "dense"):
    """Hessian, matching `jax.hessian`'s output shape.

    Equivalent to `materialize(jax.linearize(jax.grad(f), y)[1], y,
    format=format)`.

    Returns a function `(y) -> Hessian`.
    """
    def wrapped(y):
        _, lin = jax.linearize(jax.grad(f), y)
        return materialize(lin, y, format=format)
    return wrapped


def bcoo_hessian(f):
    """Hessian returned as BCOO. Alias for `hessian(f, format='bcoo')`."""
    return hessian(f, format="bcoo")


# -------------------------- demo --------------------------


def _demo():
    from sif2jax.cutest._quadratic_problems.dual1 import DUAL1
    from sif2jax.cutest._quadratic_problems.dual2 import DUAL2
    from sif2jax.cutest._quadratic_problems.dual3 import DUAL3
    from sif2jax.cutest._quadratic_problems.dual4 import DUAL4

    for cls in (DUAL1, DUAL2, DUAL3, DUAL4):
        p = cls()
        y = p.y0

        def f(z):
            return p.objective(z, p.args)

        _, hvp = jax.linearize(jax.grad(f), y)
        H_ref = jax.vmap(hvp)(jnp.eye(p.n)).T
        H_ours = materialize(hvp, y)
        err = float(jnp.max(jnp.abs(H_ours - H_ref)))
        sym = float(jnp.max(jnp.abs(H_ours - H_ours.T)))
        print(
            f"{cls.__name__:6s}  n={p.n:3d}  "
            f"max|H_ours - H_ref| = {err:.2e}  |H - H.T| = {sym:.2e}"
        )

        def run_jit(y_):
            _, hvp_ = jax.linearize(jax.grad(f), y_)
            return materialize(hvp_, y_)

        H_jit = jax.jit(run_jit)(y)
        err_jit = float(jnp.max(jnp.abs(H_jit - H_ref)))
        print(f"         inside-jit err: {err_jit:.2e}")


if __name__ == "__main__":
    _demo()
