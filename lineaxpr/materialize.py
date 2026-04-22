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
"""

from __future__ import annotations

import functools
import operator
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
    # Out-size-broadcast path: scale expands a size-1 out axis to
    # `scale.shape[-1]`. Triggered by the NONMSQRT-class pattern where
    # an aval-(B, 1) BEllpack (from `bid`-trailing-singleton + slice /
    # reduce_sum chain) multiplies by a (B, S) closure — the primal
    # broadcasts to (B, S) and each of the S new rows is a scaled copy
    # of the traced op's single row. Stays structural: new BEllpack
    # batch_shape=batch, out_size=S, k unchanged; values are one
    # broadcast-mul (no per-band loop); cols broadcast statically
    # across the new out axis.
    #
    # Gated on `traced.k >= 2` as a proxy for "dense alternative is
    # expensive enough that sparse emit pays off". A K=1 traced BE
    # typically just arrived from `bid(row-vector)` or similar and the
    # downstream op is often `add_any(..., dense_closure)` which
    # densifies the mul result anyway — making the structural emit
    # pure overhead. K>=2 indicates the walk has already accumulated
    # bands (mul/add/concat), correlating with a larger dense
    # alternative where the scatter-into-zeros + dense-add path
    # beats XLA's fused broadcast-mul+add on small tensors. Measured
    # on EIGENALS/BLS/CLS (n=2550, dense ~6.4M fits in L3): without
    # gate, ~30ms regression vs master (97ms → 126ms) from the
    # extra structural-BE-then-densify work. NONMSQRT (k=70, dense
    # ~24M > L3) retains its 2.5× win under the gate.
    if (isinstance(traced_op, BEllpack)
            and traced_op.k >= 2
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
    """Return the shape used to check operand compatibility in
    `_add_rule`'s mix-path. For unbatched LinOps this is the natural
    2D `(out_size, in_size)`. For batched BEllpack this is the
    flattened shape `(prod(batch_shape) * out_size, in_size)` —
    matching what `_ellpack_to_bcoo_batched` emits (an unbatched
    flat BCOO). This lets batched BEllpacks flow through the BCOO-
    concat path consistently with unbatched operands."""
    if isinstance(v, (ConstantDiagonal, Diagonal)):
        return (v.n, v.n)
    if isinstance(v, BEllpack):
        prod_batch = 1
        for d in v.batch_shape:
            prod_batch *= d
        return (prod_batch * v.out_size, v.in_size)
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
    return sparse.BCOO(
        (jnp.concatenate([v.data for v in bcoo_vals], axis=n_batch),
         jnp.concatenate([v.indices for v in bcoo_vals], axis=n_batch)),
        shape=shape,
    )


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
        first = vals[0]
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
            # Different cols: widen bands. Values shape for n_batch=0 is
            # (nrows,) for k=1 or (nrows, k) for k>=2; concat along the
            # band axis (axis=1, i.e. the k dim). For batched values
            # (n_batch>0) the band axis is n_batch+1.
            n_batch = first.n_batch
            band_axis = n_batch + 1
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
            return BEllpack(first.start_row, first.end_row,
                           new_in_cols, new_values,
                           first.out_size, first.in_size,
                           batch_shape=first.batch_shape)

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
        # Non-scalar empty-contract = outer product.
        # Output shape: traced_shape + M.shape (or M.shape + traced_shape).
        # LinOp dense: A_out[..., n] where A_out[*out_shape] is the outer.
        # For traced A of shape (*t_shape, n), closure M of shape (*m_shape):
        #   result[i, j, :] = M[j] * A[i, :]   if traced_is_first
        #   result[j, i, :] = M[j] * A[i, :]   if traced_is_second
        dense = _to_dense(traced_op, n)
        if traced_is_first:
            # dense: (*t_shape, n), M: (*m_shape). Want (*t_shape, *m_shape, n).
            M_idx = (None,) * len(traced_shape) + (slice(None),) * M.ndim + (None,)
            d_idx = (slice(None),) * len(traced_shape) + (None,) * M.ndim + (slice(None),)
            return M[M_idx] * dense[d_idx]
        else:
            # Output: (*m_shape, *t_shape, n) = M[i, ...] * A[j, :]
            d_idx = (None,) * M.ndim + (slice(None),) * len(traced_shape) + (slice(None),)
            M_idx = (slice(None),) * M.ndim + (None,) * len(traced_shape) + (None,)
            return M[M_idx] * dense[d_idx]

    if isinstance(traced_op, ConstantDiagonal):
        remaining = [a for a in range(M.ndim) if a not in c_M]
        tensor = lax.transpose(M, remaining + c_M)
        return traced_op.value * tensor

    dense = _to_dense(traced_op, n)
    if traced_is_first:
        out = lax.dot_general(
            dense, M, (((tuple(c_tr), tuple(c_M))), ((), ()))
        )
        n_rem_tr = len(traced_shape) - len(c_tr)
        M_rank = M.ndim
        perm = (
            list(range(n_rem_tr))
            + list(range(n_rem_tr + 1, n_rem_tr + 1 + (M_rank - len(c_M))))
            + [n_rem_tr]
        )
        return lax.transpose(out, perm)
    return lax.dot_general(M, dense, (((tuple(c_M), tuple(c_tr))), ((), ())))

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
    # Pass-through: BCOO whose flat shape already equals the target
    # LinOp shape `(*new_sizes, n)`. Happens when batched BEllpacks
    # were converted to unbatched-flat BCOO by `_ellpack_to_bcoo_batched`
    # (static-prune path) and the subsequent reshape is the walk's
    # final aval-flatten — structurally a no-op.
    if (isinstance(op, sparse.BCOO) and op.n_batch == 0
            and len(new_sizes) == 1
            and op.shape == (int(new_sizes[0]), op.shape[-1])):
        return op

    # Structural path: batched BEllpack → unbatched flat BCOO when the
    # reshape fully flattens the leading (batch + out) axes into a
    # single aval dimension. Delegates to `_ellpack_to_bcoo_batched`,
    # which already emits a flat BCOO of shape
    # `(prod(batch) * out_size, in_size)` via
    # `flat_row = batch_flat * out_size + row_within_batch`. Handles
    # the final reshape in LUKSAN11-16's `jnp.stack` chain
    # (`bid → concat → reshape(flatten)`) so the whole chain stays
    # structural end-to-end. Target must be rank 1 and the total equal
    # `prod(batch) * out_size`; otherwise fall through.
    if (isinstance(op, BEllpack) and op.n_batch >= 1
            and len(new_sizes) == 1
            and int(np.prod(op.batch_shape)) * op.out_size
                == int(new_sizes[0])):
        return _ellpack_to_bcoo_batched(op)
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
    dense = _to_dense(op, n)
    # Map each output axis to the corresponding input axis (or broadcast it).
    out_dims = tuple(broadcast_dimensions) + (len(shape),)  # add input axis
    return lax.broadcast_in_dim(dense, tuple(shape) + (n,), out_dims)

materialize_rules[lax.broadcast_in_dim_p] = _broadcast_in_dim_rule

def _bellpack_row_sum(ep):
    """Sum the rows of an unbatched BEllpack, returning a 1D (in_size,)
    array of per-column totals. O(nrows · k) scatter-adds — no dense
    (out_size, in_size) materialisation.

    Used as `_reduce_sum_rule(ep, axes=(0,))`'s structural path: the
    row-sum is the row-vector coefficients of the resulting scalar-LinOp.
    """
    assert ep.n_batch == 0
    nrows = ep.nrows
    k = ep.k
    result = jnp.zeros((ep.in_size,), ep.dtype)
    for b in range(k):
        cols_b = _resolve_col(ep.in_cols[b], nrows)
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
            ok = True
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
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                if K == 1:
                    # (B, O) already in natural r-major order, k=O.
                    new_values = op.values
                else:
                    # (B, O, K) → flatten to (B, O*K) in r-major, b-minor order.
                    new_values = op.values.reshape(B, O * K)
                return BEllpack(
                    start_row=0, end_row=B,
                    in_cols=tuple(new_in_cols), values=new_values,
                    out_size=B, in_size=op.in_size,
                )
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
                        # 1D shared cols — all identical for a batch-axis
                        # concat to be structural. Fall back if they differ.
                        if all(np.array_equal(np.asarray(c), np.asarray(parts[0])) for c in parts[1:]):
                            new_in_cols.append(parts[0])
                        else:
                            new_in_cols = None
                            break
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
    # Structural path: split along output axis 0 (the "out_size" dim).
    # Promote (Constant)Diagonal/BEllpack → BCOO and split each chunk by
    # masking entries whose row falls outside its range.
    if axis == 0 and isinstance(operand, (ConstantDiagonal, Diagonal,
                                          BEllpack, sparse.BCOO)):
        bcoo = _to_bcoo(operand, n)
        rows = bcoo.indices[:, 0]
        out = []
        start = 0
        for sz in sizes:
            end = start + int(sz)
            in_range = (rows >= start) & (rows < end)
            # Shift rows into [0, sz) range for entries in this chunk;
            # entries outside get row=0 but data=0 so they're harmless.
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
    if (sum(case_traced) == 1
            and isinstance(t_case, BEllpack)
            and t_case.n_batch == 0):
        pred_arr = jnp.asarray(pred)
        pred_slice = pred_arr[t_case.start_row:t_case.end_row]
        mask = (pred_slice == t_idx)
        if t_case.values.ndim > 1:
            mask_b = mask[:, None]
        else:
            mask_b = mask
        new_values = jnp.where(mask_b, t_case.values,
                               jnp.zeros((), t_case.dtype))
        return BEllpack(
            t_case.start_row, t_case.end_row, t_case.in_cols,
            new_values, t_case.out_size, t_case.in_size,
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
    permutation = params["permutation"]
    dense = _to_dense(op, n)
    # Permutation applies to output axes only; preserve trailing input axis.
    return lax.transpose(dense, tuple(permutation) + (len(permutation),))

materialize_rules[lax.transpose_p] = _transpose_rule

def _gather_rule(invals, traced, n, **params):
    operand, start_indices = invals
    to, ti = traced
    if ti:
        raise NotImplementedError("gather with traced indices")
    if not to:
        return None
    dnums = params["dimension_numbers"]
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
