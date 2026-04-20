"""Coloring-free Jacobian extraction for linear callables.

Two public entry points:

* `materialize(linear_fn, primal) -> jnp.ndarray` — always returns a dense
  matrix, mirroring `jax.hessian`'s output convention.
* `bcoo_jacobian(linear_fn, primal) -> jnp.ndarray | sparse.BCOO` — returns
  a `BCOO` when the linear function's structure is sparse-friendly, otherwise
  a dense `jnp.ndarray`. Use when downstream code can consume `BCOO` matvecs
  / `sparsify`-style ops directly.

Both work by tracing `linear_fn` to a jaxpr and walking its equations with
per-primitive rules that propagate structural per-var operators. The internal
forms (`ConstantDiagonal`, `Diagonal`) are private — they let common patterns
(scalar · I, vector-scaled I, sparse banded blocks) avoid materialising
intermediate identity matrices, but they are converted to BCOO or dense at
the boundary.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.lax import slicing as _slicing
from jax.experimental import sparse
from jax.extend import core


# -------------------------- internal structural forms --------------------------
# These are *internal*: they live only inside the env during a single walk.
# Public APIs always return BCOO or ndarray.


class ConstantDiagonal:
    """Internal: diagonal matrix with all entries equal to `value`."""

    __slots__ = ("n", "value")

    def __init__(self, n: int, value: Any = 1.0):
        self.n = n
        self.value = value


class Diagonal:
    """Internal: diagonal matrix `diag(values)` for a length-n vector."""

    __slots__ = ("values",)

    def __init__(self, values):
        self.values = values

    @property
    def n(self):
        return self.values.shape[0]


class Pivoted:
    """Internal: a linear operator with at most one nonzero per row.

    Represents the `(out_size, in_size)` matrix
        M[out_rows[i], in_cols[i]] = values[i]
        M[r, c] = 0 otherwise.

    Captures `slice(Identity)`, `gather(Identity)`, scaled versions, and
    pad/add chains thereof — common in sparse-banded problems.
    Avoids BCOO's `(nse, 2)` indices array by keeping rows + cols as 1D.
    """

    __slots__ = ("out_rows", "in_cols", "values", "out_size", "in_size")

    def __init__(self, out_rows, in_cols, values, out_size, in_size):
        self.out_rows = out_rows
        self.in_cols = in_cols
        self.values = values
        self.out_size = out_size
        self.in_size = in_size

    @property
    def shape(self):
        return (self.out_size, self.in_size)

    @property
    def n(self):
        return self.in_size

    @property
    def nse(self):
        return self.out_rows.shape[0]


def _to_dense(op, n: int) -> jnp.ndarray:
    if isinstance(op, ConstantDiagonal):
        if isinstance(op.value, float) and op.value == 1.0:
            return jnp.eye(n)
        return op.value * jnp.eye(n)
    if isinstance(op, Diagonal):
        m = op.values.shape[0]
        idx = jnp.arange(m)
        return jnp.zeros((m, m), op.values.dtype).at[idx, idx].set(op.values)
    if isinstance(op, Pivoted):
        return (jnp.zeros((op.out_size, op.in_size), op.values.dtype)
                .at[op.out_rows, op.in_cols].add(op.values))
    if isinstance(op, sparse.BCOO):
        return op.todense()
    return op


def _pivoted_to_bcoo(p: "Pivoted") -> sparse.BCOO:
    if isinstance(p.out_rows, np.ndarray) and isinstance(p.in_cols, np.ndarray):
        # Stack statically; avoid two asarray + one jnp.stack HLO ops.
        indices = np.stack([p.out_rows, p.in_cols], axis=1)
    else:
        indices = jnp.stack([jnp.asarray(p.out_rows), jnp.asarray(p.in_cols)], axis=1)
    return sparse.BCOO((p.values, indices), shape=p.shape)


def _concat(arrs):
    arrs = list(arrs)
    if all(isinstance(a, np.ndarray) for a in arrs):
        return np.concatenate(arrs)
    return jnp.concatenate(arrs)


def _diag_to_bcoo(d, n=None) -> sparse.BCOO:
    """Convert a (Constant)Diagonal to BCOO."""
    idx = jnp.arange(d.n)
    indices = jnp.stack([idx, idx], axis=1)
    if isinstance(d, ConstantDiagonal):
        v = jnp.asarray(d.value)
        data = jnp.broadcast_to(v, (d.n,))
    elif isinstance(d, Diagonal):
        data = d.values
    else:
        raise TypeError(f"_diag_to_bcoo expected diagonal LinOp, got {type(d)}")
    return sparse.BCOO((data, indices), shape=(d.n, d.n))


def _to_bcoo(op, n: int):
    """Convert any internal LinOp to BCOO (used at the bcoo_jacobian boundary)."""
    if isinstance(op, sparse.BCOO):
        return op
    if isinstance(op, (ConstantDiagonal, Diagonal)):
        return _diag_to_bcoo(op)
    if isinstance(op, Pivoted):
        return _pivoted_to_bcoo(op)
    return op  # plain ndarray — caller will keep dense


def _traced_shape(op) -> tuple:
    if isinstance(op, (ConstantDiagonal, Diagonal)):
        return (op.n,)
    if isinstance(op, Pivoted):
        return (op.out_size,)
    return tuple(op.shape[:-1])


# -------------------------- rule registry --------------------------


materialize_rules: dict[core.Primitive, Callable] = {}


def register(prim: core.Primitive):
    def deco(fn):
        materialize_rules[prim] = fn
        return fn

    return deco


# -------------------------- rules --------------------------


@register(lax.mul_p)
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
        scale = jnp.asarray(scale).reshape(())
        if isinstance(traced_op, ConstantDiagonal):
            return ConstantDiagonal(traced_op.n, scale * traced_op.value)
        if isinstance(traced_op, Diagonal):
            return Diagonal(scale * traced_op.values)
        if isinstance(traced_op, Pivoted):
            return Pivoted(traced_op.out_rows, traced_op.in_cols,
                           scale * traced_op.values,
                           traced_op.out_size, traced_op.in_size)
        if isinstance(traced_op, sparse.BCOO):
            return sparse.BCOO(
                (scale * traced_op.data, traced_op.indices),
                shape=traced_op.shape,
            )
        return scale * traced_op
    if isinstance(traced_op, ConstantDiagonal):
        return Diagonal(traced_op.value * jnp.asarray(scale))
    if isinstance(traced_op, Diagonal):
        return Diagonal(traced_op.values * jnp.asarray(scale))
    if isinstance(traced_op, Pivoted):
        # scale has shape (out_size,) matching traced_op's output dim. Scale
        # values by scale[out_rows].
        scale_arr = jnp.asarray(scale)
        if scale_arr.shape[0] == traced_op.nse:
            # Special case: scale length equals nse (true when out_rows ==
            # arange(k), i.e. before any pad). Avoid the gather.
            new_values = traced_op.values * scale_arr
        else:
            new_values = traced_op.values * jnp.take(scale_arr, traced_op.out_rows)
        return Pivoted(traced_op.out_rows, traced_op.in_cols, new_values,
                       traced_op.out_size, traced_op.in_size)
    if isinstance(traced_op, sparse.BCOO):
        row_idx = traced_op.indices[:, 0]
        scale_arr = jnp.asarray(scale)
        return sparse.BCOO(
            (traced_op.data * jnp.take(scale_arr, row_idx), traced_op.indices),
            shape=traced_op.shape,
        )
    dense = _to_dense(traced_op, n)
    return scale[..., None] * dense


def _add_like(invals, traced, n, **params):
    del params
    vals = [v for v, t in zip(invals, traced) if t]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    if all(isinstance(v, ConstantDiagonal) and v.n == vals[0].n for v in vals):
        return ConstantDiagonal(vals[0].n, sum(v.value for v in vals))
    if all(isinstance(v, (ConstantDiagonal, Diagonal)) and v.n == vals[0].n
           for v in vals):
        # Dtype from the first operand's values where available.
        dtype = next((v.values.dtype for v in vals if isinstance(v, Diagonal)),
                     jnp.result_type(float))
        total = jnp.zeros(vals[0].n, dtype=dtype)
        for v in vals:
            total = total + (
                jnp.broadcast_to(v.value, (v.n,))
                if isinstance(v, ConstantDiagonal) else v.values
            )
        return Diagonal(total)
    # Two Pivoteds at the same shape: try the same-indices fast path first.
    if all(isinstance(v, Pivoted) and v.shape == vals[0].shape for v in vals):
        # Same-indices fast path: if all out_rows AND all in_cols are
        # statically equal, just sum the values — no entry duplication.
        first = vals[0]
        if (isinstance(first.out_rows, np.ndarray)
                and isinstance(first.in_cols, np.ndarray)
                and all(
                    isinstance(v.out_rows, np.ndarray)
                    and isinstance(v.in_cols, np.ndarray)
                    and np.array_equal(v.out_rows, first.out_rows)
                    and np.array_equal(v.in_cols, first.in_cols)
                    for v in vals[1:]
                )):
            summed = vals[0].values
            for v in vals[1:]:
                summed = summed + v.values
            return Pivoted(first.out_rows, first.in_cols, summed,
                           first.out_size, first.in_size)
        # Fallback: concat entries. Disjoint row sets stay "one per row";
        # overlaps get fixed at densification by scatter-add.
        return Pivoted(
            _concat(v.out_rows for v in vals),
            _concat(v.in_cols for v in vals),
            jnp.concatenate([v.values for v in vals]),
            vals[0].out_size, vals[0].in_size,
        )
    if all(isinstance(v, sparse.BCOO) and v.shape == vals[0].shape for v in vals):
        return sparse.BCOO(
            (jnp.concatenate([v.data for v in vals]),
             jnp.concatenate([v.indices for v in vals])),
            shape=vals[0].shape,
        )
    # Mixed Pivoted + BCOO at same shape: convert Pivoted to BCOO and concat.
    if all(isinstance(v, (Pivoted, sparse.BCOO)) and v.shape == vals[0].shape
           for v in vals):
        bcoo_vals = [_pivoted_to_bcoo(v) if isinstance(v, Pivoted) else v
                     for v in vals]
        return sparse.BCOO(
            (jnp.concatenate([v.data for v in bcoo_vals]),
             jnp.concatenate([v.indices for v in bcoo_vals])),
            shape=vals[0].shape,
        )
    # Mixed Pivoted + (Constant)Diagonal at compatible shape: convert all to
    # BCOO and concatenate (Pivoted is square iff out_size == in_size).
    if all(isinstance(v, (Pivoted, ConstantDiagonal, Diagonal)) for v in vals):
        # Check shapes are compatible (square n×n where n is consistent).
        sizes = [v.out_size if isinstance(v, Pivoted) else v.n for v in vals]
        in_sizes = [v.in_size if isinstance(v, Pivoted) else v.n for v in vals]
        if all(s == sizes[0] == in_sizes[0] for s in sizes + in_sizes):
            n_out = sizes[0]
            bcoo_vals = []
            for v in vals:
                if isinstance(v, Pivoted):
                    bcoo_vals.append(_pivoted_to_bcoo(v))
                else:
                    bcoo_vals.append(_diag_to_bcoo(v, n))
            return sparse.BCOO(
                (jnp.concatenate([v.data for v in bcoo_vals]),
                 jnp.concatenate([v.indices for v in bcoo_vals])),
                shape=(n_out, n_out),
            )
    if all(isinstance(v, (ConstantDiagonal, Diagonal, sparse.BCOO)) for v in vals):
        n_out = (vals[0].n if isinstance(vals[0], (ConstantDiagonal, Diagonal))
                 else vals[0].shape[0])
        if all(
            (isinstance(v, sparse.BCOO) and v.shape == (n_out, n_out))
            or (isinstance(v, (ConstantDiagonal, Diagonal)) and v.n == n_out)
            for v in vals
        ):
            bcoo_vals = [
                _diag_to_bcoo(v, n) if not isinstance(v, sparse.BCOO) else v
                for v in vals
            ]
            return sparse.BCOO(
                (jnp.concatenate([v.data for v in bcoo_vals]),
                 jnp.concatenate([v.indices for v in bcoo_vals])),
                shape=(n_out, n_out),
            )
    dense_vals = [_to_dense(v, n) for v in vals]
    result = dense_vals[0]
    for t in dense_vals[1:]:
        result = result + t
    return result


register(lax.add_p)(_add_like)
try:
    from jax._src.ad_util import add_jaxvals_p

    register(add_jaxvals_p)(_add_like)
except ImportError:
    pass


def _identity_rule(invals, traced, n, **params):
    """For primitives that don't change value (convert_element_type, copy)."""
    del params
    (op,) = invals
    (t,) = traced
    return op if t else None


register(lax.convert_element_type_p)(_identity_rule)
register(lax.copy_p)(_identity_rule)


@register(lax.neg_p)
def _neg_rule(invals, traced, n, **params):
    del params
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    if isinstance(op, ConstantDiagonal):
        return ConstantDiagonal(op.n, -op.value)
    if isinstance(op, Diagonal):
        return Diagonal(-op.values)
    if isinstance(op, Pivoted):
        return Pivoted(op.out_rows, op.in_cols, -op.values,
                       op.out_size, op.in_size)
    if isinstance(op, sparse.BCOO):
        return sparse.BCOO((-op.data, op.indices), shape=op.shape)
    return -op


@register(lax.sub_p)
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
    return _add_like([a, neg_b], [True, True], n)


@register(lax.dot_general_p)
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


@register(lax.slice_p)
def _slice_rule(invals, traced, n, **params):
    (operand,) = invals
    (to,) = traced
    if not to:
        return None
    starts = tuple(int(s) for s in params["start_indices"])
    limits = tuple(int(l) for l in params["limit_indices"])
    strides_p = params.get("strides")
    strides = tuple(int(s) for s in strides_p) if strides_p else (1,) * len(starts)

    # Structural fast path: 1D operand with unit stride and structural form.
    if (len(starts) == 1 and strides == (1,)
            and isinstance(operand, (ConstantDiagonal, Diagonal, Pivoted, sparse.BCOO))):
        s, e = starts[0], limits[0]
        k = e - s
        if isinstance(operand, ConstantDiagonal):
            # Static numpy indices let downstream add_any dedup-by-equality.
            return Pivoted(
                np.arange(k),
                np.arange(s, e),
                jnp.broadcast_to(jnp.asarray(operand.value), (k,)),
                k, operand.n,
            )
        if isinstance(operand, Diagonal):
            return Pivoted(
                np.arange(k),
                np.arange(s, e),
                operand.values[s:e],
                k, operand.n,
            )
        if isinstance(operand, Pivoted):
            # Filter entries whose out_row is in [s, e); shift by -s. For
            # masked-out entries we both zero the value AND clip the row
            # index to a valid value (0) — the value is 0 so position
            # doesn't matter, but the index must be in-bounds for scatter.
            mask = (operand.out_rows >= s) & (operand.out_rows < e)
            shifted = operand.out_rows - s
            safe_rows = jnp.where(mask, shifted, 0)
            return Pivoted(
                safe_rows,
                operand.in_cols,
                operand.values * mask,
                k, operand.in_size,
            )
        rows = operand.indices[:, 0]
        mask = (rows >= s) & (rows < e)
        new_data = operand.data * mask
        new_rows = rows - s
        new_indices = jnp.stack([new_rows, operand.indices[:, 1]], axis=1)
        return sparse.BCOO((new_data, new_indices), shape=(k, operand.shape[1]))

    # Fallback: densify and slice along output (non-input) axes; preserve the
    # trailing input-coordinate axis with start=0, limit=n, stride=1.
    dense = _to_dense(operand, n)
    s_full = starts + (0,)
    l_full = limits + (n,)
    str_full = strides + (1,)
    return lax.slice(dense, s_full, l_full, str_full)


@register(lax.pad_p)
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
    if (isinstance(operand, Pivoted) and len(config) == 1
            and int(interior) == 0):
        out_size = operand.out_size + before + after
        new_rows = operand.out_rows + before
        if before >= 0 and after >= 0:
            return Pivoted(new_rows, operand.in_cols, operand.values,
                           out_size, operand.in_size)
        # Negative pad = truncation: drop entries with new_row out of range.
        is_np = isinstance(new_rows, np.ndarray)
        mask = (new_rows >= 0) & (new_rows < out_size)
        safe_rows = (np.where(mask, new_rows, 0) if is_np
                     else jnp.where(mask, new_rows, 0))
        val_mask = jnp.asarray(mask, dtype=operand.values.dtype) if is_np else mask
        return Pivoted(safe_rows, operand.in_cols, operand.values * val_mask,
                       out_size, operand.in_size)
    if (isinstance(operand, sparse.BCOO) and len(config) == 1
            and int(interior) == 0):
        out_size = operand.shape[0] + before + after
        new_rows = operand.indices[:, 0] + before
        new_indices = jnp.stack([new_rows, operand.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (operand.data, new_indices), shape=(out_size, operand.shape[1])
        )
    # Dense fallback: pad along output axes (input axis untouched).
    dense = _to_dense(operand, n)
    full_config = tuple((int(b), int(a), int(i)) for (b, a, i) in config) + ((0, 0, 0),)
    return lax.pad(dense, jnp.asarray(0.0, dtype=dense.dtype), full_config)


@register(lax.squeeze_p)
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
    if isinstance(op, (Pivoted, sparse.BCOO)):
        # Densify (sparse → (out_size, in_size)) then squeeze leading axes.
        return lax.squeeze(_to_dense(op, n), dimensions)
    # Dense: squeeze the specified axes (always output axes, never the last
    # input-coordinate axis).
    return lax.squeeze(op, dimensions)


@register(lax.rev_p)
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


@register(lax.reshape_p)
def _reshape_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    new_sizes = params["new_sizes"]
    dense = _to_dense(op, n)
    # Reshape applies to output axes only; preserve the trailing input axis.
    return lax.reshape(dense, tuple(new_sizes) + (n,))


@register(lax.broadcast_in_dim_p)
def _broadcast_in_dim_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    shape = params["shape"]
    broadcast_dimensions = params["broadcast_dimensions"]
    dense = _to_dense(op, n)
    # Map each output axis to the corresponding input axis (or broadcast it).
    out_dims = tuple(broadcast_dimensions) + (len(shape),)  # add input axis
    return lax.broadcast_in_dim(dense, tuple(shape) + (n,), out_dims)


@register(lax.reduce_sum_p)
def _reduce_sum_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    axes = params["axes"]
    dense = _to_dense(op, n)
    # Sum applies to output axes only — never to the last (input-coordinate) axis.
    return jnp.sum(dense, axis=tuple(axes))


@register(lax.concatenate_p)
def _concatenate_rule(invals, traced, n, **params):
    if not any(traced):
        return None
    dimension = params["dimension"]
    # Densify any traced operands; treat closure operands as their concrete
    # values (they have shape NOT carrying an input axis, so we need to add it).
    parts = []
    for v, t in zip(invals, traced):
        if t:
            parts.append(_to_dense(v, n))
        else:
            # Closure constant: extend with a zero "input axis" of size n.
            parts.append(jnp.broadcast_to(v[..., None] * 0, v.shape + (n,)))
    return lax.concatenate(parts, dimension)


@register(lax.split_p)
def _split_rule(invals, traced, n, **params):
    (operand,) = invals
    (t,) = traced
    if not t:
        return None
    sizes = params["sizes"]
    axis = params["axis"]
    dense = _to_dense(operand, n)
    out = []
    start = 0
    for sz in sizes:
        slc = [slice(None)] * dense.ndim
        slc[axis] = slice(int(start), int(start) + int(sz))
        out.append(dense[tuple(slc)])
        start += int(sz)
    return out


def _jit_rule(invals, traced, n, **params):
    """Recurse into the inner jaxpr of a `jit` (pjit) call."""
    inner_cj = params["jaxpr"]  # ClosedJaxpr
    inner = inner_cj.jaxpr
    inner_consts = inner_cj.consts

    inner_env: dict = {}
    inner_consts_env: dict = dict(zip(inner.constvars, inner_consts))
    for inner_invar, outer_val, was_traced in zip(inner.invars, invals, traced):
        if was_traced:
            inner_env[inner_invar] = outer_val
        else:
            inner_consts_env[inner_invar] = outer_val
    _walk_jaxpr(inner, inner_env, inner_consts_env, n)

    outs = []
    for outvar in inner.outvars:
        if outvar in inner_env:
            outs.append(inner_env[outvar])
        else:
            outs.append(inner_consts_env[outvar])
    return outs  # jit_p is always multiple_results


# pjit_p is the modern name for jit's primitive.
try:
    from jax._src.pjit import jit_p

    register(jit_p)(_jit_rule)
except ImportError:
    pass


@register(lax.select_n_p)
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


@register(lax.cumsum_p)
def _cumsum_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    axis = params["axis"]
    reverse = params.get("reverse", False)
    dense = _to_dense(op, n)
    return lax.cumsum(dense, axis=axis, reverse=reverse)


@register(lax.div_p)
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


@register(lax.transpose_p)
def _transpose_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    permutation = params["permutation"]
    dense = _to_dense(op, n)
    # Permutation applies to output axes only; preserve trailing input axis.
    return lax.transpose(dense, tuple(permutation) + (len(permutation),))


@register(lax.gather_p)
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
    if isinstance(operand, ConstantDiagonal):
        k = row_idx.shape[0]
        return Pivoted(
            np.arange(k),
            row_idx,
            jnp.broadcast_to(jnp.asarray(operand.value), (k,)),
            k, operand.n,
        )
    if isinstance(operand, sparse.BCOO):
        raise NotImplementedError("gather on BCOO operand")
    # Dense fallback: gather rows of the dense linop.
    dense = _to_dense(operand, n)
    return dense[row_idx]


@register(_slicing.scatter_add_p)
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
    # Out_idx comes from scatter_indices[..., 0]. If scatter_indices is 2D+,
    # out_idx is 2D+ too; flatten the batch dims so we treat each element as
    # a scalar target position.
    out_idx_flat = out_idx.reshape(-1)
    updates_nse_dims = len(updates.shape) - 1  # all axes except the input-coord
    if isinstance(updates, Pivoted):
        # Pivoted's out_rows are 1D. If updates batches more dims, we need to
        # flatten them. For now, support the 1D case directly and 2D via
        # densify-and-scatter.
        if updates.values.ndim == 1:
            return Pivoted(
                out_idx_flat[updates.out_rows],
                updates.in_cols,
                updates.values,
                out_size, n,
            )
        # Fall through to dense
        updates = _to_dense(updates, n)
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


# -------------------------- driver --------------------------


def _walk_jaxpr(jaxpr, env, consts_env, n):
    """Walk a jaxpr, mutating env. Used by both top-level and nested jit calls.

    Each var is in exactly one of:
      * `env` (traced — has a LinOp value)
      * `consts_env` (not traced — has a concrete array value)
      * neither (a Literal, read directly from .val).
    """

    def read(atom):
        if isinstance(atom, core.Literal):
            return atom.val
        if atom in env:
            return env[atom]
        return consts_env[atom]

    def is_traced(atom):
        return not isinstance(atom, core.Literal) and atom in env

    for eqn in jaxpr.eqns:
        invals = [read(v) for v in eqn.invars]
        traced = [is_traced(v) for v in eqn.invars]
        if not any(traced):
            # No traced inputs — evaluate the primitive concretely and stash
            # the result in consts_env. This is the "constant propagation"
            # path for vars derived purely from closures.
            concrete_outs = eqn.primitive.bind(*invals, **eqn.params)
            if eqn.primitive.multiple_results:
                for v, o in zip(eqn.outvars, concrete_outs):
                    consts_env[v] = o
            else:
                (outvar,) = eqn.outvars
                consts_env[outvar] = concrete_outs
            continue
        rule = materialize_rules.get(eqn.primitive)
        if rule is None:
            raise NotImplementedError(
                f"No materialize rule for primitive {eqn.primitive}"
            )
        outs = rule(invals, traced, n, **eqn.params)
        if eqn.primitive.multiple_results:
            for v, o in zip(eqn.outvars, outs):
                env[v] = o
        else:
            (outvar,) = eqn.outvars
            env[outvar] = outs


def _walk(linear_fn, primal):
    cj = jax.make_jaxpr(linear_fn)(primal)
    jaxpr = cj.jaxpr

    if len(jaxpr.invars) != 1:
        raise NotImplementedError("multi-input linear_fn not yet handled")
    (invar,) = jaxpr.invars
    if invar.aval.ndim != 1:
        raise NotImplementedError("non-1D input not yet handled")
    n = invar.aval.size

    consts_env: dict = dict(zip(jaxpr.constvars, cj.consts))
    env: dict = {invar: ConstantDiagonal(n)}
    _walk_jaxpr(jaxpr, env, consts_env, n)

    if len(jaxpr.outvars) != 1:
        raise NotImplementedError("multi-output linear_fn not yet handled")
    (outvar,) = jaxpr.outvars
    return (env[outvar] if outvar in env else consts_env[outvar]), n


_SMALL_N_VMAP_THRESHOLD = 16
"""Below this n, vmap(linear_fn)(eye) emits less HLO than the structural walk
on most problems. Above it the walk's structure exploitation dominates."""


def materialize(linear_fn, primal):
    """Return the dense `jnp.ndarray` matrix of the linear function `linear_fn`.

    Mirrors `jax.hessian`'s output convention.

    For tiny inputs (n < `_SMALL_N_VMAP_THRESHOLD`) the structural walk emits
    more HLO than the simple `vmap(linear_fn)(eye)` does — short-circuit.
    """
    n = primal.size if hasattr(primal, "size") else int(jnp.size(primal))
    if n < _SMALL_N_VMAP_THRESHOLD:
        return jax.vmap(linear_fn)(jnp.eye(n, dtype=primal.dtype)).T
    op, n2 = _walk(linear_fn, primal)
    return _to_dense(op, n2)


def bcoo_jacobian(linear_fn, primal):
    """Return a `jax.experimental.sparse.BCOO` if the linear function has
    sparse structure that survives the walk, otherwise a dense `jnp.ndarray`.
    """
    op, n = _walk(linear_fn, primal)
    if isinstance(op, (ConstantDiagonal, Diagonal, Pivoted, sparse.BCOO)):
        return _to_bcoo(op, n)
    return op


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
