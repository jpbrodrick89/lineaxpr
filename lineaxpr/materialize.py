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
    Identity,
    Pivoted,
    _concat,
    _diag_to_bcoo,
    _pivoted_to_bcoo,
    _to_bcoo,
    _to_dense,
    _traced_shape,
)


# -------------------------- rule registry --------------------------


materialize_rules: dict[core.Primitive, Callable] = {}


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
        if isinstance(traced_op, (ConstantDiagonal, Diagonal, Pivoted)):
            return traced_op.scale_scalar(s)
        if isinstance(traced_op, sparse.BCOO):
            return _bcoo_scale_scalar(traced_op, s)
        return s * traced_op
    if isinstance(traced_op, (ConstantDiagonal, Diagonal, Pivoted)):
        return traced_op.scale_per_out_row(scale)
    if isinstance(traced_op, sparse.BCOO):
        return _bcoo_scale_per_out_row(traced_op, scale)
    dense = _to_dense(traced_op, n)
    return scale[..., None] * dense

materialize_rules[lax.mul_p] = _mul_rule

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


materialize_rules[lax.add_p] = _add_like
try:
    from jax._src.ad_util import add_jaxvals_p

    materialize_rules[add_jaxvals_p] = _add_like
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
    if isinstance(op, (ConstantDiagonal, Diagonal, Pivoted)):
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
    return _add_like([a, neg_b], [True, True], n)

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
    if (isinstance(operand, Pivoted) and len(config) == 1
            and int(interior) == 0):
        return operand.pad_rows(before, after)
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
    if isinstance(op, (Pivoted, sparse.BCOO)):
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
    new_sizes = params["new_sizes"]
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
    dense = _to_dense(op, n)
    # Map each output axis to the corresponding input axis (or broadcast it).
    out_dims = tuple(broadcast_dimensions) + (len(shape),)  # add input axis
    return lax.broadcast_in_dim(dense, tuple(shape) + (n,), out_dims)

materialize_rules[lax.broadcast_in_dim_p] = _broadcast_in_dim_rule

def _reduce_sum_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    axes = params["axes"]
    dense = _to_dense(op, n)
    # Sum applies to output axes only — never to the last (input-coordinate) axis.
    return jnp.sum(dense, axis=tuple(axes))

materialize_rules[lax.reduce_sum_p] = _reduce_sum_rule

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

materialize_rules[lax.concatenate_p] = _concatenate_rule

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
    `bcoo_jacobian` build `Identity(primal.size, dtype=primal.dtype)` and
    pass it through.
    """
    def inner(seed_linop):
        return _walk_with_seed(linear_fn, seed_linop)

    return inner


# Back-compat for direct users of `_walk`.
def _walk(linear_fn, primal):
    seed = ConstantDiagonal(primal.size)
    out = _walk_with_seed(linear_fn, seed)
    return out, primal.size


_SMALL_N_VMAP_THRESHOLD = 16
"""Below this n, vmap(linear_fn)(eye) emits less HLO than the structural walk
on most problems. Above it the walk's structure exploitation dominates."""


def to_dense(op):
    """Densify a LinOp returned by `sparsify` to a jnp.ndarray.

    Uniform across all possible return types:
    - Our LinOp classes (ConstantDiagonal, Diagonal, Pivoted) → `.todense()`.
    - `jax.experimental.sparse.BCOO` → `.todense()`.
    - Plain ndarray → passthrough.
    """
    if isinstance(op, (ConstantDiagonal, Diagonal, Pivoted)):
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
    if isinstance(op, (ConstantDiagonal, Diagonal, Pivoted)):
        return op.to_bcoo()
    return op


# Legacy internal aliases (still referenced in benchmarks / older tests).
_linop_to_dense = lambda op, n: to_dense(op)  # noqa: E731
_linop_to_bcoo = lambda op: to_bcoo(op)  # noqa: E731


def materialize(linear_fn, primal):
    """Return the dense `jnp.ndarray` matrix of the linear function `linear_fn`.

    Mirrors `jax.hessian`'s output convention.

    For tiny inputs (n < `_SMALL_N_VMAP_THRESHOLD`) the structural walk emits
    more HLO than the simple `vmap(linear_fn)(eye)` does — short-circuit.
    """
    n = primal.size if hasattr(primal, "size") else int(jnp.size(primal))
    if n < _SMALL_N_VMAP_THRESHOLD:
        return jax.vmap(linear_fn)(jnp.eye(n, dtype=primal.dtype)).T
    seed = Identity(n, dtype=primal.dtype)
    return to_dense(sparsify(linear_fn)(seed))


def bcoo_jacobian(linear_fn, primal):
    """Return a `jax.experimental.sparse.BCOO` if the linear function has
    sparse structure that survives the walk, otherwise a dense `jnp.ndarray`.
    """
    n = primal.size if hasattr(primal, "size") else int(jnp.size(primal))
    seed = Identity(n, dtype=primal.dtype)
    return to_bcoo(sparsify(linear_fn)(seed))


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
