"""Primitive rule registry — maps jax.lax primitives to rule functions."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import lax
from jax._src.lax import slicing as _slicing
from jax.experimental import sparse
from jax.extend import core

from .add import _add_rule, BELLPACK_DEDUP_LIMIT, BELLPACK_DEDUP_VECTORISED_MIN
from .mul import _mul_rule
from .multilinear import _sub_rule, _dot_general_rule, _div_rule
from .control_flow import (
    _cond_rule,
    _jit_rule,
    _select_n_rule,
)
from .structural import _concatenate_rule
from .._linops import (
    BEllpack,
    ConstantDiagonal,
    Diagonal,
    LinOpProtocol,
    broadcast_in_dim_op,
    cumsum_op,
    gather_op,
    pad_op,
    reduce_sum_op,
    reshape_op,
    rev_op,
    scatter_add_op,
    slice_op,
    split_op,
    squeeze_op,
)
from .._linops.base import negate as _negate_dispatch

# Re-export constants so callers can reach them via materialize.BELLPACK_*
__all__ = [
    "materialize_rules",
    "BELLPACK_DEDUP_LIMIT",
    "BELLPACK_DEDUP_VECTORISED_MIN",
]


# ---------------------------------------------------------------------------
# Generic adapters
# ---------------------------------------------------------------------------

def _unary_rule(dispatch_fn):
    """Wrap a singledispatch unary op into the rule signature."""
    def rule(invals, traced, n, **params):
        (op,) = invals
        (t,) = traced
        if not t:
            return None
        return dispatch_fn(op, n=n, **params)
    return rule


def _identity_rule(invals, traced, n, **params):
    """For primitives that don't change value (convert_element_type, copy)."""
    del params
    (op,) = invals
    (t,) = traced
    return op if t else None


def _neg_rule(invals, traced, n, **params):
    del params, n
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    if isinstance(op, LinOpProtocol):
        return _negate_dispatch(op)
    return -op


# ---------------------------------------------------------------------------
# Pad rule (two-operand: operand + padding_value)
# ---------------------------------------------------------------------------

def _pad_rule(invals, traced, n, **params):
    operand, padding_value = invals
    to, tp = traced
    if tp:
        raise NotImplementedError("pad with traced padding_value")
    if not to:
        return None
    if hasattr(padding_value, "shape") and padding_value.shape != ():
        raise NotImplementedError("pad with non-scalar padding_value")
    return pad_op(operand, n=n, padding_value=padding_value, **params)


# ---------------------------------------------------------------------------
# Gather rule (two-operand: operand + start_indices)
# ---------------------------------------------------------------------------

def _gather_rule(invals, traced, n, **params):
    operand, start_indices = invals
    to, ti = traced
    if ti:
        raise NotImplementedError("gather with traced indices")
    if not to:
        return None
    params["dimension_numbers"]
    # For non-BEllpack BCOO operand, use BCOO-specific fallback.
    if isinstance(operand, sparse.BCOO):
        raise NotImplementedError("gather on BCOO operand")
    return gather_op(operand, n=n, start_indices=start_indices, **params)


# ---------------------------------------------------------------------------
# Scatter-add rule (three-operand: operand + scatter_indices + updates)
# ---------------------------------------------------------------------------

def _scatter_add_rule(invals, traced, n, **params):
    operand, scatter_indices, updates = invals
    to, ti, tu = traced
    if ti:
        raise NotImplementedError("scatter-add with traced indices")
    if to and not tu:
        # out = operand + scatter(constant_updates, indices)
        # Linear part: ∂out/∂input = ∂operand/∂input — the constant updates
        # vanish from the Jacobian. Return the operand's LinOp unchanged.
        return operand
    if to and tu:
        # out = operand.at[indices].add(updates)  — both linear in input.
        # J_out = J_operand + scatter(J_updates, indices).
        # J_operand and J_updates are both (*batch, rows, n) dense matrices.
        # Scatter J_updates rows into J_operand using the same index map.
        op_dense  = (operand.todense() if isinstance(operand, LinOpProtocol)
                     else jnp.asarray(operand))
        upd_dense = (updates.todense() if isinstance(updates, LinOpProtocol)
                     else jnp.asarray(updates))
        # out_rows: which rows of op_dense each update adds to.
        # upd_dense may have a leading batch-1 dim from MJX's vmap; flatten it.
        out_rows   = jnp.asarray(scatter_indices).reshape(-1)  # (update_count,)
        flat_upd   = upd_dense.reshape(-1, n)                  # (update_count, n)
        if op_dense.ndim == 2:
            # (out, n) — simple row-scatter
            return op_dense.at[out_rows].add(flat_upd)
        # (*batch, out, n) — scatter along the out axis for every batch element
        batch_shape = op_dense.shape[:-2]
        out_size    = op_dense.shape[-2]
        op_flat     = op_dense.reshape(-1, out_size, n)        # (B, out, n)
        upd_3d      = flat_upd.reshape(-1, len(out_rows), n)   # (B, update_count, n)
        result      = op_flat.at[:, out_rows, :].add(upd_3d)
        return result.reshape(op_dense.shape)
    if not tu:
        return None
    dnums = params["dimension_numbers"]

    # 2D point-scatter (HADAMALS-class): inserts updates[k] at
    # operand[scatter_indices[k,0], scatter_indices[k,1]].
    if (dnums.update_window_dims == ()
            and dnums.inserted_window_dims == (0, 1)
            and dnums.scatter_dims_to_operand_dims == (0, 1)
            and operand.ndim == 2):
        out_shape_2d = operand.shape
        updates_dense = updates.todense() if isinstance(updates, LinOpProtocol) else jnp.asarray(updates)
        si_flat = scatter_indices.reshape(-1, 2)
        updates_flat = updates_dense.reshape(-1, n)
        flat_rows = (si_flat[:, 0].astype(jnp.int64) * out_shape_2d[1]
                     + si_flat[:, 1].astype(jnp.int64))
        return (jnp.zeros((out_shape_2d[0] * out_shape_2d[1], n), updates_flat.dtype)
                .at[flat_rows].add(updates_flat)
                .reshape(out_shape_2d + (n,)))

    scatter_kept = (
        dnums.update_window_dims == (1,)
        and dnums.inserted_window_dims == ()
        and dnums.scatter_dims_to_operand_dims == (0,)
    )

    # Normalise "kept" form for non-BEllpack updates (BEllpack handles it).
    if scatter_kept and not isinstance(updates, BEllpack):
        if isinstance(updates, sparse.BCOO):
            updates = updates.todense().squeeze(axis=-2)
        else:
            updates = jnp.asarray(updates).squeeze(axis=-2)
        # Rebuild dnums for the collapsed form.
        from jax._src.lax import slicing as _sl  # noqa: PLC0415
        dnums = _sl.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        params = dict(params, dimension_numbers=dnums)

    # BCOO updates (non-BEllpack).
    if isinstance(updates, sparse.BCOO):
        scatter_collapsed = (
            dnums.update_window_dims == ()
            and dnums.inserted_window_dims == (0,)
            and dnums.scatter_dims_to_operand_dims == (0,)
        )
        if not scatter_collapsed:
            raise NotImplementedError(f"scatter-add with unhandled dnums: {dnums}")
        out_idx = scatter_indices[..., 0]
        out_idx_flat = out_idx.reshape(-1)
        out_size = operand.shape[0]
        new_rows = out_idx_flat[updates.indices[:, 0]]
        new_indices = jnp.stack([new_rows, updates.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (updates.data, new_indices), shape=(out_size, n)
        )

    # Dense/BEllpack updates: dispatch.
    return scatter_add_op(
        updates, n=n, operand=operand, scatter_indices=scatter_indices, **params
    )


# ---------------------------------------------------------------------------
# Scatter (assign) rule — lax.scatter_p  (lax.scatter, not scatter_add)
# ---------------------------------------------------------------------------

def _scatter_rule(invals, traced, n, **params):
    """Jacobian rule for lax.scatter (assignment scatter, not accumulation).

    scatter(operand, indices, updates) sets operand[indices] = updates.
    Jacobian: J_out[indices] = J_updates; J_out[non-indices] = J_operand.
    """
    operand, scatter_indices, updates = invals
    to, ti, tu = traced

    if ti:
        raise NotImplementedError("scatter with traced indices")

    dnums = params["dimension_numbers"]
    scatter_target_dim = (int(dnums.scatter_dims_to_operand_dims[0])
                          if dnums is not None else 0)
    out_idx_flat = jnp.asarray(scatter_indices)[..., 0].reshape(-1)

    if not to and not tu:
        return None

    if to and not tu:
        # Constant updates overwrite scattered rows → those rows have J=0.
        # Non-scattered rows retain J_operand.
        op_dense = (operand.todense() if isinstance(operand, LinOpProtocol)
                    else jnp.asarray(operand))
        idx = [slice(None)] * op_dense.ndim
        idx[scatter_target_dim] = out_idx_flat
        return op_dense.at[tuple(idx)].set(0)

    # tu=True: scatter J_updates into the output Jacobian.
    upd_dense = (updates.todense() if isinstance(updates, LinOpProtocol)
                 else jnp.asarray(updates))
    flat_upd = upd_dense.reshape(-1, n)

    if not to:
        # Only updates traced: J_out[non-scattered] = 0, J_out[scattered] = J_updates.
        out_shape = tuple(operand.shape) + (n,)
        result = jnp.zeros(out_shape, flat_upd.dtype)
        idx = [slice(None)] * (len(operand.shape) + 1)
        idx[scatter_target_dim] = out_idx_flat
        return result.at[tuple(idx)].set(flat_upd)

    # Both traced: merge J_operand (background) with J_updates (foreground).
    op_dense = (operand.todense() if isinstance(operand, LinOpProtocol)
                else jnp.asarray(operand))
    idx = [slice(None)] * op_dense.ndim
    idx[scatter_target_dim] = out_idx_flat
    return op_dense.at[tuple(idx)].set(flat_upd)


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

materialize_rules: dict[core.Primitive, Callable] = {}

materialize_rules[lax.mul_p] = _mul_rule
materialize_rules[lax.add_p] = _add_rule

try:
    from jax._src.ad_util import add_jaxvals_p
    materialize_rules[add_jaxvals_p] = _add_rule
except ImportError:
    pass

materialize_rules[lax.convert_element_type_p] = _identity_rule
materialize_rules[lax.copy_p] = _identity_rule
materialize_rules[lax.neg_p] = _neg_rule
materialize_rules[lax.sub_p] = _sub_rule
materialize_rules[lax.dot_general_p] = _dot_general_rule
materialize_rules[lax.slice_p] = _unary_rule(slice_op)
materialize_rules[lax.pad_p] = _pad_rule
materialize_rules[lax.squeeze_p] = _unary_rule(squeeze_op)
materialize_rules[lax.rev_p] = _unary_rule(rev_op)
materialize_rules[lax.reshape_p] = _unary_rule(reshape_op)
materialize_rules[lax.broadcast_in_dim_p] = _unary_rule(broadcast_in_dim_op)
materialize_rules[lax.reduce_sum_p] = _unary_rule(reduce_sum_op)
materialize_rules[lax.concatenate_p] = _concatenate_rule
materialize_rules[lax.split_p] = _unary_rule(split_op)

try:
    from jax._src.lax.control_flow.conditionals import cond_p
    materialize_rules[cond_p] = _cond_rule
except ImportError:
    pass

try:
    from jax._src.pjit import jit_p
    materialize_rules[jit_p] = _jit_rule
except ImportError:
    pass

materialize_rules[lax.select_n_p] = _select_n_rule
materialize_rules[lax.cumsum_p] = _unary_rule(cumsum_op)
materialize_rules[lax.div_p] = _div_rule
def _transpose_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    permutation = tuple(int(p) for p in params["permutation"])
    if isinstance(op, (ConstantDiagonal, Diagonal, BEllpack)):
        return op.transpose(axes=permutation)
    dense = op.todense() if isinstance(op, LinOpProtocol) else op
    return lax.transpose(dense, permutation + (len(permutation),))

materialize_rules[lax.transpose_p] = _transpose_rule
materialize_rules[lax.gather_p] = _gather_rule
materialize_rules[_slicing.scatter_add_p] = _scatter_add_rule
materialize_rules[_slicing.scatter_p] = _scatter_rule


# ---------------------------------------------------------------------------
# Linear-algebra primitives
# ---------------------------------------------------------------------------

def _triangular_solve_rule(invals, traced, n, **params):
    """Jacobian rule for lax.linalg.triangular_solve.

    Only the constant-a / traced-b case is supported (the common MJX path).
    J_b.shape = (*primal_b.shape, n).

    For left_side=True:  J_x = a^{-1} J_b (same solve on each column of J_b).
    For left_side=False: J_x = a^{-T} J_b — remap to left-solve with transposed a.

    When primal b is multi-dimensional (shape (*batch, m, k)), J_b has an extra
    k-dim before n: (*batch, m, k, n).  JAX requires a.ndim == b.ndim, so we
    merge the trailing (k, n) dims into one before the solve, then restore.
    """
    a, b = invals
    ta, tb = traced
    if ta:
        raise NotImplementedError(
            "triangular_solve with traced a (constant b) not supported")
    if not tb:
        return None
    J_b = b.todense() if isinstance(b, LinOpProtocol) else jnp.asarray(b)

    # Convert right-solve to left-solve with transposed a so shapes work out.
    if not params.get("left_side", True):
        params = dict(params, left_side=True,
                      transpose_a=not params.get("transpose_a", False))

    # J_b may have extra trailing dims vs a (when primal b is multi-column).
    # Merge everything after the m-dim into one column count, solve, restore.
    orig_shape = J_b.shape
    if J_b.ndim > a.ndim:
        # batch_ndim = number of leading batch dims shared by a and J_b
        batch_ndim = max(a.ndim - 2, J_b.ndim - 3)
        J_b = J_b.reshape(J_b.shape[:batch_ndim + 1] + (-1,))

    # Expand a's batch dims if J_b has more.
    a_batch_ndim = a.ndim - 2
    j_batch_ndim = J_b.ndim - 2
    if j_batch_ndim > a_batch_ndim:
        extra = j_batch_ndim - a_batch_ndim
        a = jnp.broadcast_to(
            a.reshape((1,) * extra + a.shape),
            J_b.shape[:j_batch_ndim] + a.shape[-2:],
        )

    return lax.linalg.triangular_solve(a, J_b, **params).reshape(orig_shape)


try:
    from jax._src.lax.linalg import triangular_solve_p
    materialize_rules[triangular_solve_p] = _triangular_solve_rule
except ImportError:
    pass
