"""Primitive rule registry — maps jax.lax primitives to rule functions.

PoC strategy: axis-stripping wrappers strip the vmap batch entry from params
then delegate unchanged to the existing LinOp singledispatch functions, which
retain all BEllpack structural paths without densification.

Migration path (after PoC confirmed): remove wrappers one op at a time,
update each dispatch function to natively handle the correct number of batch
dims, and loop until no wrappers remain.  The end state: every op below that
currently strips just calls its dispatch function directly with no wrapper.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import lax
from jax._src.lax import slicing as _slicing
from jax.experimental import sparse
from jax.extend import core

from .add import _add_rule, BELLPACK_DEDUP_LIMIT, BELLPACK_DEDUP_VECTORISED_MIN
from .mul import _mul_rule
from .multilinear import _sub_rule, _div_rule, _dot_general_rule
from .control_flow import (
    _cond_rule,
    _jit_rule,
    _select_n_rule,
)
from .structural import _concatenate_rule
from .._linops import (
    BEllpack,
    LinOpProtocol,
    broadcast_in_dim_op,
    gather_op,
    pad_op,
    reduce_sum_op,
    reshape_op,
    rev_op,
    scatter_add_op,
    slice_op,
    split_op,
    squeeze_op,
    replace_slots,
)
from jax.experimental.sparse.transform import (
    _zero_preserving_linear_unary_primitives,
)

__all__ = [
    "materialize_rules",
    "BELLPACK_DEDUP_LIMIT",
    "BELLPACK_DEDUP_VECTORISED_MIN",
]


# ---------------------------------------------------------------------------
# Axis-strip primitives and generic rule factory
# ---------------------------------------------------------------------------

def _make_unary_rule(dispatch_fn, *, zero_preserving=False):
    """Build a single-traced-operand rule.

    Two flavours, both keyed off `dispatch_fn`:

    - `zero_preserving=True`: `dispatch_fn` is a `prim.bind`-style
      callable for an elementwise primitive that maps zero to zero
      (`neg`, `conj`, `real`, `imag`, `convert_element_type`, ...).
      The rule applies it to `op.data` for LinOps and rebuilds via
      `replace_slots`; for non-LinOp inputs (BCOO, plain arrays) it
      applies directly. Mirrors
      `jax.experimental.sparse.transform._zero_preserving_unary_op`.
    - `zero_preserving=False` (default): `dispatch_fn` is a
      singledispatch op (`slice_op`, `squeeze_op`, ...) with per-LinOp
      implementations. Called with the LinOp, `n=n`, and the jaxpr
      params straight through (Phase B convention — no walk-frame
      translation).
    """
    def rule(invals, traced, n, **params):
        (op,), (t,) = invals, traced
        if not t:
            return None
        if zero_preserving:
            if isinstance(op, LinOpProtocol):
                return replace_slots(op, data=dispatch_fn(op.data, **params))
            return dispatch_fn(op, **params)
        return dispatch_fn(op, n=n, **params)
    return rule


# ---------------------------------------------------------------------------
# Two-input rules that need custom signatures
# ---------------------------------------------------------------------------

def _pad_rule(invals, traced, n, **params):
    """Phase B: jaxpr params pass straight through (vmap(-1, -1) puts
    batch at the last axis, matching walker's in_axis at -1)."""
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
# Gather — vmap shifts the gather dimension numbers
# ---------------------------------------------------------------------------

def _gather_rule(invals, traced, n, **params):
    """Phase B: jaxpr params pass straight through. Translation removed —
    legacy walk-frame logic was specific to vmap(in_axes=0)."""
    operand, start_indices = invals
    to, ti = traced
    if ti:
        raise NotImplementedError("gather with traced indices")
    if not to:
        return None
    if isinstance(operand, sparse.BCOO):
        raise NotImplementedError("gather on BCOO operand")
    return gather_op(operand, n=n, start_indices=start_indices, **params)


# ---------------------------------------------------------------------------
# Scatter-add — vmap shifts the scatter dimension numbers
# ---------------------------------------------------------------------------

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

    # vmapped form of scatter_collapsed — strip batch dim from dnums and operand.
    if (dnums.update_window_dims == (0,)
            and tuple(dnums.inserted_window_dims) == (1,)
            and tuple(dnums.scatter_dims_to_operand_dims) == (1,)):
        dnums = _slicing.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        params = dict(params, dimension_numbers=dnums)
        if hasattr(operand, "shape") and len(operand.shape) > 1 and operand.shape[0] == n:
            operand = operand[0]

    # vmapped 2D point-scatter: vmap adds b as update_window_dim 0 and shifts
    # inserted/scatter dims up by 1.  Convert to base form and fall through.
    if (dnums.update_window_dims == (0,)
            and tuple(dnums.inserted_window_dims) == (1, 2)
            and tuple(dnums.scatter_dims_to_operand_dims) == (1, 2)
            and hasattr(operand, "ndim") and operand.ndim == 2):
        dnums = _slicing.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1),
        )
        params = dict(params, dimension_numbers=dnums)

    # 2D point-scatter (HADAMALS-class): inserts updates[k] at
    # operand[scatter_indices[k,0], scatter_indices[k,1]]. Output has V
    # at axis 0 (match the rest of the post-vmap V-at-0 chain).
    if (dnums.update_window_dims == ()
            and dnums.inserted_window_dims == (0, 1)
            and dnums.scatter_dims_to_operand_dims == (0, 1)
            and operand.ndim == 2):
        out_shape_2d = operand.shape
        updates_dense = updates.todense() if isinstance(updates, LinOpProtocol) else jnp.asarray(updates)
        si_flat = scatter_indices.reshape(-1, 2)
        # Updates have V at axis 0 → flatten trailing dims: (V, K).
        updates_flat = updates_dense.reshape(n, -1)
        flat_cols = (si_flat[:, 0].astype(jnp.int64) * out_shape_2d[1]
                     + si_flat[:, 1].astype(jnp.int64))
        return (jnp.zeros((n, out_shape_2d[0] * out_shape_2d[1]),
                          updates_flat.dtype)
                .at[:, flat_cols].add(updates_flat)
                .reshape((n,) + out_shape_2d))

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

# Bulk-register the zero-preserving linear unary family (neg, copy, conj,
# real, imag — whatever the upstream sparsify list contains). Each rule
# applies the primitive to `.data` and rebuilds the LinOp via `replace_slots`.
# Mirrors the upstream pattern at jax.experimental.sparse.transform:539–541.
for _prim in _zero_preserving_linear_unary_primitives:
    materialize_rules[_prim] = _make_unary_rule(
        _prim.bind, zero_preserving=True,
    )
# convert_element_type_p has a `new_dtype` param and isn't in upstream's
# list, but the same body works (data dtype follows the primitive output).
materialize_rules[lax.convert_element_type_p] = _make_unary_rule(
    lax.convert_element_type_p.bind, zero_preserving=True,
)
materialize_rules[lax.sub_p] = _sub_rule
materialize_rules[lax.dot_general_p] = _dot_general_rule
materialize_rules[lax.slice_p] = _make_unary_rule(slice_op)
materialize_rules[lax.pad_p] = _pad_rule
materialize_rules[lax.squeeze_p] = _make_unary_rule(squeeze_op)
materialize_rules[lax.rev_p] = _make_unary_rule(rev_op)
materialize_rules[lax.reshape_p] = _make_unary_rule(reshape_op)
materialize_rules[lax.broadcast_in_dim_p] = _make_unary_rule(broadcast_in_dim_op)
materialize_rules[lax.reduce_sum_p] = _make_unary_rule(reduce_sum_op)
materialize_rules[lax.concatenate_p] = _concatenate_rule
materialize_rules[lax.split_p] = _make_unary_rule(split_op)

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


def _cumsum_rule(invals, traced, n, **params):
    """cumsum: densify any LinOp and call lax.cumsum.

    No structural sparsity-preserving path exists; every prior format
    registration was identical to this dense fallback, so they were
    deleted along with the cumsum_op singledispatch.
    """
    del n
    (op,), (t,) = invals, traced
    if not t:
        return None
    if isinstance(op, LinOpProtocol):
        op = op.todense()
    return lax.cumsum(op, axis=int(params["axis"]),
                      reverse=params.get("reverse", False))


materialize_rules[lax.cumsum_p] = _cumsum_rule
materialize_rules[lax.div_p] = _div_rule

def _transpose_rule(invals, traced, n, **params):
    """Phase B: dispatch to `op.transpose(perm)`. Each LinOp / BCOO /
    plain array implements its own transpose — symmetric forms return
    self, BE 2D cross-V swap flips the flag, identity perms early-out
    inside the method, and unsupported cross-V perms raise."""
    del n
    (op,), (t,) = invals, traced
    if not t:
        return None
    perm = params["permutation"]
    return op.transpose(perm)
materialize_rules[lax.transpose_p] = _transpose_rule

materialize_rules[lax.gather_p] = _gather_rule
materialize_rules[_slicing.scatter_add_p] = _scatter_add_rule
