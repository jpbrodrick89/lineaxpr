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
    ConstantDiagonal,
    Diagonal,
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

def _make_unary_rule(dispatch_fn=None, *, prim=None, zero_preserving=False):
    """Build a single-traced-operand rule.

    Two flavours:

    - `zero_preserving=True` (with `prim` given): the rule applies the
      primitive directly to `op.data` and rebuilds the same LinOp form
      via `replace_slots`. Suits elementwise primitives that map zero
      to zero (`neg`, `conj`, `real`, `imag`, `convert_element_type`,
      ...). Mirrors `jax.experimental.sparse.transform._zero_preserving_unary_op`.

    - `zero_preserving=False` (default, with `dispatch_fn` given):
      delegates to a singledispatch op (`slice_op`, `squeeze_op`, ...)
      that has per-LinOp implementations. Jaxpr params pass straight
      through (Phase B convention — no walk-frame translation).
    """
    if zero_preserving:
        assert prim is not None, "zero_preserving=True requires `prim`"
        def rule(invals, traced, n, **params):
            del n
            params.pop("_vmap_avals", None)
            (op,), (t,) = invals, traced
            if not t:
                return None
            if isinstance(op, LinOpProtocol):
                return replace_slots(op, data=prim.bind(op.data, **params))
            return prim.bind(op, **params)
        return rule
    assert dispatch_fn is not None, "zero_preserving=False requires `dispatch_fn`"
    def rule(invals, traced, n, **params):
        (op,), (t,) = invals, traced
        if not t:
            return None
        params.pop("_vmap_avals", None)
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
    params.pop("_vmap_avals", None)
    return pad_op(operand, n=n, padding_value=padding_value, **params)


def _concatenate_rule_vmap(invals, traced, n, **params):
    """Phase B: jaxpr params pass straight through."""
    return _concatenate_rule(invals, traced, n, **params)



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
    materialize_rules[_prim] = _make_unary_rule(prim=_prim, zero_preserving=True)
# convert_element_type_p has a `new_dtype` param and isn't in upstream's
# list, but the same body works (data dtype follows the primitive output).
materialize_rules[lax.convert_element_type_p] = _make_unary_rule(
    prim=lax.convert_element_type_p, zero_preserving=True,
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
materialize_rules[lax.concatenate_p] = _concatenate_rule_vmap
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
    params.pop("_vmap_avals", None)
    if isinstance(op, LinOpProtocol):
        op = op.todense()
    return lax.cumsum(op, axis=int(params["axis"]),
                      reverse=params.get("reverse", False))


materialize_rules[lax.cumsum_p] = _cumsum_rule
materialize_rules[lax.div_p] = _div_rule

def _transpose_rule(invals, traced, n, **params):
    """Phase B: jaxpr params pass straight through.

    For 2D BEllpack with row/col swap (perm=(1, 0)), use the free
    `transposed` flag flip. For BCOO, use native transpose (cheap
    index swap). For other forms, dispatch to op.transpose.
    """
    (op,), (t,) = invals, traced
    if not t:
        return None
    params.pop("_vmap_avals", None)
    perm = tuple(int(p) for p in params["permutation"])
    if perm == tuple(range(len(perm))):
        return op
    # 2D row/col swap on BEllpack: free flag flip.
    if isinstance(op, BEllpack) and op.n_batch == 0 and perm == (1, 0):
        return replace_slots(op, transposed=not op.transposed)
    # 2D BCOO: native transpose (cheap index swap).
    if isinstance(op, sparse.BCOO) and op.indices.ndim == 2 and op.indices.shape[-1] == 2:
        return op.transpose(axes=perm)
    # CD / Diagonal: symmetric, perm is no-op for square.
    if isinstance(op, (ConstantDiagonal, Diagonal)):
        return op
    return op.transpose(perm)
materialize_rules[lax.transpose_p] = _transpose_rule

materialize_rules[lax.gather_p] = _gather_rule
materialize_rules[_slicing.scatter_add_p] = _scatter_add_rule
