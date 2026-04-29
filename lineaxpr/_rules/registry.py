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

def _drop(v):
    """Drop the leading element (the vmap batch entry); no-op for None."""
    return v[1:] if v is not None else v

def _decr(v):
    """Subtract 1 from every element; works on int or tuple."""
    if hasattr(v, "__iter__"):
        return tuple(int(x) - 1 for x in v)
    return int(v) - 1

def _drop_decr(v):
    """Drop the leading element then decrement the rest."""
    return tuple(int(x) - 1 for x in v[1:])


def _vmap_strip(**ops):
    """Return a (params, n) → params function that applies one strip op per key."""
    def strip(params, n):
        return {**params, **{k: f(params[k]) for k, f in ops.items() if k in params}}
    return strip


def _vmap_unary_rule(dispatch_fn, strip=None):
    """Generic unary axis-stripping rule: strip params, call dispatch op."""
    def rule(invals, traced, n, **params):
        (op,), (t,) = invals, traced
        if not t:
            return None
        if strip:
            params = strip(params, n)
        return dispatch_fn(op, n=n, **params)
    return rule


# ---------------------------------------------------------------------------
# Unchanged rules (no axis params)
# ---------------------------------------------------------------------------

def _make_zero_preserving_linear_unary_rule(prim):
    """Mirror of `jax.experimental.sparse.transform._zero_preserving_unary_op`.

    Pushes a primitive through any LinOp by applying it to `op.data`
    and rebuilding the same form via `replace_slots` (which handles
    both our `__slots__` LinOps and BCOO's `__dict__`-backed storage
    uniformly).
    """
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


# ---------------------------------------------------------------------------
# Two-input rules that need custom signatures
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
    # Translate jaxpr-frame padding_config (vmap put n at position 0) to
    # walk-frame (n at -1). The leading entry is always (0,0,0) because
    # vmap doesn't pad its inserted batch axis; drop it from the front
    # and append at the back to preserve the n identity-pad.
    cfg = tuple(params["padding_config"])
    walk_cfg = cfg[1:] + ((0, 0, 0),)
    return pad_op(operand, n=n, padding_value=padding_value,
                  padding_config=walk_cfg)


def _concatenate_rule_vmap(invals, traced, n, **params):
    # dimension is shifted +1 by vmap; subtract 1.
    return _concatenate_rule(invals, traced, n,
                              **_vmap_strip(dimension=_decr)(params, n))



# ---------------------------------------------------------------------------
# Gather — vmap shifts the gather dimension numbers
# ---------------------------------------------------------------------------

def _gather_rule(invals, traced, n, **params):
    operand, start_indices = invals
    to, ti = traced
    if ti:
        raise NotImplementedError("gather with traced indices")
    if not to:
        return None
    if isinstance(operand, sparse.BCOO):
        raise NotImplementedError("gather on BCOO operand")
    dn = params["dimension_numbers"]
    # vmap wraps N-D point-gather-collapsed as:
    #   offset_dims=(0,), collapsed=(k1,...), start_index_map=(k1,...)  where all ki > 0
    # Convert by stripping the batch dim: decrement each ki by 1, drop batch slice_size.
    if (len(dn.offset_dims) == 1 and dn.offset_dims[0] == 0
            and 1 <= len(dn.collapsed_slice_dims) <= 2
            and all(int(k) > 0 for k in dn.collapsed_slice_dims)
            and dn.start_index_map == dn.collapsed_slice_dims):
        new_collapsed = tuple(int(k) - 1 for k in dn.collapsed_slice_dims)
        base_dn = _slicing.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=new_collapsed,
            start_index_map=new_collapsed,
            operand_batching_dims=(), start_indices_batching_dims=(),
        )
        params = dict(params, dimension_numbers=base_dn,
                      slice_sizes=tuple(params["slice_sizes"])[1:])
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
    materialize_rules[_prim] = _make_zero_preserving_linear_unary_rule(_prim)
# convert_element_type_p has a `new_dtype` param and isn't in upstream's
# list, but the same body works (data dtype follows the primitive output).
materialize_rules[lax.convert_element_type_p] = (
    _make_zero_preserving_linear_unary_rule(lax.convert_element_type_p)
)
materialize_rules[lax.sub_p] = _sub_rule
materialize_rules[lax.dot_general_p] = _dot_general_rule
materialize_rules[lax.slice_p] = _vmap_unary_rule(slice_op,
    strip=_vmap_strip(start_indices=_drop, limit_indices=_drop, strides=_drop))
materialize_rules[lax.pad_p] = _pad_rule
materialize_rules[lax.squeeze_p] = _vmap_unary_rule(squeeze_op,
    strip=_vmap_strip(dimensions=_decr))
materialize_rules[lax.rev_p] = _vmap_unary_rule(rev_op,
    strip=_vmap_strip(dimensions=_decr))
def _reshape_rule(invals, traced, n, **params):
    """Translate jaxpr-frame `new_sizes` (n at 0) to walk-frame (n at -1)."""
    (op,), (t,) = invals, traced
    if not t:
        return None
    params.pop("_vmap_avals", None)
    sizes = tuple(int(s) for s in params["new_sizes"])
    walk_sizes = sizes[1:] + (n,)
    return reshape_op(op, n=n, **{**params, "new_sizes": walk_sizes})


materialize_rules[lax.reshape_p] = _reshape_rule
def _broadcast_in_dim_rule(invals, traced, n, **params):
    """Translate jaxpr broadcast_in_dim params (n at jaxpr-output 0) to
    walk-frame (n at -1)."""
    (op,), (t,) = invals, traced
    if not t:
        return None
    params.pop("_vmap_avals", None)
    jaxpr_shape = tuple(params["shape"])
    jaxpr_bd = tuple(int(d) for d in params["broadcast_dimensions"])
    walk_shape = jaxpr_shape[1:] + (n,)
    # jaxpr output dim d → walk dim: d==0 → ndim-1 (n at end); d>0 → d-1.
    # vmap puts operand's n at jaxpr operand axis 0, mapped via bd[0] to
    # jaxpr output axis 0 (always). Reorder so walk operand axes are
    # (jaxpr operand 1+ then jaxpr operand 0).
    walk_bd = tuple(b - 1 for b in jaxpr_bd[1:]) + (len(walk_shape) - 1,)
    return broadcast_in_dim_op(
        op, n=n,
        **{**params, "shape": walk_shape, "broadcast_dimensions": walk_bd},
    )


materialize_rules[lax.broadcast_in_dim_p] = _broadcast_in_dim_rule
materialize_rules[lax.reduce_sum_p] = _vmap_unary_rule(reduce_sum_op,
    strip=_vmap_strip(axes=_decr))
materialize_rules[lax.concatenate_p] = _concatenate_rule_vmap
materialize_rules[lax.split_p] = _vmap_unary_rule(split_op,
    strip=_vmap_strip(axis=_decr))

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
    return lax.cumsum(op, axis=int(params["axis"]) - 1,
                      reverse=params.get("reverse", False))


materialize_rules[lax.cumsum_p] = _cumsum_rule
materialize_rules[lax.div_p] = _div_rule

def _transpose_rule(invals, traced, n, **params):
    """Translate a jaxpr permutation to walk-frame, then dispatch to `.transpose`.

    Walk frame fixes n at dim -1; jaxpr can have it at `bdim` (usually 0
    after vmap, but `dot_general` can shift it). We rewrite perm into
    walk axes so n's output dim disappears (it's appended at -1), then
    every form (LinOp, BCOO, jax.Array) handles `.transpose(walk_perm)`
    uniformly — BEllpack auto-strips the trailing in-axis identity entry.
    """
    (op,), (t,) = invals, traced
    if not t:
        return None
    vmap_avals = params.pop("_vmap_avals", None)
    perm = tuple(int(p) for p in params["permutation"])
    ndim = len(perm)

    traced_aval = vmap_avals[0] if vmap_avals else None
    bdim = 0
    if traced_aval is not None:
        bdim = next((i for i, s in enumerate(traced_aval) if int(s) == n), 0)

    # jaxpr dim → walk dim: bdim → ndim-1, others shift down past bdim.
    def to_walk(d):
        return ndim - 1 if d == bdim else d - (d > bdim)

    n_out = perm.index(bdim)
    walk_perm = tuple(to_walk(perm[j]) for j in range(ndim) if j != n_out) + (ndim - 1,)

    if walk_perm == tuple(range(ndim)):
        return op
    # BCOO's native transpose disallows permutations that mix batch/sparse/
    # dense axes (see jax.experimental.sparse.bcoo._validate_permutation).
    # Use it when allowed, densify only when the perm crosses a boundary.
    # TODO: once the internal Csr LinOp lands (docs/TODO.md §2), route this
    # densify branch through BCOO → Csr → relabeled transpose to preserve
    # sparsity for cross-boundary perms.
    if isinstance(op, sparse.BCOO):
        n_batch = op.indices.ndim - 2
        n_sparse = op.indices.shape[-1]
        batch_ok = (not n_batch
                    or tuple(sorted(walk_perm[:n_batch])) == tuple(range(n_batch)))
        dense_ok = (len(walk_perm) == n_batch + n_sparse
                    or tuple(sorted(walk_perm[n_batch + n_sparse:]))
                       == tuple(range(n_batch + n_sparse, len(walk_perm))))
        if not (batch_ok and dense_ok):
            op = op.todense()
    return op.transpose(walk_perm)
materialize_rules[lax.transpose_p] = _transpose_rule

materialize_rules[lax.gather_p] = _gather_rule
materialize_rules[_slicing.scatter_add_p] = _scatter_add_rule
