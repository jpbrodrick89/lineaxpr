"""Singledispatch registrations for plain jax arrays.

Registered for both `jax.Array` and `DynamicJaxprTracer`: singledispatch
dispatches via MRO, and JIT-time tracers are not in `jax.Array`'s MRO
(even though `isinstance(tracer, jax.Array)` is True), so the second
registration is required to cover JIT.

The `broadcast_in_dim_op` registration also includes a structural-
recovery special case: a `(n,)`-shaped dense linear-form broadcast to
`(1,)` is converted back to BCOO rather than staying dense.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.interpreters.partial_eval import DynamicJaxprTracer
from jax.experimental import sparse

from .base import (
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
)


@squeeze_op.register(jax.Array)
@squeeze_op.register(DynamicJaxprTracer)
def _(op, *, n, **params):
    return lax.squeeze(op, params["dimensions"])


@rev_op.register(jax.Array)
@rev_op.register(DynamicJaxprTracer)
def _(op, *, n, **params):
    return lax.rev(op, params["dimensions"])


@slice_op.register(jax.Array)
@slice_op.register(DynamicJaxprTracer)
def _(op, *, n, **params):
    return lax.slice(op, **params)


@pad_op.register(jax.Array)
@pad_op.register(DynamicJaxprTracer)
def _(op, *, n, padding_value, **params):
    return lax.pad(op, padding_value, **params)


@reshape_op.register(jax.Array)
@reshape_op.register(DynamicJaxprTracer)
def _(op, *, n, **params):
    # `sharding` (jaxpr) → `out_sharding` (lax); pass explicitly to avoid
    # name mismatch when forwarding **params.
    return lax.reshape(op, params["new_sizes"],
                       dimensions=params.get("dimensions"),
                       out_sharding=params.get("sharding"))


def _bid_with_extra_batch(dense, shape, broadcast_dimensions, n):
    """Shared tail for broadcast_in_dim dense fallbacks.

    Handles extra leading batch dims that vmap may have accumulated before
    the strip rules see the tensor (n_extra == 0 in the normal non-vmap
    path). Also imported by ellpack_transforms.py for the BEllpack dense
    fallback.
    """
    expected_ndim = len(broadcast_dimensions) + 1
    while dense.ndim > expected_ndim and dense.shape[0] == 1:
        dense = dense[0]
    n_extra = dense.ndim - expected_ndim
    if n_extra > 0:
        adj_shape = tuple(dense.shape[:n_extra]) + tuple(shape)
        adj_bds = tuple(range(n_extra)) + tuple(b + n_extra for b in broadcast_dimensions)
    else:
        adj_shape, adj_bds = tuple(shape), tuple(broadcast_dimensions)
    out_dims = adj_bds + (len(adj_shape),)
    return lax.broadcast_in_dim(dense, adj_shape + (n,), out_dims)


@broadcast_in_dim_op.register(jax.Array)
@broadcast_in_dim_op.register(DynamicJaxprTracer)
def _(op, *, n, **params):
    full_shape = tuple(params["shape"])
    full_bd = tuple(params["broadcast_dimensions"])
    # Inside-vmap: V at the front of output (bd[0]==0, shape[0]==n).
    # Operand is already in V-at-0 layout (e.g., densified upstream from a
    # transposed BE). Pass params straight through to lax.broadcast_in_dim.
    if full_bd and full_bd[0] == 0 and full_shape[0] == n:
        return lax.broadcast_in_dim(op, full_shape, full_bd)
    # Walk-frame: shape ends in n, bd ends in n's mapping (== ndim_out-1).
    # Strip both trailing entries for the spatial-only structural checks.
    shape = full_shape[:-1]
    broadcast_dimensions = full_bd[:-1]
    # Dense linear-form (n,)-ndarray broadcast to spatial-shape (1,):
    # recover BCOO form so downstream rules see structure, not a dense row.
    if broadcast_dimensions == () and shape == (1,):
        if op.ndim == 1 and op.shape[0] == n:
            zeros_row = jnp.zeros((n,), dtype=jnp.int32)
            cols = jnp.arange(n, dtype=jnp.int32)
            indices = jnp.stack([zeros_row, cols], axis=1)
            return sparse.BCOO((op, indices), shape=(1, n))
    return _bid_with_extra_batch(op, shape, broadcast_dimensions, n)


@reduce_sum_op.register(jax.Array)
@reduce_sum_op.register(DynamicJaxprTracer)
def _(op, *, n, **params):
    return lax.reduce_sum_p.bind(op, **params)


@gather_op.register(jax.Array)
@gather_op.register(DynamicJaxprTracer)
def _(op, *, n, start_indices, **params):
    return lax.gather(op, start_indices, **params)


@scatter_add_op.register(jax.Array)
@scatter_add_op.register(DynamicJaxprTracer)
def _(updates, *, n, operand, scatter_indices, **params):
    # V-at-0 layout: updates have shape (V=n, *update_batch); the
    # result has shape (V=n, out_size). `_scatter_add_rule` does the
    # dnums normalisation upstream so the canvas+at[].add() pattern
    # matches the rewritten params.
    out_idx = scatter_indices[..., 0]
    out_idx_flat = out_idx.reshape(-1)
    out_size = operand.shape[0]
    # Updates come in with V at axis 0; flatten the trailing
    # (update_batch) dims to a single axis matching `out_idx_flat`.
    flat_updates = updates.reshape(n, -1)
    # Scatter along the out_size axis (axis 1 of the canvas).
    return (jnp.zeros((n, out_size), flat_updates.dtype)
            .at[:, out_idx_flat].add(flat_updates))


@split_op.register(jax.Array)
@split_op.register(DynamicJaxprTracer)  # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    return list(lax.split(op, params["sizes"], axis=params["axis"]))
