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
    return lax.squeeze_p.bind(op, **params)


@rev_op.register(jax.Array)
@rev_op.register(DynamicJaxprTracer)
def _(op, *, n, **params):
    return lax.rev_p.bind(op, **params)


@slice_op.register(jax.Array)
@slice_op.register(DynamicJaxprTracer)
def _(op, *, n, **params):
    return lax.slice_p.bind(op, **params)


@pad_op.register(jax.Array)
@pad_op.register(DynamicJaxprTracer)
def _(op, *, n, padding_value, **params):
    return lax.pad_p.bind(op, padding_value, **params)


@reshape_op.register(jax.Array)
@reshape_op.register(DynamicJaxprTracer)
def _(op, *, n, **params):
    return lax.reshape_p.bind(op, **params)


def _bid_with_extra_batch(dense, shape, broadcast_dimensions, n):
    """Shared tail for broadcast_in_dim LinOp dense fallbacks.

    Handles extra leading batch dims that vmap may have accumulated before
    the strip rules see the tensor (n_extra == 0 in the normal non-vmap
    path). Imported by `ellpack_transforms.py` and `diagonal.py` for
    their LinOp dense fallbacks.
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
    return lax.broadcast_in_dim_p.bind(op, **params)


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
    return lax.split_p.bind(op, **params)
