"""Non-densifying singledispatch registrations for jax.experimental.sparse.BCOO.

These extend the negate/scale_scalar/scale_per_out_row interface to BCOO
without subclassing or monkeypatching the JAX class.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.experimental import sparse

from .base import negate, scale_per_out_row, scale_scalar


@negate.register(sparse.BCOO)
def _(op: sparse.BCOO) -> sparse.BCOO:
    return sparse.BCOO((-op.data, op.indices), shape=op.shape)


@scale_scalar.register(sparse.BCOO)
def _(op: sparse.BCOO, s) -> sparse.BCOO:
    return sparse.BCOO((s * op.data, op.indices), shape=op.shape)


@scale_per_out_row.register(sparse.BCOO)
def _(op: sparse.BCOO, v) -> sparse.BCOO:
    # op.indices shape: (nse, 2) for unbatched; (*batch, nse, 2) for batched.
    row_idx = op.indices[..., 0]
    v_arr = jnp.asarray(v)
    return sparse.BCOO(
        (op.data * jnp.take(v_arr, row_idx), op.indices), shape=op.shape
    )
