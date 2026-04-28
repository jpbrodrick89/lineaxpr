"""Non-densifying singledispatch registrations for jax.experimental.sparse.BCOO.

These extend the negate/scale_scalar/scale_per_out_row interface to BCOO
without subclassing or monkeypatching the JAX class.
Also registers BCOO-specific implementations for unary structural ops.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from .base import (
    negate,
    pad_op,
    reduce_sum_op,
    rev_op,
    scale_per_out_row,
    scale_scalar,
    slice_op,
)


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


@slice_op.register(sparse.BCOO)  # pyrefly: ignore [bad-argument-type]
def _(op: sparse.BCOO, *, n, **params):
    starts = tuple(int(s) for s in params["start_indices"])
    limits = tuple(int(l) for l in params["limit_indices"])
    strides_p = params.get("strides")
    strides = tuple(int(s) for s in strides_p) if strides_p else (1,) * len(starts)

    if len(starts) == 1 and strides == (1,) and op.n_batch == 0:
        s, e = starts[0], limits[0]
        k = e - s
        indices_np = None
        try:
            indices_np = np.asarray(op.indices)
        except (jax.errors.TracerArrayConversionError, TypeError):
            pass
        if isinstance(indices_np, np.ndarray):
            rows_np = indices_np[:, 0]
            keep = np.nonzero((rows_np >= s) & (rows_np < e))[0]
            new_indices = np.stack(
                [rows_np[keep] - s, indices_np[keep, 1]], axis=1
            )
            new_data = jnp.take(op.data, jnp.asarray(keep))
            return sparse.BCOO(
                # pyrefly: ignore [bad-argument-type]
                (new_data, new_indices), shape=(k, op.shape[1]),
            )
        rows = op.indices[:, 0]
        in_range = (rows >= s) & (rows < e)
        new_rows = jnp.where(in_range, rows - s, 0)
        new_data = jnp.where(in_range, op.data,
                             jnp.zeros((), op.data.dtype))
        new_indices = jnp.stack([new_rows, op.indices[:, 1]], axis=1)
        return sparse.BCOO((new_data, new_indices), shape=(k, op.shape[1]))

    # Dense fallback for other BCOO slice patterns.
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    from jax import lax  # noqa: PLC0415
    dense = _to_dense(op, n)
    s_full = starts + (0,)
    l_full = limits + (n,)
    str_full = strides + (1,)
    return lax.slice(dense, s_full, l_full, str_full)


@pad_op.register(sparse.BCOO)  # pyrefly: ignore [bad-argument-type]
def _(op: sparse.BCOO, *, n, padding_value, **params):
    config = params["padding_config"]
    before, after, interior = config[0] if len(config) >= 1 else (0, 0, 0)
    before, after = int(before), int(after)
    interior = int(interior)

    if len(config) == 1 and interior == 0:
        out_size = op.shape[0] + before + after
        new_rows = op.indices[:, 0] + before
        new_indices = jnp.stack([new_rows, op.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (op.data, new_indices), shape=(out_size, op.shape[1])
        )
    if len(config) == 1 and interior > 0:
        step = interior + 1
        old_size = op.shape[0]
        out_size = old_size + before + after + interior * max(old_size - 1, 0)
        new_rows = op.indices[:, 0] * step + before
        new_indices = jnp.stack([new_rows, op.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (op.data, new_indices), shape=(out_size, op.shape[1])
        )
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    from jax import lax  # noqa: PLC0415
    dense = _to_dense(op, n)
    full_config = tuple((int(b), int(a), int(i)) for (b, a, i) in config) + ((0, 0, 0),)
    return lax.pad(dense, jnp.asarray(0.0, dtype=dense.dtype), full_config)


@rev_op.register(sparse.BCOO)  # pyrefly: ignore [bad-argument-type]
def _(op: sparse.BCOO, *, n, **params):
    dimensions = params["dimensions"]
    if op.n_batch == 0 and dimensions == (0,):
        new_rows = (op.shape[0] - 1) - op.indices[:, 0]
        new_indices = jnp.stack([new_rows, op.indices[:, 1]], axis=1)
        return sparse.BCOO((op.data, new_indices), shape=op.shape)
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    from jax import lax  # noqa: PLC0415
    dense = _to_dense(op, n)
    return lax.rev(dense, dimensions)


@reduce_sum_op.register(sparse.BCOO)  # pyrefly: ignore [bad-argument-type]
def _(op: sparse.BCOO, *, n, **params):
    """BCOO row-sum: when indices are static np, emit a structural BE row-vector."""
    axes = params["axes"]
    from lineaxpr._linops.ellpack import BEllpack  # noqa: PLC0415
    if tuple(axes) == (0,) and op.n_batch == 0:
        try:
            indices_np = np.asarray(op.indices)
        except (jax.errors.TracerArrayConversionError, TypeError):
            indices_np = None
        if isinstance(indices_np, np.ndarray):
            in_size = int(op.shape[1])
            cols_np = indices_np[:, 1]
            uniq, inverse = np.unique(cols_np, return_inverse=True)
            n_groups = int(uniq.shape[0])
            if 0 < n_groups < in_size:
                summed = jnp.zeros((n_groups,), op.data.dtype).at[
                    jnp.asarray(inverse)].add(op.data)
                if n_groups == 1:
                    return BEllpack(
                        start_row=0, end_row=1,
                        in_cols=(np.asarray([uniq[0]], dtype=uniq.dtype),),
                        values=summed.reshape(1),
                        out_size=1, in_size=in_size,
                    )
                return BEllpack(
                    start_row=0, end_row=1,
                    in_cols=tuple(np.asarray([c], dtype=uniq.dtype)
                                  for c in uniq),
                    values=summed.reshape(1, n_groups),
                    out_size=1, in_size=in_size,
                )
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    dense = _to_dense(op, n)
    return jnp.sum(dense, axis=tuple(axes))
