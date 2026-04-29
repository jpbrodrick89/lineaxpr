"""Non-densifying singledispatch registrations for jax.experimental.sparse.BCOO.

These extend the scale_scalar/scale_per_out_row interface to BCOO
without subclassing or monkeypatching the JAX class.
Also registers BCOO-specific implementations for unary structural ops.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import sparse

from .ellpack import BEllpack
from .base import (
    broadcast_in_dim_op,
    pad_op,
    reduce_sum_op,
    rev_op,
    scale_per_out_row,
    scale_scalar,
    slice_op,
    split_op,
    squeeze_op,
)
from .dense import _bid_with_extra_batch


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


@slice_op.register(sparse.BCOO)
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
    dense = op.todense()
    s_full = starts + (0,)
    l_full = limits + (n,)
    str_full = strides + (1,)
    return lax.slice(dense, s_full, l_full, str_full)


@pad_op.register(sparse.BCOO)
def _(op: sparse.BCOO, *, n, padding_value, **params):
    # Walk-frame config has the n identity-pad at -1 (always (0,0,0));
    # structural logic operates on the spatial entries only.
    config = params["padding_config"][:-1]
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
    return lax.pad(op.todense(), padding_value, **params)


@rev_op.register(sparse.BCOO)
def _(op: sparse.BCOO, *, n, **params):
    dimensions = params["dimensions"]
    if op.n_batch == 0 and dimensions == (0,):
        new_rows = (op.shape[0] - 1) - op.indices[:, 0]
        new_indices = jnp.stack([new_rows, op.indices[:, 1]], axis=1)
        return sparse.BCOO((op.data, new_indices), shape=op.shape)
    dense = op.todense()
    return lax.rev(dense, dimensions)


@reduce_sum_op.register(sparse.BCOO)  # pyrefly: ignore [bad-argument-type]
def _(op: sparse.BCOO, *, n, **params):
    """BCOO row-sum: when indices are static np, emit a structural BE row-vector."""
    axes = params["axes"]
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
                        data=summed.reshape(1),
                        out_size=1, in_size=in_size,
                    )
                return BEllpack(
                    start_row=0, end_row=1,
                    in_cols=tuple(np.asarray([c], dtype=uniq.dtype)
                                  for c in uniq),
                    data=summed.reshape(1, n_groups),
                    out_size=1, in_size=in_size,
                )
    dense = op.todense()
    return jnp.sum(dense, axis=tuple(axes))


@split_op.register(sparse.BCOO)  # pyrefly: ignore [bad-argument-type]
def _(op: sparse.BCOO, *, n, **params):
    sizes = params["sizes"]
    axis = params["axis"]
    if axis == 0 and op.n_batch == 0:
        rows = op.indices[:, 0]
        out = []
        start = 0
        for sz in sizes:
            end = start + int(sz)
            in_range = (rows >= start) & (rows < end)
            new_rows = jnp.where(in_range, rows - start, 0)
            new_data = jnp.where(in_range, op.data,
                                 jnp.zeros((), op.data.dtype))
            new_indices = jnp.stack(
                [new_rows, op.indices[:, 1]], axis=1
            )
            out.append(sparse.BCOO(
                (new_data, new_indices), shape=(int(sz), op.shape[1])
            ))
            start = end
        return out
    dense = op.todense()
    out = []
    start = 0
    for sz in sizes:
        slc = [slice(None)] * dense.ndim
        slc[axis] = slice(int(start), int(start) + int(sz))
        out.append(dense[tuple(slc)])
        start += int(sz)
    return out


def _bcoo_concat(bcoo_vals, shape):
    """Concatenate a list of BCOO matrices by stacking their (data, indices)."""
    all_data = [b.data for b in bcoo_vals]
    all_indices = [b.indices for b in bcoo_vals]
    if all(isinstance(b.data, np.ndarray) for b in bcoo_vals):
        cat_data = np.concatenate(all_data, axis=-1)
        cat_indices = np.concatenate(all_indices, axis=-2)
    else:
        cat_data = jnp.concatenate(all_data, axis=-1)
        cat_indices = jnp.concatenate(all_indices, axis=-2)
    # pyrefly: ignore [bad-argument-type]
    return sparse.BCOO((cat_data, cat_indices), shape=shape,
                       indices_sorted=False, unique_indices=False)


@squeeze_op.register(sparse.BCOO)
def _(op: sparse.BCOO, *, n, **params) -> jax.Array:
    return lax.squeeze(op.todense(), params["dimensions"])


@broadcast_in_dim_op.register(sparse.BCOO)
def _(op: sparse.BCOO, *, n, **params) -> jax.Array:
    # Walk-frame: shape ends in n, bd ends in n's mapping. Strip both
    # for the spatial-only call to _bid_with_extra_batch.
    shape = tuple(params["shape"])[:-1]
    broadcast_dimensions = tuple(params["broadcast_dimensions"])[:-1]
    return _bid_with_extra_batch(op.todense(), shape, broadcast_dimensions, n)
