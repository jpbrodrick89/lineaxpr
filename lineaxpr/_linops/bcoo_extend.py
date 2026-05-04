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


@scale_scalar.register(sparse.BCOO)
def _(op: sparse.BCOO, s) -> sparse.BCOO:
    return sparse.BCOO((s * op.data, op.indices), shape=op.shape)


@scale_per_out_row.register(sparse.BCOO)
def _(op: sparse.BCOO, v) -> sparse.BCOO:
    # Under V-at-0 layout (post-vmap), the BCOO has V at axis 0 and the
    # primal-output axis at axis 1. `scale_per_out_row` scales by the
    # output index, so use `indices[..., 1]`. op.indices shape:
    # `(nse, 2)` for unbatched; `(*batch, nse, 2)` for batched.
    out_idx = op.indices[..., 1]
    v_arr = jnp.asarray(v)
    return sparse.BCOO(
        (op.data * jnp.take(v_arr, out_idx), op.indices), shape=op.shape
    )


@slice_op.register(sparse.BCOO)
def _(op: sparse.BCOO, *, n, **params):
    # Common case: 1D stride-1 slice on unbatched BCOO. Hand-roll
    # to avoid `sparse.bcoo_slice`'s `(idx-start) % stride` and
    # `(idx-start+stride-1) // stride` chain, which JAX wraps in
    # `jit[name=floor_divide]` / `jit[name=remainder]` sub-jaxprs
    # that XLA can't fuse with surrounding ops. Each such call adds
    # 2 jit boundaries to the jaxpr; in HS110 with 5 slice calls
    # that's 10 boundaries removed and ~50% of the HLO size delta
    # vs main. (Mirrors main's `slice_op(BCOO)` handling.)
    starts = tuple(int(s) for s in params["start_indices"])
    limits = tuple(int(l) for l in params["limit_indices"])
    strides_p = params.get("strides")
    strides = (tuple(int(s) for s in strides_p)
               if strides_p else (1,) * len(starts))

    # Apply only to 2D unbatched stride-1 slice along axis 0 (the
    # most common shape we see). Other patterns delegate.
    if (len(starts) == 2 and strides == (1, 1) and op.n_batch == 0
            and starts[1] == 0 and limits[1] == op.shape[1]):
        s, e = starts[0], limits[0]
        k = e - s
        try:
            indices_np = np.asarray(op.indices)
        except (jax.errors.TracerArrayConversionError, TypeError):
            indices_np = None
        if isinstance(indices_np, np.ndarray):
            rows_np = indices_np[:, 0]
            keep = np.nonzero((rows_np >= s) & (rows_np < e))[0]
            new_indices = np.stack(
                [rows_np[keep] - s, indices_np[keep, 1]], axis=1)
            new_data = jnp.take(op.data, jnp.asarray(keep))
            return sparse.BCOO(
                (new_data, jnp.asarray(new_indices)),  # pyrefly: ignore [bad-argument-type]
                shape=(k, op.shape[1]))
        rows = op.indices[:, 0]
        in_range = (rows >= s) & (rows < e)
        new_rows = jnp.where(in_range, rows - s, 0)
        new_data = jnp.where(in_range, op.data,
                             jnp.zeros((), op.data.dtype))
        new_indices = jnp.stack(
            [new_rows, op.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (new_data, new_indices), shape=(k, op.shape[1]))
    if (len(starts) == 2 and strides == (1, 1) and op.n_batch == 0
            and starts[0] == 0 and limits[0] == op.shape[0]):
        # Stride-1 slice along axis 1 (primal-axis on V-at-0 layout).
        s, e = starts[1], limits[1]
        k = e - s
        try:
            indices_np = np.asarray(op.indices)
        except (jax.errors.TracerArrayConversionError, TypeError):
            indices_np = None
        if isinstance(indices_np, np.ndarray):
            cols_np = indices_np[:, 1]
            keep = np.nonzero((cols_np >= s) & (cols_np < e))[0]
            new_indices = np.stack(
                [indices_np[keep, 0], cols_np[keep] - s], axis=1)
            new_data = jnp.take(op.data, jnp.asarray(keep))
            return sparse.BCOO(
                (new_data, jnp.asarray(new_indices)),  # pyrefly: ignore [bad-argument-type]
                shape=(op.shape[0], k))
        cols = op.indices[:, 1]
        in_range = (cols >= s) & (cols < e)
        new_cols = jnp.where(in_range, cols - s, 0)
        new_data = jnp.where(in_range, op.data,
                             jnp.zeros((), op.data.dtype))
        new_indices = jnp.stack(
            [op.indices[:, 0], new_cols], axis=1)
        return sparse.BCOO(
            (new_data, new_indices), shape=(op.shape[0], k))
    return sparse.bcoo_slice(op, **params)


@pad_op.register(sparse.BCOO)
def _(op: sparse.BCOO, *, n, padding_value, **params):
    config = tuple(params["padding_config"])
    in_axis_noop = len(config) >= 2 and tuple(config[-1]) == (0, 0, 0)

    if len(config) == 2 and in_axis_noop:
        before, after, interior = config[0]
        before, after, interior = int(before), int(after), int(interior)
        if interior == 0:
            out_size = op.shape[0] + before + after
            # Negative pad cropping: drop entries whose row falls outside
            # the kept window. Without this, OOB indices pass through
            # and corrupt downstream gathers (e.g. `scale * BCOO`).
            if before < 0 or after < 0:
                rows = op.indices[:, 0]
                lo = max(-before, 0)
                hi = op.shape[0] - max(-after, 0)
                in_range = (rows >= lo) & (rows < hi)
                new_rows = jnp.where(in_range, rows - lo, 0)
                new_data = jnp.where(in_range, op.data,
                                     jnp.zeros((), op.data.dtype))
                new_indices = jnp.stack(
                    [new_rows, op.indices[:, 1]], axis=1)
                return sparse.BCOO(
                    (new_data, new_indices), shape=(out_size, op.shape[1]))
            new_rows = op.indices[:, 0] + before
            new_indices = jnp.stack([new_rows, op.indices[:, 1]], axis=1)
            return sparse.BCOO(
                (op.data, new_indices), shape=(out_size, op.shape[1])
            )
        step = interior + 1
        old_size = op.shape[0]
        out_size = old_size + before + after + interior * max(old_size - 1, 0)
        new_rows = op.indices[:, 0] * step + before
        new_indices = jnp.stack([new_rows, op.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (op.data, new_indices), shape=(out_size, op.shape[1])
        )

    # `(n, m) → (n, m')` via padding axis 1 with zeros (axis 0
    # no-op). All existing entries stay valid in `[0, m)`; just bump
    # `shape[1]` and shift cols if `before > 0`. Generalises the
    # earlier `(n, 1) → (n, m)` special case to any input axis-1
    # size. Bridges `reduce_sum → broadcast_in_dim → pad` chains
    # (LUKSAN11LS-class) and pads inside the chain too.
    if len(config) == 2 and tuple(config[0]) == (0, 0, 0):
        before_c, after_c, interior_c = config[1]
        before_c, after_c, interior_c = (
            int(before_c), int(after_c), int(interior_c))
        if interior_c == 0:
            new_shape1 = op.shape[1] + before_c + after_c
            # Negative pad cropping along axis 1: drop entries whose
            # col falls outside the kept window.
            if before_c < 0 or after_c < 0:
                cols = op.indices[:, 1]
                lo = max(-before_c, 0)
                hi = op.shape[1] - max(-after_c, 0)
                in_range = (cols >= lo) & (cols < hi)
                new_cols = jnp.where(in_range, cols - lo, 0)
                new_data = jnp.where(in_range, op.data,
                                     jnp.zeros((), op.data.dtype))
                new_indices = jnp.stack(
                    [op.indices[:, 0], new_cols], axis=1)
                return sparse.BCOO(
                    (new_data, new_indices),
                    shape=(op.shape[0], new_shape1))
            if before_c == 0:
                return sparse.BCOO(
                    (op.data, op.indices),
                    shape=(op.shape[0], new_shape1),
                    indices_sorted=op.indices_sorted,
                    unique_indices=op.unique_indices,
                )
            new_cols = op.indices[:, 1] + before_c
            new_indices = jnp.stack(
                [op.indices[:, 0], new_cols], axis=1)
            return sparse.BCOO(
                (op.data, new_indices),
                shape=(op.shape[0], new_shape1),
            )
        # Interior padding along axis 1: each existing col `j` maps
        # to `j*step + before_c`. The result still has the same
        # `op.indices.shape[0]` non-zero entries; only their col
        # positions shift.
        step = interior_c + 1
        old_cols_axis = op.shape[1]
        new_shape1 = (old_cols_axis + before_c + after_c
                      + interior_c * max(old_cols_axis - 1, 0))
        new_cols = op.indices[:, 1] * step + before_c
        new_indices = jnp.stack(
            [op.indices[:, 0], new_cols], axis=1)
        return sparse.BCOO(
            (op.data, new_indices),
            shape=(op.shape[0], new_shape1),
        )

    return lax.pad(op.todense(), padding_value, **params)


@rev_op.register(sparse.BCOO)
def _(op: sparse.BCOO, *, n, **params):
    return sparse.bcoo_rev(op, **params)


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
    return sparse.bcoo_reduce_sum(op, axes=tuple(axes))


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
    # General-purpose path for split along any sparse axis (n_batch=0,
    # n_sparse>=2). Mask out-of-range entries to `(col=0, value=0)`
    # per piece and offset the split-axis index. Each piece is a BCOO
    # at the same nse as input (with sentinels for entries outside).
    if op.n_batch == 0 and axis < op.ndim and op.indices.shape[-1] >= 2:
        sparse_axis = axis  # n_batch=0
        col_along = op.indices[:, sparse_axis]
        out = []
        start = 0
        n_idx_cols = op.indices.shape[-1]
        for sz in sizes:
            end = start + int(sz)
            in_range = (col_along >= start) & (col_along < end)
            new_along = jnp.where(in_range, col_along - start, 0)
            new_data = jnp.where(in_range, op.data,
                                 jnp.zeros((), op.data.dtype))
            new_index_cols = [
                op.indices[:, j] if j != sparse_axis else new_along
                for j in range(n_idx_cols)
            ]
            new_indices = jnp.stack(new_index_cols, axis=1)
            new_shape = list(op.shape)
            new_shape[axis] = int(sz)
            out.append(sparse.BCOO(
                (new_data, new_indices), shape=tuple(new_shape),
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
def _(op: sparse.BCOO, *, n, **params):
    # Squeeze on BCOO is almost always `(1, k)` or `(k, 1)` to a 1D
    # vector. Dense is cheap at that size and the dense
    # broadcast_in_dim path has structural sparsity-recovery for
    # `(n,) → (n, 1...)` re-sparsification. (Mirrors main's behavior.)
    return lax.squeeze(op.todense(), params["dimensions"])


@broadcast_in_dim_op.register(sparse.BCOO)
def _(op: sparse.BCOO, *, n, **params):
    # Densify and route through the dense path's sparsity-recovery.
    # Keeping BCOO machinery alive through `bcoo_broadcast_in_dim`
    # produces large amounts of int64 index broadcast HLO without
    # measurable speed benefit. (Mirrors main's behavior.)
    return lax.broadcast_in_dim_p.bind(op.todense(), **params)
