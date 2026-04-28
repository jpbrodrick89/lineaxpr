"""BEllpack registrations for reshape, broadcast_in_dim, reduce_sum, cumsum.

Extracted from _rules/unary.py; singledispatch registrations extend the
base functions defined in _linops/base.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import sparse

from .base import (
    broadcast_in_dim_op,
    cumsum_op,
    reduce_sum_op,
    reshape_op,
    split_op,
)
from .ellpack import BEllpack, _bellpack_unbatch


# ---------------------------------------------------------------------------
# reshape_op registrations
# ---------------------------------------------------------------------------

@reshape_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    new_sizes = tuple(int(s) for s in params["new_sizes"])

    # Pass-through: unbatched BCOO already the target shape.
    # (handled separately via BCOO base; this path won't fire for BEllpack)

    # Batched BEllpack → unbatched BEllpack: flatten leading (batch + out).
    if (op.n_batch >= 1
            and len(new_sizes) == 1
            and int(np.prod(op.batch_shape)) * op.out_size == int(new_sizes[0])):
        prod_b = int(np.prod(op.batch_shape))
        total = prod_b * op.out_size
        if op.k == 1:
            new_values = op.values.reshape(total)
        else:
            new_values = op.values.reshape(total, op.k)
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                rs = np.arange(c.start or 0, c.stop or op.nrows, c.step or 1)
                c_full = np.broadcast_to(rs, op.batch_shape + (op.nrows,))
                new_in_cols.append(c_full.reshape(total))
            elif isinstance(c, np.ndarray):
                if c.ndim == 1:
                    c_full = np.broadcast_to(c, op.batch_shape + (op.nrows,))
                    new_in_cols.append(c_full.reshape(total))
                else:
                    new_in_cols.append(c.reshape(total))
            else:
                ca = jnp.asarray(c)
                if ca.ndim == 1:
                    ca = jnp.broadcast_to(ca, op.batch_shape + (op.nrows,))
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(ca.reshape(total))
        return BEllpack(
            start_row=0, end_row=total,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=total, in_size=op.in_size,
        )

    # Unbatched BEllpack (N,) → (A, B): unflatten into batched BE.
    if (op.n_batch == 0
            and len(new_sizes) == 2
            and int(new_sizes[0]) * int(new_sizes[1]) == op.out_size
            and int(new_sizes[1]) > 1
            and op.start_row == 0 and op.end_row == op.out_size):
        A = int(new_sizes[0])
        B_out = int(new_sizes[1])
        new_batch = (A,)
        if op.k == 1:
            new_values = op.values.reshape(A, B_out)
        else:
            new_values = op.values.reshape(A, B_out, op.k)
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                rs = np.arange(c.start or 0, c.stop or op.nrows, c.step or 1)
                new_in_cols.append(rs.reshape(A, B_out))
            elif isinstance(c, np.ndarray):
                new_in_cols.append(c.reshape(A, B_out))
            else:
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(jnp.asarray(c).reshape(A, B_out))
        return BEllpack(
            start_row=0, end_row=B_out,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=B_out, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Leading-singleton reshape: (N,) → (1, ..., 1, N).
    if (op.n_batch == 0
            and len(new_sizes) >= 2
            and new_sizes[-1] == op.out_size
            and all(s == 1 for s in new_sizes[:-1])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(new_sizes[:-1])
        prefix = (1,) * len(new_batch)
        if op.k == 1:
            new_values = op.values.reshape(prefix + (op.nrows,))
        else:
            new_values = op.values.reshape(prefix + (op.nrows, op.k))
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif isinstance(c, np.ndarray):
                if c.ndim == 1:
                    # pyrefly: ignore [bad-argument-type]
                    new_in_cols.append(c)
                else:
                    # pyrefly: ignore [bad-argument-type]
                    new_in_cols.append(c.reshape(prefix + c.shape))
            else:
                ca = jnp.asarray(c)
                if ca.ndim == 1:
                    # pyrefly: ignore [bad-argument-type]
                    new_in_cols.append(ca)
                else:
                    # pyrefly: ignore [bad-argument-type]
                    new_in_cols.append(ca.reshape(prefix + ca.shape))
        return BEllpack(
            start_row=op.start_row, end_row=op.end_row,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=op.out_size, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Singleton-axis-insert: (N,) → (N, 1, ..., 1).
    if (op.n_batch == 0
            and len(new_sizes) >= 2
            and new_sizes[0] == op.out_size
            and all(s == 1 for s in new_sizes[1:])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(new_sizes[:-1])
        if op.k == 1:
            new_values = op.values.reshape(new_batch + (1,))
        else:
            new_values = op.values.reshape(new_batch + (1, op.k))
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif isinstance(c, np.ndarray):
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(c.reshape(new_batch + (1,) + c.shape[1:]))
            else:
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(jnp.asarray(c).reshape(
                    new_batch + (1,) + c.shape[1:]))
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=1, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Dense fallback.
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    dense = _to_dense(op, n)
    return lax.reshape(dense, tuple(new_sizes) + (n,))


# BCOO batched → flat reshape registration.
@reshape_op.register(sparse.BCOO) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    new_sizes = tuple(int(s) for s in params["new_sizes"])

    # Pass-through: already the target shape.
    if (op.n_batch == 0
            and len(new_sizes) == 1
            and op.shape == (int(new_sizes[0]), op.shape[-1])):
        return op

    # Flatten batched BCOO's leading (batch + out) axes.
    if (op.n_batch >= 1
            and len(new_sizes) == 1
            and int(np.prod(op.shape[:-1])) == int(new_sizes[0])):
        nb = op.n_batch
        old_out = op.shape[nb]
        in_size = op.shape[-1]
        batch_total = int(np.prod(op.shape[:nb]))
        nse_per_batch = op.data.shape[nb]
        flat_data = op.data.reshape(batch_total, nse_per_batch)
        flat_indices = op.indices.reshape(batch_total, nse_per_batch, 2)
        offsets = jnp.arange(batch_total, dtype=flat_indices.dtype) * old_out
        new_rows = flat_indices[..., 0] + offsets[:, None]
        new_cols = flat_indices[..., 1]
        new_indices = jnp.stack(
            [new_rows.reshape(-1), new_cols.reshape(-1)], axis=1,
        )
        return sparse.BCOO(
            (flat_data.reshape(-1), new_indices),
            shape=(int(new_sizes[0]), in_size),
        )

    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    dense = _to_dense(op, n)
    return lax.reshape(dense, tuple(new_sizes) + (n,))


# ---------------------------------------------------------------------------
# broadcast_in_dim_op registrations
# ---------------------------------------------------------------------------

@broadcast_in_dim_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    shape = params["shape"]
    broadcast_dimensions = params["broadcast_dimensions"]

    # Linear form (aval ()) broadcast to shape (1,): pass through BE row-vector.
    if (broadcast_dimensions == () and tuple(shape) == (1,)
            and op.n_batch == 0 and op.out_size == 1
            and op.start_row == 0 and op.end_row == 1):
        return op

    # Tile 1-row BEllpack to N-row vector via empty broadcast_dimensions.
    if (op.n_batch == 0
            and op.out_size == 1 and op.start_row == 0 and op.end_row == 1
            and broadcast_dimensions == () and len(shape) == 1):
        N = int(shape[0])
        new_values = jnp.broadcast_to(op.values, (N,) + op.values.shape[1:])
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                resolved = c
                new_in_cols.append(np.broadcast_to(resolved, (N,)).copy())  # pyrefly: ignore [no-matching-overload]
            elif isinstance(c, np.ndarray):
                new_in_cols.append(np.broadcast_to(c, (N,)).copy())
            else:
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(jnp.broadcast_to(c, (N,)))
        return BEllpack(
            start_row=0, end_row=N,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=N, in_size=op.in_size,
        )

    # Fallback normalisation: BEllpack row-vector (aval ()) with non-trivial bd.
    if (op.n_batch == 0
            and op.out_size == 1 and op.start_row == 0 and op.end_row == 1
            and broadcast_dimensions == ()):
        from lineaxpr._linops import _to_dense  # noqa: PLC0415
        op_dense = _to_dense(op, n)[0]
        expected_ndim = len(broadcast_dimensions) + 1
        while op_dense.ndim > expected_ndim and op_dense.shape[0] == 1:
            op_dense = op_dense[0]
        out_dims = tuple(broadcast_dimensions) + (len(shape),)
        return lax.broadcast_in_dim(op_dense, tuple(shape) + (n,), out_dims)

    # Unbatched BE: append broadcast dimensions as new batch axes.
    if (op.n_batch == 0
            and len(broadcast_dimensions) == 1
            and broadcast_dimensions[0] == len(shape) - 1
            and shape[-1] == op.out_size):
        new_batch = tuple(shape[:-1])
        new_values_shape = new_batch + op.values.shape
        new_values = jnp.broadcast_to(op.values, new_values_shape)
        return BEllpack(
            start_row=op.start_row, end_row=op.end_row,
            in_cols=op.in_cols, values=new_values,
            out_size=op.out_size, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Prepend leading batch axes to an already-batched BE.
    input_rank = op.n_batch + 1
    if (len(broadcast_dimensions) == input_rank
            and len(shape) > input_rank
            and broadcast_dimensions == tuple(
                range(len(shape) - input_rank, len(shape)))
            and shape[-1] == op.out_size):
        prepend = tuple(shape[:len(shape) - input_rank])
        new_batch = prepend + op.batch_shape
        new_values = jnp.broadcast_to(op.values, prepend + op.values.shape)
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice) or c.ndim == 1:
                new_in_cols.append(c)
                continue
            target = prepend + c.shape
            if isinstance(c, np.ndarray):
                new_in_cols.append(np.broadcast_to(c, target))
            else:
                new_in_cols.append(jnp.broadcast_to(c, target))
        return BEllpack(
            start_row=op.start_row, end_row=op.end_row,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=op.out_size, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Trailing-singleton: unbatched BE aval-(n,) → (n, 1, ..., 1).
    if (op.n_batch == 0
            and len(broadcast_dimensions) == 1
            and broadcast_dimensions[0] == 0
            and len(shape) >= 2
            and shape[0] == op.out_size
            and all(s == 1 for s in shape[1:])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(shape[:-1])
        if op.k == 1:
            new_values = op.values.reshape(new_batch + (1,))
        else:
            new_values = op.values.reshape(new_batch + (1, op.k))
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            else:
                new_in_cols.append(c.reshape(new_batch + (1,) + c.shape[1:]))
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=1, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Leading-dim row-broadcast: unbatched BE out=N → (N, M_1, ..., M_{r-1}).
    if (op.n_batch == 0
            and len(broadcast_dimensions) == 1
            and broadcast_dimensions[0] == 0
            and len(shape) >= 2
            and shape[0] == op.out_size
            and any(s > 1 for s in shape[1:])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(shape[:-1])
        new_out = int(shape[-1])
        N = op.out_size
        if op.k == 1:
            reshape_shape = (N,) + (1,) * (len(new_batch) - 1) + (1,)
            new_values = jnp.broadcast_to(
                op.values.reshape(reshape_shape),
                new_batch + (new_out,),
            )
        else:
            reshape_shape = (N,) + (1,) * (len(new_batch) - 1) + (1, op.k)
            new_values = jnp.broadcast_to(
                op.values.reshape(reshape_shape),
                new_batch + (new_out, op.k),
            )
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif isinstance(c, np.ndarray):
                reshape_c = (N,) + (1,) * (len(new_batch) - 1) + (1,) + c.shape[1:]
                new_in_cols.append(
                    # pyrefly: ignore [bad-argument-type]
                    np.broadcast_to(c.reshape(reshape_c),
                                    new_batch + (new_out,) + c.shape[1:])
                )
            else:
                ca = jnp.asarray(c)
                reshape_c = (N,) + (1,) * (len(new_batch) - 1) + (1,) + c.shape[1:]
                new_in_cols.append(
                    # pyrefly: ignore [bad-argument-type]
                    jnp.broadcast_to(ca.reshape(reshape_c),
                                     new_batch + (new_out,) + c.shape[1:])
                )
        return BEllpack(
            start_row=0, end_row=new_out,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=new_out, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Dense fallback.
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    dense = _to_dense(op, n)
    expected_ndim = len(broadcast_dimensions) + 1
    while dense.ndim > expected_ndim and dense.shape[0] == 1:
        dense = dense[0]
    out_dims = tuple(broadcast_dimensions) + (len(shape),)
    return lax.broadcast_in_dim(dense, tuple(shape) + (n,), out_dims)


# Handle the jax.Array dense linear-form path for broadcast_in_dim.
@broadcast_in_dim_op.register(jax.Array) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    shape = params["shape"]
    broadcast_dimensions = params["broadcast_dimensions"]
    # Dense linear-form (n,)-ndarray broadcast to shape (1,) with empty bd.
    if broadcast_dimensions == () and tuple(shape) == (1,):
        if op.ndim == 1 and op.shape[0] == n:
            zeros_row = jnp.zeros((n,), dtype=jnp.int32)
            cols = jnp.arange(n, dtype=jnp.int32)
            indices = jnp.stack([zeros_row, cols], axis=1)
            return sparse.BCOO((op, indices), shape=(1, n))
    # General dense fallback.
    dense = op
    expected_ndim = len(broadcast_dimensions) + 1
    while dense.ndim > expected_ndim and dense.shape[0] == 1:
        dense = dense[0]
    out_dims = tuple(broadcast_dimensions) + (len(shape),)
    return lax.broadcast_in_dim(dense, tuple(shape) + (n,), out_dims)


# ---------------------------------------------------------------------------
# reduce_sum_op registrations
# ---------------------------------------------------------------------------

@reduce_sum_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    axes = params["axes"]

    # BEllpack with leading batch dims.
    if op.n_batch > 0:
        axes_t = tuple(sorted(axes))
        # Full batch reduction.
        if axes_t == tuple(range(op.n_batch)):
            from lineaxpr._rules.add import _add_rule  # noqa: PLC0415
            slices = _bellpack_unbatch(op)
            if len(slices) == 1:
                return slices[0]
            return _add_rule(list(slices), [True] * len(slices), n)
        # Partial batch-axis reduction.
        if (axes_t and axes_t[-1] < op.n_batch
                and len(axes_t) < op.n_batch):
            reduced = set(axes_t)
            safe = True
            for c in op.in_cols:
                if isinstance(c, slice):
                    continue
                if c.ndim == 1:
                    continue
                for a in axes_t:
                    if a < c.ndim and c.shape[a] != 1:
                        safe = False
                        break
                if not safe:
                    break
            if safe:
                new_values = op.values.sum(axis=axes_t)
                new_in_cols = []
                for c in op.in_cols:
                    if isinstance(c, slice) or c.ndim == 1:
                        new_in_cols.append(c)
                    else:
                        new_in_cols.append(c.squeeze(axis=axes_t))
                new_batch = tuple(
                    s for i, s in enumerate(op.batch_shape) if i not in reduced
                )
                return BEllpack(
                    op.start_row, op.end_row,
                    tuple(new_in_cols), new_values,
                    op.out_size, op.in_size,
                    batch_shape=new_batch,
                )
        # Out-axis-only reduction on single-batch-axis BEllpack.
        if (axes_t == (op.n_batch,) and op.n_batch == 1
                and op.start_row == 0 and op.end_row == op.out_size):
            from lineaxpr._rules.add import _densify_if_wider_than_dense  # noqa: PLC0415
            B = op.batch_shape[0]
            O = op.out_size
            K = op.k
            new_in_cols = []
            for r in range(O):
                for b in range(K):
                    c = op.in_cols[b]
                    if isinstance(c, slice):
                        rs = np.arange(c.start or 0, c.stop or O, c.step or 1)
                        new_in_cols.append(np.broadcast_to(
                            np.asarray(rs[r]), (B,)).copy())
                    elif isinstance(c, np.ndarray) and c.ndim == 1:
                        new_in_cols.append(np.broadcast_to(
                            c[r:r+1], (B,)).copy())
                    elif isinstance(c, np.ndarray) and c.ndim == 2:
                        new_in_cols.append(c[:, r])
                    else:
                        c_full = c if c.ndim >= 2 else jnp.broadcast_to(
                            c, op.batch_shape + (op.nrows,))
                        # pyrefly: ignore [bad-argument-type]
                        new_in_cols.append(c_full[:, r])
            if K == 1:
                new_values = op.values
            else:
                new_values = op.values.reshape(B, O * K)

            def _col_key(c):
                if isinstance(c, np.ndarray):
                    return ("np", c.shape, c.tobytes())
                if isinstance(c, slice):
                    return ("slc", c.start, c.stop, c.step)
                return ("id", id(c))

            assigned = np.empty(len(new_in_cols), dtype=np.int64)
            group_cols: list = []
            key_to_group: dict = {}
            for i, c in enumerate(new_in_cols):
                k_ = _col_key(c)
                g = key_to_group.get(k_)
                if g is None:
                    g = len(group_cols)
                    key_to_group[k_] = g
                    group_cols.append(c)
                assigned[i] = g
            n_groups = len(group_cols)
            if n_groups < len(new_in_cols):
                dedup_values = jnp.zeros(
                    (B, n_groups), dtype=new_values.dtype
                ).at[:, assigned].add(new_values)
                return _densify_if_wider_than_dense(BEllpack(
                    start_row=0, end_row=B,
                    in_cols=tuple(group_cols), values=dedup_values,
                    out_size=B, in_size=op.in_size,
                ), n)
            return _densify_if_wider_than_dense(BEllpack(
                start_row=0, end_row=B,
                in_cols=tuple(new_in_cols), values=new_values,
                out_size=B, in_size=op.in_size,
            ), n)

    # BEllpack row-sum (unbatched, axis 0): emit a BEllpack row-vector whose
    # bands hold per-col sums when static cols allow trace-time sparsity analysis.
    if tuple(axes) == (0,) and op.n_batch == 0:
        k = op.k
        in_size = op.in_size
        per_band_cols = list(op.in_cols)
        if all(isinstance(c, np.ndarray) for c in per_band_cols):
            cols_flat = np.concatenate(per_band_cols)
            valid = cols_flat >= 0
            cols_valid = cols_flat[valid]
            uniq_cols, inverse = np.unique(cols_valid, return_inverse=True)
            n_groups = uniq_cols.shape[0]
            if 0 < n_groups < in_size:
                vals_flat = op.values if k == 1 else op.values.T.reshape(-1)
                keep = np.nonzero(valid)[0]
                vals_keep = jnp.take(vals_flat, jnp.asarray(keep)) if keep.shape[0] < cols_flat.shape[0] else vals_flat
                summed = jnp.zeros((n_groups,), op.dtype).at[jnp.asarray(inverse)].add(vals_keep)
                if n_groups == 1:
                    return BEllpack(start_row=0, end_row=1,
                                   in_cols=(np.asarray([uniq_cols[0]], dtype=uniq_cols.dtype),),
                                   values=summed.reshape(1), out_size=1, in_size=in_size)
                return BEllpack(start_row=0, end_row=1,
                               in_cols=tuple(np.asarray([c], dtype=uniq_cols.dtype) for c in uniq_cols),
                               values=summed.reshape(1, n_groups), out_size=1, in_size=in_size)
        cols_stacked = jnp.concatenate([jnp.asarray(c) for c in per_band_cols], axis=0)
        vals_stacked = op.values if k == 1 else op.values.T.reshape(-1)
        mask = cols_stacked >= 0
        return jnp.zeros((in_size,), op.dtype).at[
            jnp.where(mask, cols_stacked, 0)].add(jnp.where(mask, vals_stacked, jnp.zeros((), op.dtype)))

    # Dense fallback.
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    dense = _to_dense(op, n)
    return jnp.sum(dense, axis=tuple(axes))


# ---------------------------------------------------------------------------
# cumsum_op registration
# ---------------------------------------------------------------------------

@cumsum_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    """BEllpack cumsum: dense fallback."""
    axis = params["axis"]
    reverse = params.get("reverse", False)
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    dense = _to_dense(op, n)
    return lax.cumsum(dense, axis=axis, reverse=reverse)


# ---------------------------------------------------------------------------
# split_op registrations
# ---------------------------------------------------------------------------

@split_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    sizes = params["sizes"]
    axis = params["axis"]
    # Structural path: batched BE split along the out-axis (== n_batch).
    # Slice values and each band's cols along the out axis; keep batch_shape.
    # Requires full out coverage so each chunk's rows map to [0, sz) cleanly.
    if (op.n_batch >= 1
            and axis == op.n_batch
            and op.start_row == 0
            and op.end_row == op.out_size):
        nb = op.n_batch
        out = []
        start = 0
        for sz in sizes:
            sz_i = int(sz)
            end = start + sz_i
            val_slc = [slice(None)] * op.values.ndim
            val_slc[nb] = slice(start, end)
            new_values = op.values[tuple(val_slc)]
            new_in_cols = []
            for c in op.in_cols:
                arr = c
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 1:
                        new_in_cols.append(arr[start:end])
                    else:
                        slc = [slice(None)] * arr.ndim
                        slc[nb] = slice(start, end)
                        new_in_cols.append(arr[tuple(slc)])
                else:
                    arr_j = jnp.asarray(arr)
                    if arr_j.ndim == 1:
                        # pyrefly: ignore [bad-argument-type]
                        new_in_cols.append(arr_j[start:end])
                    else:
                        slc = [slice(None)] * arr_j.ndim
                        slc[nb] = slice(start, end)
                        # pyrefly: ignore [bad-argument-type]
                        new_in_cols.append(arr_j[tuple(slc)])
            out.append(BEllpack(
                0, sz_i, tuple(new_in_cols), new_values,
                sz_i, op.in_size, batch_shape=op.batch_shape,
            ))
            start = end
        return out
    # Structural path: split along output axis 0 (the "out_size" dim).
    # For an unbatched BEllpack with static cols we slice the BE
    # per-chunk (row range + per-band-col row-slice) and emit one
    # proper BCOO per chunk. Going through `_to_bcoo` on the full BE
    # and then masking out-of-range rows to `(row=0, value=0)` would
    # leave zero-valued entries clogging row 0 of every chunk — those
    # count as BCOO nse and manufacture "duplicates" at row 0 that
    # propagate through every downstream add/concat (observed as
    # COATING's 4.5× final nse bloat).
    if (axis == 0
            and op.n_batch == 0
            and all(isinstance(c, np.ndarray) or isinstance(c, slice)
                    for c in op.in_cols)):
        out = []
        start = 0
        for sz in sizes:
            sz_i = int(sz)
            end = start + sz_i
            # Row range [start, end) intersected with BE's own
            # [start_row, end_row). Slice cols/values along the row axis.
            be_start = max(op.start_row - start, 0)
            be_end = min(op.end_row - start, sz_i)
            if be_end <= be_start:
                out.append(sparse.BCOO(
                    # pyrefly: ignore [bad-argument-type]
                    (jnp.zeros((0,), op.values.dtype),
                     np.zeros((0, 2), np.int32)),
                    shape=(sz_i, op.in_size),
                ))
                start = end
                continue
            row_lo = max(start, op.start_row) - op.start_row
            row_hi = min(end, op.end_row) - op.start_row
            new_in_cols = []
            for c in op.in_cols:
                if isinstance(c, slice):
                    c = c
                new_in_cols.append(c[row_lo:row_hi])
            if op.k == 1:
                new_values = op.values[row_lo:row_hi]
            else:
                new_values = op.values[row_lo:row_hi, :]
            chunk_be = BEllpack(
                be_start, be_end, tuple(new_in_cols), new_values,
                sz_i, op.in_size,
            )
            # pyrefly: ignore [bad-argument-type]
            out.append(chunk_be)
            start = end
        return out
    # Fall through to BCOO-based split (handled in bcoo_extend.py for
    # ConstantDiagonal/Diagonal/BCOO; for BEllpack fall through to dense).
    from lineaxpr._linops import _to_bcoo  # noqa: PLC0415
    if axis == 0:
        bcoo = _to_bcoo(op, n)
        rows = bcoo.indices[:, 0]
        out = []
        start = 0
        for sz in sizes:
            end = start + int(sz)
            in_range = (rows >= start) & (rows < end)
            new_rows = jnp.where(in_range, rows - start, 0)
            new_data = jnp.where(in_range, bcoo.data,
                                 jnp.zeros((), bcoo.data.dtype))
            new_indices = jnp.stack(
                [new_rows, bcoo.indices[:, 1]], axis=1
            )
            out.append(sparse.BCOO(
                (new_data, new_indices), shape=(int(sz), bcoo.shape[1])
            ))
            start = end
        return out
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    dense = _to_dense(op, n)
    out = []
    start = 0
    for sz in sizes:
        slc = [slice(None)] * dense.ndim
        slc[axis] = slice(int(start), int(start) + int(sz))
        out.append(dense[tuple(slc)])
        start += int(sz)
    return out
