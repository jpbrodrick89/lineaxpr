"""ConstantDiagonal and Diagonal LinOp classes."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import sparse

from .ellpack import BEllpack

from .base import (
    broadcast_in_dim_op,
    gather_op,
    pad_op,
    reduce_sum_op,
    replace_slots,
    reshape_op,
    rev_op,
    scale_per_out_row,
    scale_scalar,
    slice_op,
    split_op,
    squeeze_op,
)
from .dense import _bid_with_extra_batch


class ConstantDiagonal:
    """Diagonal matrix with all entries equal to `value`."""

    __slots__ = ("n", "data")

    def __init__(self, n: int, data: Any = 1.0):
        self.n = n
        self.data = data

    @property
    def shape(self):
        return (self.n, self.n)

    @property
    def dtype(self):
        return jnp.asarray(self.data).dtype

    def todense(self):
        if isinstance(self.data, float) and self.data == 1.0:
            return jnp.eye(self.n)
        return self.data * jnp.eye(self.n)

    def to_bcoo(self):
        return _diag_to_bcoo(self.n, jnp.full((self.n,), self.data))

    def transpose(self, axes: tuple[int, ...] | None = None):
        # Symmetric: any 2D perm preserves the matrix.
        return self

    def __neg__(self):
        return replace_slots(self, data=-self.data)


def Identity(n: int, dtype=None):
    """The n×n identity as a ConstantDiagonal(n, 1.0).

    The standard seed for `lineaxpr.sparsify` when extracting the full
    Jacobian of a linear function.
    """
    data = jnp.asarray(1.0, dtype=dtype) if dtype is not None else 1.0
    return ConstantDiagonal(n, data)


class Diagonal:
    """Diagonal matrix `diag(values)` for a length-n vector."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    @property
    def n(self):
        return self.data.shape[0]

    @property
    def shape(self):
        return (self.n, self.n)

    @property
    def dtype(self):
        return self.data.dtype

    def todense(self):
        # `where(eye_bool, v[:, None], 0)` — same shape family as
        # `jnp.diag`, but with an explicit column-broadcast. `jnp.diag`'s
        # own body passes `v` with shape `(n,)`, which NumPy/JAX
        # broadcasting aligns to `(1, n)` against the eye mask; that
        # collapses each row to a gather across `v` and is pathologically
        # slow (measured 117µs at n=200 vs 12µs here — 10×). The
        # `v[:, None]` column form avoids that.
        # Only competitively-relevant signal on the affected-problem
        # subset (Linux clean, 3 reps): TABLE8 materialize flips from
        # losing to jax.hessian-folded (178µs, v*eye) to beating it
        # (147µs, this). Other impl-dependent deltas are either low
        # signal (we're 3–60× behind asdex-bcoo), materialize-only
        # (dense path is rare in optimizer loops), or within noise.
        # Scatter was tested too: +212% regression on LIARWHD bcoo and
        # +90% on FLETBV3M family — don't consider.
        # See docs/BENCH_HARNESS_NOTES.md for why isolated macOS-native
        # numbers disagree; trust the clean Linux container.
        return jnp.where(
            jnp.eye(self.n, dtype=jnp.bool_),
            self.data[:, None],
            jnp.zeros((), self.data.dtype),
        )

    def to_bcoo(self):
        return _diag_to_bcoo(self.n, self.data)

    def transpose(self, axes: tuple[int, ...] | None = None):
        return self  # symmetric

    def __neg__(self):
        return replace_slots(self, data=-self.data)


def _diag_to_bcoo(n: int, values) -> sparse.BCOO:
    """Convert a length-n diagonal values array to an (n, n) BCOO."""
    idx = jnp.arange(n)
    indices = jnp.stack([idx, idx], axis=1)
    return sparse.BCOO((values, indices), shape=(n, n),
                       indices_sorted=True, unique_indices=True)


# ---- singledispatch registrations ----

@scale_scalar.register(ConstantDiagonal)
def _(op, s):
    return ConstantDiagonal(op.n, s * op.data)


@scale_scalar.register(Diagonal)
def _(op, s):
    return Diagonal(s * op.data)


@scale_per_out_row.register(ConstantDiagonal)
def _(op, v):
    return Diagonal(op.data * jnp.asarray(v))


@scale_per_out_row.register(Diagonal)
def _(op, v):
    return Diagonal(op.data * jnp.asarray(v))


# ---- unary structural op registrations ----

def _trailing_is_in_axis_noop(starts, limits, strides, in_size):
    """True when the trailing slice spec is a full no-op over `in_size`.

    Phase B: jaxpr params pass straight through (no `[:-1]` strip). When
    the trailing axis is a no-op slice covering the full `in_size`, it
    corresponds to the LinOp's implicit in-axis (e.g. vmap's V-axis at
    -1) and the structural branches operate on the leading out-axes.
    """
    return (len(starts) >= 2
            and int(starts[-1]) == 0
            and int(limits[-1]) == int(in_size)
            and int(strides[-1]) == 1)


@slice_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    starts = params["start_indices"]
    limits = params["limit_indices"]
    strides = params.get("strides") or (1,) * len(starts)
    if (len(starts) == 2
            and _trailing_is_in_axis_noop(starts, limits, strides, op.n)):
        s, e = starts[0], limits[0]
        stride = strides[0]
        cols = np.arange(s, e, stride)
        k_out = len(cols)
        data_b = jnp.broadcast_to(jnp.asarray(op.data), (k_out,))
        return BEllpack(
            start_row=0, end_row=k_out,
            in_cols=(cols,), data=data_b,
            out_size=k_out, in_size=op.n,
        )
    return lax.slice(op.todense(), **params)


@slice_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    starts = params["start_indices"]
    limits = params["limit_indices"]
    strides = params.get("strides") or (1,) * len(starts)
    if (len(starts) == 2
            and _trailing_is_in_axis_noop(starts, limits, strides, op.n)):
        s, e = starts[0], limits[0]
        stride = strides[0]
        cols = np.arange(s, e, stride)
        k_out = len(cols)
        return BEllpack(
            start_row=0, end_row=k_out,
            in_cols=(cols,), data=op.data[s:e:stride],
            out_size=k_out, in_size=op.n,
        )
    # Axis-1-only slice: by V-axis-detection, V was at axis 0 of the
    # input (vmap-rewrote a primal-axis slice on a V-at-0 Diagonal).
    # Output has shape (n, n_cols) with V at axis 0; emit a T=True
    # BEllpack (in_size=n=V_size, out_size=n_cols=primal). Each
    # primal-out column j has at most one non-zero, at row
    # in_cols[0][j] = s_in + j*stride_in (which sits in [0, n)).
    # No sentinels needed — every kept output column has a valid row.
    if (len(starts) == 2
            and starts[0] == 0 and limits[0] == op.n and strides[0] == 1):
        s_in, e_in = int(starts[1]), int(limits[1])
        stride_in = int(strides[1])
        n_cols = (e_in - s_in + stride_in - 1) // stride_in
        rows_for_cols = (s_in + np.arange(n_cols, dtype=np.intp)
                         * stride_in)
        data = op.data[s_in:e_in:stride_in]
        return BEllpack(
            start_row=0, end_row=n_cols,
            in_cols=(rows_for_cols,), data=data,
            out_size=n_cols, in_size=op.n,
            transposed=True,
        )
    return lax.slice(op.todense(), **params)


@squeeze_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    dimensions = params["dimensions"]
    if not dimensions:
        return op
    # For a 1×1 CD, axes 0 and 1 are interchangeable (both size 1);
    # squeeze either to a 1D row-vector BE.
    if op.n == 1 and dimensions in ((0,), (1,)):
        val = jnp.asarray(op.data).reshape(1)
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=(np.asarray([0]),), data=val,
            out_size=1, in_size=1,
        )
    raise NotImplementedError(f"squeeze on diag with dims {dimensions}")


@squeeze_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    dimensions = params["dimensions"]
    if not dimensions:
        return op
    if op.n == 1 and dimensions in ((0,), (1,)):
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=(np.asarray([0]),), data=op.data,
            out_size=1, in_size=1,
        )
    raise NotImplementedError(f"squeeze on diag with dims {dimensions}")


@rev_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    dimensions = params["dimensions"]
    # CD = a·I is symmetric, so reversing axis 0 (rows) and axis 1
    # (cols) both yield the same anti-diagonal. dim=(1,) arises under
    # vmap(in_axes=0) where vmap rewrites rev's primal-axis index.
    if dimensions in ((0,), (1,)):
        cols = np.arange(op.n - 1, -1, -1, dtype=np.int64)
        data = jnp.broadcast_to(jnp.asarray(op.data, dtype=op.dtype), (op.n,))
        return BEllpack(start_row=0, end_row=op.n,
                        in_cols=(cols,), data=data,
                        out_size=op.n, in_size=op.n)
    return op


@rev_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    dimensions = params["dimensions"]
    # Reversing rows of diag(v) gives the anti-diagonal with v[n-1-i]
    # at (i, n-1-i) — values reversed.
    # Reversing cols of diag(v) gives the anti-diagonal with v[i]
    # at (i, n-1-i) — values not reversed (original col is moved).
    if dimensions == (0,):
        cols = np.arange(op.n - 1, -1, -1, dtype=np.int64)
        return BEllpack(start_row=0, end_row=op.n,
                        in_cols=(cols,), data=op.data[::-1],
                        out_size=op.n, in_size=op.n)
    if dimensions == (1,):
        cols = np.arange(op.n - 1, -1, -1, dtype=np.int64)
        return BEllpack(start_row=0, end_row=op.n,
                        in_cols=(cols,), data=op.data,
                        out_size=op.n, in_size=op.n)
    return op


@reshape_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    # Walk-frame new_sizes has n at -1; structural shape is the prefix.
    new_sizes = params["new_sizes"][:-1]
    if len(new_sizes) >= 2 and int(np.prod(new_sizes)) == op.n:
        batch_shape = new_sizes[:-1]
        nrows = new_sizes[-1]
        flat_idx = np.arange(op.n).reshape(new_sizes)
        data = jnp.broadcast_to(jnp.asarray(op.data), new_sizes)
        return BEllpack(
            start_row=0, end_row=nrows,
            in_cols=(flat_idx,), data=data,
            out_size=nrows, in_size=op.n,
            batch_shape=batch_shape,
        )
    return lax.reshape(op.todense(), params["new_sizes"],
                       dimensions=params.get("dimensions"),
                       out_sharding=params.get("sharding"))


@reshape_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    new_sizes = params["new_sizes"][:-1]
    if len(new_sizes) >= 2 and int(np.prod(new_sizes)) == op.n:
        batch_shape = new_sizes[:-1]
        nrows = new_sizes[-1]
        flat_idx = np.arange(op.n).reshape(new_sizes)
        data = op.data.reshape(new_sizes)
        return BEllpack(
            start_row=0, end_row=nrows,
            in_cols=(flat_idx,), data=data,
            out_size=nrows, in_size=op.n,
            batch_shape=batch_shape,
        )
    return lax.reshape(op.todense(), params["new_sizes"],
                       dimensions=params.get("dimensions"),
                       out_sharding=params.get("sharding"))


def _is_inside_vmap_bcast(shape, bd, op_n):
    """Detect inside-vmap broadcast: V slot is at the front of the output.

    CD/Diagonal are symmetric so transpose is a no-op (no `transposed`
    flag), but inside-vmap params still indicate V at output axis 0
    (mapped from input axis 0). When that pattern matches, densify
    rather than running the V-at-back structural branches below.
    """
    return bool(bd) and bd[0] == 0 and len(shape) >= 1 and shape[0] == op_n


@broadcast_in_dim_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    full_shape = tuple(params["shape"])
    full_bd = tuple(params["broadcast_dimensions"])
    if _is_inside_vmap_bcast(full_shape, full_bd, op.n):
        return lax.broadcast_in_dim(op.todense(), full_shape, full_bd)
    # Walk-frame: shape ends in n, bd ends in n's mapping. Strip both
    # for the spatial-only structural checks below.
    shape = full_shape[:-1]
    broadcast_dimensions = full_bd[:-1]
    # Diagonal (aval `(n,)`) broadcast to `(*pre, n, *post)` where all
    # non-n axes are size-1.
    if (len(broadcast_dimensions) == 1
            and shape[broadcast_dimensions[0]] == op.n
            and all(s == 1 for i, s in enumerate(shape)
                    if i != broadcast_dimensions[0])):
        bcast_axis = broadcast_dimensions[0]
        leading_singletons = tuple(shape[:bcast_axis])
        trailing_singletons = tuple(shape[bcast_axis + 1:])
        v = jnp.broadcast_to(jnp.asarray(op.data), (op.n,))
        if not trailing_singletons:
            cols = (np.arange(op.n),)
            new_values = jnp.broadcast_to(
                v.reshape((1,) * len(leading_singletons) + (op.n,)),
                leading_singletons + (op.n,),
            )
            return BEllpack(
                start_row=0, end_row=op.n,
                in_cols=cols, data=new_values,
                out_size=op.n, in_size=op.n,
                batch_shape=leading_singletons,
            )
        cols_2d = np.arange(op.n).reshape(
            (1,) * len(leading_singletons) + (op.n,)
            + (1,) * (len(trailing_singletons) - 1)
        )
        new_batch = leading_singletons + (op.n,) + trailing_singletons[:-1]
        cols_full = np.broadcast_to(cols_2d, new_batch).copy()
        new_values = jnp.broadcast_to(
            v.reshape((1,) * len(leading_singletons) + (op.n,)
                       + (1,) * (len(trailing_singletons) - 1)),
            new_batch,
        ).reshape(new_batch + (1,))
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=(cols_full[..., None],), data=new_values,
            out_size=1, in_size=op.n,
            batch_shape=new_batch,
        )
    return _bid_with_extra_batch(op.todense(), shape, broadcast_dimensions, n)


@broadcast_in_dim_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    full_shape = tuple(params["shape"])
    full_bd = tuple(params["broadcast_dimensions"])
    if _is_inside_vmap_bcast(full_shape, full_bd, op.n):
        return lax.broadcast_in_dim(op.todense(), full_shape, full_bd)
    # Walk-frame: shape ends in n, bd ends in n's mapping. Strip both
    # for the spatial-only structural checks below.
    shape = full_shape[:-1]
    broadcast_dimensions = full_bd[:-1]
    if (len(broadcast_dimensions) == 1
            and shape[broadcast_dimensions[0]] == op.n
            and all(s == 1 for i, s in enumerate(shape)
                    if i != broadcast_dimensions[0])):
        bcast_axis = broadcast_dimensions[0]
        leading_singletons = tuple(shape[:bcast_axis])
        trailing_singletons = tuple(shape[bcast_axis + 1:])
        v = op.data
        if not trailing_singletons:
            cols = (np.arange(op.n),)
            new_values = jnp.broadcast_to(
                v.reshape((1,) * len(leading_singletons) + (op.n,)),
                leading_singletons + (op.n,),
            )
            return BEllpack(
                start_row=0, end_row=op.n,
                in_cols=cols, data=new_values,
                out_size=op.n, in_size=op.n,
                batch_shape=leading_singletons,
            )
        cols_2d = np.arange(op.n).reshape(
            (1,) * len(leading_singletons) + (op.n,)
            + (1,) * (len(trailing_singletons) - 1)
        )
        new_batch = leading_singletons + (op.n,) + trailing_singletons[:-1]
        cols_full = np.broadcast_to(cols_2d, new_batch).copy()
        new_values = jnp.broadcast_to(
            v.reshape((1,) * len(leading_singletons) + (op.n,)
                       + (1,) * (len(trailing_singletons) - 1)),
            new_batch,
        ).reshape(new_batch + (1,))
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=(cols_full[..., None],), data=new_values,
            out_size=1, in_size=op.n,
            batch_shape=new_batch,
        )
    return _bid_with_extra_batch(op.todense(), shape, broadcast_dimensions, n)


@reduce_sum_op.register(ConstantDiagonal)
def _(op, *, n, **params):
    """CD: dense fallback (see comment in unary.py about XLA fusion)."""
    axes = params["axes"]
    dense = op.todense()
    return jnp.sum(dense, axis=tuple(axes))


@reduce_sum_op.register(Diagonal)
def _(op, *, n, **params):
    """Diagonal: dense fallback (same XLA fusion reason as CD)."""
    axes = params["axes"]
    dense = op.todense()
    return jnp.sum(dense, axis=tuple(axes))


@gather_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, start_indices, **params):
    dnums = params["dimension_numbers"]
    point_gather_kept = (
        dnums.offset_dims == (1,)
        and dnums.collapsed_slice_dims == ()
        and dnums.start_index_map == (0,)
        and params["slice_sizes"] == (1,)
    )
    # V-augmented gather (post jax.vmap, V at axis 0): operand has
    # shape (V=n, primal_out=n); the gather collapses axis 1 (primal)
    # and preserves V as offset_dims=(0,). The output BE is the same
    # row-select but transposed=True to match the V-at-0 layout.
    sl = params["slice_sizes"]
    point_gather_v_collapsed_T = (
        tuple(dnums.offset_dims) == (0,)
        and tuple(dnums.collapsed_slice_dims) == (1,)
        and tuple(dnums.start_index_map) == (1,)
        and len(sl) == 2 and sl[0] == op.n and sl[1] == 1
    )
    row_idx = start_indices[..., 0]
    batch_shape = tuple(row_idx.shape[:-1])
    N = row_idx.shape[-1]
    vals = jnp.broadcast_to(jnp.asarray(op.data), batch_shape + (N,))
    if point_gather_kept:
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=(row_idx[..., None],),
            data=vals[..., None],
            out_size=1, in_size=op.n,
            batch_shape=batch_shape + (N,),
        )
    return BEllpack(
        start_row=0, end_row=N,
        in_cols=(row_idx,),
        data=vals,
        out_size=N, in_size=op.n,
        batch_shape=batch_shape,
        transposed=point_gather_v_collapsed_T,
    )


@split_op.register(ConstantDiagonal)
@split_op.register(Diagonal)  # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    sizes = params["sizes"]
    axis = params["axis"]
    if axis == 0:
        bcoo = op.to_bcoo() if hasattr(op, 'to_bcoo') else op
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
    dense = op.todense()
    out = []
    start = 0
    for sz in sizes:
        slc = [slice(None)] * dense.ndim
        slc[axis] = slice(int(start), int(start) + int(sz))
        out.append(dense[tuple(slc)])
        start += int(sz)
    return out


@gather_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, start_indices, **params):
    dnums = params["dimension_numbers"]
    point_gather_kept = (
        dnums.offset_dims == (1,)
        and dnums.collapsed_slice_dims == ()
        and dnums.start_index_map == (0,)
        and params["slice_sizes"] == (1,)
    )
    # V-augmented gather (V at axis 0); see ConstantDiagonal rule above.
    sl = params["slice_sizes"]
    point_gather_v_collapsed_T = (
        tuple(dnums.offset_dims) == (0,)
        and tuple(dnums.collapsed_slice_dims) == (1,)
        and tuple(dnums.start_index_map) == (1,)
        and len(sl) == 2 and sl[0] == op.n and sl[1] == 1
    )
    row_idx = start_indices[..., 0]
    batch_shape = tuple(row_idx.shape[:-1])
    N = row_idx.shape[-1]
    vals = jnp.take(op.data, row_idx)
    if point_gather_kept:
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=(row_idx[..., None],),
            data=vals[..., None],
            out_size=1, in_size=op.n,
            batch_shape=batch_shape + (N,),
        )
    return BEllpack(
        start_row=0, end_row=N,
        in_cols=(row_idx,),
        data=vals,
        out_size=N, in_size=op.n,
        batch_shape=batch_shape,
        transposed=point_gather_v_collapsed_T,
    )


@pad_op.register(ConstantDiagonal)
@pad_op.register(Diagonal)
def _(op, *, n, padding_value, **params) -> jax.Array:
    return lax.pad(op.todense(), padding_value, **params)
