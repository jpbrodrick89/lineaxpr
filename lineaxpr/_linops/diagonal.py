"""ConstantDiagonal and Diagonal LinOp classes."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from .base import (
    broadcast_in_dim_op,
    gather_op,
    negate,
    reduce_sum_op,
    reshape_op,
    rev_op,
    scale_per_out_row,
    scale_scalar,
    slice_op,
    squeeze_op,
)


class ConstantDiagonal:
    """Diagonal matrix with all entries equal to `value`."""

    __slots__ = ("n", "value")

    def __init__(self, n: int, value: Any = 1.0):
        self.n = n
        self.value = value

    @property
    def shape(self):
        return (self.n, self.n)

    @property
    def dtype(self):
        return jnp.asarray(self.value).dtype

    def todense(self):
        if isinstance(self.value, float) and self.value == 1.0:
            return jnp.eye(self.n)
        return self.value * jnp.eye(self.n)

    def to_bcoo(self):
        return _diag_to_bcoo(self.n, jnp.full((self.n,), self.value))

    def transpose(self, permutation):
        return self  # symmetric


def Identity(n: int, dtype=None):
    """The n×n identity as a ConstantDiagonal(n, 1.0).

    The standard seed for `lineaxpr.sparsify` when extracting the full
    Jacobian of a linear function.
    """
    value = jnp.asarray(1.0, dtype=dtype) if dtype is not None else 1.0
    return ConstantDiagonal(n, value)


class Diagonal:
    """Diagonal matrix `diag(values)` for a length-n vector."""

    __slots__ = ("values",)

    def __init__(self, values):
        self.values = values

    @property
    def n(self):
        return self.values.shape[0]

    @property
    def shape(self):
        return (self.n, self.n)

    @property
    def dtype(self):
        return self.values.dtype

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
            self.values[:, None],
            jnp.zeros((), self.values.dtype),
        )

    def to_bcoo(self):
        return _diag_to_bcoo(self.n, self.values)

    def transpose(self, permutation):
        return self  # symmetric


def _diag_to_bcoo(n: int, values) -> sparse.BCOO:
    """Convert a length-n diagonal values array to an (n, n) BCOO."""
    idx = jnp.arange(n)
    indices = jnp.stack([idx, idx], axis=1)
    return sparse.BCOO((values, indices), shape=(n, n),
                       indices_sorted=True, unique_indices=True)


# ---- singledispatch registrations ----

@negate.register(ConstantDiagonal)
def _(op):
    return ConstantDiagonal(op.n, -op.value)


@negate.register(Diagonal)
def _(op):
    return Diagonal(-op.values)


@scale_scalar.register(ConstantDiagonal)
def _(op, s):
    return ConstantDiagonal(op.n, s * op.value)


@scale_scalar.register(Diagonal)
def _(op, s):
    return Diagonal(s * op.values)


@scale_per_out_row.register(ConstantDiagonal)
def _(op, v):
    return Diagonal(op.value * jnp.asarray(v))


@scale_per_out_row.register(Diagonal)
def _(op, v):
    return Diagonal(op.values * jnp.asarray(v))


# ---- unary structural op registrations ----
# These implement the CD/Diagonal-specific fast paths that unary.py used to
# branch on via isinstance checks. They forward to BEllpack construction, which
# is imported lazily (to avoid the module-level circular import through _linops).



@slice_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    starts = tuple(int(x) for x in params["start_indices"])
    limits = tuple(int(x) for x in params["limit_indices"])
    strides_p = params.get("strides")
    strides = tuple(int(x) for x in strides_p) if strides_p else (1,) * len(starts)
    from lineaxpr._linops.ellpack import BEllpack  # noqa: PLC0415
    if len(starts) == 1:
        s, e = starts[0], limits[0]
        stride = strides[0]
        cols = np.arange(s, e, stride)
        k_out = len(cols)
        values_b = jnp.broadcast_to(jnp.asarray(op.value), (k_out,))
        return BEllpack(
            start_row=0, end_row=k_out,
            in_cols=(cols,), values=values_b,
            out_size=k_out, in_size=op.n,
        )
    # Multi-dim: dense fallback
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    dense = _to_dense(op, n)
    s_full = starts + (0,)
    l_full = limits + (n,)
    str_full = strides + (1,)
    from jax import lax  # noqa: PLC0415
    return lax.slice(dense, s_full, l_full, str_full)


@slice_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    starts = tuple(int(x) for x in params["start_indices"])
    limits = tuple(int(x) for x in params["limit_indices"])
    strides_p = params.get("strides")
    strides = tuple(int(x) for x in strides_p) if strides_p else (1,) * len(starts)
    from lineaxpr._linops.ellpack import BEllpack  # noqa: PLC0415
    if len(starts) == 1:
        s, e = starts[0], limits[0]
        stride = strides[0]
        cols = np.arange(s, e, stride)
        k_out = len(cols)
        return BEllpack(
            start_row=0, end_row=k_out,
            in_cols=(cols,), values=op.values[s:e:stride],
            out_size=k_out, in_size=op.n,
        )
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    dense = _to_dense(op, n)
    s_full = starts + (0,)
    l_full = limits + (n,)
    str_full = strides + (1,)
    from jax import lax  # noqa: PLC0415
    return lax.slice(dense, s_full, l_full, str_full)


@squeeze_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    dimensions = params["dimensions"]
    from lineaxpr._linops.ellpack import BEllpack  # noqa: PLC0415
    if not dimensions:
        return op
    if op.n == 1 and dimensions == (0,):
        val = jnp.asarray(op.value).reshape(1)
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=(np.asarray([0]),), values=val,
            out_size=1, in_size=1,
        )
    raise NotImplementedError(f"squeeze on diag with dims {dimensions}")


@squeeze_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    dimensions = params["dimensions"]
    from lineaxpr._linops.ellpack import BEllpack  # noqa: PLC0415
    if not dimensions:
        return op
    if op.n == 1 and dimensions == (0,):
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=(np.asarray([0]),), values=op.values,
            out_size=1, in_size=1,
        )
    raise NotImplementedError(f"squeeze on diag with dims {dimensions}")


@rev_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    return op  # constant under axis-reversal


@rev_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    dimensions = params["dimensions"]
    if dimensions == (0,):
        return Diagonal(op.values[::-1])
    return op


@reshape_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    new_sizes = tuple(int(s) for s in params["new_sizes"])
    from lineaxpr._linops.ellpack import BEllpack  # noqa: PLC0415
    if len(new_sizes) >= 2 and int(np.prod(new_sizes)) == op.n:
        batch_shape = new_sizes[:-1]
        nrows = new_sizes[-1]
        flat_idx = np.arange(op.n).reshape(new_sizes)
        values = jnp.broadcast_to(jnp.asarray(op.value), new_sizes)
        return BEllpack(
            start_row=0, end_row=nrows,
            in_cols=(flat_idx,), values=values,
            out_size=nrows, in_size=op.n,
            batch_shape=batch_shape,
        )
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    from jax import lax  # noqa: PLC0415
    dense = _to_dense(op, n)
    return lax.reshape(dense, tuple(new_sizes) + (n,))


@reshape_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    new_sizes = tuple(int(s) for s in params["new_sizes"])
    from lineaxpr._linops.ellpack import BEllpack  # noqa: PLC0415
    if len(new_sizes) >= 2 and int(np.prod(new_sizes)) == op.n:
        batch_shape = new_sizes[:-1]
        nrows = new_sizes[-1]
        flat_idx = np.arange(op.n).reshape(new_sizes)
        values = op.values.reshape(new_sizes)
        return BEllpack(
            start_row=0, end_row=nrows,
            in_cols=(flat_idx,), values=values,
            out_size=nrows, in_size=op.n,
            batch_shape=batch_shape,
        )
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    from jax import lax  # noqa: PLC0415
    dense = _to_dense(op, n)
    return lax.reshape(dense, tuple(new_sizes) + (n,))


@broadcast_in_dim_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    shape = params["shape"]
    broadcast_dimensions = params["broadcast_dimensions"]
    from lineaxpr._linops.ellpack import BEllpack  # noqa: PLC0415
    # Diagonal (aval `(n,)`) broadcast to `(*pre, n, *post)` where all
    # non-n axes are size-1.
    if (len(broadcast_dimensions) == 1
            and shape[broadcast_dimensions[0]] == op.n
            and all(s == 1 for i, s in enumerate(shape)
                    if i != broadcast_dimensions[0])):
        bcast_axis = broadcast_dimensions[0]
        leading_singletons = tuple(shape[:bcast_axis])
        trailing_singletons = tuple(shape[bcast_axis + 1:])
        v = jnp.broadcast_to(jnp.asarray(op.value), (op.n,))
        if not trailing_singletons:
            cols = (np.arange(op.n),)
            new_values = jnp.broadcast_to(
                v.reshape((1,) * len(leading_singletons) + (op.n,)),
                leading_singletons + (op.n,),
            )
            return BEllpack(
                start_row=0, end_row=op.n,
                in_cols=cols, values=new_values,
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
            in_cols=(cols_full[..., None],), values=new_values,
            out_size=1, in_size=op.n,
            batch_shape=new_batch,
        )
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    from jax import lax  # noqa: PLC0415
    dense = _to_dense(op, n)
    expected_ndim = len(broadcast_dimensions) + 1
    while dense.ndim > expected_ndim and dense.shape[0] == 1:
        dense = dense[0]
    out_dims = tuple(broadcast_dimensions) + (len(shape),)
    return lax.broadcast_in_dim(dense, tuple(shape) + (n,), out_dims)


@broadcast_in_dim_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    shape = params["shape"]
    broadcast_dimensions = params["broadcast_dimensions"]
    from lineaxpr._linops.ellpack import BEllpack  # noqa: PLC0415
    if (len(broadcast_dimensions) == 1
            and shape[broadcast_dimensions[0]] == op.n
            and all(s == 1 for i, s in enumerate(shape)
                    if i != broadcast_dimensions[0])):
        bcast_axis = broadcast_dimensions[0]
        leading_singletons = tuple(shape[:bcast_axis])
        trailing_singletons = tuple(shape[bcast_axis + 1:])
        v = op.values
        if not trailing_singletons:
            cols = (np.arange(op.n),)
            new_values = jnp.broadcast_to(
                v.reshape((1,) * len(leading_singletons) + (op.n,)),
                leading_singletons + (op.n,),
            )
            return BEllpack(
                start_row=0, end_row=op.n,
                in_cols=cols, values=new_values,
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
            in_cols=(cols_full[..., None],), values=new_values,
            out_size=1, in_size=op.n,
            batch_shape=new_batch,
        )
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    from jax import lax  # noqa: PLC0415
    dense = _to_dense(op, n)
    expected_ndim = len(broadcast_dimensions) + 1
    while dense.ndim > expected_ndim and dense.shape[0] == 1:
        dense = dense[0]
    out_dims = tuple(broadcast_dimensions) + (len(shape),)
    return lax.broadcast_in_dim(dense, tuple(shape) + (n,), out_dims)


@reduce_sum_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    """CD: dense fallback (see comment in unary.py about XLA fusion)."""
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    axes = params["axes"]
    dense = _to_dense(op, n)
    return jnp.sum(dense, axis=tuple(axes))


@reduce_sum_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    """Diagonal: dense fallback (same XLA fusion reason as CD)."""
    from lineaxpr._linops import _to_dense  # noqa: PLC0415
    axes = params["axes"]
    dense = _to_dense(op, n)
    return jnp.sum(dense, axis=tuple(axes))


@gather_op.register(ConstantDiagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, start_indices, **params):
    dnums = params["dimension_numbers"]
    from lineaxpr._linops.ellpack import BEllpack  # noqa: PLC0415
    point_gather_kept = (
        dnums.offset_dims == (1,)
        and dnums.collapsed_slice_dims == ()
        and dnums.start_index_map == (0,)
        and params["slice_sizes"] == (1,)
    )
    row_idx = start_indices[..., 0]
    batch_shape = tuple(row_idx.shape[:-1])
    N = row_idx.shape[-1]
    vals = jnp.broadcast_to(jnp.asarray(op.value), batch_shape + (N,))
    if point_gather_kept:
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=(row_idx[..., None],),
            values=vals[..., None],
            out_size=1, in_size=op.n,
            batch_shape=batch_shape + (N,),
        )
    return BEllpack(
        start_row=0, end_row=N,
        in_cols=(row_idx,),
        values=vals,
        out_size=N, in_size=op.n,
        batch_shape=batch_shape,
    )


@gather_op.register(Diagonal) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, start_indices, **params):
    dnums = params["dimension_numbers"]
    from lineaxpr._linops.ellpack import BEllpack  # noqa: PLC0415
    point_gather_kept = (
        dnums.offset_dims == (1,)
        and dnums.collapsed_slice_dims == ()
        and dnums.start_index_map == (0,)
        and params["slice_sizes"] == (1,)
    )
    row_idx = start_indices[..., 0]
    batch_shape = tuple(row_idx.shape[:-1])
    N = row_idx.shape[-1]
    vals = jnp.take(op.values, row_idx)
    if point_gather_kept:
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=(row_idx[..., None],),
            values=vals[..., None],
            out_size=1, in_size=op.n,
            batch_shape=batch_shape + (N,),
        )
    return BEllpack(
        start_row=0, end_row=N,
        in_cols=(row_idx,),
        values=vals,
        out_size=N, in_size=op.n,
        batch_shape=batch_shape,
    )
