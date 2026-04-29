"""Singledispatch op functions and LinOpProtocol.

`negate`, `scale_scalar`, `scale_per_out_row` are operations that our native
LinOp classes implement as methods but BCOO/CSR do not. They live here as
singledispatch functions so every format (LinOp, BCOO, future CSR) shares
one call-site; bcoo_extend.py registers the BCOO implementations.

LinOpProtocol is the minimal structural interface that BCOO, BCSR, and our
own LinOp classes all satisfy by duck-typing. It deliberately excludes
negate/scale_* (handled by singledispatch, not the protocol) so that external
sparse formats can be passed as LinOps without any adapter code.
Note: BCOO.to_bcoo() does not exist (BCOO is already BCOO); call sites that
need a BCOO should use `op.to_bcoo() if hasattr(op, 'to_bcoo') else op`.

Unary structural ops (squeeze_op, rev_op, etc.) live here as singledispatch
bases. Every format that can appear in the walk has an explicit registration
in diagonal.py, ellpack.py, ellpack_transforms.py, ellpack_indexing.py, or
bcoo_extend.py. The bases are plain-array fallbacks with no isinstance checks.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Protocol, Sequence, runtime_checkable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse


@runtime_checkable
class LinOpProtocol(Protocol):
    """Minimal interface shared by all walk-compatible formats.

    Satisfied by duck-typing: jax.experimental.sparse.BCOO, BCSR, plain
    ndarrays, and our own ConstantDiagonal / Diagonal / BEllpack all have
    shape and dtype. todense() is required for format conversion at the
    materialize boundary.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...

    def todense(self) -> jnp.ndarray: ...

    def transpose(self, axes: Sequence[int] | None = None) -> Any: ...


@singledispatch
def negate(op) -> Any:
    """Negate a LinOp or BCOO. Raises for unregistered types."""
    raise NotImplementedError(f"negate not implemented for {type(op)}")


@singledispatch
def scale_scalar(op, s) -> Any:
    """Multiply a LinOp or BCOO by a scalar. Raises for unregistered types."""
    raise NotImplementedError(f"scale_scalar not implemented for {type(op)}")


@singledispatch
def scale_per_out_row(op, v) -> Any:
    """Scale each output row of a LinOp or BCOO by the vector v."""
    raise NotImplementedError(
        f"scale_per_out_row not implemented for {type(op)}"
    )


# ---------------------------------------------------------------------------
# Unary structural ops — singledispatch bases.
# All LinOp and BCOO types have explicit registrations in the format files.
# These bases are plain-array fallbacks (for jax.Array results from dense
# rules flowing through the walk) — no isinstance checks needed here.
# ---------------------------------------------------------------------------

@singledispatch
def identity_op(op, *, n, **params) -> LinOpProtocol | sparse.BCOO | jax.Array:
    """Pass-through: return op unchanged (used by convert_element_type, copy)."""
    return op


@singledispatch
def squeeze_op(op, *, n, **params) -> LinOpProtocol | sparse.BCOO | jax.Array:
    """Squeeze output axes. Plain-array fallback."""
    return lax.squeeze(op, params["dimensions"])


@singledispatch
def rev_op(op, *, n, **params) -> LinOpProtocol | sparse.BCOO | jax.Array:
    """Reverse output axes. Plain-array fallback."""
    return lax.rev(op, params["dimensions"])


@singledispatch
def slice_op(op, *, n, **params) -> LinOpProtocol | sparse.BCOO | jax.Array:
    """Slice output axes. Plain-array fallback."""
    starts = tuple(int(s) for s in params["start_indices"])
    limits = tuple(int(l) for l in params["limit_indices"])
    strides_p = params.get("strides")
    strides = tuple(int(s) for s in strides_p) if strides_p else (1,) * len(starts)
    s_full = starts + (0,)
    l_full = limits + (n,)
    str_full = strides + (1,)
    return lax.slice(op, s_full, l_full, str_full)


@singledispatch
def pad_op(op, *, n, padding_value, **params) -> LinOpProtocol | sparse.BCOO | jax.Array:
    """Pad output axes. Plain-array fallback."""
    config = params["padding_config"]
    full_config = tuple((int(b), int(a), int(i)) for (b, a, i) in config) + ((0, 0, 0),)
    return lax.pad(op, jnp.asarray(0.0, dtype=op.dtype), full_config)


@singledispatch
def cumsum_op(op, *, n, **params) -> LinOpProtocol | sparse.BCOO | jax.Array:
    """Cumulative sum. Plain-array fallback."""
    return lax.cumsum(op, axis=params["axis"],
                      reverse=params.get("reverse", False))


@singledispatch
def reshape_op(op, *, n, **params) -> LinOpProtocol | sparse.BCOO | jax.Array:
    """Reshape output axes. Plain-array fallback."""
    new_sizes = tuple(int(s) for s in params["new_sizes"])
    return lax.reshape(op, tuple(new_sizes) + (n,))


@singledispatch
def broadcast_in_dim_op(op, *, n, **params) -> LinOpProtocol | sparse.BCOO | jax.Array:
    """Broadcast-in-dim. Plain-array fallback."""
    shape = params["shape"]
    broadcast_dimensions = params["broadcast_dimensions"]
    dense = op
    expected_ndim = len(broadcast_dimensions) + 1
    while dense.ndim > expected_ndim and dense.shape[0] == 1:
        dense = dense[0]
    out_dims = tuple(broadcast_dimensions) + (len(shape),)
    return lax.broadcast_in_dim(dense, tuple(shape) + (n,), out_dims)


@singledispatch
def reduce_sum_op(op, *, n, **params) -> LinOpProtocol | sparse.BCOO | jax.Array:
    """Reduce sum. Plain-array fallback."""
    return jnp.sum(op, axis=tuple(params["axes"]))


@singledispatch
def gather_op(op, *, n, start_indices, **params) -> LinOpProtocol | sparse.BCOO | jax.Array:
    """Gather rows. Plain-array fallback."""
    dnums = params["dimension_numbers"]
    point_gather_kept = (
        dnums.offset_dims == (1,)
        and dnums.collapsed_slice_dims == ()
        and dnums.start_index_map == (0,)
        and params["slice_sizes"] == (1,)
    )
    row_idx = start_indices[..., 0]
    if point_gather_kept:
        return op[row_idx][..., None, :]
    # 2D point-gather: select (row, col) pairs from a matrix.
    if (dnums.offset_dims == ()
            and dnums.collapsed_slice_dims == (0, 1)
            and dnums.start_index_map == (0, 1)):
        col_idx = start_indices[..., 1]
        return op[row_idx, col_idx]
    return op[row_idx]


@singledispatch
def scatter_add_op(updates, *, n, operand, scatter_indices,
                   **params) -> LinOpProtocol | sparse.BCOO | jax.Array:
    """Scatter-add updates into rows. Plain-array fallback."""
    out_idx = scatter_indices[..., 0]
    out_idx_flat = out_idx.reshape(-1)
    out_size = operand.shape[0]
    flat_updates = updates.reshape(-1, n)
    return (jnp.zeros((out_size, n), flat_updates.dtype)
            .at[out_idx_flat].add(flat_updates))


@singledispatch
def split_op(op, *, n, **params) -> list[LinOpProtocol | sparse.BCOO | jax.Array]:
    """Split output axis. Plain-array fallback — returns a list of arrays."""
    sizes = params["sizes"]
    axis = params["axis"]
    out = []
    start = 0
    for sz in sizes:
        slc = [slice(None)] * op.ndim
        slc[axis] = slice(int(start), int(start) + int(sz))
        out.append(op[tuple(slc)])
        start += int(sz)
    return out
