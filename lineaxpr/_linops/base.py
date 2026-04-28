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
bases with dense fallbacks; type-specific implementations are registered in
diagonal.py, ellpack.py, ellpack_transforms.py, and ellpack_indexing.py.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Protocol, runtime_checkable

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
    def dtype(self): ...

    def todense(self) -> jnp.ndarray: ...

    def transpose(self, permutation): ...


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
# Unary structural ops — singledispatch bases (dense fallback).
# Type-specific registrations live in diagonal.py / ellpack.py /
# ellpack_transforms.py / ellpack_indexing.py.
# ---------------------------------------------------------------------------

@singledispatch
def identity_op(op, *, n, **params):
    """Pass-through: return op unchanged (used by convert_element_type, copy)."""
    return op


@singledispatch
def squeeze_op(op, *, n, **params):
    """Squeeze output axes. Dense fallback."""
    dimensions = params["dimensions"]
    if isinstance(op, sparse.BCOO):
        return lax.squeeze(op.todense(), dimensions)
    return lax.squeeze(op, dimensions)


@singledispatch
def rev_op(op, *, n, **params):
    """Reverse output axes. Dense fallback."""
    dimensions = params["dimensions"]
    if isinstance(op, sparse.BCOO):
        return lax.rev(op.todense(), dimensions)
    return lax.rev(op, dimensions)


@singledispatch
def slice_op(op, *, n, **params):
    """Slice output axes. Dense fallback."""
    starts = tuple(int(s) for s in params["start_indices"])
    limits = tuple(int(l) for l in params["limit_indices"])
    strides_p = params.get("strides")
    strides = tuple(int(s) for s in strides_p) if strides_p else (1,) * len(starts)
    dense = op.todense() if isinstance(op, LinOpProtocol) else op
    s_full = starts + (0,)
    l_full = limits + (n,)
    str_full = strides + (1,)
    return lax.slice(dense, s_full, l_full, str_full)


@singledispatch
def pad_op(op, *, n, padding_value, **params):
    """Pad output axes. Dense fallback."""
    config = params["padding_config"]
    dense = op.todense() if isinstance(op, LinOpProtocol) else op
    full_config = tuple((int(b), int(a), int(i)) for (b, a, i) in config) + ((0, 0, 0),)
    return lax.pad(dense, jnp.asarray(0.0, dtype=dense.dtype), full_config)


@singledispatch
def cumsum_op(op, *, n, **params):
    """Cumulative sum. Dense fallback."""
    axis = params["axis"]
    reverse = params.get("reverse", False)
    dense = op.todense() if isinstance(op, LinOpProtocol) else op
    return lax.cumsum(dense, axis=axis, reverse=reverse)




@singledispatch
def reshape_op(op, *, n, **params):
    """Reshape output axes. Dense fallback."""
    new_sizes = tuple(int(s) for s in params["new_sizes"])
    dense = op.todense() if isinstance(op, LinOpProtocol) else op
    return lax.reshape(dense, tuple(new_sizes) + (n,))


@singledispatch
def broadcast_in_dim_op(op, *, n, **params):
    """Broadcast-in-dim. Dense fallback."""
    shape = params["shape"]
    broadcast_dimensions = params["broadcast_dimensions"]
    dense = op.todense() if isinstance(op, LinOpProtocol) else op
    expected_ndim = len(broadcast_dimensions) + 1
    while dense.ndim > expected_ndim and dense.shape[0] == 1:
        dense = dense[0]
    out_dims = tuple(broadcast_dimensions) + (len(shape),)
    return lax.broadcast_in_dim(dense, tuple(shape) + (n,), out_dims)


@singledispatch
def reduce_sum_op(op, *, n, **params):
    """Reduce sum. Dense fallback."""
    axes = params["axes"]
    dense = op.todense() if isinstance(op, LinOpProtocol) else op
    return jnp.sum(dense, axis=tuple(axes))


@singledispatch
def gather_op(op, *, n, start_indices, **params):
    """Gather rows. Dense fallback."""
    dnums = params["dimension_numbers"]
    point_gather_kept = (
        dnums.offset_dims == (1,)
        and dnums.collapsed_slice_dims == ()
        and dnums.start_index_map == (0,)
        and params["slice_sizes"] == (1,)
    )
    row_idx = start_indices[..., 0]
    dense = op.todense() if isinstance(op, LinOpProtocol) else op
    if point_gather_kept:
        return dense[row_idx][..., None, :]
    return dense[row_idx]


@singledispatch
def scatter_add_op(updates, *, n, operand, scatter_indices, **params):
    """Scatter-add updates into rows. Dense fallback."""
    out_idx = scatter_indices[..., 0]
    out_idx_flat = out_idx.reshape(-1)
    out_size = operand.shape[0]
    updates_dense = updates.todense() if isinstance(updates, LinOpProtocol) else updates
    flat_updates = updates_dense.reshape(-1, n)
    return (jnp.zeros((out_size, n), flat_updates.dtype)
            .at[out_idx_flat].add(flat_updates))


@singledispatch
def split_op(op, *, n, **params):
    """Split output axis. Dense fallback — returns a list of arrays."""
    sizes = params["sizes"]
    axis = params["axis"]
    dense = op.todense() if isinstance(op, LinOpProtocol) else op
    out = []
    start = 0
    for sz in sizes:
        slc = [slice(None)] * dense.ndim
        slc[axis] = slice(int(start), int(start) + int(sz))
        out.append(dense[tuple(slc)])
        start += int(sz)
    return out
