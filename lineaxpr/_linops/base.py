"""Singledispatch op functions and LinOpProtocol.

`scale_scalar`, `scale_per_out_row` are operations that our native LinOp
classes implement as methods but BCOO/CSR do not. They live here as
singledispatch functions so every format (LinOp, BCOO, future CSR)
shares one call-site; bcoo_extend.py registers the BCOO implementations.

`replace_slots(op, **changes)` is the structural primitive the
zero-preserving-linear-unary rule registry uses to push neg/copy/conj/etc.
through any LinOp form uniformly:
`replace_slots(op, data=prim.bind(op.data, **params))`.

LinOpProtocol is the minimal structural interface that BCOO, BCSR, and our
own LinOp classes all satisfy by duck-typing. It deliberately excludes
scale_* (handled by singledispatch, not the protocol) so that external
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

    @property
    def data(self) -> Any: ...

    def todense(self) -> jnp.ndarray: ...

    def transpose(self, axes: Sequence[int] | None = None) -> Any: ...


def replace_slots(obj, **changes) -> Any:
    """Generic `dataclasses.replace` analogue for `__slots__` classes.

    Constructs a new instance via `__new__`, copies every attribute the
    original holds (whether stored in `__slots__` walked across the MRO
    or in `__dict__`), then applies overrides from `changes`. Bypasses
    `__init__` — caller is responsible for ensuring overridden values
    are already in the canonical form the class expects (e.g. for
    BEllpack, `data` must already be normalised by `_normalize_data`).

    Handles BCOO uniformly with our own LinOps despite BCOO declaring
    `__slots__ = ()` and storing its fields in `__dict__`.

    Used to push zero-preserving linear unary primitives through any
    LinOp form: `replace_slots(op, data=prim.bind(op.data, **params))`.
    Mirrors `jax.experimental.sparse.transform`'s SparsifyEnv-mediated
    rebuild — same idea, different mechanism since we don't have an env.
    """
    cls = type(obj)
    new = cls.__new__(cls)  # pyrefly: ignore [no-matching-overload]
    if hasattr(obj, "__dict__"):
        new.__dict__.update(obj.__dict__)
    for c in cls.__mro__:
        for slot in getattr(c, "__slots__", ()):
            if hasattr(obj, slot):
                setattr(new, slot, getattr(obj, slot))
    for k, v in changes.items():
        setattr(new, k, v)
    return new


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

def _unimplemented(name):
    # Annotated return type prevents pyrefly from narrowing the inferred
    # singledispatch base to `Never`, which would then reject every
    # `@base.register(...)` registration that returns an array.
    def base(op, *args, **kwargs) -> "LinOpProtocol | sparse.BCOO | jax.Array":
        raise NotImplementedError(
            f"{name}: no registration for {type(op).__name__}. "
            f"Register the type in dense.py (jax.Array / DynamicJaxprTracer) "
            f"or in the corresponding LinOp module."
        )
    base.__name__ = name
    return base


# All explicit registrations live in: dense.py (jax.Array / DynamicJaxprTracer),
# diagonal.py (CD/Diagonal), ellpack.py + ellpack_transforms.py +
# ellpack_indexing.py (BEllpack), bcoo_extend.py (sparse.BCOO).
squeeze_op = singledispatch(_unimplemented("squeeze_op"))
rev_op = singledispatch(_unimplemented("rev_op"))
slice_op = singledispatch(_unimplemented("slice_op"))
pad_op = singledispatch(_unimplemented("pad_op"))
reshape_op = singledispatch(_unimplemented("reshape_op"))
broadcast_in_dim_op = singledispatch(_unimplemented("broadcast_in_dim_op"))
reduce_sum_op = singledispatch(_unimplemented("reduce_sum_op"))
gather_op = singledispatch(_unimplemented("gather_op"))
scatter_add_op = singledispatch(_unimplemented("scatter_add_op"))


@singledispatch
def split_op(op, *, n, **params) -> "list[LinOpProtocol | sparse.BCOO | jax.Array]":
    raise NotImplementedError(
        f"split_op: no registration for {type(op).__name__}. "
        f"Register the type in dense.py (jax.Array / DynamicJaxprTracer) "
        f"or in the corresponding LinOp module."
    )
