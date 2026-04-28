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
need a BCOO should use the module-level `_to_bcoo(op, n)` helper instead of
the method directly.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Protocol

import jax.numpy as jnp


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
