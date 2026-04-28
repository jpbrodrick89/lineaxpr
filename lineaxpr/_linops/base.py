"""Singledispatch op functions and LinOpProtocol.

`negate`, `scale_scalar`, `scale_per_out_row` are the operations that our
LinOp classes share as methods but BCOO does not. Defining them as
singledispatch functions gives a single call-site regardless of format and
lets bcoo_extend.py register non-densifying BCOO implementations.

LinOpProtocol documents the required interface for new LinOp forms and lets
pyrefly check that call sites annotated `op: LinOpProtocol` are valid.
BCOO satisfies todense() and to_bcoo() but not negate/scale_* — it is
handled via singledispatch registrations in bcoo_extend.py, not the protocol.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Protocol

import jax.numpy as jnp
from jax import core


class LinOpProtocol(Protocol):
    """Interface every native LinOp class must satisfy."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    def primal_aval(self) -> core.ShapedArray: ...

    def todense(self) -> jnp.ndarray: ...

    def to_bcoo(self): ...

    def negate(self): ...

    def scale_scalar(self, s): ...

    def scale_per_out_row(self, v): ...


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
