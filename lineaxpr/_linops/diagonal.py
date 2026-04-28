"""ConstantDiagonal and Diagonal LinOp classes."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import core
from jax.experimental import sparse

from .base import negate, scale_per_out_row, scale_scalar


class ConstantDiagonal:
    """Diagonal matrix with all entries equal to `value`."""

    __slots__ = ("n", "value")

    def __init__(self, n: int, value: Any = 1.0):
        self.n = n
        self.value = value

    @property
    def shape(self):
        return (self.n, self.n)

    def primal_aval(self):
        v = jnp.asarray(self.value)
        return core.ShapedArray((self.n,), v.dtype)

    def todense(self):
        if isinstance(self.value, float) and self.value == 1.0:
            return jnp.eye(self.n)
        return self.value * jnp.eye(self.n)

    def to_bcoo(self):
        return _diag_to_bcoo(self.n, jnp.full((self.n,), self.value))


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

    def primal_aval(self):
        return core.ShapedArray((self.n,), self.values.dtype)

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
