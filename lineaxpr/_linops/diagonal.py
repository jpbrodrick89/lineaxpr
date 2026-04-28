"""ConstantDiagonal and Diagonal LinOp classes."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import core
from jax.experimental import sparse


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
        return _diag_to_bcoo(self)

    def negate(self):
        return ConstantDiagonal(self.n, -self.value)

    def scale_scalar(self, s):
        return ConstantDiagonal(self.n, s * self.value)

    def scale_per_out_row(self, v):
        # value * diag(v) = Diagonal(value * v)
        return Diagonal(self.value * jnp.asarray(v))


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
        return _diag_to_bcoo(self)

    def negate(self):
        return Diagonal(-self.values)

    def scale_scalar(self, s):
        return Diagonal(s * self.values)

    def scale_per_out_row(self, v):
        return Diagonal(self.values * jnp.asarray(v))


def _diag_to_bcoo(d, n=None) -> sparse.BCOO:
    """Convert a (Constant)Diagonal to BCOO."""
    idx = jnp.arange(d.n)
    indices = jnp.stack([idx, idx], axis=1)
    if isinstance(d, ConstantDiagonal):
        v = jnp.asarray(d.value)
        data = jnp.broadcast_to(v, (d.n,))
    elif isinstance(d, Diagonal):
        data = d.values
    else:
        raise TypeError(f"_diag_to_bcoo expected diagonal LinOp, got {type(d)}")
    return sparse.BCOO((data, indices), shape=(d.n, d.n),
                       indices_sorted=True, unique_indices=True)
