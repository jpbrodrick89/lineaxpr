"""LinOp classes + densification helpers.

Internal structural forms used by the sparsify walk. They live in the env
during a single walk and are converted to BCOO or ndarray at the public
API boundary.

Public API consumers should use `Identity(n, dtype=...)` as the seed for
`lineaxpr.sparsify`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import core
from jax.experimental import sparse


# -------------------------- structural forms --------------------------


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
        idx = jnp.arange(self.n)
        return jnp.zeros((self.n, self.n), self.values.dtype).at[idx, idx].set(self.values)

    def to_bcoo(self):
        return _diag_to_bcoo(self)

    def negate(self):
        return Diagonal(-self.values)

    def scale_scalar(self, s):
        return Diagonal(s * self.values)

    def scale_per_out_row(self, v):
        return Diagonal(self.values * jnp.asarray(v))


class Pivoted:
    """A linear operator with at most one nonzero per row.

    Represents the `(out_size, in_size)` matrix
        M[out_rows[i], in_cols[i]] = values[i]
        M[r, c] = 0 otherwise.

    Captures `slice(Identity)`, `gather(Identity)`, scaled versions, and
    pad/add chains thereof — common in sparse-banded problems.
    Avoids BCOO's `(nse, 2)` indices array by keeping rows + cols as 1D.
    """

    __slots__ = ("out_rows", "in_cols", "values", "out_size", "in_size")

    def __init__(self, out_rows, in_cols, values, out_size, in_size):
        self.out_rows = out_rows
        self.in_cols = in_cols
        self.values = values
        self.out_size = out_size
        self.in_size = in_size

    @property
    def shape(self):
        return (self.out_size, self.in_size)

    @property
    def n(self):
        return self.in_size

    @property
    def nse(self):
        return self.out_rows.shape[0]

    def primal_aval(self):
        return core.ShapedArray((self.in_size,), self.values.dtype)

    def todense(self):
        return (jnp.zeros((self.out_size, self.in_size), self.values.dtype)
                .at[self.out_rows, self.in_cols].add(self.values))

    def to_bcoo(self):
        return _pivoted_to_bcoo(self)

    def negate(self):
        return Pivoted(self.out_rows, self.in_cols, -self.values,
                       self.out_size, self.in_size)

    def scale_scalar(self, s):
        return Pivoted(self.out_rows, self.in_cols, s * self.values,
                       self.out_size, self.in_size)

    def scale_per_out_row(self, v):
        v_arr = jnp.asarray(v)
        # Fast path: scale length equals nse (true when out_rows == arange(k)).
        if v_arr.shape[0] == self.nse:
            new_values = self.values * v_arr
        else:
            new_values = self.values * jnp.take(v_arr, self.out_rows)
        return Pivoted(self.out_rows, self.in_cols, new_values,
                       self.out_size, self.in_size)

    def pad_rows(self, before: int, after: int):
        """Pad along the out_size axis. Negative before/after truncates."""
        out_size = self.out_size + before + after
        new_rows = self.out_rows + before
        if before >= 0 and after >= 0:
            return Pivoted(new_rows, self.in_cols, self.values,
                           out_size, self.in_size)
        is_np = isinstance(new_rows, np.ndarray)
        mask = (new_rows >= 0) & (new_rows < out_size)
        safe_rows = (np.where(mask, new_rows, 0) if is_np
                     else jnp.where(mask, new_rows, 0))
        val_mask = jnp.asarray(mask, dtype=self.values.dtype) if is_np else mask
        return Pivoted(safe_rows, self.in_cols, self.values * val_mask,
                       out_size, self.in_size)


# -------------------------- densification helpers --------------------------


def _to_dense(op, n: int) -> jnp.ndarray:
    if isinstance(op, ConstantDiagonal):
        if isinstance(op.value, float) and op.value == 1.0:
            return jnp.eye(n)
        return op.value * jnp.eye(n)
    if isinstance(op, Diagonal):
        m = op.values.shape[0]
        idx = jnp.arange(m)
        return jnp.zeros((m, m), op.values.dtype).at[idx, idx].set(op.values)
    if isinstance(op, Pivoted):
        return (jnp.zeros((op.out_size, op.in_size), op.values.dtype)
                .at[op.out_rows, op.in_cols].add(op.values))
    if isinstance(op, sparse.BCOO):
        return op.todense()
    return op


def _pivoted_to_bcoo(p: "Pivoted") -> sparse.BCOO:
    if isinstance(p.out_rows, np.ndarray) and isinstance(p.in_cols, np.ndarray):
        # Stack statically; avoid two asarray + one jnp.stack HLO ops.
        indices = np.stack([p.out_rows, p.in_cols], axis=1)
    else:
        indices = jnp.stack([jnp.asarray(p.out_rows), jnp.asarray(p.in_cols)], axis=1)
    return sparse.BCOO((p.values, indices), shape=p.shape)


def _concat(arrs):
    arrs = list(arrs)
    if all(isinstance(a, np.ndarray) for a in arrs):
        return np.concatenate(arrs)
    return jnp.concatenate(arrs)


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
    return sparse.BCOO((data, indices), shape=(d.n, d.n))


def _to_bcoo(op, n: int):
    """Convert any internal LinOp to BCOO (used at the bcoo_jacobian boundary)."""
    if isinstance(op, sparse.BCOO):
        return op
    if isinstance(op, (ConstantDiagonal, Diagonal)):
        return _diag_to_bcoo(op)
    if isinstance(op, Pivoted):
        return _pivoted_to_bcoo(op)
    return op  # plain ndarray — caller will keep dense


def _traced_shape(op) -> tuple:
    if isinstance(op, (ConstantDiagonal, Diagonal)):
        return (op.n,)
    if isinstance(op, Pivoted):
        return (op.out_size,)
    return tuple(op.shape[:-1])
