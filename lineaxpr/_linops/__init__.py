"""LinOp classes and densification helpers.

Internal structural forms used by the sparsify walk. They live in the env
during a single walk and are converted to BCOO or ndarray at the public
API boundary (`materialize`, `to_dense`, `to_bcoo`).

Public API consumers should use `Identity(n, dtype=...)` as the seed for
`lineaxpr.sparsify`.

### Adding a new LinOp form

To extend the format space:

1. Define the class in the appropriate module under `_linops/`. Give it the
   standard method set: `.shape`, `.n`, `.primal_aval()`, `.todense()`,
   `.to_bcoo()`, `.negate()`, `.scale_scalar(s)`, `.scale_per_out_row(v)`,
   and any form-specific ops (e.g. `BEllpack.pad_rows`).
2. Update `_to_dense(op, n)` and `_to_bcoo(op, n)` in this module.
3. Export it from `lineaxpr/__init__.py`.
4. In `materialize.py`, touch:
   - `_linop_matrix_shape(v)` — add an `isinstance` branch for shape.
   - `_add_rule`'s kind-dispatch — decide which combos with the new
     form stay structural vs promote to BCOO. Shared path is "any mix
     of {CD, D, BEllpack, <new>, BCOO} at matching shape → BCOO via
     `_to_bcoo` and concat", so no per-combo isinstance soup needed.
   - `_mul_rule` / `_neg_rule` just dispatch to the LinOp methods — no
     new branches unless the new form needs special-case BCOO fallbacks.
   - Rules that currently return BEllpack (e.g. `_slice_rule`,
     `_gather_rule`) may opportunistically return the new form when
     the pattern warrants it.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.experimental import sparse

from .base import (
    LinOpProtocol,
    broadcast_in_dim_op,
    cumsum_op,
    gather_op,
    identity_op,
    negate,
    pad_op,
    reduce_sum_op,
    reshape_op,
    rev_op,
    scale_per_out_row,
    scale_scalar,
    scatter_add_op,
    slice_op,
    squeeze_op,
    transpose_op,
)
from .diagonal import ConstantDiagonal, Diagonal, Identity, _diag_to_bcoo
from .ellpack import (
    BEllpack,
    _ellpack_to_bcoo,
    _ellpack_to_bcoo_batched,
    _normalize_values,
    _resolve_col,
    _slice_col,
    _transpose_col_batch,
    _transpose_col_full,
)
from . import bcoo_extend as _bcoo_extend  # noqa: F401 — registers BCOO dispatchers
from . import ellpack_transforms as _ellpack_transforms  # noqa: F401 — registers BE transform dispatchers
from . import ellpack_indexing as _ellpack_indexing  # noqa: F401 — registers BE indexing dispatchers

__all__ = [
    "LinOpProtocol",
    "negate",
    "scale_scalar",
    "scale_per_out_row",
    "identity_op",
    "squeeze_op",
    "rev_op",
    "slice_op",
    "pad_op",
    "cumsum_op",
    "transpose_op",
    "reshape_op",
    "broadcast_in_dim_op",
    "reduce_sum_op",
    "gather_op",
    "scatter_add_op",
    "ConstantDiagonal",
    "Diagonal",
    "Identity",
    "BEllpack",
    "_diag_to_bcoo",
    "_ellpack_to_bcoo",
    "_ellpack_to_bcoo_batched",
    "_normalize_values",
    "_resolve_col",
    "_slice_col",
    "_transpose_col_batch",
    "_transpose_col_full",
    "_to_dense",
    "_to_bcoo",
    "_traced_shape",
]


def _to_dense(op, n: int) -> jnp.ndarray:
    if isinstance(op, ConstantDiagonal):
        if isinstance(op.value, float) and op.value == 1.0:
            return jnp.eye(n)
        return op.value * jnp.eye(n)
    if isinstance(op, Diagonal):
        # Consistent with Diagonal.todense — scatter is context-robust
        # where the alternatives regress ARGTRIGLS.
        return op.todense()
    if isinstance(op, BEllpack):
        return op.todense()
    if isinstance(op, sparse.BCOO):
        return op.todense()
    return op


def _to_bcoo(op, n: int):
    """Convert any internal LinOp to BCOO (used at the `materialize`
    boundary when `format='bcoo'`, and internally by `_add_rule` to
    promote mixed-form operands to a common BCOO before concatenation)."""
    if isinstance(op, sparse.BCOO):
        return op
    if isinstance(op, (ConstantDiagonal, Diagonal, BEllpack)):
        return op.to_bcoo()
    return op  # plain ndarray — caller will keep dense


def _traced_shape(op) -> tuple:
    """Return the aval shape of the walk-variable this LinOp represents
    (i.e., the LinOp shape minus the trailing input-coordinate axis)."""
    if isinstance(op, (ConstantDiagonal, Diagonal)):
        return (op.n,)
    if isinstance(op, BEllpack):
        return (*op.batch_shape, op.out_size)
    return tuple(op.shape[:-1])
