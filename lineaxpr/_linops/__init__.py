"""LinOp classes and singledispatch ops for the sparsify walk.

Internal structural forms live in the env during a walk and are converted
to BCOO or ndarray at the public API boundary (`materialize`).

### Adding a new LinOp form

1. Define the class in the appropriate module under `_linops/`. Give it the
   standard interface: `.shape`, `.dtype`, `.todense()`, `.to_bcoo()`,
   `.transpose(axes)`.
2. Register singledispatch implementations for any ops it supports.
3. Export the class from `lineaxpr/__init__.py`.
4. In `_rules/`, `_add_rule`'s kind-dispatch uses `v.shape` directly.
"""

from __future__ import annotations

from .base import (
    LinOpProtocol,
    broadcast_in_dim_op,
    gather_op,
    pad_op,
    replace_slots,
    reduce_sum_op,
    reshape_op,
    rev_op,
    scale_per_out_row,
    scale_scalar,
    scatter_add_op,
    slice_op,
    split_op,
    squeeze_op,
)
from .diagonal import ConstantDiagonal, Diagonal, Identity
from .ellpack import BEllpack, ColArr, canonicalize, _ellpack_to_bcoo_batched
from .bcoo_extend import _bcoo_concat
from . import dense as _dense  # noqa: F401 — registers jax.Array / DynamicJaxprTracer dispatchers
from . import bcoo_extend as _bcoo_extend  # noqa: F401 — registers BCOO dispatchers
from . import ellpack_transforms as _ellpack_transforms  # noqa: F401 — registers BE transform dispatchers
from . import ellpack_indexing as _ellpack_indexing  # noqa: F401 — registers BE indexing dispatchers

__all__ = [
    "LinOpProtocol",
    "replace_slots",
    "scale_scalar",
    "scale_per_out_row",
    "squeeze_op",
    "rev_op",
    "slice_op",
    "pad_op",
    "reshape_op",
    "broadcast_in_dim_op",
    "reduce_sum_op",
    "gather_op",
    "scatter_add_op",
    "split_op",
    "ConstantDiagonal",
    "Diagonal",
    "Identity",
    "BEllpack",
    "ColArr",
    "canonicalize",
    "_bcoo_concat",
    "_ellpack_to_bcoo_batched",
]
