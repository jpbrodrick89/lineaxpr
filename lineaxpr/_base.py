"""Shim — all content moved to lineaxpr/_linops/."""

from ._linops import *  # noqa: F401, F403
from ._linops import (  # noqa: F401
    ConstantDiagonal,
    Diagonal,
    Identity,
    BEllpack,
    _diag_to_bcoo,
    _ellpack_to_bcoo,
    _ellpack_to_bcoo_batched,
    _normalize_values,
    _resolve_col,
    _slice_col,
    _transpose_col_batch,
    _transpose_col_full,
    _to_dense,
    _to_bcoo,
    _traced_shape,
)
