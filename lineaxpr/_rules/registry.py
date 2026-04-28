"""Primitive rule registry — maps jax.lax primitives to rule functions."""

from __future__ import annotations

from typing import Callable

from jax import lax
from jax._src.lax import slicing as _slicing
from jax.extend import core

from .add import _add_rule, BELLPACK_DEDUP_LIMIT, BELLPACK_DEDUP_VECTORISED_MIN
from .mul import _mul_rule
from .unary import (
    _identity_rule,
    _neg_rule,
    _slice_rule,
    _pad_rule,
    _squeeze_rule,
    _rev_rule,
    _reshape_rule,
    _broadcast_in_dim_rule,
    _reduce_sum_rule,
    _cumsum_rule,
    _transpose_rule,
)
from .multilinear import _sub_rule, _dot_general_rule, _div_rule
from .control_flow import (
    _concatenate_rule,
    _split_rule,
    _cond_rule,
    _jit_rule,
    _select_n_rule,
)
from .indexing import _gather_rule, _scatter_add_rule

# Re-export constants so callers can reach them via materialize.BELLPACK_*
__all__ = [
    "materialize_rules",
    "BELLPACK_DEDUP_LIMIT",
    "BELLPACK_DEDUP_VECTORISED_MIN",
]

materialize_rules: dict[core.Primitive, Callable] = {}

materialize_rules[lax.mul_p] = _mul_rule
materialize_rules[lax.add_p] = _add_rule

try:
    from jax._src.ad_util import add_jaxvals_p
    materialize_rules[add_jaxvals_p] = _add_rule
except ImportError:
    pass

materialize_rules[lax.convert_element_type_p] = _identity_rule
materialize_rules[lax.copy_p] = _identity_rule
materialize_rules[lax.neg_p] = _neg_rule
materialize_rules[lax.sub_p] = _sub_rule
materialize_rules[lax.dot_general_p] = _dot_general_rule
materialize_rules[lax.slice_p] = _slice_rule
materialize_rules[lax.pad_p] = _pad_rule
materialize_rules[lax.squeeze_p] = _squeeze_rule
materialize_rules[lax.rev_p] = _rev_rule
materialize_rules[lax.reshape_p] = _reshape_rule
materialize_rules[lax.broadcast_in_dim_p] = _broadcast_in_dim_rule
materialize_rules[lax.reduce_sum_p] = _reduce_sum_rule
materialize_rules[lax.concatenate_p] = _concatenate_rule
materialize_rules[lax.split_p] = _split_rule

try:
    from jax._src.lax.control_flow.conditionals import cond_p
    materialize_rules[cond_p] = _cond_rule
except ImportError:
    pass

try:
    from jax._src.pjit import jit_p
    materialize_rules[jit_p] = _jit_rule
except ImportError:
    pass

materialize_rules[lax.select_n_p] = _select_n_rule
materialize_rules[lax.cumsum_p] = _cumsum_rule
materialize_rules[lax.div_p] = _div_rule
materialize_rules[lax.transpose_p] = _transpose_rule
materialize_rules[lax.gather_p] = _gather_rule
materialize_rules[_slicing.scatter_add_p] = _scatter_add_rule
