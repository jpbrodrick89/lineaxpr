"""Primitive rules for the sparsify walk."""

from .add import _add_rule
from .mul import _mul_rule
from .registry import (
    _identity_rule,
    _neg_rule,
    _pad_rule,
    _gather_rule,
    _scatter_add_rule,
    materialize_rules,
)
from .multilinear import _sub_rule, _dot_general_rule, _div_rule
from .control_flow import (
    _concatenate_rule,
    _split_rule,
    _cond_rule,
    _jit_rule,
    _squeeze_leading_ones,
    _select_n_rule,
)
from .._linops import (
    slice_op as _slice_rule,
    pad_op as _pad_rule_dispatch,
    squeeze_op as _squeeze_rule,
    rev_op as _rev_rule,
    reshape_op as _reshape_rule,
    broadcast_in_dim_op as _broadcast_in_dim_rule,
    reduce_sum_op as _reduce_sum_rule,
    cumsum_op as _cumsum_rule,

    gather_op as _gather_rule_dispatch,
    scatter_add_op as _scatter_add_rule_dispatch,
)
from .._linops.ellpack_transforms import _bellpack_row_sum

__all__ = [
    "materialize_rules",
    "_add_rule",
    "_mul_rule",
    "_identity_rule",
    "_neg_rule",
    "_pad_rule",
    "_gather_rule",
    "_scatter_add_rule",
    "_sub_rule",
    "_dot_general_rule",
    "_div_rule",
    "_concatenate_rule",
    "_split_rule",
    "_cond_rule",
    "_jit_rule",
    "_squeeze_leading_ones",
    "_select_n_rule",
    "_bellpack_row_sum", "_broadcast_in_dim_rule", "_cumsum_rule", "_gather_rule_dispatch", "_pad_rule_dispatch", "_reduce_sum_rule", "_reshape_rule", "_rev_rule", "_scatter_add_rule_dispatch", "_slice_rule", "_squeeze_rule",
]
