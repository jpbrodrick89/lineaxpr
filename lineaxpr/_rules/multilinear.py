"""Multi-operand structural rules: sub, dot_general, div."""

from __future__ import annotations

import string

import jax.numpy as jnp
import numpy as np
from jax import lax

from .._linops import (
    BEllpack,
    ConstantDiagonal,
    _to_dense,
    _traced_shape,
)
from .mul import _mul_rule
from .._linops.base import LinOpProtocol as _LinOpProtocol
from .._linops.base import negate as _negate_dispatch


def _neg_rule(invals, traced, n, **params):
    """Local neg rule copy (avoids circular import with registry)."""
    del params, n
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    if isinstance(op, _LinOpProtocol):
        return _negate_dispatch(op)
    return -op


def _bcast(arr, shape):
    return (np if isinstance(arr, np.ndarray) else jnp).broadcast_to(arr, shape)


def _resolve_full(c, nrows, batch_shape):
    """Resolve a ColArr (slice | 1D | N-D) to shape `(*batch, nrows)`."""
    if isinstance(c, slice):
        c = c
    if c.ndim == 1:
        return _bcast(c, batch_shape + (nrows,))
    return c


def _be_dot_closure_matrix(be: BEllpack, M, c_be: int, c_M: int,
                           traced_is_first: bool):
    """Structural `BEllpack ⊗ closure_matrix` contract. Returns `None`
    when dense would be no larger (gate `k_old * A >= in_size`), or
    when the contract is on an earlier-batch axis (unsupported — would
    require reshuffling every per-band col tensor).

    Band enumeration (β, a) flattens to `β * A + a`; the einsum output
    order `(remaining, J, [K], contract)` is chosen so a trailing
    reshape produces that layout directly.
    """
    n_batch = be.n_batch
    aval_shape = (*be.batch_shape, be.out_size)
    k_old, in_size = be.k, be.in_size
    if not (n_batch - 1 <= c_be <= n_batch):
        return None
    A = aval_shape[c_be]
    if A != M.shape[c_M] or k_old * A >= in_size:
        return None

    B = M.shape[1 - c_M]
    M_AB = M if c_M == 0 else M.T
    new_aval = aval_shape[:c_be] + aval_shape[c_be + 1:] + (B,)
    new_batch, new_out = new_aval[:-1], new_aval[-1]

    new_in_cols = tuple(
        _bcast(c_full.take(a, axis=c_be)[..., None], new_aval)
        for c_full in (_resolve_full(c, be.nrows, be.batch_shape)
                       for c in be.in_cols)
        for a in range(A)
    )

    # einsum: one letter per aval axis, K for optional k, J for M's
    # free axis. Trailing (K, contract) reshape gives the β*A+a layout.
    letters = string.ascii_lowercase[:len(aval_shape)]
    assert len(letters) == len(aval_shape), "aval rank exceeds letter pool"
    ctr = letters[c_be]
    remaining = letters[:c_be] + letters[c_be + 1:]
    k_let = "K" if k_old > 1 else ""
    eq = f"{letters}{k_let},{ctr}J->{remaining}J{k_let}{ctr}"
    new_vals = jnp.einsum(eq, be.values, M_AB)
    if k_old > 1:
        new_vals = new_vals.reshape(new_aval + (k_old * A,))

    out_be = BEllpack(
        start_row=0, end_row=new_out,
        in_cols=new_in_cols, values=new_vals,
        out_size=new_out, in_size=in_size, batch_shape=new_batch,
    )
    if not traced_is_first:
        # dot_general(closure, BE) aval is (*remaining_M, *remaining_BE);
        # BE's out axis is structurally last so we permute batch↔out.
        # Cheap: reorders the in_cols tuple + one values transpose.
        out_be = out_be.transpose((n_batch,) + tuple(range(n_batch)))
    return out_be


def _dot_general_rule(invals, traced, n, **params):
    x, y = invals
    tx, ty = traced
    (contract, batch) = params["dimension_numbers"]
    (cx, cy) = contract
    if batch != ((), ()):
        raise NotImplementedError("dot_general with batch dims not yet handled")

    if tx and ty:
        raise NotImplementedError("dot_general of two traced operands")
    if tx:
        traced_op, c_tr, M, c_M = x, list(cx), y, list(cy)
    else:
        traced_op, c_tr, M, c_M = y, list(cy), x, list(cx)
    traced_is_first = tx
    traced_shape = _traced_shape(traced_op)

    if len(c_tr) == 0 and len(c_M) == 0 and M.shape == ():
        if isinstance(traced_op, ConstantDiagonal):
            return ConstantDiagonal(traced_op.n, M * traced_op.value)
        return M * traced_op
    if len(c_tr) == 0 and len(c_M) == 0:
        # Outer product. BE's trailing `n` axis stays last.
        dense = _to_dense(traced_op, n)
        if traced_is_first:
            # (*t, n) × (*m,) → (*t, *m, n)
            d = dense.reshape(traced_shape + (1,) * M.ndim + dense.shape[-1:])
            return d * M[..., None]
        # (*m,) × (*t, n) → (*m, *t, n)
        return M.reshape(M.shape + (1,) * (len(traced_shape) + 1)) * dense

    if isinstance(traced_op, ConstantDiagonal):
        remaining = [a for a in range(M.ndim) if a not in c_M]
        tensor = lax.transpose(M, remaining + c_M)
        return traced_op.value * tensor

    # Structural BEllpack × closure-matrix path (see
    # `_be_dot_closure_matrix` for the gate: `k_new >= in_size`
    # falls through to dense).
    if (isinstance(traced_op, BEllpack)
            and M.ndim == 2
            and len(c_tr) == 1 and len(c_M) == 1
            and traced_op.start_row == 0
            and traced_op.end_row == traced_op.out_size):
        be_result = _be_dot_closure_matrix(
            traced_op, M, c_tr[0], c_M[0], traced_is_first,
        )
        if be_result is not None:
            return be_result

    dense = _to_dense(traced_op, n)
    if traced_is_first:
        out = lax.dot_general(
            dense, M, ((tuple(c_tr), tuple(c_M)), ((), ()))
        )
        # dense's trailing `n` axis is never contracted; dot_general's
        # output places it at `len(traced_shape) - len(c_tr)`. Move to end.
        return jnp.moveaxis(out, len(traced_shape) - len(c_tr), -1)
    return lax.dot_general(M, dense, ((tuple(c_M), tuple(c_tr)), ((), ())))


def _sub_rule(invals, traced, n, **params):
    """a - b = a + (-b). Reuse add via negating the second operand if traced."""
    from .add import _add_rule
    a, b = invals
    ta, tb = traced
    if not ta and not tb:
        return None
    if not tb:
        # a is traced, b is closure. a - b is still linear with same A as a.
        return a if ta else None
    if not ta:
        # -b only. Negate via _neg_rule equivalent.
        return _neg_rule([b], [True], n)
    # both traced
    neg_b = _neg_rule([b], [True], n)
    return _add_rule([a, neg_b], [True, True], n)


def _div_rule(invals, traced, n, **params):
    a, b = invals
    ta, tb = traced
    if not ta and not tb:
        return None
    if tb:
        raise NotImplementedError("div with traced denominator")
    # a / b where b is closure: equivalent to mul(a, 1/b).
    inv_b = jnp.reciprocal(jnp.asarray(b))
    return _mul_rule([a, inv_b], [True, False], n)
