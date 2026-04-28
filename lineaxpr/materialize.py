"""Coloring-free Jacobian extraction for linear callables.

Public API (see `lineaxpr/__init__.py`):

* `jacfwd(f)(y)` / `bcoo_jacfwd(f)(y)` — forward-mode Jacobian.
* `jacrev(f)(y)` / `bcoo_jacrev(f)(y)` — reverse-mode Jacobian.
* `hessian(f)(y)` / `bcoo_hessian(f)(y)` — full Hessian.
* `materialize(linear_fn, primal, format='dense'|'bcoo')` — core helper,
  when you already have a linearized callable.
* `sparsify(linear_fn)(seed_linop)` — primitive transform returning a
  LinOp (before format conversion).

All of the above trace `linear_fn` to a jaxpr and walk its equations
with per-primitive rules that propagate structural per-var operators.
The LinOp classes (`ConstantDiagonal`, `Diagonal`, `BEllpack`; see
`_base.py`) let common patterns (scalar · I, vector-scaled I, sparse
banded blocks) avoid materialising intermediate identity matrices; they
are converted to BCOO or dense at the boundary.

## Known gap: non-finite closures in structural paths

Our structural rules assume `0 * x = 0` for any `x`. This is correct
when `x` is finite but wrong for `x ∈ {inf, nan}` (where `0 * inf = nan`).
When a mul/div/add structural rule emits a BEllpack/BCOO that skips
zero positions, it silently drops positions where the closure operand
has `inf`/`nan`. CUTEst objectives don't produce non-finite intermediate
values in practice, so this is a latent correctness gap rather than an
observed issue. A fully-general fix would require reading the closure
for non-finite entries (essentially densifying), losing the structural
optimisation — not worth it unless the gap bites.
"""

from __future__ import annotations

import string
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.lax import slicing as _slicing
from jax.experimental import sparse
from jax.extend import core

from ._base import (
    ConstantDiagonal,
    Diagonal,
    BEllpack,
    Identity,
    _resolve_col,
    _to_bcoo,
    _to_dense,
    _traced_shape,
)
from ._linops import LinOpProtocol, negate
from ._rules import _add_rule, _mul_rule
from ._rules.add import (
    _bcoo_concat,
    _bellpack_unbatch,
    _cols_equal,
    _densify_if_wider_than_dense,
)


# -------------------------- rule registry --------------------------


materialize_rules: dict[core.Primitive, Callable] = {}


# -------------------------- rules --------------------------


materialize_rules[lax.mul_p] = _mul_rule

materialize_rules[lax.add_p] = _add_rule
try:
    from jax._src.ad_util import add_jaxvals_p

    materialize_rules[add_jaxvals_p] = _add_rule
except ImportError:
    pass


def _identity_rule(invals, traced, n, **params):
    """For primitives that don't change value (convert_element_type, copy)."""
    del params
    (op,) = invals
    (t,) = traced
    return op if t else None


materialize_rules[lax.convert_element_type_p] = _identity_rule
materialize_rules[lax.copy_p] = _identity_rule


def _neg_rule(invals, traced, n, **params):
    del params, n
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    if isinstance(op, LinOpProtocol):
        return negate(op)
    return -op

materialize_rules[lax.neg_p] = _neg_rule

def _sub_rule(invals, traced, n, **params):
    """a - b = a + (-b). Reuse add via negating the second operand if traced."""
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

materialize_rules[lax.sub_p] = _sub_rule

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


def _bcast(arr, shape):
    return (np if isinstance(arr, np.ndarray) else jnp).broadcast_to(arr, shape)


def _resolve_full(c, nrows, batch_shape):
    """Resolve a ColArr (slice | 1D | N-D) to shape `(*batch, nrows)`."""
    if isinstance(c, slice):
        c = _resolve_col(c, nrows)
    if c.ndim == 1:
        return _bcast(c, batch_shape + (nrows,))
    return c


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

materialize_rules[lax.dot_general_p] = _dot_general_rule

def _slice_rule(invals, traced, n, **params):
    (operand,) = invals
    (to,) = traced
    if not to:
        return None
    starts = tuple(int(s) for s in params["start_indices"])
    limits = tuple(int(l) for l in params["limit_indices"])
    strides_p = params.get("strides")
    strides = tuple(int(s) for s in strides_p) if strides_p else (1,) * len(starts)

    # Structural fast path — unit-stride 1D (hot path, unchanged HLO).
    if (len(starts) == 1 and strides == (1,)
            and isinstance(operand, LinOpProtocol)):
        s, e = starts[0], limits[0]
        k = e - s
        if isinstance(operand, ConstantDiagonal):
            values_b = jnp.broadcast_to(jnp.asarray(operand.value), (k,))
            return BEllpack(
                start_row=0, end_row=k,
                in_cols=(np.arange(s, e),),
                values=values_b,
                out_size=k, in_size=operand.n,
            )
        if isinstance(operand, Diagonal):
            return BEllpack(
                start_row=0, end_row=k,
                in_cols=(np.arange(s, e),),
                values=operand.values[s:e],
                out_size=k, in_size=operand.n,
            )
        if isinstance(operand, BEllpack):
            return operand.pad_rows(-s, -(operand.out_size - e))
        if isinstance(operand, sparse.BCOO):
            # BCOO slice axis=0: drop out-of-range rows rather than mask to
            # zero. The mask-to-zero variant leaves entries with row indices
            # outside `[0, k)` in the output — downstream `_bcoo_scale_*`
            # rules index with those rows and emit NaN / wrong values. When
            # indices are a static np array (the common case for our BCOO
            # construction), filter at trace time; otherwise drop via
            # runtime gather. Mirrors the fix applied to `_split_rule` for
            # the same reason.
            indices_np = None
            try:
                indices_np = np.asarray(operand.indices)
            except (jax.errors.TracerArrayConversionError, TypeError):
                pass
            if isinstance(indices_np, np.ndarray):
                rows_np = indices_np[:, 0]
                keep = np.nonzero((rows_np >= s) & (rows_np < e))[0]
                new_indices = np.stack(
                    [rows_np[keep] - s, indices_np[keep, 1]], axis=1
                )
                new_data = jnp.take(operand.data, jnp.asarray(keep))
                return sparse.BCOO(
                    # pyrefly: ignore [bad-argument-type]
                    (new_data, new_indices), shape=(k, operand.shape[1]),
                )
            rows = operand.indices[:, 0]
            in_range = (rows >= s) & (rows < e)
            new_rows = jnp.where(in_range, rows - s, 0)
            new_data = jnp.where(in_range, operand.data,
                                 jnp.zeros((), operand.data.dtype))
            new_indices = jnp.stack([new_rows, operand.indices[:, 1]], axis=1)
            return sparse.BCOO((new_data, new_indices), shape=(k, operand.shape[1]))

    # Strided 1D slice on ConstantDiagonal/Diagonal — emit a BEllpack
    # whose in_cols carry the strided index pattern. Used by RAYBENDL's
    # `y[::2]` pattern. BEllpack/BCOO with stride > 1 fall through to
    # dense (pad_rows can't express strided row selection).
    if (len(starts) == 1
            and isinstance(operand, (ConstantDiagonal, Diagonal))):
        s, e = starts[0], limits[0]
        stride = strides[0]
        cols = np.arange(s, e, stride)
        k_out = len(cols)
        if isinstance(operand, ConstantDiagonal):
            values_b = jnp.broadcast_to(jnp.asarray(operand.value), (k_out,))
            return BEllpack(
                start_row=0, end_row=k_out,
                in_cols=(cols,),
                values=values_b,
                out_size=k_out, in_size=operand.n,
            )
        return BEllpack(
            start_row=0, end_row=k_out,
            in_cols=(cols,),
            values=operand.values[s:e:stride],
            out_size=k_out, in_size=operand.n,
        )

    # Structural path: n-D unit-stride slice on a batched BEllpack where
    # the slice axes cover `batch_shape + (out_size,)`. Slice the batch
    # dims on `values` and per-batch `in_cols` via basic indexing, then
    # slice the out_size axis via `pad_rows`. Triggers on problems that
    # reshape(Identity) to n-D then slice (e.g. MSQRT, SPARSINE-like,
    # DRCAV1LQ's 2D stencil).
    if (isinstance(operand, BEllpack) and operand.n_batch > 0
            and len(starts) == operand.n_batch + 1
            and all(st == 1 for st in strides)):
        batch_slicer = tuple(slice(int(s), int(e))
                             for s, e in zip(starts[:-1], limits[:-1]))
        out_start, out_limit = int(starts[-1]), int(limits[-1])
        # values shape (*batch, nrows) for k=1 or (*batch, nrows, k) for k>=2.
        tail = (slice(None),) * (operand.values.ndim - operand.n_batch)
        new_values = operand.values[batch_slicer + tail]
        new_in_cols = []
        for c in operand.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif hasattr(c, "ndim") and c.ndim > 1:
                new_in_cols.append(c[batch_slicer + (slice(None),)])
            else:
                new_in_cols.append(c)  # 1D shared cols
        new_batch = tuple(b.stop - b.start for b in batch_slicer)
        sliced = BEllpack(
            operand.start_row, operand.end_row,
            tuple(new_in_cols), new_values,
            operand.out_size, operand.in_size,
            batch_shape=new_batch,
        )
        return sliced.pad_rows(-out_start, -(operand.out_size - out_limit))

    # Fallback: densify and slice along output (non-input) axes; preserve the
    # trailing input-coordinate axis with start=0, limit=n, stride=1.
    dense = _to_dense(operand, n)
    s_full = starts + (0,)
    l_full = limits + (n,)
    str_full = strides + (1,)
    return lax.slice(dense, s_full, l_full, str_full)

materialize_rules[lax.slice_p] = _slice_rule

def _pad_rule(invals, traced, n, **params):
    operand, padding_value = invals
    to, tp = traced
    if tp:
        raise NotImplementedError("pad with traced padding_value")
    if not to:
        return None
    if hasattr(padding_value, "shape") and padding_value.shape != ():
        raise NotImplementedError("pad with non-scalar padding_value")
    config = params["padding_config"]
    before, after, interior = config[0] if len(config) >= 1 else (0, 0, 0)
    before, after = int(before), int(after)
    interior = int(interior)
    if isinstance(operand, BEllpack) and len(config) == 1 and interior == 0:
        return operand.pad_rows(before, after)
    if isinstance(operand, sparse.BCOO) and len(config) == 1 and interior == 0:
        out_size = operand.shape[0] + before + after
        new_rows = operand.indices[:, 0] + before
        new_indices = jnp.stack([new_rows, operand.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (operand.data, new_indices), shape=(out_size, operand.shape[1])
        )
    # Structural n-D zero-interior pad on a batched BEllpack where pad
    # axes cover `batch_shape + (out_size,)`. Inner (out_size) axis:
    # `pad_rows(before, after)` shifts start_row / end_row. Outer (batch)
    # axes: zero-pad `values` on the batch axes and extend `batch_shape`.
    # Per-batch `in_cols` get padded with `-1` sentinels at new batch
    # slots (values are 0 there so col doesn't matter for correctness;
    # BCOO conversion filters). Used by DRCAV1LQ/DRCAV2LQ to pad each
    # 13-point stencil window back to the full 2D grid.
    if (isinstance(operand, BEllpack) and operand.n_batch > 0
            and len(config) == operand.n_batch + 1
            and all(int(c[2]) == 0 for c in config)):
        batch_pads = tuple((int(c[0]), int(c[1])) for c in config[:-1])
        out_before, out_after = int(config[-1][0]), int(config[-1][1])
        new_batch_shape = tuple(
            b + s + a for (b, a), s in zip(batch_pads, operand.batch_shape)
        )
        tail_pad = ((0, 0),) * (operand.values.ndim - operand.n_batch)
        new_values = jnp.pad(operand.values, batch_pads + tail_pad)
        new_in_cols = []
        for c in operand.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif hasattr(c, "ndim") and c.ndim > 1:
                pad_cfg = batch_pads + ((0, 0),)
                if isinstance(c, np.ndarray):
                    # pyrefly: ignore [bad-argument-type]
                    new_in_cols.append(np.pad(c, pad_cfg, constant_values=-1))
                else:
                    # pyrefly: ignore [bad-argument-type]
                    new_in_cols.append(jnp.pad(c, pad_cfg, constant_values=-1))
            else:
                new_in_cols.append(c)  # 1D shared cols — broadcast OK
        padded_batch = BEllpack(
            operand.start_row, operand.end_row,
            tuple(new_in_cols), new_values,
            operand.out_size, operand.in_size,
            batch_shape=new_batch_shape,
        )
        return padded_batch.pad_rows(out_before, out_after)

    if len(config) == 1 and interior > 0 and isinstance(operand, (sparse.BCOO, BEllpack)):
        # Interior padding inserts `interior` zeros between each original
        # entry — the adjoint of a strided slice. Promote BEllpack to
        # BCOO, then `new_row = old_row * (interior + 1) + before`.
        if isinstance(operand, BEllpack):
            from ._base import _to_bcoo
            operand = _to_bcoo(operand, n)
        step = interior + 1
        old_size = operand.shape[0]
        out_size = old_size + before + after + interior * max(old_size - 1, 0)
        new_rows = operand.indices[:, 0] * step + before
        new_indices = jnp.stack([new_rows, operand.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (operand.data, new_indices), shape=(out_size, operand.shape[1])
        )
    # Dense fallback: pad along output axes (input axis untouched).
    dense = _to_dense(operand, n)
    full_config = tuple((int(b), int(a), int(i)) for (b, a, i) in config) + ((0, 0, 0),)
    return lax.pad(dense, jnp.asarray(0.0, dtype=dense.dtype), full_config)

materialize_rules[lax.pad_p] = _pad_rule

def _squeeze_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    dimensions = params["dimensions"]
    # For 1D structural forms, the var is shape (n,) so there's nothing to
    # squeeze. Only fail if some other rule produced a higher-dim form.
    if isinstance(op, (ConstantDiagonal, Diagonal)):
        if not dimensions:
            return op
        # n=1 special case: a Diagonal/ConstantDiagonal of size 1 is
        # semantically a scalar times 1×1 identity. Squeezing dim (0,)
        # collapses the single out row to a scalar-aval linear form;
        # return a BEllpack(out_size=1) row-vector whose single band
        # has col 0 and the scalar value. Matches the same form
        # produced by the `_squeeze_rule` BEllpack path below.
        if op.n == 1 and dimensions == (0,):
            val = (jnp.asarray(op.value).reshape(1)
                   if isinstance(op, ConstantDiagonal) else op.values)
            return BEllpack(
                start_row=0, end_row=1,
                in_cols=(np.asarray([0]),), values=val,
                out_size=1, in_size=1,
            )
        raise NotImplementedError(f"squeeze on diag with dims {dimensions}")
    if isinstance(op, BEllpack) and op.n_batch == 0 and dimensions == (0,) \
            and op.out_size == 1 and op.start_row == 0 and op.end_row == 1:
        # BEllpack row-vector squeezed along its row axis: the result has
        # aval () — a *linear form* (1×n row vector, the Jacobian of a
        # scalar-aval variable w.r.t. the n-dim input). Keep it as a
        # BEllpack row-vector (shape (1, in_size)) so downstream broadcast-
        # add in `_add_rule` can tile the sparse row cheaply — instead
        # of the old densify-to-(n,)-ndarray path which forced
        # subsequent linear_form + vector adds to materialise (n, n)
        # dense. Only valid when the row is sparse (few bands); dense
        # rows should go via BCOO, not a k-many-bands BEllpack.
        return op
    # Batched BEllpack with out_size=1 squeezing the out axis: flatten
    # batch axes into a new unbatched `out_size = prod(batch)`. Each
    # original batch's single output row becomes a row of the new
    # unbatched BEllpack. Attacks WOODS (`reshape → slice → squeeze` on
    # the `(1000, 4)` variable grid, 123× vs asdex).
    if (isinstance(op, BEllpack) and op.n_batch >= 1
            and op.out_size == 1
            and op.start_row == 0 and op.end_row == 1
            and dimensions == (op.n_batch,)):
        B = int(np.prod(op.batch_shape))
        # Values: k=1 is (*batch, 1) → (B,); k>=2 is (*batch, 1, k) → (B, k).
        if op.k == 1:
            new_values = op.values.reshape(B)
        else:
            new_values = op.values.reshape(B, op.k)
        new_in_cols = []
        ok = True
        for c in op.in_cols:
            if isinstance(c, slice):
                rs = np.arange(c.start or 0, c.stop or 1, c.step or 1)
                # nrows=1 so slice yields 1 col; tile to (B,).
                if len(rs) == 1:
                    new_in_cols.append(np.broadcast_to(rs, (B,)).copy())
                else:
                    ok = False; break
            elif isinstance(c, np.ndarray):
                if c.ndim == op.n_batch + 1:  # (*batch, 1) per-batch
                    new_in_cols.append(c.reshape(B))
                elif c.ndim == 1 and c.shape[0] == 1:  # shared (1,)
                    new_in_cols.append(np.broadcast_to(c, (B,)).copy())
                elif c.ndim == 1 and c.shape[0] == B:
                    new_in_cols.append(c)
                else:
                    ok = False; break
            else:  # traced cols
                ok = False; break
        if ok:
            return BEllpack(
                start_row=0, end_row=B,
                in_cols=tuple(new_in_cols), values=new_values,
                out_size=B, in_size=op.in_size,
            )
    if isinstance(op, (BEllpack, sparse.BCOO)):
        # Densify (sparse → (out_size, in_size)) then squeeze leading axes.
        return lax.squeeze(_to_dense(op, n), dimensions)
    # Dense: squeeze the specified axes (always output axes, never the last
    # input-coordinate axis).
    return lax.squeeze(op, dimensions)

materialize_rules[lax.squeeze_p] = _squeeze_rule

def _rev_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    dimensions = params["dimensions"]
    if isinstance(op, ConstantDiagonal):
        return op  # constant under axis-reversal
    if isinstance(op, Diagonal):
        # Reverse the diagonal values along axis 0.
        if dimensions == (0,):
            return Diagonal(op.values[::-1])
        return op
    # BEllpack unbatched, reverse out axis: flip values + per-band cols
    # along the row axis; remap `[start_row, end_row)` to mirror around
    # `out_size`. Metadata + one jnp.flip per dim — no densify.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and dimensions == (0,)):
        new_start = op.out_size - op.end_row
        new_end = op.out_size - op.start_row
        new_values = jnp.flip(op.values, axis=0)
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(_resolve_col(c, op.nrows)[::-1].copy())
            elif isinstance(c, np.ndarray):
                new_in_cols.append(c[::-1].copy())
            else:
                new_in_cols.append(jnp.flip(c, axis=0))
        return BEllpack(
            start_row=new_start, end_row=new_end,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=op.out_size, in_size=op.in_size,
        )
    # BCOO unbatched, reverse out axis: remap row indices to
    # `shape[0] - 1 - row`. One jnp op on indices column 0.
    if (isinstance(op, sparse.BCOO) and op.n_batch == 0
            and dimensions == (0,)):
        new_rows = (op.shape[0] - 1) - op.indices[:, 0]
        new_indices = jnp.stack([new_rows, op.indices[:, 1]], axis=1)
        return sparse.BCOO((op.data, new_indices), shape=op.shape)
    # Fallback.
    dense = _to_dense(op, n)
    return lax.rev(dense, dimensions)

materialize_rules[lax.rev_p] = _rev_rule

# TODO(structural): reshape/broadcast_in_dim/reduce_sum/cumsum/split/transpose
# all densify unconditionally. Structural alternatives exist (e.g. transpose on
# BCOO swaps index columns; reduce_sum on a sparse axis drops it). Deferred —
# see docs/RESEARCH_NOTES.md §10 "Densifying vs structure-preserving" audit.
def _reshape_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    new_sizes = tuple(int(s) for s in params["new_sizes"])
    # Structural path: splitting the output axis of a (Constant)Diagonal
    # into batch-shape + nrows encodes the reshape permutation into a
    # batched BEllpack with k=1. `in_cols[*batch_idx, r] =
    # flat_index(*batch_idx, r)` so that densifying recovers
    # `lax.reshape(diag.todense(), (*new_sizes, n))` bit-exactly. Storage
    # is O(n) — no (67, 67, 4489)-class dense intermediate. Triggered by
    # 2D-stencil problems (DRCAV1LQ etc.) that start with
    # `reshape(Identity, (sqrt_n, sqrt_n))`. Downstream rules that don't
    # yet support batched BEllpack will still densify at their own site;
    # this keeps the walk structural at least through reshape.
    if (isinstance(op, (ConstantDiagonal, Diagonal))
            and len(new_sizes) >= 2
            and int(np.prod(new_sizes)) == op.n):
        batch_shape = new_sizes[:-1]
        nrows = new_sizes[-1]
        flat_idx = np.arange(op.n).reshape(new_sizes)
        if isinstance(op, ConstantDiagonal):
            values = jnp.broadcast_to(jnp.asarray(op.value), new_sizes)
        else:
            values = op.values.reshape(new_sizes)
        return BEllpack(
            start_row=0, end_row=nrows,
            in_cols=(flat_idx,), values=values,
            out_size=nrows, in_size=op.n,
            batch_shape=batch_shape,
        )
    # Pass-through: unbatched BCOO whose shape already equals the
    # target `(*new_sizes, n)`. Rare post-0j (batched BEllpacks now
    # convert to batched BCOO, not unbatched-flat), but still fires
    # when an already-unbatched BCOO reaches a final aval-flatten
    # reshape that's structurally a no-op.
    if (isinstance(op, sparse.BCOO) and op.n_batch == 0
            and len(new_sizes) == 1
            and op.shape == (int(new_sizes[0]), op.shape[-1])):
        return op

    # Structural path: batched BEllpack → unbatched BEllpack when the
    # reshape fully flattens the leading (batch + out) axes into one
    # aval dimension. Stays in BE form so downstream ops (`mul`,
    # band-widen `add`, `pad`) can carry BE-specific fast paths.
    # Values reshape `(*batch, nrows) → (B*O,)` for k=1 or `(*batch,
    # nrows, k) → (B*O, k)` for k>=2. Per-band cols broadcast to
    # `(*batch, nrows)` if 1D, then reshape to `(B*O,)`. Final BCOO
    # conversion (if needed) happens at the public-API boundary via
    # the now-vectorized `_ellpack_to_bcoo`. Target must be rank 1
    # and the total equal `prod(batch) * out_size`; otherwise fall
    # through.
    if (isinstance(op, BEllpack) and op.n_batch >= 1
            and len(new_sizes) == 1
            and int(np.prod(op.batch_shape)) * op.out_size
                == int(new_sizes[0])):
        prod_b = int(np.prod(op.batch_shape))
        total = prod_b * op.out_size
        if op.k == 1:
            new_values = op.values.reshape(total)
        else:
            new_values = op.values.reshape(total, op.k)
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                rs = np.arange(c.start or 0, c.stop or op.nrows,
                               c.step or 1)
                c_full = np.broadcast_to(
                    rs, op.batch_shape + (op.nrows,)
                )
                new_in_cols.append(c_full.reshape(total))
            elif isinstance(c, np.ndarray):
                if c.ndim == 1:
                    c_full = np.broadcast_to(
                        c, op.batch_shape + (op.nrows,)
                    )
                    new_in_cols.append(c_full.reshape(total))
                else:
                    new_in_cols.append(c.reshape(total))
            else:
                ca = jnp.asarray(c)
                if ca.ndim == 1:
                    ca = jnp.broadcast_to(
                        ca, op.batch_shape + (op.nrows,)
                    )
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(ca.reshape(total))
        return BEllpack(
            start_row=0, end_row=total,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=total, in_size=op.in_size,
        )
    # Unflatten on unbatched BEllpack: target `(A, B)` from aval `(N,)`
    # where `N == op.out_size == A * B` and B > 1. Inverse of the
    # batch+out flatten emit above. Each flat row i becomes (batch_idx,
    # local_row) = (i // B, i % B); `values` and each band's cols
    # reshape directly from `(N,)` to `(A, B)` (or `(A, B, k)` for
    # values at k>=2). Closes LUKSAN11-15LS's `reshape → mul → reshape`
    # chain where the intermediate (198,) flat BE was previously
    # densified at the unflatten step.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and len(new_sizes) == 2
            and int(new_sizes[0]) * int(new_sizes[1]) == op.out_size
            and int(new_sizes[1]) > 1
            and op.start_row == 0 and op.end_row == op.out_size):
        A = int(new_sizes[0])
        B_out = int(new_sizes[1])
        new_batch = (A,)
        # Values: (N,) → (A, B) for k=1, (N, k) → (A, B, k) for k>=2.
        if op.k == 1:
            new_values = op.values.reshape(A, B_out)
        else:
            new_values = op.values.reshape(A, B_out, op.k)
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                # Slice covers [0, N); reshape to per-(batch, row) by
                # resolving then reshaping.
                rs = np.arange(c.start or 0, c.stop or op.nrows, c.step or 1)
                new_in_cols.append(rs.reshape(A, B_out))
            elif isinstance(c, np.ndarray):
                new_in_cols.append(c.reshape(A, B_out))
            else:
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(jnp.asarray(c).reshape(A, B_out))
        return BEllpack(
            start_row=0, end_row=B_out,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=B_out, in_size=op.in_size,
            batch_shape=new_batch,
        )
    # Leading-singleton reshape on unbatched BEllpack: target
    # `(1, ..., 1, N)` from aval `(N,)` where `N == op.out_size`.
    # Adds size-1 leading batch axes. Values reshape to add singleton
    # axes; cols stay 1D shared. Used by LUKSAN16LS reshape from
    # aval `(49,)` to `(1, 1, 49)`.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and len(new_sizes) >= 2
            and new_sizes[-1] == op.out_size
            and all(s == 1 for s in new_sizes[:-1])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(new_sizes[:-1])
        prefix = (1,) * len(new_batch)
        if op.k == 1:
            new_values = op.values.reshape(prefix + (op.nrows,))
        else:
            new_values = op.values.reshape(prefix + (op.nrows, op.k))
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif isinstance(c, np.ndarray):
                if c.ndim == 1:
                    # pyrefly: ignore [bad-argument-type]
                    new_in_cols.append(c)  # 1D shared cols
                else:
                    # pyrefly: ignore [bad-argument-type]
                    new_in_cols.append(c.reshape(prefix + c.shape))
            else:
                ca = jnp.asarray(c)
                if ca.ndim == 1:
                    # pyrefly: ignore [bad-argument-type]
                    new_in_cols.append(ca)
                else:
                    # pyrefly: ignore [bad-argument-type]
                    new_in_cols.append(ca.reshape(prefix + ca.shape))
        return BEllpack(
            start_row=op.start_row, end_row=op.end_row,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=op.out_size, in_size=op.in_size,
            batch_shape=new_batch,
        )
    # Singleton-axis-insert on unbatched BEllpack: target
    # `(N, 1, ..., 1)` from aval `(N,)` where `N == op.out_size`. The
    # original rows become separate batches and the trailing size-1
    # axes become the new out axis plus singleton batch axes. Mirrors
    # the Change 3 `bid` trailing-singleton path (commit 0123250);
    # reshape can produce the same aval shift via different primitives.
    # Triggered by NONMSQRT's `reshape(BE(out=N), (N, 1))` step that
    # previously densified after the second reduce_sum.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and len(new_sizes) >= 2
            and new_sizes[0] == op.out_size
            and all(s == 1 for s in new_sizes[1:])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(new_sizes[:-1])
        if op.k == 1:
            new_values = op.values.reshape(new_batch + (1,))
        else:
            new_values = op.values.reshape(new_batch + (1, op.k))
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif isinstance(c, np.ndarray):
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(c.reshape(new_batch + (1,) + c.shape[1:]))
            else:
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(jnp.asarray(c).reshape(
                    new_batch + (1,) + c.shape[1:]))
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=1, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Structural path: flatten a batched BCOO's leading (batch + out)
    # axes into a single flat out axis. Handles the final reshape in
    # DRCAV1LQ/2LQ (`(67, 67) → (4489,)` as aval, LinOp `(67, 67, n) →
    # (4489, n)`). Remaps `new_row = batch_flat * old_out + old_row`
    # with `batch_flat = ravel_multi_index(batch_idx, batch_shape)`,
    # cols unchanged. Only supports fully-flattening the leading dims
    # (target rank 1); partial flattens fall through to dense.
    if (isinstance(op, sparse.BCOO) and op.n_batch >= 1
            and len(new_sizes) == 1
            and int(np.prod(op.shape[:-1])) == int(new_sizes[0])):
        nb = op.n_batch
        old_out = op.shape[nb]  # out axis (after batch, before in)
        in_size = op.shape[-1]
        batch_total = int(np.prod(op.shape[:nb]))
        nse_per_batch = op.data.shape[nb]
        flat_data = op.data.reshape(batch_total, nse_per_batch)
        flat_indices = op.indices.reshape(batch_total, nse_per_batch, 2)
        # Flat batch index b contributes row offset b * old_out.
        offsets = jnp.arange(batch_total, dtype=flat_indices.dtype) * old_out
        new_rows = flat_indices[..., 0] + offsets[:, None]
        new_cols = flat_indices[..., 1]
        new_indices = jnp.stack(
            [new_rows.reshape(-1), new_cols.reshape(-1)], axis=1,
        )
        return sparse.BCOO(
            (flat_data.reshape(-1), new_indices),
            shape=(int(new_sizes[0]), in_size),
        )
    dense = _to_dense(op, n)
    # Reshape applies to output axes only; preserve the trailing input axis.
    return lax.reshape(dense, tuple(new_sizes) + (n,))

materialize_rules[lax.reshape_p] = _reshape_rule

def _broadcast_in_dim_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    shape = params["shape"]
    broadcast_dimensions = params["broadcast_dimensions"]
    # Structural fast path: adding new leading axes to an unbatched BEllpack
    # (the pattern emitted by e.g. `jnp.sum(a[perm_indices], axis=0)`'s
    # backwards-linearize). Produces a BEllpack with the new dims in
    # batch_shape — values broadcast-tiled, in_cols shared across batches.
    # Linear form (aval ()) broadcast to shape (1,): target matrix shape
    # is (1, n) — already what a BEllpack row-vector carries, so pass through.
    # For a (n,)-ndarray linear form, promote to BCOO(1, n) so the
    # subsequent `pad` stays structural (its BCOO path just shifts row
    # indices; the dense fallback would zero-fill an (out, n) block).
    # Triggered by the `reduce_sum → neg → broadcast_in_dim → pad`
    # chain in LIARWHD-class problems.
    if broadcast_dimensions == () and tuple(shape) == (1,):
        if (isinstance(op, BEllpack) and op.n_batch == 0
                and op.out_size == 1 and op.start_row == 0
                and op.end_row == 1):
            return op
        if isinstance(op, jax.Array) and op.ndim == 1 and op.shape[0] == n:
            zeros_row = jnp.zeros((n,), dtype=jnp.int32)
            cols = jnp.arange(n, dtype=jnp.int32)
            indices = jnp.stack([zeros_row, cols], axis=1)
            return sparse.BCOO((op, indices), shape=(1, n))
    # Tile a 1-row BEllpack (aval-() linear form) to a N-row vector
    # `(N,)` via empty broadcast_dimensions. Each output row carries the
    # same sparse linear form as the input. Values broadcast along the
    # new row axis; cols broadcast similarly (1D shape (1,) → (N,)).
    # ~22 hits across the sweep on small problems with `bid → linear
    # form → vector` chains.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and op.out_size == 1 and op.start_row == 0 and op.end_row == 1
            and broadcast_dimensions == () and len(shape) == 1):
        N = int(shape[0])
        new_values = jnp.broadcast_to(op.values, (N,) + op.values.shape[1:])
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                resolved = _resolve_col(c, 1)
                new_in_cols.append(np.broadcast_to(resolved, (N,)).copy())
            elif isinstance(c, np.ndarray):
                new_in_cols.append(np.broadcast_to(c, (N,)).copy())
            else:
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(jnp.broadcast_to(c, (N,)))
        return BEllpack(
            start_row=0, end_row=N,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=N, in_size=op.in_size,
        )
    # Diagonal (aval `(n,)`) broadcast to `(*pre, n, *post)` where
    # `pre`/`post` only contain size-1 axes — i.e. inserts size-1
    # batch / out dims around the existing aval axis. Promote to a
    # batched BEllpack with `out_size = 1` (post=trailing-singletons)
    # or `out_size = n` (post empty), batch_shape carrying any
    # leading singletons. Six sweep hits split between
    # `Diagonal → (n, 1)` and `Diagonal → (1, n)` patterns.
    if (isinstance(op, (Diagonal, ConstantDiagonal))
            and len(broadcast_dimensions) == 1
            and shape[broadcast_dimensions[0]] == op.n
            and all(s == 1 for i, s in enumerate(shape)
                    if i != broadcast_dimensions[0])):
        bcast_axis = broadcast_dimensions[0]
        leading_singletons = tuple(shape[:bcast_axis])
        trailing_singletons = tuple(shape[bcast_axis + 1:])
        # Materialise diagonal values once.
        if isinstance(op, ConstantDiagonal):
            v = jnp.broadcast_to(jnp.asarray(op.value), (op.n,))
        else:
            v = op.values
        # No trailing singletons: BE(batch=leading_singletons,
        # out=n, in=n, k=1, cols=arange(n)).
        if not trailing_singletons:
            cols = (np.arange(op.n),)
            # values shape: (*leading_singletons, n)
            new_values = jnp.broadcast_to(
                v.reshape((1,) * len(leading_singletons) + (op.n,)),
                leading_singletons + (op.n,),
            )
            return BEllpack(
                start_row=0, end_row=op.n,
                in_cols=cols, values=new_values,
                out_size=op.n, in_size=op.n,
                batch_shape=leading_singletons,
            )
        # Trailing singletons present: each batch index i has a
        # 1-row matrix with col i and value v[i]. Total batch shape
        # = leading_singletons + (n,) + trailing_singletons[:-1];
        # out_size = trailing_singletons[-1] (which must be 1, since
        # all trailings are 1). Simplest: batch=(leading + (n,) +
        # trailing[:-1]), out_size=1, k=1, per-batch cols.
        # Cols (per-batch) = arange(n) along the n-axis.
        cols_2d = np.arange(op.n).reshape(
            (1,) * len(leading_singletons) + (op.n,)
            + (1,) * (len(trailing_singletons) - 1)
        )
        new_batch = leading_singletons + (op.n,) + trailing_singletons[:-1]
        cols_full = np.broadcast_to(cols_2d, new_batch).copy()
        # values per batch — 1 entry. Shape: new_batch + (1,) for k=1.
        new_values = jnp.broadcast_to(
            v.reshape((1,) * len(leading_singletons) + (op.n,)
                       + (1,) * (len(trailing_singletons) - 1)),
            new_batch,
        ).reshape(new_batch + (1,))
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=(cols_full[..., None],), values=new_values,
            out_size=1, in_size=op.n,
            batch_shape=new_batch,
        )
    # Fallback normalisation: a BEllpack row-vector represents an aval-()
    # linear form. For other broadcast patterns the dense fallback
    # below expects the canonical (n,)-ndarray linear-form shape, so
    # squeeze the BEllpack row first.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and op.out_size == 1 and op.start_row == 0 and op.end_row == 1
            and broadcast_dimensions == ()):
        op = _to_dense(op, n)[0]
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and len(broadcast_dimensions) == 1
            and broadcast_dimensions[0] == len(shape) - 1
            and shape[-1] == op.out_size):
        new_batch = tuple(shape[:-1])
        new_values_shape = new_batch + op.values.shape
        new_values = jnp.broadcast_to(op.values, new_values_shape)
        return BEllpack(
            start_row=op.start_row, end_row=op.end_row,
            in_cols=op.in_cols, values=new_values,
            out_size=op.out_size, in_size=op.in_size,
            batch_shape=new_batch,
        )
    # Prepend leading batch axes to an already-batched BE.
    # `broadcast_dimensions` is a contiguous tail mapping the input's
    # (batch, out) axes to the output's (batch, out) axes, with new
    # axes prepended. Values get broadcast-tiled along the new leading
    # dims; N-D cols (if any) get the same broadcast along those axes.
    # Used by LUKSAN16LS's step 22: BE(batch=(4,), out=49) broadcast
    # to aval (3, 4, 49), keeping the chain structural past step 14.
    input_rank = op.n_batch + 1 if isinstance(op, BEllpack) else None
    if (isinstance(op, BEllpack)
            and input_rank is not None
            and len(broadcast_dimensions) == input_rank
            and len(shape) > input_rank
            and broadcast_dimensions == tuple(
                range(len(shape) - input_rank, len(shape)))
            and shape[-1] == op.out_size):
        prepend = tuple(shape[:len(shape) - input_rank])
        new_batch = prepend + op.batch_shape
        new_values = jnp.broadcast_to(op.values, prepend + op.values.shape)
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice) or c.ndim == 1:
                new_in_cols.append(c)
                continue
            # N-D cols: broadcast along the new leading axes.
            target = prepend + c.shape
            if isinstance(c, np.ndarray):
                new_in_cols.append(np.broadcast_to(c, target))
            else:
                new_in_cols.append(jnp.broadcast_to(c, target))
        return BEllpack(
            start_row=op.start_row, end_row=op.end_row,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=op.out_size, in_size=op.in_size,
            batch_shape=new_batch,
        )
    # Trailing-singleton (the `jnp.stack` pattern in LUKSAN11–16):
    # unbatched BEllpack aval-(n,) broadcast to aval-(n, 1, ..., 1). The
    # original rows become separate batches; the trailing size-1 axis
    # becomes the new out_size=1, and any extra middle-1 axes become
    # additional singleton batch axes. Triggered by `bid(shape=(n,
    # 1, ..., 1), broadcast_dimensions=(0,))`.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and len(broadcast_dimensions) == 1
            and broadcast_dimensions[0] == 0
            and len(shape) >= 2
            and shape[0] == op.out_size
            and all(s == 1 for s in shape[1:])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(shape[:-1])  # (n,) or (n, 1, ...)
        # Values: (nrows,) → (*new_batch, 1) for k=1; (nrows, k) →
        # (*new_batch, 1, k) for k>=2. All added axes are size-1 so a
        # plain reshape suffices (no broadcast_to needed).
        if op.k == 1:
            new_values = op.values.reshape(new_batch + (1,))
        else:
            new_values = op.values.reshape(new_batch + (1, op.k))
        # Cols: slice stays shared; ndarray (nrows,) or (nrows, k_band)
        # reshapes to (*new_batch, 1, ...k_band).
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            else:
                new_in_cols.append(c.reshape(new_batch + (1,) + c.shape[1:]))
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=1, in_size=op.in_size,
            batch_shape=new_batch,
        )
    # Leading-dim row-broadcast: `bid(unbatched BE out=N, shape=(N,
    # M_1, ..., M_{r-1}), bd=(0,))`. Output aval adds trailing
    # broadcast axes — each of the N original rows is replicated
    # across the new axes. Represent as batched BE `bs=(N, M_1, ...,
    # M_{r-2}), out=M_{r-1}`, values and cols broadcast-tiled over the
    # new axes. Chains with the reshape singleton-insert
    # (`_reshape_rule`) and dedup-in-reduce_sum to unblock NONMSQRT's
    # final `bid` step. An earlier version of this rule (reverted)
    # regressed because the upstream dedup was missing — K_intermediate
    # blew up past the dense alternative. With dedup, K at this step
    # is ≤ 70 for NONMSQRT-class (vs 5000+ without), and the bid
    # produces BE with nse close to the true matrix nnz.
    if (isinstance(op, BEllpack) and op.n_batch == 0
            and len(broadcast_dimensions) == 1
            and broadcast_dimensions[0] == 0
            and len(shape) >= 2
            and shape[0] == op.out_size
            and any(s > 1 for s in shape[1:])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(shape[:-1])           # (N, M_1, ..., M_{r-2})
        new_out = int(shape[-1])                # M_{r-1}
        N = op.out_size
        if op.k == 1:
            reshape_shape = (N,) + (1,) * (len(new_batch) - 1) + (1,)
            new_values = jnp.broadcast_to(
                op.values.reshape(reshape_shape),
                new_batch + (new_out,),
            )
        else:
            reshape_shape = (N,) + (1,) * (len(new_batch) - 1) + (1, op.k)
            new_values = jnp.broadcast_to(
                op.values.reshape(reshape_shape),
                new_batch + (new_out, op.k),
            )
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, slice):
                new_in_cols.append(c)
            elif isinstance(c, np.ndarray):
                reshape_c = (N,) + (1,) * (len(new_batch) - 1) + (1,) + c.shape[1:]
                new_in_cols.append(
                    # pyrefly: ignore [bad-argument-type]
                    np.broadcast_to(c.reshape(reshape_c),
                                    new_batch + (new_out,) + c.shape[1:])
                )
            else:
                ca = jnp.asarray(c)
                reshape_c = (N,) + (1,) * (len(new_batch) - 1) + (1,) + c.shape[1:]
                new_in_cols.append(
                    # pyrefly: ignore [bad-argument-type]
                    jnp.broadcast_to(ca.reshape(reshape_c),
                                     new_batch + (new_out,) + c.shape[1:])
                )
        return BEllpack(
            start_row=0, end_row=new_out,
            in_cols=tuple(new_in_cols), values=new_values,
            out_size=new_out, in_size=op.in_size,
            batch_shape=new_batch,
        )
    dense = _to_dense(op, n)
    # Dense-operand normalisation: when a prior rule has already
    # densified a 1-row LinOp (aval ndim 0, represented with a leading
    # size-1 axis), `dense` carries an extra axis beyond what the
    # broadcast_in_dim semantics expect. Squeeze leading size-1 axes
    # so `dense.ndim == len(broadcast_dimensions) + 1` (one for the
    # trailing input-coord axis). Example trigger: BEALE n=2 after
    # `_to_dense` on a 1-row BEllpack produces `(1, n)` then hits
    # `broadcast_in_dim(shape=(1,), dims=())` for scalar-aval input.
    expected_ndim = len(broadcast_dimensions) + 1
    while dense.ndim > expected_ndim and dense.shape[0] == 1:
        dense = dense[0]
    # Map each output axis to the corresponding input axis (or broadcast it).
    out_dims = tuple(broadcast_dimensions) + (len(shape),)  # add input axis
    return lax.broadcast_in_dim(dense, tuple(shape) + (n,), out_dims)

materialize_rules[lax.broadcast_in_dim_p] = _broadcast_in_dim_rule

def _bellpack_row_sum(ep):
    """Sum the rows of an unbatched BEllpack. When static cols let us
    compute the set of touched columns at trace time AND the result is
    structurally sparse (distinct cols < in_size), emit a BEllpack
    row-vector `(1, in_size)` whose bands hold the per-col sums —
    preserving sparsity through downstream broadcast_in_dim / pad /
    add_any chains. Otherwise fall back to a dense `(in_size,)` array.

    Used as `_reduce_sum_rule(ep, axes=(0,))`'s structural path.
    """
    assert ep.n_batch == 0
    nrows = ep.nrows
    k = ep.k
    in_size = ep.in_size
    per_band_cols = [_resolve_col(c, nrows) for c in ep.in_cols]
    if all(isinstance(c, np.ndarray) for c in per_band_cols):
        cols_flat = np.concatenate(per_band_cols)
        valid = cols_flat >= 0
        cols_valid = cols_flat[valid]
        uniq_cols, inverse = np.unique(cols_valid, return_inverse=True)
        n_groups = uniq_cols.shape[0]
        if 0 < n_groups < in_size:
            vals_flat = ep.values if k == 1 else ep.values.T.reshape(-1)
            keep = np.nonzero(valid)[0]
            if keep.shape[0] < cols_flat.shape[0]:
                vals_keep = jnp.take(vals_flat, jnp.asarray(keep))
            else:
                vals_keep = vals_flat
            summed = jnp.zeros((n_groups,), ep.dtype).at[
                jnp.asarray(inverse)].add(vals_keep)
            if n_groups == 1:
                return BEllpack(
                    start_row=0, end_row=1,
                    in_cols=(np.asarray([uniq_cols[0]], dtype=uniq_cols.dtype),),
                    values=summed.reshape(1),
                    out_size=1, in_size=in_size,
                )
            return BEllpack(
                start_row=0, end_row=1,
                in_cols=tuple(np.asarray([c], dtype=uniq_cols.dtype)
                              for c in uniq_cols),
                values=summed.reshape(1, n_groups),
                out_size=1, in_size=in_size,
            )
    # Tracer-cols fallback: flatten all bands at once. Values `(nrows,)`
    # for k=1 stay as-is; `(nrows, k)` for k>=2 goes band-major via
    # `.T.reshape(-1)`. Cols stack via jnp (some may be tracers). One
    # scatter into the result. Replaces the previous per-band Python
    # loop that emitted k scatter-add ops.
    cols_stacked = jnp.concatenate(
        [jnp.asarray(c) for c in per_band_cols], axis=0
    )
    vals_stacked = (ep.values if k == 1
                    else ep.values.T.reshape(-1))
    mask = cols_stacked >= 0
    safe_cols = jnp.where(mask, cols_stacked, 0)
    safe_vals = jnp.where(mask, vals_stacked, jnp.zeros((), ep.dtype))
    return jnp.zeros((in_size,), ep.dtype).at[safe_cols].add(safe_vals)


def _reduce_sum_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    axes = params["axes"]
    # BEllpack with leading batch dims: if `axes` cover the entire
    # batch_shape, split into per-batch Ellpack slices and sum them via
    # `_add_rule` (which handles same-cols dedup / band widening).
    if isinstance(op, BEllpack) and op.n_batch > 0:
        axes_t = tuple(sorted(axes))
        if axes_t == tuple(range(op.n_batch)):
            slices = _bellpack_unbatch(op)
            if len(slices) == 1:
                return slices[0]
            return _add_rule(list(slices), [True] * len(slices), n)
        # Partial batch-axis reduction: `axes` is a strict subset of
        # the batch axes. Safe to stay structural when every band's
        # cols are shared across the reduced axes (1-D shared cols, or
        # N-D cols with size-1 along those axes). Sum values along
        # those axes; shrink batch_shape accordingly. Used by
        # LUKSAN16LS: a `broadcast_in_dim`-tiled BE feeds a
        # reduce_sum(axes=(0,)) with n_batch=2 — previously densified
        # at this step, triggering 22 dense ops downstream.
        if (axes_t and axes_t[-1] < op.n_batch
                and len(axes_t) < op.n_batch):
            reduced = set(axes_t)
            safe = True
            for c in op.in_cols:
                if isinstance(c, slice):
                    continue
                if c.ndim == 1:
                    continue
                # 2D+ cols: must be size-1 along every reduced axis.
                for a in axes_t:
                    if a < c.ndim and c.shape[a] != 1:
                        safe = False
                        break
                if not safe:
                    break
            if safe:
                new_values = op.values.sum(axis=axes_t)
                new_in_cols = []
                for c in op.in_cols:
                    if isinstance(c, slice) or c.ndim == 1:
                        new_in_cols.append(c)
                    else:
                        # Drop the size-1 reduced axes.
                        new_in_cols.append(c.squeeze(axis=axes_t))
                new_batch = tuple(
                    s for i, s in enumerate(op.batch_shape)
                    if i not in reduced
                )
                return BEllpack(
                    op.start_row, op.end_row,
                    tuple(new_in_cols), new_values,
                    op.out_size, op.in_size,
                    batch_shape=new_batch,
                )
        # Out-axis-only reduction on a single-batch-axis BEllpack:
        # `axes == (n_batch,)` sums the out_size rows within each batch,
        # yielding an unbatched `(batch, in_size)` operator whose row-b
        # is the sum of batch b's out rows. Structural: unbatched
        # BEllpack `out_size=B, in_size=in_size, k=O*K_orig` with
        # (batch, orig_row, orig_band) → (new_band). Values reshape;
        # cols stack per (orig_row, orig_band). Downstream densify
        # goes through the single-scatter fused `.todense()` path, so
        # wide-K does not emit per-band loops.
        # Attacks NONMSQRT (42× vs asdex) — operator is 1/sqrt(n) dense
        # per row (K=70 at n=4900), a natural sparsity width.
        if (axes_t == (op.n_batch,) and op.n_batch == 1
                and op.start_row == 0 and op.end_row == op.out_size):
            B = op.batch_shape[0]
            O = op.out_size
            K = op.k
            # New unbatched BE has k_new = O * K. The iteration order
            # below MUST match the flattening of values: outer loop over
            # original rows r, inner loop over original bands b, so
            # new_band_idx = r * K + b — which matches the default
            # C-order flatten of `values.shape = (*batch, nrows=O, k=K)`
            # to `(B, O*K)`.
            new_in_cols = []
            for r in range(O):
                for b in range(K):
                    c = op.in_cols[b]
                    if isinstance(c, slice):
                        rs = np.arange(c.start or 0, c.stop or O, c.step or 1)
                        new_in_cols.append(np.broadcast_to(
                            np.asarray(rs[r]), (B,)).copy())
                    elif isinstance(c, np.ndarray) and c.ndim == 1:
                        new_in_cols.append(np.broadcast_to(
                            c[r:r+1], (B,)).copy())
                    elif isinstance(c, np.ndarray) and c.ndim == 2:
                        new_in_cols.append(c[:, r])
                    else:
                        # jnp tracer cols — 1D shared or (*batch, nrows).
                        c_full = c if c.ndim >= 2 else jnp.broadcast_to(
                            c, op.batch_shape + (op.nrows,))
                        # pyrefly: ignore [bad-argument-type]
                        new_in_cols.append(c_full[:, r])
            if K == 1:
                # (B, O) already in natural r-major order, k=O.
                new_values = op.values
            else:
                # (B, O, K) → flatten to (B, O*K) in r-major, b-minor order.
                new_values = op.values.reshape(B, O * K)
            # Dedup the O*K emitted bands. After out-axis reduction
            # many (r, b) pairs produce cols-identical bands — e.g.
            # NONMSQRT at n=4900 has O*K=5040 bands but only 70
            # unique cols (72× savings). Hash-group by cols bytes
            # (O(K) at trace time) and sum per-group values via one
            # scatter-add HLO op. Not gated on BELLPACK_DEDUP_LIMIT:
            # the dedup savings on wide-K justify the linear-time
            # cost (and the values scatter-add is one HLO op
            # regardless of K).
            def _col_key(c):
                if isinstance(c, np.ndarray):
                    return ("np", c.shape, c.tobytes())
                if isinstance(c, slice):
                    return ("slc", c.start, c.stop, c.step)
                return ("id", id(c))  # traced — won't group
            assigned = np.empty(len(new_in_cols), dtype=np.int64)
            group_cols: list = []
            key_to_group: dict = {}
            for i, c in enumerate(new_in_cols):
                k_ = _col_key(c)
                g = key_to_group.get(k_)
                if g is None:
                    g = len(group_cols)
                    key_to_group[k_] = g
                    group_cols.append(c)
                assigned[i] = g
            n_groups = len(group_cols)
            if n_groups < len(new_in_cols):
                # Scatter-add: new_values shape (B, O*K),
                # assigned shape (O*K,), output shape (B, n_groups).
                # `.at[..., assigned].add(new_values)` accumulates
                # along the last axis with repeated indices.
                dedup_values = jnp.zeros(
                    (B, n_groups), dtype=new_values.dtype
                ).at[:, assigned].add(new_values)
                return _densify_if_wider_than_dense(BEllpack(
                    start_row=0, end_row=B,
                    in_cols=tuple(group_cols), values=dedup_values,
                    out_size=B, in_size=op.in_size,
                ), n)
            return _densify_if_wider_than_dense(BEllpack(
                start_row=0, end_row=B,
                in_cols=tuple(new_in_cols), values=new_values,
                out_size=B, in_size=op.in_size,
            ), n)
    # BEllpack row-sum: accumulate per-column values via scatter-add.
    # Returns a 1D (in_size,) ndarray linear form — the Jacobian
    # coefficients of the resulting scalar-aval variable. Avoids the
    # (out_size, in_size) dense materialisation. Used by the LIARWHD-
    # class walk where the walker arrives at reduce_sum with a
    # structural BEllpack still carrying the sparsity (squeeze kept the
    # BEllpack row-vector, _add_rule tiled it into a banded matrix).
    if tuple(axes) == (0,) and isinstance(op, BEllpack) and op.n_batch == 0:
        return _bellpack_row_sum(op)
    # BCOO row-sum: when indices are static np, emit a structural BE
    # row-vector if `n_unique_cols < in_size`. Avoids densifying the
    # full (out_size, in_size) matrix. ~24 sweep hits.
    if (tuple(axes) == (0,) and isinstance(op, sparse.BCOO)
            and op.n_batch == 0):
        try:
            indices_np = np.asarray(op.indices)
        except (jax.errors.TracerArrayConversionError, TypeError):
            indices_np = None
        if isinstance(indices_np, np.ndarray):
            in_size = int(op.shape[1])
            cols_np = indices_np[:, 1]
            uniq, inverse = np.unique(cols_np, return_inverse=True)
            n_groups = int(uniq.shape[0])
            if 0 < n_groups < in_size:
                summed = jnp.zeros((n_groups,), op.data.dtype).at[
                    jnp.asarray(inverse)].add(op.data)
                if n_groups == 1:
                    return BEllpack(
                        start_row=0, end_row=1,
                        in_cols=(np.asarray([uniq[0]], dtype=uniq.dtype),),
                        values=summed.reshape(1),
                        out_size=1, in_size=in_size,
                    )
                return BEllpack(
                    start_row=0, end_row=1,
                    in_cols=tuple(np.asarray([c], dtype=uniq.dtype)
                                  for c in uniq),
                    values=summed.reshape(1, n_groups),
                    out_size=1, in_size=in_size,
                )
    # Fallback: densify + sum. Intentionally no shortcut for Diagonal /
    # ConstantDiagonal — those always densify to a (n,) linear form
    # anyway (1ᵀ diag(v) = v, dense) and returning `op.values` directly
    # breaks XLA fusion (measured 2.25× regression on ARGTRIGLS's dense
    # Hessian where the walk aggregates Diagonals through reduce_sum
    # before the inevitable dense-add downstream). Letting
    # `_to_dense + jnp.sum` stay in the jaxpr lets XLA reduce the
    # combined graph; the (n, n) intermediate is DCE'd.
    dense = _to_dense(op, n)
    return jnp.sum(dense, axis=tuple(axes))

materialize_rules[lax.reduce_sum_p] = _reduce_sum_rule

def _concatenate_rule(invals, traced, n, **params):
    if not any(traced):
        return None
    dimension = params["dimension"]
    traced_idxs = [i for i, t in enumerate(traced) if t]

    # Structural path: all-traced BEllpack concat, each operand spans
    # its full (0, out_size) range, same batch_shape and same in_size.
    # `dimension` in the aval is either a batch axis (dim < n_batch) or
    # the out_size axis (dim == n_batch). For unbatched operands, dim
    # must be 0 (the out_size axis). Produces a single BEllpack:
    #   * dim < n_batch: extend `batch_shape` at that axis.
    #   * dim == n_batch: extend `out_size`, widen to max_k bands.
    # Closure operands in the mix fall through to the pad-based paths.
    if (len(traced_idxs) == len(invals)
            and all(isinstance(v, BEllpack) for v in invals)
            and all(v.batch_shape == invals[0].batch_shape for v in invals)
            and all(v.in_size == invals[0].in_size for v in invals)
            and all(v.start_row == 0 and v.end_row == v.out_size for v in invals)
            and all(v.out_size == invals[0].out_size for v in invals[1:]
                    if dimension < invals[0].n_batch)):
        nb = invals[0].n_batch
        in_size = invals[0].in_size
        if dimension < nb:
            # Batch-axis concat: same out_size, same k assumed, just
            # concatenate values + per-batch in_cols along that axis
            # and grow batch_shape[dim].
            if not all(v.k == invals[0].k for v in invals[1:]):
                pass  # fall through to dense fallback (k mismatch rare
                      # on batch-axis concat; avoid complexity)
            else:
                new_values = jnp.concatenate([v.values for v in invals], axis=dimension)
                new_in_cols = []
                for b in range(invals[0].k):
                    parts = []
                    has_per_batch = False
                    for v in invals:
                        c = v.in_cols[b]
                        if isinstance(c, slice):
                            c = np.arange(c.start or 0, c.stop or v.nrows, c.step or 1)
                        if hasattr(c, "ndim") and c.ndim > 1:
                            has_per_batch = True
                        parts.append(c)
                    if has_per_batch:
                        norm = []
                        for v, c in zip(invals, parts):
                            if hasattr(c, "ndim") and c.ndim == 1:
                                shape = v.batch_shape + (v.nrows,)
                                if isinstance(c, np.ndarray):
                                    c = np.broadcast_to(c, shape)
                                else:
                                    c = jnp.broadcast_to(c, shape)
                            norm.append(c)
                        parts = norm
                        if all(isinstance(c, np.ndarray) for c in parts):
                            new_in_cols.append(np.concatenate(parts, axis=dimension))
                        else:
                            new_in_cols.append(jnp.concatenate(
                                [jnp.asarray(c) for c in parts], axis=dimension))
                    else:
                        # All 1D cols. Two sub-cases:
                        #   - All identical across operands: keep as 1D
                        #     (most efficient — broadcasts across batches).
                        #   - Differ: broadcast each to `(batch_shape[dim],
                        #     nrows)` and concatenate along `dim`, giving
                        #     per-batch 2D cols of shape `(sum_batch,
                        #     nrows)`. Closes LUKSAN17LS's
                        #     `concat(4 × BE(bs=(1,), out=49), dim=0)`
                        #     where each operand has different strided
                        #     1D cols (0,2,4,…; 1,3,5,…; etc.) — this
                        #     previously fell through to dense fallback.
                        if all(np.array_equal(np.asarray(c), np.asarray(parts[0])) for c in parts[1:]):
                            new_in_cols.append(parts[0])
                        else:
                            norm = []
                            for v, c in zip(invals, parts):
                                shape = v.batch_shape + (v.nrows,)
                                if isinstance(c, np.ndarray):
                                    norm.append(np.broadcast_to(c, shape))
                                else:
                                    # pyrefly: ignore [bad-argument-type]
                                    norm.append(jnp.broadcast_to(c, shape))
                            if all(isinstance(c, np.ndarray) for c in norm):
                                new_in_cols.append(
                                    np.concatenate(norm, axis=dimension))
                            else:
                                new_in_cols.append(jnp.concatenate(
                                    [jnp.asarray(c) for c in norm],
                                    axis=dimension))
                if new_in_cols is not None:
                    new_batch = list(invals[0].batch_shape)
                    new_batch[dimension] = sum(v.batch_shape[dimension] for v in invals)
                    return BEllpack(
                        0, invals[0].out_size, tuple(new_in_cols), new_values,
                        invals[0].out_size, in_size,
                        batch_shape=tuple(new_batch),
                    )
        elif dimension == nb:
            # Out-axis concat: extend out_size, widen bands to max_k
            # (shorter operands pad with -1 sentinels + 0 values).
            max_k = max(v.k for v in invals)
            def _widen_values(v):
                if max_k == 1:
                    return v.values
                vals = v.values if v.values.ndim == nb + 2 else v.values[..., None]
                if v.k < max_k:
                    pad = [(0, 0)] * vals.ndim
                    pad[-1] = (0, max_k - v.k)
                    vals = jnp.pad(vals, pad)
                return vals
            new_values = jnp.concatenate([_widen_values(v) for v in invals], axis=nb)
            new_in_cols = []
            for b in range(max_k):
                band_parts = []
                has_per_batch = False
                for v in invals:
                    if b < v.k:
                        c = v.in_cols[b]
                        if isinstance(c, slice):
                            c = np.arange(c.start or 0, c.stop or v.nrows, c.step or 1)
                    else:
                        c = np.full((v.nrows,), -1, dtype=np.int64)
                    if hasattr(c, "ndim") and c.ndim > 1:
                        has_per_batch = True
                    band_parts.append(c)
                if has_per_batch:
                    normalized = []
                    for v, c in zip(invals, band_parts):
                        if hasattr(c, "ndim") and c.ndim == 1:
                            shape = v.batch_shape + (v.nrows,)
                            if isinstance(c, np.ndarray):
                                c = np.broadcast_to(c, shape)
                            else:
                                c = jnp.broadcast_to(c, shape)
                        normalized.append(c)
                    axis = nb
                    band_parts = normalized
                else:
                    axis = 0
                if all(isinstance(c, np.ndarray) for c in band_parts):
                    new_in_cols.append(np.concatenate(band_parts, axis=axis))
                else:
                    # pyrefly: ignore [bad-argument-type]
                    new_in_cols.append(jnp.concatenate(
                        [jnp.asarray(c) for c in band_parts], axis=axis))
            total_out = sum(v.out_size for v in invals)
            return BEllpack(
                0, total_out, tuple(new_in_cols), new_values,
                total_out, in_size,
                batch_shape=invals[0].batch_shape,
            )

    # Structural fast path: `concatenate([C, ..., traced_op, ..., C], axis=0)`
    # — exactly one traced operand sandwiched by closures. Closures have no
    # dependency on the traced input, so their Jacobian rows are zero and the
    # result is structurally `op.pad_rows(left_total, right_total)`. Promote
    # (Constant)Diagonal to BEllpack first so pad_rows is available.
    if dimension == 0 and len(traced_idxs) == 1:
        idx = traced_idxs[0]
        op = invals[idx]
        left_total = sum(int(invals[i].shape[0]) for i in range(idx))
        right_total = sum(int(invals[i].shape[0])
                          for i in range(idx + 1, len(invals)))
        if isinstance(op, ConstantDiagonal):
            op = BEllpack(
                0, op.n, (np.arange(op.n),),
                jnp.broadcast_to(jnp.asarray(op.value), (op.n,)),
                op.n, op.n,
            )
        elif isinstance(op, Diagonal):
            op = BEllpack(
                0, op.n, (np.arange(op.n),), op.values, op.n, op.n,
            )
        if isinstance(op, BEllpack) and op.n_batch == 0:
            return op.pad_rows(left_total, right_total)
        if isinstance(op, sparse.BCOO):
            out_size = op.shape[0] + left_total + right_total
            new_rows = op.indices[:, 0] + left_total
            new_indices = jnp.stack([new_rows, op.indices[:, 1]], axis=1)
            return sparse.BCOO(
                (op.data, new_indices), shape=(out_size, op.shape[1])
            )
    # Fallback: densify everything.
    parts = []
    for v, t in zip(invals, traced):
        if t:
            parts.append(_to_dense(v, n))
        else:
            # Closure constant: extend with a zero "input axis" of size n.
            parts.append(jnp.broadcast_to(v[..., None] * 0, v.shape + (n,)))
    return lax.concatenate(parts, dimension)

materialize_rules[lax.concatenate_p] = _concatenate_rule

def _split_rule(invals, traced, n, **params):
    (operand,) = invals
    (t,) = traced
    if not t:
        return None
    sizes = params["sizes"]
    axis = params["axis"]
    # Structural path: batched BE split along the out-axis (== n_batch).
    # Slice values and each band's cols along the out axis; keep batch_shape.
    # Requires full out coverage so each chunk's rows map to [0, sz) cleanly.
    if (isinstance(operand, BEllpack)
            and operand.n_batch >= 1
            and axis == operand.n_batch
            and operand.start_row == 0
            and operand.end_row == operand.out_size):
        nb = operand.n_batch
        out = []
        start = 0
        for sz in sizes:
            sz_i = int(sz)
            end = start + sz_i
            val_slc = [slice(None)] * operand.values.ndim
            val_slc[nb] = slice(start, end)
            new_values = operand.values[tuple(val_slc)]
            new_in_cols = []
            for c in operand.in_cols:
                arr = c
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 1:
                        new_in_cols.append(arr[start:end])
                    else:
                        slc = [slice(None)] * arr.ndim
                        slc[nb] = slice(start, end)
                        new_in_cols.append(arr[tuple(slc)])
                else:
                    arr_j = jnp.asarray(arr)
                    if arr_j.ndim == 1:
                        # pyrefly: ignore [bad-argument-type]
                        new_in_cols.append(arr_j[start:end])
                    else:
                        slc = [slice(None)] * arr_j.ndim
                        slc[nb] = slice(start, end)
                        # pyrefly: ignore [bad-argument-type]
                        new_in_cols.append(arr_j[tuple(slc)])
            out.append(BEllpack(
                0, sz_i, tuple(new_in_cols), new_values,
                sz_i, operand.in_size, batch_shape=operand.batch_shape,
            ))
            start = end
        return out
    # Structural path: split along output axis 0 (the "out_size" dim).
    # For an unbatched BEllpack with static cols we slice the BE
    # per-chunk (row range + per-band-col row-slice) and emit one
    # proper BCOO per chunk. Going through `_to_bcoo` on the full BE
    # and then masking out-of-range rows to `(row=0, value=0)` would
    # leave zero-valued entries clogging row 0 of every chunk — those
    # count as BCOO nse and manufacture "duplicates" at row 0 that
    # propagate through every downstream add/concat (observed as
    # COATING's 4.5× final nse bloat).
    if (axis == 0 and isinstance(operand, BEllpack)
            and operand.n_batch == 0
            and all(isinstance(c, np.ndarray) or isinstance(c, slice)
                    for c in operand.in_cols)):
        out = []
        start = 0
        for sz in sizes:
            sz_i = int(sz)
            end = start + sz_i
            # Row range [start, end) intersected with BE's own
            # [start_row, end_row). Slice cols/values along the row axis.
            be_start = max(operand.start_row - start, 0)
            be_end = min(operand.end_row - start, sz_i)
            if be_end <= be_start:
                out.append(sparse.BCOO(
                    # pyrefly: ignore [bad-argument-type]
                    (jnp.zeros((0,), operand.values.dtype),
                     np.zeros((0, 2), np.int32)),
                    shape=(sz_i, operand.in_size),
                ))
                start = end
                continue
            row_lo = max(start, operand.start_row) - operand.start_row
            row_hi = min(end, operand.end_row) - operand.start_row
            new_in_cols = []
            for c in operand.in_cols:
                if isinstance(c, slice):
                    c = _resolve_col(c, operand.nrows)
                new_in_cols.append(c[row_lo:row_hi])
            if operand.k == 1:
                new_values = operand.values[row_lo:row_hi]
            else:
                new_values = operand.values[row_lo:row_hi, :]
            chunk_be = BEllpack(
                be_start, be_end, tuple(new_in_cols), new_values,
                sz_i, operand.in_size,
            )
            # pyrefly: ignore [bad-argument-type]
            out.append(chunk_be)
            start = end
        return out
    if axis == 0 and isinstance(operand, (ConstantDiagonal, Diagonal,
                                          BEllpack, sparse.BCOO)):
        bcoo = _to_bcoo(operand, n)
        rows = bcoo.indices[:, 0]
        out = []
        start = 0
        for sz in sizes:
            end = start + int(sz)
            in_range = (rows >= start) & (rows < end)
            new_rows = jnp.where(in_range, rows - start, 0)
            new_data = jnp.where(in_range, bcoo.data,
                                 jnp.zeros((), bcoo.data.dtype))
            new_indices = jnp.stack(
                [new_rows, bcoo.indices[:, 1]], axis=1
            )
            out.append(sparse.BCOO(
                (new_data, new_indices), shape=(int(sz), bcoo.shape[1])
            ))
            start = end
        return out
    dense = _to_dense(operand, n)
    out = []
    start = 0
    for sz in sizes:
        slc = [slice(None)] * dense.ndim
        slc[axis] = slice(int(start), int(start) + int(sz))
        out.append(dense[tuple(slc)])
        start += int(sz)
    return out

materialize_rules[lax.split_p] = _split_rule

def _cond_rule(invals, traced, n, **params):
    """`lax.cond` with a compile-time-decidable branch choice.

    Two senses of "traced" coexist here and are orthogonal:
      * `traced[0]` (walker-level): does the index depend on the
        walker's seed input? `True` means the branch choice is part
        of the structural chain — genuinely data-dependent control
        flow we don't support (non-linear).
      * `isinstance(invals[0], Tracer)` (JAX-level): is the VALUE a
        `DynamicJaxprTracer`? Under outer `jax.jit`, every closure
        `jnp.ndarray` gets lifted into the traced graph, so even
        walker-static closures can be tracers at the value level.

    Two structural (no-densify) patterns we handle:

    1. **Closure-concrete index**: un-jitted walks, or jitted walks
       under `EAGER_CONSTANT_FOLDING=TRUE`. `int(invals[0])` succeeds
       (0-d int arrays support `__index__`); we pick the branch.

    2. **`lax.platform_dependent`**: emits a `cond` whose index is
       `platform_index_p`, which stays an abstract tracer under outer
       jit without ECF. By the `platform_dependent` contract all
       branches are semantically equivalent — the actual platform is
       decided at lowering time. The eqn carries `branches_platforms`;
       we detect that and pick the `None` (default) branch. Covers
       the `jnp.diagonal` mosaic-vs-default dispatch HADAMALS hits
       under un-ECF jit.

    Neither case requires densification.
    """
    if traced[0]:
        raise NotImplementedError(
            "cond with walker-traced index (genuine control flow)"
        )
    branches = params["branches"]
    try:
        # Tuple-indexing uses `__index__`; 0-d int arrays / np scalars
        # / Python ints all support it. Under concrete conditions
        # (un-jit, or jit + ECF) this picks the right branch directly —
        # including the `platform_dependent` case, where
        # `platform_index_p` evaluates eagerly to the current
        # platform's branch index.
        chosen = branches[invals[0]]
    except jax.errors.TracerIntegerConversionError as e:
        # Tracer index (outer jit, no ECF). If this is a
        # `platform_dependent` cond, all branches are semantically
        # equivalent per its contract — pick the default (`None`
        # platform) branch. Otherwise we can't decide without
        # densifying.
        bp = params.get("branches_platforms")
        if bp is None:
            raise NotImplementedError(
                f"cond with tracer index ({type(invals[0]).__name__}) "
                f"and no `branches_platforms` hint — can't pick a branch "
                f"without densifying both"
            ) from e
        chosen = branches[next((i for i, pl in enumerate(bp) if pl is None), 0)]
    inner = chosen.jaxpr
    operand_invals = invals[1:]
    operand_traced = traced[1:]
    inner_env: dict = {v: (False, c) for v, c in zip(inner.constvars, chosen.consts)}
    for inner_invar, outer_val, was_traced in zip(
        inner.invars, operand_invals, operand_traced
    ):
        inner_env[inner_invar] = (was_traced, outer_val)
    _walk_jaxpr(inner, inner_env, n)
    return [inner_env[outvar][1] for outvar in inner.outvars]


try:
    from jax._src.lax.control_flow.conditionals import cond_p
    materialize_rules[cond_p] = _cond_rule
except ImportError:
    pass


def _jit_rule(invals, traced, n, **params):
    """Recurse into the inner jaxpr of a `jit` (pjit) call."""
    inner_cj = params["jaxpr"]  # ClosedJaxpr
    inner = inner_cj.jaxpr
    inner_consts = inner_cj.consts

    inner_env: dict = {v: (False, c) for v, c in zip(inner.constvars, inner_consts)}
    for inner_invar, outer_val, was_traced in zip(inner.invars, invals, traced):
        inner_env[inner_invar] = (was_traced, outer_val)
    _walk_jaxpr(inner, inner_env, n)

    # jit_p is always multiple_results; walker sets all outputs to traced
    # since _jit_rule is only called when any input is traced.
    return [inner_env[outvar][1] for outvar in inner.outvars]


# pjit_p is the modern name for jit's primitive.
try:
    from jax._src.pjit import jit_p

    materialize_rules[jit_p] = _jit_rule
except ImportError:
    pass


def _squeeze_leading_ones(arr, k):
    """Squeeze `k` leading size-1 axes from `arr`. Used to align
    densified LinOp case shapes in `_select_n_rule` (1-row BEs
    densify to `(1, n)` but represent the same aval as a scalar-aval
    LinOp densifying to `(n,)`)."""
    for _ in range(k):
        if arr.ndim == 0 or arr.shape[0] != 1:
            break
        arr = arr[0]
    return arr


def _select_n_rule(invals, traced, n, **params):
    """`select_n(pred, *cases)` for constant `pred`. Predicates derived from
    traced inputs would imply a data-dependent branch, which is non-linear and
    not supported.
    """
    del params
    pred = invals[0]
    cases = invals[1:]
    pred_traced = traced[0]
    case_traced = traced[1:]
    if pred_traced:
        raise NotImplementedError("select_n with traced predicate")
    if not any(case_traced):
        return None  # caller's constant-prop path handles concrete eval

    # Structural fast path (single traced op, rest closures): output row i
    # is zero wherever pred picks a closure case, else matches the traced
    # op's row i. Promote (Constant)Diagonal → BEllpack first, then mask.
    if sum(case_traced) == 1:
        t_idx = case_traced.index(True)
        t_case = cases[t_idx]
        if isinstance(t_case, ConstantDiagonal):
            t_case = BEllpack(
                0, t_case.n, (np.arange(t_case.n),),
                jnp.broadcast_to(jnp.asarray(t_case.value), (t_case.n,)),
                t_case.n, t_case.n,
            )
        elif isinstance(t_case, Diagonal):
            t_case = BEllpack(
                0, t_case.n, (np.arange(t_case.n),), t_case.values,
                t_case.n, t_case.n,
            )
        # BCOO: mask data entries by row-predicate.
        if isinstance(t_case, sparse.BCOO):
            pred_arr = jnp.asarray(pred)
            entry_rows = t_case.indices[:, 0]
            entry_mask = (pred_arr[entry_rows] == t_idx)
            new_data = jnp.where(entry_mask, t_case.data,
                                 jnp.zeros((), t_case.data.dtype))
            return sparse.BCOO(
                (new_data, t_case.indices), shape=t_case.shape
            )
    # pyrefly: ignore [unbound-name]
    if sum(case_traced) == 1 and isinstance(t_case, BEllpack):
        # pred has the BE's aval shape `(*batch_shape, out_size)`;
        # slice the last axis to the active row range. mask is
        # `(*batch_shape, nrows)`, broadcasting over the trailing k
        # axis for k>=2 values. Scalar pred (aval=()) applies
        # uniformly — skip the slice.
        pred_arr = jnp.asarray(pred)
        if pred_arr.ndim == 0:
            # pyrefly: ignore [unbound-name]
            mask = (pred_arr == t_idx)
        else:
            pred_slice = pred_arr[..., t_case.start_row:t_case.end_row]
            # pyrefly: ignore [unbound-name]
            mask = (pred_slice == t_idx)
            if t_case.k >= 2:
                mask = mask[..., None]
        new_values = jnp.where(mask, t_case.values,
                               jnp.zeros((), t_case.dtype))
        return BEllpack(
            t_case.start_row, t_case.end_row, t_case.in_cols,
            new_values, t_case.out_size, t_case.in_size,
            batch_shape=t_case.batch_shape,
        )

    # Structural fast path: all cases are BEllpack with matching
    # (start_row, end_row, out_size, in_size) and identical in_cols tuples.
    # Then select_n is a per-row choice among their values — emit one
    # BEllpack with `values = where(pred_slice, case_0.values, case_1.values, ...)`.
    if all(t and isinstance(c, BEllpack) and c.n_batch == 0
           for c, t in zip(cases, case_traced)):
        first = cases[0]
        same_shape = all(
            c.start_row == first.start_row and c.end_row == first.end_row
            and c.out_size == first.out_size and c.in_size == first.in_size
            for c in cases[1:]
        )
        same_cols = same_shape and all(
            len(c.in_cols) == len(first.in_cols)
            and all(_cols_equal(a, b)
                    for a, b in zip(c.in_cols, first.in_cols))
            for c in cases[1:]
        )
        if same_cols:
            pred_arr = jnp.asarray(pred)
            if pred_arr.ndim == 0:
                # Scalar pred applies uniformly across rows (HELIX /
                # PFIT* at n=3). No slicing or row-axis broadcast
                # needed; values broadcast against scalar naturally.
                pred_b = pred_arr
            else:
                pred_slice = pred_arr[first.start_row:first.end_row]
                pred_b = pred_slice[:, None] if first.values.ndim > 1 else pred_slice
            # select_n with bool pred: cases[0] when pred is False, cases[1] when True
            # (matching lax.select_n semantics for 2-case).
            if len(cases) == 2:
                new_values = jnp.where(pred_b, cases[1].values, cases[0].values)
            else:
                # N-way: use lax.select_n on stacked values.
                stacked = jnp.stack([c.values for c in cases], axis=0)
                new_values = lax.select_n(pred_b, *[stacked[i] for i in range(len(cases))])
            return BEllpack(
                first.start_row, first.end_row, first.in_cols,
                new_values, first.out_size, first.in_size,
            )

    # Multi-traced BCOO / BE (mismatched cols): mask each traced case by
    # its own `pred[row] == case_idx` predicate and concat the results as
    # a BCOO. Non-traced cases contribute zero to the linear-in-input
    # part, so we drop them. BE operands are promoted via `_to_bcoo` +
    # row-mask. Used by BROYDN7D (2×BCOO select_n over 5000-row state —
    # the dense fallback would emit a 25M-element matrix).
    # Gate: only fire when at least one operand is already BCOO AND
    # the aval has rank 1 (simple row-select). Pure CD/Diag cases
    # should go through the existing dense fallback to preserve the
    # bit-exact contract the `select_n(pred, d0, d1)` HLO provides.
    any_bcoo = any(isinstance(c, sparse.BCOO)
                   for c, t in zip(cases, case_traced) if t)
    if (any_bcoo and pred.ndim == 1
            and all(isinstance(c, (sparse.BCOO, BEllpack, Diagonal,
                                   ConstantDiagonal))
                    for c, t in zip(cases, case_traced) if t)):
        pred_arr = jnp.asarray(pred)
        masked_bcoos = []
        for c_idx, (c, t) in enumerate(zip(cases, case_traced)):
            if not t:
                continue
            bc = _to_bcoo(c, n)
            if bc.n_batch != 0:
                masked_bcoos = None
                break
            entry_rows = bc.indices[:, 0]
            mask = pred_arr[entry_rows] == c_idx
            new_data = jnp.where(mask, bc.data,
                                 jnp.zeros((), bc.data.dtype))
            masked_bcoos.append(sparse.BCOO(
                (new_data, bc.indices), shape=bc.shape,
            ))
        if masked_bcoos and len(masked_bcoos) == 1:
            return masked_bcoos[0]
        if masked_bcoos:
            return _bcoo_concat(masked_bcoos, shape=masked_bcoos[0].shape)

    # Densify each case to shape (*var_shape, n). Non-traced cases contribute
    # zero to the linear-in-input part (their dependence on the traced input
    # is zero), so we represent them as a zero tensor of the right shape.
    case_dense = []
    for c, t in zip(cases, case_traced):
        if t:
            case_dense.append(_to_dense(c, n))
        else:
            arr = jnp.asarray(c)
            zero_shape = arr.shape + (n,)
            case_dense.append(jnp.zeros(zero_shape, dtype=arr.dtype))
    # Normalise densified cases to the lowest aval-rank by squeezing
    # leading size-1 axes. A 1-row BEllpack (aval ndim 0) densifies to
    # `(1, n)`; a scalar-aval LinOp densifies to `(n,)`. Without this
    # align, `lax.select_n` rejects mismatched case shapes (HELIX n=3
    # repro: one case `(1, 3)` vs another `(3,)`).
    min_ndim = min(d.ndim for d in case_dense)
    case_dense = [
        d if d.ndim == min_ndim
        else _squeeze_leading_ones(d, d.ndim - min_ndim)
        for d in case_dense
    ]

    pred_arr = jnp.asarray(pred)
    # pred has shape (*var_shape,); broadcast it to (*var_shape, n) so each
    # row across the input-coord axis is selected the same way.
    target_shape = case_dense[0].shape
    pred_b = jnp.broadcast_to(pred_arr[..., None], target_shape)
    return lax.select_n(pred_b, *case_dense)

materialize_rules[lax.select_n_p] = _select_n_rule

def _cumsum_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    axis = params["axis"]
    reverse = params.get("reverse", False)
    dense = _to_dense(op, n)
    return lax.cumsum(dense, axis=axis, reverse=reverse)

materialize_rules[lax.cumsum_p] = _cumsum_rule

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

materialize_rules[lax.div_p] = _div_rule

def _transpose_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    permutation = tuple(params["permutation"])
    if isinstance(op, BEllpack):
        return op.transpose(permutation)
    dense = _to_dense(op, n)
    # Permutation applies to output axes only; preserve trailing input axis.
    return lax.transpose(dense, permutation + (len(permutation),))

materialize_rules[lax.transpose_p] = _transpose_rule

def _gather_rule(invals, traced, n, **params):
    operand, start_indices = invals
    to, ti = traced
    if ti:
        raise NotImplementedError("gather with traced indices")
    if not to:
        return None
    dnums = params["dimension_numbers"]
    # 2D point-gather fallback (HADAMALS-class): `M[b[i,0], b[i,1]]`
    # with dnums `collapsed_slice_dims=(0, 1), start_index_map=(0, 1),
    # slice_sizes=(1, 1)`. Materialize dense and gather. Correct but
    # not structural — leaves optimization for follow-up.
    if (
        dnums.offset_dims == ()
        and dnums.collapsed_slice_dims == (0, 1)
        and dnums.start_index_map == (0, 1)
        and params["slice_sizes"] == (1, 1)
    ):
        # Structural 2D point-gather on batched BEllpack (0d): each
        # gather index `(r, c)` picks the k entries stored at
        # `operand[r, c]` across all bands. Shapes: operand aval
        # `(R, C)` with nrows = C, per-band cols resolved to `(R, C)`,
        # values `(R, C[, k])`. Output LinOp batch_shape = leading
        # axes of `start_indices`, out_size = last leading axis,
        # `k = k_old`. Leaves the walk sparse where the dense fallback
        # would densify at `k_old * N` entries.
        if (isinstance(operand, BEllpack)
                and operand.n_batch == 1
                and operand.start_row == 0
                and operand.end_row == operand.out_size
                and start_indices.ndim >= 2):
            leading = start_indices.shape[:-1]
            new_batch = leading[:-1]
            new_out = leading[-1]
            # Fancy indexing with `row_flat`/`col_flat` needs jnp on
            # both sides when indices are outer-jit tracers. Cols that
            # were static np.ndarrays become jnp tracers after this —
            # downstream just treats them as traced cols.
            idx_static = isinstance(start_indices, np.ndarray)
            if idx_static:
                row_flat = start_indices[..., 0].reshape(-1)
                col_flat = start_indices[..., 1].reshape(-1)
            else:
                row_flat = jnp.asarray(start_indices[..., 0]).reshape(-1)
                col_flat = jnp.asarray(start_indices[..., 1]).reshape(-1)
            new_in_cols = []
            for c in operand.in_cols:
                c_full = _resolve_full(c, operand.nrows, operand.batch_shape)
                if not idx_static and isinstance(c_full, np.ndarray):
                    c_full = jnp.asarray(c_full)
                new_in_cols.append(c_full[row_flat, col_flat].reshape(leading))
            vals = operand.values[row_flat, col_flat]
            if operand.k == 1:
                vals = vals.reshape(leading)
            else:
                vals = vals.reshape(leading + (operand.k,))
            return BEllpack(
                start_row=0, end_row=new_out,
                in_cols=tuple(new_in_cols), values=vals,
                out_size=new_out, in_size=operand.in_size,
                batch_shape=new_batch,
            )
        dense = _to_dense(operand, n)
        # dense has shape `(*operand_primal_shape, n)` where the last
        # axis is the input axis; gather collapses the first two.
        row_idx = start_indices[..., 0]
        col_idx = start_indices[..., 1]
        return dense[row_idx, col_idx]
    # Two equivalent point-gather forms differing only in whether the
    # size-1 slice axis is collapsed away or kept in the output:
    #   collapsed: offset_dims=(),  collapsed_slice_dims=(0,) → out shape (..., )
    #   kept:      offset_dims=(1,), collapsed_slice_dims=()  → out shape (..., 1)
    # The kept form is what `vmap`-ed scalar indexing (e.g. older sif2jax
    # GENROSE's `vmap(lambda i: y[i])`) lowers to. Treat both, with the
    # kept form taking the dense fallback that adds the trailing axis.
    point_gather_collapsed = (
        dnums.offset_dims == ()
        and dnums.collapsed_slice_dims == (0,)
        and dnums.start_index_map == (0,)
        and params["slice_sizes"] == (1,)
    )
    point_gather_kept = (
        dnums.offset_dims == (1,)
        and dnums.collapsed_slice_dims == ()
        and dnums.start_index_map == (0,)
        and params["slice_sizes"] == (1,)
    )
    if not (point_gather_collapsed or point_gather_kept):
        raise NotImplementedError(f"gather with unhandled dnums: {dnums}")
    row_idx = start_indices[..., 0]
    # Build a BEllpack for (Constant)Diagonal operand. row_idx has shape
    # (*batch_shape, N) — 1D for the standard gather case, multi-dim for
    # batched gathers like SPARSINE's `sine_values[perm_indices]` with
    # perm_indices shape (B, N).
    if isinstance(operand, (ConstantDiagonal, Diagonal)):
        batch_shape = tuple(row_idx.shape[:-1])
        N = row_idx.shape[-1]
        if isinstance(operand, ConstantDiagonal):
            vals = jnp.broadcast_to(
                jnp.asarray(operand.value), batch_shape + (N,)
            )
        else:
            # Diagonal(v) — value at col c is v[c]. Gather v[row_idx].
            vals = jnp.take(operand.values, row_idx)
        if point_gather_kept:
            # Primal aval is (*batch_shape, N, 1). Encode by extending
            # batch with N and setting out_size=1 + nrows=1; cols and
            # values get an extra trailing axis of size 1.
            return BEllpack(
                start_row=0, end_row=1,
                in_cols=(row_idx[..., None],),
                values=vals[..., None],
                out_size=1, in_size=operand.n,
                batch_shape=batch_shape + (N,),
            )
        return BEllpack(
            start_row=0, end_row=N,
            in_cols=(row_idx,),
            values=vals,
            out_size=N, in_size=operand.n,
            batch_shape=batch_shape,
        )
    # 1-D gather on an unbatched BEllpack: pick rows by `row_idx`,
    # producing a BE with new out_size and the same `k`. Cols stay
    # static when both `op.in_cols[b]` and `row_idx` are np.ndarray.
    # Used by TOINTGOR's `M[indices]` pattern (top-10 loss).
    if (isinstance(operand, BEllpack) and operand.n_batch == 0
            and operand.start_row == 0
            and operand.end_row == operand.out_size):
        batch_shape = tuple(row_idx.shape[:-1])
        N = row_idx.shape[-1]
        idx_static = isinstance(row_idx, np.ndarray)
        ridx_flat = row_idx.reshape(-1) if idx_static else jnp.asarray(
            row_idx).reshape(-1)
        new_in_cols = []
        for c in operand.in_cols:
            cr = _resolve_col(c, operand.nrows)
            if idx_static and isinstance(cr, np.ndarray):
                gathered = cr[ridx_flat].reshape(batch_shape + (N,))
            else:
                gathered = jnp.asarray(cr)[ridx_flat].reshape(
                    batch_shape + (N,))
            new_in_cols.append(gathered)
        if operand.k == 1:
            vals = operand.values[ridx_flat].reshape(batch_shape + (N,))
        else:
            vals = operand.values[ridx_flat].reshape(
                batch_shape + (N, operand.k))
        if point_gather_kept:
            # Same trailing-1 trick as the Diagonal branch above.
            new_in_cols = tuple(c[..., None] for c in new_in_cols)
            vals = vals[..., None] if operand.k == 1 else vals[..., None, :]
            return BEllpack(
                start_row=0, end_row=1,
                in_cols=new_in_cols, values=vals,
                out_size=1, in_size=operand.in_size,
                batch_shape=batch_shape + (N,),
            )
        return BEllpack(
            start_row=0, end_row=N,
            in_cols=tuple(new_in_cols), values=vals,
            out_size=N, in_size=operand.in_size,
            batch_shape=batch_shape,
        )
    if isinstance(operand, sparse.BCOO):
        raise NotImplementedError("gather on BCOO operand")
    # Dense fallback: gather rows of the dense linop.
    dense = _to_dense(operand, n)
    if point_gather_kept:
        return dense[row_idx][..., None, :]
    return dense[row_idx]

materialize_rules[lax.gather_p] = _gather_rule

def _scatter_add_rule(invals, traced, n, **params):
    operand, scatter_indices, updates = invals
    to, ti, tu = traced
    if ti:
        raise NotImplementedError("scatter-add with traced indices")
    if to:
        raise NotImplementedError("scatter-add with traced operand")
    if not tu:
        return None
    dnums = params["dimension_numbers"]
    # 2D point-scatter fallback (HADAMALS-class): inserts `updates[k]`
    # at `operand[scatter_indices[k, 0], scatter_indices[k, 1]]`. Emit
    # as a BCOO with the static (i, j) indices and the traced values.
    if (
        dnums.update_window_dims == ()
        and dnums.inserted_window_dims == (0, 1)
        and dnums.scatter_dims_to_operand_dims == (0, 1)
        and operand.ndim == 2
    ):
        operand_dense = jnp.asarray(operand)
        out_shape_2d = operand_dense.shape  # (R, C)
        # updates shape: scatter_indices' leading dims (N,) + input_axis (n,)
        updates_dense = _to_dense(updates, n)  # (N, n) or broader
        # Flatten any leading batch shape of scatter_indices to N.
        si_flat = scatter_indices.reshape(-1, 2)
        updates_flat = updates_dense.reshape(-1, n)
        # Emit to a flat 2D BCOO matching operand shape: the output has
        # shape `(R*C, n)` — flatten the (R, C) output axis so it lives
        # in LinOp's (out_size, in_size) layout.
        flat_rows = (
            si_flat[:, 0].astype(jnp.int64) * out_shape_2d[1]
            + si_flat[:, 1].astype(jnp.int64)
        )
        # Dense approach: scatter into flattened zeros.
        out_size_flat = out_shape_2d[0] * out_shape_2d[1]
        return (
            jnp.zeros((out_size_flat, n), updates_flat.dtype)
            .at[flat_rows]
            .add(updates_flat)
            .reshape(out_shape_2d + (n,))
        )
    # Two equivalent point-scatter forms (transposes of the two gather
    # variants in `_gather_rule`):
    #   collapsed: update_window_dims=(),  inserted_window_dims=(0,)
    #   kept:      update_window_dims=(1,), inserted_window_dims=()
    # The kept form arises from `linear_transpose` of vmap-ed scalar
    # indexing (older sif2jax GENROSE). Its updates carry an extra
    # size-1 axis; squeeze it and route through the collapsed logic.
    scatter_collapsed = (
        dnums.update_window_dims == ()
        and dnums.inserted_window_dims == (0,)
        and dnums.scatter_dims_to_operand_dims == (0,)
    )
    scatter_kept = (
        dnums.update_window_dims == (1,)
        and dnums.inserted_window_dims == ()
        and dnums.scatter_dims_to_operand_dims == (0,)
    )
    if not (scatter_collapsed or scatter_kept):
        raise NotImplementedError(f"scatter-add with unhandled dnums: {dnums}")
    if scatter_kept:
        # Drop the size-1 trailing axis from updates and rerun as collapsed.
        # The kept form arises as the linear_transpose of the corresponding
        # gather kept form, which encodes the trailing 1 by setting
        # out_size=1 and appending the "real" out axis to batch_shape.
        # Inverting that is structural — no dense fallback needed.
        if (isinstance(updates, BEllpack)
                and updates.out_size == 1
                and updates.start_row == 0
                and updates.end_row == 1
                and len(updates.batch_shape) >= 1):
            inner = updates.batch_shape[-1]
            new_batch = updates.batch_shape[:-1]
            new_in_cols = tuple(
                c if isinstance(c, slice) else c.reshape(c.shape[:-1])
                for c in updates.in_cols
            )
            if updates.k == 1:
                new_values = updates.values.reshape(updates.values.shape[:-1])
            else:
                # values shape (..., 1, k) → drop the size-1 axis
                shape = updates.values.shape
                new_values = updates.values.reshape(shape[:-2] + shape[-1:])
            updates = BEllpack(
                start_row=0, end_row=inner,
                in_cols=new_in_cols, values=new_values,
                out_size=inner, in_size=updates.in_size,
                batch_shape=new_batch,
            )
        elif isinstance(updates, BEllpack):
            updates = _to_dense(updates, n).squeeze(axis=-2)
        elif isinstance(updates, sparse.BCOO):
            updates = _to_dense(updates, n).squeeze(axis=-2)
        else:
            updates = jnp.asarray(updates).squeeze(axis=-2)
    out_idx = scatter_indices[..., 0]
    out_size = operand.shape[0]
    # BEllpack updates: batched case handled per-slice (each batch's
    # Ellpack rows get remapped via scatter_indices[b]), then concat as
    # BCOO. Unbatched case falls through to the 1D-BCOO path below.
    if isinstance(updates, BEllpack):
        if updates.n_batch == 0:
            updates = _to_bcoo(updates, n)
        else:
            # Batched: unbatch, remap each slice's rows via the static
            # inverse-scatter map, then combine via _add_rule dedup.
            # Gather-based: ep.values[inv_map] is a jnp gather with
            # static indices into dynamic data — ECF-safe. The previous
            # zeros.at[inverse].add(data) scatter was ECF-toxic when
            # the output size was large (unique count ~150K for SPARSINE)
            # because XLA constant-folded the scatter at compile time.
            # The gather approach also avoids BCOO entirely when all
            # batch slices have static out_idx and static in_cols; the
            # resulting BEllpack stays sparse through downstream rules.
            slices = _bellpack_unbatch(updates)
            be_pieces = []
            static_ok = True
            for b_idx, ep in enumerate(slices):
                try:
                    out_idx_b = np.asarray(out_idx[b_idx])
                except (jax.errors.TracerArrayConversionError, TypeError):
                    static_ok = False
                    break
                nrows_b = ep.nrows
                # Source-row count must match scatter-target count, else
                # the slice scatters partially and we can't gather.
                if out_idx_b.shape[0] != nrows_b:
                    static_ok = False
                    break
                # Group source rows by output row at trace time. unique_out
                # is sorted; counts[r] = number of source rows scattering to
                # unique_out[r]. We emit a structural BE iff the active rows
                # form a contiguous range AND the duplicate factor is uniform
                # — otherwise sentinels would inflate nse beyond what the
                # BCOO concat fallback gives. SPARSINE's k=2 slice maps to
                # the odd outputs {1, 3, ..., 4999} (non-contiguous) → BCOO.
                unique_out, inverse = np.unique(out_idx_b, return_inverse=True)
                n_unique = unique_out.shape[0]
                counts = np.bincount(inverse, minlength=n_unique)
                contiguous = (
                    n_unique > 0
                    and int(unique_out[-1] - unique_out[0]) + 1 == n_unique
                )
                if not (contiguous and int(counts.min()) == int(counts.max())):
                    static_ok = False
                    break
                dup = int(counts[0])
                # order[r, d] = source row index for d-th writer to unique_out[r]
                order = np.argsort(inverse, kind="stable").reshape(n_unique, dup)
                try:
                    new_in_cols = []
                    for d in range(dup):
                        src = order[:, d]
                        for col in ep.in_cols:
                            col_np = _resolve_col(col, nrows_b)
                            if not isinstance(col_np, np.ndarray):
                                col_np = np.asarray(col_np)
                            new_in_cols.append(col_np[src].astype(np.intp))
                except (jax.errors.TracerArrayConversionError, TypeError):
                    static_ok = False
                    break
                if ep.k == 1:
                    new_vals = ep.values[order]  # (n_unique, dup)
                else:
                    new_vals = ep.values[order].reshape(n_unique, dup * ep.k)
                start = int(unique_out[0])
                be_pieces.append(BEllpack(
                    start, start + n_unique, tuple(new_in_cols), new_vals,
                    out_size, ep.in_size,
                ))
            if static_ok:
                return _add_rule(be_pieces, [True] * len(be_pieces), n)
            # Traced indices fallback: original concat-as-BCOO path.
            bcoo_pieces = []
            for b_idx, ep in enumerate(slices):
                bc = _to_bcoo(ep, n)
                old_rows = bc.indices[:, 0]
                new_rows = out_idx[b_idx][old_rows]
                new_indices = jnp.stack(
                    [new_rows, bc.indices[:, 1]], axis=1
                )
                bcoo_pieces.append(sparse.BCOO(
                    (bc.data, new_indices),
                    shape=(out_size, updates.in_size),
                ))
            return _bcoo_concat(
                bcoo_pieces, shape=(out_size, updates.in_size)
            )
    out_idx_flat = out_idx.reshape(-1)
    if isinstance(updates, sparse.BCOO):
        new_rows = out_idx_flat[updates.indices[:, 0]]
        new_indices = jnp.stack([new_rows, updates.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (updates.data, new_indices), shape=(out_size, n)
        )
    # Dense fallback: updates has shape (*batch, n). Reshape to (prod_batch, n)
    # and scatter-add row i of flat into row out_idx_flat[i] of a zero (out_size, n).
    updates_dense = _to_dense(updates, n)
    flat_updates = updates_dense.reshape(-1, n)
    return (jnp.zeros((out_size, n), flat_updates.dtype)
            .at[out_idx_flat].add(flat_updates))

materialize_rules[_slicing.scatter_add_p] = _scatter_add_rule

# -------------------------- driver --------------------------


def _walk_jaxpr(jaxpr, env, n):
    """Walk a jaxpr, mutating env.

    Env is `dict[Var, tuple[bool, Any]]` where the bool is `traced`:
      * (True, LinOp) — this var depends on the walk's input; value is a LinOp.
      * (False, concrete_array) — this var is pure closure data.
    Literals are read directly from `.val`; traced status comes from the
    invars the caller seeded.
    """

    def read(atom):
        if isinstance(atom, core.Literal):
            return (False, atom.val)
        return env[atom]

    for eqn in jaxpr.eqns:
        entries = [read(v) for v in eqn.invars]
        invals = [e[1] for e in entries]
        traced = [e[0] for e in entries]
        if not any(traced):
            # Constant propagation: no traced inputs → evaluate concretely
            # and stash as closure data. Important for constant-H problems
            # (DUAL, CMPC) — lets the whole walk fold to a trace-time BCOO
            # literal. See docs/RESEARCH_NOTES.md §10.
            concrete_outs = eqn.primitive.bind(*invals, **eqn.params)
            if eqn.primitive.multiple_results:
                for v, o in zip(eqn.outvars, concrete_outs):
                    env[v] = (False, o)
            else:
                (outvar,) = eqn.outvars
                env[outvar] = (False, concrete_outs)
            continue
        rule = materialize_rules.get(eqn.primitive)
        if rule is None:
            forms = ", ".join(
                type(v).__name__ if t else f"closure:{type(v).__name__}"
                for v, t in zip(invals, traced)
            )
            raise NotImplementedError(
                f"No lineaxpr rule for primitive '{eqn.primitive}'.\n"
                f"  Input forms: [{forms}]\n"
                f"  To add a rule: register at lineaxpr.materialize_rules[{eqn.primitive}] = your_rule\n"
                f"  Or file an issue at https://github.com/jpbrodrick89/lineaxpr/issues "
                f"with the minimal f(y) that triggers this."
            )
        outs = rule(invals, traced, n, **eqn.params)
        if eqn.primitive.multiple_results:
            for v, o in zip(eqn.outvars, outs):
                env[v] = (True, o)
        else:
            (outvar,) = eqn.outvars
            env[outvar] = (True, outs)


def _walk_with_seed(linear_fn, seed_linop):
    """Trace `linear_fn` with the aval implied by `seed_linop`, walk the
    jaxpr, return the output LinOp."""
    placeholder = jax.ShapeDtypeStruct((seed_linop.shape[-1],), seed_linop.dtype)
    cj = jax.make_jaxpr(linear_fn)(placeholder)
    jaxpr = cj.jaxpr

    if len(jaxpr.invars) != 1:
        raise NotImplementedError("multi-input linear_fn not yet handled")
    (invar,) = jaxpr.invars
    n = invar.aval.size

    env: dict = {v: (False, c) for v, c in zip(jaxpr.constvars, cj.consts)}
    env[invar] = (True, seed_linop)
    _walk_jaxpr(jaxpr, env, n)

    if len(jaxpr.outvars) != 1:
        raise NotImplementedError("multi-output linear_fn not yet handled")
    (outvar,) = jaxpr.outvars
    return env[outvar][1]


def sparsify(linear_fn):
    """Transform a linear function into one that operates on LinOps.

    `sparsify(linear_fn)(seed_linop)` traces `linear_fn` against the aval
    implied by `seed_linop.primal_aval()`, walks the resulting jaxpr with
    per-primitive structural rules, and returns a LinOp representing the
    linear function's matrix.

    Seeds are explicit — no automatic Identity cast. For the common case
    of extracting the full Jacobian, the public wrappers `materialize` /
    `jacfwd` / `jacrev` / `hessian` build
    `Identity(primal.size, dtype=primal.dtype)` and pass it through.
    """
    def inner(seed_linop):
        return _walk_with_seed(linear_fn, seed_linop)

    return inner


_SMALL_N_VMAP_THRESHOLD = 16
"""Deprecated: shortcut removed — all problems exercise the structural
walk. Threshold retained as a sentinel so the `lineaxpr` re-export stays
valid; no live code path uses it. Formerly: below this n,
`vmap(linear_fn)(eye)` emits less HLO than the structural walk
on most problems. Above it the walk's structure exploitation dominates."""


def to_dense(op):
    """Densify a LinOp returned by `sparsify` to a jnp.ndarray.

    Uniform across all possible return types:
    - Our LinOp classes (ConstantDiagonal, Diagonal, BEllpack) → `.todense()`.
    - `jax.experimental.sparse.BCOO` → `.todense()`.
    - Plain ndarray → passthrough.
    """
    if isinstance(op, (ConstantDiagonal, Diagonal, BEllpack)):
        return op.todense()
    if isinstance(op, sparse.BCOO):
        return op.todense()
    return op


def to_bcoo(op):
    """Convert a LinOp returned by `sparsify` to a BCOO (or ndarray if
    the walk produced a dense fallback that can't be usefully sparsified).

    - Our LinOp classes → `.to_bcoo()`.
    - `BCOO` passthrough.
    - Plain ndarray passthrough (caller decides what to do).
    """
    if isinstance(op, sparse.BCOO):
        return op
    if isinstance(op, (ConstantDiagonal, Diagonal, BEllpack)):
        return op.to_bcoo()
    return op


_VALID_FORMATS = ("dense", "bcoo")


def materialize(linear_fn, primal, format: str = "dense"):
    """Materialize the Jacobian matrix of a linear callable.

    Args:
      linear_fn: a linear callable `R^n -> R^m` (typically the output of
        `jax.linearize(...)[1]` or `jax.linear_transpose(...)`).
      primal: a shape/dtype witness for the input to `linear_fn`. Only
        `primal.size` and `primal.dtype` are read, so this can be any of:
        a concrete array, a `jax.Array` / `jnp.ndarray`, or a
        `jax.ShapeDtypeStruct` (matching the convention used by
        `jax.linear_transpose` / `jax.eval_shape`). Passing a
        ShapeDtypeStruct is the preferred option when you don't already
        have a concrete primal on hand.
      format: one of `"dense"` or `"bcoo"`.
        - `"dense"` returns a `jnp.ndarray`.
        - `"bcoo"` returns a `jax.experimental.sparse.BCOO` when the
          walk preserves structural sparsity, otherwise a dense ndarray
          (dense fallbacks surface to the caller unchanged).

    """
    if format not in _VALID_FORMATS:
        raise ValueError(f"format must be one of {_VALID_FORMATS}, got {format!r}")
    n = primal.size if hasattr(primal, "size") else int(jnp.size(primal))
    seed = Identity(n, dtype=primal.dtype)
    linop = sparsify(linear_fn)(seed)
    if format == "dense":
        return to_dense(linop)
    bcoo = to_bcoo(linop)
    # Smart-densify at output: at `nse >= prod(shape)` the BCOO stores
    # at least as many float values as dense AND carries 2·nse index
    # ints on top — strictly worse than dense. DUAL-class problems
    # (small n, highly-connected) hit this when `_bcoo_concat` stacks
    # many overlapping BCOO operands without deduping.
    if isinstance(bcoo, sparse.BCOO):
        total = 1
        for s in bcoo.shape:
            total *= int(s)
        if bcoo.nse >= total:
            return bcoo.todense()
    return bcoo


# -------------------------- jax-like public API --------------------------


def _jacfwd_impl(f, y, format: str):
    """materialize ∘ jax.linearize — forward-mode Jacobian."""
    y_out, lin = jax.linearize(f, y)
    del y_out  # only needed for shape in jacrev
    return materialize(lin, y, format=format)


def _jacrev_impl(f, y, format: str):
    """materialize ∘ jax.linear_transpose ∘ jax.linearize — reverse-mode
    Jacobian. linear_transpose of the JVP is the VJP; materializing gives
    Jᵀ, so we transpose the result to match `jax.jacrev`'s shape."""
    y_out, lin = jax.linearize(f, y)
    vjp = jax.linear_transpose(lin, y)
    # vjp: R^m -> R^n, where m = y_out.shape.
    # jax.linear_transpose wraps the result in a tuple (multi-output),
    # so we unpack.
    def vjp_unpacked(w):
        (out,) = vjp(w)
        return out
    # Primal for the VJP is a shape/dtype witness of y_out.
    jt = materialize(vjp_unpacked, y_out, format=format)
    # jt has shape (n, m); we want (m, n) to match jax.jacrev.
    if format == "dense":
        return jt.T
    # BCOO supports .T via transpose().
    return jt.T


def jacfwd(f, *, format: str = "dense"):
    """Forward-mode Jacobian, matching `jax.jacfwd`'s output shape.

    Equivalent to `materialize(jax.linearize(f, y)[1], y, format=format)`.

    Returns a function `(y) -> Jacobian`. `format='dense'` (default)
    returns a `jnp.ndarray`; `format='bcoo'` returns a BCOO when
    structural sparsity survives, else a dense ndarray.

    Only single-input / single-output `f` with 1D `y` is currently
    supported — see `docs/TODO.md` for the multi-input / multi-output
    roadmap.
    """
    def wrapped(y):
        return _jacfwd_impl(f, y, format)
    return wrapped


def bcoo_jacfwd(f):
    """Forward-mode Jacobian returned as BCOO. Alias for
    `jacfwd(f, format='bcoo')`."""
    return jacfwd(f, format="bcoo")


def jacrev(f, *, format: str = "dense"):
    """Reverse-mode Jacobian, matching `jax.jacrev`'s output shape.

    Equivalent to `materialize(linear_transpose(linearize(f, y)[1], y),
    y_out, format=format).T`.

    Returns a function `(y) -> Jacobian`.
    """
    def wrapped(y):
        return _jacrev_impl(f, y, format)
    return wrapped


def bcoo_jacrev(f):
    """Reverse-mode Jacobian returned as BCOO. Alias for
    `jacrev(f, format='bcoo')`."""
    return jacrev(f, format="bcoo")


def hessian(f, *, format: str = "dense"):
    """Hessian, matching `jax.hessian`'s output shape.

    Equivalent to `materialize(jax.linearize(jax.grad(f), y)[1], y,
    format=format)`.

    Returns a function `(y) -> Hessian`.
    """
    def wrapped(y):
        _, lin = jax.linearize(jax.grad(f), y)
        return materialize(lin, y, format=format)
    return wrapped


def bcoo_hessian(f):
    """Hessian returned as BCOO. Alias for `hessian(f, format='bcoo')`."""
    return hessian(f, format="bcoo")


# -------------------------- demo --------------------------


def _demo():
    from sif2jax.cutest._quadratic_problems.dual1 import DUAL1
    from sif2jax.cutest._quadratic_problems.dual2 import DUAL2
    from sif2jax.cutest._quadratic_problems.dual3 import DUAL3
    from sif2jax.cutest._quadratic_problems.dual4 import DUAL4

    for cls in (DUAL1, DUAL2, DUAL3, DUAL4):
        p = cls()
        y = p.y0

        def f(z):
            return p.objective(z, p.args)

        _, hvp = jax.linearize(jax.grad(f), y)
        H_ref = jax.vmap(hvp)(jnp.eye(p.n)).T
        H_ours = materialize(hvp, y)
        err = float(jnp.max(jnp.abs(H_ours - H_ref)))
        # pyrefly: ignore [unsupported-operation]
        sym = float(jnp.max(jnp.abs(H_ours - H_ours.T)))
        print(
            f"{cls.__name__:6s}  n={p.n:3d}  "
            f"max|H_ours - H_ref| = {err:.2e}  |H - H.T| = {sym:.2e}"
        )

        def run_jit(y_):
            _, hvp_ = jax.linearize(jax.grad(f), y_)
            return materialize(hvp_, y_)

        H_jit = jax.jit(run_jit)(y)
        err_jit = float(jnp.max(jnp.abs(H_jit - H_ref)))
        print(f"         inside-jit err: {err_jit:.2e}")


if __name__ == "__main__":
    _demo()
