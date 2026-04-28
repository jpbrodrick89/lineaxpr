"""Unary structural rules: identity, neg, slice, pad, squeeze, rev, reshape,
broadcast_in_dim, reduce_sum, cumsum, transpose."""

from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import sparse

from .._linops import (
    BEllpack,
    ConstantDiagonal,
    Diagonal,
    LinOpProtocol,
    _resolve_col,
    _to_dense,
)
from .._linops import negate
from .add import (
    _add_rule,
    _bellpack_unbatch,
    _densify_if_wider_than_dense,
)


def _identity_rule(invals, traced, n, **params):
    """For primitives that don't change value (convert_element_type, copy)."""
    del params
    (op,) = invals
    (t,) = traced
    return op if t else None


def _neg_rule(invals, traced, n, **params):
    del params, n
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    if isinstance(op, LinOpProtocol):
        return negate(op)
    return -op


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
            from .._base import _to_bcoo
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


def _cumsum_rule(invals, traced, n, **params):
    (op,) = invals
    (t,) = traced
    if not t:
        return None
    axis = params["axis"]
    reverse = params.get("reverse", False)
    dense = _to_dense(op, n)
    return lax.cumsum(dense, axis=axis, reverse=reverse)


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
