"""Structural rules: concatenate and split."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import sparse

from .._linops import (
    BEllpack,
    ColArr,
    ConstantDiagonal,
    Diagonal,
    LinOpProtocol,
)


def _concatenate_rule(invals, traced, n, **params):
    if not any(traced):
        return None
    dimension = params["dimension"]
    traced_idxs = [i for i, t in enumerate(traced) if t]

    # Phase B: all-traced BE_T concat — flip all to canonical (free
    # flag flip), shift `dimension` to the canonical frame (V was at
    # axis 0, now at axis -1; structural axes shift down by 1), recurse
    # through the canonical body, flip the BE result back to T=True.
    # Bridges concat-along-out chains (BROYDN3DLS-class:
    # `concat([BE_T(n,1), BE_T(n,m), BE_T(n,1)], dim=1)`) whose
    # canonical-frame analogue is `dim=0` and hits the existing
    # out-axis structural path (line ~116).
    from .._linops.base import replace_slots  # noqa: PLC0415
    traced_be = [v for v, t in zip(invals, traced) if t and isinstance(v, BEllpack)]
    if (traced_be
            and len(traced_be) == sum(traced)
            and all(v.transposed for v in traced_be)
            and dimension >= 1):
        flipped = [
            replace_slots(v, transposed=False) if isinstance(v, BEllpack) else v
            for v in invals
        ]
        new_params = dict(params, dimension=dimension - 1)
        res = _concatenate_rule(flipped, traced, n, **new_params)
        if isinstance(res, BEllpack):
            return replace_slots(res, transposed=True)
        if (isinstance(res, sparse.BCOO)
                and res.indices.ndim == 2
                and res.indices.shape[-1] == 2):
            return res.transpose(axes=(1, 0))
        if hasattr(res, "ndim") and res.ndim >= 2:
            perm = (tuple(range(res.ndim - 2))
                    + (res.ndim - 1, res.ndim - 2))
            return jnp.transpose(res, perm)
        return res

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
            # Batch-axis concat: same out_size; pad shorter k to max_k
            # with `-1` sentinel cols / 0 values, then concatenate
            # values + per-band in_cols along the batch axis and grow
            # batch_shape[dim]. LUKSAN11LS's
            # `concat(BE(bs=(99,1), out=1, k=2), BE(bs=(99,1), out=1, k=1))`
            # — different k from upstream sub vs slice — used to fall
            # through to dense.
            max_k = max(v.k for v in invals)
            def _widen_values_batch(v):
                if max_k == 1:
                    return v.data
                vals = v.data if v.data.ndim == nb + 2 else v.data[..., None]
                if v.k < max_k:
                    pad = [(0, 0)] * vals.ndim
                    pad[-1] = (0, max_k - v.k)
                    vals = jnp.pad(vals, pad)
                return vals
            if True:
                new_values = jnp.concatenate(
                    [_widen_values_batch(v) for v in invals],
                    axis=dimension,
                )
                new_in_cols: list[ColArr] = []
                for b in range(max_k):
                    parts = []
                    has_per_batch = False
                    for v in invals:
                        if b >= v.k:
                            # Sentinel band: -1 cols, 0 values for the
                            # whole batch shape.
                            c = np.full((v.nrows,), -1, dtype=np.int64)
                            parts.append(c)
                            continue
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
                    return v.data
                vals = v.data if v.data.ndim == nb + 2 else v.data[..., None]
                if v.k < max_k:
                    pad = [(0, 0)] * vals.ndim
                    pad[-1] = (0, max_k - v.k)
                    vals = jnp.pad(vals, pad)
                return vals
            new_values = jnp.concatenate([_widen_values(v) for v in invals], axis=nb)
            new_in_cols: list[ColArr] = []
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

    # Structural fast path: `concatenate([C, ..., traced_op, ..., C])`
    # along the primal out_size axis — exactly one traced operand
    # sandwiched by closures. Closures from `jax.linearize` are
    # structurally zero, so the result is `op.pad_rows(left, right)`.
    # `pad_rows` preserves the transposed flag and pads `out_size`,
    # which is the primal-output axis under either flag.
    closures_1d = all(
        len(v.shape) == 1
        for v, t in zip(invals, traced) if not t and hasattr(v, "shape")
    )
    fast_path_dim = (dimension == 0) or (dimension == 1 and closures_1d)
    if fast_path_dim and len(traced_idxs) == 1:
        idx = traced_idxs[0]
        op = invals[idx]
        left_total = sum(int(invals[i].shape[0]) for i in range(idx))
        right_total = sum(int(invals[i].shape[0])
                          for i in range(idx + 1, len(invals)))
        # Promote symmetric LinOps to BE. ConstantDiagonal/Diagonal
        # are symmetric so dense rendering is identical either way; use
        # `transposed=True` (V at axis 0) when entering via vmap rewrite
        # so downstream rules see the canonical V-at-0 layout.
        promote_transposed = (dimension == 1)
        if isinstance(op, ConstantDiagonal):
            op = BEllpack(
                0, op.n, (np.arange(op.n),),
                jnp.broadcast_to(jnp.asarray(op.data), (op.n,)),
                op.n, op.n, transposed=promote_transposed,
            )
        elif isinstance(op, Diagonal):
            op = BEllpack(
                0, op.n, (np.arange(op.n),), op.data, op.n, op.n,
                transposed=promote_transposed,
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
    # All-traced BCOO concat: delegate to `sparse.bcoo_concatenate`
    # for an all-sparse result. Closes the LUKSAN11LS-class chain
    # `concat(BCOO(...), BCOO(...)) → BCOO(...)` that previously
    # densified at this rule's dense fallback below. Closures
    # contribute structural zeros, so we'd need to materialize them
    # as BCOO too — keep this path narrow to all-traced for now.
    if (len(traced_idxs) == len(invals)
            and all(isinstance(v, sparse.BCOO) for v in invals)
            and all(v.n_batch == invals[0].n_batch for v in invals[1:])):
        return sparse.bcoo_concatenate(
            list(invals), dimension=dimension,
        )

    # Fallback: densify everything. When any traced operand has V at
    # axis 0 (densified-from-transposed-BE chain — detected by
    # `shape[0]==n, shape[-1]!=n`), build closure zeros with V at 0 to
    # match. Otherwise V at -1 (Phase B default).
    raw_traced = [v.todense() if isinstance(v, LinOpProtocol) else v
                  for v, t in zip(invals, traced) if t]
    v_at_zero = any(d.ndim >= 2 and d.shape[0] == n and d.shape[-1] != n
                    for d in raw_traced)
    parts = []
    traced_iter = iter(raw_traced)
    for v, t in zip(invals, traced):
        if t:
            parts.append(next(traced_iter))
        else:
            zero_shape = (n,) + v.shape if v_at_zero else v.shape + (n,)
            parts.append(jnp.broadcast_to(v[None] * 0 if v_at_zero
                                          else v[..., None] * 0, zero_shape))
    return lax.concatenate(parts, dimension)
