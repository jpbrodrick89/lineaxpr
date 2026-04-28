"""Gather and scatter-add structural rules."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from .._linops import (
    BEllpack,
    ConstantDiagonal,
    Diagonal,
    _resolve_col,
    _to_bcoo,
    _to_dense,
)
from .add import _add_rule, _bcoo_concat, _bellpack_unbatch
from .multilinear import _resolve_full


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
