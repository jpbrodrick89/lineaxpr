"""BEllpack registrations for gather_op and scatter_add_op.

Extracted from _rules/indexing.py; singledispatch registrations extend the
base functions defined in _linops/base.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from jax.experimental import sparse

from .base import LinOpProtocol, gather_op, scatter_add_op
from .bcoo_extend import _bcoo_concat
from .ellpack import BEllpack, _bellpack_unbatch, ColArr


@gather_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, start_indices, **params):
    dnums = params["dimension_numbers"]

    # 2D point-gather on batched BEllpack (n_batch==1).
    if (dnums.offset_dims == ()
            and dnums.collapsed_slice_dims == (0, 1)
            and dnums.start_index_map == (0, 1)
            and params["slice_sizes"] == (1, 1)):
        if (op.n_batch == 1
                and op.start_row == 0
                and op.end_row == op.out_size
                and start_indices.ndim >= 2):
            leading = start_indices.shape[:-1]
            new_batch = leading[:-1]
            new_out = leading[-1]
            idx_static = isinstance(start_indices, np.ndarray)
            if idx_static:
                row_flat = start_indices[..., 0].reshape(-1)
                col_flat = start_indices[..., 1].reshape(-1)
            else:
                row_flat = jnp.asarray(start_indices[..., 0]).reshape(-1)
                col_flat = jnp.asarray(start_indices[..., 1]).reshape(-1)
            from lineaxpr._rules.multilinear import _resolve_full  # noqa: PLC0415
            new_in_cols: list[ColArr] = []
            for c in op.in_cols:
                c_full = _resolve_full(c, op.nrows, op.batch_shape)
                if not idx_static and isinstance(c_full, np.ndarray):
                    c_full = jnp.asarray(c_full)
                new_in_cols.append(c_full[row_flat, col_flat].reshape(leading))
            vals = op.data[row_flat, col_flat]
            if op.k == 1:
                vals = vals.reshape(leading)
            else:
                vals = vals.reshape(leading + (op.k,))
            return BEllpack(
                start_row=0, end_row=new_out,
                in_cols=tuple(new_in_cols), data=vals,
                out_size=new_out, in_size=op.in_size,
                batch_shape=new_batch,
            )
        dense = op.todense()
        row_idx = start_indices[..., 0]
        col_idx = start_indices[..., 1]
        return dense[row_idx, col_idx]

    # 1-D gather on unbatched BEllpack.
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
    # V-augmented gather (post jax.vmap, V at axis 0): the operand is a
    # `transposed=True` BE whose dense shape is (V=in_size, out_size).
    # The gather collapses the out_size axis (axis 1) and keeps V (axis
    # 0) as offset_dims. Structurally this gathers rows of the (out,
    # in)-canonical Jacobian — same semantics as point_gather_collapsed
    # but the dnums describe the V-augmented operand layout.
    sl = params["slice_sizes"]
    point_gather_v_collapsed_T = (
        op.transposed
        and op.n_batch == 0
        and tuple(dnums.offset_dims) == (0,)
        and tuple(dnums.collapsed_slice_dims) == (1,)
        and tuple(dnums.start_index_map) == (1,)
        and len(sl) == 2 and sl[0] == op.in_size and sl[1] == 1
    )
    if not (point_gather_collapsed or point_gather_kept
            or point_gather_v_collapsed_T):
        # Unhandled dnums — densify and let `lax.gather` handle it.
        return lax.gather(
            op.todense(), start_indices, **params)

    if op.n_batch == 0:
        row_idx = start_indices[..., 0]
        batch_shape = tuple(row_idx.shape[:-1])
        N = row_idx.shape[-1]
        idx_static = isinstance(row_idx, np.ndarray)
        ridx_flat = row_idx.reshape(-1) if idx_static else jnp.asarray(
            row_idx).reshape(-1)
        # Padded-range support: rows outside [start_row, end_row) are
        # structurally zero. Clamp the lookup index and mask out-of-range
        # elements to zero.
        nrows = op.end_row - op.start_row
        local = ridx_flat - op.start_row
        if op.start_row == 0 and op.end_row == op.out_size:
            local_clipped = local
            mask = None
        else:
            clip_hi = max(nrows - 1, 0)
            if idx_static:
                local_clipped = np.clip(local, 0, clip_hi)
            else:
                local_clipped = jnp.clip(local, 0, clip_hi)
            mask = ((ridx_flat >= op.start_row) & (ridx_flat < op.end_row))
        new_in_cols: list[ColArr] = []
        for c in op.in_cols:
            cr = c
            if idx_static and isinstance(cr, np.ndarray):
                gathered = cr[local_clipped].reshape(batch_shape + (N,))
            else:
                gathered = jnp.asarray(cr)[local_clipped].reshape(
                    batch_shape + (N,))
            new_in_cols.append(gathered)
        if op.k == 1:
            vals = op.data[local_clipped].reshape(batch_shape + (N,))
        else:
            vals = op.data[local_clipped].reshape(batch_shape + (N, op.k))
        if mask is not None:
            mask_r = mask.reshape(batch_shape + (N,))
            if op.k == 1:
                vals = jnp.where(mask_r, vals, 0)
            else:
                vals = jnp.where(mask_r[..., None], vals, 0)
        if point_gather_kept:
            return BEllpack(
                start_row=0, end_row=1,
                in_cols=tuple(c[..., None] for c in new_in_cols),
                data=vals[..., None] if op.k == 1 else vals[..., None, :],
                out_size=1, in_size=op.in_size,
                batch_shape=batch_shape + (N,),
            )
        return BEllpack(
            start_row=0, end_row=N,
            in_cols=tuple(new_in_cols), data=vals,
            out_size=N, in_size=op.in_size,
            batch_shape=batch_shape,
            transposed=op.transposed,
        )

    # Dense fallback for other BEllpack gather patterns.
    row_idx = start_indices[..., 0]
    dense = op.todense()
    if point_gather_kept:
        return dense[row_idx][..., None, :]
    return dense[row_idx]


@scatter_add_op.register(BEllpack)
def _(updates, *, n, operand, scatter_indices, **params):
    dnums = params["dimension_numbers"]

    # 2D point-scatter: handled via dense fallback in base.
    if (dnums.update_window_dims == ()
            and dnums.inserted_window_dims == (0, 1)
            and dnums.scatter_dims_to_operand_dims == (0, 1)
            and operand.ndim == 2):
        operand_dense = jnp.asarray(operand)
        out_shape_2d = operand_dense.shape
        updates_dense = updates.todense()
        si_flat = scatter_indices.reshape(-1, 2)
        updates_flat = updates_dense.reshape(-1, n)
        flat_rows = (
            si_flat[:, 0].astype(jnp.int64) * out_shape_2d[1]
            + si_flat[:, 1].astype(jnp.int64)
        )
        out_size_flat = out_shape_2d[0] * out_shape_2d[1]
        return (
            jnp.zeros((out_size_flat, n), updates_flat.dtype)
            .at[flat_rows]
            .add(updates_flat)
            .reshape(out_shape_2d + (n,))
        )

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
        raise NotImplementedError(
            f"scatter-add with unhandled dnums: {dnums}")

    # Canonicalise the "kept" form by squeezing the trailing size-1 axis.
    if scatter_kept:
        if (updates.out_size == 1
                and updates.start_row == 0
                and updates.end_row == 1
                and len(updates.batch_shape) >= 1):
            inner = updates.batch_shape[-1]
            new_batch = updates.batch_shape[:-1]
            squeezed_cols = tuple(
                c.reshape(c.shape[:-1]) for c in updates.in_cols
            )
            if updates.k == 1:
                new_values = updates.data.reshape(updates.data.shape[:-1])
            else:
                shape = updates.data.shape
                new_values = updates.data.reshape(shape[:-2] + shape[-1:])
            updates = BEllpack(
                start_row=0, end_row=inner,
                in_cols=squeezed_cols, data=new_values,
                out_size=inner, in_size=updates.in_size,
                batch_shape=new_batch,
            )
        else:
            updates = updates.todense().squeeze(axis=-2)
            # Now updates is a plain array — fall through to dense path.

    out_idx = scatter_indices[..., 0]
    out_size = operand.shape[0]

    if not isinstance(updates, BEllpack):
        # Dense fallback (updates was converted above or was BCOO).
        out_idx_flat = out_idx.reshape(-1)
        updates_dense = (updates.todense() if isinstance(updates, LinOpProtocol)  # pyrefly: ignore [unsafe-overlap]
                         else updates)
        flat_updates = updates_dense.reshape(-1, n)
        return (jnp.zeros((out_size, n), flat_updates.dtype)
                .at[out_idx_flat].add(flat_updates))

    # BEllpack batched case.
    was_transposed = False
    if updates.n_batch == 0:
        # `to_bcoo()` is flag-aware: for T=True it produces the LOGICAL
        # `(V, struct)` view, matching the dense `scatter_add_op`
        # rule's V-at-0 output convention. For T=False it produces
        # canonical `(struct, V)` (matches legacy behaviour).
        was_transposed = updates.transposed
        updates = updates.to_bcoo()
    else:
        from lineaxpr._rules.add import _add_rule  # noqa: PLC0415
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
            if out_idx_b.shape[0] != nrows_b:
                static_ok = False
                break
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
            order = np.argsort(inverse, kind="stable").reshape(n_unique, dup)
            try:
                new_in_cols: list[ColArr] = []
                for d in range(dup):
                    src = order[:, d]
                    for col in ep.in_cols:
                        col_np = col
                        if not isinstance(col_np, np.ndarray):
                            col_np = np.asarray(col_np)
                        new_in_cols.append(col_np[src].astype(np.intp))
            except (jax.errors.TracerArrayConversionError, TypeError):
                static_ok = False
                break
            if ep.k == 1:
                new_vals = ep.data[order]
            else:
                new_vals = ep.data[order].reshape(n_unique, dup * ep.k)
            start = int(unique_out[0])
            be_pieces.append(BEllpack(
                start, start + n_unique, tuple(new_in_cols), new_vals,
                out_size, ep.in_size,
                transposed=updates.transposed,
            ))
        if static_ok:
            return _add_rule(be_pieces, [True] * len(be_pieces), n)
        # Traced indices fallback. For T=True BE input, `to_bcoo()`
        # produces BCOO at `(in, out)` ordering — the OUT axis is at
        # index 1 of `indices`. For T=False, OUT is at index 0. Match
        # the V-at-0 post-vmap BCOO convention by always emitting at
        # `(in, out)` ordering for T=True inputs (and `(out, in)` for
        # T=False), and route the scatter through the OUT axis.
        bcoo_pieces = []
        for b_idx, ep in enumerate(slices):
            bc = ep.to_bcoo()
            if ep.transposed:
                # `(in, out)` ordering: OUT at axis 1.
                old_out = bc.indices[:, 1]
                new_out = out_idx[b_idx][old_out]
                new_indices = jnp.stack(
                    [bc.indices[:, 0], new_out], axis=1)
                new_shape = (updates.in_size, out_size)
            else:
                # `(out, in)` ordering: OUT at axis 0.
                old_out = bc.indices[:, 0]
                new_out = out_idx[b_idx][old_out]
                new_indices = jnp.stack(
                    [new_out, bc.indices[:, 1]], axis=1)
                new_shape = (out_size, updates.in_size)
            bcoo_pieces.append(sparse.BCOO(
                (bc.data, new_indices), shape=new_shape,
            ))
        return _bcoo_concat(
            bcoo_pieces,
            shape=bcoo_pieces[0].shape if bcoo_pieces
            else (out_size, updates.in_size),
        )

    # BCOO (converted from unbatched BEllpack above). For T=True BE
    # input, BCOO is at LOGICAL `(V, struct)` layout — axis 1 is
    # struct, gets remapped via `out_idx`; result is `(V=n, out_size)`
    # matching the dense path. For T=False input, BCOO is at canonical
    # `(struct, V)` — axis 0 is struct.
    if isinstance(updates, sparse.BCOO):
        out_idx_flat = out_idx.reshape(-1)
        if was_transposed:
            v_idx = updates.indices[:, 0]
            struct_idx = updates.indices[:, 1]
            new_struct = out_idx_flat[struct_idx]
            new_indices = jnp.stack([v_idx, new_struct], axis=1)
            return sparse.BCOO(
                (updates.data, new_indices), shape=(n, out_size)
            )
        new_rows = out_idx_flat[updates.indices[:, 0]]
        new_indices = jnp.stack([new_rows, updates.indices[:, 1]], axis=1)
        return sparse.BCOO(
            (updates.data, new_indices), shape=(out_size, n)
        )

    # Plain array fallback.
    out_idx_flat = out_idx.reshape(-1)
    flat_updates = jnp.asarray(updates).reshape(-1, n)  # pyrefly: ignore [bad-argument-type]
    return (jnp.zeros((out_size, n), flat_updates.dtype)
            .at[out_idx_flat].add(flat_updates))
