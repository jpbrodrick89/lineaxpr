"""BEllpack registrations for reshape, broadcast_in_dim, reduce_sum, cumsum.

Extracted from _rules/unary.py; singledispatch registrations extend the
base functions defined in _linops/base.py.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import sparse

from .base import (
    broadcast_in_dim_op,
    reduce_sum_op,
    reshape_op,
    split_op,
)
from .dense import _bid_with_extra_batch
from .ellpack import BEllpack, _bellpack_unbatch, ColArr


# ---------------------------------------------------------------------------
# reshape_op registrations
# ---------------------------------------------------------------------------

@reshape_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    # T=True unbatched → batched: `(in, total) → (in, *new_batch, out)`
    # where `total == prod(new_batch) * out`. Inverts the unbatching
    # reshape below (LUKSAN11LS: `BE_T(100, 198) k=2 → (100, 99, 2)`).
    if (op.transposed and op.n_batch == 0
            and op.start_row == 0 and op.end_row == op.out_size):
        full_new = tuple(int(s) for s in params["new_sizes"])
        if (len(full_new) >= 3
                and full_new[0] == op.in_size
                and int(np.prod(full_new[1:])) == op.out_size):
            new_batch = full_new[1:-1]
            new_out = full_new[-1]
            target_data_shape = new_batch + (new_out,)
            if op.k == 1:
                new_values = op.data.reshape(target_data_shape)
            else:
                new_values = op.data.reshape(target_data_shape + (op.k,))
            new_in_cols: list[ColArr] = []
            for c in op.in_cols:
                if isinstance(c, np.ndarray):
                    new_in_cols.append(c.reshape(target_data_shape))
                elif isinstance(c, slice):
                    rs = np.arange(
                        c.start or 0, c.stop or op.nrows, c.step or 1)
                    new_in_cols.append(rs.reshape(target_data_shape))
                else:
                    new_in_cols.append(jnp.asarray(c).reshape(
                        target_data_shape))
            return BEllpack(
                start_row=0, end_row=new_out,
                in_cols=tuple(new_in_cols), data=new_values,
                out_size=new_out, in_size=op.in_size,
                batch_shape=new_batch, transposed=True,
            )
    # T=True (V-at-0): `params["new_sizes"]` is the full V-augmented
    # target shape WITH V at axis 0 (no walk-frame strip). Flatten
    # `*batch, out_size → new out_size` while keeping in_size at axis 0.
    if op.transposed and op.n_batch >= 1:
        full_new = tuple(int(s) for s in params["new_sizes"])
        # Expected output of an unbatching reshape: (in_size, total) where
        # total = prod(batch) * out_size.
        if (len(full_new) == 2
                and full_new[0] == op.in_size
                and full_new[1] == int(np.prod(op.batch_shape)) * op.out_size):
            total = full_new[1]
            nrows = total
            if op.k == 1:
                new_values = op.data.reshape(nrows)
            else:
                new_values = op.data.reshape(nrows, op.k)
            new_in_cols: list[ColArr] = []
            for c in op.in_cols:
                if isinstance(c, np.ndarray):
                    if c.ndim == 1:
                        # 1D shared cols — broadcast across batch first,
                        # then reshape.
                        c_full = np.broadcast_to(
                            c, op.batch_shape + (op.nrows,))
                        new_in_cols.append(c_full.reshape(nrows))
                    else:
                        new_in_cols.append(c.reshape(nrows))
                elif isinstance(c, slice):
                    rs = np.arange(
                        c.start or 0, c.stop or op.nrows, c.step or 1)
                    c_full = np.broadcast_to(
                        rs, op.batch_shape + (op.nrows,))
                    new_in_cols.append(c_full.reshape(nrows))
                else:
                    ca = jnp.asarray(c)
                    if ca.ndim == 1:
                        ca = jnp.broadcast_to(
                            ca, op.batch_shape + (op.nrows,))
                    new_in_cols.append(ca.reshape(nrows))
            return BEllpack(
                start_row=0, end_row=nrows,
                in_cols=tuple(new_in_cols), data=new_values,
                out_size=nrows, in_size=op.in_size,
                transposed=True,
            )
    # Walk-frame new_sizes has n at -1; structural shape is the prefix.
    new_sizes = params["new_sizes"][:-1]

    # Pass-through: unbatched BCOO already the target shape.
    # (handled separately via BCOO base; this path won't fire for BEllpack)

    # Batched BEllpack → unbatched BEllpack: flatten leading (batch + out).
    if (op.n_batch >= 1
            and len(new_sizes) == 1
            and int(np.prod(op.batch_shape)) * op.out_size == int(new_sizes[0])):
        prod_b = int(np.prod(op.batch_shape))
        total = prod_b * op.out_size
        if op.k == 1:
            new_values = op.data.reshape(total)
        else:
            new_values = op.data.reshape(total, op.k)
        new_in_cols: list[ColArr] = []
        for c in op.in_cols:
            if isinstance(c, slice):
                rs = np.arange(c.start or 0, c.stop or op.nrows, c.step or 1)
                c_full = np.broadcast_to(rs, op.batch_shape + (op.nrows,))
                new_in_cols.append(c_full.reshape(total))
            elif isinstance(c, np.ndarray):
                if c.ndim == 1:
                    c_full = np.broadcast_to(c, op.batch_shape + (op.nrows,))
                    new_in_cols.append(c_full.reshape(total))
                else:
                    new_in_cols.append(c.reshape(total))
            else:
                ca = jnp.asarray(c)
                if ca.ndim == 1:
                    ca = jnp.broadcast_to(ca, op.batch_shape + (op.nrows,))
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(ca.reshape(total))
        return BEllpack(
            start_row=0, end_row=total,
            in_cols=tuple(new_in_cols), data=new_values,
            out_size=total, in_size=op.in_size,
        )

    # Unbatched BEllpack (N,) → (A, B): unflatten into batched BE.
    if (op.n_batch == 0
            and len(new_sizes) == 2
            and int(new_sizes[0]) * int(new_sizes[1]) == op.out_size
            and int(new_sizes[1]) > 1
            and op.start_row == 0 and op.end_row == op.out_size):
        A = int(new_sizes[0])
        B_out = int(new_sizes[1])
        new_batch = (A,)
        if op.k == 1:
            new_values = op.data.reshape(A, B_out)
        else:
            new_values = op.data.reshape(A, B_out, op.k)
        new_in_cols: list[ColArr] = []
        for c in op.in_cols:
            if isinstance(c, slice):
                rs = np.arange(c.start or 0, c.stop or op.nrows, c.step or 1)
                new_in_cols.append(rs.reshape(A, B_out))
            elif isinstance(c, np.ndarray):
                new_in_cols.append(c.reshape(A, B_out))
            else:
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(jnp.asarray(c).reshape(A, B_out))
        return BEllpack(
            start_row=0, end_row=B_out,
            in_cols=tuple(new_in_cols), data=new_values,
            out_size=B_out, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Leading-singleton reshape: (N,) → (1, ..., 1, N).
    if (op.n_batch == 0
            and len(new_sizes) >= 2
            and new_sizes[-1] == op.out_size
            and all(s == 1 for s in new_sizes[:-1])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(new_sizes[:-1])
        prefix = (1,) * len(new_batch)
        if op.k == 1:
            new_values = op.data.reshape(prefix + (op.nrows,))
        else:
            new_values = op.data.reshape(prefix + (op.nrows, op.k))
        new_in_cols: list[ColArr] = []
        for c in op.in_cols:
            if isinstance(c, np.ndarray):
                if c.ndim == 1:
                    # pyrefly: ignore [bad-argument-type]
                    new_in_cols.append(c)
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
            in_cols=tuple(new_in_cols), data=new_values,
            out_size=op.out_size, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Singleton-axis-insert: (N,) → (N, 1, ..., 1).
    if (op.n_batch == 0
            and len(new_sizes) >= 2
            and new_sizes[0] == op.out_size
            and all(s == 1 for s in new_sizes[1:])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(new_sizes[:-1])
        if op.k == 1:
            new_values = op.data.reshape(new_batch + (1,))
        else:
            new_values = op.data.reshape(new_batch + (1, op.k))
        new_in_cols: list[ColArr] = []
        for c in op.in_cols:
            if isinstance(c, np.ndarray):
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(c.reshape(new_batch + (1,) + c.shape[1:]))
            else:
                # pyrefly: ignore [bad-argument-type]
                new_in_cols.append(jnp.asarray(c).reshape(
                    new_batch + (1,) + c.shape[1:]))
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=tuple(new_in_cols), data=new_values,
            out_size=1, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Dense fallback.
    return lax.reshape(op.todense(), params["new_sizes"],
                       dimensions=params.get("dimensions"),
                       out_sharding=params.get("sharding"))


# BCOO batched → flat reshape registration.
@reshape_op.register(sparse.BCOO)
def _(op, *, n, **params):
    full_sizes = tuple(params["new_sizes"])
    new_sizes = full_sizes[:-1]

    # Pass-through: already the target shape.
    if (op.n_batch == 0
            and len(new_sizes) == 1
            and op.shape == (int(new_sizes[0]), op.shape[-1])):
        return op

    # General-purpose path via `sparse.bcoo_reshape`. Try with the
    # full `new_sizes` first (V-at-0 BCOOs and V-at-last BCOOs both
    # ship with the V dim already in `new_sizes`), then fall back to
    # the legacy walk-frame stripped form. Wrapped in try/except
    # because `bcoo_reshape` raises on permutations that mix batch
    # with sparse axes.
    for candidate in (full_sizes, new_sizes):
        if op.shape == candidate:
            return op
        try:
            res = sparse.bcoo_reshape(op, new_sizes=candidate)
            if res.shape == candidate:
                return res
        except (NotImplementedError, ValueError):
            pass

    # Flatten batched BCOO's leading (batch + out) axes.
    if (op.n_batch >= 1
            and len(new_sizes) == 1
            and int(np.prod(op.shape[:-1])) == int(new_sizes[0])):
        nb = op.n_batch
        old_out = op.shape[nb]
        in_size = op.shape[-1]
        batch_total = int(np.prod(op.shape[:nb]))
        nse_per_batch = op.data.shape[nb]
        flat_data = op.data.reshape(batch_total, nse_per_batch)
        flat_indices = op.indices.reshape(batch_total, nse_per_batch, 2)
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

    return lax.reshape(op.todense(), params["new_sizes"],
                       dimensions=params.get("dimensions"),
                       out_sharding=params.get("sharding"))


# ---------------------------------------------------------------------------
# broadcast_in_dim_op registrations
# ---------------------------------------------------------------------------

@broadcast_in_dim_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    # Post-no-op-squeeze 1-row BE (transposed=True, out_size=1) carries
    # logical-1D semantics in 2D form. The rewritten jaxpr's
    # broadcast_in_dim params describe a 1D operand input (the squeezed
    # form), but our BE has shape (in, 1).
    full_bd = tuple(params["broadcast_dimensions"])
    full_shape = tuple(params["shape"])
    if (op.transposed and op.n_batch == 0
            and op.out_size == 1
            and op.start_row == 0 and op.end_row == 1
            and len(full_bd) == 1 and full_bd[0] == 0
            and len(full_shape) == 2 and full_shape[0] == op.in_size):
        # Output shape (in_size, N) with bd=(0,) means: tile the
        # 1D-semantic row vector to N rows. Each output row is a copy
        # of the BE's single canonical row — broadcast values & cols
        # along the new out_size axis.
        N = int(full_shape[1])
        if N == 1:
            return op  # (in, 1) → (in, 1): no-op.
        new_values = jnp.broadcast_to(
            op.data, (N,) + op.data.shape[1:]
        )
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, np.ndarray):
                new_in_cols.append(np.broadcast_to(c, (N,)).copy())
            else:
                new_in_cols.append(jnp.broadcast_to(jnp.asarray(c), (N,)))
        return BEllpack(
            start_row=0, end_row=N,
            in_cols=tuple(new_in_cols), data=new_values,
            out_size=N, in_size=op.in_size, transposed=True,
        )
    # transposed=True BE with broadcast that only adds trailing
    # size-1 axes (LUKSAN-class): emit a batched BE_T whose new
    # `batch_shape` absorbs the original out_size and the trailing
    # ones absorb the new singletons. Keeps everything as BE so
    # downstream `concat / reshape / split / reduce_sum` rules walk
    # via the BE-structural paths (band dedup) instead of the BCOO
    # paths (which accumulate sentinel entries).
    if (op.transposed
            and op.n_batch == 0
            and tuple(full_bd) == tuple(range(len(op.shape)))
            and full_shape[:len(op.shape)] == tuple(op.shape)
            and all(s == 1 for s in full_shape[len(op.shape):])):
        # Original 2D BE_T: shape (in_size, out_size). Goal: 3D shape
        # (in_size, out_size, 1), interpreted as
        # `batch=(out_size,), in=in_size, out=1`.
        n_extra = len(full_shape) - 2
        new_batch = (op.out_size,) + (1,) * (n_extra - 1)
        if op.k == 1:
            new_data = op.data.reshape(op.out_size, *([1] * n_extra))
        else:
            new_data = op.data.reshape(op.out_size, *([1] * n_extra), op.k)
        new_in_cols = []
        for c in op.in_cols:
            if isinstance(c, np.ndarray):
                new_in_cols.append(
                    c.reshape((op.out_size,) + (1,) * (n_extra - 1) + (1,))
                )
            else:
                new_in_cols.append(jnp.asarray(c).reshape(
                    (op.out_size,) + (1,) * (n_extra - 1) + (1,)
                ))
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=tuple(new_in_cols), data=new_data,
            out_size=1, in_size=op.in_size,
            batch_shape=new_batch, transposed=True,
        )
    # Inside-vmap (transposed=True): V is at axis 0 in the jaxpr-frame
    # operand (externally), but BE's batch-at-front structure can't
    # natively represent V-at-front for rank > 2 outputs. Densify via
    # op.todense() (which puts V at 0 for transposed 2D BE) and use
    # lax.broadcast_in_dim — params apply unchanged. Only arises in
    # 1D→ND primal cases (the bcast+prim grid); sif2jax stays 2D and
    # falls through to the structural branches below with transposed=False.
    if op.transposed:
        # Insert a new size>1 batch axis between the V (axis 0) and
        # OUT (last) axes of an unbatched T=True BE: shape (in, out)
        # → (in, B, out) with bd=(0, 2). Structurally equivalent to
        # batched T=True BE with batch=(B,) and the same in_cols /
        # data broadcast across the batch. Triggered by the
        # `_SYN_SCATTER_COMPACT_DUP`-class chain
        # `gather → BE_T → reduce_sum → ... → broadcast_in_dim → scatter`.
        # Insert a new size>1 batch axis between the V (axis 0) and
        # OUT (last) axes of an unbatched T=True BE: shape (in, out)
        # → (in, B, out) with bd=(0, 2). Structurally equivalent to
        # batched T=True BE with batch=(B,) and the same in_cols /
        # data broadcast across the batch. Triggered by the
        # `_SYN_SCATTER_COMPACT_DUP`-class chain
        # `gather → BE_T → reduce_sum → ... → broadcast_in_dim → scatter`.
        # Size-1 placeholder middle dims (LUKSAN16LS / SPARSINE-class:
        # `(in, 1, ..., 1, out)`) need different downstream handling
        # — turning those into a batched BE_T breaks subsequent muls
        # that broadcast against the size-1 dims. Densify those
        # patterns instead via the fallback below.
        full_shape_t = tuple(params["shape"])
        full_bd_t = tuple(params["broadcast_dimensions"])
        if (op.n_batch == 0
                and len(full_shape_t) == 3
                and len(full_bd_t) == 2
                and full_bd_t[0] == 0
                and full_bd_t[1] == 2
                and full_shape_t[0] == op.shape[0]
                and full_shape_t[-1] == op.shape[1]
                and full_shape_t[1] > 1):
            new_batch = (int(full_shape_t[1]),)
            new_data = jnp.broadcast_to(op.data, new_batch + op.data.shape)
            new_in_cols: list[ColArr] = []
            for c in op.in_cols:
                if isinstance(c, np.ndarray):
                    new_in_cols.append(np.broadcast_to(c, new_batch + c.shape).copy())
                else:
                    new_in_cols.append(jnp.broadcast_to(jnp.asarray(c),
                                                       new_batch + jnp.asarray(c).shape))
            return BEllpack(
                start_row=op.start_row, end_row=op.end_row,
                in_cols=tuple(new_in_cols), data=new_data,
                out_size=op.out_size, in_size=op.in_size,
                batch_shape=new_batch, transposed=True,
            )
        # Trailing tile pattern: `BE_T(in, out)` → `(in, out, B)` with
        # `bd=(0, 1)`. The new trailing dim is a tile (size B>=1) —
        # each (V, OUT) entry replicated B times. Emit a batched
        # T=True BE that re-labels the original out as a batch and
        # the new trailing dim as the new out: shape
        # `(in, *batch, out)` = `(in, orig_out, B)`. The data and
        # in_cols are broadcast across the new out axis. Triggered by
        # YATP1LS-class chains: `gather/slice/transpose → BE_T(in, k)
        # → broadcast_in_dim` where `bd[1]=1, len(shape)=3,
        # shape[1]==op.shape[1]`.
        if (op.n_batch == 0
                and len(full_shape_t) == 3
                and len(full_bd_t) == 2
                and full_bd_t[0] == 0
                and full_bd_t[1] == 1
                and full_shape_t[0] == op.shape[0]
                and full_shape_t[1] == op.shape[1]):
            B = int(full_shape_t[2])
            new_batch = (int(op.shape[1]),)  # original out becomes batch
            # Data: (out, k) for k>=2 unbatched → (out, B, k) batched
            # by broadcasting the new trailing tile axis. For k=1:
            # (out,) → (out, B).
            if op.k == 1:
                expanded = jnp.expand_dims(op.data, axis=-1)
                new_data = jnp.broadcast_to(expanded, op.data.shape + (B,))
            else:
                # data shape (out, k) → (out, B, k)
                expanded = jnp.expand_dims(op.data, axis=-2)
                target = op.data.shape[:-1] + (B,) + op.data.shape[-1:]
                new_data = jnp.broadcast_to(expanded, target)
            new_in_cols: list[ColArr] = []
            for c in op.in_cols:
                if isinstance(c, np.ndarray):
                    expanded_c = c.reshape(c.shape + (1,))
                    new_in_cols.append(np.broadcast_to(
                        expanded_c, c.shape + (B,)).copy())
                else:
                    c_arr = jnp.asarray(c)
                    expanded_c = c_arr.reshape(c_arr.shape + (1,))
                    new_in_cols.append(jnp.broadcast_to(
                        expanded_c, c_arr.shape + (B,)))
            return BEllpack(
                start_row=0, end_row=B,
                in_cols=tuple(new_in_cols), data=new_data,
                out_size=B, in_size=op.in_size,
                batch_shape=new_batch, transposed=True,
            )
        return lax.broadcast_in_dim(
            op.todense(),
            tuple(params["shape"]),
            tuple(params["broadcast_dimensions"]),
        )

    # Inside-vmap untransposed row-vector BE (out_size=1): the strip
    # below assumes V at output axis -1, but row-vector chains routed
    # through transpose+squeeze+mul end up here with V at output axis
    # bd[0] != ndim_out - 1 (e.g. arrowhead's `bcast(q:(3,), bd=(0,),
    # shape=(3,1))` where V stays at axis 0). The strip then mis-tiles
    # the singleton primal axis. Sticking plaster: densify and use
    # direct lax.broadcast_in_dim. Loses any sparsity we'd have
    # recovered structurally, but fixes the wrong-shape output.
    full_bd = tuple(params["broadcast_dimensions"])
    full_shape = tuple(params["shape"])
    # Inside-vmap row-vector pattern (e.g. arrowhead's
    # `bcast(q:(3,), bd=(0,), shape=(3,1))`): jaxpr-frame input is 1D
    # (V only — the row vector's scalar primal output collapses to no
    # axis in jaxpr) and bd[0] != ndim_out - 1 means V is NOT at the
    # output's trailing axis. The `[:-1]` strip below would remove the
    # wrong slot (the new singleton primal axis instead of V's slot)
    # and the structural branches would mis-tile. Sticking plaster:
    # densify and let lax.broadcast_in_dim handle it.
    if (op.out_size == 1 and op.n_batch == 0
            and op.start_row == 0 and op.end_row == 1
            and len(full_bd) == 1
            and full_bd[0] != len(full_shape) - 1):
        # Inside-vmap row-vector BE viewed as 1D under jaxpr-frame:
        # the BE's `(1, in_size)` shape represents a sparse vector of
        # length `in_size`, and the bcast spreads it across an output
        # whose `bd[0]` axis takes that vector's values. Structurally:
        # build a BCOO of the output shape with one entry per non-
        # sentinel band, placed at `axis=bd[0]` and zero on every
        # other axis. Static cols → static indices (no sentinel
        # contamination); traced cols fall through to dense.
        if all(isinstance(c, np.ndarray) and c.ndim == 1 and c.shape[0] == 1
               for c in op.in_cols):
            # k bands × 1 row → at most k nonzeros
            cols_concat = np.concatenate(
                [c.reshape(1) for c in op.in_cols]
            )
            valid_idx = np.where(cols_concat >= 0)[0]
            valid_cols = cols_concat[valid_idx]
            # `op.data` is traced under jit. `valid_idx` is static, so
            # gather via integer indexing rather than boolean indexing.
            vals_concat = jnp.asarray(op.data).reshape(-1)
            valid_vals = vals_concat[jnp.asarray(valid_idx)]
            ndim = len(full_shape)
            ax = int(full_bd[0])
            indices = np.zeros((valid_idx.shape[0], ndim), np.intp)
            indices[:, ax] = valid_cols
            return sparse.BCOO(
                (valid_vals, jnp.asarray(indices)), shape=full_shape,
            )
        return lax.broadcast_in_dim(
            op.todense().squeeze(axis=0), full_shape, full_bd
        )

    # Walk-frame (transposed=False): shape ends in n, bd ends in n's
    # mapping. Strip both for the spatial-only structural checks below.
    shape = full_shape[:-1]
    broadcast_dimensions = full_bd[:-1]

    # Linear form (aval ()) broadcast to shape (1,): pass through BE row-vector.
    if (broadcast_dimensions == () and tuple(shape) == (1,)
            and op.n_batch == 0 and op.out_size == 1
            and op.start_row == 0 and op.end_row == 1):
        return op

    # Tile 1-row BEllpack to N-row vector via empty broadcast_dimensions.
    if (op.n_batch == 0
            and op.out_size == 1 and op.start_row == 0 and op.end_row == 1
            and broadcast_dimensions == () and len(shape) == 1):
        N = int(shape[0])
        new_values = jnp.broadcast_to(op.data, (N,) + op.data.shape[1:])
        new_in_cols: list[ColArr] = []
        for c in op.in_cols:
            if isinstance(c, np.ndarray):
                new_in_cols.append(np.broadcast_to(c, (N,)).copy())
            else:
                new_in_cols.append(jnp.broadcast_to(jnp.asarray(c), (N,)))
        return BEllpack(
            start_row=0, end_row=N,
            in_cols=tuple(new_in_cols), data=new_values,
            out_size=N, in_size=op.in_size,
        )

    # Fallback normalisation: BEllpack row-vector (aval ()) with non-trivial bd.
    if (op.n_batch == 0
            and op.out_size == 1 and op.start_row == 0 and op.end_row == 1
            and broadcast_dimensions == ()):
        op_dense = op.todense()[0]
        expected_ndim = len(broadcast_dimensions) + 1
        while op_dense.ndim > expected_ndim and op_dense.shape[0] == 1:
            op_dense = op_dense[0]
        out_dims = tuple(broadcast_dimensions) + (len(shape),)
        return lax.broadcast_in_dim(op_dense, tuple(shape) + (n,), out_dims)

    # Unbatched BE: append broadcast dimensions as new batch axes.
    if (op.n_batch == 0
            and len(broadcast_dimensions) == 1
            and broadcast_dimensions[0] == len(shape) - 1
            and shape[-1] == op.out_size):
        new_batch = tuple(shape[:-1])
        new_values_shape = new_batch + op.data.shape
        new_values = jnp.broadcast_to(op.data, new_values_shape)
        return BEllpack(
            start_row=op.start_row, end_row=op.end_row,
            in_cols=op.in_cols, data=new_values,
            out_size=op.out_size, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Prepend leading batch axes to an already-batched BE.
    input_rank = op.n_batch + 1
    if (len(broadcast_dimensions) == input_rank
            and len(shape) > input_rank
            and broadcast_dimensions == tuple(
                range(len(shape) - input_rank, len(shape)))
            and shape[-1] == op.out_size):
        prepend = tuple(shape[:len(shape) - input_rank])
        new_batch = prepend + op.batch_shape
        new_values = jnp.broadcast_to(op.data, prepend + op.data.shape)
        new_in_cols: list[ColArr] = []
        for c in op.in_cols:
            if c.ndim == 1:
                new_in_cols.append(c)
                continue
            target = prepend + c.shape
            if isinstance(c, np.ndarray):
                new_in_cols.append(np.broadcast_to(c, target))
            else:
                new_in_cols.append(jnp.broadcast_to(c, target))
        return BEllpack(
            start_row=op.start_row, end_row=op.end_row,
            in_cols=tuple(new_in_cols), data=new_values,
            out_size=op.out_size, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Trailing-singleton: unbatched BE aval-(n,) → (n, 1, ..., 1).
    if (op.n_batch == 0
            and len(broadcast_dimensions) == 1
            and broadcast_dimensions[0] == 0
            and len(shape) >= 2
            and shape[0] == op.out_size
            and all(s == 1 for s in shape[1:])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(shape[:-1])
        if op.k == 1:
            new_values = op.data.reshape(new_batch + (1,))
        else:
            new_values = op.data.reshape(new_batch + (1, op.k))
        new_in_cols: list[ColArr] = []
        for c in op.in_cols:
            new_in_cols.append(c.reshape(new_batch + (1,) + c.shape[1:]))
        return BEllpack(
            start_row=0, end_row=1,
            in_cols=tuple(new_in_cols), data=new_values,
            out_size=1, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Leading-dim row-broadcast: unbatched BE out=N → (N, M_1, ..., M_{r-1}).
    if (op.n_batch == 0
            and len(broadcast_dimensions) == 1
            and broadcast_dimensions[0] == 0
            and len(shape) >= 2
            and shape[0] == op.out_size
            and any(s > 1 for s in shape[1:])
            and op.start_row == 0 and op.end_row == op.out_size):
        new_batch = tuple(shape[:-1])
        new_out = int(shape[-1])
        N = op.out_size
        if op.k == 1:
            reshape_shape = (N,) + (1,) * (len(new_batch) - 1) + (1,)
            new_values = jnp.broadcast_to(
                op.data.reshape(reshape_shape),
                new_batch + (new_out,),
            )
        else:
            reshape_shape = (N,) + (1,) * (len(new_batch) - 1) + (1, op.k)
            new_values = jnp.broadcast_to(
                op.data.reshape(reshape_shape),
                new_batch + (new_out, op.k),
            )
        new_in_cols: list[ColArr] = []
        for c in op.in_cols:
            if isinstance(c, np.ndarray):
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
            in_cols=tuple(new_in_cols), data=new_values,
            out_size=new_out, in_size=op.in_size,
            batch_shape=new_batch,
        )

    # Dense fallback.
    return _bid_with_extra_batch(op.todense(), shape, broadcast_dimensions, n)


# ---------------------------------------------------------------------------
# reduce_sum_op registrations
# ---------------------------------------------------------------------------

@reduce_sum_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    axes = params["axes"]
    # Translate V-augmented axes to structural axes (batch + out).
    # transposed=True: V at axis 0, structural axes shift down by 1.
    # transposed=False: V at axis -1, structural axes are 0..n_batch.
    if op.transposed:
        axes = tuple(a - 1 for a in axes)

    # BEllpack with leading batch dims.
    if op.n_batch > 0:
        axes_t = tuple(sorted(axes))
        # Full batch reduction.
        if axes_t == tuple(range(op.n_batch)):
            from lineaxpr._rules.add import _add_rule  # noqa: PLC0415
            slices = _bellpack_unbatch(op)
            if len(slices) == 1:
                return slices[0]
            return _add_rule(list(slices), [True] * len(slices), n)
        # Partial batch-axis reduction.
        if (axes_t and axes_t[-1] < op.n_batch
                and len(axes_t) < op.n_batch):
            reduced = set(axes_t)
            safe = True
            for c in op.in_cols:
                if isinstance(c, slice):
                    continue
                if c.ndim == 1:
                    continue
                for a in axes_t:
                    if a < c.ndim and c.shape[a] != 1:
                        safe = False
                        break
                if not safe:
                    break
            if safe:
                new_values = op.data.sum(axis=axes_t)
                new_in_cols: list[ColArr] = []
                for c in op.in_cols:
                    if c.ndim == 1:
                        new_in_cols.append(c)
                    else:
                        new_in_cols.append(c.squeeze(axis=axes_t))
                new_batch = tuple(
                    s for i, s in enumerate(op.batch_shape) if i not in reduced
                )
                return BEllpack(
                    op.start_row, op.end_row,
                    tuple(new_in_cols), new_values,
                    op.out_size, op.in_size,
                    batch_shape=new_batch,
                )
        # Out-axis-only reduction on single-batch-axis BEllpack.
        if (axes_t == (op.n_batch,) and op.n_batch == 1
                and op.start_row == 0 and op.end_row == op.out_size):
            from lineaxpr._rules.add import _densify_if_wider_than_dense  # noqa: PLC0415
            B = op.batch_shape[0]
            O = op.out_size
            K = op.k
            new_in_cols: list[ColArr] = []
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
                        c_full = c if c.ndim >= 2 else jnp.broadcast_to(
                            c, op.batch_shape + (op.nrows,))
                        # pyrefly: ignore [bad-argument-type]
                        new_in_cols.append(c_full[:, r])
            if K == 1:
                new_values = op.data
            else:
                new_values = op.data.reshape(B, O * K)

            def _col_key(c):
                if isinstance(c, np.ndarray):
                    return ("np", c.shape, c.tobytes())
                if isinstance(c, slice):
                    return ("slc", c.start, c.stop, c.step)
                return ("id", id(c))

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
                dedup_values = jnp.zeros(
                    (B, n_groups), dtype=new_values.dtype
                ).at[:, assigned].add(new_values)
                return _densify_if_wider_than_dense(BEllpack(
                    start_row=0, end_row=B,
                    in_cols=tuple(group_cols), data=dedup_values,
                    out_size=B, in_size=op.in_size,
                    transposed=op.transposed,
                ), n)
            return _densify_if_wider_than_dense(BEllpack(
                start_row=0, end_row=B,
                in_cols=tuple(new_in_cols), data=new_values,
                out_size=B, in_size=op.in_size,
                transposed=op.transposed,
            ), n)

    # BEllpack row-sum (unbatched, axis 0): emit a BEllpack row-vector whose
    # bands hold per-col sums when static cols allow trace-time sparsity analysis.
    if tuple(axes) == (0,) and op.n_batch == 0:
        k = op.k
        in_size = op.in_size
        per_band_cols = list(op.in_cols)
        if all(isinstance(c, np.ndarray) for c in per_band_cols):
            cols_flat = np.concatenate(per_band_cols)
            valid = cols_flat >= 0
            cols_valid = cols_flat[valid]
            uniq_cols, inverse = np.unique(cols_valid, return_inverse=True)
            n_groups = uniq_cols.shape[0]
            if 0 < n_groups < in_size:
                vals_flat = op.data if k == 1 else op.data.T.reshape(-1)
                keep = np.nonzero(valid)[0]
                vals_keep = jnp.take(vals_flat, jnp.asarray(keep)) if keep.shape[0] < cols_flat.shape[0] else vals_flat
                summed = jnp.zeros((n_groups,), op.dtype).at[jnp.asarray(inverse)].add(vals_keep)
                if n_groups == 1:
                    return BEllpack(start_row=0, end_row=1,
                                   in_cols=(np.asarray([uniq_cols[0]], dtype=uniq_cols.dtype),),
                                   data=summed.reshape(1), out_size=1, in_size=in_size)
                return BEllpack(start_row=0, end_row=1,
                               in_cols=tuple(np.asarray([c], dtype=uniq_cols.dtype) for c in uniq_cols),
                               data=summed.reshape(1, n_groups), out_size=1, in_size=in_size)
        cols_stacked = jnp.concatenate([jnp.asarray(c) for c in per_band_cols], axis=0)
        vals_stacked = op.data if k == 1 else op.data.T.reshape(-1)
        mask = cols_stacked >= 0
        return jnp.zeros((in_size,), op.dtype).at[
            jnp.where(mask, cols_stacked, 0)].add(jnp.where(mask, vals_stacked, jnp.zeros((), op.dtype)))

    # Dense fallback.
    dense = op.todense()
    return jnp.sum(dense, axis=tuple(axes))


# ---------------------------------------------------------------------------
# split_op registrations
# ---------------------------------------------------------------------------

@split_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    sizes = params["sizes"]
    axis = params["axis"]
    # Structural path: batched BE split along the out-axis. For T=False
    # the out-axis is at frame index `n_batch`; for T=True (V-at-0) it
    # shifts to `n_batch + 1`. Slice values and each band's cols along
    # the out axis; keep batch_shape. Requires full out coverage.
    out_axis_in_frame = (op.n_batch + 1) if op.transposed else op.n_batch
    if (op.n_batch >= 1
            and axis == out_axis_in_frame
            and op.start_row == 0
            and op.end_row == op.out_size):
        nb = op.n_batch
        out = []
        start = 0
        for sz in sizes:
            sz_i = int(sz)
            end = start + sz_i
            val_slc = [slice(None)] * op.data.ndim
            val_slc[nb] = slice(start, end)
            new_values = op.data[tuple(val_slc)]
            new_in_cols: list[ColArr] = []
            for c in op.in_cols:
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
                sz_i, op.in_size, batch_shape=op.batch_shape,
                transposed=op.transposed,
            ))
            start = end
        return out
    # Structural path: unbatched T=True split along the out axis
    # (axis 1 of `(in_size, out_size)`). Slice `data` along its
    # leading axis (the out-axis in T=True data layout: `(out, k)` for
    # k>=2, `(out,)` for k=1) and each band's col array along its
    # only axis. Emits one BE T=True per chunk with full row coverage.
    if (op.transposed and op.n_batch == 0 and axis == 1
            and op.start_row == 0 and op.end_row == op.out_size):
        out = []
        start = 0
        for sz in sizes:
            sz_i = int(sz)
            end = start + sz_i
            if op.k == 1:
                new_values = op.data[start:end]
            else:
                new_values = op.data[start:end, :]
            new_in_cols: list[ColArr] = []
            for c in op.in_cols:
                if isinstance(c, np.ndarray):
                    new_in_cols.append(c[start:end])
                else:
                    new_in_cols.append(jnp.asarray(c)[start:end])
            out.append(BEllpack(
                0, sz_i, tuple(new_in_cols), new_values,
                sz_i, op.in_size, transposed=True,
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
    if (axis == 0
            and op.n_batch == 0
            and all(isinstance(c, np.ndarray) or isinstance(c, slice)
                    for c in op.in_cols)):
        out = []
        start = 0
        for sz in sizes:
            sz_i = int(sz)
            end = start + sz_i
            # Row range [start, end) intersected with BE's own
            # [start_row, end_row). Slice cols/values along the row axis.
            be_start = max(op.start_row - start, 0)
            be_end = min(op.end_row - start, sz_i)
            if be_end <= be_start:
                out.append(sparse.BCOO(
                    # pyrefly: ignore [bad-argument-type]
                    (jnp.zeros((0,), op.data.dtype),
                     np.zeros((0, 2), np.int32)),
                    shape=(sz_i, op.in_size),
                ))
                start = end
                continue
            row_lo = max(start, op.start_row) - op.start_row
            row_hi = min(end, op.end_row) - op.start_row
            new_in_cols: list[ColArr] = []
            for c in op.in_cols:
                if isinstance(c, slice):
                    c = c
                new_in_cols.append(c[row_lo:row_hi])
            if op.k == 1:
                new_values = op.data[row_lo:row_hi]
            else:
                new_values = op.data[row_lo:row_hi, :]
            chunk_be = BEllpack(
                be_start, be_end, tuple(new_in_cols), new_values,
                sz_i, op.in_size,
            )
            # pyrefly: ignore [bad-argument-type]
            out.append(chunk_be)
            start = end
        return out
    # Fall through to BCOO-based split (handled in bcoo_extend.py for
    # ConstantDiagonal/Diagonal/BCOO; for BEllpack fall through to dense).
    if axis == 0:
        bcoo = op.to_bcoo() if hasattr(op, 'to_bcoo') else op
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
    dense = op.todense()
    out = []
    start = 0
    for sz in sizes:
        slc = [slice(None)] * dense.ndim
        slc[axis] = slice(int(start), int(start) + int(sz))
        out.append(dense[tuple(slc)])
        start += int(sz)
    return out
