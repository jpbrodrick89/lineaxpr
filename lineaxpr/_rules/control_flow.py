"""Control-flow structural rules: concatenate, split, cond, jit, select_n."""

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
    _to_bcoo,
    _to_dense,
)
from .add import _bcoo_concat
from .add import _cols_equal


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
                    c = c
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
    # Late import to avoid circular dependency.
    import importlib
    _materialize_module = importlib.import_module("lineaxpr.materialize")
    _materialize_module._walk_jaxpr(inner, inner_env, n)
    return [inner_env[outvar][1] for outvar in inner.outvars]


def _jit_rule(invals, traced, n, **params):
    """Recurse into the inner jaxpr of a `jit` (pjit) call."""
    inner_cj = params["jaxpr"]  # ClosedJaxpr
    inner = inner_cj.jaxpr
    inner_consts = inner_cj.consts

    inner_env: dict = {v: (False, c) for v, c in zip(inner.constvars, inner_consts)}
    for inner_invar, outer_val, was_traced in zip(inner.invars, invals, traced):
        inner_env[inner_invar] = (was_traced, outer_val)
    # Late import to avoid circular dependency.
    import importlib
    _materialize_module = importlib.import_module("lineaxpr.materialize")
    _materialize_module._walk_jaxpr(inner, inner_env, n)

    # jit_p is always multiple_results; walker sets all outputs to traced
    # since _jit_rule is only called when any input is traced.
    return [inner_env[outvar][1] for outvar in inner.outvars]


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
