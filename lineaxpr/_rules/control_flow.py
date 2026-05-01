"""Control-flow structural rules: cond, jit, select_n."""

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
)
from .._linops import _bcoo_concat
from .add import _cols_equal


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
    _materialize_module = importlib.import_module("lineaxpr._transform")
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
    _materialize_module = importlib.import_module("lineaxpr._transform")
    _materialize_module._walk_jaxpr(inner, inner_env, n)

    # jit_p is always multiple_results; walker sets all outputs to traced
    # since _jit_rule is only called when any input is traced.
    return [inner_env[outvar][1] for outvar in inner.outvars]


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
                jnp.broadcast_to(jnp.asarray(t_case.data), (t_case.n,)),
                t_case.n, t_case.n,
            )
        elif isinstance(t_case, Diagonal):
            t_case = BEllpack(
                0, t_case.n, (np.arange(t_case.n),), t_case.data,
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
        # BEllpack: mask values by row-predicate.
        # pred has the BE's aval shape `(*batch_shape, out_size)`;
        # slice the last axis to the active row range. Scalar pred
        # (aval=()) applies uniformly — skip the slice.
        if isinstance(t_case, BEllpack):
            pred_arr = jnp.asarray(pred)
            if pred_arr.ndim == 0:
                mask = (pred_arr == t_idx)
            else:
                pred_slice = pred_arr[..., t_case.start_row:t_case.end_row]
                mask = (pred_slice == t_idx)
                if t_case.k >= 2:
                    mask = mask[..., None]
            new_values = jnp.where(mask, t_case.data,
                                   jnp.zeros((), t_case.dtype))
            return BEllpack(
                t_case.start_row, t_case.end_row, t_case.in_cols,
                new_values, t_case.out_size, t_case.in_size,
                batch_shape=t_case.batch_shape,
                transposed=t_case.transposed,
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
        same_transposed = all(c.transposed == first.transposed for c in cases[1:])
        if same_cols and same_transposed:
            pred_arr = jnp.asarray(pred)
            if pred_arr.ndim == 0:
                # Scalar pred applies uniformly across rows (HELIX /
                # PFIT* at n=3). No slicing or row-axis broadcast
                # needed; values broadcast against scalar naturally.
                pred_b = pred_arr
            else:
                pred_slice = pred_arr[first.start_row:first.end_row]
                pred_b = pred_slice[:, None] if first.data.ndim > 1 else pred_slice
            # select_n with bool pred: cases[0] when pred is False, cases[1] when True
            # (matching lax.select_n semantics for 2-case).
            if len(cases) == 2:
                new_values = jnp.where(pred_b, cases[1].data, cases[0].data)
            else:
                # N-way: use lax.select_n on stacked values.
                stacked = jnp.stack([c.data for c in cases], axis=0)
                new_values = lax.select_n(pred_b, *[stacked[i] for i in range(len(cases))])
            return BEllpack(
                first.start_row, first.end_row, first.in_cols,
                new_values, first.out_size, first.in_size,
                transposed=first.transposed,
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
            bc = c.to_bcoo() if hasattr(c, 'to_bcoo') else c
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

    # Densify each case. Non-traced cases contribute zero to the
    # linear-in-input part (their dependence on the traced input is
    # zero), so we represent them as a zero tensor with V appended in
    # the convention-appropriate position.
    #
    # Inside-vmap, the traced cases may have V at axis 0 (when their
    # upstream chain came through a transposed-BE → densify path)
    # rather than at axis -1. Detect that via shape inspection (only
    # unambiguous for non-square traced shapes, but those are the ones
    # that actually crash) and build closure zeros with V at 0 to
    # match.
    raw_traced = [
        c.todense() if isinstance(c, LinOpProtocol) else c
        for c, t in zip(cases, case_traced) if t
    ]
    v_at_zero = any(d.ndim >= 2 and d.shape[0] == n and d.shape[-1] != n
                    for d in raw_traced)
    case_dense = []
    traced_iter = iter(raw_traced)
    for c, t in zip(cases, case_traced):
        if t:
            case_dense.append(next(traced_iter))
        else:
            arr = jnp.asarray(c)
            zero_shape = (n,) + arr.shape if v_at_zero else arr.shape + (n,)
            case_dense.append(jnp.zeros(zero_shape, dtype=arr.dtype))
    # Normalise densified cases to the lowest aval-rank by squeezing
    # leading size-1 axes only. A 1-row BEllpack (aval ndim 0)
    # densifies to `(1, n)`; a scalar-aval LinOp densifies to `(n,)`.
    # Without this align, `lax.select_n` rejects mismatched case shapes
    # (HELIX n=3 repro: one case `(1, 3)` vs another `(3,)`). Only
    # squeeze size-1 leading axes — a leading non-1 axis is meaningful
    # data (e.g. a closure zero of shape `(out_dim, n, V)` should NOT
    # be sliced down to `(n, V)`; that would pick its 0-th slice).
    def _squeeze_leading_ones(d, target_ndim):
        while d.ndim > target_ndim and d.shape[0] == 1:
            d = d[0]
        return d
    min_ndim = min(d.ndim for d in case_dense)
    case_dense = [_squeeze_leading_ones(d, min_ndim) for d in case_dense]
    # Broadcast all cases to a common shape (handles vmap-accumulated batch dims).
    target_shape = case_dense[0].shape
    for d in case_dense[1:]:
        target_shape = jnp.broadcast_shapes(target_shape, d.shape)
    case_dense = [jnp.broadcast_to(d, target_shape) for d in case_dense]

    pred_arr = jnp.asarray(pred)
    target_shape = case_dense[0].shape
    while pred_arr.ndim >= len(target_shape):
        pred_arr = pred_arr[0]
    # pred carries var_shape axes; insert V at the convention-appropriate
    # position to make broadcast_to work.
    if v_at_zero:
        pred_b = jnp.broadcast_to(pred_arr[None, ...], target_shape)
    else:
        pred_b = jnp.broadcast_to(pred_arr[..., None], target_shape)
    return lax.select_n(pred_b, *case_dense)
