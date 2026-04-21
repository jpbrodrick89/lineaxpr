"""LinOp classes + densification helpers.

Internal structural forms used by the sparsify walk. They live in the env
during a single walk and are converted to BCOO or ndarray at the public
API boundary (`materialize`, `to_dense`, `to_bcoo`).

Public API consumers should use `Identity(n, dtype=...)` as the seed for
`lineaxpr.sparsify`.

### Adding a new LinOp form

To extend the format space:

1. Define the class in this file. Give it the standard method set:
   `.shape`, `.n`, `.primal_aval()`, `.todense()`, `.to_bcoo()`,
   `.negate()`, `.scale_scalar(s)`, `.scale_per_out_row(v)`, and any
   form-specific ops (e.g. `BEllpack.pad_rows`).
2. Update `_to_dense(op, n)` and `_to_bcoo(op, n)` to handle it.
3. Export it from `lineaxpr/__init__.py`.
4. In `materialize.py`, touch:
   - `_linop_matrix_shape(v)` — add an `isinstance` branch for shape.
   - `_add_rule`'s kind-dispatch — decide which combos with the new
     form stay structural vs promote to BCOO. Shared path is "any mix
     of {CD, D, BEllpack, <new>, BCOO} at matching shape → BCOO via
     `_to_bcoo` and concat", so no per-combo isinstance soup needed.
   - `_mul_rule` / `_neg_rule` just dispatch to the LinOp methods — no
     new branches unless the new form needs special-case BCOO fallbacks.
   - Rules that currently return BEllpack (e.g. `_slice_rule`,
     `_gather_rule`) may opportunistically return the new form when
     the pattern warrants it.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import core
from jax.experimental import sparse


# -------------------------- structural forms --------------------------


class ConstantDiagonal:
    """Diagonal matrix with all entries equal to `value`."""

    __slots__ = ("n", "value")

    def __init__(self, n: int, value: Any = 1.0):
        self.n = n
        self.value = value

    @property
    def shape(self):
        return (self.n, self.n)

    def primal_aval(self):
        v = jnp.asarray(self.value)
        return core.ShapedArray((self.n,), v.dtype)

    def todense(self):
        if isinstance(self.value, float) and self.value == 1.0:
            return jnp.eye(self.n)
        return self.value * jnp.eye(self.n)

    def to_bcoo(self):
        return _diag_to_bcoo(self)

    def negate(self):
        return ConstantDiagonal(self.n, -self.value)

    def scale_scalar(self, s):
        return ConstantDiagonal(self.n, s * self.value)

    def scale_per_out_row(self, v):
        # value * diag(v) = Diagonal(value * v)
        return Diagonal(self.value * jnp.asarray(v))


def Identity(n: int, dtype=None):
    """The n×n identity as a ConstantDiagonal(n, 1.0).

    The standard seed for `lineaxpr.sparsify` when extracting the full
    Jacobian of a linear function.
    """
    value = jnp.asarray(1.0, dtype=dtype) if dtype is not None else 1.0
    return ConstantDiagonal(n, value)


class Diagonal:
    """Diagonal matrix `diag(values)` for a length-n vector."""

    __slots__ = ("values",)

    def __init__(self, values):
        self.values = values

    @property
    def n(self):
        return self.values.shape[0]

    @property
    def shape(self):
        return (self.n, self.n)

    def primal_aval(self):
        return core.ShapedArray((self.n,), self.values.dtype)

    def todense(self):
        # Kept as explicit `.at[i,i].set(v)` scatter rather than the
        # arguably-cleaner `jnp.diag(v)` or `where(eye, v, 0)` patterns.
        # Reason (measured 2026-04-21 on ARGTRIGLS n=200): although both
        # alternatives produce simpler HLO in isolation, `jnp.diag` adds
        # a `call @_diag` boundary that XLA can't fuse through in
        # complex walks (82µs → 300µs on ARGTRIGLS), and the inlined
        # `where(eye, v, 0)` also regresses (different XLA fusion path).
        # The scatter pattern is context-robust across all measured walks.
        idx = jnp.arange(self.n)
        return jnp.zeros((self.n, self.n), self.values.dtype).at[idx, idx].set(self.values)

    def to_bcoo(self):
        return _diag_to_bcoo(self)

    def negate(self):
        return Diagonal(-self.values)

    def scale_scalar(self, s):
        return Diagonal(s * self.values)

    def scale_per_out_row(self, v):
        return Diagonal(self.values * jnp.asarray(v))


class BEllpack:
    """Batched multi-band sparse operator with a contiguous row range.

    Represents the `(*batch_shape, out_size, in_size)` tensor where
    each batch slice is an Ellpack matrix. Rows in `[start_row, end_row)`
    each hold up to k entries (one per band); rows outside are zero.
    When `batch_shape == ()`, this is a plain Ellpack matrix.

    Shape convention mirrors `jax.experimental.sparse.BCOO`:
    `shape = (*batch_shape, *sparse_shape)` with `n_sparse = 2`, and
    `n_batch = len(batch_shape)`.

    Storage:
      - `in_cols`: tuple of length k. Each entry is a ColArr —
        `slice` (step=1), static `np.ndarray`, or traced
        `jnp.ndarray`. For unbatched (`batch_shape == ()`) the cols
        are 1D `(nrows,)`. For batched, cols may be `(*batch_shape,
        nrows)` (per-batch varying cols) or still 1D (shared cols
        across batches). `-1` entries are sentinels: that slot
        contributes 0 and is filtered at CSR/BCOO conversion.
      - `values`: a single `jnp.ndarray`. Shape `(*batch_shape, nrows)`
        when k==1, `(*batch_shape, nrows, k)` when k>=2.

    The **hybrid 1D-for-k=1 / 2D-for-k>=2** layout (tuned 2026-04-20):
      * k=1 with 1D values avoids the `v[:, None]` axis-insertion
        `broadcast_in_dim` that XLA would emit for `(n, 1)` values.
        This is the common case for slice/gather/scaled-Identity walks.
      * k>=2 with 2D values lets a single `(n, k)` multiply fuse in
        one XLA kernel instead of k separate per-band kernels. For
        k=10 this was ~5× faster than tuple-per-band in isolation.
      * Band-widening from two k=1 Ellpacks stacks to 2D via one
        `jnp.stack(axis=1)`; from mixed k via `jnp.concatenate`.

    Duplicate `(row, col)` entries across bands are allowed; they sum
    at densification (`.at[...].add`) and are left unsummed in BCOO
    output (downstream `segment_sum` dedups).
    """

    __slots__ = ("start_row", "end_row", "in_cols", "values",
                 "out_size", "in_size", "batch_shape")

    def __init__(self, start_row, end_row, in_cols, values,
                 out_size, in_size, batch_shape=()):
        self.start_row = int(start_row)
        self.end_row = int(end_row)
        self.in_cols = tuple(in_cols)
        self.batch_shape = tuple(int(d) for d in batch_shape)
        self.values = _normalize_values(
            values, len(self.in_cols), self.batch_shape,
            self.end_row - self.start_row,
        )
        self.out_size = int(out_size)
        self.in_size = int(in_size)

    @property
    def nrows(self):
        return self.end_row - self.start_row

    @property
    def k(self):
        return len(self.in_cols)

    @property
    def n_batch(self):
        return len(self.batch_shape)

    @property
    def n_sparse(self):
        return 2  # out_size + in_size are both sparse dims

    @property
    def shape(self):
        return (*self.batch_shape, self.out_size, self.in_size)

    @property
    def n(self):
        return self.in_size

    @property
    def nse(self):
        """Number of structural entries per batch element (= nrows * k)."""
        return self.nrows * self.k

    @property
    def dtype(self):
        return self.values.dtype

    def primal_aval(self):
        return core.ShapedArray((self.in_size,), self.dtype)

    def todense(self):
        # For batched BEllpack, materialize each batch slice via the
        # unbatched todense and stack on the leading axes. Keeps the
        # densification logic simple and shares it with the n_batch==0
        # case (loop body below).
        if self.n_batch > 0:
            # Flatten batch dims, densify per slice, reshape.
            B_total = 1
            for d in self.batch_shape:
                B_total *= d
            slices = []
            for flat_idx in range(B_total):
                # Unravel to per-axis index.
                idx = []
                rem = flat_idx
                for d in self.batch_shape[::-1]:
                    idx.insert(0, rem % d); rem //= d
                idx = tuple(idx)
                # Build an unbatched BEllpack for this slice.
                in_cols_s = tuple(
                    _col_batch_index(c, idx) for c in self.in_cols
                )
                vals_s = self.values[idx]
                one = BEllpack(
                    self.start_row, self.end_row, in_cols_s, vals_s,
                    self.out_size, self.in_size, batch_shape=(),
                )
                slices.append(one.todense())
            stacked = jnp.stack(slices, axis=0)
            return stacked.reshape(self.batch_shape + (self.out_size, self.in_size))
        rows_1d = np.arange(self.start_row, self.end_row)
        dense = jnp.zeros((self.out_size, self.in_size), self.dtype)
        k = self.k
        for b in range(k):
            cols_b = _resolve_col(self.in_cols[b], self.nrows)
            vals_b = self.values if k == 1 else self.values[..., b]
            if isinstance(cols_b, np.ndarray) and (cols_b >= 0).all():
                dense = dense.at[rows_1d, cols_b].add(vals_b)
            else:
                mask = cols_b >= 0
                safe_cols = jnp.where(mask, cols_b, 0)
                safe_vals = jnp.where(mask, vals_b, jnp.zeros((), self.dtype))
                dense = dense.at[rows_1d, safe_cols].add(safe_vals)
        return dense

    def to_bcoo(self):
        return _ellpack_to_bcoo(self)

    def negate(self):
        return BEllpack(self.start_row, self.end_row, self.in_cols,
                       -self.values, self.out_size, self.in_size)

    def scale_scalar(self, s):
        return BEllpack(self.start_row, self.end_row, self.in_cols,
                       s * self.values, self.out_size, self.in_size)

    def scale_per_out_row(self, v):
        v_arr = jnp.asarray(v)
        if v_arr.shape[0] == self.nrows:
            v_slice = v_arr
        else:
            # v is length out_size; slice to our row range.
            v_slice = v_arr[self.start_row:self.end_row]
        # 1D values: direct 1D mul. 2D values: broadcast v along columns.
        if self.values.ndim == 1:
            scaled = v_slice * self.values
        else:
            scaled = v_slice[:, None] * self.values
        return BEllpack(self.start_row, self.end_row, self.in_cols,
                       scaled, self.out_size, self.in_size)

    def pad_rows(self, before: int, after: int):
        """Pad along the out_size axis. Negative before/after truncates."""
        new_out_size = self.out_size + before + after
        new_start = self.start_row + before
        new_end = self.end_row + before
        # Clip the row range to [0, new_out_size), slicing bands/values in sync.
        trim_top = max(0, -new_start)
        trim_bottom = max(0, new_end - new_out_size)
        if trim_top == 0 and trim_bottom == 0:
            return BEllpack(new_start, new_end, self.in_cols, self.values,
                           new_out_size, self.in_size)
        nrows_old = self.nrows
        lo = trim_top
        hi = nrows_old - trim_bottom
        if hi <= lo:
            empty_cols = tuple(np.empty(0, dtype=np.int64) for _ in self.in_cols)
            empty_shape = (0,) if self.k == 1 else (0, self.k)
            empty_vals = jnp.empty(empty_shape, self.dtype)
            return BEllpack(0, 0, empty_cols, empty_vals,
                           new_out_size, self.in_size)
        new_in_cols = tuple(_slice_col(c, lo, hi) for c in self.in_cols)
        new_values = self.values[lo:hi]
        return BEllpack(new_start + lo, new_end - trim_bottom,
                       new_in_cols, new_values,
                       new_out_size, self.in_size)


def _normalize_values(values, k: int, batch_shape=(), nrows=None):
    """Coerce a user-supplied `values` into the canonical hybrid layout.

    Accepts:
      - 1D `jnp.ndarray` (only when k==1 and batch_shape==()).
      - 2D `jnp.ndarray` of shape `(nrows, k)` (only when k>=2 and
        batch_shape==()).
      - Tuple of k 1D arrays (stacks axis=1 if k>=2, unwraps if k==1).
      - For `batch_shape=(*B,)`: `(*B, nrows)` array when k==1,
        `(*B, nrows, k)` when k>=2, or a tuple of k `(*B, nrows)` arrays.
    """
    n_batch = len(batch_shape)
    if isinstance(values, tuple):
        if k == 1:
            assert len(values) == 1, f"k=1 but got {len(values)} bands"
            return jnp.asarray(values[0])
        assert len(values) == k, f"k={k} but got {len(values)} bands"
        # Each band has shape (*batch_shape, nrows) → stack on last axis.
        return jnp.stack(list(values), axis=-1)
    arr = jnp.asarray(values)
    if k == 1:
        assert arr.ndim == n_batch + 1, (
            f"k=1 with batch_shape={batch_shape} needs ndim={n_batch+1} values, "
            f"got shape {arr.shape}"
        )
    else:
        assert arr.ndim == n_batch + 2 and arr.shape[-1] == k, (
            f"k={k} with batch_shape={batch_shape} needs "
            f"(*batch, nrows, k) values, got shape {arr.shape}"
        )
    return arr


def _col_batch_index(col, batch_idx):
    """Index a ColArr at a batch position tuple.

    - `slice` (shared across batches): pass through.
    - `ndarray`: if col.ndim > 1, index the leading batch dims; else shared.
    """
    if isinstance(col, slice):
        return col
    if col.ndim > 1:
        return col[batch_idx]
    return col


def _resolve_col(col, nrows):
    """Materialize a ColArr (slice | ndarray) to a length-nrows 1D array."""
    if isinstance(col, slice):
        start = 0 if col.start is None else col.start
        stop = nrows if col.stop is None else col.stop
        return np.arange(start, stop)
    return col


def _slice_col(col, lo, hi):
    """Slice a ColArr along its row axis: col[lo:hi]."""
    if isinstance(col, slice):
        start = 0 if col.start is None else col.start
        return slice(start + lo, start + hi)
    return col[lo:hi]


# -------------------------- densification helpers --------------------------


def _to_dense(op, n: int) -> jnp.ndarray:
    if isinstance(op, ConstantDiagonal):
        if isinstance(op.value, float) and op.value == 1.0:
            return jnp.eye(n)
        return op.value * jnp.eye(n)
    if isinstance(op, Diagonal):
        m = op.values.shape[0]
        idx = jnp.arange(m)
        return jnp.zeros((m, m), op.values.dtype).at[idx, idx].set(op.values)
    if isinstance(op, BEllpack):
        return op.todense()
    if isinstance(op, sparse.BCOO):
        return op.todense()
    return op


def _ellpack_to_bcoo(e: "BEllpack") -> sparse.BCOO:
    """Flatten BEllpack to BCOO, filtering -1-sentinel cols.

    k=1: values is 1D, indices stack rows + band 0's cols. One HLO op
    total for the happy path.

    k>=2: values is (nrows, k); flatten band-by-band (all of band 0,
    then band 1, ...) so `jnp.concatenate(values_band_slices, axis=0)`
    emits a single 1D concat with no shape-manipulation overhead.
    """
    rows_1d = np.arange(e.start_row, e.end_row)
    nrows = e.nrows
    k = e.k

    # k=1 fast path — single band, values already 1D.
    if k == 1:
        cols_b = _resolve_col(e.in_cols[0], nrows)
        if isinstance(cols_b, np.ndarray):
            if (cols_b >= 0).all():
                indices = np.stack([rows_1d, cols_b], axis=1)
                return sparse.BCOO((e.values, indices), shape=e.shape)
            keep = np.nonzero(cols_b >= 0)[0]
            indices = np.stack([rows_1d[keep], cols_b[keep]], axis=1)
            return sparse.BCOO((jnp.take(e.values, keep), indices),
                               shape=e.shape)
        # Traced cols — mask values.
        cols_j = jnp.asarray(cols_b)
        mask = cols_j >= 0
        cols_safe = jnp.where(mask, cols_j, 0)
        vals_safe = jnp.where(mask, e.values, jnp.zeros((), e.dtype))
        indices = jnp.stack([jnp.asarray(rows_1d), cols_safe], axis=1)
        return sparse.BCOO((vals_safe, indices), shape=e.shape)

    # k>=2 path — values is (nrows, k).
    per_band_cols = [_resolve_col(c, nrows) for c in e.in_cols]
    any_traced_cols = any(not isinstance(c, np.ndarray) for c in per_band_cols)
    if not any_traced_cols:
        kept_rows, kept_cols, kept_vals = [], [], []
        for b in range(k):
            cols_b = per_band_cols[b]
            if (cols_b >= 0).all():
                kept_rows.append(rows_1d)
                kept_cols.append(cols_b)
                kept_vals.append(e.values[:, b])
            else:
                keep = np.nonzero(cols_b >= 0)[0]
                kept_rows.append(rows_1d[keep])
                kept_cols.append(cols_b[keep])
                kept_vals.append(jnp.take(e.values[:, b], keep))
        rows_flat = np.concatenate(kept_rows)
        cols_flat = np.concatenate(kept_cols)
        indices = np.stack([rows_flat, cols_flat], axis=1)
        vals_flat = jnp.concatenate(kept_vals)
        return sparse.BCOO((vals_flat, indices), shape=e.shape)
    # Traced cols in some band — mask values band-by-band.
    rows_parts, cols_parts, vals_parts = [], [], []
    for b in range(k):
        cols_j = jnp.asarray(per_band_cols[b])
        mask = cols_j >= 0
        rows_parts.append(jnp.asarray(rows_1d))
        cols_parts.append(jnp.where(mask, cols_j, 0))
        vals_parts.append(jnp.where(mask, e.values[:, b],
                                    jnp.zeros((), e.dtype)))
    rows_flat = jnp.concatenate(rows_parts)
    cols_flat = jnp.concatenate(cols_parts)
    vals_flat = jnp.concatenate(vals_parts)
    indices = jnp.stack([rows_flat, cols_flat], axis=1)
    return sparse.BCOO((vals_flat, indices), shape=e.shape)


def _diag_to_bcoo(d, n=None) -> sparse.BCOO:
    """Convert a (Constant)Diagonal to BCOO."""
    idx = jnp.arange(d.n)
    indices = jnp.stack([idx, idx], axis=1)
    if isinstance(d, ConstantDiagonal):
        v = jnp.asarray(d.value)
        data = jnp.broadcast_to(v, (d.n,))
    elif isinstance(d, Diagonal):
        data = d.values
    else:
        raise TypeError(f"_diag_to_bcoo expected diagonal LinOp, got {type(d)}")
    return sparse.BCOO((data, indices), shape=(d.n, d.n))


def _to_bcoo(op, n: int):
    """Convert any internal LinOp to BCOO (used at the `materialize`
    boundary when `format='bcoo'`, and internally by `_add_rule` to
    promote mixed-form operands to a common BCOO before concatenation)."""
    if isinstance(op, sparse.BCOO):
        return op
    if isinstance(op, (ConstantDiagonal, Diagonal)):
        return _diag_to_bcoo(op)
    if isinstance(op, BEllpack):
        return _ellpack_to_bcoo(op)
    return op  # plain ndarray — caller will keep dense


def _traced_shape(op) -> tuple:
    if isinstance(op, (ConstantDiagonal, Diagonal)):
        return (op.n,)
    if isinstance(op, BEllpack):
        return (op.out_size,)
    return tuple(op.shape[:-1])
