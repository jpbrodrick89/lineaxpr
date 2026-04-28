"""BEllpack LinOp class and conversion helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import core
from jax.experimental import sparse

from .base import negate, scale_per_out_row, scale_scalar


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
        static `np.ndarray` or traced `jnp.ndarray`. For unbatched
        (`batch_shape == ()`) the cols are 1D `(nrows,)`. For batched,
        cols may be `(*batch_shape, nrows)` (per-batch varying cols) or
        still 1D (shared cols across batches). `-1` entries are
        sentinels: that slot contributes 0 and is filtered at
        CSR/BCOO conversion.
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
        # K=1 retains the old single-scatter form (already optimal — no
        # per-band loop to begin with). K>=2 fuses all bands' cols and
        # values into one scatter-add via (nrows, k) index arrays, per
        # the "never loop over arrays" rule (CLAUDE.md). Cols stack is
        # static (tuple iteration at trace time), values is passed
        # whole (no per-band slicing).
        if self.n_batch > 0:
            return self._todense_batched()
        return self._todense_unbatched()

    def _todense_unbatched(self):
        rows_1d = np.arange(self.start_row, self.end_row)
        dense = jnp.zeros((self.out_size, self.in_size), self.dtype)
        k = self.k
        if k == 1:
            cols_b = _resolve_col(self.in_cols[0], self.nrows)
            if isinstance(cols_b, np.ndarray) and (cols_b >= 0).all():
                return dense.at[rows_1d, cols_b].add(
                    self.values, unique_indices=True, indices_are_sorted=True)
            mask = cols_b >= 0
            safe_cols = jnp.where(mask, cols_b, 0)
            safe_vals = jnp.where(mask, self.values,
                                  jnp.zeros((), self.dtype))
            return dense.at[rows_1d, safe_cols].add(
                safe_vals, unique_indices=True, indices_are_sorted=True)
        resolved = [_resolve_col(c, self.nrows) for c in self.in_cols]
        all_np = all(isinstance(c, np.ndarray) for c in resolved)
        if all_np:
            cols_nk = np.stack(resolved, axis=-1)  # (nrows, k), static
            static_ok = bool((cols_nk >= 0).all())
        else:
            cols_nk = jnp.stack([jnp.asarray(c) for c in resolved], axis=-1)
            static_ok = False
        rows_nk = np.broadcast_to(rows_1d[:, None], (self.nrows, k))
        if static_ok:
            return dense.at[rows_nk, cols_nk].add(self.values)
        mask = cols_nk >= 0
        safe_cols = jnp.where(mask, cols_nk, 0)
        safe_vals = jnp.where(mask, self.values, jnp.zeros((), self.dtype))
        return dense.at[rows_nk, safe_cols].add(safe_vals)

    def _todense_batched(self):
        out = jnp.zeros(self.batch_shape + (self.out_size, self.in_size),
                        self.dtype)
        k = self.k
        batch_grids = np.meshgrid(
            *[np.arange(d) for d in self.batch_shape],
            np.arange(self.nrows), indexing="ij",
        )
        batch_idx_arrays = batch_grids[:-1]
        row_idx = batch_grids[-1] + self.start_row
        nb_shape = self.batch_shape + (self.nrows,)
        if k == 1:
            cols_b = _resolve_col(self.in_cols[0], self.nrows)
            if isinstance(cols_b, np.ndarray) and cols_b.ndim == 1:
                cols_nd = np.broadcast_to(cols_b, nb_shape)
            else:
                cols_nd = jnp.asarray(cols_b)
            if isinstance(cols_nd, np.ndarray) and (cols_nd >= 0).all():
                return out.at[tuple(batch_idx_arrays) + (row_idx, cols_nd)].add(
                    self.values, unique_indices=True, indices_are_sorted=True)
            mask = cols_nd >= 0
            safe_cols = jnp.where(mask, cols_nd, 0)
            safe_vals = jnp.where(mask, self.values,
                                  jnp.zeros((), self.dtype))
            return out.at[tuple(batch_idx_arrays) + (row_idx, safe_cols)].add(
                safe_vals, unique_indices=True, indices_are_sorted=True)
        resolved = [_resolve_col(c, self.nrows) for c in self.in_cols]
        all_np = all(isinstance(c, np.ndarray) for c in resolved)
        if all_np:
            cols_per_band = []
            for c in resolved:
                if c.ndim == 1:
                    cols_per_band.append(np.broadcast_to(c, nb_shape))
                else:
                    cols_per_band.append(c)
            cols_ndk = np.stack(cols_per_band, axis=-1)
            static_ok = bool((cols_ndk >= 0).all())
        else:
            cols_per_band = []
            for c in resolved:
                ca = jnp.asarray(c)
                if ca.ndim == 1:
                    ca = jnp.broadcast_to(ca, nb_shape)
                cols_per_band.append(ca)
            cols_ndk = jnp.stack(cols_per_band, axis=-1)
            static_ok = False
        def _expand(ix):
            return np.broadcast_to(ix[..., None], nb_shape + (k,))
        batch_idx_k = tuple(_expand(a) for a in batch_idx_arrays)
        row_idx_k = _expand(row_idx)
        if static_ok:
            return out.at[batch_idx_k + (row_idx_k, cols_ndk)].add(self.values)
        mask = cols_ndk >= 0
        safe_cols = jnp.where(mask, cols_ndk, 0)
        safe_vals = jnp.where(mask, self.values, jnp.zeros((), self.dtype))
        return out.at[batch_idx_k + (row_idx_k, safe_cols)].add(safe_vals)

    def to_bcoo(self):
        return _ellpack_to_bcoo(self)

    def negate(self):
        return BEllpack(self.start_row, self.end_row, self.in_cols,
                       -self.values, self.out_size, self.in_size,
                       batch_shape=self.batch_shape)

    def scale_scalar(self, s):
        return BEllpack(self.start_row, self.end_row, self.in_cols,
                       s * self.values, self.out_size, self.in_size,
                       batch_shape=self.batch_shape)

    def scale_per_out_row(self, v):
        v_arr = jnp.asarray(v)
        # For batched BEllpack, `v` comes from a closure at the output
        # aval's shape (`*batch_shape, out_size`); slice the trailing
        # axis to the active row range. For unbatched, same shape
        # convention (v_arr is 1D).
        if self.n_batch > 0:
            v_slice = v_arr[..., self.start_row:self.end_row]
        elif v_arr.shape[0] == self.nrows:
            v_slice = v_arr
        else:
            v_slice = v_arr[self.start_row:self.end_row]
        # values shape: (*batch, nrows) for k=1, (*batch, nrows, k) for k>=2.
        # v_slice shape: (*batch, nrows). For k>=2 broadcast a trailing axis.
        if self.values.ndim == self.n_batch + 1:  # k=1
            scaled = v_slice * self.values
        else:
            scaled = v_slice[..., None] * self.values
        return BEllpack(self.start_row, self.end_row, self.in_cols,
                       scaled, self.out_size, self.in_size,
                       batch_shape=self.batch_shape)

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
                           new_out_size, self.in_size,
                           batch_shape=self.batch_shape)
        nrows_old = self.nrows
        lo = trim_top
        hi = nrows_old - trim_bottom
        if hi <= lo:
            empty_cols = tuple(np.empty(0, dtype=np.int64) for _ in self.in_cols)
            empty_shape = (0,) if self.k == 1 else (0, self.k)
            empty_vals = jnp.empty(empty_shape, self.dtype)
            return BEllpack(0, 0, empty_cols, empty_vals,
                           new_out_size, self.in_size,
                           batch_shape=self.batch_shape)
        new_in_cols = tuple(_slice_col(c, lo, hi) for c in self.in_cols)
        # Slice values along the nrows axis (after batch dims).
        if self.n_batch == 0:
            new_values = self.values[lo:hi]
        else:
            new_values = self.values[(slice(None),) * self.n_batch + (slice(lo, hi),)]
        return BEllpack(new_start + lo, new_end - trim_bottom,
                       new_in_cols, new_values,
                       new_out_size, self.in_size,
                       batch_shape=self.batch_shape)

    def transpose(self, permutation):
        """Permute the `(*batch_shape, out_size)` axes; in-axis stays last.

        The returned BEllpack is structurally equivalent to
        `jnp.transpose(self.todense(), permutation + (ndim-1,))` but
        without densifying.
        """
        nb = self.n_batch
        permutation = tuple(int(p) for p in permutation)
        assert len(permutation) == nb + 1
        old_sizes = self.batch_shape + (self.out_size,)
        new_sizes = tuple(old_sizes[p] for p in permutation)
        new_batch_shape = new_sizes[:-1]
        new_out_size = new_sizes[-1]

        # Fast path: out axis stays last — permute batch axes only,
        # preserving the compressed row range.
        if permutation[-1] == nb:
            batch_perm = permutation[:-1]
            if self.k == 1:
                val_perm = batch_perm + (nb,)
            else:
                val_perm = batch_perm + (nb, nb + 1)
            new_values = jnp.transpose(self.values, val_perm)
            new_in_cols = tuple(
                _transpose_col_batch(c, batch_perm) for c in self.in_cols
            )
            return BEllpack(self.start_row, self.end_row, new_in_cols,
                           new_values, new_out_size, self.in_size,
                           batch_shape=new_batch_shape)

        # General case: out axis moves. Pad the compressed row axis to
        # full out_size, then apply the full permutation to values and
        # each resolved col band. New row range is [0, new_out_size).
        out_size = self.out_size
        pad_before = self.start_row
        pad_after = out_size - self.end_row
        if self.k == 1:
            val_pad = [(0, 0)] * nb + [(pad_before, pad_after)]
            values_full = jnp.pad(self.values, val_pad)
            new_values = jnp.transpose(values_full, permutation)
        else:
            val_pad = [(0, 0)] * nb + [(pad_before, pad_after), (0, 0)]
            values_full = jnp.pad(self.values, val_pad)
            new_values = jnp.transpose(values_full, permutation + (nb + 1,))

        new_in_cols = tuple(
            _transpose_col_full(c, self.batch_shape, self.start_row,
                                self.end_row, out_size, permutation)
            for c in self.in_cols
        )
        return BEllpack(0, new_out_size, new_in_cols, new_values,
                       new_out_size, self.in_size,
                       batch_shape=new_batch_shape)


def _transpose_col_batch(col, batch_perm):
    """Permute only the batch axes of a ColArr. Shared 1D cols are unaffected."""
    if col.ndim == 1:
        return col
    nb = col.ndim - 1
    axes = tuple(batch_perm) + (nb,)
    if isinstance(col, np.ndarray):
        return np.transpose(col, axes)
    return jnp.transpose(col, axes)


def _transpose_col_full(col, batch_shape, start_row, end_row, out_size,
                        permutation):
    """Resolve a ColArr to `(*batch_shape, out_size)` with -1 sentinels
    outside `[start_row, end_row)`, then apply `permutation` across the
    combined `(*batch, out)` axes.
    """
    nb = len(batch_shape)
    pad_before = start_row
    pad_after = out_size - end_row
    if col.ndim == 1:
        # Shared across batches. Pad row axis first, then broadcast.
        if pad_before or pad_after:
            if isinstance(col, np.ndarray):
                col = np.pad(col, (pad_before, pad_after), constant_values=-1)
            else:
                col = jnp.pad(col, (pad_before, pad_after),
                              constant_values=-1)
        if nb:
            if isinstance(col, np.ndarray):
                col = np.broadcast_to(col, batch_shape + (out_size,))
            else:
                col = jnp.broadcast_to(col, batch_shape + (out_size,))
        if isinstance(col, np.ndarray):
            return np.transpose(col, permutation)
        return jnp.transpose(col, permutation)
    # Per-batch `(*batch, nrows)` cols — pad row axis, then permute.
    pad_spec = [(0, 0)] * nb + [(pad_before, pad_after)]
    if isinstance(col, np.ndarray):
        padded = np.pad(col, pad_spec, constant_values=-1)
        return np.transpose(padded, permutation)
    padded = jnp.pad(col, pad_spec, constant_values=-1)
    return jnp.transpose(padded, permutation)


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


def _resolve_col(col, nrows):
    """Return a ColArr (ndarray) unchanged. nrows unused; kept for call-site symmetry."""
    return col


def _slice_col(col, lo, hi):
    """Slice a ColArr along its row axis (the last axis for batched cols)."""
    # Per-batch cols shape (*batch_shape, nrows) — slice the nrows axis.
    if col.ndim > 1:
        return col[..., lo:hi]
    return col[lo:hi]


def _ellpack_to_bcoo(e: "BEllpack") -> sparse.BCOO:
    """Flatten BEllpack to BCOO, filtering -1-sentinel cols.

    Unbatched (n_batch == 0):
      k=1: values is 1D, indices stack rows + band 0's cols. One HLO op
      total for the happy path.
      k>=2: values is (nrows, k); flatten band-by-band (all of band 0,
      then band 1, ...).

    Batched (n_batch > 0):
      Produces a batched `sparse.BCOO` with `n_batch = len(batch_shape)`
      (so `data.shape == (*batch, nse)`, `indices.shape == (*batch, nse,
      2)`). Per-batch cols are used directly; 1D shared cols are
      broadcast to `(*batch, nrows)`. `-1` sentinels keep their
      position in `indices` with col set to 0 and value set to 0,
      since batched BCOO can't have variable `nse` across batches.
    """
    if e.n_batch > 0:
        return _ellpack_to_bcoo_batched(e)
    rows_1d = np.arange(e.start_row, e.end_row)
    nrows = e.nrows
    k = e.k

    # k=1 fast path — single band, values already 1D.
    if k == 1:
        cols_b = _resolve_col(e.in_cols[0], nrows)
        if isinstance(cols_b, np.ndarray):
            if (cols_b >= 0).all():
                indices = np.stack([rows_1d, cols_b], axis=1)
                # pyrefly: ignore [bad-argument-type]
                return sparse.BCOO((e.values, indices), shape=e.shape,
                                   indices_sorted=True, unique_indices=True)
            keep = np.nonzero(cols_b >= 0)[0]
            indices = np.stack([rows_1d[keep], cols_b[keep]], axis=1)
            # pyrefly: ignore [bad-argument-type]
            return sparse.BCOO((jnp.take(e.values, keep), indices),
                               shape=e.shape,
                               indices_sorted=True, unique_indices=True)
        # Traced cols — mask values.
        cols_j = jnp.asarray(cols_b)
        mask = cols_j >= 0
        cols_safe = jnp.where(mask, cols_j, 0)
        vals_safe = jnp.where(mask, e.values, jnp.zeros((), e.dtype))
        indices = jnp.stack([jnp.asarray(rows_1d), cols_safe], axis=1)
        return sparse.BCOO((vals_safe, indices), shape=e.shape,
                           indices_sorted=True, unique_indices=True)

    # k>=2 path — values is (nrows, k). Dispatch by cols-type and k:
    #   * Static cols: always vectorize. Indices are built in pure np
    #     (single compile-time constant through BCOO; the downstream
    #     `_bcoo_concat` np.concatenate path preserves foldability);
    #     values are a single `.T.reshape(-1)` op regardless of K.
    #   * Traced cols: dispatch by k. Small k (k ≤ threshold, NONCVX-
    #     class): loop form — K per-band jnp.where + K-input concat
    #     stays at top level where XLA SIMD-parallelizes (measured 4×
    #     on NONCVX k=3 nrows=5000; confirmed same failure mode at
    #     small nrows when the K-input pattern gets fusion-trapped).
    #     Large k (SPARSINE k=6, LUKSAN k=4-6, NONMSQRT k=71): vec
    #     form — per-op dispatch cost of K jnp.wheres dominates at
    #     large K; single-shot stack+reshape wins. The boundary was
    #     chosen to match the empirical split: NONCVX k=3 is the only
    #     known case needing loop; SPARSINE k=6 and up want vec.
    _BE_TO_BCOO_LOOP_K_THRESHOLD = 3
    per_band_cols = [_resolve_col(c, nrows) for c in e.in_cols]
    any_traced_cols = any(not isinstance(c, np.ndarray) for c in per_band_cols)
    if not any_traced_cols:
        # Static cols: always vectorize (any K, any nrows).
        rows_flat = np.concatenate([rows_1d] * k)
        cols_flat = np.concatenate(per_band_cols)
        vals_flat = e.values.T.reshape(-1)
        mask = cols_flat >= 0
        if mask.all():
            indices = np.stack([rows_flat, cols_flat], axis=1)
            # pyrefly: ignore [bad-argument-type]
            return sparse.BCOO((vals_flat, indices), shape=e.shape)
        keep = np.nonzero(mask)[0]
        indices = np.stack([rows_flat[keep], cols_flat[keep]], axis=1)
        # pyrefly: ignore [bad-argument-type]
        return sparse.BCOO((jnp.take(vals_flat, keep), indices),
                           shape=e.shape)
    # Traced cols — gate by k.
    if k <= _BE_TO_BCOO_LOOP_K_THRESHOLD:
        # Loop form (old): K per-band jnp.where + K-input concat.
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
    # Vectorized for small nrows (traced cols). `per_band_cols` is
    # already a list of K 1D arrays (mixed np/jnp); `jnp.concatenate`
    # handles the mix and goes straight to band-major flat — no
    # intermediate stack+reshape.
    rows_flat_np = np.concatenate([rows_1d] * k)
    cols_flat = jnp.concatenate(per_band_cols)             # band-major
    vals_flat = e.values.T.reshape(-1)                     # band-major
    mask = cols_flat >= 0
    cols_safe = jnp.where(mask, cols_flat, 0)
    vals_safe = jnp.where(mask, vals_flat, jnp.zeros((), e.dtype))
    indices = jnp.stack([jnp.asarray(rows_flat_np), cols_safe], axis=1)
    return sparse.BCOO((vals_safe, indices), shape=e.shape)


def _ellpack_to_bcoo_batched(e: "BEllpack") -> sparse.BCOO:
    """Convert a batched BEllpack to a batched `sparse.BCOO` of shape
    `(*batch_shape, out_size, in_size)` with `n_batch = len(batch_shape)`.

    Sentinel pruning: batched BCOO requires uniform nse per batch, so a
    slot can only be dropped when enough batches have a sentinel there.
    Static-cols path applies the **min-shared-sentinel prune**: stable-
    argsort puts non-sentinels first within each batch, take the first
    `uniform_nse = nrows*k - min(sentinels_per_batch)` slots per batch.
    Residual sentinels in batches whose count exceeds the minimum stay
    as `(col=0, value=0)` masked entries. Traced-cols path keeps all
    `nrows * k` slots with sentinel masking.

    Previously this helper flattened to an unbatched 2D BCOO
    `(prod_batch * out, in)` — semantically a 3D→2D collapse. That
    hid that the walker should have unbatched at its final reshape,
    and broke downstream rules (e.g. `_transpose_rule`) that key off
    LinOp rank. The walker's final `reshape (*output, n) → (total, n)`
    step (`_reshape_rule` batched-BE → unbatched-BE at line 1293 and
    batched-BCOO → unbatched-BCOO at line 1411) is what now
    collapses batch rank at the correct point.
    """
    if e.n_batch == 0:
        return _ellpack_to_bcoo(e)
    B = e.batch_shape
    nrows = e.nrows
    k = e.k
    rows_1d = np.arange(e.start_row, e.end_row)

    per_band = [_resolve_col(c, nrows) for c in e.in_cols]
    all_static = all(isinstance(c, np.ndarray) for c in per_band)
    # (debug hook removed; see git blame if you need to re-instrument)

    if all_static:
        # Broadcast each band's cols to `(*B, nrows)` and concatenate
        # band-major into `(*B, nrows * k)`. Per-batch sentinel patterns
        # are now visible at trace time.
        per_band_full = [
            np.broadcast_to(c, B + (nrows,)) if c.ndim == 1 else c
            for c in per_band
        ]
        cols_bb = np.concatenate(per_band_full, axis=-1)  # (*B, nrows * k)
        rows_bb = np.broadcast_to(
            np.concatenate([rows_1d] * k) if k > 1 else rows_1d,
            B + (nrows * k,),
        )
        sentinel_mask = cols_bb < 0
        # Uniform-nse pruning: drop the **same number** of sentinels from
        # every batch, equal to the minimum sentinel count across
        # batches. Positions dropped within a batch are the first
        # `min_sentinels` sentinel slots in band-major flat order.
        # Remaining sentinels (in batches that had more than the
        # minimum) stay as `(col=0, value=0)` masked entries. This
        # preserves the batched-BCOO uniform-per-batch-nse invariant
        # while recovering most of the prune savings the flat
        # `_ellpack_to_bcoo_batched` gets for free.
        sentinel_count_per_batch = sentinel_mask.reshape(
            -1, nrows * k
        ).sum(axis=-1)
        min_sentinels = (int(sentinel_count_per_batch.min())
                         if sentinel_count_per_batch.size else 0)
        uniform_nse = nrows * k - min_sentinels
        if min_sentinels == 0:
            # No prune possible — every batch has at least one slot we
            # can't drop uniformly.
            safe_cols = np.where(sentinel_mask, 0, cols_bb)
            # Values in band-major order to match `cols_bb`.
            if k == 1:
                vals_bb = e.values
            else:
                nb = len(B)
                vals_bb = jnp.moveaxis(e.values, -1, nb).reshape(
                    B + (nrows * k,)
                )
            safe_vals = jnp.where(
                jnp.asarray(sentinel_mask),
                jnp.zeros((), e.dtype), vals_bb,
            )
            indices = jnp.stack(
                [jnp.asarray(rows_bb), jnp.asarray(safe_cols)], axis=-1,
            )
            return sparse.BCOO(
                (safe_vals, indices),
                shape=B + (e.out_size, e.in_size),
                indices_sorted=False, unique_indices=False,
            )
        # Build per-batch `keep` indices: stable argsort puts False (non-
        # sentinels) before True (sentinels); take the first
        # `uniform_nse` slots so every batch drops exactly
        # `min_sentinels` sentinel positions.
        order = np.argsort(sentinel_mask, axis=-1, kind="stable")
        keep = order[..., :uniform_nse]  # (*B, uniform_nse)
        cols_pruned = np.take_along_axis(cols_bb, keep, axis=-1)
        rows_pruned = np.take_along_axis(rows_bb, keep, axis=-1)
        # Residual sentinels (for batches that had more than the min)
        # stay in `cols_pruned` as `-1`; mask them to `(col=0, value=0)`.
        residual_mask = cols_pruned < 0
        safe_cols_pruned = np.where(residual_mask, 0, cols_pruned)
        if k == 1:
            vals_bb = e.values
        else:
            nb = len(B)
            vals_bb = jnp.moveaxis(e.values, -1, nb).reshape(
                B + (nrows * k,)
            )
        vals_pruned = jnp.take_along_axis(
            vals_bb, jnp.asarray(keep), axis=-1,
        )
        safe_vals = jnp.where(
            jnp.asarray(residual_mask),
            jnp.zeros((), e.dtype), vals_pruned,
        )
        indices = jnp.stack(
            [jnp.asarray(rows_pruned), jnp.asarray(safe_cols_pruned)],
            axis=-1,
        )
        return sparse.BCOO(
            (safe_vals, indices),
            shape=B + (e.out_size, e.in_size),
            indices_sorted=False, unique_indices=False,
        )

    # Traced-cols fallback: can't analyze sentinels at trace time —
    # keep all `nrows * k` slots, mask sentinels to `(col=0, value=0)`.
    rows_broad = np.broadcast_to(rows_1d, B + (nrows,))
    nse = nrows * k

    def resolve_to_batch(col):
        if isinstance(col, np.ndarray):
            if col.ndim == 1:
                return jnp.asarray(np.broadcast_to(col, B + (nrows,)))
            return jnp.asarray(col)
        if col.ndim == 1:
            return jnp.broadcast_to(col, B + (nrows,))
        return col

    if k == 1:
        cols = resolve_to_batch(per_band[0])
        mask = cols >= 0
        safe_cols = jnp.where(mask, cols, 0)
        safe_vals = jnp.where(mask, e.values, jnp.zeros((), e.dtype))
        rows_per_batch = jnp.asarray(rows_broad)
        indices = jnp.stack([rows_per_batch, safe_cols], axis=-1)
        data = safe_vals
    else:
        per_band_bcast = [resolve_to_batch(c) for c in per_band]
        cols_stacked = jnp.stack(per_band_bcast, axis=-1)
        rows_stacked = jnp.broadcast_to(
            jnp.asarray(rows_broad)[..., None], cols_stacked.shape
        )
        mask = cols_stacked >= 0
        safe_cols = jnp.where(mask, cols_stacked, 0)
        safe_vals = jnp.where(mask, e.values, jnp.zeros((), e.dtype))
        indices = jnp.stack(
            [rows_stacked.reshape(B + (nse,)),
             safe_cols.reshape(B + (nse,))],
            axis=-1,
        )
        data = safe_vals.reshape(B + (nse,))

    return sparse.BCOO(
        (data, indices),
        shape=B + (e.out_size, e.in_size),
        indices_sorted=False, unique_indices=False,
    )


# ---- singledispatch registrations ----

@negate.register(BEllpack)
def _(op):
    return op.negate()


@scale_scalar.register(BEllpack)
def _(op, s):
    return op.scale_scalar(s)


@scale_per_out_row.register(BEllpack)
def _(op, v):
    return op.scale_per_out_row(v)
