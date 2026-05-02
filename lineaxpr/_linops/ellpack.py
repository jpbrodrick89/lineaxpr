"""BEllpack LinOp class and conversion helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import sparse

# ColArr: a column-index array — either a static numpy array (trace-time
# constant, allows compile-time sentinel filtering) or a traced JAX array.
ColArr = np.ndarray | jax.Array

from .base import (
    pad_op,
    replace_slots,
    rev_op,
    scale_per_out_row,
    scale_scalar,
    slice_op,
    squeeze_op,
)


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

    __slots__ = ("start_row", "end_row", "in_cols", "data",
                 "out_size", "in_size", "batch_shape", "transposed")

    start_row: int
    end_row: int
    in_cols: tuple[ColArr, ...]
    data: jax.Array
    out_size: int
    in_size: int
    batch_shape: tuple[int, ...]
    transposed: bool

    def __init__(self, start_row: int, end_row: int,
                 in_cols: tuple[ColArr, ...] | list[ColArr],
                 data: jax.Array,
                 out_size: int, in_size: int,
                 batch_shape: tuple[int, ...] = (),
                 transposed: bool = False):
        self.start_row = int(start_row)
        self.end_row = int(end_row)
        self.in_cols = tuple(in_cols)
        self.batch_shape = tuple(int(d) for d in batch_shape)
        self.data = _normalize_data(
            data, len(self.in_cols), self.batch_shape,
            self.end_row - self.start_row,
        )
        self.out_size = int(out_size)
        self.in_size = int(in_size)
        self.transposed = bool(transposed)

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
        if self.transposed:
            # V (=in_size) at axis 0, batch_shape and out_size following.
            return (self.in_size, *self.batch_shape, self.out_size)
        return (*self.batch_shape, self.out_size, self.in_size)

    @property
    def n(self):
        return self.in_size

    @property
    def nse(self):
        """Number of structural entries per batch element (= nrows * k)."""
        return self.nrows * self.k

    @property
    def data_2d(self):
        """data always in (*batch, nrows, k) shape regardless of k."""
        if self.data.ndim == self.n_batch + 1:  # k==1
            return self.data[..., None]
        return self.data

    @property
    def dtype(self):
        return self.data.dtype

    def todense(self):
        # K=1 retains the old single-scatter form (already optimal — no
        # per-band loop to begin with). K>=2 fuses all bands' cols and
        # values into one scatter-add via (nrows, k) index arrays, per
        # the "never loop over arrays" rule (CLAUDE.md). Cols stack is
        # static (tuple iteration at trace time), values is passed
        # whole (no per-band slicing).
        dense = (self._todense_batched() if self.n_batch > 0
                 else self._todense_unbatched())
        if self.transposed:
            # Canonical layout is (*batch_shape, out_size, in_size).
            # transposed=True moves in_size (V) to axis 0.
            ndim = dense.ndim
            perm = (ndim - 1,) + tuple(range(ndim - 1))
            dense = jnp.transpose(dense, perm)
        return dense

    def _todense_unbatched(self):
        rows_1d = np.arange(self.start_row, self.end_row)
        dense = jnp.zeros((self.out_size, self.in_size), self.dtype)
        k = self.k
        if k == 1:
            cols_b = self.in_cols[0]
            if isinstance(cols_b, np.ndarray) and (cols_b >= 0).all():
                return dense.at[rows_1d, cols_b].add(
                    self.data, unique_indices=True, indices_are_sorted=True)
            mask = cols_b >= 0
            safe_cols = jnp.where(mask, cols_b, 0)
            safe_vals = jnp.where(mask, self.data,
                                  jnp.zeros((), self.dtype))
            return dense.at[rows_1d, safe_cols].add(
                safe_vals, unique_indices=True, indices_are_sorted=True)
        resolved = list(self.in_cols)
        all_np = all(isinstance(c, np.ndarray) for c in resolved)
        if all_np:
            cols_nk = np.stack(resolved, axis=-1)  # (nrows, k), static
            static_ok = bool((cols_nk >= 0).all())
        else:
            cols_nk = jnp.stack([jnp.asarray(c) for c in resolved], axis=-1)
            static_ok = False
        rows_nk = np.broadcast_to(rows_1d[:, None], (self.nrows, k))
        if static_ok:
            return dense.at[rows_nk, cols_nk].add(self.data)
        mask = cols_nk >= 0
        safe_cols = jnp.where(mask, cols_nk, 0)
        safe_vals = jnp.where(mask, self.data, jnp.zeros((), self.dtype))
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
            cols_b = self.in_cols[0]
            if isinstance(cols_b, np.ndarray) and cols_b.ndim == 1:
                cols_nd = np.broadcast_to(cols_b, nb_shape)
            else:
                cols_nd = jnp.asarray(cols_b)
            if isinstance(cols_nd, np.ndarray) and (cols_nd >= 0).all():
                return out.at[tuple(batch_idx_arrays) + (row_idx, cols_nd)].add(
                    self.data, unique_indices=True, indices_are_sorted=True)
            mask = cols_nd >= 0
            safe_cols = jnp.where(mask, cols_nd, 0)
            safe_vals = jnp.where(mask, self.data,
                                  jnp.zeros((), self.dtype))
            return out.at[tuple(batch_idx_arrays) + (row_idx, safe_cols)].add(
                safe_vals, unique_indices=True, indices_are_sorted=True)
        resolved = list(self.in_cols)
        all_np = all(isinstance(c, np.ndarray) for c in resolved)
        if all_np:
            cols_per_band: list[ColArr] = []
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
            return out.at[batch_idx_k + (row_idx_k, cols_ndk)].add(self.data)
        mask = cols_ndk >= 0
        safe_cols = jnp.where(mask, cols_ndk, 0)
        safe_vals = jnp.where(mask, self.data, jnp.zeros((), self.dtype))
        return out.at[batch_idx_k + (row_idx_k, safe_cols)].add(safe_vals)

    def __neg__(self):
        return replace_slots(self, data=-self.data)

    def to_bcoo(self):
        # Delegated to `_ellpack_to_bcoo` (which is flag-aware). Both
        # helpers produce BCOOs at the LOGICAL view — see the
        # `_ellpack_to_bcoo` docstring for details.
        return _ellpack_to_bcoo(self)

    def pad_rows(self, before: int, after: int):
        """Pad along the out_size axis. Negative before/after truncates."""
        new_out_size = self.out_size + before + after
        new_start = self.start_row + before
        new_end = self.end_row + before
        # Clip the row range to [0, new_out_size), slicing bands/values in sync.
        trim_top = max(0, -new_start)
        trim_bottom = max(0, new_end - new_out_size)
        if trim_top == 0 and trim_bottom == 0:
            return BEllpack(new_start, new_end, self.in_cols, self.data,
                           new_out_size, self.in_size,
                           batch_shape=self.batch_shape,
                           transposed=self.transposed)
        nrows_old = self.nrows
        lo = trim_top
        hi = nrows_old - trim_bottom
        if hi <= lo:
            empty_cols = tuple(np.empty(0, dtype=np.int64) for _ in self.in_cols)
            empty_shape = (0,) if self.k == 1 else (0, self.k)
            empty_vals = jnp.empty(empty_shape, self.dtype)
            return BEllpack(0, 0, empty_cols, empty_vals,
                           new_out_size, self.in_size,
                           batch_shape=self.batch_shape,
                           transposed=self.transposed)
        new_in_cols = tuple(_slice_col(c, lo, hi) for c in self.in_cols)
        # Slice values along the nrows axis (after batch dims).
        if self.n_batch == 0:
            new_values = self.data[lo:hi]
        else:
            new_values = self.data[(slice(None),) * self.n_batch + (slice(lo, hi),)]
        return BEllpack(new_start + lo, new_end - trim_bottom,
                       new_in_cols, new_values,
                       new_out_size, self.in_size,
                       batch_shape=self.batch_shape,
                       transposed=self.transposed)

    def transpose(self, axes: tuple[int, ...] | None = None):
        """Permute the BE's axes.

        Accepts three forms:

        - `axes is None`: reverse `(*batch_shape, out_size)` — the
          structural form. V (in-axis) stays at the structural tail.
        - `axes` of length `n_batch + 1`: structural perm over
          `(*batch_shape, out_size)`.
        - `axes` of length `n_batch + 2`: full V-augmented perm. If V's
          original position (axis 0 for transposed=True, axis -1 for
          transposed=False) ends up unmoved, this reduces to the
          structural form. The 2D unbatched cross-V swap (`(1, 0)`)
          flips `transposed` for free. Other cross-V perms (V swapped
          with a non-primal axis on rank > 2) are unsupported and
          raise — BE's structural representation can't natively place
          V mid-batch.
        """
        nb = self.n_batch
        permutation = (tuple(int(p) for p in axes) if axes is not None
                       else tuple(range(nb + 1))[::-1])
        # Identity perm — return self.
        if permutation == tuple(range(len(permutation))):
            return self
        # Full V-augmented perm: classify what happens to V.
        if len(permutation) == nb + 2:
            v_axis_in = 0 if self.transposed else nb + 1
            v_axis_out = permutation.index(v_axis_in)
            # Output transposed flag is determined by where V ends up:
            # axis 0 → transposed=True, axis nb+1 (tail) → transposed=False.
            # Anything in between means V crossed a batch axis and isn't
            # structurally representable.
            if v_axis_out == 0:
                new_transposed = True
            elif v_axis_out == nb + 1:
                new_transposed = False
            else:
                raise NotImplementedError(
                    f"BEllpack.transpose: V ends up at axis {v_axis_out} "
                    f"of {nb + 2} under perm {permutation} (transposed="
                    f"{self.transposed}). V must land on either axis 0 "
                    f"or axis {nb + 1}; intermediate positions cross a "
                    f"batch axis and aren't structurally representable."
                )
            # Strip V from the perm and re-index, leaving a structural
            # perm over `(*batch, out_size)`.
            permutation = tuple(p for p in permutation if p != v_axis_in)
            permutation = tuple(p - 1 if p > v_axis_in else p
                                for p in permutation)
            # Apply the structural perm via the recursive code below,
            # then update the flag.
            structural = self.transpose(permutation)
            return replace_slots(structural, transposed=new_transposed)
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
            new_values = jnp.transpose(self.data, val_perm)
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
            values_full = jnp.pad(self.data, val_pad)
            new_values = jnp.transpose(values_full, permutation)
        else:
            val_pad = [(0, 0)] * nb + [(pad_before, pad_after), (0, 0)]
            values_full = jnp.pad(self.data, val_pad)
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


def _normalize_data(data, k: int, batch_shape=(), nrows=None):
    """Coerce a user-supplied `data` into the canonical hybrid layout.

    Accepts:
      - 1D `jnp.ndarray` (only when k==1 and batch_shape==()).
      - 2D `jnp.ndarray` of shape `(nrows, k)` (only when k>=2 and
        batch_shape==()).
      - Tuple of k 1D arrays (stacks axis=1 if k>=2, unwraps if k==1).
      - For `batch_shape=(*B,)`: `(*B, nrows)` array when k==1,
        `(*B, nrows, k)` when k>=2, or a tuple of k `(*B, nrows)` arrays.
    """
    n_batch = len(batch_shape)
    if isinstance(data, tuple):
        if k == 1:
            assert len(data) == 1, f"k=1 but got {len(data)} bands"
            return jnp.asarray(data[0])
        assert len(data) == k, f"k={k} but got {len(data)} bands"
        # Each band has shape (*batch_shape, nrows) → stack on last axis.
        return jnp.stack(list(data), axis=-1)
    arr = jnp.asarray(data)
    if k == 1:
        assert arr.ndim == n_batch + 1, (
            f"k=1 with batch_shape={batch_shape} needs ndim={n_batch+1} data, "
            f"got shape {arr.shape}"
        )
    else:
        assert arr.ndim == n_batch + 2 and arr.shape[-1] == k, (
            f"k={k} with batch_shape={batch_shape} needs "
            f"(*batch, nrows, k) data, got shape {arr.shape}"
        )
    return arr


def _slice_col(col, lo, hi):
    """Slice a ColArr along its row axis (the last axis for batched cols)."""
    # Per-batch cols shape (*batch_shape, nrows) — slice the nrows axis.
    if col.ndim > 1:
        return col[..., lo:hi]
    return col[lo:hi]


def _bcoo_swap_last_two_sparse_axes(bcoo: sparse.BCOO) -> sparse.BCOO:
    """Swap the last two (sparse) axes of a BCOO. Cheap — reorders the
    last two index columns, leaves data and n_batch alone."""
    n_batch = bcoo.n_batch
    perm = tuple(range(n_batch)) + (n_batch + 1, n_batch)
    return bcoo.transpose(axes=perm)


def canonicalize(op):
    """Defensive guard for rules that aren't yet transposed-flag-aware.

    For a `transposed=True` BEllpack, returns its dense view (loses
    sparsity but stays correct). Pass-through for everything else.
    Rules that don't yet inspect `op.transposed` should call this on
    every BEllpack input so they never see a transposed=True operand
    in a code path that interprets jaxpr axes as canonical.

    Per-rule conversion replaces `op = canonicalize(op)` with proper
    flag handling. Until every rule is converted, this guard ensures
    that introducing `transposed=True` in any producer doesn't break
    correctness anywhere downstream.
    """
    if isinstance(op, BEllpack) and op.transposed:
        return op.todense()
    return op


def _ellpack_to_bcoo(e: "BEllpack") -> sparse.BCOO:
    """Flatten BEllpack to BCOO at the LOGICAL view (respects
    `transposed`).

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

    Flag handling: a `transposed=True` BEllpack is flipped to canonical
    (free flag flip — same data, T=False interpretation), the canonical
    BCOO is built, then its last two sparse axes are swapped to
    recover the LOGICAL `(in, out)` indexing. Identical behaviour to
    `BEllpack.to_bcoo`. Mixing the two helpers is safe — both produce
    BCOOs at the LOGICAL view regardless of the BE's flag.
    """
    if e.transposed:
        canonical = replace_slots(e, transposed=False)
        bcoo = _ellpack_to_bcoo(canonical)
        return _bcoo_swap_last_two_sparse_axes(bcoo)
    if e.n_batch > 0:
        return _ellpack_to_bcoo_batched(e)
    rows_1d = np.arange(e.start_row, e.end_row)
    k = e.k

    # k=1 fast path — single band, values already 1D.
    if k == 1:
        cols_b = e.in_cols[0]
        if isinstance(cols_b, np.ndarray):
            if (cols_b >= 0).all():
                indices = np.stack([rows_1d, cols_b], axis=1)
                # pyrefly: ignore [bad-argument-type]
                return sparse.BCOO((e.data, indices), shape=e.shape,
                                   indices_sorted=True, unique_indices=True)
            keep = np.nonzero(cols_b >= 0)[0]
            indices = np.stack([rows_1d[keep], cols_b[keep]], axis=1)
            # pyrefly: ignore [bad-argument-type]
            return sparse.BCOO((jnp.take(e.data, keep), indices),
                               shape=e.shape,
                               indices_sorted=True, unique_indices=True)
        # Traced cols — mask values.
        cols_j = jnp.asarray(cols_b)
        mask = cols_j >= 0
        cols_safe = jnp.where(mask, cols_j, 0)
        vals_safe = jnp.where(mask, e.data, jnp.zeros((), e.dtype))
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
    per_band_cols = list(e.in_cols)
    any_traced_cols = any(not isinstance(c, np.ndarray) for c in per_band_cols)
    if not any_traced_cols:
        # Static cols: always vectorize (any K, any nrows).
        rows_flat = np.concatenate([rows_1d] * k)
        cols_flat = np.concatenate(per_band_cols)
        vals_flat = e.data.T.reshape(-1)
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
            vals_parts.append(jnp.where(mask, e.data[:, b],
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
    vals_flat = e.data.T.reshape(-1)                     # band-major
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
    if e.transposed:
        # Mirror `_ellpack_to_bcoo`'s flag handling: flip-and-swap so
        # callers always see a BCOO at the LOGICAL view regardless of
        # the BE's flag.
        canonical = replace_slots(e, transposed=False)
        bcoo = _ellpack_to_bcoo_batched(canonical)
        return _bcoo_swap_last_two_sparse_axes(bcoo)
    B = e.batch_shape
    nrows = e.nrows
    k = e.k
    rows_1d = np.arange(e.start_row, e.end_row)

    per_band = list(e.in_cols)
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
                vals_bb = e.data
            else:
                nb = len(B)
                vals_bb = jnp.moveaxis(e.data, -1, nb).reshape(
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
            vals_bb = e.data
        else:
            nb = len(B)
            vals_bb = jnp.moveaxis(e.data, -1, nb).reshape(
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
        safe_vals = jnp.where(mask, e.data, jnp.zeros((), e.dtype))
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
        safe_vals = jnp.where(mask, e.data, jnp.zeros((), e.dtype))
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

@scale_scalar.register(BEllpack)
def _(op, s):
    return BEllpack(op.start_row, op.end_row, op.in_cols,
                   s * op.data, op.out_size, op.in_size,
                   batch_shape=op.batch_shape, transposed=op.transposed)


@scale_per_out_row.register(BEllpack)
def _(op, v):
    v_arr = jnp.asarray(v)
    # For batched BEllpack, `v` comes from a closure at the output
    # aval's shape (`*batch_shape, out_size`); slice the trailing
    # axis to the active row range. For unbatched, same shape
    # convention (v_arr is 1D).
    if op.n_batch > 0:
        v_slice = v_arr[..., op.start_row:op.end_row]
    elif v_arr.shape[0] == op.nrows:
        v_slice = v_arr
    else:
        v_slice = v_arr[op.start_row:op.end_row]
    # values shape: (*batch, nrows) for k=1, (*batch, nrows, k) for k>=2.
    # v_slice shape: (*batch, nrows). For k>=2 broadcast a trailing axis.
    if op.data.ndim == op.n_batch + 1:  # k=1
        scaled = v_slice * op.data
    else:
        scaled = v_slice[..., None] * op.data
    return BEllpack(op.start_row, op.end_row, op.in_cols,
                   scaled, op.out_size, op.in_size,
                   batch_shape=op.batch_shape, transposed=op.transposed)


# ---- unary structural op registrations for BEllpack ----

@slice_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    starts = params["start_indices"]
    limits = params["limit_indices"]
    strides = params.get("strides") or (1,) * len(starts)
    # V (in_size) is at axis 0 for transposed=True, axis -1 otherwise.
    # The "in_axis_noop" check confirms the slice doesn't touch V; the
    # primal_out (= structural row axis) sits at the opposite end.
    v_axis = 0 if op.transposed else len(starts) - 1
    primal_axis = len(starts) - 1 if op.transposed else 0
    in_axis_noop = (
        len(starts) >= 2
        and starts[v_axis] == 0
        and limits[v_axis] == op.in_size
        and strides[v_axis] == 1
    )

    # Unit-stride 1D primal slice on unbatched BEllpack.
    if (len(starts) == 2 and in_axis_noop
            and strides[primal_axis] == 1 and op.n_batch == 0):
        s, e = starts[primal_axis], limits[primal_axis]
        return op.pad_rows(-s, -(op.out_size - e))

    # N-D unit-stride slice on batched BEllpack: out_axis is the
    # second-to-last (preceding the in-axis no-op).
    if (op.n_batch > 0
            and in_axis_noop
            and len(starts) == op.n_batch + 2
            and all(st == 1 for st in strides[:-1])):
        batch_slicer = tuple(slice(int(s), int(e))
                             for s, e in zip(starts[:-2], limits[:-2]))
        out_start, out_limit = int(starts[-2]), int(limits[-2])
        tail = (slice(None),) * (op.data.ndim - op.n_batch)
        new_values = op.data[batch_slicer + tail]
        new_in_cols: list[ColArr] = []
        for c in op.in_cols:
            if hasattr(c, "ndim") and c.ndim > 1:
                new_in_cols.append(c[batch_slicer + (slice(None),)])
            else:
                new_in_cols.append(c)
        new_batch = tuple(b.stop - b.start for b in batch_slicer)
        sliced = BEllpack(
            op.start_row, op.end_row,
            tuple(new_in_cols), new_values,
            op.out_size, op.in_size,
            batch_shape=new_batch,
        )
        return sliced.pad_rows(-out_start, -(op.out_size - out_limit))

    return lax.slice(op.todense(), **params)


@pad_op.register(BEllpack)
def _(op, *, n, padding_value, **params):
    config = tuple(params["padding_config"])
    # V (in_size) sits at axis -1 for transposed=False, axis 0 for
    # transposed=True. Padding the V axis is unsupported here; we only
    # accept configs where the V axis pad is a noop.
    v_axis = 0 if op.transposed else len(config) - 1
    primal_axis = len(config) - 1 if op.transposed else 0  # for unbatched
    in_axis_noop = (
        len(config) >= 2 and tuple(config[v_axis]) == (0, 0, 0)
    )

    # 1D primal no-interior pad on unbatched BEllpack.
    if len(config) == 2 and in_axis_noop and op.n_batch == 0:
        before, after, interior = config[primal_axis]
        if int(interior) == 0:
            return op.pad_rows(int(before), int(after))

    # N-D zero-interior pad on batched BEllpack.
    if (op.n_batch > 0
            and in_axis_noop
            and len(config) == op.n_batch + 2
            and all(int(c[2]) == 0 for c in config[:-1])):
        batch_pads = tuple((int(c[0]), int(c[1])) for c in config[:-2])
        out_before, out_after = int(config[-2][0]), int(config[-2][1])
        new_batch_shape = tuple(
            b + s + a for (b, a), s in zip(batch_pads, op.batch_shape)
        )
        tail_pad = ((0, 0),) * (op.data.ndim - op.n_batch)
        new_values = jnp.pad(op.data, batch_pads + tail_pad)
        new_in_cols: list[ColArr] = []
        for c in op.in_cols:
            if hasattr(c, "ndim") and c.ndim > 1:
                pad_cfg = batch_pads + ((0, 0),)
                if isinstance(c, np.ndarray):
                    new_in_cols.append(np.pad(c, pad_cfg, constant_values=-1))
                else:
                    new_in_cols.append(jnp.pad(c, pad_cfg, constant_values=-1))
            else:
                new_in_cols.append(c)
        padded_batch = BEllpack(
            op.start_row, op.end_row,
            tuple(new_in_cols), new_values,
            op.out_size, op.in_size,
            batch_shape=new_batch_shape,
        )
        return padded_batch.pad_rows(out_before, out_after)

    # Interior padding — promote to BCOO then shift rows.
    if len(config) == 2 and in_axis_noop:
        before, after, interior = config[0]
        before, after, interior = int(before), int(after), int(interior)
        if interior > 0 and op.n_batch == 0:
            bcoo_op = op.to_bcoo() if hasattr(op, 'to_bcoo') else op
            step = interior + 1
            old_size = bcoo_op.shape[0]
            out_size = old_size + before + after + interior * max(old_size - 1, 0)
            new_rows = bcoo_op.indices[:, 0] * step + before
            new_indices = jnp.stack([new_rows, bcoo_op.indices[:, 1]], axis=1)
            return sparse.BCOO(
                (bcoo_op.data, new_indices), shape=(out_size, bcoo_op.shape[1])
            )

    # Dense fallback.
    return lax.pad(op.todense(), padding_value, **params)


@squeeze_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    dimensions = params["dimensions"]
    # Unbatched 1-row BE: squeeze of the size-1 out axis is a no-op.
    # The out axis sits at position 0 for transposed=False (shape (1, n))
    # and at position 1 for transposed=True (shape (n, 1)). Returning
    # the BE unchanged lets downstream rules consume the structural
    # form rather than a dense (n,) vector.
    out_axis = (op.n_batch + 1) if op.transposed else op.n_batch
    if (op.n_batch == 0 and dimensions == (out_axis,)
            and op.out_size == 1 and op.start_row == 0 and op.end_row == 1):
        return op
    if (op.n_batch >= 1
            and op.out_size == 1
            and op.start_row == 0 and op.end_row == 1
            and dimensions == (op.n_batch,)):
        B = int(np.prod(op.batch_shape))
        if op.k == 1:
            new_values = op.data.reshape(B)
        else:
            new_values = op.data.reshape(B, op.k)
        new_in_cols: list[ColArr] = []
        ok = True
        for c in op.in_cols:
            if isinstance(c, slice):
                rs = np.arange(c.start or 0, c.stop or 1, c.step or 1)
                if len(rs) == 1:
                    new_in_cols.append(np.broadcast_to(rs, (B,)).copy())
                else:
                    ok = False; break
            elif isinstance(c, np.ndarray):
                if c.ndim == op.n_batch + 1:
                    new_in_cols.append(c.reshape(B))
                elif c.ndim == 1 and c.shape[0] == 1:
                    new_in_cols.append(np.broadcast_to(c, (B,)).copy())
                elif c.ndim == 1 and c.shape[0] == B:
                    new_in_cols.append(c)
                else:
                    ok = False; break
            else:
                ok = False; break
        if ok:
            return BEllpack(
                start_row=0, end_row=B,
                in_cols=tuple(new_in_cols), data=new_values,
                out_size=B, in_size=op.in_size,
            )
    # Densify (sparse → (out_size, in_size)) then squeeze.
    return lax.squeeze(op.todense(), dimensions)


@rev_op.register(BEllpack) # pyrefly: ignore [bad-argument-type]
def _(op, *, n, **params):
    dimensions = params["dimensions"]
    # Unbatched reverse along out axis.
    if op.n_batch == 0 and dimensions == (0,):
        new_start = op.out_size - op.end_row
        new_end = op.out_size - op.start_row
        new_values = jnp.flip(op.data, axis=0)
        new_in_cols: list[ColArr] = []
        for c in op.in_cols:
            if isinstance(c, np.ndarray):
                new_in_cols.append(c[::-1].copy())
            else:
                new_in_cols.append(jnp.flip(c, axis=0))  # pyrefly: ignore [bad-argument-type]
        return BEllpack(
            start_row=new_start, end_row=new_end,
            in_cols=tuple(new_in_cols), data=new_values,
            out_size=op.out_size, in_size=op.in_size,
        )
    dense = op.todense()
    return lax.rev(dense, dimensions)


def _bellpack_unbatch(bep):
    """Split a BEllpack with n_batch >= 1 into a tuple of unbatched BEllpacks.

    For n_batch == 1 the split is direct. For n_batch > 1 we flatten the
    batch axes first: values `(*batch, nrows, k) -> (prod_batch, nrows, k)`;
    N-D cols `(*batch, nrows, ...)` similarly collapse. 1-D shared cols
    and slices stay as-is. Each emitted slice shares
    `(start_row, end_row, out_size, in_size)`.
    """
    assert bep.n_batch >= 1, "use only when n_batch > 0"
    if bep.n_batch > 1:
        prod_B = int(np.prod(bep.batch_shape))
        trailing = bep.data.shape[bep.n_batch:]
        flat_values = bep.data.reshape((prod_B,) + trailing)
        flat_cols: list[ColArr] = []
        for c in bep.in_cols:
            if c.ndim == 1:
                flat_cols.append(c)
            else:
                flat_cols.append(c.reshape((prod_B,) + c.shape[bep.n_batch:]))
        bep = BEllpack(
            start_row=bep.start_row, end_row=bep.end_row,
            in_cols=tuple(flat_cols), data=flat_values,
            out_size=bep.out_size, in_size=bep.in_size,
            batch_shape=(prod_B,),
            transposed=bep.transposed,
        )
    B = bep.batch_shape[0]
    result = []
    for b in range(B):
        in_cols_b = tuple(c[b] if hasattr(c, "ndim") and c.ndim >= 2 else c
                          for c in bep.in_cols)
        values_b = bep.data[b]
        result.append(BEllpack(
            start_row=bep.start_row, end_row=bep.end_row,
            in_cols=in_cols_b, data=values_b,
            out_size=bep.out_size, in_size=bep.in_size,
            batch_shape=(),
            transposed=bep.transposed,
        ))
    return tuple(result)
