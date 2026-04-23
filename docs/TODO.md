# TODO — prioritized

Empirically-grounded; see `RESEARCH_NOTES.md` for reasoning / history.

## Handoff context

Key files:

- `lineaxpr/materialize.py` — rules + public API
- `lineaxpr/_base.py` — LinOp classes (ConstantDiagonal, Diagonal, BEllpack)
- `tests/test_sif2jax_sweep.py` — sweep test with SIZE_OVERRIDES + KNOWN_UNIMPLEMENTED + `nse <= n*n` invariant
- `tests/nse_manifest.json` — per-problem nse baselines (regen via
  `uv run python -m tests.update_nse_manifest`)
- `benchmarks/plots.py` — `--problem-filter unconstrained` for
  publication plots

Rule patterns that have been load-bearing (keep in mind when writing
new rules):

- **Sparsity-preserving structural paths** — `mul(BE, dense)` etc.
  preserve per-batch-element nse; replicate pattern over a new batch
  when size-1 axes are expanded. See `_mul_rule` batch-expand and
  `_add_rule` batch-broadcast.
- **`_densify_if_wider_than_dense`** — fires when `k >= in_size`.
  Prevents BE from carrying effectively-dense state. Applied at
  `_reduce_sum_rule` (out-axis) and `_add_rule` (band-widen) emissions.
- **`_bcoo_concat` uses np.concatenate** for indices when possible —
  avoids XLA iota-decompression of tile-able constants.
- **CLAUDE.md rule**: never Python-loop over jnp arrays. Exception:
  `_ellpack_to_bcoo` loop-form stays for large nrows (k-gate at
  `_BE_TO_BCOO_LOOP_K_THRESHOLD = 3`) — see commit e82ab33.

Sweep-contamination awareness: sweep-level "regressions" are often
false positives — always isolate-verify before believing. See
CLAUDE.md "Sweep vs isolated min". EIGEN\*, DMN15102, LUKSAN16LS have
all been false positives this session.

## Priority 0 — open

### 0c. Structural `_transpose_rule` for BEllpack

**Current**: `_transpose_rule` unconditionally densifies via
`_to_dense(op, n); lax.transpose(...)`, regardless of op form.

**Why it matters**: HADAMALS (2026-04-23) showed step 1 of its HVP
walk is `transpose(BEllpack(20, 20, 400)) → ArrayImpl(20, 20, 400)`,
densifying immediately. Everything downstream (dot_general, diagonal,
scatter-add) then runs dense because the chain started dense.

**Fix**: for batched BEllpack, `transpose` on the non-input axes is a
structural permutation — swap axes of `values` and per-batch
`in_cols`, update `batch_shape`/`out_size`. No data copy. ~30 LoC.

**Cascade**: unblocks `dot_general` (Q.T @ Q linearized) to stay
structural via the existing sparse-vs-closure matmul path. HADAMALS
itself is tiny (dense fits in L2) so the direct win is modest; larger
transposed-reshape patterns would benefit more.

### 0d. Structural 2D point-gather / scatter-add

**Current**: `_gather_rule` + `_scatter_add_rule` have dense fallback
paths for `collapsed_slice_dims=(0, 1)` / `inserted_window_dims=(0, 1)`
(commit bc46c45). Correct but lossy on sparsity.

**Fix**: when operand is BEllpack with batch shape, `M[b[i,0], b[i,1]]`
= pick batch=`b[i,0]`, out-idx=`b[i,1]`, keep input axis. Emit a new
BEllpack with `batch_shape=()`, `out_size=N` (gather length),
values/cols gathered along the (batch, out) axes. Scatter-add is the
dual.

**Dependency**: only useful after 0c (transpose structural) lands —
currently the 2D gather's operand arrives dense because `transpose`
upstream densifies. If 0c doesn't happen, 0d has no beneficiary.

### 0f. Tiny-n `hessian`/`bcoo_hessian` overhead (~1µs vs `jax.hessian`)

**Observed** (2026-04-23 sweep): 147 problems with n ≤ 16 are ~1µs
slower than `jax.hessian`. All have <10µs baseline runtimes, so the
ratio looks bad (1.2×) but absolute gap is tiny.

**Root**: our `hessian(f)` calls `jax.linearize(jax.grad(f), y)` before
`materialize(lin, y, format)`. For n < `_SMALL_N_VMAP_THRESHOLD=16`,
materialize short-circuits to `vmap(linear_fn)(eye)` — same as
jax.hessian internally. The linearize step itself is pure overhead
for tiny-n.

**Fix**: in `hessian`/`bcoo_hessian` public wrappers, bypass linearize
for n < threshold and call `jax.hessian(f)(y)` directly (+ `to_bcoo`
for bcoo flavour). ~10 LoC. May also allow relaxing the threshold.

**Also**: `bcoo_hessian` has ~2-3µs floor vs `hessian` even after the
short-circuit — attributable to BCOO pytree return overhead at the
jit-dispatch boundary. Non-compile-amortisable. Possibly a case to
just document rather than fix.

### 0g. Extend SIZE_OVERRIDES upstream in sif2jax

**Context**: 45 problems still skip the sweep on size with no
override. Most are constrained-quadratic problems (CVXQP1-3,
NCVXQP1-9, NCVXBQP1-3, CVXBQP1, OBSTCL\*, QPBAND/QPNBAND, A0E\*/A0N\*,
GOULDQP2/3, JUNKTURN, 10FOLDTRLS) where the class hardcodes `n` via a
`@property` — no size constructor arg. Adding `n: int = default` as a
constructor kwarg per class would unlock all of them.

**Fix**: sif2jax PR adding constructor size kwargs to the CVXQP /
NCVXQP / bounded-quad families. Mechanical — each class's `n`
property returns a hardcoded int; change to `n: int = <current>` field.

## Priority 1 — structural improvements

### 1. Deferred `Sum` LinOp for order-independent add-chain bucketing

**Motivation**: the heat-equation test in `tests/test_materialize.py`
bloats a 3n-2 tridiagonal to 2.66× dense. `pad(X, (0,1)) + pad(X, (1,0))`
hits different `(start_row, end_row)` ranges, forces an early BCOO
promote, and the remaining chain piles up concat duplicates. Walk
becomes order-sensitive.

**Proposal**: `Sum(operands: tuple[LinOp, ...])` LinOp that `_add_rule`
returns instead of eagerly BCOO-promoting mismatched kinds. Unary
rules (mul/neg/pad/slice/squeeze/rev) distribute through Sum by
pushing into each operand. Densifying rules (dot_general, scatter_add,
reshape, broadcast_in_dim, reduce_sum, dense fallbacks, final walk
output) flush via `_flush_sum` — bucket operands by matrix shape +
`(start_row, end_row)`, emit per-bucket Ellpacks, BCOO-concat.

**Expected win**: heat-equation 56 → 28 for n=8. Plus potential
recovery of FREUROTH (+51%) and BDEXP (+25%) Ellpack regressions
(2026-04-20 — different-cols widen emits 2 broadcast_in_dim + concat
that Old Pivoted did in 1 op).

**Affects**: new class in `_base.py`, new `_flush_sum` helper,
~8 rule touches. Any path reading operand attributes directly (e.g.
`_linop_matrix_shape`) must handle Sum or be flushed first.

**Ordering vs CSR (#2)**: start with CSR — it partially subsumes Sum
(disjoint-range union is natural in CSR).

### 2. Internal CSR LinOp for arrowhead / disjoint-row adds

**Motivation**: `Ellpack`'s `(start_row, end_row)` + uniform `k` forces
padding or BCOO promotion when rows have very different entry counts
(classic arrowhead: one dense row, many unit rows → padded `k = n`).
`_scatter_add_rule` and mismatched-range `_add_rule` fall through to
BCOO concat, dropping structural info downstream rules could exploit.

**Proposal**: `Csr(row_ptr, col_ind, values, out_size, in_size)` that
stays structural through arbitrary scatter-add, disjoint-range add,
and arrowhead patterns. Also natural output target for
`materialize(..., format='csr')` — direct hand-off to cuDSS without a
reconstruction pass.

**Affects**: new class in `_base.py`, `_to_csr` helpers,
`_scatter_add_rule` emits Csr instead of BCOO, `_add_rule` gets a
mixed-Ellpack → Csr path, `materialize` accepts `format='csr'`.

**Win**: unlocks arrowhead problems that currently densify; cuDSS-
native output.

### 3. Test infra follow-ups

- **Sweep runtime**: the slow sweep (~4 min) now exercises 323
  problems; grows as overrides are added. If it becomes a bottleneck,
  split into a fast subset (LUKSAN, DMN, NONCVX, NONMSQRT canaries)
  and a full mode.
- **Publication plot refinement**: `--problem-filter bounded-min`,
  `--problem-filter bounded-quad`, etc. already supported by the
  current CLI. Consider a single-shot `--all-filters` variant for
  batch-generating all 4 variants.

## Priority 2 — densifying rules remaining audit

Per `RESEARCH_NOTES.md` §10 audit of rules that unconditionally
densify where a structural alternative is plausible:

- **`_dot_general`**: currently densifies when both operands traced.
  Could stay BE×BE → BE via sparse outer-product bucketing when
  operands are row-orthogonal; hard case.
- **`_broadcast_in_dim`**: has several structural paths already.
  Remaining case: leading-dim broadcast on BCOO operand (currently
  dense fallback). Rare.
- **`_reshape_rule`**: has batch+out flatten + inverse-flatten
  structural paths. Remaining edge cases around reshapes that cross
  the in_size axis don't have natural BE representation.

None of these have a clear isolated beneficiary yet. Wait for a
problem to motivate the work.

## Recently landed (2026-04-23, `a96520a..57c3093`)

- `_mul_rule` batch-expand + `_add_rule` batch-broadcast (mirror
  pair). Structural `mul(BE, dense)` and `add(BE, BE)` when one side
  has size-1 batch axes.
- `_densify_if_wider_than_dense` helper. Gates wide-k BE emissions at
  `_reduce_sum_rule` and `_add_rule`.
- `_bcoo_concat` np.concatenate for static indices (avoids XLA
  iota-decompression).
- `_ellpack_to_bcoo` k-gate for traced-cols branch.
- `_cond_rule` for closure-index cond (HADAMALS prep).
- 2D point-gather/scatter-add dense fallback (HADAMALS coverage).
- `_split_rule` batched-BE axis=n_batch.
- Test infra: SIZE_OVERRIDES (28 problems covered), KNOWN_UNIMPLEMENTED
  (CURLY/SCURLY conv_general_dilated), drop-skip-on-NotImplementedError,
  `assert nse <= n*n` invariant, `--problem-filter unconstrained`
  for plots.

Main wins vs baseline (e8fd7d5): LUKSAN11-15 (3-6×), DIXMAAN B/C/D
(1.7-1.9×), HADAMALS newly covered, DMN15102/3 fully recovered with
ac3c7a6+4ff0881. No significant regressions after isolate-verification.

---

Anything not listed here that comes up in sweep analysis: isolate-
verify first (CLAUDE.md), then check if it's a new pattern or a
re-occurrence of one of the closed items above.
