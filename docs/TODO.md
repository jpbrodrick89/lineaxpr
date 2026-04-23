# TODO — prioritized

Empirically-grounded; see `RESEARCH_NOTES.md` for the reasoning.

## Priority 0 — next up

### 0a. ~~Ensure unimplemented rules lead to test failures, not skips~~ — DONE (f564c63)

Dropped the `try/except NotImplementedError: pytest.skip` around the
walk in `tests/test_sif2jax_sweep.py`. Added `KNOWN_UNIMPLEMENTED`
dict for explicit documentation of any problems that hit unsupported
primitives (empty now; HADAMALS covered by cond + 2D-gather in
bc46c45).

### 0b. ~~Expand correctness coverage by running large problems at smaller sizes~~ — DONE (f564c63)

Added `SIZE_OVERRIDES` dict in the sweep test. 22 smaller-n variants
run correctness coverage (but not nse-manifest check, since that's
keyed by default-n). Covers: BOX, COSINE, CURLY10/20/30, DIXON3DQ,
INDEFM, POWER, YATP1LS/CLS, TORSION1-6/A-F. Easy to extend: add
entries to `SIZE_OVERRIDES` when new size-parameterized problems
appear.

Remaining skips that aren't currently overrideable: most CVXQP/NCVXQP
variants (only y0_iD as init-arg), CYCLIC3LS/CYCLOOCFLS/CYCLOOCTLS,
DEGDIAG/DEGTRID/DEGTRID2, FMINSURF/FMINSRF2, 10FOLDTRLS, SCURLY\*\_0/\_1
(size fixed in class), A0Nxx/A0Exx. Add overrides as their
constructor conventions are verified.

### 0c. Fix densification in `_transpose_rule`

**Context**: HADAMALS walk instrumentation (2026-04-23) showed the
very first structural op after `reshape` (`transpose`) eagerly
densifies:

```
0: reshape    ConstantDiagonal(400,400) → BEllpack(20, 20, 400)  ✓
1: transpose  BEllpack(20, 20, 400)     → ArrayImpl(20, 20, 400)  ← dense
```

The entire downstream matmul/diagonal/scatter chain then runs dense.
`_transpose_rule` currently does `_to_dense(op, n); lax.transpose(...)`
unconditionally regardless of op form.

**Proposal**: for BEllpack with batched structure, `transpose` on
the non-input axes is a structural permutation — swap axes of
`values` and per-batch `in_cols`, update `batch_shape`/`out_size`.
No data copy. ~30 LoC.

**Cascade impact**: unblocks the downstream `dot_general` (Q.T @ Q
linearized) to stay structural via the existing sparse-vs-closure
matmul path. Benefits any problem with transposed-reshape patterns.
HADAMALS itself is small (n=20, 1.3MB dense fits in L2) so the
direct win is modest; larger-dim transpose chains would benefit more.

### 0d-pre. Investigate DMN15102LS regression source (not scatter-add)

**Context**: 094627a sweep (2026-04-23) flagged DMN15102LS as
regressed 1.25× (+4.5 ms, 20 → 24.5 ms) vs baseline. Bisect
attributed it to commit 93e8c3f (step 2, reshape emit BE instead
of BCOO).

**Investigation**: rule invocation counts are **identical** between
baseline and current (25 mul, 10 add_any, 5 reduce_sum, 4
broadcast_in_dim, etc.). Walker does the exact same work. Only
the intermediate LinOp forms differ.

Walk shows 6× add_any BE+dense→dense via `_add_rule`'s fallback
at current, vs baseline's mix (6× dense+dense, 2× dense+BE→dense,
1× BE+BE→BE, 1× BE+dense→dense). All shapes `(4643, 33, 66)` or
`(1, 33, 66)`.

**Microbench ruled out scatter-add fix**: `BE.todense() + dense`
vs `dense.at[indices].add(BE.values)` — both 3273 µs at shape
(4643, 33, 66). XLA fuses the zeros+scatter+add into an efficient
single op. Swapping the walker path wouldn't help.

**Probable root cause**: XLA emits different HLO based on the
intermediate LinOp shapes/forms flowing through jit tracing.
Baseline's earlier-densification pattern may fuse better than
current's late-densification pattern.

**Status**: no trivial fix; needs HLO-level diff between baseline
and current to identify the specific XLA optimization gap. Low
priority — single problem, ~25% slowdown on a 20ms baseline,
massively outweighed by LUKSAN 11-15 wins (3-6×).

### 0d. Implement structural 2D point-gather / scatter-add

**Context**: `jnp.diagonal` emits a 2-branch cond (platform
dispatch) whose active branch uses 2D point-gather /
scatter-add with `collapsed_slice_dims=(0, 1)`. Current rules
(`bc46c45`) fall back to dense for correctness.

**Proposal**: structural emit — when operand is BEllpack with batch
shape, `M[b[i,0], b[i,1]]` is: pick batch=`b[i,0]`, out-idx=`b[i,1]`,
keep input axis. Emit a new BEllpack with `batch_shape=()`,
`out_size=N` (gather length), values/cols gathered along the
(batch, out) axes. Scatter-add is the dual.

**Impact**: benefits HADAMALS-class problems only if `0c` (transpose
fix) also lands, since the 2D gather currently receives dense input
because of upstream densification.

### 1. Deferred `Sum` LinOp for order-independent add-chain bucketing

**Motivation**: the heat-equation test in `tests/test_materialize.py`
bloats a 3n-2 tridiagonal to 2.66× that. Walk instrumentation (2026-04-20)
confirmed: `pad(X, (0,1)) + pad(X, (1,0))` hits different
`(start_row, end_row)` ranges, forces an early BCOO promote, and the
remaining `pad + add_any` chain for the neg/mirror term piles up concat
duplicates. Reordering the add tree to group by-range first would keep
everything Ellpack and flush to BCOO once.

**Proposal**: `Sum(operands: tuple[LinOp, ...])` LinOp that `_add_rule`
returns instead of eagerly BCOO-promoting mismatched kinds. Unary
structural rules (`_mul_rule`, `_neg_rule`, `_pad_rule`, `_slice_rule`,
`_squeeze_rule`, `_rev_rule`) distribute through Sum by pushing into
each operand. Densifying rules (`_dot_general_rule`, `_scatter_add_rule`,
`_reshape_rule`, `_broadcast_in_dim_rule`, `_reduce_sum_rule`, dense
fallbacks, final walk output) flush via `_flush_sum` — bucket operands
by matrix shape + `(start_row, end_row)`, emit per-bucket Ellpacks,
BCOO-concat across buckets.

**Expected win**: heat-equation 56 → 28 for n=8 (2.66× → 1.27× in
general). Walk becomes order-independent for add-chains, robust against
authorial style variations in sif2jax.

**Additional benefit (Ellpack perf fix, 2026-04-20)**: on full bench,
FREUROTH (+51%) and BDEXP (+25%) regressed vs old Pivoted because
every different-cols `_add_rule` widen emits 2 `broadcast_in_dim` +
1 `concatenate` to stack two 1D k=1 values into `(n, 2)` 2D. Old
Pivoted flat-1D concat was 1 kernel. Tested `jnp.stack(axis=1)` as
fix — lowers to identical HLO. A deferred-Sum approach that stays
tuple-per-band through the widen and only stacks at scale-time
(where the fused k≥2 SIMD is worth the axis insert) would recover
these regressions without losing the high-k fused-mul win on
DIXMAAN-class problems.

**Affects**: new class in `_base.py`, new `_flush_sum` helper, ~8 rule
touches. Carefully: any path reading operand attributes directly (e.g.
`_linop_matrix_shape`) must handle Sum or be flushed first.

**Ordering vs #2 (CSR)**: CSR partially subsumes Sum — disjoint-range
union is natural in CSR. Start with CSR; land Sum only if CSR leaves
add-order sensitivities on the table.

### 2. Internal CSR representation for arrowhead / disjoint-row adds

**Motivation**: `Ellpack`'s `(start_row, end_row)` + uniform `k` forces
padding or BCOO promotion when rows have very different entry counts
(classic arrowhead: one dense row, many unit rows → padded `k = n`).
Currently `_scatter_add_rule` and mismatched-range `_add_rule` fall
through to BCOO concat, dropping structural information that downstream
rules could exploit.

**Proposal**: add a CSR-shaped LinOp class `Csr(row_ptr, col_ind,
values, out_size, in_size)` that stays structural through:

- `_scatter_add_rule` with arbitrary scatter targets
- `_add_rule` with disjoint `(start_row, end_row)` Ellpacks
- Arrowhead patterns (one dense row stitched onto a banded body)

**Also**: the final conversion path for `materialize(...,
format='csr')` lands naturally — direct hand-off to cuDSS without a
reconstruction pass.

**Affects**: new class in `_base.py`, new `_to_csr` helpers,
`_scatter_add_rule` emits `Csr` instead of BCOO, `_add_rule` gets a
"mixed-Ellpack → Csr" path, `materialize` accepts `format='csr'`.

**Win**: unlocks arrowhead problems that currently densify; enables
cuDSS-native output.

## Priority 1 — further structural wins

### 3. Structural upgrades for the remaining "densifying 9"

Per `RESEARCH_NOTES.md` §10 "Densifying vs structure-preserving" audit,
these rules unconditionally densify but have plausible structural
alternatives:

- `transpose`: swap `out_rows ↔ in_cols` on Ellpack / swap BCOO index
  columns (note: only fires on 2D+ output, which doesn't happen in
  R^n → R^n linearize-of-grad).
- ~~`reshape`~~: **partial (2026-04-22, commit `ccdcb38`+`d1c004e`)**
  — ConstantDiagonal/Diagonal reshape to `(*batch, n)` now emits a
  batched BEllpack encoding the reshape permutation
  (`in_cols[*batch_idx, r] = flat_index(*batch_idx, r)`). DRCAV1LQ
  runtime: 210 ms → 144 ms (30% on mac). **Not full closure** — the
  next op is 2D `slice` whose structural path in `_slice_rule` is
  1D-only, so the batched BEllpack densifies at slice (via the
  newly-vectorised `BEllpack.todense` batched path in `d1c004e`; the
  old Python-loop batched densify inflated NONMSQRT 2× before the fix
  and was a latent pre-existing bug). Full DRCAV closure needs
  batched support across `_slice_rule` (2D), `_pad_rule` (2D),
  `_add_rule` (13 differently-shifted batched BEllpacks from the
  biharmonic stencil). Likely subsumed by TODO #2 (CSR), which
  handles arbitrary row-column mappings natively and is dramatically
  cleaner for 2D-stencil patterns — see §3b DRCAV analysis.
- `reduce_sum`: **partial (2026-04-22)** —
  Diagonal/ConstantDiagonal/BEllpack row-sum (`axes=(0,)`, unbatched)
  emit the column-sum as a canonical (n,) ndarray linear form (commit
  `e3fa885`). BEllpack full-batch-axis reduction (`axes ==
range(n_batch)`) splits into per-batch slices and sums via
  `_add_rule`. **Still missing**: batched BEllpack out-axis-only
  reduction (`axes=(n_batch,)`) — the LUKSAN11-16 pattern where a
  batched BEllpack `bs=(m,), out=k, in=N` reduces the out axis to
  yield unbatched `(m, N)`. Current fallback densifies.
- `broadcast_in_dim`: BCOO can broadcast length-1 sparse dims (sparsify
  already supports this; length-≠1 fails upstream). **Partial (2026-
  04-22)**: scalar → (1,) now promotes to BCOO(1, n) or passes a
  BEllpack row-vector through. Trailing-singleton (`shape=(n,
1, ..., 1), bd=(0,)`) on unbatched BEllpack promotes to batched
  `bs=(n,), out=1` (2026-04-22, commit `0123250`). Length-≠1
  broadcast of sparse LinOps still densifies.
- `split`: **partial (2026-04-22)** — `axis=0` promotes operand to
  BCOO and masks entries outside each chunk's row-range. **Still
  missing**: non-zero axis splits on batched operands (LUKSAN's
  `split(axis=1)` on `(m, k)` batched BEllpack).

Low priority — the remaining gaps rarely fire on the curated set.
Revisit if a benchmark flags them.

### 3b. Close the asdex-bcoo gap on large sparse-Hessian problems

**Progress (2026-04-22, commit `87a30df`)**: 5/11 problems closed by
the linear-form structural walk, confirmed on the clean Linux sweep
(`.benchmarks/Linux-CPython-3.12-64bit/0029-0031_*_full_linux_*.json`):

| problem  | pre (Linux) | post (Linux) | asdex bcoo | post/asdex | status                          |
| -------- | ----------- | ------------ | ---------- | ---------- | ------------------------------- |
| LIARWHD  | 8,374 µs    | 49 µs        | 35 µs      | 1.40×      | **closed** (171× vs pre)        |
| EG2      | 591 µs      | 41 µs        | 46 µs      | 0.90×      | **closed, ahead** (14× vs pre)  |
| NONDQUAR | 19,558 µs   | 60 µs        | 73 µs      | 0.82×      | **closed, ahead** (326× vs pre) |
| FLETBV3M | 5,820 µs    | 82 µs        | 73 µs      | 1.12×      | **closed** (71× vs pre)         |
| FLETCBV3 | 5,940 µs    | 50 µs        | 79 µs      | 0.63×      | **closed, ahead** (118× vs pre) |
| BDQRTIC  | —           | 48,590 µs    | 157 µs     | 310×       | improved ~1.03× from pre, open  |
| BROYDN7D | —           | 27,531 µs    | 210 µs     | 131×       | open — needs Sum (TODO #1)      |
| DRCAV1LQ | —           | OOM (bench)  | 1,473 µs   | —          | open — needs Csr (TODO #2)      |
| DRCAV2LQ | —           | OOM (bench)  | 1,432 µs   | —          | open — needs Csr (TODO #2)      |
| NONMSQRT | —           | 16,006 µs    | 308 µs     | 52×        | open — densifies; needs trace   |
| RAYBENDL | —           | 4,465 µs     | 91 µs      | 49×        | open — complex tree; needs Sum  |

Bonus closures outside the original §3b list, same sweep:

- **ARWHEAD** 16,318 µs → 52 µs (314× vs pre)
- **INDEF** 35,007 µs → 184 µs (190× vs pre)
- **FLETCBV2** 5,552 µs → 70 µs (79× vs pre)

Root cause was **linear-form densification** at `add(linear_form,
vector-LinOp)` in the linearized grad. `slice(Identity, [0:1])`
emits a 1-row BEllpack; `_squeeze_rule` used to densify its row to
a `(n,)` ndarray, forcing the subsequent `add` onto the dense
fallback and bloating intermediates to `(n, n)`. Fixed by keeping
the BEllpack row-vector through squeeze, adding a broadcast-add
branch in `_add_rule` that tiles the sparse row to a column-constant
BEllpack of shape (m, n), and adding structural paths in
`_reduce_sum_rule` (Diagonal / ConstantDiagonal / BEllpack) and
`_broadcast_in_dim_rule` (scalar → (1,) → BCOO).

**Remaining open**:

- **DRCAV1LQ / DRCAV2LQ** — 2D cavity stencil, disjoint-row adds.
  Also OOMs pytest-benchmark at n=4489 (pre-existing, not our
  regression — confirmed at commit `8563bf9`). Root cause traced
  2026-04-22: first `reshape(ConstantDiagonal, (67, 67))` densifies
  the entire walk; see TODO #3 `reshape` entry for fix scope. Wants
  internal `Csr` (TODO #2) + batched BEllpack slice/pad/add paths.
- **BROYDN7D / RAYBENDL** — complex mul/add trees. Wants deferred
  `Sum` (TODO #1) for order-independent add-chain bucketing.
- **BDQRTIC** — still densifies further in. Next: single-rule trace.
- **NONMSQRT** — `mul(BEllpack, dense)` somewhere densifies. Next:
  single-rule trace.
- **SBRYBND** — 893 µs asdex, 4,725 µs ours (5.3×). Similar to
  BDQRTIC class; trace needed.

### 3c. Additional asdex-gap problems (not in original §3b)

Cross-checked 2026-04-22 via `lx_ratio_vs_jax > 1.2 * asdex_ratio_vs_jax`
on the full Linux sweep (commit `d1c004e`). Below are the 20+ problems
where lineaxpr is materially behind asdex-bcoo but weren't in §3b's
original list.

**Big-n jax-fails cluster** (jax.hessian times out at n=5000; ratios
are `lx/asdex_bcoo`). All plausible add-chain / complex-tree patterns,
same flavor as BROYDN7D:

| problem    | n    | lineaxpr  | asdex  | lx/as | category                      |
| ---------- | ---- | --------- | ------ | ----- | ----------------------------- |
| BDQRTIC    | 5000 | 52,135 µs | 137 µs | 380×  | (already in §3b)              |
| BROYDN3DLS | 5000 | 21,171 µs | 68 µs  | 313×  | Broyden tridiagonal — Sum/Csr |
| CRAGGLVY   | 5000 | 18,916 µs | 79 µs  | 239×  | Cragg-Levy — complex tree     |
| BROYDN7D   | 5000 | 26,863 µs | 183 µs | 147×  | (already in §3b)              |
| WOODS      | 4000 | 3,329 µs  | 28 µs  | 118×  | Wood's function — mul tree    |
| CHAINWOO   | 4000 | 5,204 µs  | 56 µs  | 93×   | Chained Wood — same as WOODS? |
| RAYBENDL   | 2050 | 4,460 µs  | 59 µs  | 76×   | (already in §3b; see below)   |
| SROSENBR   | 5000 | 3,217 µs  | 44 µs  | 74×   | Sum-of-Rosenbrocks            |
| NONMSQRT   | 4900 | 13,954 µs | 269 µs | 52×   | (already in §3b)              |
| TOINTGSS   | 5000 | 294 µs    | 117 µs | 2.5×  | small abs gap                 |
| BDEXP      | 5000 | 159 µs    | 68 µs  | 2.3×  | Boundary exp                  |
| GENHUMPS   | 5000 | 437 µs    | 221 µs | 2.0×  | Gen humps                     |
| FREUROTH   | 5000 | 144 µs    | 74 µs  | 2.0×  | Freudenstein-Roth             |
| SBRYBND    | 5000 | 1,621 µs  | 846 µs | 1.9×  | (already in §3b regression)   |

**Small-n LUKSAN cluster** (n≈100, all L-BFGS test problems). Ratios
are `lx/jax_min` vs `asdex/jax_min` — asdex is 3–7× further ahead of
jax than lineaxpr is. Trace of LUKSAN11LS shows the densifier is
`broadcast_in_dim[BEllpack → (n, 1, n)]` from `jnp.stack` pattern in
the objective (stack → concat → reshape chain in the linearized grad):

| problem    | n   | lineaxpr | jax    | asdex | lx/jax | as/jax | lx/as |
| ---------- | --- | -------- | ------ | ----- | ------ | ------ | ----- |
| LUKSAN11LS | 100 | 30 µs    | 42 µs  | 9 µs  | 0.71×  | 0.21×  | 3.4×  |
| LUKSAN12LS | 98  | 64 µs    | 77 µs  | 12 µs | 0.83×  | 0.15×  | 5.4×  |
| LUKSAN13LS | 98  | 39 µs    | 66 µs  | 10 µs | 0.59×  | 0.15×  | 4.0×  |
| LUKSAN14LS | 98  | 33 µs    | 52 µs  | 9 µs  | 0.63×  | 0.18×  | 3.5×  |
| LUKSAN15LS | 100 | 133 µs   | 153 µs | 50 µs | 0.87×  | 0.33×  | 2.7×  |
| LUKSAN16LS | 100 | 54 µs    | 72 µs  | 17 µs | 0.75×  | 0.23×  | 3.2×  |
| LUKSAN17LS | 100 | 241 µs   | 323 µs | 34 µs | 0.75×  | 0.11×  | 7.0×  |

Structurally: `jnp.stack([a, b])` decomposes to
`broadcast_in_dim(singleton) + concatenate(axis=1) + reshape(flatten)`.
Each operand is a BEllpack up to `broadcast_in_dim`, then densifies.
BEllpack path could work with 3 rule extensions (middle-singleton
broadcast, same-batch concat along out axis, batch+out flatten). CSR
path is trivial — the whole chain is metadata updates + triple-stack.

**Others**:

| problem  | n    | lineaxpr | jax      | asdex    | note                                |
| -------- | ---- | -------- | -------- | -------- | ----------------------------------- |
| TOINTGOR | 50   | 19 µs    | 23 µs    | 10 µs    | small, 1.9× lx/as-ratio             |
| EDENSCH  | 2000 | 42 µs    | 988 µs   | 26 µs    | jax.hessian slow — likely densifies |
| PENALTY3 | 200  | 8,598 µs | 6,612 µs | 6,941 µs | 1.24× lx/as-ratio, large absolute   |

RAYBENDL's hard step is the **strided slice** `y[::2]` / `y[1::2]` in
the first two primitives — `_slice_rule` only handles `strides=(1,)`,
so strided slices densify immediately and everything downstream is
dense. ~5 LoC extension to emit
`BEllpack(in_cols=(np.arange(start, limit, stride),), ...)` probably
closes it end-to-end. Independent of CSR.

**Summary**: CSR / deferred Sum address the big-n and LUKSAN clusters;
strided-slice fix closes RAYBENDL; LUKSAN could in principle stay
BEllpack but is much cleaner in CSR. Rough priority for the next
structural work:

1. Strided slice in `_slice_rule` (RAYBENDL, ~5 LoC, independent)
2. CSR (TODO #2) — unlocks big-n jax-fails, LUKSAN cluster, DRCAV
3. Deferred Sum (TODO #1) — may subsume some of BDQRTIC / BROYDN\* /
   WOODS / CHAINWOO / CRAGGLVY if CSR leaves add-order sensitivities.

### Loop 2026-04-22: structural concat + BCOO reshape + sentinel prune

Three incremental rule additions landed (with one reverted), each
validated in isolation and via full sweep + plots:

1. **Structural concat for BEllpacks (`2c601da`)** — all-traced
   concat along dim < n_batch or dim == n_batch, with band widening
   to max_k. Huge wins:
   - BROYDN3DLS: 20.5 ms → 128 µs (160×)
   - BDQRTIC: 47.9 ms → 6.7 ms (7.2×)
   - DIXMAAN[DFGHL,O]: ~90 µs → ~47 µs (≈2×) each
2. **BCOO reshape-flatten (`e4a5f6e`)** — final reshape in DRCAV
   flattens leading batch + out into a single out axis without
   densification:
   - DRCAV1LQ: 44 ms → 26 ms, now BCOO output (was dense)
   - DRCAV2LQ: 130 ms → 94 ms
3. **Sentinel-prune + flat BCOO from `_ellpack_to_bcoo_batched`
   (`c9cfcac`)** — static-prune `-1` sentinels at trace time, emit
   unbatched flat BCOO of shape `(prod(batch) * out, in)`. Also
   changed `_linop_matrix_shape(BEllpack)` to return the flat shape.
   - DRCAV1LQ: 26 ms → 16 ms (1.6× further, total 3.1× from 6fd1d15)
   - DRCAV2LQ: similar
4. **Relanded with fix (2026-04-22, `0123250`): BEllpack
   broadcast_in_dim trailing-singleton.** Prior attempt had been
   reverted for a CHARDIS0 correctness bug: two batched BEllpacks
   `(n, 1, m)` and `(1, n, m)` collided to the same flat
   `_linop_matrix_shape` `(n, m)` and were mis-combined by the
   BCOO-concat mix-path. Fix: added a batch-shape guard to the mix
   path — if multiple BEllpack operands have flat-equal shapes but
   different `batch_shape`s, reject and fall through to the dense
   reducer (which broadcasts correctly). Preserves DRCAV's
   sentinel-prune win. LUKSAN11-15LS picked up modest wins
   (12-49% range); LUKSAN16/17LS unchanged — downstream densifiers
   (split on axis≠0, reduce_sum on out-axis-only) still block full
   closure.
5. **Landed (2026-04-22, `b8d2bf8`): batch+out flatten in
   `_reshape_rule`.** Batched BEllpack → unbatched flat BCOO when
   the reshape fully flattens `(batch + out) → (prod(batch)*out,)`.
   Closes the `bid → concat → reshape(flatten)` trio of rule
   extensions needed for LUKSAN. No standalone perf delta today —
   the reshape no longer densifies but downstream rules (`split`
   axis=1, `reduce_sum` out-axis) still do. Kept as a prerequisite
   that unblocks those rules.

Remaining asdex-gap problems (after this loop) that aren't addressed
by any rule listed in §3 and remain truly CSR / Sum territory:

- LUKSAN11LS–17LS (`jnp.stack` pattern — bid+concat+reshape)
- NONMSQRT (`reduce_sum` partial-batch)
- WOODS (`squeeze` multi-dim)
- BROYDN7D, TOINTGOR (`select_n` with structural branches)

Plus diminishing-returns tail on TOINTGSS, BDEXP, GENHUMPS,
FREUROTH, SBRYBND, EDENSCH, etc. (all < 3× from asdex, small
absolute).

**Post-sweep fix (commit `25bd419`)**: the initial sweep flagged
ARGTRIGLS 2.25× slower than pre-change (98 → 222 µs). Root-caused
to the `_reduce_sum_rule(Diagonal) → op.values` shortcut breaking
XLA fusion (6 reduce-window kernel launches vs baseline's 3, despite
lower total flops). Since `reduce_sum(Diagonal(v))` always yields a
dense `(n,)` linear form anyway (no structural rep possible),
reverting the Diagonal/ConstantDiagonal branches of
`_reduce_sum_rule` — while keeping the BEllpack branch — fixes
ARGTRIGLS (recovered to 86.8 µs, faster than old baseline) AND
mildly improves FLETBV3M (82 → 54 µs) and FLETCBV2 (70 → 45 µs).
§3b closures preserved.

**Remaining smaller regressions vs old lineaxpr** (all still ahead
of or near asdex-bcoo):

- **SBRYBND** bcoo: 3,000 µs → 4,725 µs (1.58×). Still 5.3× behind
  asdex; no clean win lost. Likely tied to the same add-chain
  pattern as BDQRTIC.
- **GAUSS1LS** bcoo: 62 → 81 µs (1.32×). Small absolute, low signal.
- **NASH** bcoo (5.7 → 8.6 µs), **LEVYMONT** materialize (10 → 14
  µs) — too small to call real regressions; within cross-platform
  noise even on clean Linux.

**Compile-time note (2026-04-22)**: asdex runs `hessian_sparsity`
(per-primitive jaxpr index-set walk) + graph coloring, costing
~50–250 ms on most problems but blowing up on dependency-heavy
patterns: DMN15102LS 5.0s, DRCAV1LQ 1.4s, SBRYBND 890ms, PENALTY3
692ms. Lineaxpr compile is steady (median 81ms, p90 173ms) except
**NONDQUAR which hits 1290ms** (5× asdex). NONDQUAR runtime is now
good but compile is still a lineaxpr outlier — profile separately.

Post-closure break-even: on the 5 closed problems lineaxpr beats
asdex on _both_ compile and runtime. For the open 5 runtime losers,
asdex breaks even after ~4 Hessian evaluations until Sum/Csr land.

### 4. Upstream `add_any` / `pad` / `scatter-add` to `jax.experimental.sparse`

**Motivation**: `experiments/sparsify_monkeypatch.py` provides working
implementations. 14/16 CUTEst curated compile as pure-BCOO with these
three rules (+ fix for `bcoo_broadcast_in_dim` length-≠1 case for
HART6/ARGTRIGLS). Benefits the whole JAX ecosystem and makes
`sparsify(vmap(linearize(grad(f), y)[1]))(sparse.eye(n))` work for users
who don't need lineaxpr's specializations.

**Scope**: PR to `jax.experimental.sparse.transform` — ~200 LoC, mostly
mechanical. Out of scope for lineaxpr itself.

## Priority 2 — JAX idiom alignment

### 5. Use `jax.experimental.sparse` ops instead of hand-rolling

Current: `jnp.concatenate([v.data, v.indices])` for BCOO adds, manual
scatter for scale, etc. Lose `indices_sorted` / `unique_indices` metadata.

Proposal: delegate to `sparse.bcoo_concatenate`, `sparse.bcoo_multiply_dense`,
`sparse.bcoo_slice`, `sparse.bcoo_reduce_sum`, etc. Preserve metadata flags.

**Risk**: per-op perf validation needed (some sparse ops have overhead).

### 6. `safe_map` / `safe_zip` from `jax._src.util`

Replace raw `zip()` calls to catch length mismatches early.

### 7. Tracer mode for `sparsify`

Implement `LineaxprTrace(core.Trace)` / `LineaxprTracer` hooking
`process_primitive`, matching `jax.experimental.sparse.sparsify(...,
use_tracer=True)`. Lets `sparsify` compose with other transforms without
upfront `make_jaxpr`. Defer until a concrete use case emerges.

### 8a. jax-style kwargs on jacfwd/jacrev/hessian

Match `jax.jacfwd` / `jax.jacrev` / `jax.hessian`'s full signature:

- `argnums: int | Sequence[int]` — differentiate w.r.t. multiple args
- `has_aux: bool` — allow `f` to return `(output, aux)`; aux passes
  through untouched
- `holomorphic: bool` — complex-valued inputs/outputs
- `allow_int: bool` — allow integer inputs (where result is identically zero)

Today our wrappers hard-code `argnums=0`, single-output, float, no aux.
Low priority until a user hits one; the building block (`materialize` on
a traced linear_fn) doesn't care.

### 8b. Multi-input / multi-output + vmap composition

Currently `sparsify(linear_fn)(seed)` rejects multi-input and
multi-output linear fns. And the walk hasn't been tested against
`jax.vmap`-composed callers. For NeurIPS demos with more general
differentiation targets, either:

- extend the walk to handle multi-output linear fns (flatten outputs
  to a single 1D output, walk, reshape), OR
- document multi-output as a hard limit and require users to flatten
  themselves.

Also: `vmap(hessian(f))(batched_y)` should work in principle — the walk
happens once at trace time, then vmap batches the compiled result.
Verify + test; no rule changes expected.

### 9. `custom_jvp_call_p` rule

Audit sif2jax for `@jax.custom_jvp` usage; add a rule that `lift_jvp`'s
the custom-JVP'd function and re-runs through our walk. Sparsify has
this; we don't. (Confirmed 2026-04-20: sif2jax has 0 `custom_jvp`
calls, so this is not blocking. Revisit if a user reports it.)

### 10. Primitive-coverage gaps — 18 problems skipped by --full sweep

Full bench at commit `4562b8f` found 18 `bcoo_jacobian` compile failures
(NotImplementedError in walk). They form two clusters:

- **PALMER1A/1C/1D, PALMER2A/2C/2E, PALMER3A/3C/3E, PALMER4C/4E,
  PALMER6C/6E, PALMER7C/7E, PALMER8C/8E** — 17 problems. Polynomial
  curve-fitting to spectroscopy data. Likely one shared primitive
  missing across the family.
- **BQP1VAR** — bounded quadratic with n=1. Degenerate shape case.

Audit: set `_SMALL_N_VMAP_THRESHOLD = 0` and run one of them; the
improved NIE message names the missing primitive. Add the rule or
document the shape limitation.

### 10a. ARGLIN family — 6–45× slower than jax.hessian (folded)

The `--full` sweep (commit 4562b8f) + full-refs shows only 3 problems

> 2× slower than `min(jax.hessian folded, unfolded)`, all linear-
> regression quadratics:

ARGLINC n=200 44.5x slower (471µs vs 10.6µs)
ARGLINB n=200 23.0x slower (241µs vs 10.5µs)
ARGLINA n=200 12.0x slower (127µs vs 10.6µs)

Both `lineaxpr.hessian` and `lineaxpr.bcoo_hessian` are affected
(bcoo slightly worse). All are constant-H problems. `jax.hessian` with
`eager_constant_folding=True` folds the entire `AᵀA` matrix to a
literal (~10µs to memcpy 320KB into the runtime). Our walk computes
a dense ndarray at trace time but may not be emitting it as a folded
literal — need to audit the `dot_general(closure_matrix, Identity_seed)`
path.

Hypothesis: the walk produces `closure_matrix @ closure_matrix`-style
intermediates that are closure-only but still carry trace-time arith
ops, rather than a single concrete ndarray literal. Next steps:

1. Inspect the jaxpr our walk produces on ARGLINA.
2. Test `with eager_constant_folding(True)` around
   `lineaxpr.hessian(ARGLINA)` to see if XLA can fold what we emit.
3. If XLA folding doesn't help, force-fold at trace time via explicit
   `jnp.asarray(...)` on closure-produced arrays.

Not BandedPivoted-blocking; absolute times are modest (≤471µs).

## Priority 3 — memory / hygiene

### 11. Last-use analysis in `_walk_jaxpr`

`env` retains every intermediate LinOp until walk ends. For long jaxprs
with dense fallbacks, retained dense tensors can OOM. Compute `last_use`
per var, `del env[v]` after its last read.

### 12. Per-rule unit tests

`tests/test_rules.py` — isolates each rule with synthetic LinOp inputs.
Gates refactors more finely than end-to-end CUTEst correctness.

## Priority 4 — deferred / low ROI

### 13. Coloring-based alternative extractor

For small-n sparse problems, 3-HVP coloring (asdex's approach) beats our
structural walk (LEVYMONT: 24 µs asdex-bcoo vs 50 µs ours-bcoo).

Proposal: `materialize(fn, primal, mode="coloring")` as opt-in. Detect
pattern via one structural walk, color it, emit `vmap(hvp)(seeds)` +
gather.

Low priority because: only wins on small-n (asdex itself loses at large
n and on tiny/dense problems); adds substantial new machinery. Also
partly subsumed by Ellpack (for banded cases).

### 14. True `scan` structural support

All CUTEst scan-using problems pass via the n<16 short-circuit (PALMER
etc. are tiny). No large-n scan problems currently fail. Real support
would require body-unroll or vmap-over-axis; non-trivial. Wait for a
concrete failing problem to motivate.

### 15. Cross-eqn pattern matching for prod-tree

HS110's `(∏x)^k` HVP fragments into many small Ellpacks that our walk
processes one-by-one. asdex also doesn't fix this. The analytical form
is `α · uuᵀ + diag(...)`. Detecting this from the jaxpr structure would
need multi-eqn lookahead — substantial new machinery. Currently handled
acceptably by the n<16 short-circuit.

### 16. Symmetric output option

`materialize(fn, primal, symmetric=False)` that computes lower triangle
and mirrors. ~30% dense-path savings. No BCOO win.

Complication: requires detecting/asserting symmetry — for Hessians it's
guaranteed, but adds a new API surface.

## Priority 5 — cleanup

### 17. `_dot_general_rule` variable renames

`c_tr`, `c_M`, `cx`, `cy` → `traced_contract`, `closure_contract`, etc.
Readability across ~80 lines of that rule.

### 18. Honest README reframe

Per `RESEARCH_NOTES.md` §10 monkeypatch findings: headline should be
"2–4× over pure-BCOO sparsify on y-dependent problems + robust handling
of upstream-sparsify primitive gaps (HART6, ARGTRIGLS); const-H wins
over jax.hessian depend on `EAGER_CONSTANT_FOLDING` regime and
closure-vs-input placement of y." Not the "100-6000×" story the
benchmarks-vs-jax.hessian alone would suggest.

## Recently landed

### Linear-form preservation through squeeze + broadcast-add (2026-04-22)

Closed 5/11 §3b problems (LIARWHD, EG2, NONDQUAR, FLETBV3M, FLETCBV3)
— now matching or slightly ahead of asdex-bcoo. Previous behaviour:
`squeeze(BEllpack(1, n))` densified the row to a `(n,)` ndarray; the
subsequent `add(linear_form, Diagonal)` broadcast fell to the dense
fallback and allocated an `(n, n)` intermediate, cascading the walk
onto the dense path. Fix:

- `_squeeze_rule`: keep the BEllpack row-vector (shape (1, n)) for
  sparse linear forms (from 1-row slice/gather).
- `_add_rule`: new `_tile_1row_bellpack` helper + broadcast-add
  branch — tiles the BEllpack row-vector to a column-constant
  BEllpack of shape (m, n) where each band's in_cols is
  `np.full(m, col_j)`, values broadcasted. Merges with the matrix
  operand via the existing same-range band-widening path.
- `_add_rule` final normalisation: before the dense fallback, coerces
  mixed linear-form operand representations (`(n,)` ndarray vs
  BEllpack row-vector vs BCOO row-vector) to canonical `(n,)` ndarray
  so numpy-broadcast doesn't leak an ill-shaped `(1, n)`.
- `_reduce_sum_rule`: structural paths for `Diagonal` /
  `ConstantDiagonal` (diag values as the linear-form's row
  coefficients) and unbatched `BEllpack` via new `_bellpack_row_sum`
  scatter-add. Avoids the `(out_size, in_size)` dense materialisation.
- `_broadcast_in_dim_rule`: scalar-aval → (1,) passes a BEllpack
  row-vector through unchanged (shape already (1, n)) or promotes a
  canonical `(n,)` ndarray linear form to `BCOO(1, n)` so the
  subsequent `pad` stays structural.

Headline: LIARWHD 8,738µs Linux → 43µs mac (BCOO). Also follow-up
commit `af5d7ba` renames "1-row BEllpack" to "BEllpack row-vector"
in comments for clarity — what's stored is semantically a sparse
row vector (Jacobian of a scalar-aval variable), not a matrix with
one populated row.

### BEllpack + SPARSINE dense-fallback closed (2026-04-21)

`Ellpack` → `BEllpack` mirroring JAX's `BCOO`/`BCSR` batching
convention (`shape = (*batch_shape, out_size, in_size)`, `n_batch`
attribute). Unbatched case unchanged. `_gather_rule` on diagonal
operands with multi-dim indices emits a batched BEllpack;
`_reduce_sum_rule` collapses batch dims by summing slices;
`_broadcast_in_dim_rule` adds leading axes to batch_shape;
`_scatter_add_rule` handles batched updates via per-slice BCOO
remap + concat. SPARSINE at n=5000 went from 183ms (dense fallback)
to 360µs (~500× speedup). Also: `_mul_rule` now guards
`scale_per_out_row` with a shape-compat check so multi-dim scale
patterns (outer-product in jaxpr) fall back to dense rather than
raise — fixed a PENALTY3 regression uncovered by the full sweep.

### `_add_rule` Diagonal/ConstantDiagonal absorption into Ellpack bands (2026-04-21)

Mix of `{ConstantDiagonal, Diagonal, Ellpack}` at matching square
shape now promotes diagonals to Ellpack bands and merges via the
standard same-range path, avoiding BCOO promote. Prerequisite for
BEllpack work — keeps the walk in Ellpack through `Diagonal + Ellpack`
adds that are common in HVP patterns.

### Ellpack replaces Pivoted (2026-04-20)

Pivoted's "at most one nonzero per row" representation is replaced by
Ellpack: `(start_row, end_row)` contiguous row range + tuple of bands
`(in_cols, values)` where `values` is a uniform 2D `(nrows, k)` array.
`in_cols` entries of `-1` are sentinels for padding. Same-range adds
extend the bands tuple rather than concatenating entry arrays; same-
`in_cols` adds sum values (old Pivoted same-indices fast path). Rules
touched: `_slice_rule`, `_pad_rule`, `_gather_rule`, `_scatter_add_rule`
(promotes to BCOO since non-contiguous output rows), `_add_rule`
(new same-range / same-cols paths), `_mul_rule`, `_neg_rule`.
Motivated the subsequent SPARSINE and CSR-internal work above.

### `_add_rule` kind-dispatch refactor (pre-Ellpack)

Compressed 7 cascading `all(isinstance...)` passes into 4 kind-set
checks + a shared `_bcoo_concat` helper + `_linop_matrix_shape`.
