# TODO — prioritized

Empirically-grounded; see `RESEARCH_NOTES.md` for the reasoning.

## Priority 0 — next up

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
- `reshape`: map linear → multi-dim indices on BCOO when shape is
  preserved structurally.
- `reduce_sum`: sum along axis of BCOO → smaller BCOO; Ellpack row-sum
  → Diagonal.
- `broadcast_in_dim`: BCOO can broadcast length-1 sparse dims (sparsify
  already supports this; length-≠1 fails upstream).
- `split`: partition entries by index along split axis.

Low priority — these rarely fire on the curated set. Revisit if a
benchmark flags them.

### 3b. Close the asdex-bcoo gap on large sparse-Hessian problems

**Motivation**: clean Linux sweep (2026-04-21) — 3 reps, min-of-mins:

| problem  | lineaxpr bcoo | asdex bcoo | ratio |
| -------- | ------------- | ---------- | ----- |
| BDQRTIC  | 50,009 µs     | 137 µs     | 364×  |
| NONDQUAR | 17,149 µs     | 60 µs      | 284×  |
| BROYDN7D | 26,970 µs     | 183 µs     | 147×  |
| DRCAV1LQ | 194,119 µs    | 1,369 µs   | 142×  |
| DRCAV2LQ | 193,718 µs    | 1,380 µs   | 140×  |
| FLETBV3M | 5,135 µs      | 73 µs      | 70×   |
| FLETCBV3 | 5,257 µs      | 79 µs      | 67×   |
| NONMSQRT | 15,888 µs     | 269 µs     | 59×   |
| LIARWHD  | 8,738 µs      | 35 µs      | 250×  |
| RAYBENDL | 4,410 µs      | 72 µs      | 61×   |
| EG2      | 489 µs        | 46 µs      | 11×   |

All are `test_bcoo_jacobian` (BCOO build path). asdex evaluates a small
number of JVPs from its coloring and fills the BCOO directly; lineaxpr's
walk emits one linop per primitive and flushes to BCOO at densification.
The gap suggests: (a) the walk is doing more work than necessary when
the whole tree is already structurally sparse, and/or (b) the final
BCOO assembly has overhead that dominates for simple patterns. LIARWHD
is pure-diagonal — asdex ships a 35 µs answer where we ship 8.7 ms.

**Next step**: profile the BCOO emission path for LIARWHD specifically
(simplest case) — is it the walk, the `_to_bcoo` flush, or the
`concatenate` at the output? Compare HLO against what asdex emits.

**Compile-time note (2026-04-22)**: the runtime gap comes with a
compile-time trade-off. asdex runs its `hessian_sparsity` (a per-primitive
jaxpr index-set walk, conceptually similar to lineaxpr's walk) followed
by graph coloring. That costs ~50-250ms on most problems, but blows up
when the pattern has many dependencies: DMN15102LS 5.0s, DRCAV1LQ 1.4s,
SBRYBND 890ms, PENALTY3 692ms. Lineaxpr compile is steady (median 81ms,
p90 173ms) except **NONDQUAR which hits 1290ms** — 5× asdex on the same
problem. NONDQUAR is a lineaxpr compile outlier to profile alongside
the runtime gap. Asdex has no fast path for low-sparsity cases (e.g.
pure-diagonal LIARWHD still triggers full detection+coloring, 156ms).

Break-even calculation: on §3b-gap problems asdex pays ~100-400ms more
compile for a 100-400× runtime win. For BDQRTIC (380ms compile, 137µs
runtime vs lineaxpr 209ms compile, 50ms runtime): asdex breaks even
after ~4 hessian evaluations. For single-shot hessian use cases,
lineaxpr may still be faster total.

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
