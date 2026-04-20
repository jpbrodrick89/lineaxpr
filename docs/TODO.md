# TODO — prioritized

Empirically-grounded; see `RESEARCH_NOTES.md` for the reasoning.

## Priority 0 — next up

### 1. SPARSINE — static gather densifies (should stay sparse)

Full sweep flagged **SPARSINE at n=5000: 183ms bcoo_jacobian**. The
problem uses a static gather rather than slice; `_gather_rule` has
narrower structural support than `_slice_rule` and hits the dense
fallback. The output SHOULD be sparse.

**Scope**: broaden `_gather_rule`'s `dnums` guard at
`lineaxpr/materialize.py:_gather_rule` to emit `Ellpack` for the gather
patterns SPARSINE actually uses (likely non-scalar `slice_sizes` or
different `offset_dims`/`collapsed_slice_dims`). The `ConstantDiagonal`
→ `Ellpack` path already handles the 1D indexed case; extend to the
patterns that currently raise `NotImplementedError`.

**Win**: closes SPARSINE regression; small code change; high leverage.

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
