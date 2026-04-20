# TODO — prioritized

Empirically-grounded; see `RESEARCH_NOTES.md` for the reasoning.

## Priority 0 — next up (post-refactor)

### 1. Banded Pivoted via `(out_rows, list[(in_cols, values)])`

**Motivation** (reinforced by `RESEARCH_NOTES.md` §10 monkeypatch): on
LEVYMONT, pure-BCOO sparsify BEATS us (17µs vs 20µs). We also lose to
asdex on the same problem (24µs vs 50µs). Banded Hessians are the main
regression class; BandedPivoted directly addresses it.

**Context**: 15/20 observed `add_any(Pivoted, Pivoted)` cases in our
benchmark set have **same out_rows, different in_cols** (diagonal +
off-diagonal band patterns in LEVYMONT, FLETCHCR, DIXMAAN). Current impl
concatenates entries → `nse` doubles per-row for bandwidth-2 Hessians.

**Proposal**: when add_any detects same out_rows with different in_cols,
produce a "multi-column Pivoted" that stores one out_row array and a list
of `(in_cols, values)` pairs. Matvec: `sum(v * values[i] for each col)`.
Densify: scatter per pair.

**Affects**: `_add_like` (new fast path), new class in `_base.py` with
`.to_dense`/`.to_bcoo`/`.negate`/`.scale_scalar`/`.scale_per_out_row`
methods mirroring Pivoted. After the Step-3 method extraction, the only
rule touch is `_add_like`.

**Win**: ~2× BCOO size reduction on banded Hessians. Closes LEVYMONT
regression. Moves us closer to CSR-native output.

### 2. Upstream `add_any` / `pad` / `scatter-add` to `jax.experimental.sparse`

**Motivation**: `experiments/sparsify_monkeypatch.py` provides working
implementations. 14/16 CUTEst curated compile as pure-BCOO with these
three rules (+ fix for `bcoo_broadcast_in_dim` length-≠1 case for
HART6/ARGTRIGLS). Benefits the whole JAX ecosystem and makes
`sparsify(vmap(linearize(grad(f), y)[1]))(sparse.eye(n))` work for users
who don't need lineaxpr's specializations.

**Scope**: PR to `jax.experimental.sparse.transform` — ~200 LoC, mostly
mechanical. Out of scope for lineaxpr itself.

## Priority 1 — further structural wins

### 3. Range-based `out_rows` when no scatter has fired

**Context**: before any `scatter_add`, `out_rows` is always a concatenation
of contiguous ranges (arange(k) + offsets from pad). We currently
materialize these as full arrays.

**Proposal**: represent `out_rows` as `list[(start, stop)]` until the
first `scatter_add` forces materialization. Benefits:
- Trace-time state: O(#ranges) vs O(k) ints
- O(1) equality for same-indices fast path
- Natural merging of adjacent ranges

**Affects**: `Pivoted` class (new field type), all rules that construct
or mutate `out_rows`.

**Win**: minor trace-time speedup; enables more accurate same-indices
detection.

### 4. Static-numpy filter in slice/pad negative-range paths

**Context**: current `slice(Pivoted)` and `pad(Pivoted, negative)` use
`values * mask` to zero-out-of-range entries but keep them in the Pivoted.
This bloats `nse` permanently.

**Proposal**: when `out_rows` is static numpy, use `np.nonzero(mask)[0]` to
actually filter entries. Values array shrinks via `jnp.take(values, keep)`.

**Affects**: `_slice_rule`, `Pivoted.pad_rows`.

**Win**: genuine `nse` compression; real runtime savings downstream.

### 5. Structural upgrades for the remaining "densifying 9"

Per `RESEARCH_NOTES.md` §10 "Densifying vs structure-preserving" audit,
these rules unconditionally densify but have plausible structural
alternatives:

- `transpose`: swap `out_rows ↔ in_cols` on Pivoted / swap BCOO index
  columns (note: only fires on 2D+ output, which doesn't happen in
  R^n → R^n linearize-of-grad).
- `reshape`: map linear → multi-dim indices on BCOO when shape is
  preserved structurally.
- `reduce_sum`: sum along axis of BCOO → smaller BCOO; Pivoted → Diagonal.
- `broadcast_in_dim`: BCOO can broadcast length-1 sparse dims (sparsify
  already supports this; length-≠1 fails upstream).
- `split`: partition entries by index along split axis.

Low priority — these rarely fire on the curated set. Revisit if a
benchmark flags them.

## Priority 2 — JAX idiom alignment

### 6. Use `jax.experimental.sparse` ops instead of hand-rolling

Current: `jnp.concatenate([v.data, v.indices])` for BCOO adds, manual
scatter for scale, etc. Lose `indices_sorted` / `unique_indices` metadata.

Proposal: delegate to `sparse.bcoo_concatenate`, `sparse.bcoo_multiply_dense`,
`sparse.bcoo_slice`, `sparse.bcoo_reduce_sum`, etc. Preserve metadata flags.

**Risk**: per-op perf validation needed (some sparse ops have overhead).

### 7. `safe_map` / `safe_zip` from `jax._src.util`

Replace raw `zip()` calls to catch length mismatches early.

### 8. Tracer mode for `sparsify`

Implement `LineaxprTrace(core.Trace)` / `LineaxprTracer` hooking
`process_primitive`, matching `jax.experimental.sparse.sparsify(...,
use_tracer=True)`. Lets `sparsify` compose with other transforms without
upfront `make_jaxpr`. Defer until a concrete use case emerges.

### 9. `custom_jvp_call_p` rule

Audit sif2jax for `@jax.custom_jvp` usage; add a rule that `lift_jvp`'s
the custom-JVP'd function and re-runs through our walk. Sparsify has
this; we don't.

## Priority 3 — memory / hygiene

### 10. Last-use analysis in `_walk_jaxpr`

`env` retains every intermediate LinOp until walk ends. For long jaxprs
with dense fallbacks, retained dense tensors can OOM. Compute `last_use`
per var, `del env[v]` after its last read.

### 11. `_add_like` kind-dispatch refactor

Currently 5 cascading `all(isinstance...)` passes. Compute
`kinds = tuple(type(v) for v in vals)` once, dispatch by membership.
Cleaner; no runtime effect (trace-time only).

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
somewhat subsumed by BandedPivoted (#1).

### 14. True `scan` structural support

All CUTEst scan-using problems pass via the n<16 short-circuit (PALMER
etc. are tiny). No large-n scan problems currently fail. Real support
would require body-unroll or vmap-over-axis; non-trivial. Wait for a
concrete failing problem to motivate.

### 15. Cross-eqn pattern matching for prod-tree

HS110's `(∏x)^k` HVP fragments into many small Pivoteds that our walk
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

### 18. Delete dead `_slice_rule` Pivoted branch

Empirically `slice(Pivoted)` never fires in our problem set (only
`slice(ConstantDiagonal)` does). Can delete until a use case appears.

### 19. Honest README reframe

Per `RESEARCH_NOTES.md` §10 monkeypatch findings: headline should be
"2–4× over pure-BCOO sparsify on y-dependent problems + robust handling
of upstream-sparsify primitive gaps (HART6, ARGTRIGLS); const-H wins
over jax.hessian depend on `EAGER_CONSTANT_FOLDING` regime and
closure-vs-input placement of y." Not the "100-6000×" story the
benchmarks-vs-jax.hessian alone would suggest.
