# TODO — prioritized

Empirically-grounded; see `RESEARCH_NOTES.md` for the reasoning.

## Priority 1 — structural wins

### 1. Banded Pivoted via `(out_rows, list[(in_cols, values)])`

**Context**: 15/20 observed `add_any(Pivoted, Pivoted)` cases in our
benchmark set have **same out_rows, different in_cols** (diagonal +
off-diagonal band patterns in LEVYMONT, FLETCHCR, DIXMAAN). Current impl
concatenates entries → `nse` doubles per-row for bandwidth-2 Hessians.

**Proposal**: when add_any detects same out_rows with different in_cols,
produce a "multi-column Pivoted" that stores one out_row array and a list
of `(in_cols, values)` pairs. Matvec: `sum(v * values[i] for each col)`.
Densify: scatter per pair.

**Affects**: `_add_like` (new fast path), `_to_dense`, `_to_bcoo` (with
`nse * num_bands` entries), densification rule for BCOO concat.

**Win**: ~2× BCOO size reduction on banded Hessians. Moves us closer to
CSR output. Also naturally expressible as CSR if we want to expose it.

### 2. Range-based `out_rows` when no scatter has fired

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

### 3. Static-numpy filter in slice/pad negative-range paths

**Context**: current `slice(Pivoted)` and `pad(Pivoted, negative)` use
`values * mask` to zero-out-of-range entries but keep them in the Pivoted.
This bloats `nse` permanently.

**Proposal**: when `out_rows` is static numpy, use `np.nonzero(mask)[0]` to
actually filter entries. Values array shrinks via `jnp.take(values, keep)`.

**Affects**: `_slice_rule`, `_pad_rule`.

**Win**: genuine `nse` compression; real runtime savings downstream.

## Priority 2 — JAX idiom alignment (for eventual upstreaming)

### 4. Use `jax.experimental.sparse` ops instead of hand-rolling

Current: `jnp.concatenate([v.data, v.indices])` for BCOO adds, manual
scatter for scale, etc. Lose `indices_sorted` / `unique_indices` metadata.

Proposal: delegate to `sparse.bcoo_concatenate`, `sparse.bcoo_multiply_dense`,
`sparse.bcoo_slice`, `sparse.bcoo_reduce_sum`, etc. Preserve metadata flags.

**Risk**: per-op perf validation needed (some sparse ops have overhead).

### 5. Single-env walk pattern (like `core.eval_jaxpr`)

Currently split `env` (traced) + `consts_env` (non-traced). JAX uses a
single env with type-based distinction. Cleaner for `_jit_rule` recursion.

### 6. `safe_map` / `safe_zip` from `jax._src.util`

Replace raw `zip()` calls to catch length mismatches early.

## Priority 3 — memory / hygiene

### 7. Last-use analysis in `_walk_jaxpr`

`env` retains every intermediate LinOp until walk ends. For long jaxprs
with dense fallbacks, retained dense tensors can OOM. Compute `last_use`
per var, `del env[v]` after its last read.

### 8. `_add_like` kind-dispatch refactor

Currently 5 cascading `all(isinstance...)` passes. Compute
`kinds = tuple(type(v) for v in vals)` once, dispatch by membership.
Cleaner; no runtime effect (trace-time only).

## Priority 4 — deferred / low ROI

### 9. Coloring-based alternative extractor

For small-n sparse problems, 3-HVP coloring (asdex's approach) beats our
structural walk (LEVYMONT: 24 µs asdex-bcoo vs 50 µs ours-bcoo).

Proposal: `materialize(fn, primal, mode="coloring")` as opt-in. Detect
pattern via one structural walk, color it, emit `vmap(hvp)(seeds)` +
gather.

Low priority because: only wins on small-n (asdex itself loses at large
n and on tiny/dense problems); adds substantial new machinery.

### 10. True `scan` structural support

All CUTEst scan-using problems pass via the n<16 short-circuit (PALMER
etc. are tiny). No large-n scan problems currently fail. Real support
would require body-unroll or vmap-over-axis; non-trivial. Wait for a
concrete failing problem to motivate.

### 11. Cross-eqn pattern matching for prod-tree

HS110's `(∏x)^k` HVP fragments into many small Pivoteds that our walk
processes one-by-one. asdex also doesn't fix this. The analytical form
is `α · uuᵀ + diag(...)`. Detecting this from the jaxpr structure would
need multi-eqn lookahead — substantial new machinery. Currently handled
acceptably by the n<16 short-circuit.

### 12. Symmetric output option

`materialize(fn, primal, symmetric=False)` that computes lower triangle
and mirrors. ~30% dense-path savings. No BCOO win.

Complication: requires detecting/asserting symmetry — for Hessians it's
guaranteed, but adds a new API surface.

## Priority 5 — cleanup

### 13. File split

Current `materialize.py` is 1000 lines. Natural splits:
- `forms.py` — ConstantDiagonal, Diagonal, Pivoted + helpers
- `rules.py` — per-primitive rules
- `materialize.py` — driver + public API

Deferred because "design will shift as we go" (per user) and single-file
is simpler for exploration.

### 14. `_dot_general_rule` variable renames

`c_tr`, `c_M`, `cx`, `cy` → `traced_contract`, `closure_contract`, etc.
Readability across ~80 lines of that rule.

### 15. Delete dead `_slice_rule` Pivoted branch

Empirically `slice(Pivoted)` never fires in our problem set (only
`slice(ConstantDiagonal)` does). Can delete until a use case appears.
