# lineaxpr — notes for Claude

## What this is
A JAX transformation that extracts dense / sparse Hessian/Jacobian matrices
from linear callables (the output of `jax.linearize`). Walks the linear
jaxpr once with per-primitive structural rules.

Public API (jax-like, preferred):
- `hessian(f)(y)`, `bcoo_hessian(f)(y)` — matches `jax.hessian`.
- `jacfwd(f)(y)`, `bcoo_jacfwd(f)(y)` — matches `jax.jacfwd`.
- `jacrev(f)(y)`, `bcoo_jacrev(f)(y)` — matches `jax.jacrev`.
  All six accept `format='dense'|'bcoo'` as a kwarg too; the
  `bcoo_`-prefixed variants are shorthand for `format='bcoo'`.

Lower-level building blocks:
- `materialize(linear_fn, primal, format='dense'|'bcoo')` — core
  entry point. Default `'dense'`.
- `sparsify(linear_fn)(seed_linop) -> LinOp` — primitive transform.
  No implicit Identity cast; caller provides the seed.
- `to_dense(linop)` / `to_bcoo(linop)` — format conversion helpers
  that work uniformly on LinOp classes, BCOO, and plain ndarrays.
- `Identity(n, dtype=...)` — standard seed for `sparsify`.
- `ConstantDiagonal`, `Diagonal`, `Pivoted` — LinOp classes (exposed
  for tests/debugging and custom seeds; not a pytree-registered API).

Current limitations (TODO #9c/#9d): no argnums, no has_aux, single
input, single output, 1D primal. `jax.vmap` composition not yet
verified.

## Testing philosophy

Tests should call the public jax-like API (`hessian`, `jacfwd`,
`jacrev`, and `bcoo_*` variants) wherever possible. Reserve direct
`materialize` / `sparsify` calls for:

- `tests/test_ops.py` — LinOp class methods (unit).
- `tests/test_sparsify.py` — transform-level behavior (custom seeds,
  const-prop, missing-primitive errors). `sparsify` IS the thing under
  test; public API would just be indirection.
- `tests/test_public_api.py` — unit tests of the `format` kwarg,
  invalid-format error, default-dense behavior.

Everything else (Hessian correctness, sweep, hand-rolled problems)
should go through the public API since that's how users exercise the
library.

## Non-goals
- Coloring-based extraction (that's asdex's approach; we do per-linearization
  structural walk instead, yielding exact — not conservative — sparsity).
- Second AD pass (we stay inside `jax.linearize`'s output).
- A full linear-operator algebra library like lineax (name inspired by it,
  but we're narrower in scope).
- Migrating onto `jax.experimental.sparse.sparsify`'s framework — see
  `docs/RESEARCH_NOTES.md` §10. We mirror its patterns (direct-assignment
  rule registry, single-env walk) but keep our own mixed-format space.

## Key invariants

- The walk produces **bit-exact** output (vs `vmap(hvp)(eye)` reference).
  Any rule change must preserve this. `tests/test_correctness.py` is the
  ground truth; `tests/test_materialize.py` pins explicit expected
  matrices for hand-rolled problems.
- LinOp forms live in `lineaxpr/_base.py` with methods `.todense()`,
  `.to_bcoo()`, `.negate()`, `.scale_scalar(s)`, `.scale_per_out_row(v)`,
  `.primal_aval()` (and `Pivoted.pad_rows(lo, hi)`). Rules dispatch to
  these methods rather than branching on LinOp type inside every rule.
- Primitive rules are registered by direct dict assignment:
  `materialize_rules[lax.PRIM_p] = _rule_fn`. Each rule receives
  `(invals, traced, n, **params)` and returns a LinOp (or list for
  multi-result primitives). No `@register` decorator — it was ceremony.
- Walk env is `dict[Var, tuple[bool, Any]]`. The `bool` is `traced`; the
  `Any` is a LinOp (traced) or concrete array (closure). Walker emits
  `(False, value)` via constant-propagation for eqns with all-closure
  inputs — load-bearing for constant-H problems (DUAL, CMPC) where it
  produces a trace-time BCOO literal.
- nse regression is pinned in `tests/nse_manifest.json` — increases in
  `bcoo_jacobian(...).nse` fail the sweep; decreases require a manifest
  bump via `uv run python -m tests.update_nse_manifest`.

## Benchmarks

Run with the docker runner (see `benchmarks/run_in_container.sh`) to get
the EAGER_CONSTANT_FOLDING=TRUE regime that matches JAX's release config.

**Critical**: always pass the linearization point `y` as a `jit` INPUT,
never a closure. Otherwise `jax.hessian`'s output gets constant-folded and
spurious speedups appear. See `docs/RESEARCH_NOTES.md` §10 for the
full picture on how EAGER_CONSTANT_FOLDING + closure-y interact.

## Useful quick commands

```bash
# Fast tests (unit + hand-rolled + transform-level + curated CUTEst):
uv run pytest tests/ -v

# Slow sweep over all sif2jax problems (~200, ~2 min):
uv run pytest tests/test_sif2jax_sweep.py -m slow -v

# Regenerate the nse manifest after an intentional structural change:
uv run python -m tests.update_nse_manifest

# Per-commit lineaxpr bench (materialize + bcoo_jacobian only, commit-tagged):
bash benchmarks/run_bench.sh

# Reference bench (jax.hessian / asdex; tagged by JAX version, rerun only on upstream update):
bash benchmarks/run_bench.sh --refs

# Full sweep: lineaxpr on all ~275 sif2jax problems (~10 min, per-commit):
bash benchmarks/run_bench.sh --full

# Full refs sweep: jax.hessian + asdex across all problems (~30 min, rarely rerun):
bash benchmarks/run_bench.sh --full-refs

# View combined report with min(folded, unfolded) as the jax baseline:
uv run python -m benchmarks.report --tag lineaxpr   # curated table
uv run python -m benchmarks.report --tag full --summary  # full with aggregate stats

# Other modes: --curated (5-way), --highn (n>2500)
# Add USE_CONTAINER=1 for EAGER_CONSTANT_FOLDING=TRUE release parity.

# Monkeypatch experiment (pure-BCOO comparison):
uv run python -m experiments.run_monkeypatch
```

## Repo layout

```
lineaxpr/
  __init__.py           # public API exports
  _base.py              # LinOp classes + densify helpers
  materialize.py        # sparsify transform, rule registry, rules, public wrappers
experiments/
  sparsify_monkeypatch.py   # adds add_any/pad/scatter-add rules to jax.experimental.sparse
  run_monkeypatch.py        # reports pure-BCOO vs lineaxpr timings
tests/
  conftest.py               # enables jax_enable_x64
  test_ops.py               # unit tests for LinOp methods
  test_materialize.py       # hand-rolled problems with explicit expected matrices
  test_sparsify.py          # transform-level tests (seeds, const-prop, jit, errors)
  test_correctness.py       # curated CUTEst smoke tests
  test_sif2jax_sweep.py     # slow sweep + nse regression
  nse_manifest.json         # golden nse values per problem
  update_nse_manifest.py    # regenerator
benchmarks/
  run_bench.sh              # commit-tagged bench runner (lineaxpr/refs/curated/highn/full)
  test_curated.py           # 16-problem curated bench (5-way comparison)
  test_full.py              # all sif2jax scalar problems (DENSE_MAX=2000, BCOO_MAX=5000)
  test_highn.py             # n > 2500 sweep
  run_in_container.sh       # docker runner for EAGER_CONSTANT_FOLDING parity
docs/
  ARCHITECTURE.md           # walk algorithm, LinOp forms, rule pattern
  RESEARCH_NOTES.md         # empirical findings, sparsify comparison, DUAL deep-dive
  TODO.md                   # prioritized future work (BandedPivoted is #1)
```

## References

- Historical gist (pre-repo exploration):
  https://gist.github.com/jpbrodrick89/a3657522e7218d2cc98dae9f80258216
- Comparison library — asdex (coloring-based):
  https://github.com/adrhill/asdex
- Upstream reference — `jax.experimental.sparse.sparsify`:
  `.venv/lib/python3.13/site-packages/jax/experimental/sparse/transform.py`

## Documents to read

- `docs/ARCHITECTURE.md` — data structures + walk algorithm + LinOp methods
- `docs/RESEARCH_NOTES.md` — empirical findings (monkeypatch §9–§10 is key)
- `docs/TODO.md` — prioritized future work
