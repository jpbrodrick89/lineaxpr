# lineaxpr — notes for Claude

## What this is
A JAX transformation that extracts dense / sparse Hessian/Jacobian matrices
from linear callables (the output of `jax.linearize`). Walks the linear
jaxpr once with per-primitive structural rules.

Public API:
- `materialize(linear_fn, primal) -> jnp.ndarray` — always dense.
- `bcoo_jacobian(linear_fn, primal) -> BCOO | jnp.ndarray` — BCOO when
  sparse structure survives; dense otherwise.
- `sparsify(linear_fn)(seed_linop) -> LinOp` — primitive transform. No
  implicit Identity cast; caller provides the seed.
- `to_dense(linop)` / `to_bcoo(linop)` — format conversion helpers that
  work uniformly on our LinOp classes, BCOO, and plain ndarrays.
- `Identity(n, dtype=...)` — standard seed constructor for `sparsify`.
- `ConstantDiagonal`, `Diagonal`, `Pivoted` — LinOp classes (exposed for
  tests/debugging and custom seeds; not a pytree-registered API).

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
  `.tobcoo()`, `.negate()`, `.scale_scalar(s)`, `.scale_per_out_row(v)`,
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

# Benchmarks (local):
uv run pytest benchmarks/ --benchmark-only

# In-container benchmarks (matches CI config):
bash benchmarks/run_in_container.sh benchmarks/test_curated.py --benchmark-save=0001_curated

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
  coverage_sweep.py         # legacy standalone sweep (pre-pytest version)
benchmarks/
  test_curated.py           # curated comparison bench
  test_full.py / test_highn.py  # broader perf sweeps
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
