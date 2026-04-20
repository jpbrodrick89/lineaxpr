# lineaxpr — notes for Claude

## What this is
A JAX transformation that extracts dense / sparse Hessian/Jacobian matrices
from linear callables (the output of `jax.linearize`). Walks the linear
jaxpr once with per-primitive structural rules.

Two public entry points:
- `materialize(linear_fn, primal) -> jnp.ndarray` (always dense)
- `bcoo_jacobian(linear_fn, primal) -> BCOO | jnp.ndarray`

## Non-goals
- Coloring-based extraction (that's asdex's approach; we do per-linearization
  structural walk instead, yielding exact — not conservative — sparsity).
- Second AD pass (we stay inside `jax.linearize`'s output).
- A full linear-operator algebra library like lineax (name inspired by it,
  but we're narrower in scope).

## Key invariants

- The walk produces **bit-exact** output (vs `vmap(hvp)(eye)` reference).
  Any rule change must preserve this. `tests/test_correctness.py` is the
  ground truth.
- Internal forms (`ConstantDiagonal`, `Diagonal`, `Pivoted`) are **private**.
  Only `BCOO` or `ndarray` flow out. Adding a new public form needs a
  deliberate API decision.
- Primitive rules are registered via `@register(lax.PRIM_p)`. Each rule
  receives `(invals, traced_mask, n, **params)` and returns a LinOp. See
  `lineaxpr/materialize.py` for the pattern.

## Benchmarks

Run with the docker runner (see `benchmarks/run_in_container.sh`) to get
the EAGER_CONSTANT_FOLDING=TRUE regime that matches JAX's release config.

**Critical**: always pass the linearization point `y` as a `jit` INPUT,
never a closure. Otherwise `jax.hessian`'s output gets constant-folded and
spurious speedups appear.

## Useful quick commands

```bash
# Correctness on a curated set:
uv run pytest tests/ -v

# Benchmarks (local):
uv run pytest benchmarks/ --benchmark-only

# In-container benchmarks (matches CI config):
bash benchmarks/run_in_container.sh benchmarks/test_curated.py --benchmark-save=0001_curated
```

## References

- Historical gist (pre-repo exploration):
  https://gist.github.com/jpbrodrick89/a3657522e7218d2cc98dae9f80258216
- Comparison library — asdex (coloring-based):
  https://github.com/adrhill/asdex

## Documents to read

- `docs/ARCHITECTURE.md` — data structures + walk algorithm
- `docs/RESEARCH_NOTES.md` — empirical findings from the exploration phase
- `docs/TODO.md` — prioritized future work
