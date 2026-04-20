# lineaxpr

Coloring-free Jacobian/Hessian extraction for JAX. Walks the linearized
jaxpr of a callable and emits either a dense `jnp.ndarray` or a
`jax.experimental.sparse.BCOO` matrix with per-linearization-point
sparsity — no up-front coloring, no second AD pass, no closure-of-constants
tricks.

> **Status**: exploratory / pre-alpha. Design will shift as we go.

## Install

```bash
pip install -e .
# or with asdex for head-to-head benchmarks:
pip install -e ".[benchmark]"
```

## Quick example

```python
import jax
import jax.numpy as jnp
from lineaxpr import materialize, bcoo_jacobian

def f(y):
    return jnp.sum(y ** 2) - jnp.prod(y) ** 0.2

y = jnp.array([9.1, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0])

@jax.jit
def H_of(y):
    _, hvp = jax.linearize(jax.grad(f), y)
    return materialize(hvp, y)          # -> jnp.ndarray (n, n)

@jax.jit
def H_sparse_of(y):
    _, hvp = jax.linearize(jax.grad(f), y)
    return bcoo_jacobian(hvp, y)        # -> BCOO or ndarray
```

## Why lineaxpr

- **Non-conservative sparsity**: the pattern reflects the Hessian at this
  specific `y`, not a conservative union over all possible inputs.
- **One pass over the linear jaxpr**, not two (coloring + AD).
- **Fully jittable**: construction happens at trace time, runtime is the
  fused emitted HLO.
- **BCOO output** for sparse problems yields massive speedups (100-6000×
  over `jax.hessian` at n=3000-5000 on CUTEst problems).

## Coverage

Bit-exact on **100%** of 300 CUTEst unconstrained / bounded / quadratic
problems at `n ≤ 5000` tested so far. See `docs/RESEARCH_NOTES.md`.

## Layout

```
lineaxpr/
├── lineaxpr/              # package
│   ├── __init__.py        # public API
│   └── materialize.py     # structural walk + per-primitive rules
├── tests/                 # correctness (vs vmap(hvp)(eye) reference)
├── benchmarks/            # pytest-benchmark suites
└── docs/
    ├── ARCHITECTURE.md    # how the walk / forms work
    ├── RESEARCH_NOTES.md  # key empirical findings
    └── TODO.md            # prioritized future work
```

## Origin

Built starting from a gist exploration. Historical snapshot:
https://gist.github.com/jpbrodrick89/a3657522e7218d2cc98dae9f80258216
