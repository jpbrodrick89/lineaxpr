# lineaxpr

Structural Jacobian/Hessian extraction for JAX. Walks the linearized
jaxpr of a callable with per-primitive rules over mixed sparse forms
(Identity / Diagonal / Pivoted / BCOO / ndarray) and emits either a
dense `jnp.ndarray` or a `jax.experimental.sparse.BCOO` matrix. No
up-front coloring, no second AD pass, per-linearization-point sparsity.

Inspired by — and interoperable with — `jax.experimental.sparse.sparsify`;
see `docs/RESEARCH_NOTES.md` §10 for the design comparison.

> **Status**: exploratory / pre-alpha. Design will shift as we go.

## Install

```bash
pip install -e .
# or with asdex for head-to-head benchmarks:
pip install -e ".[benchmark]"
```

## Quick example — jax-like API

```python
import jax.numpy as jnp
import lineaxpr

def f(y):
    return jnp.sum(y ** 2) - jnp.prod(y) ** 0.2

y = jnp.array([9.1, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0])

H = lineaxpr.hessian(f)(y)          # -> jnp.ndarray (n, n)
H_bcoo = lineaxpr.bcoo_hessian(f)(y) # -> BCOO or ndarray

# Fwd and rev Jacobians (also accept a format='dense'|'bcoo' kwarg):
J_fwd = lineaxpr.jacfwd(f)(y)
J_rev = lineaxpr.jacrev(f)(y)
J_sparse = lineaxpr.bcoo_jacfwd(f)(y)

# Equivalent longer form — useful when you already have a linearized fn:
_, hvp = jax.linearize(jax.grad(f), y)
H = lineaxpr.materialize(hvp, y, format="dense")
H_bcoo = lineaxpr.materialize(hvp, y, format="bcoo")
```

Signatures mirror `jax.jacfwd` / `jax.jacrev` / `jax.hessian`. Current
limitations: single-input, single-output `f`, 1D `y` only — see
`docs/TODO.md` #9c/#9d for `argnums` / `has_aux` / multi-output plans.

## Lower-level transform

The jax-like wrappers are thin compositions over a more general
transform:

```python
from lineaxpr import sparsify, Identity, to_dense, to_bcoo

seed = Identity(y.size, dtype=y.dtype)
linop = sparsify(hvp)(seed)          # returns a LinOp (our class, BCOO, or ndarray)
dense = to_dense(linop)              # uniform densification
bcoo  = to_bcoo(linop)               # uniform sparsification

# Non-identity seeds also work — useful for matrix-matrix products
# where one factor is structured:
from lineaxpr import ConstantDiagonal, Diagonal
scaled = to_dense(sparsify(hvp)(ConstantDiagonal(y.size, 2.0)))  # == 2 * dense
```

No implicit Identity cast inside `sparsify` — pass your seed explicitly.
The seed's `.primal_aval()` method provides the shape/dtype that
`jax.make_jaxpr` traces against.

## Why lineaxpr

- **Non-conservative sparsity**: the pattern reflects the Hessian at this
  specific `y`, not a conservative union over all possible inputs.
- **One pass over the linear jaxpr**, not two (coloring + AD).
- **Fully jittable**: construction happens at trace time, runtime is the
  fused emitted HLO.
- **Mixed-format walk**: Identity / Diagonal / Pivoted / BCOO coexist in
  the same env with promotion at incompatibility; no unnecessary
  densification.
- **2–4× over pure-BCOO** `jax.experimental.sparse.sparsify` on
  y-dependent sparse problems (DIXMAAN / EDENSCH / FLETCHCR), plus
  robust handling of primitive-coverage gaps where upstream sparsify
  fails (HART6, ARGTRIGLS). Const-H wins over `jax.hessian` depend on
  the EAGER_CONSTANT_FOLDING regime — see `docs/RESEARCH_NOTES.md` §10.

## Coverage

Bit-exact on **100%** of ~200 CUTEst unconstrained / bounded / quadratic
problems at `n ≤ 5000` tested so far. See `docs/RESEARCH_NOTES.md` and
the `tests/test_sif2jax_sweep.py` slow sweep.

## Tests

```bash
# Fast (unit + hand-rolled + transform + curated CUTEst):
uv run pytest tests/

# Slow sweep across all sif2jax problems + nse regression:
uv run pytest -m slow tests/test_sif2jax_sweep.py
```

nse (number of stored entries) per problem is pinned in
`tests/nse_manifest.json`. Increases fail; decreases require running
`uv run python -m tests.update_nse_manifest` to update the golden file.
Regressions need an explicit justification in the commit message.

## Layout

```
lineaxpr/
├── lineaxpr/              # package
│   ├── __init__.py        # public API:
│   │                      #   jacfwd / bcoo_jacfwd
│   │                      #   jacrev / bcoo_jacrev
│   │                      #   hessian / bcoo_hessian
│   │                      #   materialize(..., format='dense'|'bcoo')
│   │                      #   sparsify, to_dense, to_bcoo
│   │                      #   Identity, ConstantDiagonal, Diagonal, Pivoted
│   │                      #   materialize_rules
│   ├── _base.py           # LinOp classes with methods
│   └── materialize.py     # sparsify transform + rule registry + rules
├── experiments/           # monkeypatch studies (not production)
├── tests/                 # unit + hand-rolled + transform + sweep + nse manifest
├── benchmarks/            # pytest-benchmark suites
└── docs/
    ├── ARCHITECTURE.md    # walk algorithm, LinOp methods, rule pattern
    ├── RESEARCH_NOTES.md  # empirical findings + sparsify comparison
    └── TODO.md            # prioritized future work
```

## Origin

Built starting from a gist exploration. Historical snapshot:
https://gist.github.com/jpbrodrick89/a3657522e7218d2cc98dae9f80258216
