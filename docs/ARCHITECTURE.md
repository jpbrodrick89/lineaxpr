# Architecture

## Core idea

Given a linear callable `L: R^n -> R^m` (typically the output of
`jax.linearize(jax.grad(f), y)`), produce the matrix `M` such that
`L(v) = M @ v` for all `v`.

We trace `L` to a jaxpr and walk it, propagating a per-var "LinOp" that
describes how each intermediate depends linearly on the input.

## Public API

```python
# Full Jacobian, dense:
H = lineaxpr.materialize(linear_fn, primal)

# Full Jacobian, BCOO when sparse structure survives:
S = lineaxpr.bcoo_jacobian(linear_fn, primal)

# Lower-level transform — explicit seed, returns a LinOp:
seed = lineaxpr.Identity(primal.size, dtype=primal.dtype)
linop = lineaxpr.sparsify(linear_fn)(seed)
linop.to_dense()   # jnp.ndarray
linop.to_bcoo()    # sparse.BCOO
```

`sparsify` is the primitive transform; `materialize` and `bcoo_jacobian`
are convenience wrappers that build an `Identity` seed and densify at the
boundary. Inspired by `jax.experimental.sparse.sparsify`, but specialized
for the linearize-of-grad jaxpr pattern with a richer format space
(Identity / Diagonal / Pivoted / BCOO / ndarray).

## The walk

```
_walk_jaxpr(jaxpr, env, n):
    for eqn in jaxpr.eqns:
        entries = [env[v] for v in eqn.invars]   # (traced, value) tuples
        invals  = [e[1] for e in entries]
        traced  = [e[0] for e in entries]
        if not any(traced):
            # Constant-prop path: evaluate concretely, stash as closure.
            # Important for constant-H problems (DUAL, CMPC) — lets the
            # entire walk fold to a trace-time BCOO literal.
            out = eqn.primitive.bind(*invals, **eqn.params)
            env[outvar] = (False, out)
        else:
            out = materialize_rules[eqn.primitive](invals, traced, n, **eqn.params)
            env[outvar] = (True, out)
```

Each primitive has a rule that knows how to propagate LinOps.

## LinOp forms

Env values are `(traced: bool, value)`. For traced vars, `value` is one
of:

### `ConstantDiagonal(n, value)`
Represents `value · I_n`. `Identity(n, dtype=...)` is the standard seed
constructor for this form. Survives scalar multiplication.

### `Diagonal(values)`
Represents `diag(values)`. Emerges from `mul(closure_vec, ConstantDiagonal)`.

### `Pivoted(out_rows, in_cols, values, out_size, in_size)`
A sparse operator with at most one nonzero per row. Emerges from
`slice(Identity)` and `gather(Identity)`, survives `mul`, `pad`,
`add_any` chains.

**Key invariants**:
- Fresh slice/gather gives `out_rows = np.arange(k)` (identity permutation).
- `pad_rows(before, after)` shifts: `out_rows = prev + before` (still a single range).
- `add_any` concatenates: `out_rows = concat([a.out_rows, b.out_rows, ...])`.
- `scatter_add` permutes: `new_rows = scatter_indices[prev.out_rows]`.

### `jax.experimental.sparse.BCOO`
Standard sparse matrix. Emerges when mixing forms that can't share a
compact representation. Used as the "lowest common denominator" for
structural intermediates.

### `jnp.ndarray`
Dense fallback when no structural form fits. Shape `(*out_shape, n)` —
the trailing axis is the input coordinate.

## LinOp methods

Each LinOp class defines:

- `.primal_aval() -> ShapedArray` — the input aval the walk should trace
  against when this LinOp is used as a seed.
- `.to_dense() -> jnp.ndarray` — densify to an (out_size, in_size) matrix.
- `.to_bcoo() -> sparse.BCOO` — convert to BCOO.
- `.negate()` — returns same form with negated values.
- `.scale_scalar(s)` — scale all entries by scalar `s`.
- `.scale_per_out_row(v)` — scale row `i` by `v[i]`.

`Pivoted` additionally has `.pad_rows(before, after)` for row-axis
padding/truncation.

`sparse.BCOO` is external, so the equivalents live as helper functions
(`_bcoo_negate`, `_bcoo_scale_scalar`, `_bcoo_scale_per_out_row`) in
`materialize.py`.

## Rule registry

```python
materialize_rules: dict[core.Primitive, Callable] = {}

def _mul_rule(invals, traced, n, **params):
    ...
materialize_rules[lax.mul_p] = _mul_rule
```

No decorator — direct assignment, mirroring
`jax.experimental.sparse.transform.sparse_rules_bcoo`. Rules receive:
- `invals` — list of values (LinOp for traced, concrete array for closure)
- `traced` — list of bool flags (True = LinOp / traced)
- `n` — size of flattened top-level input
- `**params` — the eqn's primitive parameters

They return a single LinOp (or list if `primitive.multiple_results`).

## File layout

- `lineaxpr/_base.py` — LinOp classes + densification helpers. Standard
  seed constructors (`Identity`). Mirrors `jax.experimental.sparse._base`.
- `lineaxpr/materialize.py` — the sparsify transform, rule registry, all
  rules (under `# ------ rules ------` section banner), public entry
  points. Single-file like sparsify's `transform.py`.

## The n<16 short-circuit

For tiny problems, `vmap(linear_fn)(jnp.eye(n))` emits cleaner HLO than
our structural walk (which has per-op dispatch overhead). We short-circuit
when `primal.size < _SMALL_N_VMAP_THRESHOLD` (default 16) in
`materialize` / `bcoo_jacobian`, NOT inside `sparsify` — the transform
should do exactly what it says.

## Why the walk stays ours (not sparsify's)

We considered migrating to `jax.experimental.sparse.sparsify`'s
framework — adding our LinOp forms as sparsify formats alongside
BCOO/BCSR — and rejected it. See `docs/RESEARCH_NOTES.md` §10 for full
reasoning. Short version: sparsify's format space is closed
(`SparsifyValue` is a NamedTuple encoding "BCOO vs BCSR" via
`indptr_ref`), upstream coordination cost is high, and our walk already
mirrors sparsify's structure where it matters (interpreter mode via
`make_jaxpr`, `dict[Primitive, Callable]` rule registry).

We do contribute to sparsify by upstreaming `add_any`/`pad`/`scatter-add`
rules to `sparse_rules_bcoo` — see `docs/TODO.md`.

## What we explicitly don't do

- **No coloring pass.** asdex does coloring + compressed HVP; we do
  structural walk + direct matrix emit. Different tradeoffs at different
  scales.
- **No cross-eqn pattern matching.** Each rule looks at one eqn. This
  means e.g. we can't recognise a whole `prod(y)^k` tree as a single
  rank-1 update — we process it level by level.
- **No pytree registration** of LinOp forms. They never leave the env
  during a walk.
- **No tracer mode** — `sparsify` uses `make_jaxpr` + walk. A
  `LineaxprTrace` that hooks `core.Trace.process_primitive` is a
  plausible follow-up (see `docs/TODO.md`).
