# Architecture

## Core idea

Given a linear callable `L: R^n -> R^m` (typically the output of
`jax.linearize(jax.grad(f), y)`), produce the matrix `M` such that
`L(v) = M @ v` for all `v`.

We trace `L` to a jaxpr and walk it, propagating a per-var "LinOp" that
describes how each intermediate depends linearly on the input.

## The walk

```
_walk_jaxpr(jaxpr, env, consts_env, n):
    for eqn in jaxpr.eqns:
        invals  = [read(v) for v in eqn.invars]
        traced  = [is_traced(v) for v in eqn.invars]
        if not any(traced):
            # Pure closure eqn — evaluate concretely
            out = eqn.primitive.bind(*invals, **eqn.params)
            stash in consts_env
        else:
            out = materialize_rules[eqn.primitive](invals, traced, n, **eqn.params)
            stash in env
    return env[jaxpr.outvars[0]]
```

Each primitive has a rule that knows how to propagate LinOps.

## LinOp forms (all internal)

Each `env[var]` is one of:

### `ConstantDiagonal(n, value)`
Represents `value · I_n`. Appears at the invar entry and survives scalar
multiplication. Densifies as `value * jnp.eye(n)`.

### `Diagonal(values)`
Represents `diag(values)`. Emerges from `mul(closure_vec, ConstantDiagonal)`.
Densifies via `zeros.at[arange, arange].set(values)`.

### `Pivoted(out_rows, in_cols, values, out_size, in_size)`
A sparse operator with at most one nonzero per row. Represents
`M[out_rows[i], in_cols[i]] = values[i]`. Emerges from `slice(Identity)` and
`gather(Identity)`, survives `mul`, `pad`, `add_any` chains. Densifies via
scatter.

**Key invariants**:
- Fresh slice/gather gives `out_rows = np.arange(k)` (identity permutation).
- `pad` shifts: `out_rows = prev + before` (still a single range).
- `add_any` concatenates: `out_rows = concat([a.out_rows, b.out_rows, ...])`.
- `scatter_add` permutes: `new_rows = scatter_indices[prev.out_rows]`
  (only genuinely-arbitrary case).

### `jax.experimental.sparse.BCOO`
Standard sparse matrix. Emerges when mixing forms that can't share a
compact representation (e.g., `add(Pivoted, Diagonal)` with different
shape semantics). Used as the "lowest common denominator" for structural
intermediates.

### `jnp.ndarray`
Dense fallback when no structural form fits. Shape `(*out_shape, n)` — the
trailing axis is the input coordinate.

## Boundary conversions

- `materialize`: always densify to `jnp.ndarray`.
- `bcoo_jacobian`: convert `(Constant)Diagonal` and `Pivoted` to `BCOO`;
  `BCOO` passes through; `ndarray` passes through.

## Rule registry pattern

```python
materialize_rules: dict[core.Primitive, Callable] = {}

def register(prim):
    def deco(fn):
        materialize_rules[prim] = fn
        return fn
    return deco

@register(lax.mul_p)
def _mul_rule(invals, traced, n, **params):
    ...
```

Rules receive:
- `invals` — list of values (LinOp for traced, concrete array for closure)
- `traced` — list of bool flags (True = LinOp / traced)
- `n` — size of flattened top-level input (for context)
- `**params` — the eqn's primitive parameters

They return a single LinOp (or list if `primitive.multiple_results`).

## The n<16 short-circuit

For tiny problems, `vmap(linear_fn)(jnp.eye(n))` emits cleaner HLO than our
structural walk (which has per-op dispatch overhead). We short-circuit when
`primal.size < _SMALL_N_VMAP_THRESHOLD` (default 16) and fall back to the
`vmap(eye)` path. Correctness is preserved; just avoids a ~2× regression
on problems like HS110 (n=10) where the prod-tree HVP produces many tiny
intermediate Pivoteds that each pay scatter overhead.

## What we explicitly don't do

- **No coloring pass.** asdex does coloring + compressed HVP; we do
  structural walk + direct matrix emit. Different tradeoffs at different
  scales.
- **No cross-eqn pattern matching.** Each rule looks at one eqn. This
  means e.g. we can't recognise a whole `prod(y)^k` tree as a single
  rank-1 update — we process it level by level.
- **No pytree registration** of internal LinOp forms. They never leave
  the `env` during a walk.
