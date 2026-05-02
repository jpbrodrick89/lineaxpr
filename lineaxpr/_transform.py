"""Coloring-free Jacobian extraction for linear callables.

Public API (see `lineaxpr/__init__.py`):

* `jacfwd(f)(y)` / `bcoo_jacfwd(f)(y)` — forward-mode Jacobian.
* `jacrev(f)(y)` / `bcoo_jacrev(f)(y)` — reverse-mode Jacobian.
* `hessian(f)(y)` / `bcoo_hessian(f)(y)` — full Hessian.
* `materialize(linear_fn, primal, format='dense'|'bcoo')` — core helper,
  when you already have a linearized callable.
* `sparsify(linear_fn)(seed_linop)` — primitive transform returning a
  LinOp (before format conversion).

All of the above trace `linear_fn` to a jaxpr and walk its equations
with per-primitive rules that propagate structural per-var operators.
The LinOp classes (`ConstantDiagonal`, `Diagonal`, `BEllpack`; see
`_linops/`) let common patterns (scalar · I, vector-scaled I, sparse
banded blocks) avoid materialising intermediate identity matrices; they
are converted to BCOO or dense at the boundary.

## Column-independence invariant (Phase B convention)

The walker is a sparsify-style transform: it tracks the *Jacobian*
of a linearised function, and that Jacobian acts on each COLUMN of a
2D operand independently. There is no primitive that should couple
columns. `sparsify(f)(seed)` is "f applied to each column of seed,
columns stacked".

Concretely, with `vmap(in_axes=±1, out_axes=±1)` (batch at the LAST
axis of a 2D input, output stacked along the LAST axis), this
matches `jax.experimental.sparse.sparsify` semantics exactly:

* For seed `Identity(n)`, each column is `e_j`. Output column j is
  `f(e_j)` = J's column j. Result = J.
* For asymmetric seed `M` shape `(m, n)`, output column j is `f(M[:, j])`.
  Independent across j; structural sparsity propagates per-column.

Composition follows: `sparsify(f ∘ g)(eye) = J_f @ J_g` because
`f(g(eye))` decomposes into f acting on each col of g(eye), and
each col of g(eye) is `g(e_j) = J_g[:, j]`.

**MUST-meet design invariants**:

1. `sparsify(f)(linop.todense()) == sparsify(f)(linop).todense()` for
   non-transposed LinOps and any vmap(in=±1, out=±1)'d flat linear
   function (R^n → R^m) with supported primitives.
2. `sparsify(f)(BEllpack)` returns a BEllpack with the same
   `transposed` flag (non-transposed → non-transposed).

**Materialize uses `vmap(in=0, out=-1)`**: this is the column-
independent view (each row of input is a per-sample input → vmap
batches over rows; output stacked at -1 = columns). Combined with
Identity seed, gives J directly. (Note: `in_axes=0` on a 2D input
is equivalent to "iterate over rows", which under the column-
independent invariant means columns of the seed are treated row-
by-row — for symmetric Identity that's the same.)

JAX vmap's broadcasting rules don't always meet the column-
independence assumption directly — vmap inserts boundary transposes
to maintain its convention. The walker handles these transposes;
the key invariant — original columns stay independent — is preserved.

## Known gap: non-finite closures in structural paths

Our structural rules assume `0 * x = 0` for any `x`. This is correct
when `x` is finite but wrong for `x ∈ {inf, nan}` (where `0 * inf = nan`).
When a mul/div/add structural rule emits a BEllpack/BCOO that skips
zero positions, it silently drops positions where the closure operand
has `inf`/`nan`. CUTEst objectives don't produce non-finite intermediate
values in practice, so this is a latent correctness gap rather than an
observed issue. A fully-general fix would require reading the closure
for non-finite entries (essentially densifying), losing the structural
optimisation — not worth it unless the gap bites.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse
from jax.extend import core

from ._linops import Identity, LinOpProtocol
from ._rules.registry import materialize_rules
from ._rules.add import (
    BELLPACK_DEDUP_LIMIT,  # noqa: F401 — accessible as lineaxpr._transform.BELLPACK_DEDUP_LIMIT
    BELLPACK_DEDUP_VECTORISED_MIN,  # noqa: F401 — same
    _add_rule,  # noqa: F401 — re-exported for test compatibility
)

# -------------------------- rule registry --------------------------
# Rules are registered in `lineaxpr/_rules/registry.py`.
# Re-exported here for backward-compatibility:
#   lineaxpr._transform.materialize_rules
#   lineaxpr._transform.BELLPACK_DEDUP_LIMIT
#   lineaxpr._transform.BELLPACK_DEDUP_VECTORISED_MIN

# -------------------------- driver --------------------------


def _walk_jaxpr(jaxpr, env, n):
    """Walk a jaxpr, mutating env.

    Env is `dict[Var, tuple[bool, Any]]` where the bool is `traced`:
      * (True, LinOp) — this var depends on the walk's input; value is a LinOp.
      * (False, concrete_array) — this var is pure closure data.
    Literals are read directly from `.val`; traced status comes from the
    invars the caller seeded.
    """
    def read(atom):
        if isinstance(atom, core.Literal):
            return (False, atom.val)
        return env[atom]

    for eqn in jaxpr.eqns:
        entries = [read(v) for v in eqn.invars]
        invals = [e[1] for e in entries]
        traced = [e[0] for e in entries]
        if not any(traced):
            # Constant propagation: no traced inputs → evaluate concretely
            # and stash as closure data. Important for constant-H problems
            # (DUAL, CMPC) — lets the whole walk fold to a trace-time BCOO
            # literal. See docs/RESEARCH_NOTES.md §10.
            concrete_outs = eqn.primitive.bind(*invals, **eqn.params)
            # In the vmap jaxpr, broadcast_in_dim of a constant may add the
            # vmap batch dim b=n as a NEW leading dimension — recognisable by
            # dim 0 not being in broadcast_dimensions (it was added, not mapped).
            # Strip it so downstream rules see the pre-vmap (structural) shape.
            if (eqn.primitive is lax.broadcast_in_dim_p
                    and hasattr(concrete_outs, 'ndim')
                    and concrete_outs.ndim > 0
                    and int(concrete_outs.shape[0]) == n
                    and 0 not in eqn.params.get('broadcast_dimensions', ())):
                concrete_outs = concrete_outs[0]
            if eqn.primitive.multiple_results:
                for v, o in zip(eqn.outvars, concrete_outs):
                    env[v] = (False, o)
            else:
                (outvar,) = eqn.outvars
                env[outvar] = (False, concrete_outs)
            continue
        rule = materialize_rules.get(eqn.primitive)
        if rule is None:
            forms = ", ".join(
                type(v).__name__ if t else f"closure:{type(v).__name__}"
                for v, t in zip(invals, traced)
            )
            raise NotImplementedError(
                f"No lineaxpr rule for primitive '{eqn.primitive}'.\n"
                f"  Input forms: [{forms}]\n"
                f"  To add a rule: register at lineaxpr.materialize_rules[{eqn.primitive}] = your_rule\n"
                f"  Or file an issue at https://github.com/jpbrodrick89/lineaxpr/issues "
                f"with the minimal f(y) that triggers this."
            )
        outs = rule(invals, traced, n, **eqn.params)
        if eqn.primitive.multiple_results:
            for v, o in zip(eqn.outvars, outs):
                env[v] = (True, o)
        else:
            (outvar,) = eqn.outvars
            env[outvar] = (True, outs)


def _walk_with_seed(linear_fn, seed_linop):
    """Trace `linear_fn` with the aval implied by `seed_linop`, walk the
    jaxpr, return the output LinOp.

    `linear_fn` is expected to be already vmapped by the caller (see
    `materialize`). `sparsify(vmapped_fn)(seed)` is the lower-level
    entry point — callers may supply any vmap configuration.

    The jaxpr's invar has shape `seed_linop.shape`. For the standard
    `Identity(n)` seed under default `jax.vmap`, that's `(n, n)`.
    """
    placeholder = jax.ShapeDtypeStruct(seed_linop.shape, seed_linop.dtype)
    cj = jax.make_jaxpr(linear_fn)(placeholder)
    jaxpr = cj.jaxpr

    if len(jaxpr.invars) != 1:
        raise NotImplementedError("multi-input linear_fn not yet handled")
    (invar,) = jaxpr.invars
    n = seed_linop.shape[-1]

    # Under EAGER_CONSTANT_FOLDING=TRUE, broadcast_in_dim equations that add b=n
    # as a new leading dim are pre-folded into constvars — our equation-level
    # strip in _walk_jaxpr never fires for them.  A broadcast-result constvar has
    # all identical rows (c == c[0]), unlike genuine non-uniform closure data.
    # Strip those here so the rest of the walk sees the pre-vmap (structural) shape.
    def _strip_if_broadcast(c):
        if (hasattr(c, "shape") and len(c.shape) > 1
                and int(c.shape[0]) == n):
            try:
                if jnp.all(c == c[0]):
                    return c[0]
            except Exception:
                pass
        return c

    env: dict = {v: (False, _strip_if_broadcast(c))
                 for v, c in zip(jaxpr.constvars, cj.consts)}
    env[invar] = (True, seed_linop)
    _walk_jaxpr(jaxpr, env, n)

    if len(jaxpr.outvars) != 1:
        raise NotImplementedError("multi-output linear_fn not yet handled")
    (outvar,) = jaxpr.outvars
    return env[outvar][1]


def sparsify(linear_fn):
    """Transform a linear function into one that operates on LinOps.

    `sparsify(linear_fn)(seed_linop)` traces `linear_fn` against the
    aval implied by `seed_linop.shape`, walks the resulting jaxpr with
    per-primitive structural rules, and returns a LinOp representing
    the linear function's matrix.

    `linear_fn` is **not** vmapped by sparsify — pass an already-vmapped
    function if you want a per-sample-then-stack interpretation. The
    high-level entry points (`materialize` / `jacfwd` / `jacrev` /
    `hessian`) handle vmap themselves.

    Seeds are explicit — no automatic Identity cast.
    """
    def inner(seed_linop):
        return _walk_with_seed(linear_fn, seed_linop)

    return inner



_VALID_FORMATS = ("dense", "bcoo")


def materialize(linear_fn, primal, format: str = "dense"):
    """Materialize the Jacobian matrix of a linear callable.

    Args:
      linear_fn: a linear callable `R^n -> R^m` (typically the output of
        `jax.linearize(...)[1]` or `jax.linear_transpose(...)`).
      primal: a shape/dtype witness for the input to `linear_fn`. Only
        `primal.size` and `primal.dtype` are read, so this can be any of:
        a concrete array, a `jax.Array` / `jnp.ndarray`, or a
        `jax.ShapeDtypeStruct` (matching the convention used by
        `jax.linear_transpose` / `jax.eval_shape`). Passing a
        ShapeDtypeStruct is the preferred option when you don't already
        have a concrete primal on hand.
      format: one of `"dense"` or `"bcoo"`.
        - `"dense"` returns a `jnp.ndarray`.
        - `"bcoo"` returns a `jax.experimental.sparse.BCOO` when the
          walk preserves structural sparsity, otherwise a dense ndarray
          (dense fallbacks surface to the caller unchanged).

    """
    if format not in _VALID_FORMATS:
        raise ValueError(f"format must be one of {_VALID_FORMATS}, got {format!r}")
    n = primal.size if hasattr(primal, "size") else int(jnp.size(primal))
    seed = Identity(n, dtype=primal.dtype)
    # vmap config is centralised here so callers (sparsify directly)
    # can supply any vmap configuration via jax.vmap on linear_fn.
    #
    # Phase B convention: (in_axes=-1, out_axes=-1). Both batch and
    # per-sample stacking happen at the last axis, which aligns with
    # the walker's "in_axis at -1" convention. Rules pass jaxpr params
    # straight through (no walk-frame translation); the walker's
    # natural output equals dense vmap, so `sparsify` matches
    # `vmap(lin, in_ax, out_ax)(seed_dense)` for any vmap config.
    linop = sparsify(jax.vmap(linear_fn, in_axes=-1, out_axes=-1))(seed)
    if format == "dense":
        return linop.todense() if isinstance(linop, LinOpProtocol) else linop
    bcoo = linop.to_bcoo() if hasattr(linop, 'to_bcoo') else linop
    # Smart-densify at output: at `nse >= prod(shape)` the BCOO stores
    # at least as many float values as dense AND carries 2·nse index
    # ints on top — strictly worse than dense. DUAL-class problems
    # (small n, highly-connected) hit this when `_bcoo_concat` stacks
    # many overlapping BCOO operands without deduping.
    if isinstance(bcoo, sparse.BCOO):
        total = 1
        for s in bcoo.shape:
            total *= int(s)
        if bcoo.nse >= total:
            return bcoo.todense()
    return bcoo


# -------------------------- jax-like public API --------------------------


def jacfwd(f, *, format: str = "dense"):
    """Forward-mode Jacobian, matching `jax.jacfwd`'s output shape.

    Equivalent to `materialize(jax.linearize(f, y)[1], y, format=format)`.

    Returns a function `(y) -> Jacobian`. `format='dense'` (default)
    returns a `jnp.ndarray`; `format='bcoo'` returns a BCOO when
    structural sparsity survives, else a dense ndarray.

    Only single-input / single-output `f` with 1D `y` is currently
    supported — see `docs/TODO.md` for the multi-input / multi-output
    roadmap.
    """
    def wrapped(y):
        _, lin = jax.linearize(f, y)
        return materialize(lin, y, format=format)
    return wrapped


def bcoo_jacfwd(f):
    """Forward-mode Jacobian returned as BCOO. Alias for
    `jacfwd(f, format='bcoo')`."""
    return jacfwd(f, format="bcoo")


def jacrev(f, *, format: str = "dense"):
    """Reverse-mode Jacobian, matching `jax.jacrev`'s output shape.

    Equivalent to `materialize(linear_transpose(linearize(f, y)[1], y),
    y_out, format=format).T`.

    Returns a function `(y) -> Jacobian`.
    """
    def wrapped(y):
        y_out, lin = jax.linearize(f, y)
        vjp = jax.linear_transpose(lin, y)
        def vjp_unpacked(w):
            (out,) = vjp(w)
            return out
        return materialize(vjp_unpacked, y_out, format=format).T
    return wrapped


def bcoo_jacrev(f):
    """Reverse-mode Jacobian returned as BCOO. Alias for
    `jacrev(f, format='bcoo')`."""
    return jacrev(f, format="bcoo")


def hessian(f, *, format: str = "dense"):
    """Hessian, matching `jax.hessian`'s output shape.

    Equivalent to `materialize(jax.linearize(jax.grad(f), y)[1], y,
    format=format)`.

    Returns a function `(y) -> Hessian`.
    """
    def wrapped(y):
        _, lin = jax.linearize(jax.grad(f), y)
        return materialize(lin, y, format=format)
    return wrapped


def bcoo_hessian(f):
    """Hessian returned as BCOO. Alias for `hessian(f, format='bcoo')`."""
    return hessian(f, format="bcoo")
