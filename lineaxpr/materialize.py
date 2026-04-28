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
from jax.experimental import sparse
from jax.extend import core

from ._linops import Identity, LinOpProtocol
from ._rules.registry import materialize_rules
from ._rules.add import (
    BELLPACK_DEDUP_LIMIT,  # noqa: F401 — accessible as lineaxpr.materialize.BELLPACK_DEDUP_LIMIT
    BELLPACK_DEDUP_VECTORISED_MIN,  # noqa: F401 — same
    _add_rule,  # noqa: F401 — re-exported for test compatibility
)

# -------------------------- rule registry --------------------------
# Rules are registered in `lineaxpr/_rules/registry.py`.
# Re-exported here for backward-compatibility:
#   lineaxpr.materialize.materialize_rules
#   lineaxpr.materialize.BELLPACK_DEDUP_LIMIT
#   lineaxpr.materialize.BELLPACK_DEDUP_VECTORISED_MIN

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
    jaxpr, return the output LinOp."""
    placeholder = jax.ShapeDtypeStruct((seed_linop.shape[-1],), seed_linop.dtype)
    cj = jax.make_jaxpr(linear_fn)(placeholder)
    jaxpr = cj.jaxpr

    if len(jaxpr.invars) != 1:
        raise NotImplementedError("multi-input linear_fn not yet handled")
    (invar,) = jaxpr.invars
    n = invar.aval.size

    env: dict = {v: (False, c) for v, c in zip(jaxpr.constvars, cj.consts)}
    env[invar] = (True, seed_linop)
    _walk_jaxpr(jaxpr, env, n)

    if len(jaxpr.outvars) != 1:
        raise NotImplementedError("multi-output linear_fn not yet handled")
    (outvar,) = jaxpr.outvars
    return env[outvar][1]


def sparsify(linear_fn):
    """Transform a linear function into one that operates on LinOps.

    `sparsify(linear_fn)(seed_linop)` traces `linear_fn` against the aval
    implied by `seed_linop.primal_aval()`, walks the resulting jaxpr with
    per-primitive structural rules, and returns a LinOp representing the
    linear function's matrix.

    Seeds are explicit — no automatic Identity cast. For the common case
    of extracting the full Jacobian, the public wrappers `materialize` /
    `jacfwd` / `jacrev` / `hessian` build
    `Identity(primal.size, dtype=primal.dtype)` and pass it through.
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
    linop = sparsify(linear_fn)(seed)
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
