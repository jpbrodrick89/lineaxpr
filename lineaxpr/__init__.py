"""lineaxpr — structural Jacobian/Hessian extraction for JAX linear callables.

Works on linearized jaxprs. Inspired by lineax linear-algebra rules;
focused on per-linearization-point sparsity and non-conservative
extraction.

Public API:

    import lineaxpr

    # jax-like wrappers (preferred):
    H  = lineaxpr.hessian(f)(y)          # -> jnp.ndarray
    Hs = lineaxpr.bcoo_hessian(f)(y)     # -> BCOO | jnp.ndarray
    Jf = lineaxpr.jacfwd(f)(y)           # same as jax.jacfwd shape
    Jr = lineaxpr.jacrev(f)(y)           # same as jax.jacrev shape
    # All six accept format='dense'|'bcoo'; bcoo_* are shorthands.

    # Core helper — use when you already have a linear callable:
    H = lineaxpr.materialize(linear_fn, primal, format='dense')
    S = lineaxpr.materialize(linear_fn, primal, format='bcoo')

    # Primitive transform — returns a LinOp to post-process:
    seed = lineaxpr.Identity(primal.size, dtype=primal.dtype)
    linop = lineaxpr.sparsify(linear_fn)(seed)
    H = lineaxpr.to_dense(linop)
    S = lineaxpr.to_bcoo(linop)
"""

from .materialize import (
    bcoo_hessian,
    bcoo_jacfwd,
    bcoo_jacrev,
    hessian,
    jacfwd,
    jacrev,
    materialize,
    materialize_rules,
    sparsify,
    to_bcoo,
    to_dense,
    _SMALL_N_VMAP_THRESHOLD,
)

from ._base import (
    ConstantDiagonal,
    Diagonal,
    Identity,
    Pivoted,
)

__all__ = [
    # jax-like public API
    "jacfwd",
    "bcoo_jacfwd",
    "jacrev",
    "bcoo_jacrev",
    "hessian",
    "bcoo_hessian",
    # lower-level building blocks
    "sparsify",
    "materialize",
    "to_dense",
    "to_bcoo",
    # rule registry
    "materialize_rules",
    # LinOp classes (exposed for custom seeds / debugging)
    "ConstantDiagonal",
    "Diagonal",
    "Identity",
    "Pivoted",
]
