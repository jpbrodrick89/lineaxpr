"""lineaxpr — coloring-free Jacobian/Hessian extraction for JAX linear callables.

Works on linearized jaxprs. Inspired by lineax linear-algebra rules; focused
on per-linearization-point sparsity and non-conservative extraction.

Public API:

    from lineaxpr import materialize, bcoo_jacobian

    # Hessian of a scalar objective f at y:
    import jax
    _, hvp = jax.linearize(jax.grad(f), y)

    H_dense = materialize(hvp, y)             # -> jnp.ndarray
    H_bcoo  = bcoo_jacobian(hvp, y)           # -> BCOO | jnp.ndarray
"""

from .materialize import (
    bcoo_hessian,
    bcoo_jacfwd,
    bcoo_jacobian,
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
    # jax-like public API (preferred)
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
    # deprecated — kept for back-compat
    "bcoo_jacobian",
]
