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
    materialize,
    bcoo_jacobian,
    materialize_rules,
    register,
    _SMALL_N_VMAP_THRESHOLD,
)

# Internal structural forms — exposed for tests / debugging, not stable API.
from .materialize import (
    ConstantDiagonal,
    Diagonal,
    Pivoted,
)

__all__ = [
    "materialize",
    "bcoo_jacobian",
    "materialize_rules",
    "register",
    "ConstantDiagonal",
    "Diagonal",
    "Pivoted",
]
