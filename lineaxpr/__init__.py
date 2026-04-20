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
    bcoo_jacobian,
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
    "sparsify",
    "materialize",
    "bcoo_jacobian",
    "to_dense",
    "to_bcoo",
    "materialize_rules",
    "ConstantDiagonal",
    "Diagonal",
    "Identity",
    "Pivoted",
]
