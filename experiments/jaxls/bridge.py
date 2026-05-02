"""Bridge: use lineaxpr.jacfwd as the Jacobian engine for jaxls costs.

Usage
-----
    def residual(vals: jaxls.VarValues, var: MyVar, ...) -> jax.Array:
        return vals[var] - target

    jac_fn = make_lineaxpr_jac(residual)

    @jaxls.Cost.factory(jac_custom_fn=jac_fn)
    def my_cost(vals, var, ...):
        return residual(vals, var, ...)

Design
------
lineaxpr.jacfwd calls jax.linearize(f, y) internally so it handles
arbitrary nonlinear residuals — JAX does the linearisation, lineaxpr
walks the resulting linear jaxpr to extract the sparse matrix.

The key difficulty is that inside the traced function we must avoid
VarValues.make / VarValues.get_value, both of which use jnp.searchsorted
and dynamic gathers that lineaxpr cannot yet trace through.  We bypass
them entirely with _FlatVarValues: a lightweight mock whose __getitem__
returns a static-indexed slice of the flat input vector.  JAX traces
through this without emitting any gather ops.

Only Euclidean variables are supported (tangent space = parameter space).
The Jacobian columns are ordered by the iteration order of variables found
in cost.args (depth-first pytree walk), which matches jaxls' sorted order
for costs where all variables share the same type.  For mixed-type costs
the caller must ensure the variable types sort (alphabetically by class
name) into the column order expected by jaxls.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.tree_util import default_registry

import lineaxpr
import jaxls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yield_vars(obj: Any):
    """Yield all jaxls.Var leaves depth-first."""
    if isinstance(obj, jaxls.Var):
        yield obj
        return
    children_and_meta = default_registry.flatten_one_level(obj)
    if children_and_meta is None:
        return
    for child in children_and_meta[0]:
        yield from _yield_vars(child)


def _split_sizes(variables: list[jaxls.Var], vals: jaxls.VarValues) -> list[int]:
    """Return flat sizes (= tangent_dim) for each variable in order."""
    return [type(v).tangent_dim for v in variables]


class _FlatVarValues:
    """Minimal VarValues mock that returns slices of a flat vector.

    Avoids jnp.searchsorted / dynamic gather inside the traced function,
    so lineaxpr can walk the resulting jaxpr without hitting unsupported ops.

    Lookup uses Python object identity (id(var)) so it works inside jaxls'
    jax.vmap context where variable IDs are BatchTracers (not concretisable).
    This is safe because _yield_vars yields the *same* Python Var objects that
    appear in cost.args, which are the same objects compute_residual receives.
    """

    def __init__(
        self,
        variables: list[jaxls.Var],
        pieces: list[jax.Array],
    ) -> None:
        self._pieces = pieces
        # Keyed by Python object identity — never touches v.id
        self._index: dict[int, int] = {id(v): i for i, v in enumerate(variables)}

    def __getitem__(self, var: jaxls.Var) -> jax.Array:
        return self._pieces[self._index[id(var)]]

    def get_value(self, var: jaxls.Var) -> jax.Array:
        return self[var]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_lineaxpr_jac(
    compute_residual: Callable,
    *,
    use_jacrev: bool = False,
    format: str = "dense",
) -> Callable:
    """Return a jac_custom_fn that uses lineaxpr to compute Jacobians.

    Parameters
    ----------
    compute_residual:
        Same function passed to Cost.factory.  Must only access the Var
        objects present in its *args* (standard jaxls practice).
    use_jacrev:
        Use lineaxpr.jacrev instead of jacfwd.  Default False (jacfwd is
        optimal for n→n Jacobians; prefer jacrev when residual_dim << tangent_dim).
    format:
        'dense' (default) or 'bcoo'.

    Returns
    -------
    jac_fn : (vals, *args) -> jax.Array  shape (residual_dim, tangent_dim)
        Suitable for Cost.factory(jac_custom_fn=...).
    """
    materialize_jac = lineaxpr.jacrev if use_jacrev else lineaxpr.jacfwd

    def jac_fn(vals: jaxls.VarValues, *args: Any, **kwargs: Any) -> jax.Array:
        # Vars live in positional args; kwargs hold plain data (e.g. weight)
        variables_raw = list(_yield_vars(args))
        if not variables_raw:
            raise ValueError("No jaxls.Var found in cost args.")

        # Reorder to match jaxls' expected column layout:
        #   types sorted by str(type) (alphabetically), within-type order preserved.
        # jaxls uses the same key in its _sort_key / VarTypeOrdering.
        # Within each type the variables must be in ascending-ID order; callers
        # are responsible for passing them that way (standard practice).
        from collections import defaultdict as _dd
        type_groups: dict = _dd(list)
        for v in variables_raw:
            type_groups[type(v)].append(v)
        variables = []
        for t in sorted(type_groups.keys(), key=lambda t: str(t)):
            variables.extend(type_groups[t])

        sizes = _split_sizes(variables, vals)

        # Flat current values (the linearisation point for jacfwd)
        flat_pieces = [vals[v].ravel() for v in variables]
        flat_current = jnp.concatenate(flat_pieces) if len(flat_pieces) > 1 else flat_pieces[0]

        # Shapes for reconstruction inside the traced function
        shapes = [vals[v].shape for v in variables]

        def residual_of_flat(flat_vec: jax.Array) -> jax.Array:
            # Split flat_vec using static offsets — no dynamic gather.
            offset = 0
            pieces = []
            for size, shape in zip(sizes, shapes):
                chunk = flat_vec[offset : offset + size]
                pieces.append(chunk.reshape(shape) if shape else chunk.squeeze())
                offset += size

            mock_vals = _FlatVarValues(variables, pieces)
            return compute_residual(mock_vals, *args, **kwargs).ravel()

        return materialize_jac(residual_of_flat, format=format)(flat_current)

    return jac_fn
