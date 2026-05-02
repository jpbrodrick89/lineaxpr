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
Linearisation point
    jaxls' own autodiff path differentiates

        g(δ) = residual(val_subset._retract(δ, ordering), *args)

    at δ = 0.  We do the same: linearise at zeros(tangent_dim) and use
    var_type.retract_fn(current_val, δ_chunk) to build the perturbed
    values inside the traced function.  This is correct for both
    Euclidean variables (retract = add) and Lie-group variables
    (SE2/SE3, retract = x @ exp(δ)).

Why not linearise at flat_current?
    For Euclidean vars both approaches give the same Jacobian, but for
    Lie-group vars the parameter space ≠ tangent space so linearising
    at the flattened parameter is wrong.  The retract-at-zero approach
    is universally correct.

VarValues mock
    Inside the traced function (after jax.linearize) we avoid
    VarValues.make / VarValues.get_value because both use
    jnp.searchsorted + dynamic gather which lineaxpr cannot walk.
    _FlatVarValues returns the retracted values (plain arrays for
    Euclidean vars, SE2/SO3/... objects for Lie-group vars) via a
    Python id() lookup — safe inside jaxls' jax.vmap context where
    variable IDs are BatchTracers.

Column ordering
    Variables are sorted by str(type) before building the flat tangent
    vector, matching jaxls' VarTypeOrdering (alphabetical type sort).
    Within each type the caller must provide variables in ascending-ID
    order (standard for any sensibly-constructed problem).

Output format
    jaxls' jac_custom_fn must return a dense ndarray — jaxls' internal
    block-sparse assembly (compute_column_norms, etc.) does not accept
    BCOO.  lineaxpr.jacfwd always uses its structural walk internally;
    the output is always densified before being returned.
"""

from __future__ import annotations

from collections import defaultdict
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


class _FlatVarValues:
    """Minimal VarValues mock backed by a list of (retracted) values.

    Avoids jnp.searchsorted / dynamic gather inside the traced function.
    Lookup uses Python object identity (id(var)) — never touches var.id,
    so it works inside jaxls' vmap where IDs are BatchTracers.
    Values can be plain arrays (Euclidean) or Lie-group objects (SE2 etc.).
    """

    def __init__(self, variables: list[jaxls.Var], values: list[Any]) -> None:
        self._values = values
        self._index: dict[int, int] = {id(v): i for i, v in enumerate(variables)}

    def __getitem__(self, var: jaxls.Var) -> Any:
        return self._values[self._index[id(var)]]

    def get_value(self, var: jaxls.Var) -> Any:
        return self[var]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_lineaxpr_jac(
    compute_residual: Callable,
    *,
    use_jacrev: bool = False,
) -> Callable:
    """Return a jac_custom_fn that uses lineaxpr to compute Jacobians.

    Works for both Euclidean variables (tangent = parameter perturbation)
    and Lie-group variables (SE2Var, SE3Var, SO2Var, SO3Var).

    Parameters
    ----------
    compute_residual:
        Same function passed to Cost.factory.  Must only access the Var
        objects present in its *args* (standard jaxls practice).
    use_jacrev:
        Use lineaxpr.jacrev instead of jacfwd.  jacfwd (default) is
        optimal for n→n or overdetermined problems.

    Returns
    -------
    jac_fn : (vals, *args, **kwargs) -> jax.Array  shape (residual_dim, tangent_dim)
        Suitable for Cost.factory(jac_custom_fn=...).  Always returns a
        dense ndarray — jaxls requires dense Jacobians from custom jac fns.
    """
    materialize_jac = lineaxpr.jacrev if use_jacrev else lineaxpr.jacfwd

    def jac_fn(vals: jaxls.VarValues, *args: Any, **kwargs: Any) -> jax.Array:
        # Vars in positional args; kwargs hold plain data (weight, target, etc.)
        variables_raw = list(_yield_vars(args))
        if not variables_raw:
            raise ValueError("No jaxls.Var found in cost args.")

        # Sort to match jaxls' VarTypeOrdering (alphabetical by str(type)).
        # Within each type, preserve appearance order (caller must supply
        # variables in ascending-ID order — standard practice).
        type_groups: dict = defaultdict(list)
        for v in variables_raw:
            type_groups[type(v)].append(v)
        variables = []
        for t in sorted(type_groups.keys(), key=lambda t: str(t)):
            variables.extend(type_groups[t])

        sizes = [type(v).tangent_dim for v in variables]
        total = sum(sizes)

        # Current values — captured here, outside the linearise trace.
        # Inside jaxls' vmap these are BatchTracers (array or SE2/SE3 objects
        # with BatchTracer leaves); jax.linearize handles them as constants.
        current_vals = [vals[v] for v in variables]

        def residual_of_delta(delta: jax.Array) -> jax.Array:
            # Apply retraction for each variable — correct for both Euclidean
            # (retract = add) and Lie-group (retract = x @ exp(δ)).
            offset = 0
            retracted = []
            for v, current, size in zip(variables, current_vals, sizes):
                d_chunk = delta[offset : offset + size]
                retracted.append(type(v).retract_fn(current, d_chunk))
                offset += size

            mock_vals = _FlatVarValues(variables, retracted)
            return compute_residual(mock_vals, *args, **kwargs).ravel()

        return materialize_jac(residual_of_delta)(jnp.zeros(total))

    return jac_fn
