"""Per-primitive `sparsify`-vs-dense Jacobian tests parameterised
over vmap (in_axes, out_axes) ∈ {-1, 0, 1}².

Phase 5 makes lineaxpr's walker layout-flexible: rules pass jaxpr
params through unchanged, format ops consult `row_axis`/`col_axis`.
This grid exercises each primitive's rule under every input/output
vmap configuration so silent bugs in axis interpretation surface
immediately, without paying the slow-sweep cost.

We test `sparsify` directly (not `materialize`) because `sparsify` is
the layer that consumes the vmapped jaxpr; phase 5's invariant is
that sparsify handles any vmap configuration. `materialize` will pin
a single (in_axes, out_axes) choice once phase 5 lands; users who
need a different layout can call `sparsify(jax.vmap(f, in, out))`
directly.

Comparison: dense reference uses
`jax.vmap(linear_fn, in_axes=in_ax, out_axes=out_ax)(eye)`.
Under-test path is `lineaxpr.sparsify(jax.vmap(linear_fn, in_axes,
out_axes))(seed)` densified. Both should produce the same array.

Stage A scaffold: `slice` covered. Stage C adds one row per primitive
as it's converted.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from lineaxpr import Identity, sparsify


# (in_axes, out_axes) ∈ {-1, 0, 1}² for inputs/outputs of rank ≥ 2.
# At stage A+B, only combos where the vmap layout coincides with
# lineaxpr's (out, in) convention pass. Stage C activates row_axis /
# col_axis interpretation per primitive — the corresponding
# parametrize calls drop the xfail marker as each primitive lands.
def _xfail_unless_natural(in_ax, out_ax):
    natural = (in_ax == 0 and out_ax in (-1, 1))
    if natural:
        return pytest.param(in_ax, out_ax)
    return pytest.param(
        in_ax, out_ax,
        marks=pytest.mark.xfail(
            reason="stage C: sparsify needs row_axis/col_axis for this layout",
            strict=False,
        ),
    )


VMAP_AXES_GRID = [
    _xfail_unless_natural(in_ax, out_ax)
    for in_ax in (-1, 0, 1)
    for out_ax in (-1, 0, 1)
]


def _densify(linop):
    """Convert a sparsify result to ndarray for comparison."""
    if hasattr(linop, "todense"):
        return np.asarray(linop.todense())
    return np.asarray(linop)


def _check(prim_name, partial_prim, y, in_ax, out_ax):
    """Compute reference (dense vmap) and under-test (sparsify) Jacobians
    in the same (in_ax, out_ax) layout and assert equality."""
    lin_fn = jax.linearize(partial_prim, y)[1]
    n = y.size
    eye = jnp.eye(n, dtype=y.dtype)
    vmapped = jax.vmap(lin_fn, in_axes=in_ax, out_axes=out_ax)

    ref = np.asarray(vmapped(eye))
    seed = Identity(n, dtype=y.dtype)
    got = _densify(sparsify(vmapped)(seed))

    np.testing.assert_allclose(
        got, ref, atol=1e-12, rtol=1e-12,
        err_msg=f"{prim_name} (in={in_ax}, out={out_ax})",
    )


# ---------------------------------------------------------------------------
# slice (stage A scaffold; full grid lights up in stage C)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_ax,out_ax", VMAP_AXES_GRID)
def test_slice(in_ax, out_ax):
    n = 6
    partial_prim = functools.partial(
        lax.slice, start_indices=(1,), limit_indices=(5,), strides=(1,),
    )
    y = jnp.linspace(0.1, 1.0, n)
    _check("slice", partial_prim, y, in_ax, out_ax)
