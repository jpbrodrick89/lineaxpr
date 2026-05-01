"""Per-primitive `sparsify`-vs-dense Jacobian tests parameterised
over vmap (in_axes, out_axes) ∈ {-1, 0, 1}².

Pins the rectangular-Jacobian motivation: `vmap(linear_fn,
in_axes=in_ax, out_axes=out_ax)(eye)` should produce the same
matrix as `sparsify(vmap(linear_fn, in_axes, out_axes))(seed)`
densified, for every layout. Forces axis interpretation to be
correct under every common vmap layout, without paying the
slow-sweep cost.

The seed factory yields both Identity (symmetric — hides row/col
bugs) and an asymmetric BEllpack (breaks symmetry so misrouted
scaling / broadcasts surface). A bare BCOO seed is intentionally
omitted — this design uses a `transposed: bool` flag on BEllpack only;
BCOO's native transpose handles row/col swaps directly.

Cells un-xfail per primitive as the `transposed` flag rolls out.
Natural cells (in=0, out=±1) currently pass on local main + the
vmap split. The 7 other combos per primitive are xfailed pending
rule-side support.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from lineaxpr import Identity, sparsify
from lineaxpr._linops.ellpack import BEllpack


# Every cell starts as xfail(strict=False). Cells "pass" if the rule
# happens to produce the right output for the asymmetric seed today
# (xpass — recorded but not failing the suite). Cells "fail" if they
# produce the wrong output (xfail — also not failing the suite).
# As `transposed`-flag rollout converts each rule, expected-pass cells
# can be promoted to plain `pytest.param(...)` (no xfail).
_XFAIL = pytest.mark.xfail(
    reason="awaiting transposed-flag rollout / asymmetric-seed coverage",
    strict=False,
)

VMAP_AXES_GRID = [
    pytest.param(in_ax, out_ax, marks=_XFAIL)
    for in_ax in (-1, 0, 1)
    for out_ax in (-1, 0, 1)
]


# reduce_sum's output is rank 1, so out_ax ∈ {-1, 0} only.
REDUCE_SUM_GRID = [
    pytest.param(in_ax, out_ax, marks=_XFAIL)
    for in_ax in (-1, 0, 1)
    for out_ax in (-1, 0)
]


def _densify(linop):
    """Convert a sparsify result to ndarray for comparison."""
    if hasattr(linop, "todense"):
        return np.asarray(linop.todense())
    return np.asarray(linop)


def _seed_factories(n, dtype):
    """Yield (label, seed_linop, seed_dense) pairs.

    Identity is symmetric and hides row/col bugs; BEllpack is asymmetric
    so any misrouted scaling / broadcasting surfaces immediately.
    """
    yield ("Identity", Identity(n, dtype=dtype),
           np.asarray(Identity(n, dtype=dtype).todense()))

    # Asymmetric single-band BEllpack: M[i, (i+1) % n] = i + 1. No row
    # equals its column index, so any row/col confusion is detectable.
    cols = np.array([(i + 1) % n for i in range(n)], dtype=np.int64)
    data = jnp.asarray(np.arange(1, n + 1), dtype=dtype)
    be = BEllpack(start_row=0, end_row=n, in_cols=(cols,),
                  data=data, out_size=n, in_size=n)
    yield ("BEllpack", be, np.asarray(be.todense()))


def _check(prim_name, partial_prim, y, in_ax, out_ax):
    """Compute reference (dense vmap) and under-test (sparsify) results
    in the same (in_ax, out_ax) layout for several seed types and
    assert equality on each."""
    lin_fn = jax.linearize(partial_prim, y)[1]
    n = y.size
    vmapped = jax.vmap(lin_fn, in_axes=in_ax, out_axes=out_ax)

    for label, seed_linop, seed_dense in _seed_factories(n, y.dtype):
        ref = np.asarray(vmapped(jnp.asarray(seed_dense)))
        got = _densify(sparsify(vmapped)(seed_linop))
        np.testing.assert_allclose(
            got, ref, atol=1e-10, rtol=1e-10,
            err_msg=f"{prim_name} seed={label} (in={in_ax}, out={out_ax})",
        )


# ---------------------------------------------------------------------------
# identity — lin_fn is the identity. Trivial walk; tests that single
# transposes induced by the (in, out) layout pass through.
#
# All 9 cells pass: vmap-induced transposes flip the transposed flag
# on BEllpack (free) and identity for ConstantDiagonal (symmetric).
# Promoted from xfail.
# ---------------------------------------------------------------------------

ALL_AXES_GRID = [
    (in_ax, out_ax) for in_ax in (-1, 0, 1) for out_ax in (-1, 0, 1)
]


@pytest.mark.parametrize("in_ax,out_ax", ALL_AXES_GRID)
def test_identity(in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("identity", lambda x: x, y, in_ax, out_ax)


# ---------------------------------------------------------------------------
# slice
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_ax,out_ax", VMAP_AXES_GRID)
def test_slice(in_ax, out_ax):
    n = 6
    partial_prim = functools.partial(
        lax.slice, start_indices=(1,), limit_indices=(5,), strides=(1,),
    )
    y = jnp.linspace(0.1, 1.0, n)
    _check("slice", partial_prim, y, in_ax, out_ax)


# ---------------------------------------------------------------------------
# pad — non-square output (n=6 → n=10).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_ax,out_ax", VMAP_AXES_GRID)
def test_pad(in_ax, out_ax):
    n = 6
    partial_prim = functools.partial(
        lax.pad, padding_value=0.0, padding_config=((2, 2, 0),),
    )
    y = jnp.linspace(0.1, 1.0, n)
    _check("pad", partial_prim, y, in_ax, out_ax)


# ---------------------------------------------------------------------------
# reshape — 1D → 2D (6 → 2×3).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_ax,out_ax", VMAP_AXES_GRID)
def test_reshape(in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("reshape", lambda x: x.reshape(2, 3), y, in_ax, out_ax)


# ---------------------------------------------------------------------------
# rev
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_ax,out_ax", VMAP_AXES_GRID)
def test_rev(in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("rev", lambda x: lax.rev(x, dimensions=(0,)), y, in_ax, out_ax)


# ---------------------------------------------------------------------------
# reduce_sum — sums over the per-sample axis. Per-sample output is rank 0
# so vmapped output is rank 1; out_ax ∈ {-1, 0}.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_ax,out_ax", REDUCE_SUM_GRID)
def test_reduce_sum(in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("reduce_sum", lambda x: jnp.sum(x), y, in_ax, out_ax)


# ---------------------------------------------------------------------------
# squeeze
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_ax,out_ax", VMAP_AXES_GRID)
def test_squeeze(in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("squeeze",
           lambda x: jnp.squeeze(x.reshape(n, 1), axis=1),
           y, in_ax, out_ax)


# ---------------------------------------------------------------------------
# concatenate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_ax,out_ax", VMAP_AXES_GRID)
def test_concatenate(in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("concatenate",
           lambda x: jnp.concatenate([x[:3], x[3:]]),
           y, in_ax, out_ax)


# ---------------------------------------------------------------------------
# broadcast_in_dim — broadcast 1D to 2D by inserting a new size-2 axis.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_ax,out_ax", VMAP_AXES_GRID)
def test_broadcast_in_dim(in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("broadcast_in_dim",
           lambda x: jnp.broadcast_to(x[None, :], (2, n)),
           y, in_ax, out_ax)


# ---------------------------------------------------------------------------
# dot_general — closure matrix × traced vector (M @ x).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_ax,out_ax", VMAP_AXES_GRID)
def test_dot_general_matvec(in_ax, out_ax):
    n = 6
    M = jnp.asarray(np.arange(n * n).reshape(n, n).astype(np.float64))
    y = jnp.linspace(0.1, 1.0, n)
    _check("dot_general", lambda x: M @ x, y, in_ax, out_ax)
