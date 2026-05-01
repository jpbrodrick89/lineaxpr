"""Per-primitive `sparsify`-vs-dense Jacobian tests parameterised over
vmap (in_axes, out_axes) ∈ {-1, 0, 1}² and seed kind.

Pins the rectangular-Jacobian motivation: `vmap(linear_fn, in_axes,
out_axes)(seed_dense)` should equal `sparsify(vmap(linear_fn, in_axes,
out_axes))(seed_linop).todense()` for every layout and seed type.

Two seed kinds:
- **Identity**: symmetric. Most cells already pass under the current
  walker. Only `(0, -1)` is universally-passing across all primitives
  (the layout where dense `vmap(lin, 0, -1)(seed) == J` and the walker's
  natural canonical output also equals J after the boundary transpose
  no-op). That cell is a plain regression test. Other Identity cells
  may xpass for some primitives; those become promotion candidates as
  Phase B aligns sparsify with dense vmap.
- **BEllpack**: asymmetric. Walker's semantics for asymmetric seeds
  doesn't match dense vmap until Phase B; BE seed cells xfail by
  design and are tracked as the desired end goal.
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


# Universal passing set: the (seed_kind, in_ax, out_ax) cells that pass
# for every primitive's grid. Cells in this set are plain pytest.param
# and act as regression tests; everything else is xfail(strict=False).
UNIVERSAL_PASSING: dict[str, set[tuple[int, int]]] = {
    "Identity": {(0, -1)},
    "BEllpack": set(),
}

_XFAIL_MARK = pytest.mark.xfail(
    reason="awaiting Phase-B walker invariant alignment with dense vmap",
    strict=False,
)


def _grid(out_axes=(-1, 0, 1)):
    cells = []
    for seed_kind in ("Identity", "BEllpack"):
        passing = UNIVERSAL_PASSING[seed_kind]
        for in_ax in (-1, 0, 1):
            for out_ax in out_axes:
                pid = f"{seed_kind}-{in_ax}-{out_ax}"
                if (in_ax, out_ax) in passing:
                    cells.append(pytest.param(seed_kind, in_ax, out_ax, id=pid))
                else:
                    cells.append(pytest.param(
                        seed_kind, in_ax, out_ax, id=pid, marks=_XFAIL_MARK,
                    ))
    return cells


GRID = _grid()
REDUCE_SUM_GRID = _grid(out_axes=(-1, 0))


def _densify(linop):
    if hasattr(linop, "todense"):
        return np.asarray(linop.todense())
    return np.asarray(linop)


def _make_seed(seed_kind: str, n: int, dtype):
    if seed_kind == "Identity":
        seed_linop = Identity(n, dtype=dtype)
        return seed_linop, np.asarray(seed_linop.todense())
    if seed_kind == "BEllpack":
        # Asymmetric single-band: M[i, (i+1) % n] = i + 1. No row equals
        # its column index, so any row/col confusion is detectable.
        cols = np.array([(i + 1) % n for i in range(n)], dtype=np.int64)
        data = jnp.asarray(np.arange(1, n + 1), dtype=dtype)
        be = BEllpack(start_row=0, end_row=n, in_cols=(cols,),
                      data=data, out_size=n, in_size=n)
        return be, np.asarray(be.todense())
    raise ValueError(f"unknown seed_kind: {seed_kind}")


def _check(prim_name, partial_prim, y, seed_kind, in_ax, out_ax):
    n = y.size
    seed_linop, seed_dense = _make_seed(seed_kind, n, y.dtype)
    lin_fn = jax.linearize(partial_prim, y)[1]
    vmapped = jax.vmap(lin_fn, in_axes=in_ax, out_axes=out_ax)
    ref = np.asarray(vmapped(jnp.asarray(seed_dense)))
    got = _densify(sparsify(vmapped)(seed_linop))
    np.testing.assert_allclose(
        got, ref, atol=1e-10, rtol=1e-10,
        err_msg=f"{prim_name} seed={seed_kind} (in={in_ax}, out={out_ax})",
    )


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed_kind,in_ax,out_ax", GRID)
def test_identity(seed_kind, in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("identity", lambda x: x, y, seed_kind, in_ax, out_ax)


@pytest.mark.parametrize("seed_kind,in_ax,out_ax", GRID)
def test_slice(seed_kind, in_ax, out_ax):
    n = 6
    partial_prim = functools.partial(
        lax.slice, start_indices=(1,), limit_indices=(5,), strides=(1,),
    )
    y = jnp.linspace(0.1, 1.0, n)
    _check("slice", partial_prim, y, seed_kind, in_ax, out_ax)


@pytest.mark.parametrize("seed_kind,in_ax,out_ax", GRID)
def test_pad(seed_kind, in_ax, out_ax):
    n = 6
    partial_prim = functools.partial(
        lax.pad, padding_value=0.0, padding_config=((2, 2, 0),),
    )
    y = jnp.linspace(0.1, 1.0, n)
    _check("pad", partial_prim, y, seed_kind, in_ax, out_ax)


@pytest.mark.parametrize("seed_kind,in_ax,out_ax", GRID)
def test_reshape(seed_kind, in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("reshape", lambda x: x.reshape(2, 3), y, seed_kind, in_ax, out_ax)


@pytest.mark.parametrize("seed_kind,in_ax,out_ax", GRID)
def test_rev(seed_kind, in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("rev", lambda x: lax.rev(x, dimensions=(0,)), y, seed_kind, in_ax, out_ax)


@pytest.mark.parametrize("seed_kind,in_ax,out_ax", REDUCE_SUM_GRID)
def test_reduce_sum(seed_kind, in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("reduce_sum", lambda x: jnp.sum(x), y, seed_kind, in_ax, out_ax)


@pytest.mark.parametrize("seed_kind,in_ax,out_ax", GRID)
def test_squeeze(seed_kind, in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("squeeze",
           lambda x: jnp.squeeze(x.reshape(n, 1), axis=1),
           y, seed_kind, in_ax, out_ax)


@pytest.mark.parametrize("seed_kind,in_ax,out_ax", GRID)
def test_concatenate(seed_kind, in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("concatenate",
           lambda x: jnp.concatenate([x[:3], x[3:]]),
           y, seed_kind, in_ax, out_ax)


@pytest.mark.parametrize("seed_kind,in_ax,out_ax", GRID)
def test_broadcast_in_dim(seed_kind, in_ax, out_ax):
    n = 6
    y = jnp.linspace(0.1, 1.0, n)
    _check("broadcast_in_dim",
           lambda x: jnp.broadcast_to(x[None, :], (2, n)),
           y, seed_kind, in_ax, out_ax)


@pytest.mark.parametrize("seed_kind,in_ax,out_ax", GRID)
def test_dot_general_matvec(seed_kind, in_ax, out_ax):
    n = 6
    M = jnp.asarray(np.arange(n * n).reshape(n, n).astype(np.float64))
    y = jnp.linspace(0.1, 1.0, n)
    _check("dot_general", lambda x: M @ x, y, seed_kind, in_ax, out_ax)
