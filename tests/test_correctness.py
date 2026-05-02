"""Bit-exact correctness on a curated CUTEst subset.

Each test: `lineaxpr.hessian(f)(y)` and `lineaxpr.bcoo_hessian(f)(y)`
should match `jax.vmap(hvp)(jnp.eye(n)).T` reference to within float64
noise.

Requires `sif2jax` installed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import lineaxpr


pytest.importorskip("sif2jax")


CASES = [
    # (module, class_name, expected_tolerance)
    ("sif2jax.cutest._quadratic_problems.dual1", "DUAL1", 0.0),
    ("sif2jax.cutest._unconstrained_minimisation.fletchcr", "FLETCHCR", 0.0),
    ("sif2jax.cutest._unconstrained_minimisation.genrose", "GENROSE", 1e-12),
    ("sif2jax.cutest._bounded_minimisation.hs110", "HS110", 1e-15),
    ("sif2jax.cutest._bounded_minimisation.levymont", "LEVYMONT", 1e-14),
    ("sif2jax.cutest._unconstrained_minimisation.argtrigls", "ARGTRIGLS", 1e-10),
    ("sif2jax.cutest._unconstrained_minimisation.dixmaana1", "DIXMAANA1", 0.0),
    ("sif2jax.cutest._quadratic_problems.cmpc1", "CMPC1", 0.0),
    ("sif2jax.cutest._unconstrained_minimisation.luksan17ls", "LUKSAN17LS", 1e-11),
    ("sif2jax.cutest._unconstrained_minimisation.tointgor", "TOINTGOR", 1e-13),
]


def _load(modpath, clsname):
    mod = __import__(modpath, fromlist=[clsname])
    return getattr(mod, clsname)()


def _reference_hessian(f, y):
    """jax.vmap(hvp)(jnp.eye(n)).T — our correctness ground truth."""
    _, hvp = jax.linearize(jax.grad(f), y)
    return jax.vmap(hvp)(jnp.eye(y.shape[0])).T


@pytest.mark.parametrize("modpath,clsname,tol", CASES,
                          ids=[c[1] for c in CASES])
def test_hessian_matches_vmap_eye(modpath, clsname, tol):
    p = _load(modpath, clsname)
    y = p.y0

    def f(z):
        return p.objective(z, p.args)

    H_ref = _reference_hessian(f, y)

    H_dense = lineaxpr.hessian(f)(y)
    err_dense = float(jnp.max(jnp.abs(H_dense - H_ref)))
    assert err_dense <= tol, f"hessian err {err_dense} > tol {tol}"

    H_bcoo = lineaxpr.bcoo_hessian(f)(y)
    H_bcoo_dense = H_bcoo.todense() if hasattr(H_bcoo, "todense") else H_bcoo
    err_bcoo = float(jnp.max(jnp.abs(H_bcoo_dense - H_ref)))
    assert err_bcoo <= tol, f"bcoo_hessian err {err_bcoo} > tol {tol}"


@pytest.mark.parametrize("modpath,clsname,tol", CASES,
                          ids=[c[1] for c in CASES])
def test_hessian_inside_jit(modpath, clsname, tol):
    p = _load(modpath, clsname)
    y = p.y0

    def f(z):
        return p.objective(z, p.args)

    hessian_jit = jax.jit(lineaxpr.hessian(f))
    H_ref = _reference_hessian(f, y)
    H = hessian_jit(y)
    err = float(jnp.max(jnp.abs(H - H_ref)))
    assert err <= tol
