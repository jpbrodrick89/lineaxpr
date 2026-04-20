"""Bit-exact correctness on a curated CUTEst subset.

Each test: materialize + bcoo_jacobian of the Hessian should match
`jax.vmap(hvp)(jnp.eye(n)).T` reference to within float64 noise.

Requires `sif2jax` installed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from lineaxpr import bcoo_jacobian, materialize

jax.config.update("jax_enable_x64", True)


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
    ("sif2jax.cutest._unconstrained_minimisation.tointgor", "TOINTGOR", 0.0),
]


def _load(modpath, clsname):
    mod = __import__(modpath, fromlist=[clsname])
    return getattr(mod, clsname)()


@pytest.mark.parametrize("modpath,clsname,tol", CASES,
                          ids=[c[1] for c in CASES])
def test_materialize_matches_vmap_eye(modpath, clsname, tol):
    p = _load(modpath, clsname)
    y = p.y0
    n = y.shape[0]

    def f(z):
        return p.objective(z, p.args)

    _, hvp = jax.linearize(jax.grad(f), y)
    H_ref = jax.vmap(hvp)(jnp.eye(n)).T

    H_mat = materialize(hvp, y)
    err_mat = float(jnp.max(jnp.abs(H_mat - H_ref)))
    assert err_mat <= tol, f"materialize err {err_mat} > tol {tol}"

    H_bcoo = bcoo_jacobian(hvp, y)
    H_bcoo_dense = H_bcoo.todense() if hasattr(H_bcoo, "todense") else H_bcoo
    err_bcoo = float(jnp.max(jnp.abs(H_bcoo_dense - H_ref)))
    assert err_bcoo <= tol, f"bcoo_jacobian err {err_bcoo} > tol {tol}"


@pytest.mark.parametrize("modpath,clsname,tol", CASES,
                          ids=[c[1] for c in CASES])
def test_inside_jit(modpath, clsname, tol):
    p = _load(modpath, clsname)
    y = p.y0
    n = y.shape[0]

    def f(z):
        return p.objective(z, p.args)

    @jax.jit
    def H_of(y_):
        _, hvp = jax.linearize(jax.grad(f), y_)
        return materialize(hvp, y_)

    _, hvp = jax.linearize(jax.grad(f), y)
    H_ref = jax.vmap(hvp)(jnp.eye(n)).T
    H = H_of(y)
    err = float(jnp.max(jnp.abs(H - H_ref)))
    assert err <= tol
