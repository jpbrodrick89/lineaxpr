"""Tests for the jax-like public API: jacfwd, jacrev, hessian, and their
bcoo_-prefixed counterparts.

These are the user-facing entry points for most workflows. They wrap the
lower-level `materialize(linear_fn, primal, format=...)` transform.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

import lineaxpr


# --------------------------- fixtures / helpers ---------------------------


def _arrowhead(n):
    """f(x) = 0.5 * sum(x^2) + x[0] * sum(x[1:]). Known-sparse Hessian."""
    def f(x):
        return 0.5 * jnp.sum(x**2) + x[0] * jnp.sum(x[1:])
    return f


def _expected_arrowhead_H(n):
    H = np.eye(n)
    H[0, 1:] = 1.0
    H[1:, 0] = 1.0
    return H


def _fetch_dense(out):
    """Normalize BCOO-or-ndarray to a dense ndarray for comparison."""
    if isinstance(out, sparse.BCOO):
        return out.todense()
    return out


# --------------------------- hessian --------------------------------------


@pytest.mark.parametrize("n", [4, 20, 64])
def test_hessian_matches_jax_hessian(n):
    f = _arrowhead(n)
    y = jnp.ones(n)
    ours = lineaxpr.hessian(f)(y)
    ref = jax.hessian(f)(y)
    np.testing.assert_allclose(np.asarray(ours), np.asarray(ref), atol=1e-12)


@pytest.mark.parametrize("n", [4, 20, 64])
def test_bcoo_hessian_matches_jax_hessian(n):
    f = _arrowhead(n)
    y = jnp.ones(n)
    ours = lineaxpr.bcoo_hessian(f)(y)
    ref = jax.hessian(f)(y)
    np.testing.assert_allclose(_fetch_dense(ours), np.asarray(ref), atol=1e-12)


def test_hessian_format_kwarg_equivalent_to_prefixed():
    f = _arrowhead(10)
    y = jnp.ones(10)
    H_named = lineaxpr.bcoo_hessian(f)(y)
    H_kwarg = lineaxpr.hessian(f, format="bcoo")(y)
    np.testing.assert_array_equal(_fetch_dense(H_named), _fetch_dense(H_kwarg))


# --------------------------- jacfwd / jacrev ------------------------------


@pytest.mark.parametrize("n", [5, 32])
def test_jacfwd_jnp_diff_matches_jax(n):
    """jnp.diff is R^n -> R^{n-1}, non-square."""
    f = jnp.diff
    y = jnp.arange(n, dtype=jnp.float64)
    ref = jax.jacfwd(f)(y)
    ours = lineaxpr.jacfwd(f)(y)
    assert ours.shape == ref.shape == (n - 1, n)
    np.testing.assert_allclose(np.asarray(ours), np.asarray(ref), atol=1e-12)


@pytest.mark.parametrize("n", [5, 32])
def test_jacrev_jnp_diff_matches_jax(n):
    """jacrev should give same matrix as jacfwd — same Jacobian, diff algo."""
    f = jnp.diff
    y = jnp.arange(n, dtype=jnp.float64)
    ref = jax.jacrev(f)(y)
    ours = lineaxpr.jacrev(f)(y)
    assert ours.shape == ref.shape == (n - 1, n)
    np.testing.assert_allclose(np.asarray(ours), np.asarray(ref), atol=1e-12)


@pytest.mark.parametrize("n", [8, 32])
def test_jacfwd_jacrev_agree(n):
    """Matrix should be identical regardless of fwd vs rev."""
    f = lambda x: jnp.sin(x) * jnp.cumsum(x)  # noqa: E731
    y = jnp.arange(1, n + 1, dtype=jnp.float64) * 0.1
    Jf = lineaxpr.jacfwd(f)(y)
    Jr = lineaxpr.jacrev(f)(y)
    np.testing.assert_allclose(np.asarray(Jf), np.asarray(Jr), atol=1e-10)


@pytest.mark.parametrize("fn_name,n", [("jacfwd", 20), ("jacrev", 20)])
def test_bcoo_variants_match_dense(fn_name, n):
    f = lambda x: jnp.diff(x) * x[:-1]  # noqa: E731
    y = jnp.arange(1, n + 1, dtype=jnp.float64)
    dense_fn = getattr(lineaxpr, fn_name)
    bcoo_fn = getattr(lineaxpr, f"bcoo_{fn_name}")
    dense = dense_fn(f)(y)
    bcoo = bcoo_fn(f)(y)
    np.testing.assert_allclose(np.asarray(dense), _fetch_dense(bcoo), atol=1e-12)


# --------------------------- materialize format ---------------------------


def test_materialize_format_dense():
    def lin(x): return jnp.cumsum(x)
    y = jnp.zeros(32, dtype=jnp.float64)
    out = lineaxpr.materialize(lin, y, format="dense")
    assert not isinstance(out, sparse.BCOO)
    np.testing.assert_array_equal(np.asarray(out), np.tril(np.ones((32, 32))))


def test_materialize_format_bcoo():
    def lin(x): return jnp.diff(x)  # sparse bidiagonal
    y = jnp.zeros(32, dtype=jnp.float64)
    out = lineaxpr.materialize(lin, y, format="bcoo")
    # bidiagonal should survive as BCOO.
    assert isinstance(out, sparse.BCOO)


def test_materialize_invalid_format():
    def lin(x): return x
    y = jnp.zeros(32, dtype=jnp.float64)
    with pytest.raises(ValueError, match="format must be"):
        lineaxpr.materialize(lin, y, format="csr")


def test_materialize_default_is_dense():
    def lin(x): return x
    y = jnp.zeros(32, dtype=jnp.float64)
    out = lineaxpr.materialize(lin, y)  # no format kwarg
    assert not isinstance(out, sparse.BCOO)


# --------------------------- inside jit -----------------------------------


def test_hessian_inside_jit():
    """The composed transform should jit cleanly."""
    def f(x): return 0.5 * jnp.sum(x**2) + x[0] * jnp.sum(x[1:])

    @jax.jit
    def H_of(y):
        return lineaxpr.hessian(f)(y)

    y = jnp.arange(1, 21, dtype=jnp.float64)
    H = H_of(y)
    ref = jax.hessian(f)(y)
    np.testing.assert_allclose(np.asarray(H), np.asarray(ref), atol=1e-10)


def test_bcoo_jacfwd_inside_jit():
    def f(x): return jnp.diff(x) * x[:-1]

    @jax.jit
    def J_of(y):
        return lineaxpr.bcoo_jacfwd(f)(y)

    y = jnp.arange(1, 17, dtype=jnp.float64)
    J = J_of(y)
    ref = jax.jacfwd(f)(y)
    np.testing.assert_allclose(_fetch_dense(J), np.asarray(ref), atol=1e-12)
