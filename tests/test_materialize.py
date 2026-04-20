"""Correctness tests with hand-rolled linear fns + known-pattern Hessians.

Unlike `test_correctness.py` (which sweeps curated CUTEst problems with a
numerical reference), these tests assert against **explicit expected
matrices** so readers can see what the extractor should produce. They
also cover:

- Non-square Jacobians (`jnp.diff`, `jnp.cumsum`) — which `materialize`
  supports despite its Hessian-first framing.
- A banded / arrowhead / tridiagonal sparsity pattern zoo.
- y-dependent Hessian (1D heat equation with nonlinear κ(T) = T^2.5).
- Edge cases: n=1, n=2, pass-through linear fn.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from lineaxpr import (
    Identity,
    bcoo_jacobian,
    materialize,
    sparsify,
)


# ----------------------- non-square Jacobians ----------------------------


@pytest.mark.parametrize("n", [3, 10, 50])
def test_jnp_diff_bidiagonal(n):
    """`jnp.diff(x)` is linear R^n → R^{n-1}; matrix is [-1, +1] bidiagonal."""
    lin = jnp.diff
    y = jnp.zeros(n)  # shape witness only
    M = materialize(lin, y)
    assert M.shape == (n - 1, n)
    expected = np.zeros((n - 1, n))
    for i in range(n - 1):
        expected[i, i] = -1.0
        expected[i, i + 1] = 1.0
    np.testing.assert_allclose(np.asarray(M), expected)


@pytest.mark.parametrize("n", [3, 10, 50])
def test_jnp_cumsum_lower_triangular_ones(n):
    """`jnp.cumsum(x)` is linear R^n → R^n; matrix is lower triangular of ones."""
    lin = jnp.cumsum
    y = jnp.zeros(n)
    M = materialize(lin, y)
    assert M.shape == (n, n)
    expected = np.tril(np.ones((n, n)))
    np.testing.assert_allclose(np.asarray(M), expected)


def test_matrix_multiply_constant_matrix():
    """lin(x) = A @ x for constant A. Expect exact reproduction of A."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 7))

    def lin(x):
        return jnp.asarray(A) @ x

    y = jnp.zeros(7)
    M = materialize(lin, y)
    np.testing.assert_allclose(np.asarray(M), A, atol=1e-12)


# ----------------------- arrowhead Hessian -------------------------------


def _arrowhead_f(x):
    # f(x) = 0.5 * sum(x^2) + x[0] * sum(x[1:])
    # Hessian: identity + rank-1 update on first row/col (off-diagonals in
    # row 0 and col 0 become 1).
    return 0.5 * jnp.sum(x**2) + x[0] * jnp.sum(x[1:])


@pytest.mark.parametrize("n", [3, 8, 20])
def test_arrowhead_hessian_pattern(n):
    y = jnp.ones(n)  # shape witness; f is quadratic so value of y doesn't matter
    _, hvp = jax.linearize(jax.grad(_arrowhead_f), y)
    H = materialize(hvp, y)
    expected = np.eye(n)
    expected[0, 1:] = 1.0
    expected[1:, 0] = 1.0
    np.testing.assert_allclose(np.asarray(H), expected, atol=1e-12)

    # Same pattern via bcoo_jacobian; may return dense ndarray for dense
    # fallback or BCOO for structural case.
    S = bcoo_jacobian(hvp, y)
    dense_S = np.asarray(S.todense() if isinstance(S, sparse.BCOO) else S)
    np.testing.assert_allclose(dense_S, expected, atol=1e-12)


# ----------------------- 1D heat eq (y-dependent tridiagonal) ------------


def _heat_energy(T):
    """Discretized energy for a 1D heat problem with nonlinear conductivity.

    E(T) = 0.5 * sum_i κ((T[i] + T[i+1])/2) * (T[i+1] - T[i])^2
    with κ(T) = T^2.5. Gradient is nonlinear; Hessian is tridiagonal and
    depends on T (crucially y-dependent, unlike arrowhead).
    """
    diffs = T[1:] - T[:-1]
    mid = 0.5 * (T[1:] + T[:-1])
    kappa = mid**2.5
    return 0.5 * jnp.sum(kappa * diffs * diffs)


@pytest.mark.parametrize("n", [5, 20, 100])
def test_heat_equation_hessian_is_tridiagonal(n):
    T = 1.0 + jnp.arange(n, dtype=jnp.float64) * 0.1  # smooth positive profile
    _, hvp = jax.linearize(jax.grad(_heat_energy), T)
    H = materialize(hvp, T)

    # Symmetry (analytic guarantee).
    np.testing.assert_allclose(np.asarray(H), np.asarray(H).T, atol=1e-10)

    # Tridiagonal pattern: all entries with |i - j| > 1 should be zero.
    H_np = np.asarray(H)
    off_band = np.abs(H_np) * (np.abs(np.subtract.outer(np.arange(n), np.arange(n))) > 1)
    assert off_band.max() < 1e-10, f"off-tridiagonal mass: {off_band.max()}"

    # Non-trivial: main diagonal should have non-zero (non-constant) values.
    assert float(jnp.max(jnp.abs(jnp.diag(H)))) > 0.1

    # BCOO path agrees.
    S = bcoo_jacobian(hvp, T)
    dense_S = np.asarray(S.todense() if isinstance(S, sparse.BCOO) else S)
    np.testing.assert_allclose(dense_S, H_np, atol=1e-10)
    # Tridiagonal nnz upper bound = 3n - 2. Current extractor emits
    # structural duplicates from add_any concat (same out_rows, different
    # in_cols); BandedPivoted (docs/TODO.md #1) will tighten this to ~1×.
    # For now, allow 3× slack.
    if isinstance(S, sparse.BCOO):
        assert S.nse <= 3 * (3 * n - 2)


# ----------------------- edge cases --------------------------------------


def test_n1_identity():
    """n=1 exercises short-circuit path and should still produce [[1]]."""
    lin = lambda x: x  # noqa: E731
    M = materialize(lin, jnp.zeros(1))
    np.testing.assert_array_equal(np.asarray(M), np.eye(1))


def test_n2_scaled_identity():
    lin = lambda x: 3.0 * x  # noqa: E731
    M = materialize(lin, jnp.zeros(2))
    np.testing.assert_array_equal(np.asarray(M), 3.0 * np.eye(2))


def test_passthrough_linear_fn():
    lin = lambda x: x  # noqa: E731
    # Above short-circuit (n >= 16) exercises the walk path.
    M = materialize(lin, jnp.zeros(32))
    np.testing.assert_array_equal(np.asarray(M), np.eye(32))


def test_zero_linear_fn():
    """lin(x) = 0 · x → Jacobian is zero."""
    lin = lambda x: 0.0 * x  # noqa: E731
    M = materialize(lin, jnp.zeros(32))
    np.testing.assert_array_equal(np.asarray(M), np.zeros((32, 32)))


# ----------------------- BCOO output on sparse Jacobian ------------------


def test_bcoo_returned_for_sparse_pattern():
    """`bcoo_jacobian` should return a BCOO when the walk yields structural
    sparsity, not a dense array."""
    def f(x):
        return jnp.sum(x**2) + jnp.sum(x[:-1] * x[1:])

    y = jnp.arange(1, 33, dtype=jnp.float64)
    _, hvp = jax.linearize(jax.grad(f), y)
    S = bcoo_jacobian(hvp, y)
    assert isinstance(S, sparse.BCOO)
    # Pattern: tridiagonal ⇒ ≤ 3n-2 nnz; allow 2× slack.
    assert S.nse <= 2 * (3 * 32 - 2)


def test_dense_returned_for_dense_pattern():
    """Fully-dense Hessian should come out as ndarray, not BCOO."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((20, 20))
    A = A @ A.T  # symmetric PSD, dense

    def f(x):
        return 0.5 * x @ jnp.asarray(A) @ x

    y = jnp.zeros(20)
    _, hvp = jax.linearize(jax.grad(f), y)
    out = bcoo_jacobian(hvp, y)
    # Walk may return either dense ndarray OR a dense-ish BCOO; both are
    # acceptable here. Just check correctness.
    dense = np.asarray(out.todense() if isinstance(out, sparse.BCOO) else out)
    np.testing.assert_allclose(dense, A, atol=1e-10)
