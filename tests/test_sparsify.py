"""Transform-level tests for `lineaxpr.sparsify`.

Unlike `test_materialize.py` (hand-rolled problems via the jax-like
public API), these tests exercise the primitive transform — custom
seeds, const-prop folding, nested jit, missing-primitive error format,
and multi-output error handling.

`sparsify` IS the thing under test here, so direct calls are correct;
higher-level tests should prefer `lineaxpr.jacfwd` / `hessian` / etc.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineaxpr import (
    ConstantDiagonal,
    Diagonal,
    Identity,
    materialize_rules,
    sparsify,
    to_dense,
)


# ----------------------- identity seed vs reference ----------------------


@pytest.mark.parametrize("n", [4, 17, 64])
def test_identity_seed_matches_vmap_eye(n):
    def lin(x):
        return 2.0 * x + jnp.pad(x[:-1], (1, 0))

    seed = Identity(n, dtype=jnp.float64)
    linop = sparsify(lin)(seed)
    ours = to_dense(linop)
    ref = jax.vmap(lin)(jnp.eye(n)).T
    np.testing.assert_allclose(np.asarray(ours), np.asarray(ref), atol=1e-12)


# ----------------------- custom seeds ------------------------------------


def test_constant_diagonal_seed_scales_output():
    """Seeding with ConstantDiagonal(n, k) should produce k × output."""
    n = 8

    def lin(x):
        return 3.0 * x

    out_identity = to_dense(sparsify(lin)(Identity(n, dtype=jnp.float64)))
    out_scaled = to_dense(sparsify(lin)(ConstantDiagonal(n, value=2.0)))
    np.testing.assert_allclose(
        np.asarray(out_scaled), 2.0 * np.asarray(out_identity), atol=1e-12
    )


def test_diagonal_seed_scales_per_column():
    """Seeding with Diagonal(v) should produce output @ diag(v) = cols scaled."""
    n = 6
    v = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def lin(x):
        return jnp.cumsum(x)

    out_identity = to_dense(sparsify(lin)(Identity(n, dtype=jnp.float64)))
    out_diag = to_dense(sparsify(lin)(Diagonal(v)))
    # sparsify(lin)(Diagonal(v)) computes lin as applied to diag(v)'s columns:
    # column i of output is lin(v[i] * e_i) = v[i] * lin(e_i). So the
    # result is out_identity with each column j scaled by v[j].
    expected = np.asarray(out_identity) * np.asarray(v)[None, :]
    np.testing.assert_allclose(np.asarray(out_diag), expected, atol=1e-12)


# ----------------------- constant propagation ----------------------------


def test_constant_hessian_folds_to_literal():
    """For a constant-H problem (pure quadratic), the walk's const-prop
    path should produce a trace-time literal. Verify by tracing under jit
    and checking the output is returned (same values, same compile-time
    constant shape)."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((12, 12))
    A = A @ A.T  # symmetric constant

    def f(x):
        return 0.5 * x @ jnp.asarray(A) @ x

    @jax.jit
    def extract(y):
        _, hvp = jax.linearize(jax.grad(f), y)
        return to_dense(sparsify(hvp)(Identity(y.size, dtype=y.dtype)))

    y0 = jnp.zeros(12)
    H = extract(y0)
    np.testing.assert_allclose(np.asarray(H), A, atol=1e-10)


# ----------------------- nested jit --------------------------------------


def test_nested_jit_inside_linear_fn():
    """The linear fn body contains a nested @jax.jit — walker's _jit_rule
    should recurse into the inner jaxpr."""

    @jax.jit
    def scaled(x):
        return 2.5 * x

    def f(x):
        return 0.5 * jnp.sum(scaled(x) ** 2)

    y = jnp.zeros(20)
    _, hvp = jax.linearize(jax.grad(f), y)
    out = to_dense(sparsify(hvp)(Identity(20, dtype=y.dtype)))
    # Hessian is (2.5)^2 · I = 6.25 · I.
    np.testing.assert_allclose(np.asarray(out), 6.25 * np.eye(20), atol=1e-12)


# ----------------------- error messages ----------------------------------


def test_missing_primitive_error_format():
    """Missing-rule error should name the primitive and include the input
    LinOp forms so users can diagnose quickly."""
    # Trigger by temporarily removing a rule and invoking a fn that needs it.
    prim = None
    for p in materialize_rules:
        if p.name == "mul":
            prim = p
            break
    assert prim is not None, "could not find mul primitive in registry"

    saved = materialize_rules.pop(prim)
    try:
        with pytest.raises(NotImplementedError) as excinfo:
            sparsify(lambda x: 2.0 * x)(Identity(4, dtype=jnp.float64))
        msg = str(excinfo.value)
        assert "mul" in msg
        assert "ConstantDiagonal" in msg or "Identity" in msg or "forms" in msg
        assert "materialize_rules" in msg
    finally:
        materialize_rules[prim] = saved


# ----------------------- unsupported shape errors ------------------------


def test_multi_output_linear_fn_rejected():
    """Our walker seeds a single invar; multi-output linear fns are rejected."""
    def lin(x):
        return x, 2.0 * x

    with pytest.raises(NotImplementedError, match="multi-output"):
        sparsify(lin)(Identity(4, dtype=jnp.float64))


def test_non_1d_input_rejected():
    """The walker currently requires a 1D input invar."""
    def lin(x):
        return x.flatten()

    with pytest.raises(NotImplementedError, match="non-1D"):
        # Seed with a LinOp whose primal_aval is 2D — synthesize via a
        # BEllpack of the right shape? Easiest: use a 1D seed but pass a
        # linear_fn that trace-expects 2D. Do that via a wrapper.
        class Fake:
            def primal_aval(self):
                import jax
                return jax.core.ShapedArray((3, 3), jnp.float64)
        sparsify(lin)(Fake())


# ----------------------- shape-witness correctness -----------------------


def test_add_rule_absorbs_diagonal_into_ellpack():
    """Diagonal + BEllpack at matching square shape should stay BEllpack
    (promoted diagonals become an extra band), not fall through to BCOO."""
    from lineaxpr import BEllpack
    from lineaxpr.materialize import _add_rule

    D = Diagonal(jnp.asarray([1.0, 2.0, 3.0]))
    E = BEllpack(0, 3, (np.array([1, 0, 2]),), jnp.asarray([5.0, 6.0, 7.0]), 3, 3)
    result = _add_rule([D, E], [True, True], n=3)
    assert isinstance(result, BEllpack), f"expected BEllpack, got {type(result)}"
    assert result.k == 2, f"expected k=2 (Diagonal + 1-band BEllpack), got k={result.k}"
    # Dense-equivalence check.
    expected = np.diag([1.0, 2.0, 3.0]) + np.asarray(E.todense())
    np.testing.assert_allclose(np.asarray(result.todense()), expected)


def test_add_rule_absorbs_constant_diagonal_into_ellpack():
    """ConstantDiagonal + Diagonal + BEllpack → BEllpack with CD and D bands
    merged (both carry `cols=arange(3)`) and E's two distinct bands preserved.
    The partial-match band dedup in `_add_rule` is a strict generalisation of
    the `same_cols` fast path — same dense output, fewer stored bands."""
    from lineaxpr import BEllpack
    from lineaxpr.materialize import _add_rule

    CD = ConstantDiagonal(3, 0.5)
    D = Diagonal(jnp.asarray([1.0, 2.0, 3.0]))
    E = BEllpack(0, 3, (np.array([1, 0, 2]), np.array([2, 2, 0])),
                jnp.asarray([[5.0, 50.0], [6.0, 60.0], [7.0, 70.0]]), 3, 3)
    result = _add_rule([CD, D, E], [True, True, True], n=3)
    assert isinstance(result, BEllpack)
    # Dedup: CD and D both have cols=arange(3) → merge into 1 band with
    # values [1.5, 2.5, 3.5]. E's two cols are distinct → stay as 2 bands.
    assert result.k == 3
    expected = 0.5 * np.eye(3) + np.diag([1.0, 2.0, 3.0]) + np.asarray(E.todense())
    np.testing.assert_allclose(np.asarray(result.todense()), expected)


def test_seed_dtype_flows_to_trace():
    """The trace placeholder should use the seed's aval dtype."""
    captured = {}

    def lin(x):
        captured["dtype"] = x.dtype
        return x

    sparsify(lin)(Identity(4, dtype=jnp.float32))
    assert captured["dtype"] == jnp.float32
