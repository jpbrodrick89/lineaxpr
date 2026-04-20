"""Unit tests for LinOp classes in lineaxpr._base.

Each LinOp method is tested in isolation with synthetic inputs, so
regressions in class logic are caught without any jaxpr-walk confounds.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import core
from jax.experimental import sparse

from lineaxpr import ConstantDiagonal, Diagonal, Identity, Pivoted


# ---------------------------- Identity / ConstantDiagonal -----------------


def test_identity_is_constant_diagonal_with_value_one():
    I = Identity(5)
    assert isinstance(I, ConstantDiagonal)
    assert I.n == 5
    assert float(jnp.asarray(I.value)) == 1.0


def test_identity_dtype_propagates():
    I = Identity(5, dtype=jnp.float32)
    aval = I.primal_aval()
    assert aval.shape == (5,)
    assert aval.dtype == jnp.float32


def test_constant_diagonal_to_dense_identity():
    out = Identity(4).to_dense()
    np.testing.assert_array_equal(np.asarray(out), np.eye(4))


def test_constant_diagonal_to_dense_scaled():
    out = ConstantDiagonal(3, value=2.5).to_dense()
    np.testing.assert_array_equal(np.asarray(out), 2.5 * np.eye(3))


def test_constant_diagonal_to_bcoo_roundtrip():
    cd = ConstantDiagonal(4, value=3.0)
    b = cd.to_bcoo()
    assert isinstance(b, sparse.BCOO)
    assert b.shape == (4, 4)
    np.testing.assert_array_equal(np.asarray(b.todense()), cd.to_dense())


def test_constant_diagonal_negate():
    cd = ConstantDiagonal(3, 2.0).negate()
    np.testing.assert_array_equal(np.asarray(cd.to_dense()), -2.0 * np.eye(3))


def test_constant_diagonal_scale_scalar():
    cd = ConstantDiagonal(3, 2.0).scale_scalar(jnp.asarray(4.0))
    np.testing.assert_array_equal(np.asarray(cd.to_dense()), 8.0 * np.eye(3))


def test_constant_diagonal_scale_per_out_row():
    v = jnp.asarray([1.0, 2.0, 3.0])
    result = ConstantDiagonal(3, value=2.0).scale_per_out_row(v)
    assert isinstance(result, Diagonal)
    expected = np.diag([2.0, 4.0, 6.0])
    np.testing.assert_array_equal(np.asarray(result.to_dense()), expected)


def test_constant_diagonal_primal_aval():
    cd = ConstantDiagonal(7, value=jnp.asarray(1.0, dtype=jnp.float64))
    aval = cd.primal_aval()
    assert isinstance(aval, core.ShapedArray)
    assert aval.shape == (7,)
    assert aval.dtype == jnp.float64


# ---------------------------- Diagonal -----------------------------------


def test_diagonal_to_dense():
    values = jnp.asarray([1.0, 2.0, 3.0])
    out = Diagonal(values).to_dense()
    np.testing.assert_array_equal(np.asarray(out), np.diag([1.0, 2.0, 3.0]))


def test_diagonal_to_bcoo_roundtrip():
    values = jnp.asarray([1.0, 2.0, 3.0])
    d = Diagonal(values)
    b = d.to_bcoo()
    assert isinstance(b, sparse.BCOO)
    np.testing.assert_array_equal(np.asarray(b.todense()), d.to_dense())


def test_diagonal_negate():
    d = Diagonal(jnp.asarray([1.0, -2.0, 3.0])).negate()
    np.testing.assert_array_equal(
        np.asarray(d.to_dense()), np.diag([-1.0, 2.0, -3.0])
    )


def test_diagonal_scale_scalar():
    d = Diagonal(jnp.asarray([1.0, 2.0])).scale_scalar(jnp.asarray(3.0))
    np.testing.assert_array_equal(np.asarray(d.to_dense()), np.diag([3.0, 6.0]))


def test_diagonal_scale_per_out_row():
    d = Diagonal(jnp.asarray([1.0, 2.0, 3.0])).scale_per_out_row(
        jnp.asarray([2.0, 2.0, 2.0])
    )
    np.testing.assert_array_equal(np.asarray(d.to_dense()), np.diag([2.0, 4.0, 6.0]))


def test_diagonal_primal_aval():
    d = Diagonal(jnp.asarray([1.0, 2.0], dtype=jnp.float32))
    aval = d.primal_aval()
    assert aval.shape == (2,)
    assert aval.dtype == jnp.float32


# ---------------------------- Pivoted ------------------------------------


def _simple_pivoted():
    """3×4 Pivoted with entries M[0, 1]=5, M[2, 3]=7."""
    return Pivoted(
        out_rows=np.array([0, 2]),
        in_cols=np.array([1, 3]),
        values=jnp.asarray([5.0, 7.0]),
        out_size=3,
        in_size=4,
    )


def test_pivoted_to_dense():
    p = _simple_pivoted()
    expected = np.zeros((3, 4))
    expected[0, 1] = 5.0
    expected[2, 3] = 7.0
    np.testing.assert_array_equal(np.asarray(p.to_dense()), expected)


def test_pivoted_to_bcoo_roundtrip():
    p = _simple_pivoted()
    b = p.to_bcoo()
    assert isinstance(b, sparse.BCOO)
    assert b.shape == (3, 4)
    np.testing.assert_array_equal(np.asarray(b.todense()), p.to_dense())


def test_pivoted_negate():
    p = _simple_pivoted().negate()
    expected = np.zeros((3, 4))
    expected[0, 1] = -5.0
    expected[2, 3] = -7.0
    np.testing.assert_array_equal(np.asarray(p.to_dense()), expected)


def test_pivoted_scale_scalar():
    p = _simple_pivoted().scale_scalar(jnp.asarray(2.0))
    expected = np.zeros((3, 4))
    expected[0, 1] = 10.0
    expected[2, 3] = 14.0
    np.testing.assert_array_equal(np.asarray(p.to_dense()), expected)


def test_pivoted_scale_per_out_row_fast_path():
    # nse == scale.shape[0] fires the "no gather" fast path.
    p = _simple_pivoted()
    assert p.nse == 2
    scaled = p.scale_per_out_row(jnp.asarray([10.0, 100.0]))
    # Entries scaled element-wise (nse-length path).
    expected = np.zeros((3, 4))
    expected[0, 1] = 50.0
    expected[2, 3] = 700.0
    np.testing.assert_array_equal(np.asarray(scaled.to_dense()), expected)


def test_pivoted_scale_per_out_row_gather_path():
    # scale.shape[0] == out_size takes the gather path: values *= scale[out_rows].
    p = _simple_pivoted()
    scaled = p.scale_per_out_row(jnp.asarray([10.0, 20.0, 100.0]))  # len 3 = out_size
    expected = np.zeros((3, 4))
    expected[0, 1] = 50.0      # 5 * scale[0]
    expected[2, 3] = 700.0     # 7 * scale[2]
    np.testing.assert_array_equal(np.asarray(scaled.to_dense()), expected)


def test_pivoted_pad_rows_positive():
    p = _simple_pivoted()           # shape (3, 4)
    padded = p.pad_rows(1, 2)       # shape (6, 4); rows shifted by +1
    assert padded.shape == (6, 4)
    expected = np.zeros((6, 4))
    expected[1, 1] = 5.0
    expected[3, 3] = 7.0
    np.testing.assert_array_equal(np.asarray(padded.to_dense()), expected)


def test_pivoted_pad_rows_negative_truncates():
    # Truncate top row with before=-1: entry at row 0 falls out; entry at row 2 stays at 1.
    p = _simple_pivoted()                    # out_rows [0, 2]
    padded = p.pad_rows(-1, 0)               # out_size 2; new rows [-1 (drop), 1 (keep)]
    assert padded.shape == (2, 4)
    expected = np.zeros((2, 4))
    expected[1, 3] = 7.0
    np.testing.assert_array_equal(np.asarray(padded.to_dense()), expected)


def test_pivoted_primal_aval():
    p = _simple_pivoted()
    aval = p.primal_aval()
    assert aval.shape == (4,)
    assert aval.dtype == p.values.dtype


def test_pivoted_nse_reports_entries():
    p = _simple_pivoted()
    assert p.nse == 2


# ---------------------------- invariants ---------------------------------


@pytest.mark.parametrize(
    "op_factory",
    [
        lambda: Identity(5),
        lambda: ConstantDiagonal(5, 2.5),
        lambda: Diagonal(jnp.arange(5, dtype=jnp.float64)),
        lambda: Pivoted(np.arange(3), np.arange(3), jnp.ones(3), 5, 5),
    ],
    ids=["Identity", "ConstantDiagonal", "Diagonal", "Pivoted"],
)
def test_to_bcoo_dense_agreement(op_factory):
    op = op_factory()
    dense = np.asarray(op.to_dense())
    bcoo = op.to_bcoo()
    np.testing.assert_array_equal(np.asarray(bcoo.todense()), dense)


@pytest.mark.parametrize(
    "op_factory",
    [
        lambda: ConstantDiagonal(5, 2.5),
        lambda: Diagonal(jnp.arange(5, dtype=jnp.float64) + 1.0),
        lambda: Pivoted(np.arange(3), np.arange(3), jnp.ones(3), 5, 5),
    ],
    ids=["ConstantDiagonal", "Diagonal", "Pivoted"],
)
def test_negate_then_scale_minus_one_agree(op_factory):
    op = op_factory()
    neg_direct = op.negate().to_dense()
    neg_via_scale = op.scale_scalar(jnp.asarray(-1.0)).to_dense()
    np.testing.assert_allclose(np.asarray(neg_direct), np.asarray(neg_via_scale))
