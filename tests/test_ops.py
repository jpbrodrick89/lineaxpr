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

from lineaxpr import ConstantDiagonal, Diagonal, BEllpack, Identity


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
    out = Identity(4).todense()
    np.testing.assert_array_equal(np.asarray(out), np.eye(4))


def test_constant_diagonal_to_dense_scaled():
    out = ConstantDiagonal(3, value=2.5).todense()
    np.testing.assert_array_equal(np.asarray(out), 2.5 * np.eye(3))


def test_constant_diagonal_to_bcoo_roundtrip():
    cd = ConstantDiagonal(4, value=3.0)
    b = cd.to_bcoo()
    assert isinstance(b, sparse.BCOO)
    assert b.shape == (4, 4)
    np.testing.assert_array_equal(np.asarray(b.todense()), cd.todense())


def test_constant_diagonal_negate():
    cd = ConstantDiagonal(3, 2.0).negate()
    np.testing.assert_array_equal(np.asarray(cd.todense()), -2.0 * np.eye(3))


def test_constant_diagonal_scale_scalar():
    cd = ConstantDiagonal(3, 2.0).scale_scalar(jnp.asarray(4.0))
    np.testing.assert_array_equal(np.asarray(cd.todense()), 8.0 * np.eye(3))


def test_constant_diagonal_scale_per_out_row():
    v = jnp.asarray([1.0, 2.0, 3.0])
    result = ConstantDiagonal(3, value=2.0).scale_per_out_row(v)
    assert isinstance(result, Diagonal)
    expected = np.diag([2.0, 4.0, 6.0])
    np.testing.assert_array_equal(np.asarray(result.todense()), expected)


def test_constant_diagonal_primal_aval():
    cd = ConstantDiagonal(7, value=jnp.asarray(1.0, dtype=jnp.float64))
    aval = cd.primal_aval()
    assert isinstance(aval, core.ShapedArray)
    assert aval.shape == (7,)
    assert aval.dtype == jnp.float64


# ---------------------------- Diagonal -----------------------------------


def test_diagonal_to_dense():
    values = jnp.asarray([1.0, 2.0, 3.0])
    out = Diagonal(values).todense()
    np.testing.assert_array_equal(np.asarray(out), np.diag([1.0, 2.0, 3.0]))


def test_diagonal_to_bcoo_roundtrip():
    values = jnp.asarray([1.0, 2.0, 3.0])
    d = Diagonal(values)
    b = d.to_bcoo()
    assert isinstance(b, sparse.BCOO)
    np.testing.assert_array_equal(np.asarray(b.todense()), d.todense())


def test_diagonal_negate():
    d = Diagonal(jnp.asarray([1.0, -2.0, 3.0])).negate()
    np.testing.assert_array_equal(
        np.asarray(d.todense()), np.diag([-1.0, 2.0, -3.0])
    )


def test_diagonal_scale_scalar():
    d = Diagonal(jnp.asarray([1.0, 2.0])).scale_scalar(jnp.asarray(3.0))
    np.testing.assert_array_equal(np.asarray(d.todense()), np.diag([3.0, 6.0]))


def test_diagonal_scale_per_out_row():
    d = Diagonal(jnp.asarray([1.0, 2.0, 3.0])).scale_per_out_row(
        jnp.asarray([2.0, 2.0, 2.0])
    )
    np.testing.assert_array_equal(np.asarray(d.todense()), np.diag([2.0, 4.0, 6.0]))


def test_diagonal_primal_aval():
    d = Diagonal(jnp.asarray([1.0, 2.0], dtype=jnp.float32))
    aval = d.primal_aval()
    assert aval.shape == (2,)
    assert aval.dtype == jnp.float32


# ---------------------------- BEllpack ------------------------------------


def _simple_ellpack():
    """3×4 BEllpack, two bands, rows [0, 3).

    M[0,1]=5, M[0,2]=50; M[1,0]=6, M[1,3]=60; M[2,3]=7, M[2,2]=70.
    """
    return BEllpack(
        start_row=0,
        end_row=3,
        in_cols=(np.array([1, 0, 3]), np.array([2, 3, 2])),
        values=(jnp.asarray([5.0, 6.0, 7.0]),
                jnp.asarray([50.0, 60.0, 70.0])),
        out_size=3,
        in_size=4,
    )


def _ellpack_expected_dense():
    m = np.zeros((3, 4))
    m[0, 1] = 5.0; m[0, 2] = 50.0
    m[1, 0] = 6.0; m[1, 3] = 60.0
    m[2, 3] = 7.0; m[2, 2] = 70.0
    return m


def test_ellpack_basic_properties():
    e = _simple_ellpack()
    assert e.shape == (3, 4)
    assert e.nrows == 3
    assert e.k == 2
    assert e.nse == 6
    assert e.n == 4


def test_ellpack_to_dense():
    e = _simple_ellpack()
    np.testing.assert_array_equal(np.asarray(e.todense()), _ellpack_expected_dense())


def test_ellpack_to_bcoo_roundtrip():
    e = _simple_ellpack()
    b = e.to_bcoo()
    assert isinstance(b, sparse.BCOO)
    assert b.shape == (3, 4)
    np.testing.assert_array_equal(np.asarray(b.todense()), _ellpack_expected_dense())


def test_ellpack_negate():
    e = _simple_ellpack().negate()
    np.testing.assert_array_equal(
        np.asarray(e.todense()), -_ellpack_expected_dense()
    )


def test_ellpack_scale_scalar():
    e = _simple_ellpack().scale_scalar(jnp.asarray(2.0))
    np.testing.assert_array_equal(
        np.asarray(e.todense()), 2.0 * _ellpack_expected_dense()
    )


def test_ellpack_scale_per_out_row_nrows_length():
    e = _simple_ellpack()
    scaled = e.scale_per_out_row(jnp.asarray([10.0, 100.0, 1000.0]))
    expected = _ellpack_expected_dense() * np.array([[10.0], [100.0], [1000.0]])
    np.testing.assert_array_equal(np.asarray(scaled.todense()), expected)


def test_ellpack_scale_per_out_row_out_size_length():
    # BEllpack with start_row > 0 so out_size > nrows.
    e = BEllpack(
        start_row=1, end_row=3,
        in_cols=(np.array([0, 2]),),
        values=(jnp.asarray([5.0, 7.0]),),
        out_size=4, in_size=3,
    )
    # Scale vector is length out_size = 4; only entries at rows 1 and 2 hit.
    scaled = e.scale_per_out_row(jnp.asarray([1.0, 10.0, 100.0, 1000.0]))
    expected = np.zeros((4, 3))
    expected[1, 0] = 50.0
    expected[2, 2] = 700.0
    np.testing.assert_array_equal(np.asarray(scaled.todense()), expected)


def test_ellpack_pad_rows_positive():
    e = _simple_ellpack()
    padded = e.pad_rows(1, 2)  # out_size 3+1+2=6, rows shift by +1
    assert padded.shape == (6, 4)
    expected = np.zeros((6, 4))
    src = _ellpack_expected_dense()
    expected[1:4, :] = src
    np.testing.assert_array_equal(np.asarray(padded.todense()), expected)


def test_ellpack_pad_rows_negative_truncates_top():
    e = _simple_ellpack()
    # before=-1: row 0 falls out; rows 1,2 shift to 0,1. new out_size = 2.
    padded = e.pad_rows(-1, 0)
    assert padded.shape == (2, 4)
    expected = np.zeros((2, 4))
    expected[0, 0] = 6.0; expected[0, 3] = 60.0
    expected[1, 3] = 7.0; expected[1, 2] = 70.0
    np.testing.assert_array_equal(np.asarray(padded.todense()), expected)


def test_ellpack_pad_rows_negative_truncates_bottom():
    e = _simple_ellpack()
    # after=-2: new out_size = 1; only row 0 survives.
    padded = e.pad_rows(0, -2)
    assert padded.shape == (1, 4)
    expected = np.zeros((1, 4))
    expected[0, 1] = 5.0; expected[0, 2] = 50.0
    np.testing.assert_array_equal(np.asarray(padded.todense()), expected)


def test_ellpack_minus_one_sentinel_masks_slot():
    e = BEllpack(
        start_row=0, end_row=3,
        in_cols=(np.array([0, 1, 2]), np.array([2, -1, 0])),
        values=(jnp.asarray([1.0, 2.0, 3.0]),
                jnp.asarray([10.0, 20.0, 30.0])),
        out_size=3, in_size=3,
    )
    expected = np.zeros((3, 3))
    expected[0, 0] = 1.0; expected[0, 2] = 10.0
    expected[1, 1] = 2.0  # sentinel drops the 20.0
    expected[2, 2] = 3.0; expected[2, 0] = 30.0
    np.testing.assert_array_equal(np.asarray(e.todense()), expected)
    np.testing.assert_array_equal(np.asarray(e.to_bcoo().todense()), expected)


def test_ellpack_slice_band_resolves():
    # in_cols = slice(1, 5) on a length-4 row range => cols [1,2,3,4].
    e = BEllpack(
        start_row=0, end_row=4,
        in_cols=(slice(1, 5),),
        values=(jnp.ones((4,)),),
        out_size=4, in_size=5,
    )
    expected = np.zeros((4, 5))
    for r in range(4):
        expected[r, r + 1] = 1.0
    np.testing.assert_array_equal(np.asarray(e.todense()), expected)
    np.testing.assert_array_equal(np.asarray(e.to_bcoo().todense()), expected)


def test_ellpack_intra_row_duplicate_cols_sum():
    # Two bands both hitting col 0 on row 0 — densify must sum.
    e = BEllpack(
        start_row=0, end_row=1,
        in_cols=(np.array([0]), np.array([0])),
        values=(jnp.asarray([3.0]), jnp.asarray([4.0])),
        out_size=1, in_size=2,
    )
    expected = np.zeros((1, 2))
    expected[0, 0] = 7.0
    np.testing.assert_array_equal(np.asarray(e.todense()), expected)


def test_ellpack_primal_aval():
    e = _simple_ellpack()
    aval = e.primal_aval()
    assert isinstance(aval, core.ShapedArray)
    assert aval.shape == (4,)
    assert aval.dtype == e.dtype


# ---------------------------- invariants ---------------------------------


@pytest.mark.parametrize(
    "op_factory",
    [
        lambda: Identity(5),
        lambda: ConstantDiagonal(5, 2.5),
        lambda: Diagonal(jnp.arange(5, dtype=jnp.float64)),
        lambda: _simple_ellpack(),
    ],
    ids=["Identity", "ConstantDiagonal", "Diagonal", "BEllpack"],
)
def test_to_bcoo_dense_agreement(op_factory):
    op = op_factory()
    dense = np.asarray(op.todense())
    bcoo = op.to_bcoo()
    np.testing.assert_array_equal(np.asarray(bcoo.todense()), dense)


@pytest.mark.parametrize(
    "op_factory",
    [
        lambda: ConstantDiagonal(5, 2.5),
        lambda: Diagonal(jnp.arange(5, dtype=jnp.float64) + 1.0),
        lambda: _simple_ellpack(),
    ],
    ids=["ConstantDiagonal", "Diagonal", "BEllpack"],
)
def test_negate_then_scale_minus_one_agree(op_factory):
    op = op_factory()
    neg_direct = op.negate().todense()
    neg_via_scale = op.scale_scalar(jnp.asarray(-1.0)).todense()
    np.testing.assert_allclose(np.asarray(neg_direct), np.asarray(neg_via_scale))


# ---------------------------- batched BEllpack → BCOO ----------------------


def test_batched_ellpack_to_bcoo_k1_shared_cols():
    """Shared 1D in_cols, k=1. BCOO should have n_batch==1 and match
    the densified BEllpack entrywise."""
    B, O, N = 3, 4, 6
    cols = np.array([0, 2, 3, 1])  # shape (O,), shared across batches
    values = jnp.arange(B * O, dtype=jnp.float64).reshape(B, O) + 1.0
    ep = BEllpack(
        start_row=0, end_row=O,
        in_cols=(cols,), values=values,
        out_size=O, in_size=N,
        batch_shape=(B,),
    )
    bcoo = ep.to_bcoo()
    assert bcoo.shape == (B, O, N)
    np.testing.assert_allclose(
        np.asarray(bcoo.todense()), np.asarray(ep.todense())
    )


def test_batched_ellpack_to_bcoo_k1_per_batch_cols():
    """Per-batch in_cols (shape (*B, nrows)), k=1."""
    B, O, N = 2, 3, 5
    cols_per_batch = np.array([[0, 1, 2], [2, 3, 4]])  # (B, O)
    values = jnp.ones((B, O), dtype=jnp.float64) * jnp.asarray([[1.0], [10.0]])
    ep = BEllpack(
        start_row=0, end_row=O,
        in_cols=(cols_per_batch,), values=values,
        out_size=O, in_size=N,
        batch_shape=(B,),
    )
    bcoo = ep.to_bcoo()
    np.testing.assert_allclose(
        np.asarray(bcoo.todense()), np.asarray(ep.todense())
    )


def test_batched_ellpack_to_bcoo_k2_and_sentinels():
    """k=2 bands plus a -1 sentinel in one position — BCOO keeps the
    slot but masks value to 0."""
    B, O, N = 2, 4, 6
    cols_b0 = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    cols_b1 = np.array([[1, -1, 3, 4], [2, 3, -1, 5]])  # two sentinels
    values = jnp.arange(B * O * 2, dtype=jnp.float64).reshape(B, O, 2) + 1.0
    ep = BEllpack(
        start_row=0, end_row=O,
        in_cols=(cols_b0, cols_b1), values=values,
        out_size=O, in_size=N,
        batch_shape=(B,),
    )
    bcoo = ep.to_bcoo()
    np.testing.assert_allclose(
        np.asarray(bcoo.todense()), np.asarray(ep.todense())
    )


def test_batched_ellpack_to_bcoo_2d_batch():
    """n_batch=2 — batch_shape=(B1, B2)."""
    B1, B2, O, N = 2, 3, 4, 5
    cols = np.arange(O)  # 1D shared
    values = jnp.arange(B1 * B2 * O, dtype=jnp.float64).reshape(B1, B2, O)
    ep = BEllpack(
        start_row=0, end_row=O,
        in_cols=(cols,), values=values,
        out_size=O, in_size=N,
        batch_shape=(B1, B2),
    )
    bcoo = ep.to_bcoo()
    assert bcoo.shape == (B1, B2, O, N)
    np.testing.assert_allclose(
        np.asarray(bcoo.todense()), np.asarray(ep.todense())
    )
