"""Conversion-path invariants for LinOps.

Pins the contracts that layout-aware rules rely on when promoting one
LinOp form to another:

- **Shape preservation**: `op.shape == converted.shape`.
- **Roundtrip equivalence**: `op.todense() == converted.todense()`.
- **Flag preservation**: BEllpack.to_bcoo() respects the `transposed`
  flag — it produces the LOGICAL view, NOT the canonical-data view.
- **Helper unification**: `_ellpack_to_bcoo` and `BEllpack.to_bcoo`
  agree (the no-trap contract from the `_ellpack_to_bcoo` flag-aware
  rewrite). A caller wanting the canonical-data view must explicitly
  flip the flag first via `replace_slots(transposed=False)` and then
  call the helper — the function call is unambiguous either way.

These were added after a `_scatter_add_rule` regression where the BE
branch silently used the canonical-data view (via flag-flip +
`_ellpack_to_bcoo`) while a parallel BCOO branch used the logical
view, producing transpose-shaped results that only surfaced in
end-to-end Hessian extraction. The tests below would have caught
this at the unit level.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lineaxpr import BEllpack, ConstantDiagonal, Diagonal
from lineaxpr._linops.diagonal import _diag_to_bcoo
from lineaxpr._linops.ellpack import (
    _ellpack_to_bcoo,
    _ellpack_to_bcoo_batched,
    replace_slots,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _todense(op):
    """Densify a LinOp / bare BCOO / ndarray uniformly."""
    if hasattr(op, "todense"):
        return np.asarray(op.todense())
    return np.asarray(op)


def _be_unbatched(*, out=4, in_=5, k=1, transposed=False, start_row=0,
                  end_row=None):
    """Asymmetric BEllpack (any row/col confusion will show up in todense).

    k=1: row r → col r mod in_ with values [1..nrows].
    k=2: extra band at col (r+1) mod in_ with values [101..].
    """
    if end_row is None:
        end_row = out
    nrows = end_row - start_row
    if k == 1:
        cols = (np.array([r % in_ for r in range(nrows)], dtype=np.int64),)
        data = jnp.asarray(np.arange(1, nrows + 1, dtype=np.float64))
    else:
        cols0 = np.array([r % in_ for r in range(nrows)], dtype=np.int64)
        cols1 = np.array([(r + 1) % in_ for r in range(nrows)], dtype=np.int64)
        cols = (cols0, cols1)
        data = jnp.asarray(np.stack([
            np.arange(1, nrows + 1, dtype=np.float64),
            np.arange(101, nrows + 101, dtype=np.float64),
        ], axis=-1))
    return BEllpack(
        start_row=start_row, end_row=end_row, in_cols=cols, data=data,
        out_size=out, in_size=in_, transposed=transposed,
    )


def _be_batched_3d(*, b=2, out=3, in_=4, transposed=False):
    nrows = out
    cols = (np.broadcast_to(
        np.array([r % in_ for r in range(nrows)], dtype=np.int64),
        (b, nrows),
    ).copy(),)
    data = jnp.asarray(
        np.arange(1, b * nrows + 1, dtype=np.float64).reshape(b, nrows)
    )
    return BEllpack(
        start_row=0, end_row=nrows, in_cols=cols, data=data,
        out_size=out, in_size=in_, batch_shape=(b,),
        transposed=transposed,
    )


# ---------------------------------------------------------------------------
# Diagonal / ConstantDiagonal: todense + to_bcoo agree on shape and values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["cd", "diag"])
def test_diag_to_bcoo_shape_and_values(kind):
    n = 5
    if kind == "cd":
        op = ConstantDiagonal(n, 3.0)
    else:
        op = Diagonal(jnp.arange(1, n + 1, dtype=jnp.float64))
    bcoo = op.to_bcoo()
    assert bcoo.shape == op.shape, (bcoo.shape, op.shape)
    np.testing.assert_allclose(_todense(bcoo), _todense(op))


def test_diag_to_bcoo_helper_matches_method():
    """`_diag_to_bcoo(n, values)` and `Diagonal(values).to_bcoo()`
    produce the same matrix."""
    n = 5
    values = jnp.arange(1, n + 1, dtype=jnp.float64)
    via_helper = _diag_to_bcoo(n, values)
    via_method = Diagonal(values).to_bcoo()
    assert via_helper.shape == via_method.shape
    np.testing.assert_allclose(_todense(via_helper), _todense(via_method))


# ---------------------------------------------------------------------------
# BEllpack -> BCOO: shape, todense roundtrip, flag respected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("k", [1, 2])
def test_ellpack_to_bcoo_unbatched(k, transposed):
    be = _be_unbatched(out=4, in_=5, k=k, transposed=transposed)
    bcoo = be.to_bcoo()
    assert bcoo.shape == be.shape, (bcoo.shape, be.shape)
    np.testing.assert_allclose(_todense(bcoo), _todense(be))


@pytest.mark.parametrize("transposed", [
    False,
    pytest.param(True, marks=pytest.mark.xfail(
        reason="Batched T=True to_bcoo: `_bcoo_swap_last_two_sparse_axes` "
               "swaps axes (out, in) but the LOGICAL shape "
               "`(in, *batch, out)` puts in at axis 0, requiring a full "
               "rotate not just a last-two swap. Pre-existing limitation; "
               "no current rule produces a batched T=True BE that flows "
               "through `.to_bcoo`.",
        strict=True,
    )),
])
def test_ellpack_to_bcoo_3d_batched(transposed):
    be = _be_batched_3d(b=2, out=3, in_=4, transposed=transposed)
    bcoo = be.to_bcoo()
    assert bcoo.shape == be.shape, (bcoo.shape, be.shape)
    np.testing.assert_allclose(_todense(bcoo), _todense(be))


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("start_row,end_row", [(0, 4), (1, 4), (0, 3)])
def test_ellpack_to_bcoo_partial_rows(start_row, end_row, transposed):
    """Partial-row BEs (start_row > 0 or end_row < out_size) must
    still round-trip cleanly. Hit by SBRYBND's slice-then-add chain."""
    be = _be_unbatched(out=4, in_=5, k=1, transposed=transposed,
                      start_row=start_row, end_row=end_row)
    bcoo = be.to_bcoo()
    assert bcoo.shape == be.shape
    np.testing.assert_allclose(_todense(bcoo), _todense(be))


# ---------------------------------------------------------------------------
# `_ellpack_to_bcoo` and `BEllpack.to_bcoo` agree (no-trap unification).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("k", [1, 2])
def test_helper_and_method_agree_unbatched(k, transposed):
    be = _be_unbatched(out=4, in_=5, k=k, transposed=transposed)
    via_helper = _ellpack_to_bcoo(be)
    via_method = be.to_bcoo()
    assert via_helper.shape == via_method.shape
    np.testing.assert_allclose(_todense(via_helper), _todense(via_method))


@pytest.mark.parametrize("transposed", [False, True])
def test_helper_and_method_agree_batched(transposed):
    """`_ellpack_to_bcoo_batched` and `BEllpack.to_bcoo` produce the
    same BCOO for batched input — the no-trap unification holds even
    in the batched T=True case where both are pre-existingly buggy
    (they agree on the same wrong shape). Future fix should keep
    them in sync."""
    be = _be_batched_3d(b=2, out=3, in_=4, transposed=transposed)
    via_helper = _ellpack_to_bcoo_batched(be)
    via_method = be.to_bcoo()
    assert via_helper.shape == via_method.shape
    np.testing.assert_allclose(_todense(via_helper), _todense(via_method))


# ---------------------------------------------------------------------------
# Canonical-data-view escape hatch: explicit flag flip + helper.
#
# A caller that genuinely wants the canonical-data BCOO (indices in
# `(out_idx, in_idx)` layout, ignoring the operand's `transposed`
# flag) MUST flip the flag explicitly before calling the helper —
# this is the documented escape hatch. Verify it still works.
# ---------------------------------------------------------------------------


def test_canonical_data_view_via_explicit_flag_flip():
    be_t = _be_unbatched(out=4, in_=5, k=1, transposed=True)
    # Logical view (T=True) — what `to_bcoo()` produces.
    logical = be_t.to_bcoo()
    # Canonical-data view (T=False reinterpretation) — flag flip first.
    flipped = replace_slots(be_t, transposed=False)
    canonical_data = _ellpack_to_bcoo(flipped)

    # Canonical-data BCOO has shape `(out_size, in_size)`, NOT
    # `(in_size, out_size)`. Indices are at `(out_idx, in_idx)`.
    assert canonical_data.shape == (be_t.out_size, be_t.in_size)
    assert logical.shape == be_t.shape  # `(in_size, out_size)` for T=True

    # The two are TRANSPOSES of each other in dense form.
    np.testing.assert_allclose(
        _todense(canonical_data).T, _todense(logical),
    )


# ---------------------------------------------------------------------------
# Self-roundtrip: todense should be idempotent on already-dense input
# (sanity check for the helper).
# ---------------------------------------------------------------------------


def test_ndarray_todense_is_identity():
    x = jnp.asarray(np.arange(20, dtype=np.float64).reshape(4, 5))
    assert np.array_equal(_todense(x), np.asarray(x))


# ---------------------------------------------------------------------------
# BEllpack negate / scale_scalar: flag is preserved.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transposed", [False, True])
def test_ellpack_negate_preserves_flag(transposed):
    be = _be_unbatched(out=4, in_=5, k=1, transposed=transposed)
    neg = -be
    assert neg.transposed == be.transposed
    assert neg.shape == be.shape
    np.testing.assert_allclose(_todense(neg), -_todense(be))
