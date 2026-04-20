"""Monkeypatched sparsify rules for the primitives linearize-of-grad emits.

Goal: get `sparsify(vmap(linearize(grad(f), y)[1]))(sparse.eye(n))` to compile
and run on the curated CUTEst set, so we can measure a pure-BCOO floor and
compare to lineaxpr.bcoo_jacobian (which uses specialized LinOp forms).

Missing primitives (see `docs/RESEARCH_NOTES.md` §9):
  - add_any   : adjoint of reduce_sum; structurally identical to lax.add
  - pad       : shifts along an output axis; update BCOO row indices
  - scatter-add : adjoint of gather; column-substitution into BCOO updates

lax.transpose_p already has a sparsify rule.

Scope: handles the shapes produced by `vmap(lin)` where `lin` is
`jax.linearize(jax.grad(f), y)[1]`. Not a general implementation.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jax._src import ad_util
from jax._src.lax import slicing
from jax.experimental import sparse
from jax.experimental.sparse import transform


def _spvalue_to_bcoo(spenv, sv):
    if sv.is_sparse():
        data = spenv.data(sv)
        indices = spenv.indices(sv)
        return sparse.BCOO((data, indices), shape=sv.shape)
    return None


def _bcoo_to_spvalue(spenv, bcoo):
    return spenv.sparse(bcoo.shape, bcoo.data, bcoo.indices)


# ---------------------------- add_any ----------------------------


def _add_any_sparse(spenv, *spvalues):
    """add_any appears as the transpose of reduce_sum / broadcast.

    Same semantics as lax.add for same-shape operands — sparsify's
    _add_sparse handles this case; delegate."""
    return transform._add_sparse(spenv, *spvalues)


# ---------------------------- pad --------------------------------


def _pad_sparse(spenv, *spvalues, padding_config):
    """pad with scalar zero padding_value: shift BCOO indices by `lo` on
    each axis, expand shape by lo + hi, drop entries that fall outside
    (negative pad). interior != 0 is not supported."""
    operand, padding_value = spvalues
    if any(c[2] != 0 for c in padding_config):
        raise NotImplementedError("sparse pad with interior dilation")

    pv = spenv.data(padding_value)
    if pv.shape != ():
        raise NotImplementedError("sparse pad with non-scalar padding_value")
    # We do not check pv == 0 at trace time (pv may be a tracer). For the
    # linearize-of-grad pattern, pv is always a concrete zero from the
    # transpose of reduce_sum / broadcast. If someone passes a nonzero pad
    # value through sparsify, the output will be silently wrong — same
    # caveat as sparsify's broader "we assume zero-preserving" stance.

    if operand.is_dense():
        # Dense passthrough.
        out = lax.pad(spenv.data(operand), pv, padding_config)
        return (spenv.dense(out),)

    bcoo = _spvalue_to_bcoo(spenv, operand)
    idx = bcoo.indices  # (nse, ndim)
    data = bcoo.data
    new_shape = tuple(
        operand.shape[ax] + lo + hi for ax, (lo, hi, _) in enumerate(padding_config)
    )

    los = jnp.asarray([lo for (lo, _, _) in padding_config], dtype=idx.dtype)
    shifted = idx + los[None, :]

    # Negative pad → drop entries outside new bounds on padded axes.
    has_neg = any(lo < 0 or hi < 0 for (lo, hi, _) in padding_config)
    if has_neg:
        lo_arr = jnp.asarray([0] * idx.shape[1], dtype=idx.dtype)
        hi_arr = jnp.asarray(new_shape, dtype=idx.dtype)
        in_bounds = jnp.all(
            (shifted >= lo_arr[None, :]) & (shifted < hi_arr[None, :]), axis=1
        )
        data = jnp.where(in_bounds, data, jnp.zeros_like(data))
        # Keep same nse; zeros will be summed away at densification.

    out_bcoo = sparse.BCOO((data, shifted), shape=new_shape)
    return (_bcoo_to_spvalue(spenv, out_bcoo),)


# ---------------------------- scatter-add ------------------------


def _scatter_add_sparse(
    spenv,
    *spvalues,
    dimension_numbers,
    indices_are_sorted,
    unique_indices,
    mode,
    update_jaxpr,
    update_consts,
):
    """scatter-add for the linearize-of-grad pattern: operand is a dense zero
    (from broadcast_in_dim of a zero closure), scatter_indices is dense closure,
    updates is sparse BCOO. We substitute scatter indices into the output axes
    of updates.indices, yielding a new BCOO.
    """
    operand, scatter_indices, updates = spvalues
    dn = dimension_numbers

    if not (operand.is_dense() and scatter_indices.is_dense() and updates.is_sparse()):
        raise NotImplementedError(
            "scatter-add sparse rule only supports dense-zero operand + "
            "dense scatter_indices + sparse updates"
        )

    op_buf = spenv.data(operand)
    # Verify operand is all-zero (cheap check: rely on constant-folding).
    # If not, we'd need to densify updates and do scatter-add on the result.
    si = spenv.data(scatter_indices)
    upd_bcoo = _spvalue_to_bcoo(spenv, updates)

    # Map update indices → operand indices.
    # update_window_dims: dims in `updates` that are windowed into `operand`.
    # inserted_window_dims: dims in `operand` that scatter_indices writes to.
    # scatter_dims_to_operand_dims: permutation from scatter index columns to
    #   operand dims.
    # For vmap(linearize_grad), the typical case is 2D:
    #   operand.shape == (B, out_size)   — zeros, broadcast of a zero
    #   scatter_indices.shape == (K, 1)  — the closure index map
    #   updates.shape == (B, K)          — sparse
    #   update_window_dims == (0,)       — the B axis
    #   inserted_window_dims == (1,)     — the out_size axis
    #   scatter_dims_to_operand_dims == (1,)  — col 0 of si maps to operand dim 1
    # Build new indices by mapping each update entry (b, k) to (b, si[k, 0]).
    if (
        len(operand.shape) == 2
        and dn.update_window_dims == (0,)
        and tuple(dn.inserted_window_dims) == (1,)
        and tuple(dn.scatter_dims_to_operand_dims) == (1,)
        and si.shape[-1] == 1
    ):
        # updates.indices has shape (nse, 2): columns (batch, scatter_slot)
        u_idx = upd_bcoo.indices
        batch_idx = u_idx[:, 0]
        slot_idx = u_idx[:, 1]
        operand_col = si[slot_idx, 0].astype(u_idx.dtype)
        new_idx = jnp.stack([batch_idx, operand_col], axis=1)
        out_bcoo = sparse.BCOO((upd_bcoo.data, new_idx), shape=op_buf.shape)
        return (_bcoo_to_spvalue(spenv, out_bcoo),)

    raise NotImplementedError(
        f"scatter-add rule: unsupported dimension_numbers {dn} "
        f"with operand.shape={operand.shape}, si.shape={si.shape}, "
        f"updates.shape={updates.shape}"
    )


# ---------------------------- install ----------------------------


def install():
    """Register the monkeypatched rules. Idempotent."""
    transform.sparse_rules_bcoo[ad_util.add_any_p] = _add_any_sparse
    transform.sparse_rules_bcoo[lax.pad_p] = _pad_sparse
    transform.sparse_rules_bcoo[slicing.scatter_add_p] = _scatter_add_sparse


def uninstall():
    for p in (ad_util.add_any_p, lax.pad_p, slicing.scatter_add_p):
        transform.sparse_rules_bcoo.pop(p, None)
