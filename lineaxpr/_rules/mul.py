"""mul primitive rule."""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from .._linops import (
    BEllpack,
    ColArr,
    LinOpProtocol,
    scale_per_out_row,
    scale_scalar,
)


def _mul_rule(invals, traced, n, **params):
    del params
    x, y = invals
    tx, ty = traced
    if not tx and not ty:
        return None
    if not tx:
        scale, traced_op = x, y
    elif not ty:
        scale, traced_op = y, x
    else:
        raise NotImplementedError("mul of two traced operands — not linear")

    # Primal axes (= traced_op shape minus V). For transposed=True
    # BE, V is at axis 0 → primal = shape[1:]; for transposed=False
    # (default for non-BE LinOps and canonical BEs) V is at axis -1 →
    # primal = shape[:-1].
    is_transposed_be = (
        isinstance(traced_op, BEllpack) and traced_op.transposed
    )
    if is_transposed_be:
        traced_var_shape = traced_op.shape[1:]
    else:
        traced_var_shape = traced_op.shape[:-1]

    # vmap inserts broadcast_in_dim on scalar closures, giving them shape
    # (1, k) instead of (k,). Squeeze leading (1,) dims so the existing
    # scale_ok / scale_per_out_row paths see the expected shape.
    if hasattr(scale, "shape"):
        target_ndim = len(traced_var_shape)
        while hasattr(scale, "shape") and len(scale.shape) > target_ndim and scale.shape[0] == 1:
            scale = scale[0]

    scalar_like = not hasattr(scale, "shape") or scale.shape in ((), (1,))
    if scalar_like:
        s = jnp.asarray(scale).reshape(())
        if isinstance(traced_op, LinOpProtocol):
            return scale_scalar(traced_op, s)
        return s * traced_op
    # scale_per_out_row assumes scale has shape that broadcasts cleanly
    # against the op's var_shape (batch_shape + (out_size,)). If scale has
    # extra dims (jaxpr outer-product-like broadcasts), fall back to dense.
    scale_ok = (
        hasattr(scale, "shape")
        and len(scale.shape) <= len(traced_var_shape)
        and all(
            s in (1, t)
            for s, t in zip(scale.shape[::-1], traced_var_shape[::-1])
        )
    )
    if scale_ok and isinstance(traced_op, LinOpProtocol):
        return scale_per_out_row(traced_op, scale)
    # Batch-expand path: scale broadcasts same-ndim as traced_var_shape
    # but expands one or more size-1 batch axes of the BE (dims where BE
    # has 1 and scale has K > 1). Structurally: new BEllpack with
    # enlarged batch_shape, values broadcast-mul'd. Cols pattern
    # preserved. Motivation: DMN15102LS's `BE(batch=(1,), out=33, k=2) /
    # closure(4643, 33)` should stay structural (nse per batch preserved
    # by mul-sparsity-preservation).
    if (isinstance(traced_op, BEllpack)
            and hasattr(scale, "shape")
            and len(scale.shape) == len(traced_var_shape)
            and all(s == t or t == 1 for s, t in
                    zip(scale.shape, traced_var_shape))
            and scale.shape[-1] == traced_var_shape[-1]):
        new_batch = scale.shape[:-1]
        scale_arr = jnp.asarray(scale)
        if traced_op.k == 1:
            new_values = scale_arr * traced_op.data
        else:
            new_values = scale_arr[..., None] * traced_op.data
        new_in_cols: list[ColArr] = []
        can_emit = True
        for c in traced_op.in_cols:
            if isinstance(c, np.ndarray):
                if c.ndim == 1:
                    new_in_cols.append(c)
                elif c.shape[:len(traced_op.batch_shape)] == traced_op.batch_shape \
                        and all(t == 1 for t in traced_op.batch_shape):
                    new_in_cols.append(
                        np.broadcast_to(c, new_batch + c.shape[-1:]).copy()
                    )
                else:
                    can_emit = False
                    break
            else:
                can_emit = False
                break
        if can_emit:
            return BEllpack(
                start_row=traced_op.start_row, end_row=traced_op.end_row,
                in_cols=tuple(new_in_cols), data=new_values,
                out_size=traced_op.out_size, in_size=traced_op.in_size,
                batch_shape=new_batch,
            )
    # Out-size-broadcast path: scale expands a size-1 out axis to
    # `scale.shape[-1]`. Triggered by the NONMSQRT / EIGENALS-class
    # pattern where an aval-(B, 1) BEllpack (from `bid`-trailing-
    # singleton + slice / reduce_sum chain) multiplies by a (B, S)
    # closure — the primal broadcasts to (B, S) and each of the S new
    # rows is a scaled copy of the traced op's single row. Stays
    # structural: new BEllpack `batch_shape=batch, out_size=S`, k
    # unchanged; for k=1 the value mul is a direct scale*values with
    # no axis insertion; for k>=2 we insert `[..., None]` to broadcast
    # over the band axis; cols broadcast statically across new out.
    #
    # Previously gated on `traced.k >= 2` after an EIGENALS regression
    # measurement (~30ms, 97→126ms). That gate was set before
    # 0c/0d/0l — downstream rules densified a k=1 broadcast-expand BE
    # almost immediately via `add_any(..., dense_closure)`. With 0d's
    # structural select_n / gather paths now consuming the output
    # without densifying, the earlier regression no longer applies.
    if (isinstance(traced_op, BEllpack)
            and traced_op.k >= 1
            and traced_op.out_size == 1
            and traced_op.start_row == 0 and traced_op.end_row == 1
            and hasattr(scale, "shape")
            and len(scale.shape) == traced_op.n_batch + 1
            and all(s in (1, t)
                    for s, t in zip(scale.shape[:-1], traced_op.batch_shape))
            and int(scale.shape[-1]) >= 1):
        new_out = int(scale.shape[-1])
        scale_arr = jnp.asarray(scale)
        if traced_op.k == 1:
            # traced values (*batch, 1). scale (*batch, new_out).
            # Result (*batch, new_out).
            new_values = scale_arr * traced_op.data
        else:
            # traced values (*batch, 1, k). Insert new_out axis then mul.
            new_values = scale_arr[..., None] * traced_op.data
        new_in_cols: list[ColArr] = []
        can_emit2 = True
        for c in traced_op.in_cols:
            if isinstance(c, np.ndarray):
                if c.ndim == traced_op.n_batch + 1:  # (*batch, 1) per-batch
                    new_in_cols.append(
                        np.broadcast_to(c, traced_op.batch_shape + (new_out,)).copy()
                    )
                else:
                    new_in_cols.append(c)
            else:
                # Traced cols — would need jnp broadcast; keep simple by
                # falling through to dense. Rare.
                can_emit2 = False
                break
        if can_emit2:
            return BEllpack(
                start_row=0, end_row=new_out,
                in_cols=tuple(new_in_cols), data=new_values,
                out_size=new_out, in_size=traced_op.in_size,
                batch_shape=traced_op.batch_shape,
            )
    # Dense fallback: just trust natural broadcasting. Under vmap, JAX
    # already wraps scale appropriately so `scale * dense` does the
    # right thing — manual axis-insertion logic was bogus and silently
    # dropped axes when scale was a (1, k) closure broadcast row.
    dense = traced_op.todense() if isinstance(traced_op, LinOpProtocol) else traced_op
    return jnp.asarray(scale) * dense
