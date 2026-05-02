"""MJX direct-collocation residual as a monolithic JAX function.

make_collocation_residual(model_xml, T) returns:

    f : z -> defects
        z       : f32[T*(nq+nv+nu)]  flat [qpos,qvel,ctrl] per step
        defects : f32[(T-1)*(nq+nv)] stacked dynamics defects

Each defect is  x_{t+1} - mjx_step(x_t, u_t)  where x = [qpos, qvel].

The Python for-loop over T is unrolled at JAX trace-time, producing a
flat jaxpr with NO while_loop / scan / cond primitives — only the
primitives emitted by mjx.step itself for a contact-free model
(add, mul, dot_general, gather, scatter, select_n, …).  This is
essential: lineaxpr can walk a flat jaxpr but cannot enter while_loops.
"""
from __future__ import annotations

import mujoco
import mujoco.mjx as mjx
import jax.numpy as jnp


def make_collocation_residual(model_xml: str, T: int):
    """Build the collocation residual for horizon T.

    Parameters
    ----------
    model_xml : MuJoCo XML string (inline).
    T         : number of knot points (T-1 intervals).

    Returns
    -------
    f    : callable  z -> defects
    info : dict with keys nq, nv, nu, nx, stride, residual_dim, z_dim
    """
    model = mujoco.MjModel.from_xml_string(model_xml)
    mx    = mjx.put_model(model)
    d0    = mjx.make_data(mx)

    nq, nv, nu = model.nq, model.nv, model.nu
    nx     = nq + nv          # state dim
    stride = nx + nu           # variables per knot: [qpos, qvel, ctrl]

    def f(z: jnp.ndarray) -> jnp.ndarray:
        defects = []
        for t in range(T - 1):
            zt  = z[t       * stride : (t + 1) * stride]
            zt1 = z[(t + 1) * stride : (t + 2) * stride]

            qpos_t  = zt[:nq]
            qvel_t  = zt[nq:nx]
            ctrl_t  = zt[nx:]

            qpos_t1 = zt1[:nq]
            qvel_t1 = zt1[nq:nx]

            d_in  = d0.replace(qpos=qpos_t, qvel=qvel_t, ctrl=ctrl_t)
            d_out = mjx.step(mx, d_in)

            defects.append(jnp.concatenate([
                qpos_t1 - d_out.qpos,
                qvel_t1 - d_out.qvel,
            ]))

        return jnp.concatenate(defects)

    info = dict(
        nq=nq, nv=nv, nu=nu,
        nx=nx, stride=stride,
        residual_dim=(T - 1) * nx,
        z_dim=T * stride,
        T=T,
    )
    return f, info
