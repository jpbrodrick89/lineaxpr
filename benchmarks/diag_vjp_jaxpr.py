"""Compare jaxpr of the vjp for scatter / jnp.diag / v*eye / where(eye, v, 0).

Motivation: benchmarks/diag_benchmark.py finds the four implementations
have very different VJP runtimes, and the ranking flips between CPU and
GPU. Reading the jaxprs explains where the work goes.

Run:
    uv run python benchmarks/diag_vjp_jaxpr.py
"""

from __future__ import annotations

from collections import Counter

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

N = 100


def scatter_impl(v):
    """zeros.at[i, i].set(v) — scatter pattern."""
    n = v.shape[0]
    idx = jnp.arange(n)
    return jnp.zeros((n, n), v.dtype).at[idx, idx].set(v)


def v_times_eye_impl(v):
    """v[:, None] * jnp.eye(n) — arithmetic mask."""
    n = v.shape[0]
    return v[:, None] * jnp.eye(n, dtype=v.dtype)


def jnp_diag_impl(v):
    """jnp.diag(v) — JAX's first-class primitive (pad+slice internally)."""
    return jnp.diag(v)


def where_eye_impl(v):
    """jnp.where(eye_bool, v, 0) — mask-and-select pattern."""
    n = v.shape[0]
    return jnp.where(jnp.eye(n, dtype=jnp.bool_), v, jnp.zeros((), v.dtype))


def fn_for_ad(diag_impl, v):
    """sum along axis=0 of the diagonal matrix — matches diag_benchmark's
    `build+reduce` / jvp / vjp scenarios."""
    return diag_impl(v).sum(axis=0)


def vjp_fn(diag_impl, v, cotangent):
    _, vjp = jax.vjp(lambda y: fn_for_ad(diag_impl, y), v)
    return vjp(cotangent)[0]


def _count_recursive(jaxpr_in, counts: Counter):
    """Count primitives across the whole jaxpr including jit / call bodies."""
    for eqn in jaxpr_in.eqns:
        counts[str(eqn.primitive)] += 1
        for p_name in ("jaxpr", "fun_jaxpr", "call_jaxpr"):
            sub = eqn.params.get(p_name)
            if sub is not None:
                inner = sub.jaxpr if hasattr(sub, "jaxpr") else sub
                _count_recursive(inner, counts)


def main():
    v = jnp.ones(N)
    ct = jnp.ones(N)
    for name, impl in [
        ("scatter",        scatter_impl),
        ("jnp.diag",       jnp_diag_impl),
        ("v*eye",          v_times_eye_impl),
        ("where(eye,v,0)", where_eye_impl),
    ]:
        jaxpr = jax.make_jaxpr(lambda v, ct, impl=impl: vjp_fn(impl, v, ct))(v, ct)
        print(f"\n========== vjp jaxpr — {name} — n={N} ==========")
        print(jaxpr)
        flat = Counter()
        _count_recursive(jaxpr.jaxpr, flat)
        top = len(jaxpr.jaxpr.eqns)
        total = sum(flat.values())
        print(f"  {top} top-level eqns, {total} total eqns (incl. jit/call bodies):")
        for p, c in flat.most_common():
            print(f"    {p:25s} {c}")


if __name__ == "__main__":
    main()
