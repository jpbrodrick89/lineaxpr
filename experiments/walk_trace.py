"""Trace the lineaxpr walk step-by-step, printing the LinOp at each equation.

Useful for comparing the walk on two branches (e.g. main vs jbp/vmap-poc):

    # On each branch:
    uv run python -m experiments.walk_trace HADAMALS 2 > /tmp/trace_main.txt
    uv run python -m experiments.walk_trace HADAMALS 2 > /tmp/trace_vmap.txt
    diff /tmp/trace_main.txt /tmp/trace_vmap.txt

Usage:
    uv run python -m experiments.walk_trace [PROBLEM_CLASS [CTOR_ARG ...]]

    PROBLEM_CLASS  sif2jax.cutest class name, e.g. HADAMALS  (default: HADAMALS)
    CTOR_ARG       positional int args to the constructor,    (default: 2)
                   e.g. "2" for HADAMALS(n=2)

The walk is inferred from _walk_with_seed source: vmap-poc uses vmap(lin),
main uses lin directly.
"""
from __future__ import annotations

import sys
import inspect
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core
import sif2jax

from lineaxpr._linops import Identity, LinOpProtocol
from lineaxpr._rules.registry import materialize_rules

jax.config.update("jax_enable_x64", True)

# ── parse args ────────────────────────────────────────────────────────────────
args = sys.argv[1:]
prob_name = args[0] if args else "HADAMALS"
ctor_args = [int(a) for a in args[1:]] if len(args) > 1 else [2]

prob_cls = getattr(sif2jax.cutest, prob_name)
prob     = prob_cls(*ctor_args)
y        = prob.y0
n        = y.shape[0]

def f(z):
    return prob.objective(z, prob.args)

_, lin = jax.linearize(jax.grad(f), y)

# ── detect which walk strategy this branch uses ───────────────────────────────
src       = inspect.getsource(
    __import__("lineaxpr._transform", fromlist=["_walk_with_seed"])._walk_with_seed
)
VMAP_WALK = "jax.vmap" in src
branch_tag = "vmap-poc" if VMAP_WALK else "main"

print(f"# branch:    {branch_tag}")
print(f"# problem:   {prob_name}({', '.join(str(a) for a in ctor_args)})")
print(f"# n:         {n}")
print(f"# y0:        {y.tolist()}")
print(f"# VMAP_WALK: {VMAP_WALK}")

# ── build jaxpr the same way _walk_with_seed does ────────────────────────────
seed = Identity(n, dtype=y.dtype)
if VMAP_WALK:
    placeholder = jax.ShapeDtypeStruct(seed.shape, seed.dtype)     # (n, n)
    cj = jax.make_jaxpr(jax.vmap(lin))(placeholder)
else:
    placeholder = jax.ShapeDtypeStruct((seed.shape[-1],), seed.dtype)  # (n,)
    cj = jax.make_jaxpr(lin)(placeholder)

jaxpr = cj.jaxpr
print(f"# invar:     {jaxpr.invars[0].aval}")
print(f"# eqns:      {len(jaxpr.eqns)}")
print()

# ── helpers ───────────────────────────────────────────────────────────────────
def todense(v):
    if v is None:
        return None
    if isinstance(v, LinOpProtocol):
        return v.todense()
    return jnp.asarray(v)


# ── walk ──────────────────────────────────────────────────────────────────────
env: dict = {v: (False, c) for v, c in zip(jaxpr.constvars, cj.consts)}
env[jaxpr.invars[0]] = (True, seed)

for i, eqn in enumerate(jaxpr.eqns):
    entries = [
        (False, v.val) if isinstance(v, core.Literal) else env.get(v, (False, None))
        for v in eqn.invars
    ]
    invals = [e[1] for e in entries]
    traced = [e[0] for e in entries]

    if not any(traced):
        # constant propagation
        if any(v is None for v in invals):
            continue
        try:
            concrete = eqn.primitive.bind(*invals, **eqn.params)
        except Exception:
            continue
        # strip vmap batch dim from broadcast_in_dim constants
        if VMAP_WALK and eqn.primitive is lax.broadcast_in_dim_p:
            if (hasattr(concrete, "ndim") and concrete.ndim > 0
                    and int(concrete.shape[0]) == n
                    and 0 not in eqn.params.get("broadcast_dimensions", ())):
                concrete = concrete[0]
        if eqn.primitive.multiple_results:
            for v, o in zip(eqn.outvars, concrete):
                env[v] = (False, o)
        else:
            env[eqn.outvars[0]] = (False, concrete)
        continue

    rule = materialize_rules.get(eqn.primitive)
    if rule is None:
        print(f"eqn {i:3d}  {eqn.primitive.name}  NO RULE")
        for v in eqn.outvars:
            env[v] = (True, None)
        continue

    vmap_avals = tuple(
        v.aval.shape if hasattr(v, "aval") else None for v in eqn.invars
    )
    try:
        out = rule(invals, traced, n, _vmap_avals=vmap_avals, **eqn.params)
    except Exception as e:
        print(f"eqn {i:3d}  {eqn.primitive.name}  ERROR: {e}")
        for v in eqn.outvars:
            env[v] = (True, None)
        continue

    if eqn.primitive.multiple_results:
        for v, o in zip(eqn.outvars, out):
            env[v] = (True, o)
        walk_val = out[0] if out else None
    else:
        env[eqn.outvars[0]] = (True, out)
        walk_val = out

    d = todense(walk_val)
    jaxpr_shape = str(eqn.outvars[0].aval)
    walk_shape  = str(d.shape) if d is not None else "None"

    print(f"eqn {i:3d}  {eqn.primitive.name:22s}  jaxpr={jaxpr_shape:22s}  walk={walk_shape}")
    if d is not None:
        for row in jnp.round(d.reshape(-1, n), 4).tolist():
            print(f"          {row}")

# ── final result ──────────────────────────────────────────────────────────────
fv = env.get(jaxpr.outvars[0])
print()
if fv and fv[1] is not None:
    fd = todense(fv[1])
    if fd is not None and fd.size == n * n:
        h = fd.reshape(n, n)
        print(f"# FINAL Hessian ({n}×{n}):")
        for row in jnp.round(h, 4).tolist():
            print(f"  {row}")
        ref = jax.hessian(f)(y)
        rel = float(jnp.linalg.norm(h - ref) / jnp.linalg.norm(ref))
        print(f"# rel_err vs jax.hessian: {rel:.6f}")
