"""Compile-check + correctness + timing for the monkeypatched sparsify path
on the curated CUTEst set. Compares to lineaxpr.bcoo_jacobian."""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp
from jax.experimental import sparse

from experiments.sparsify_monkeypatch import install
from lineaxpr import bcoo_jacobian

install()

CURATED = [
    ("sif2jax.cutest._bounded_minimisation.hs110", "HS110"),
    ("sif2jax.cutest._bounded_minimisation.hart6", "HART6"),
    ("sif2jax.cutest._unconstrained_minimisation.qing", "QING"),
    ("sif2jax.cutest._unconstrained_minimisation.chnrosnb", "CHNROSNB"),
    ("sif2jax.cutest._quadratic_problems.dual1", "DUAL1"),
    ("sif2jax.cutest._quadratic_problems.dual3", "DUAL3"),
    ("sif2jax.cutest._bounded_minimisation.levymont", "LEVYMONT"),
    ("sif2jax.cutest._unconstrained_minimisation.argtrigls", "ARGTRIGLS"),
    ("sif2jax.cutest._unconstrained_minimisation.genrose", "GENROSE"),
    ("sif2jax.cutest._unconstrained_minimisation.fletchcr", "FLETCHCR"),
    ("sif2jax.cutest._quadratic_problems.cmpc1", "CMPC1"),
    ("sif2jax.cutest._quadratic_problems.cmpc2", "CMPC2"),
    ("sif2jax.cutest._unconstrained_minimisation.dixmaanb", "DIXMAANB"),
    ("sif2jax.cutest._unconstrained_minimisation.dixmaane1", "DIXMAANE1"),
    ("sif2jax.cutest._unconstrained_minimisation.dixmaani1", "DIXMAANI1"),
    ("sif2jax.cutest._unconstrained_minimisation.edensch", "EDENSCH"),
]


def _load(modpath, cls):
    mod = __import__(modpath, fromlist=[cls])
    return getattr(mod, cls)()


def _block(out):
    for l in jax.tree_util.tree_leaves(out):
        jax.block_until_ready(l)
    return out


def _bench(fn, y0, warmup=2, reps=20):
    for _ in range(warmup):
        _block(fn(y0))
    t = []
    for _ in range(reps):
        start = time.perf_counter()
        _block(fn(y0))
        t.append(time.perf_counter() - start)
    return min(t)


def _sparsify_bcoo(problem):
    args_c = problem.args

    def f(y):
        return problem.objective(y, args_c)

    y0 = problem.y0
    n = y0.size

    @jax.jit
    def extract(y):
        _, lin = jax.linearize(jax.grad(f), y)
        I = sparse.eye(n, sparse_format="bcoo")
        return sparse.sparsify(jax.vmap(lin))(I)

    return extract, y0


def _lineaxpr_bcoo(problem):
    args_c = problem.args

    def f(y):
        return problem.objective(y, args_c)

    y0 = problem.y0

    @jax.jit
    def extract(y):
        _, lin = jax.linearize(jax.grad(f), y)
        return bcoo_jacobian(lin, y)

    return extract, y0


def main():
    problems = []
    for modpath, cls in CURATED:
        try:
            problems.append(_load(modpath, cls))
        except Exception as e:
            print(f"skip {cls}: {e}", file=sys.stderr)

    print(
        f"{'Problem':<12} {'n':>5} "
        f"{'sp_compile':>11} {'mp_compile':>11} "
        f"{'sp_us':>8} {'mp_us':>8} {'ratio':>6} {'match':>5}"
    )
    for p in problems:
        n = p.y0.size
        sp_fn, y0 = _sparsify_bcoo(p)
        mp_fn, _ = _lineaxpr_bcoo(p)

        # Compile both.
        sp_ok = True
        sp_err = ""
        try:
            sp_compiled = sp_fn.lower(y0).compile()
            sp_out = sp_compiled(y0)
            _block(sp_out)
        except Exception as e:
            sp_ok = False
            sp_err = f"{type(e).__name__}: {str(e)[:40]}"

        mp_ok = True
        try:
            mp_compiled = mp_fn.lower(y0).compile()
            mp_out = mp_compiled(y0)
            _block(mp_out)
        except Exception as e:
            mp_ok = False
            sp_err += f" | mp: {type(e).__name__}"

        if not sp_ok:
            print(f"{p.name:<12} {n:>5} {'FAIL':>11} {'OK' if mp_ok else 'FAIL':>11}  {sp_err}")
            continue
        if not mp_ok:
            print(f"{p.name:<12} {n:>5} OK FAIL")
            continue

        # Correctness.
        sp_dense = sp_out.todense() if hasattr(sp_out, "todense") else sp_out
        mp_dense = mp_out.todense() if hasattr(mp_out, "todense") else mp_out
        match = jnp.allclose(sp_dense, mp_dense, atol=1e-5)

        # Timing (best-of).
        sp_t = _bench(lambda y, c=sp_compiled: c(y), y0) * 1e6
        mp_t = _bench(lambda y, c=mp_compiled: c(y), y0) * 1e6

        print(
            f"{p.name:<12} {n:>5} {'OK':>11} {'OK':>11} "
            f"{sp_t:>8.1f} {mp_t:>8.1f} {sp_t / mp_t:>6.2f}x {str(bool(match)):>5}"
        )
        jax.clear_caches()


if __name__ == "__main__":
    main()
