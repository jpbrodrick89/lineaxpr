"""Bench materialize on 1500 < n <= 3000 problems.

CRITICAL: linearization point `y` is passed as a jit INPUT (not closure),
so quadratic-Hessians that are y-independent still trace real dataflow.
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
import os
import sys


import jax
import pytest
import sif2jax  # noqa: E402
from lineaxpr import materialize  # noqa: E402
from sif2jax._problem import (  # noqa: E402
    AbstractBoundedMinimisation,
    AbstractConstrainedQuadraticProblem,
    AbstractUnconstrainedMinimisation,
)


MIN_N = 2500
MAX_N = 5100
_ALLOW_PREFIXES = (
    "DIXMAAN",   # n=3000, y-dependent
    "CHAINWOO",  # n=4000
    "WOODS",     # n=4000
    "EIGENALS",  # n=2550
    "EIGENBLS",
    "EIGENCLS",
    "ARWHEAD",   # n=5000, simple
    "DQRTIC",    # n=5000
    "CRAGGLVY",  # n=5000
    "BDEXP",     # n=5000
    "EDENSCH",   # n=2000, just above min
    "FLETCHCR",  # n=1000, but will be skipped
)

ALL = list(sif2jax.problems)
SCALAR = [
    p for p in ALL
    if isinstance(p, (AbstractUnconstrainedMinimisation,
                      AbstractBoundedMinimisation,
                      AbstractConstrainedQuadraticProblem))
]
PROBLEMS = [p for p in SCALAR
            if MIN_N < p.y0.size <= MAX_N
            and any(p.name.startswith(pref) for pref in _ALLOW_PREFIXES)]


def _cls(p):
    for c in type(p).__mro__:
        if c.__name__.startswith("Abstract"):
            return c.__name__
    return type(p).__name__


def _id(p):
    return f"{_cls(p)}-{p.name}"


def _info(p, impl):
    return {"problem_name": p.name, "dimensionality": p.y0.size,
            "problem_type": _cls(p), "implementation": impl}


def _block(out):
    for l in jax.tree_util.tree_leaves(out): jax.block_until_ready(l)
    return out


def _make(extractor, problem):
    args_c = problem.args

    if extractor is None:
        @jax.jit
        def fn(y):
            return jax.hessian(problem.objective, argnums=0)(y, args_c)
    else:
        @jax.jit
        def fn(y):
            _, h = jax.linearize(
                jax.grad(lambda z: problem.objective(z, args_c)), y)
            return extractor(h, y)

    try:
        compiled = fn.lower(problem.y0).compile()
        _block(compiled(problem.y0))
        return compiled
    except Exception:
        return None


@pytest.mark.parametrize("problem", PROBLEMS, ids=_id)
def test_jax_hessian(benchmark, problem):
    benchmark.group = f"jax_hessian-{_cls(problem)}"
    benchmark.name = f"test_jax_hessian[{problem.name}]"
    c = _make(None, problem)
    if c is None: pytest.skip("failed")
    benchmark(lambda y: _block(c(y)), jax.device_put(problem.y0))
    benchmark.extra_info.update(_info(problem, "jax.hessian"))
    jax.clear_caches()


@pytest.mark.parametrize("problem", PROBLEMS, ids=_id)
def test_materialize(benchmark, problem):
    benchmark.group = f"materialize-{_cls(problem)}"
    benchmark.name = f"test_materialize[{problem.name}]"
    c = _make(materialize, problem)
    if c is None: pytest.skip("failed")
    benchmark(lambda y: _block(c(y)), jax.device_put(problem.y0))
    benchmark.extra_info.update(_info(problem, "materialize"))
    jax.clear_caches()


@pytest.mark.parametrize("problem", PROBLEMS, ids=_id)
def test_bcoo_jacobian(benchmark, problem):
    benchmark.group = f"bcoo_jacobian-{_cls(problem)}"
    benchmark.name = f"test_bcoo_jacobian[{problem.name}]"
    c = _make(lambda h, y: materialize(h, y, format="bcoo"), problem)
    if c is None: pytest.skip("failed")
    benchmark(lambda y: _block(c(y)), jax.device_put(problem.y0))
    benchmark.extra_info.update(_info(problem, "bcoo_jacobian"))
    jax.clear_caches()
