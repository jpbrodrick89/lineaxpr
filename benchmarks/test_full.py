"""Comprehensive Hessian benchmark: jax.hessian, materialize, bcoo_jacobian,
asdex (dense + BCOO), on all scalar-objective CUTEst problems where we can
fit in memory.

Two size tiers:
  - DENSE_MAX (default 2000): dense methods (jax.hessian, materialize, asdex-dense)
  - BCOO_MAX  (default 5000): sparse methods (bcoo_jacobian, asdex-bcoo)
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
import os
import sys


import jax
import pytest
import sif2jax  # noqa: E402
from lineaxpr import bcoo_jacobian, materialize  # noqa: E402
from sif2jax._problem import (  # noqa: E402
    AbstractBoundedMinimisation,
    AbstractConstrainedQuadraticProblem,
    AbstractUnconstrainedMinimisation,
)

try:
    import asdex
    HAS_ASDEX = True
except ImportError:
    HAS_ASDEX = False

# Thresholds empirically derived (see threshold probe in docs/RESEARCH_NOTES.md
# or re-run with `uv run python benchmarks/probe_thresholds.py`):
#   DENSE_MAX=2000 keeps materialize runtime under ~5ms/call (dense alloc
#     dominates above this; n=5000 still works but costs ~15ms/call).
#   BCOO_MAX=5000 is the empirical ceiling for bcoo_jacobian; problems
#     that would produce near-dense BCOO fall back to ndarray at runtime
#     and cost similar to materialize (e.g. CRAGGLVY at n=5000 ≈ 20ms).
DENSE_MAX = int(os.environ.get("DENSE_MAX", "2000"))
BCOO_MAX = int(os.environ.get("BCOO_MAX", "5000"))

ALL = list(sif2jax.problems)
SCALAR = [
    p for p in ALL
    if isinstance(p, (AbstractUnconstrainedMinimisation,
                      AbstractBoundedMinimisation,
                      AbstractConstrainedQuadraticProblem))
]
DENSE_PROBLEMS = [p for p in SCALAR if p.y0.size <= DENSE_MAX]
BCOO_PROBLEMS = [p for p in SCALAR if p.y0.size <= BCOO_MAX]


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


def _compile(fn, primal):
    try:
        compiled = fn.lower(primal).compile()
        _block(compiled(primal))
        return compiled
    except Exception:
        return None


def _make(extractor_kind, problem):
    """extractor_kind: one of 'jaxhes', 'jaxhes_folded', 'mat', 'bcoo',
    'asdex_bcoo', 'asdex_dense'."""
    args_c = problem.args

    def f(y):
        return problem.objective(y, args_c)

    if extractor_kind == "jaxhes":
        @jax.jit
        def fn(y): return jax.hessian(f)(y)
        return _compile(fn, problem.y0)

    if extractor_kind == "jaxhes_folded":
        # jax.hessian WITH eager_constant_folding (release config).
        from jax._src import config
        @jax.jit
        def fn(y): return jax.hessian(f)(y)
        with config.eager_constant_folding(True):
            return _compile(fn, problem.y0)

    if extractor_kind == "mat":
        @jax.jit
        def fn(y):
            _, h = jax.linearize(jax.grad(f), y)
            return materialize(h, y)
        return _compile(fn, problem.y0)

    if extractor_kind == "bcoo":
        @jax.jit
        def fn(y):
            _, h = jax.linearize(jax.grad(f), y)
            return bcoo_jacobian(h, y)
        return _compile(fn, problem.y0)

    if extractor_kind == "asdex_bcoo":
        if not HAS_ASDEX: return None
        try:
            fn = jax.jit(asdex.hessian(f, input_shape=problem.y0.shape,
                                        output_format="bcoo", symmetric=True))
        except Exception:
            return None
        return _compile(fn, problem.y0)

    if extractor_kind == "asdex_dense":
        if not HAS_ASDEX: return None
        try:
            fn = jax.jit(asdex.hessian(f, input_shape=problem.y0.shape,
                                        output_format="dense", symmetric=True))
        except Exception:
            return None
        return _compile(fn, problem.y0)

    raise ValueError(extractor_kind)


# ------------------- dense-output methods (cap DENSE_MAX) -------------------


@pytest.mark.parametrize("problem", DENSE_PROBLEMS, ids=_id)
def test_jax_hessian(benchmark, problem):
    """jax.hessian with eager_constant_folding OFF (default)."""
    benchmark.group = f"jax_hessian-{_cls(problem)}"
    benchmark.name = f"test_jax_hessian[{problem.name}]"
    c = _make("jaxhes", problem)
    if c is None: pytest.skip("jax.hessian failed/too big")
    benchmark(lambda y: _block(c(y)), jax.device_put(problem.y0))
    benchmark.extra_info.update(_info(problem, "jax.hessian"))
    jax.clear_caches()


@pytest.mark.parametrize("problem", DENSE_PROBLEMS, ids=_id)
def test_jax_hessian_folded(benchmark, problem):
    """jax.hessian with eager_constant_folding=True (release config)."""
    benchmark.group = f"jax_hessian_folded-{_cls(problem)}"
    benchmark.name = f"test_jax_hessian_folded[{problem.name}]"
    c = _make("jaxhes_folded", problem)
    if c is None: pytest.skip("jax.hessian (folded) failed/too big")
    benchmark(lambda y: _block(c(y)), jax.device_put(problem.y0))
    benchmark.extra_info.update(_info(problem, "jax.hessian (folded)"))
    jax.clear_caches()


@pytest.mark.parametrize("problem", DENSE_PROBLEMS, ids=_id)
def test_materialize(benchmark, problem):
    benchmark.group = f"materialize-{_cls(problem)}"
    benchmark.name = f"test_materialize[{problem.name}]"
    c = _make("mat", problem)
    if c is None: pytest.skip("materialize failed")
    benchmark(lambda y: _block(c(y)), jax.device_put(problem.y0))
    benchmark.extra_info.update(_info(problem, "materialize"))
    jax.clear_caches()


@pytest.mark.parametrize("problem", DENSE_PROBLEMS, ids=_id)
def test_asdex_dense(benchmark, problem):
    if not HAS_ASDEX: pytest.skip("asdex not installed")
    benchmark.group = f"asdex_dense-{_cls(problem)}"
    benchmark.name = f"test_asdex_dense[{problem.name}]"
    c = _make("asdex_dense", problem)
    if c is None: pytest.skip("asdex dense failed")
    benchmark(lambda y: _block(c(y)), jax.device_put(problem.y0))
    benchmark.extra_info.update(_info(problem, "asdex_dense"))
    jax.clear_caches()


# ------------------- sparse-output methods (cap BCOO_MAX) -------------------


@pytest.mark.parametrize("problem", BCOO_PROBLEMS, ids=_id)
def test_bcoo_jacobian(benchmark, problem):
    benchmark.group = f"bcoo_jacobian-{_cls(problem)}"
    benchmark.name = f"test_bcoo_jacobian[{problem.name}]"
    c = _make("bcoo", problem)
    if c is None: pytest.skip("bcoo_jacobian failed")
    benchmark(lambda y: _block(c(y)), jax.device_put(problem.y0))
    benchmark.extra_info.update(_info(problem, "bcoo_jacobian"))
    jax.clear_caches()


@pytest.mark.parametrize("problem", BCOO_PROBLEMS, ids=_id)
def test_asdex_bcoo(benchmark, problem):
    if not HAS_ASDEX: pytest.skip("asdex not installed")
    benchmark.group = f"asdex_bcoo-{_cls(problem)}"
    benchmark.name = f"test_asdex_bcoo[{problem.name}]"
    c = _make("asdex_bcoo", problem)
    if c is None: pytest.skip("asdex bcoo failed")
    benchmark(lambda y: _block(c(y)), jax.device_put(problem.y0))
    benchmark.extra_info.update(_info(problem, "asdex_bcoo"))
    jax.clear_caches()
