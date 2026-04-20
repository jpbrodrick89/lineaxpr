"""Curated 5-way comparison: jax.hessian, materialize, bcoo_jacobian, asdex dense, asdex bcoo.

Hand-picked problems covering: tiny, small-dense, medium-dense,
quadratic-constant-H, sparse-banded, sparse-COO, low-rank, large.
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
import os
import sys


import jax
import pytest
import sif2jax  # noqa: E402
from lineaxpr import bcoo_jacobian, materialize  # noqa: E402

try:
    import asdex
    HAS_ASDEX = True
except ImportError:
    HAS_ASDEX = False


# Curated list: (module, class_name)
CURATED = [
    # Tiny
    ("sif2jax.cutest._bounded_minimisation.hs110", "HS110"),
    ("sif2jax.cutest._bounded_minimisation.hart6", "HART6"),
    # Small
    ("sif2jax.cutest._unconstrained_minimisation.qing", "QING"),
    ("sif2jax.cutest._unconstrained_minimisation.chnrosnb", "CHNROSNB"),
    # Quadratic (constant-H)
    ("sif2jax.cutest._quadratic_problems.dual1", "DUAL1"),
    ("sif2jax.cutest._quadratic_problems.dual3", "DUAL3"),
    ("sif2jax.cutest._bounded_minimisation.levymont", "LEVYMONT"),
    ("sif2jax.cutest._unconstrained_minimisation.argtrigls", "ARGTRIGLS"),
    # Medium dense
    ("sif2jax.cutest._unconstrained_minimisation.genrose", "GENROSE"),
    # Medium sparse / banded
    ("sif2jax.cutest._unconstrained_minimisation.fletchcr", "FLETCHCR"),
    # Large sparse quadratic
    ("sif2jax.cutest._quadratic_problems.cmpc1", "CMPC1"),
    ("sif2jax.cutest._quadratic_problems.cmpc2", "CMPC2"),
    # Large y-dependent
    ("sif2jax.cutest._unconstrained_minimisation.dixmaanb", "DIXMAANB"),
    ("sif2jax.cutest._unconstrained_minimisation.dixmaane1", "DIXMAANE1"),
    ("sif2jax.cutest._unconstrained_minimisation.dixmaani1", "DIXMAANI1"),
    ("sif2jax.cutest._unconstrained_minimisation.edensch", "EDENSCH"),
]


def _load(modpath, clsname):
    mod = __import__(modpath, fromlist=[clsname])
    return getattr(mod, clsname)()


PROBLEMS = []
for mod, cls in CURATED:
    try:
        PROBLEMS.append(_load(mod, cls))
    except Exception as e:
        print(f"Could not load {cls}: {e}", file=sys.stderr)


def _id(p):
    return p.name


def _info(p, impl):
    return {"problem_name": p.name, "dimensionality": int(p.y0.size),
            "implementation": impl}


def _block(out):
    for l in jax.tree_util.tree_leaves(out): jax.block_until_ready(l)
    return out


def _compile(fn, primal):
    """Compile `fn.lower(primal).compile()` — helper for _make."""
    try:
        compiled = fn.lower(primal).compile()
        _block(compiled(primal))
        return compiled
    except Exception:
        return None


def _make(extractor_kind, problem):
    args_c = problem.args

    def f(y):
        return problem.objective(y, args_c)

    if extractor_kind == "jaxhes":
        # jax.hessian WITHOUT eager_constant_folding (off by default) —
        # constant-H quadratics stay symbolic, pay n HVPs at runtime.
        @jax.jit
        def fn(y): return jax.hessian(f)(y)
        return _compile(fn, problem.y0)

    if extractor_kind == "jaxhes_folded":
        # jax.hessian WITH eager_constant_folding — the "release config"
        # that JAX's production uses. For constant-H problems this
        # folds the entire Hessian to a literal at compile time (at a
        # possibly large compile-time cost on big sparse H). This is
        # the fair "vs jax" comparison for release-target users.
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


def _run(benchmark, kind, problem, label):
    benchmark.group = f"{kind}"
    benchmark.name = f"test_{kind}[{problem.name}]"
    c = _make(kind, problem)
    if c is None: pytest.skip(f"{kind} failed/unsupported")
    benchmark(lambda y: _block(c(y)), jax.device_put(problem.y0))
    benchmark.extra_info.update(_info(problem, label))
    jax.clear_caches()


@pytest.mark.parametrize("problem", PROBLEMS, ids=_id)
def test_jax_hessian(benchmark, problem):
    """jax.hessian with eager_constant_folding OFF (default JAX config).

    Kept for visibility — on constant-H problems this does n HVPs per
    call at runtime. Most release-target users run with folding on; for
    the fair headline ratio, see `test_jax_hessian_folded`.
    """
    _run(benchmark, "jaxhes", problem, "jax.hessian")


@pytest.mark.parametrize("problem", PROBLEMS, ids=_id)
def test_jax_hessian_folded(benchmark, problem):
    """jax.hessian with eager_constant_folding=True (release config).

    The fair "vs jax.hessian" comparison for release-target users.
    Constant-H quadratics fold to a dense literal at compile time.
    """
    _run(benchmark, "jaxhes_folded", problem, "jax.hessian (folded)")


@pytest.mark.parametrize("problem", PROBLEMS, ids=_id)
def test_materialize(benchmark, problem):
    _run(benchmark, "mat", problem, "materialize")


@pytest.mark.parametrize("problem", PROBLEMS, ids=_id)
def test_bcoo_jacobian(benchmark, problem):
    _run(benchmark, "bcoo", problem, "bcoo_jacobian")


@pytest.mark.parametrize("problem", PROBLEMS, ids=_id)
def test_asdex_dense(benchmark, problem):
    _run(benchmark, "asdex_dense", problem, "asdex_dense")


@pytest.mark.parametrize("problem", PROBLEMS, ids=_id)
def test_asdex_bcoo(benchmark, problem):
    _run(benchmark, "asdex_bcoo", problem, "asdex_bcoo")
