# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""Compile-time benchmarks for jax.hessian, asdex bcoo, and lineaxpr bcoo.

"Compile time" here = full cold path from problem to a ready-to-call
compiled artifact. Includes:

- jax.hessian   : trace through `jax.hessian(obj)` → lower → XLA compile
- asdex bcoo    : `asdex.hessian_coloring` (pattern analysis) + wrapping
                  via `hessian_from_coloring` + trace + lower + XLA compile
- lineaxpr bcoo : trace through `linearize(grad(obj), y)` + walk the
                  linear jaxpr (LinOp tree) + BCOO emission + lower +
                  XLA compile

Each round is an independent build against a fresh `jax.clear_caches()`
state so no prior-compile cache hits. We use `benchmark.pedantic` with
`iterations=1` (compile is a once-off event, not something pytest-bench
should try to auto-calibrate) and a small `rounds` (default 3).

Running
-------
In container:

    USE_CONTAINER=1 NO_EAGER=1 bash benchmarks/run_in_container.sh \
        benchmarks/test_compile_time.py \
        --benchmark-only \
        --benchmark-save=compile_times \
        --benchmark-columns=min,median,mean,rounds \
        --timeout=90 --timeout-method=signal

Filter to a subset:

    ... -k "ARGTRIGLS or TABLE8 or LIARWHD"

Env vars:

- `COMPILE_ROUNDS=3`      rounds per benchmark (default 3)
- `COMPILE_MAX_DENSE=2000` n cap for dense methods (jax.hessian)
- `COMPILE_MAX_BCOO=5000`  n cap for bcoo methods
- `ASDEX_PATHOLOGICAL_OVERRIDE=1` force-run asdex on coloring-pathological
  problems (EIGEN*, CHARDIS0, MSQRTALS) — normally skipped.

Caveats
-------
- CHARDIS0 is excluded from bcoo path: it OOMs the container's default
  memory. Use `-k "not CHARDIS0"` if you see it.
- asdex `hessian_coloring` can take >60s on its pathological list; those
  are skipped up-front rather than triggering the pytest-timeout
  fallback on each one.
"""

from __future__ import annotations
import os

import jax
import pytest
import sif2jax  # noqa: E402
from sif2jax._problem import (  # noqa: E402
    AbstractBoundedMinimisation,
    AbstractConstrainedQuadraticProblem,
    AbstractUnconstrainedMinimisation,
)

from lineaxpr import bcoo_hessian  # noqa: E402

try:
    import asdex

    HAS_ASDEX = True
except ImportError:
    HAS_ASDEX = False


# --- config -----------------------------------------------------------

COMPILE_ROUNDS = int(os.environ.get("COMPILE_ROUNDS", "3"))
MAX_DENSE = int(os.environ.get("COMPILE_MAX_DENSE", "2000"))
MAX_BCOO = int(os.environ.get("COMPILE_MAX_BCOO", "5000"))

# Same list as test_full.py — asdex `hessian_coloring` takes >30s on
# these and hits the outer timeout every run. Not useful signal.
ASDEX_PATHOLOGICAL = {"EIGENALS", "EIGENBLS", "EIGENCLS", "CHARDIS0", "MSQRTALS"}

ALL = list(sif2jax.problems)
SCALAR = [
    p
    for p in ALL
    if isinstance(
        p,
        (
            AbstractUnconstrainedMinimisation,
            AbstractBoundedMinimisation,
            AbstractConstrainedQuadraticProblem,
        ),
    )
]


def _cls(p):
    for c in type(p).__mro__:
        if c.__name__.startswith("Abstract"):
            return c.__name__
    return type(p).__name__


def _id(p):
    return f"{_cls(p)}-{p.name}"


def _info(p, impl):
    return {
        "problem_name": p.name,
        "dimensionality": p.y0.size,
        "problem_type": _cls(p),
        "implementation": impl,
    }


# --- build functions: each is ONE round of cold compile ----------------
#
# Kept as module-level closures (not `lambda` inside the test body) so
# every round gets a NEW Python object — guarantees jax can't hit its
# per-function compile cache.


def _fresh_obj(problem):
    """Return a fresh closure over problem args each call — ensures
    distinct jit cache identities across rounds."""
    args_c = problem.args

    def obj(y):
        return problem.objective(y, args_c)

    return obj


def _build_jaxhes(problem):
    obj = _fresh_obj(problem)
    fn = jax.jit(jax.hessian(obj))
    return fn.lower(problem.y0).compile()


def _build_asdex_bcoo(problem):
    obj = _fresh_obj(problem)
    coloring = asdex.hessian_coloring(obj, input_shape=problem.y0.shape, symmetric=True)
    fn = jax.jit(asdex.hessian_from_coloring(obj, coloring, output_format="bcoo"))
    return fn.lower(problem.y0).compile()


def _build_lineaxpr_bcoo(problem):
    obj = _fresh_obj(problem)
    fn = jax.jit(bcoo_hessian(obj))
    return fn.lower(problem.y0).compile()


# --- pedantic wrapper --------------------------------------------------


def _bench_compile(benchmark, build_fn, problem, impl_name):
    """Run `build_fn(problem)` COMPILE_ROUNDS times, one iteration per
    round, with `jax.clear_caches()` in setup so each round is cold.

    We use `benchmark.pedantic` because the default pytest-benchmark
    auto-calibration is wrong for compile: it wants to find a sub-1s
    inner duration and then repeat; compile events are 100ms–10s one-offs
    and should be measured exactly that many times, not more.
    """
    import gc

    def setup():
        jax.clear_caches()
        gc.collect()
        return (problem,), {}

    try:
        benchmark.pedantic(
            build_fn,
            setup=setup,
            rounds=COMPILE_ROUNDS,
            iterations=1,
            warmup_rounds=0,
        )
    except Exception as e:
        pytest.skip(f"{impl_name} build failed: {type(e).__name__}: {e}")

    benchmark.extra_info.update(_info(problem, impl_name))


# --- tests: one per method × problem ----------------------------------

DENSE_PROBLEMS = [p for p in SCALAR if p.y0.size <= MAX_DENSE]
BCOO_PROBLEMS = [p for p in SCALAR if p.y0.size <= MAX_BCOO]


@pytest.mark.parametrize("problem", DENSE_PROBLEMS, ids=_id)
def test_compile_jax_hessian(benchmark, problem):
    benchmark.group = f"compile_jax_hessian-{_cls(problem)}"
    benchmark.name = f"compile_jax_hessian[{problem.name}]"
    _bench_compile(benchmark, _build_jaxhes, problem, "jax.hessian")


@pytest.mark.parametrize("problem", BCOO_PROBLEMS, ids=_id)
def test_compile_asdex_bcoo(benchmark, problem):
    if not HAS_ASDEX:
        pytest.skip("asdex not installed")
    if problem.name in ASDEX_PATHOLOGICAL and not os.environ.get(
        "ASDEX_PATHOLOGICAL_OVERRIDE"
    ):
        pytest.skip(
            f"asdex coloring >30s on {problem.name} — "
            f"override with ASDEX_PATHOLOGICAL_OVERRIDE=1"
        )
    benchmark.group = f"compile_asdex_bcoo-{_cls(problem)}"
    benchmark.name = f"compile_asdex_bcoo[{problem.name}]"
    _bench_compile(benchmark, _build_asdex_bcoo, problem, "asdex_bcoo")


@pytest.mark.parametrize("problem", BCOO_PROBLEMS, ids=_id)
def test_compile_lineaxpr_bcoo(benchmark, problem):
    benchmark.group = f"compile_lineaxpr_bcoo-{_cls(problem)}"
    benchmark.name = f"compile_lineaxpr_bcoo[{problem.name}]"
    _bench_compile(benchmark, _build_lineaxpr_bcoo, problem, "lineaxpr_bcoo")
