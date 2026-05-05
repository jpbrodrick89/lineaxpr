# pyright: reportMissingImports=false
"""Small targeted probe for full-sweep contamination on dense-broadcast
Diagonal.todense impls.

Matrix:
  - impl     : scatter / veye / where
  - cleanup  : none / gc_sync / clear_backends
  - warmup   : default / 50 / 500
  - position : cold (problem first, clean harness) / hot (after preamble)

The preamble runs a handful of n×n dense-allocating benches before the
target, replicating the state the main sweep lands in. Comparing cold vs
hot min isolates the persistent-floor contamination; comparing cleanup
variants tells us which (if any) resets it.

Run with:
  uv run pytest benchmarks/test_contamination_probe.py --benchmark-only \
    --benchmark-columns=min,median,rounds --benchmark-group-by=func -v

Reads TARGETS env var to pick which problems (comma-separated); default
runs the full 10-problem probe. Use TARGETS=TABLE8 etc for a tight loop.
"""
from __future__ import annotations
import gc
import os

import jax
import jax.numpy as jnp
import pytest
import sif2jax

from lineaxpr import materialize
from lineaxpr import _base


# --- impl variants for Diagonal.todense -------------------------------


def _todense_scatter(self):
    idx = jnp.arange(self.n)
    return jnp.zeros((self.n, self.n), self.data.dtype).at[idx, idx].set(self.data)


def _todense_veye(self):
    return self.data[:, None] * jnp.eye(self.n, dtype=self.data.dtype)


def _todense_where(self):
    return jnp.where(
        jnp.eye(self.n, dtype=jnp.bool_),
        self.data[:, None],
        jnp.zeros((), self.data.dtype),
    )


IMPLS = {
    "scatter": _todense_scatter,
    "veye": _todense_veye,
    "where": _todense_where,
}


# --- cleanup variants -------------------------------------------------


def _cleanup_none(): pass


def _cleanup_gc_sync():
    jax.clear_caches()
    gc.collect()
    try:
        for d in jax.devices():
            d.synchronize_all_activity()
    except AttributeError:
        pass


def _cleanup_clear_backends():
    jax.clear_caches()
    # jax.clear_backends is deprecated but still callable; if gone, fall
    # through to clear_caches + gc.
    try:
        jax.clear_backends()
    except Exception:
        pass
    gc.collect()


CLEANUPS = {
    "none": _cleanup_none,
    "gc_sync": _cleanup_gc_sync,
    "clear_backends": _cleanup_clear_backends,
}


# --- problem selection ------------------------------------------------


_ALL_BY_NAME = {type(p).__name__: p for p in sif2jax.problems}

_DEFAULT_TARGETS = [
    # dense-contamination suspects
    "TABLE7", "TABLE8", "EXPLIN", "EXPLIN2", "EG2",
    # fixture-hurt suspects (bcoo at n=5000)
    "NONCVXUN", "NONCVXU2",
    # mid-n pure-diagonal
    "DIAGPQB", "DIAGIQT",
    # controls
    "ARGTRIGLS",  # small-n
    "LIARWHD",    # large pure-diagonal (bcoo path)
]

TARGETS_ENV = os.environ.get("TARGETS")
if TARGETS_ENV:
    TARGET_NAMES = [s.strip() for s in TARGETS_ENV.split(",") if s.strip()]
else:
    TARGET_NAMES = _DEFAULT_TARGETS

TARGETS = [_ALL_BY_NAME[n] for n in TARGET_NAMES if n in _ALL_BY_NAME]


# Preamble: problems that allocate n×n densely to simulate full-sweep
# state. Chosen to be fast enough (~20 total ms) but exercise the
# allocator + XLA runtime like a real sweep would.
_PREAMBLE_NAMES = [
    # mid-n dense materialize hits
    "CHNROSNB", "GENROSE", "GAUSS1LS", "GAUSS2LS", "GAUSS3LS",
    "LUKSAN11LS", "LUKSAN12LS", "LUKSAN13LS", "LUKSAN14LS",
    "ARGLINA", "ARGLINB", "ARGLINC",
    "VESUVIOLS", "VESUVIOULS", "VESUVIALS",
    "FLETCHCR", "INTEQNELS",
]
PREAMBLE = [_ALL_BY_NAME[n] for n in _PREAMBLE_NAMES if n in _ALL_BY_NAME]


# --- compile + warmup helpers ----------------------------------------


def _block(out):
    for l in jax.tree_util.tree_leaves(out):
        jax.block_until_ready(l)
    return out


def _build_mat_fn(problem):
    args_c = problem.args

    @jax.jit
    def fn(y):
        def f(z): return problem.objective(z, args_c)
        _, h = jax.linearize(jax.grad(f), y)
        return materialize(h, y)
    return fn


def _build_bcoo_fn(problem):
    args_c = problem.args

    @jax.jit
    def fn(y):
        def f(z): return problem.objective(z, args_c)
        _, h = jax.linearize(jax.grad(f), y)
        return materialize(h, y, format="bcoo")
    return fn


def _compile_and_warm(build_fn, problem, extra_warmups: int):
    y = jax.device_put(problem.y0)
    fn = build_fn(problem)
    compiled = fn.lower(problem.y0).compile()
    _block(compiled(y))
    for _ in range(extra_warmups):
        _block(compiled(y))
    return compiled, y


def _run_preamble():
    """Run materialize + bcoo on the preamble problems to simulate sweep
    state. Measured in-line, not via pytest-benchmark — just executes."""
    for p in PREAMBLE:
        try:
            c, y = _compile_and_warm(_build_mat_fn, p, extra_warmups=0)
            for _ in range(10):
                _block(c(y))
        except Exception:
            pass
        jax.clear_caches()
        gc.collect()


# --- parametrization --------------------------------------------------


def _ids_impl(v): return f"impl-{v}"
def _ids_cleanup(v): return f"cu-{v}"
def _ids_warmup(v): return f"wu-{v}"
def _ids_position(v): return f"pos-{v}"
def _ids_problem(p): return p.name


@pytest.fixture
def impl(request, monkeypatch):
    monkeypatch.setattr(_base.Diagonal, "todense", IMPLS[request.param])
    return request.param


@pytest.fixture
def cleanup(request):
    CLEANUPS[request.param]()
    return request.param


# Preamble runs once per session *unless* PREAMBLE=0 is set. Compare two
# invocations (with/without preamble) to isolate position-dependent
# contamination from impl/cleanup/warmup effects.
_PREAMBLE_ENABLED = os.environ.get("PREAMBLE", "1") == "1"


@pytest.fixture(scope="session", autouse=True)
def _maybe_preamble():
    if _PREAMBLE_ENABLED:
        _run_preamble()
    yield


@pytest.mark.parametrize("problem", TARGETS, ids=_ids_problem)
@pytest.mark.parametrize("extra_warmups", [0, 50, 500], ids=_ids_warmup)
@pytest.mark.parametrize("cleanup", list(CLEANUPS), ids=_ids_cleanup, indirect=True)
@pytest.mark.parametrize("impl", list(IMPLS), ids=_ids_impl, indirect=True)
def test_probe_mat(benchmark, problem, extra_warmups, cleanup, impl):
    if problem.y0.size > 2000:
        pytest.skip("too large for materialize")
    c, y = _compile_and_warm(_build_mat_fn, problem, extra_warmups)
    tag = "hot" if _PREAMBLE_ENABLED else "cold"
    benchmark.group = f"mat_{problem.name}"
    benchmark.name = f"mat[{problem.name}|{impl}|{cleanup}|wu{extra_warmups}|{tag}]"
    benchmark(lambda yy: _block(c(yy)), y)


@pytest.mark.parametrize("problem", TARGETS, ids=_ids_problem)
@pytest.mark.parametrize("extra_warmups", [0, 50, 500], ids=_ids_warmup)
@pytest.mark.parametrize("cleanup", list(CLEANUPS), ids=_ids_cleanup, indirect=True)
@pytest.mark.parametrize("impl", list(IMPLS), ids=_ids_impl, indirect=True)
def test_probe_bcoo(benchmark, problem, extra_warmups, cleanup, impl):
    if problem.y0.size > 5000:
        pytest.skip("too large for bcoo")
    c, y = _compile_and_warm(_build_bcoo_fn, problem, extra_warmups)
    tag = "hot" if _PREAMBLE_ENABLED else "cold"
    benchmark.group = f"bcoo_{problem.name}"
    benchmark.name = f"bcoo[{problem.name}|{impl}|{cleanup}|wu{extra_warmups}|{tag}]"
    benchmark(lambda yy: _block(c(yy)), y)
