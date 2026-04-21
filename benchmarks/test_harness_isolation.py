# pyright: reportMissingImports=false
"""Pin where the ~200µs contamination lives: pytest itself, or
pytest-benchmark's measurement specifically.

Runs the preamble, then measures TABLE8 materialize two ways in the
same test: (1) manual time.perf_counter loop, (2) benchmark(fn, y).
Side-by-side min tells us whether the benchmark fixture adds the tax.
"""
from __future__ import annotations
import gc
import time

import jax
import sif2jax

from lineaxpr import materialize


def _block(out):
    for l in jax.tree_util.tree_leaves(out):
        jax.block_until_ready(l)
    return out


def _build(problem):
    args_c = problem.args

    @jax.jit
    def fn(y):
        def f(z): return problem.objective(z, args_c)
        _, h = jax.linearize(jax.grad(f), y)
        return materialize(h, y)
    return fn


_BY_NAME = {type(p).__name__: p for p in sif2jax.problems}
_PREAMBLE_NAMES = [
    "CHNROSNB", "GENROSE", "GAUSS1LS", "GAUSS2LS", "GAUSS3LS",
    "LUKSAN11LS", "LUKSAN12LS", "LUKSAN13LS", "LUKSAN14LS",
    "ARGLINA", "ARGLINB", "ARGLINC",
    "VESUVIOLS", "VESUVIOULS", "VESUVIALS",
    "FLETCHCR", "INTEQNELS",
]


def _run_preamble():
    for name in _PREAMBLE_NAMES:
        p = _BY_NAME.get(name)
        if p is None:
            continue
        try:
            c = _build(p)
            y = jax.device_put(p.y0)
            _block(c(y))
            for _ in range(10):
                _block(c(y))
            del c, y
        except Exception:
            pass
        gc.collect()


def test_manual_vs_benchmark(benchmark):
    """Measure TABLE8 min manually, then via benchmark() fixture, in the
    same process after a realistic preamble."""
    _run_preamble()

    target = _BY_NAME["TABLE8"]
    c = _build(target)
    y = jax.device_put(target.y0)

    # Warm
    _block(c(y))
    for _ in range(20):
        _block(c(y))

    # Manual loop, 2000 iters, take min
    times = []
    for _ in range(2000):
        t = time.perf_counter()
        _block(c(y))
        times.append(time.perf_counter() - t)
    manual_min_us = min(times) * 1e6
    manual_median_us = sorted(times)[len(times) // 2] * 1e6

    # pytest-benchmark's measurement on the SAME function in the SAME
    # process immediately after
    benchmark(lambda yy: _block(c(yy)), y)

    # Compare after-the-fact — stash manual result on benchmark.extra_info
    benchmark.extra_info["manual_min_us"] = manual_min_us
    benchmark.extra_info["manual_median_us"] = manual_median_us
    print(f"\n[harness-isolation] manual: min={manual_min_us:.1f}us "
          f"median={manual_median_us:.1f}us")
    print(f"[harness-isolation] benchmark: "
          f"min={benchmark.stats.stats.min*1e6:.1f}us "
          f"median={benchmark.stats.stats.median*1e6:.1f}us "
          f"rounds={benchmark.stats.stats.rounds}")
