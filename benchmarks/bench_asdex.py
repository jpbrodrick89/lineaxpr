"""Compare asdex vs materialize on a curated problem set.

Requires `pip install asdex` in the runtime environment.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("EAGER_CONSTANT_FOLDING", "TRUE")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
from lineaxpr import materialize  # noqa: E402

try:
    import asdex  # noqa: E402
    HAS_ASDEX = True
except ImportError:
    HAS_ASDEX = False
    print("asdex not installed. `pip install asdex` first. Exiting.")
    sys.exit(0)


def bench(fn, args, N=200, W=10):
    for _ in range(W):
        out = fn(*args)
        for l in jax.tree_util.tree_leaves(out):
            jax.block_until_ready(l)
    t0 = time.perf_counter()
    for _ in range(N):
        out = fn(*args)
        for l in jax.tree_util.tree_leaves(out):
            jax.block_until_ready(l)
    return (time.perf_counter() - t0) / N


def test_one(problem):
    """y passed as JIT INPUT (not closure)."""
    args_c = problem.args
    y = problem.y0 + 0.001
    n = y.shape[0]

    def f(y):
        return problem.objective(y, args_c)

    @jax.jit
    def jaxhes(y): return jax.hessian(f)(y)

    @jax.jit
    def mat(y):
        _, h = jax.linearize(jax.grad(f), y)
        return materialize(h, y)

    @jax.jit
    def bcoo(y):
        _, h = jax.linearize(jax.grad(f), y)
        return materialize(h, y, format="bcoo")

    # asdex: static coloring, needs input_shape upfront
    try:
        asdex_hess_fn = asdex.hessian(f, input_shape=y.shape,
                                       output_format="bcoo", symmetric=True)
        asdex_jit = jax.jit(asdex_hess_fn)
        # verify correctness
        asdex_H = asdex_jit(y)
        asdex_dense = asdex_H.todense() if hasattr(asdex_H, "todense") else asdex_H
        ref = jaxhes(y)
        err = float(jnp.max(jnp.abs(asdex_dense - ref)))
        asdex_ok = err < 1e-8
    except Exception:
        asdex_jit = None
        asdex_ok = False
        err = 0

    th = bench(jaxhes, (y,))
    tm = bench(mat, (y,))
    tb = bench(bcoo, (y,))
    ta = bench(asdex_jit, (y,)) if asdex_jit else float("inf")

    print(f"{problem.name:15s} n={n:>5d}  "
          f"hess {th*1e6:>9.1f}µ  mat {tm*1e6:>9.1f}µ ({th/tm:>5.2f}x)  "
          f"bcoo {tb*1e6:>9.1f}µ ({th/tb:>7.2f}x)  "
          f"asdex {ta*1e6:>9.1f}µ ({th/ta:>6.2f}x)  asdex_ok={asdex_ok}")


def main():
    from sif2jax.cutest._unconstrained_minimisation.fletchcr import FLETCHCR
    from sif2jax.cutest._unconstrained_minimisation.dixmaanb import DIXMAANB
    from sif2jax.cutest._unconstrained_minimisation.dixmaani1 import DIXMAANI1
    from sif2jax.cutest._unconstrained_minimisation.genrose import GENROSE
    from sif2jax.cutest._unconstrained_minimisation.argtrigls import ARGTRIGLS
    from sif2jax.cutest._quadratic_problems.cmpc1 import CMPC1
    from sif2jax.cutest._quadratic_problems.dual1 import DUAL1
    from sif2jax.cutest._bounded_minimisation.hs110 import HS110
    from sif2jax.cutest._bounded_minimisation.levymont import LEVYMONT

    problems = [HS110, DUAL1, GENROSE, ARGTRIGLS, LEVYMONT, FLETCHCR,
                DIXMAANI1, DIXMAANB, CMPC1]
    for P in problems:
        try:
            test_one(P())
        except Exception as e:
            print(f"{P.__name__}: FAIL {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
