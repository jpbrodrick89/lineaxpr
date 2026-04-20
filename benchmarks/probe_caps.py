"""Probe the practical n-cap of each reference method.

Runs each method on a spectrum of problems, recording (compile_s, run_s)
or failure reason. Meant to calibrate the DENSE_MAX / BCOO_MAX caps in
run_bench.sh's per-method modes.

Each method runs in its own subprocess so eager_constant_folding,
JAX cache state, and asdex coloring memory don't leak across measurements.

Usage:
    uv run python -m benchmarks.probe_caps
    # or with a custom timeout per method:
    PROBE_TIMEOUT=30 uv run python -m benchmarks.probe_caps
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


# Problems ordered by n, spanning sparse-banded / dense-ish / constant-H.
PROBES = [
    # (module, class, n, note)
    ("sif2jax.cutest._bounded_minimisation.levymont", "LEVYMONT", 100, "banded y-dep"),
    ("sif2jax.cutest._unconstrained_minimisation.argtrigls", "ARGTRIGLS", 200, "dense-ish"),
    ("sif2jax.cutest._unconstrained_minimisation.fletchcr", "FLETCHCR", 1000, "banded y-dep"),
    ("sif2jax.cutest._quadratic_problems.cmpc2", "CMPC2", 1530, "constant-H sparse"),
    ("sif2jax.cutest._unconstrained_minimisation.edensch", "EDENSCH", 2000, "y-dep sparse"),
    ("sif2jax.cutest._quadratic_problems.cmpc1", "CMPC1", 2550, "constant-H sparse"),
    ("sif2jax.cutest._unconstrained_minimisation.dixmaanb", "DIXMAANB", 3000, "y-dep banded"),
    ("sif2jax.cutest._unconstrained_minimisation.bdqrtic", "BDQRTIC", 5000, "banded y-dep"),
    ("sif2jax.cutest._bounded_minimisation.bdexp", "BDEXP", 5000, "bounded sparse"),
]

METHODS = ["jaxhes", "jaxhes_folded", "asdex_dense", "asdex_bcoo"]

TIMEOUT = int(os.environ.get("PROBE_TIMEOUT", "60"))


_CHILD_SCRIPT = """
import sys, time, json, os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from jax._src import config
import jax
jax.config.update("jax_enable_x64", True)

modpath, clsname, method = sys.argv[1], sys.argv[2], sys.argv[3]
mod = __import__(modpath, fromlist=[clsname])
p = getattr(mod, clsname)()
args_c = p.args

def f(y):
    return p.objective(y, args_c)

try:
    if method == "jaxhes":
        @jax.jit
        def fn(y): return jax.hessian(f)(y)
        t_c = time.perf_counter()
        c = fn.lower(p.y0).compile()
        t_compile = time.perf_counter() - t_c

    elif method == "jaxhes_folded":
        @jax.jit
        def fn(y): return jax.hessian(f)(y)
        with config.eager_constant_folding(True):
            t_c = time.perf_counter()
            c = fn.lower(p.y0).compile()
            t_compile = time.perf_counter() - t_c

    elif method == "asdex_dense":
        import asdex
        afn = jax.jit(asdex.hessian(f, input_shape=p.y0.shape,
                                    output_format="dense", symmetric=True))
        t_c = time.perf_counter()
        c = afn.lower(p.y0).compile()
        t_compile = time.perf_counter() - t_c

    elif method == "asdex_bcoo":
        import asdex
        afn = jax.jit(asdex.hessian(f, input_shape=p.y0.shape,
                                    output_format="bcoo", symmetric=True))
        t_c = time.perf_counter()
        c = afn.lower(p.y0).compile()
        t_compile = time.perf_counter() - t_c

    else:
        raise ValueError(method)

    # Warmup (excluded from runtime).
    out = c(p.y0)
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()

    # Time 3 iterations; report min.
    ts = []
    for _ in range(3):
        s = time.perf_counter()
        out = c(p.y0)
        for leaf in jax.tree_util.tree_leaves(out):
            leaf.block_until_ready()
        ts.append(time.perf_counter() - s)
    t_run = min(ts)

    print(json.dumps({"ok": True, "compile_s": t_compile, "run_s": t_run}))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:100]}"}))
"""


def _run(modpath, clsname, method):
    """Run one (problem, method) in a subprocess with a timeout."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "-c", _CHILD_SCRIPT, modpath, clsname, method],
            capture_output=True, text=True, timeout=TIMEOUT,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if out.returncode != 0:
            return {"ok": False, "error": f"exit {out.returncode}: {out.stderr[-200:]}"}
        # Last JSON line is the result.
        for line in out.stdout.strip().split("\n")[::-1]:
            if line.startswith("{"):
                return json.loads(line)
        return {"ok": False, "error": "no JSON output"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout >{TIMEOUT}s"}


def main():
    print(f"{'problem':12s} {'n':>5s} {'note':18s}  "
          + "  ".join(f"{m:24s}" for m in METHODS))
    print("-" * (12 + 1 + 5 + 1 + 18 + 2 + 26 * len(METHODS)))
    for modpath, clsname, n, note in PROBES:
        cells = []
        for method in METHODS:
            r = _run(modpath, clsname, method)
            if r["ok"]:
                cells.append(f"{r['compile_s']*1000:6.0f}ms/{r['run_s']*1e6:8.0f}µs")
            else:
                cells.append(f"FAIL {r['error'][:18]:18s}")
        print(f"{clsname:12s} {n:>5d} {note:18s}  " + "  ".join(f"{c:24s}" for c in cells))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
