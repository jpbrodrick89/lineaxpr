"""Minimal benchmark: `jnp.diag(v)` vs `v[:, None] * jnp.eye(n)`.

Investigates whether the lineaxpr-internal finding (v*eye is strictly
faster in walks involving downstream `reduce_sum`/`broadcast`/`add`)
generalises to a pattern that would justify a JAX PR.

Runs each impl across a size sweep in four scenarios:

  (A) `build`          : just construct the diagonal matrix.
  (B) `build+reduce`   : `reduce_sum(diag_impl(v), axis=0)` — the
                         cross-talk pattern where the difference
                         showed up in ARGTRIGLS.
  (C) `build+matvec`   : `diag_impl(v) @ x` — common downstream use.
  (D) `jvp`, `vjp`     : forward- and reverse-mode AD of (B).

Usage:
  uv run python benchmarks/diag_benchmark.py
  uv run python benchmarks/diag_benchmark.py --sizes 50,100,500,1000,5000
  uv run python benchmarks/diag_benchmark.py --folded    # EAGER_CONSTANT_FOLDING
  uv run python benchmarks/diag_benchmark.py --out results/cpu
  uv run python benchmarks/diag_benchmark.py --out results/a100_gpu

Output:
  results/<tag>/diag_bench.json    all raw timings + env info
  results/<tag>/diag_bench_*.png   figures per scenario

Re-runnable across machines (CPU, container, A100 GPU). JSONs can be
compared to track how the answer shifts with backend.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path


# IMPORTANT: env-var must be read by JAX at import; set it before `import jax`.
if "--folded" in sys.argv:
    os.environ["EAGER_CONSTANT_FOLDING"] = "true"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------- impls -------------------------------


def scatter_impl(v):
    """`zeros.at[i, i].set(v)` — scatter pattern."""
    n = v.shape[0]
    idx = jnp.arange(n)
    return jnp.zeros((n, n), v.dtype).at[idx, idx].set(v)


def jnp_diag_impl(v):
    """`jnp.diag(v)` — wraps `jnp._diag`, which is itself `@jit`'d."""
    return jnp.diag(v)


def v_times_eye_impl(v):
    """`v[:, None] * jnp.eye(n)` — pure dense multiply."""
    n = v.shape[0]
    return v[:, None] * jnp.eye(n, dtype=v.dtype)


def where_eye_impl(v):
    """`where(eye, v, 0)` — mask-and-select pattern (inline jnp._diag body)."""
    n = v.shape[0]
    return jnp.where(jnp.eye(n, dtype=jnp.bool_), v, jnp.zeros((), v.dtype))


IMPLS = {
    "scatter": scatter_impl,
    "jnp.diag": jnp_diag_impl,
    "v*eye": v_times_eye_impl,
    "where(eye,v,0)": where_eye_impl,
}


# ------------------------------- scenarios -------------------------------


def make_scenarios(diag_impl):
    """Returns `{scenario_name: jitted_fn_of_(v, x)}` for a given diag impl."""

    @jax.jit
    def build(v, x):
        del x
        return diag_impl(v)

    @jax.jit
    def build_reduce(v, x):
        del x
        return diag_impl(v).sum(axis=0)

    @jax.jit
    def build_matvec(v, x):
        return diag_impl(v) @ x

    def fn_for_ad(v):
        return diag_impl(v).sum(axis=0)

    @jax.jit
    def jvp_fn(v, dv):
        return jax.jvp(fn_for_ad, (v,), (dv,))[1]

    @jax.jit
    def vjp_fn(v, cotangent):
        _, vjp = jax.vjp(fn_for_ad, v)
        return vjp(cotangent)[0]

    return {
        "build":        lambda v, x: build(v, x),
        "build+reduce": lambda v, x: build_reduce(v, x),
        "build+matvec": lambda v, x: build_matvec(v, x),
        "jvp":          lambda v, x: jvp_fn(v, x),     # x here is dv
        "vjp":          lambda v, x: vjp_fn(v, x),     # x here is cotangent
    }


# ------------------------------- timing -------------------------------


def _block(x):
    jax.block_until_ready(x)
    return x


def measure(fn, v, x, warmup=100, trials=5, rounds=500):
    # Compile + warm up.
    for _ in range(warmup):
        _block(fn(v, x))
    mins = []
    for _ in range(trials):
        times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            _block(fn(v, x))
            times.append(time.perf_counter() - t0)
        mins.append(min(times) * 1e6)
    return {
        "min_of_mins_us": min(mins),
        "median_of_mins_us": statistics.median(mins),
        "mean_of_mins_us": statistics.mean(mins),
        "trial_mins_us": mins,
    }


# ------------------------------- driver -------------------------------


def run_sweep(sizes, scenarios, trials, rounds):
    rng = np.random.default_rng(0)
    results = {impl: {} for impl in IMPLS}

    for impl_name, impl_fn in IMPLS.items():
        sc = make_scenarios(impl_fn)
        for n in sizes:
            v = jnp.asarray(rng.standard_normal(n))
            # Matvec needs a vector; AD needs tangent / cotangent.
            # build+matvec / build / build+reduce use `x` as rhs or unused.
            x_matvec = jnp.asarray(rng.standard_normal(n))
            x_cotangent = jnp.asarray(rng.standard_normal(n))
            results[impl_name][n] = {}
            for scenario_name, fn in sc.items():
                # Pick the right `x` for this scenario.
                if scenario_name == "build+matvec":
                    x = x_matvec
                elif scenario_name == "jvp":
                    x = x_matvec  # tangent dv
                elif scenario_name == "vjp":
                    x = x_cotangent  # cotangent (matches output shape (n,))
                else:
                    x = x_matvec
                try:
                    r = measure(fn, v, x, trials=trials, rounds=rounds)
                except Exception as e:
                    r = {"error": f"{type(e).__name__}: {e}"}
                results[impl_name][n][scenario_name] = r
                gc.collect()
            jax.clear_caches()
            # Status line.
            cell = results[impl_name][n].get("build", {})
            if "min_of_mins_us" in cell:
                print(f"  {impl_name:18s}  n={n:>6}  build={cell['min_of_mins_us']:8.2f}us")
    return results


def print_table(results, sizes, scenarios):
    impls = list(IMPLS)
    for scenario in scenarios:
        print(f"\n=== {scenario} (µs, min-of-trial-mins) ===")
        header = f'{"n":>6}  ' + "  ".join(f"{i:>16s}" for i in impls)
        print(header)
        for n in sizes:
            row = [f"{n:>6d}"]
            for impl in impls:
                cell = results[impl][n][scenario]
                if "min_of_mins_us" in cell:
                    row.append(f"{cell['min_of_mins_us']:>16.2f}")
                else:
                    row.append(f"{'err':>16s}")
            print("  ".join(row))


def plot_results(results, sizes, scenarios, out_dir, title_suffix):
    colors = {"scatter": "C0", "jnp.diag": "C1", "v*eye": "C2", "where(eye,v,0)": "C3"}
    markers = {"scatter": "o", "jnp.diag": "s", "v*eye": "^", "where(eye,v,0)": "x"}

    for scenario in scenarios:
        fig, (ax_abs, ax_ratio) = plt.subplots(1, 2, figsize=(14, 5))
        # Absolute runtime
        for impl in IMPLS:
            ys = []
            for n in sizes:
                cell = results[impl][n][scenario]
                ys.append(cell.get("min_of_mins_us", float("nan")))
            ax_abs.plot(sizes, ys, marker=markers[impl], color=colors[impl],
                        label=impl, linewidth=1.5)
        ax_abs.set_xscale("log")
        ax_abs.set_yscale("log")
        ax_abs.set_xlabel("n")
        ax_abs.set_ylabel("min runtime (µs)")
        ax_abs.set_title(f"{scenario} — absolute{title_suffix}")
        ax_abs.legend()
        ax_abs.grid(True, alpha=0.3)

        # Ratio vs scatter (baseline)
        for impl in IMPLS:
            if impl == "scatter":
                continue
            ys = []
            for n in sizes:
                base = results["scatter"][n][scenario].get("min_of_mins_us")
                here = results[impl][n][scenario].get("min_of_mins_us")
                ys.append(here / base if base and here else float("nan"))
            ax_ratio.plot(sizes, ys, marker=markers[impl], color=colors[impl],
                          label=f"{impl} / scatter", linewidth=1.5)
        ax_ratio.axhline(1.0, color="k", linestyle="--", alpha=0.3)
        ax_ratio.set_xscale("log")
        ax_ratio.set_xlabel("n")
        ax_ratio.set_ylabel("ratio vs scatter")
        ax_ratio.set_title(f"{scenario} — ratio vs scatter{title_suffix}")
        ax_ratio.legend()
        ax_ratio.grid(True, alpha=0.3)

        fig.tight_layout()
        path = out_dir / f"diag_bench_{scenario.replace('+', '_').replace('(', '').replace(')', '').replace(',', '_').replace(' ', '')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")


# ------------------------------- env capture -------------------------------


def env_info():
    return {
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "eager_constant_folding": bool(jax.config.eager_constant_folding),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy_version": np.__version__,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="50,100,500,1000,5000,10000",
                    help="Comma-separated list of n values.")
    ap.add_argument("--scenarios",
                    default="build,build+reduce,build+matvec,jvp,vjp",
                    help="Scenarios to run. See module docstring.")
    ap.add_argument("--out", default="results/diag_bench",
                    help="Output directory (JSON + plots).")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--folded", action="store_true",
                    help="Run with EAGER_CONSTANT_FOLDING=true (set before import).")
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", True)

    sizes = [int(s) for s in args.sizes.split(",")]
    scenarios = [s.strip() for s in args.scenarios.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = env_info()
    print("=== env ===")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print(f"  sizes: {sizes}")
    print(f"  scenarios: {scenarios}")
    print(f"  trials: {args.trials}, rounds: {args.rounds}")

    print("\n=== running sweep ===")
    results = run_sweep(sizes, scenarios, args.trials, args.rounds)

    print_table(results, sizes, scenarios)

    tag = "folded" if env["eager_constant_folding"] else "unfolded"
    suffix = f" (n_backend={env['jax_backend']}, {tag})"

    print("\n=== plots ===")
    plot_results(results, sizes, scenarios, out_dir, title_suffix=suffix)

    json_path = out_dir / f"diag_bench_{env['jax_backend']}_{tag}.json"
    with json_path.open("w") as f:
        json.dump({"env": env, "sizes": sizes, "scenarios": scenarios,
                   "results": results}, f, indent=2)
    print(f"  wrote {json_path}")


if __name__ == "__main__":
    main()
