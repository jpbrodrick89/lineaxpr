"""Signal denoising — lineaxpr vs autodiff on large sparse Jacobians.

A single SignalVar of size N is denoised by minimising:

  data fidelity   ||x - y_obs||²         Jacobian: N×N identity
  smoothness      ||D2 @ x||²             Jacobian: (N-2)×N, 3-band BEllpack

With N=200, the smoothness Jacobian is 198×200 but has only 594 nonzero
entries (3 bands).  lineaxpr extracts the BEllpack in one structural pass.
jax.jacfwd (jaxls default) needs N=200 forward passes over the same function.

lineaxpr wins come from the mismatch between matrix size and nonzero count:
  N=200  → 198×200 = 39 600 elements  vs  594 nonzero  (66× sparse ratio)
  N=500  → 498×500 = 249 000 elements vs 1494 nonzero (167× sparse ratio)

The data-fidelity Jacobian is weight*I — lineaxpr extracts it as
ConstantDiagonal in 1 pass (N×N matrix, N nonzero).

Run:
    uv run python -m experiments.jaxls.signal_denoising
    uv run python -m experiments.jaxls.signal_denoising --n 500 --trials 8
"""

from __future__ import annotations

import time
import argparse

import jax
import jax.numpy as jnp
import jaxls

from experiments.jaxls.bridge import make_lineaxpr_jac

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Variable and residuals
# ---------------------------------------------------------------------------

def make_signal_var(n: int):
    return type("SignalVar", (jaxls.Var[jax.Array],), {}, default_factory=lambda: jnp.zeros(n))


def _data_residual(
    vals: jaxls.VarValues,
    sv,
    y_obs: jax.Array,
    weight: float,
) -> jax.Array:
    """Data fidelity: weight*(x - y_obs).  Jacobian = weight*I_N."""
    return weight * (vals[sv] - y_obs)


def _smooth_residual(
    vals: jaxls.VarValues,
    sv,
    weight: float,
) -> jax.Array:
    """Second-difference smoothness.  Jacobian: (N-2)×N, 3-band BEllpack."""
    x = vals[sv]
    return weight * (x[:-2] - 2.0 * x[1:-1] + x[2:])


# ---------------------------------------------------------------------------
# Problem builder
# ---------------------------------------------------------------------------

def build_problem(n: int = 200, use_lineaxpr: bool = True, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    x_true = jnp.cumsum(jax.random.normal(k1, (n,))) * 0.1
    y_obs  = x_true + jax.random.normal(k2, (n,)) * 0.5

    SignalVar = make_signal_var(n)
    sv = SignalVar(0)

    smooth_w = 1.0
    data_w   = 2.0

    if use_lineaxpr:
        data_jac   = make_lineaxpr_jac(_data_residual)
        smooth_jac = make_lineaxpr_jac(_smooth_residual)

        @jaxls.Cost.factory(jac_custom_fn=data_jac)
        def data_cost(vals, sv, y_obs, weight):
            return _data_residual(vals, sv, y_obs, weight)

        @jaxls.Cost.factory(jac_custom_fn=smooth_jac)
        def smooth_cost(vals, sv, weight):
            return _smooth_residual(vals, sv, weight)
    else:
        @jaxls.Cost.factory
        def data_cost(vals, sv, y_obs, weight):
            return _data_residual(vals, sv, y_obs, weight)

        @jaxls.Cost.factory
        def smooth_cost(vals, sv, weight):
            return _smooth_residual(vals, sv, weight)

    costs = [
        data_cost(sv, y_obs, data_w),
        smooth_cost(sv, smooth_w),
    ]

    initial_vals = jaxls.VarValues.make([sv.with_value(y_obs)])
    problem = jaxls.LeastSquaresProblem(costs=costs, variables=[sv]).analyze()
    return problem, sv, initial_vals, y_obs


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _timed_solve(problem, initial_vals, n_warmup: int = 2, n_trials: int = 5):
    sol = problem.solve(initial_vals=initial_vals, verbose=False)
    jax.block_until_ready(sol)
    for _ in range(n_warmup):
        sol = problem.solve(initial_vals=initial_vals, verbose=False)
        jax.block_until_ready(sol)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        sol = problem.solve(initial_vals=initial_vals, verbose=False)
        jax.block_until_ready(sol)
        times.append(time.perf_counter() - t0)
    return sol, times


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(n: int = 200, n_trials: int = 5):
    print(f"\n{'='*62}")
    print(f"  Signal denoising  n={n}  ({n_trials} trials, 2 warmups)")
    print(f"  smooth Jacobian: {n-2}×{n}  data Jacobian: {n}×{n}")
    print(f"  smooth nonzeros: {3*(n-2)}  (vs {(n-2)*n} dense elements)")
    print(f"{'='*62}")

    results = {}

    for use_lax, tag in [(False, "autodiff"), (True, "lineaxpr")]:
        prob, sv, iv, y_obs = build_problem(n, use_lineaxpr=use_lax)
        sol, times = _timed_solve(prob, iv, n_trials=n_trials)
        ms = sorted(t * 1e3 for t in times)
        x_sol = sol[sv]
        residual = float(jnp.mean((x_sol - y_obs) ** 2))
        print(f"  {tag:10s}  min={ms[0]:.2f}ms  median={ms[len(ms)//2]:.2f}ms  "
              f"MSE(vs obs)={residual:.4f}")
        results[tag] = (x_sol, times)

    # verify solutions agree
    x_lax, x_ad = results["lineaxpr"][0], results["autodiff"][0]
    print(f"  max |Δx| lineaxpr vs autodiff: {float(jnp.max(jnp.abs(x_lax - x_ad))):.2e}")

    ad_min  = min(results["autodiff"][2]) * 1e3
    lax_min = min(results["lineaxpr"][2]) * 1e3
    ratio = ad_min / lax_min
    direction = f"{ratio:.2f}× faster" if ratio > 1 else f"{1/ratio:.2f}× slower"
    print(f"\n  lineaxpr vs autodiff (min time): {direction}  ({lax_min:.2f}ms vs {ad_min:.2f}ms)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    compare(n=args.n, n_trials=args.trials)
