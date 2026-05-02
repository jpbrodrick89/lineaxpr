"""Path planning with obstacle avoidance — lineaxpr vs autodiff Jacobians.

50 waypoints on a 2D path from [0,0] to [4,0], two circular obstacles.

Costs
-----
  smoothness (l2_squared)     wp[k+1] - wp[k]                 residual 2D, tangent 4D
  fix_waypoint (eq_zero)      wp[0/N-1] - target              residual 2D, tangent 2D
  obstacle_avoidance (leq_0)  r² - ||wp - centre||²           residual 1D, tangent 2D

Sparsity story
--------------
Each cost touches 1 or 2 of 50 variables (100D tangent space total).
The smoothness residual is wp[k+1] - wp[k] which linearises to [I, -I]
— a BEllpack with two ±Identity bands that lineaxpr extracts in one pass.
The obstacle term linearises to -2*(wp-c) — a 1×2 dense row.

For the smoothness cost jaxls' auto mode picks jacrev (2D residual < 4D
tangent → 2 reverse passes).  lineaxpr does 1 linearise + structural walk.

Note: jaxls requires dense arrays from jac_custom_fn; format='bcoo' is not
supported here.  lineaxpr still uses its structural walk internally; the
output is densified before being handed to jaxls.

Run:
    uv run python -m experiments.jaxls.obstacle_avoidance
    uv run python -m experiments.jaxls.obstacle_avoidance --n 200 --trials 8
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
# Variable
# ---------------------------------------------------------------------------

class WaypointVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)):
    """2D waypoint position (Euclidean)."""


# ---------------------------------------------------------------------------
# Residual functions (shared between lineaxpr and autodiff builds)
# ---------------------------------------------------------------------------

def _smoothness(vals: jaxls.VarValues, wp1: WaypointVar, wp2: WaypointVar) -> jax.Array:
    return vals[wp1] - vals[wp2]


def _fix_waypoint(vals: jaxls.VarValues, wp: WaypointVar, target: jax.Array) -> jax.Array:
    return vals[wp] - target


def _obstacle(
    vals: jaxls.VarValues,
    wp: WaypointVar,
    centre: jax.Array,
    radius: jax.Array,
) -> jax.Array:
    dist_sq = jnp.sum((vals[wp] - centre) ** 2)
    return jnp.array([radius**2 - dist_sq])


# ---------------------------------------------------------------------------
# Problem builder
# ---------------------------------------------------------------------------

OBSTACLES = [
    (jnp.array([1.0, 0.3]),  0.6),
    (jnp.array([3.0, -0.3]), 0.6),
]

START = jnp.array([0.0, 0.0])
GOAL  = jnp.array([4.0, 0.0])


def build_problem(n: int = 50, use_lineaxpr: bool = True):
    waypoints = [WaypointVar(i) for i in range(n)]

    if use_lineaxpr:
        smooth_jac   = make_lineaxpr_jac(_smoothness)
        fix_jac      = make_lineaxpr_jac(_fix_waypoint)
        obstacle_jac = make_lineaxpr_jac(_obstacle)

        @jaxls.Cost.factory(jac_custom_fn=smooth_jac)
        def smoothness(vals, wp1, wp2):
            return _smoothness(vals, wp1, wp2)

        @jaxls.Cost.factory(jac_custom_fn=fix_jac, kind="constraint_eq_zero")
        def fix_waypoint(vals, wp, target):
            return _fix_waypoint(vals, wp, target)

        @jaxls.Cost.factory(jac_custom_fn=obstacle_jac, kind="constraint_leq_zero")
        def obstacle(vals, wp, centre, radius):
            return _obstacle(vals, wp, centre, radius)
    else:
        @jaxls.Cost.factory
        def smoothness(vals, wp1, wp2):
            return _smoothness(vals, wp1, wp2)

        @jaxls.Cost.factory(kind="constraint_eq_zero")
        def fix_waypoint(vals, wp, target):
            return _fix_waypoint(vals, wp, target)

        @jaxls.Cost.factory(kind="constraint_leq_zero")
        def obstacle(vals, wp, centre, radius):
            return _obstacle(vals, wp, centre, radius)

    costs = []
    for k in range(n - 1):
        costs.append(smoothness(waypoints[k], waypoints[k + 1]))
    costs.append(fix_waypoint(waypoints[0],    START))
    costs.append(fix_waypoint(waypoints[-1],   GOAL))
    for k in range(1, n - 1):
        for centre, radius in OBSTACLES:
            costs.append(obstacle(waypoints[k], centre, jnp.array(radius)))

    # Straight-line initialisation
    initial_vals = jaxls.VarValues.make([
        wp.with_value(START + (GOAL - START) * i / (n - 1))
        for i, wp in enumerate(waypoints)
    ])

    problem = jaxls.LeastSquaresProblem(costs=costs, variables=waypoints).analyze()
    return problem, waypoints, initial_vals


# ---------------------------------------------------------------------------
# Timing helper (same pattern as cart_pole)
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
# Quality check
# ---------------------------------------------------------------------------

def _check_solution(sol, waypoints):
    n = len(waypoints)
    # Boundary
    err0 = float(jnp.linalg.norm(sol[waypoints[0]]  - START))
    errN = float(jnp.linalg.norm(sol[waypoints[-1]] - GOAL))
    # Obstacle clearances
    min_clearance = float("inf")
    for k in range(1, n - 1):
        pos = sol[waypoints[k]]
        for centre, radius in OBSTACLES:
            clearance = float(jnp.linalg.norm(pos - centre)) - radius
            min_clearance = min(min_clearance, clearance)
    return err0, errN, min_clearance


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare(n: int = 50, n_trials: int = 5):
    print(f"\n{'='*62}")
    print(f"  Obstacle avoidance  n_waypoints={n}  ({n_trials} trials, 2 warmups)")
    print(f"{'='*62}")

    results = {}

    # autodiff baseline
    prob, wps, iv = build_problem(n, use_lineaxpr=False)
    sol, times = _timed_solve(prob, iv, n_trials=n_trials)
    err0, errN, clearance = _check_solution(sol, wps)
    ms = sorted(t * 1e3 for t in times)
    print(f"  {'autodiff':12s}  min={ms[0]:.2f}ms  median={ms[len(ms)//2]:.2f}ms  "
          f"clearance={clearance:.3f}")
    results["autodiff"] = (sol, wps, times)

    # lineaxpr
    prob, wps, iv = build_problem(n, use_lineaxpr=True)
    sol_lax, times_lax = _timed_solve(prob, iv, n_trials=n_trials)
    err0_lax, errN_lax, clearance_lax = _check_solution(sol_lax, wps)
    ms_lax = sorted(t * 1e3 for t in times_lax)
    print(f"  {'lineaxpr':12s}  min={ms_lax[0]:.2f}ms  median={ms_lax[len(ms_lax)//2]:.2f}ms  "
          f"clearance={clearance_lax:.3f}")
    results["lineaxpr"] = (sol_lax, wps, times_lax)

    # verify solutions agree
    ad_sol, ad_wps = results["autodiff"][:2]
    diffs = [float(jnp.max(jnp.abs(sol_lax[wps[k]] - ad_sol[ad_wps[k]])))
             for k in range(n)]
    print(f"  {'':12s}  max |Δwp| vs autodiff: {max(diffs):.2e}")

    # ratio summary
    ad_min = min(results["autodiff"][2]) * 1e3
    lax_min = min(results["lineaxpr"][2]) * 1e3
    ratio = ad_min / lax_min
    direction = f"{ratio:.2f}× faster" if ratio > 1 else f"{1/ratio:.2f}× slower"
    print(f"\n  lineaxpr vs autodiff (min time): {direction}  ({lax_min:.2f}ms vs {ad_min:.2f}ms)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    compare(n=args.n, n_trials=args.trials)
