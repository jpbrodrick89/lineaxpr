"""Cart-pole direct collocation via jaxls with lineaxpr Jacobians.

The cart-pole system:
    state x = [q, dq]  where q = [cart_pos, pole_angle]  (4D)
    control u = [force_on_cart]  (1D)

Equations of motion (simplified, unit masses/lengths):
    ddot_pos  = u - sin(theta)*dot_theta^2 + cos(theta)*sin(theta)*g / denom
    ddot_theta = g*sin(theta) - cos(theta)*(u - sin(theta)*dot_theta^2) / denom
    denom = 1 + sin^2(theta)   (from Lagrangian with m_cart=m_pole=L=1)

where theta = pole_angle, g = 9.81.

Collocation with trapezoidal quadrature over N intervals:
    (x_{k+1} - x_k) / h  =  0.5 * (f(x_k, u_k) + f(x_{k+1}, u_{k+1}))

Variables per node: StateVar (4D), ControlVar (1D).
Cost: regularisation on control effort + boundary penalties.

The interesting structure: collocation residuals couple adjacent (x_k, u_k)
and (x_{k+1}, u_{k+1}).  lineaxpr.jacfwd extracts the Jacobian block
directly from the linearised dynamics.  On CPU (n=20) lineaxpr is currently
~1.8× slower per re-solve than jax.jacfwd; the per-cost Jacobian is a dense
4×10 block so the structural extraction overhead dominates.  The bcoo format
(make_lineaxpr_jac(..., format='bcoo')) may help when the per-cost block is
sparser, or when the tangent dimension is much larger than the residual dimension.

Usage:
    uv run python -m experiments.jaxls.cart_pole
    uv run python -m experiments.jaxls.cart_pole --n 50 --compare  # vs jax.jacfwd
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import jaxls

from experiments.jaxls.bridge import make_lineaxpr_jac

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

G = 9.81


def cart_pole_dynamics(state: jax.Array, u: jax.Array) -> jax.Array:
    """Returns dx/dt = f(state, u) for the cart-pole system.

    state = [cart_pos, pole_angle, cart_vel, pole_angvel]
    u     = [force]  (scalar or shape (1,))
    """
    _pos, theta, vel, omega = state
    force = u[0]
    s, c = jnp.sin(theta), jnp.cos(theta)
    denom = 1.0 + s**2
    ddpos = (force - s * omega**2 + c * s * G) / denom
    ddtheta = (G * s - c * (force - s * omega**2)) / denom
    return jnp.array([vel, omega, ddpos, ddtheta])


# ---------------------------------------------------------------------------
# Variable types
# ---------------------------------------------------------------------------

class StateVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.zeros(4),
):
    pass


class ControlVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.zeros(1),
):
    pass


# ---------------------------------------------------------------------------
# Cost factories
# ---------------------------------------------------------------------------

def _collocation_residual(
    vals: jaxls.VarValues,
    s0: StateVar,
    u0: ControlVar,
    s1: StateVar,
    u1: ControlVar,
    h: float,
) -> jax.Array:
    """Trapezoidal collocation defect: 4D residual."""
    x0, x1 = vals[s0], vals[s1]
    f0 = cart_pole_dynamics(x0, vals[u0])
    f1 = cart_pole_dynamics(x1, vals[u1])
    return (x1 - x0) / h - 0.5 * (f0 + f1)


def _boundary_residual(
    vals: jaxls.VarValues,
    sv: StateVar,
    target: jax.Array,
    weight: float,
) -> jax.Array:
    return weight * (vals[sv] - target)


def _control_residual(
    vals: jaxls.VarValues,
    uv: ControlVar,
    weight: float,
) -> jax.Array:
    return weight * vals[uv]


def _build_costs(n_nodes: int, h: float, use_lineaxpr: bool):
    """Build cost factories, optionally with lineaxpr Jacobians."""
    if use_lineaxpr:
        colloc_jac  = make_lineaxpr_jac(_collocation_residual)
        boundary_jac = make_lineaxpr_jac(_boundary_residual)
        control_jac  = make_lineaxpr_jac(_control_residual)

        @jaxls.Cost.factory(jac_custom_fn=colloc_jac)
        def colloc_cost(vals, s0, u0, s1, u1, h):
            return _collocation_residual(vals, s0, u0, s1, u1, h)

        @jaxls.Cost.factory(jac_custom_fn=boundary_jac)
        def boundary_cost(vals, sv, target, weight):
            return _boundary_residual(vals, sv, target, weight)

        @jaxls.Cost.factory(jac_custom_fn=control_jac)
        def control_cost(vals, uv, weight):
            return _control_residual(vals, uv, weight)
    else:
        @jaxls.Cost.factory
        def colloc_cost(vals, s0, u0, s1, u1, h):
            return _collocation_residual(vals, s0, u0, s1, u1, h)

        @jaxls.Cost.factory
        def boundary_cost(vals, sv, target, weight):
            return _boundary_residual(vals, sv, target, weight)

        @jaxls.Cost.factory
        def control_cost(vals, uv, weight):
            return _control_residual(vals, uv, weight)

    return colloc_cost, boundary_cost, control_cost


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def build_problem(
    n_nodes: int = 20,
    T: float = 2.0,
    use_lineaxpr: bool = True,
) -> tuple[jaxls.AnalyzedLeastSquaresProblem, list, list]:
    h = T / (n_nodes - 1)
    state_vars = [StateVar(i) for i in range(n_nodes)]
    ctrl_vars = [ControlVar(i) for i in range(n_nodes)]

    colloc_cost, boundary_cost, control_cost = _build_costs(n_nodes, h, use_lineaxpr)

    costs = []

    # Collocation constraints (couple adjacent nodes)
    for k in range(n_nodes - 1):
        costs.append(colloc_cost(state_vars[k], ctrl_vars[k],
                                 state_vars[k + 1], ctrl_vars[k + 1], h))

    # Boundary: start at upright rest
    x_start = jnp.array([0.0, 0.0, 0.0, 0.0])
    # Boundary: end at swung-up position (cart shifted, pole upright)
    x_end = jnp.array([1.0, 0.0, 0.0, 0.0])
    costs.append(boundary_cost(state_vars[0],        x_start, weight=100.0))
    costs.append(boundary_cost(state_vars[-1],       x_end,   weight=100.0))

    # Control effort regularisation
    for k in range(n_nodes):
        costs.append(control_cost(ctrl_vars[k], weight=0.1))

    problem = jaxls.LeastSquaresProblem(
        costs=costs,
        variables=state_vars + ctrl_vars,
    ).analyze()

    return problem, state_vars, ctrl_vars


# ---------------------------------------------------------------------------
# Solve and report
# ---------------------------------------------------------------------------

def _timed_solve(problem: jaxls.AnalyzedLeastSquaresProblem, n_warmup: int = 2, n_trials: int = 5):
    """JIT-compile the solve, warm it up, then return timed trial results.

    problem.solve() is decorated with @jdc.jit internally, so:
      call 0   → XLA compile + execute  (excluded from timing)
      calls 1-n_warmup → warm-up executions (excluded from timing)
      calls after that → timed trials
    """
    # Compile
    sol = problem.solve(verbose=False)
    jax.block_until_ready(sol)
    # Warm-up
    for _ in range(n_warmup):
        sol = problem.solve(verbose=False)
        jax.block_until_ready(sol)
    # Timed trials
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        sol = problem.solve(verbose=False)
        jax.block_until_ready(sol)
        times.append(time.perf_counter() - t0)
    return sol, times


def solve_and_report(n_nodes: int = 20, use_lineaxpr: bool = True, label: str = "") -> None:
    tag = label or ("lineaxpr" if use_lineaxpr else "autodiff")
    print(f"\n{'='*60}")
    print(f"  Cart-pole collocation  n_nodes={n_nodes}  [{tag}]")
    print(f"{'='*60}")

    problem, state_vars, ctrl_vars = build_problem(n_nodes, use_lineaxpr=use_lineaxpr)
    sol, times = _timed_solve(problem, n_warmup=2, n_trials=5)

    print(f"  solve  min={min(times)*1e3:.1f}ms  median={sorted(times)[2]*1e3:.1f}ms  (5 trials, 2 warmups)")

    x0 = sol[state_vars[0]]
    xN = sol[state_vars[-1]]
    print(f"  ||x0||_2 = {jnp.linalg.norm(x0):.3e}   "
          f"||xN-[1,0,0,0]||_2 = {jnp.linalg.norm(xN - jnp.array([1.,0.,0.,0.])):.3e}")

    T = 2.0
    h = T / (n_nodes - 1)
    max_defect = max(
        float(jnp.max(jnp.abs(
            _collocation_residual(sol, state_vars[k], ctrl_vars[k],
                                  state_vars[k+1], ctrl_vars[k+1], h)
        )))
        for k in range(n_nodes - 1)
    )
    print(f"  Max collocation defect: {max_defect:.2e}")


def compare_lineaxpr_vs_autodiff(n_nodes: int = 20, n_trials: int = 5) -> None:
    """Solve with both backends; compare timing and solution quality."""
    print(f"\n{'='*60}")
    print(f"  Comparing lineaxpr vs autodiff  n_nodes={n_nodes}  ({n_trials} timed trials, 2 warmups each)")
    print(f"{'='*60}")

    results = {}
    for use_lax, tag in [(True, "lineaxpr"), (False, "autodiff")]:
        problem, state_vars, ctrl_vars = build_problem(n_nodes, use_lineaxpr=use_lax)
        sol, times = _timed_solve(problem, n_warmup=2, n_trials=n_trials)
        results[tag] = (sol, state_vars, ctrl_vars, times)
        times_ms = sorted(t * 1e3 for t in times)
        print(f"  {tag:10s}  min={times_ms[0]:.1f}ms  median={times_ms[len(times_ms)//2]:.1f}ms")

    # Verify solutions agree
    sol_la, sv_la = results["lineaxpr"][:2]
    sol_ad, sv_ad = results["autodiff"][:2]
    diffs = [
        float(jnp.max(jnp.abs(sol_la[sv_la[k]] - sol_ad[sv_ad[k]])))
        for k in range(n_nodes)
    ]
    print(f"  Max state difference lineaxpr vs autodiff: {max(diffs):.2e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of nodes")
    parser.add_argument("--compare", action="store_true",
                        help="Compare lineaxpr vs autodiff")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    if args.compare:
        compare_lineaxpr_vs_autodiff(args.n, args.trials)
    else:
        solve_and_report(args.n, use_lineaxpr=True)
        solve_and_report(args.n, use_lineaxpr=False)
