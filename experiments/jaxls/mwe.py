"""Minimum working example: lineaxpr Jacobian inside jaxls.

Two problems of increasing complexity:

1. Linear regression (linear residual, Jacobian is constant A).
2. Nonlinear least squares (residual = sin(x) - target, Jacobian = diag(cos(x))).

Both verify that the lineaxpr-supplied Jacobian matches jax.jacfwd numerically
and that the solver converges to the correct solution.

Run:
    uv run python -m experiments.jaxls.mwe
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxls

from experiments.jaxls.bridge import make_lineaxpr_jac

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Variable type: a flat n-vector in Euclidean space
# ---------------------------------------------------------------------------

def make_var_type(n: int, name: str = "Var"):
    """Dynamically create a jaxls Var subclass for an n-vector."""
    return type(
        name,
        (jaxls.Var[jax.Array],),
        {},
        default_factory=lambda: jnp.zeros(n),
    )


# ---------------------------------------------------------------------------
# Problem 1: linear regression  Ax = b
# ---------------------------------------------------------------------------

def run_linear_regression(n: int = 4, seed: int = 0) -> None:
    print("=" * 60)
    print("Problem 1: linear regression  Ax = b")
    print(f"  n = {n}")

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (n, n))
    b = jax.random.normal(k2, (n,))
    x_true = jnp.linalg.solve(A, b)

    XVar = make_var_type(n, "XVar")
    x_var = XVar(0)

    def residual(vals: jaxls.VarValues, var: XVar, A: jax.Array, b: jax.Array):
        return A @ vals[var] - b

    jac_fn = make_lineaxpr_jac(residual)
    ref_jac = jax.jacfwd(lambda x: A @ x - b)(jnp.zeros(n))
    test_jac = jac_fn(jaxls.VarValues.make([x_var.with_value(jnp.zeros(n))]), x_var, A, b)

    print(f"  Jacobian matches jax.jacfwd: {jnp.allclose(test_jac, ref_jac, atol=1e-10)}")
    print(f"  Jacobian == A:               {jnp.allclose(test_jac, A, atol=1e-10)}")

    @jaxls.Cost.factory(jac_custom_fn=jac_fn)
    def linear_cost(vals, var, A, b):
        return residual(vals, var, A, b)

    problem = jaxls.LeastSquaresProblem(
        costs=[linear_cost(x_var, A, b)],
        variables=[x_var],
    ).analyze()
    sol = problem.solve(verbose=False)
    x_sol = sol[x_var]

    print(f"  ||x_sol - x_true||_inf = {jnp.max(jnp.abs(x_sol - x_true)):.2e}")
    assert jnp.allclose(x_sol, x_true, atol=1e-6), "Linear regression failed"
    print("  PASS")


# ---------------------------------------------------------------------------
# Problem 2: nonlinear least squares   sin(x) = target
# ---------------------------------------------------------------------------

def run_nonlinear_lsq(n: int = 6, seed: int = 1) -> None:
    print("=" * 60)
    print("Problem 2: nonlinear least squares  sin(x) = target")
    print(f"  n = {n}")

    key = jax.random.PRNGKey(seed)
    # Targets in (-1, 1) so arcsin is well-defined
    target = jax.random.uniform(key, (n,), minval=-0.9, maxval=0.9)
    x_true = jnp.arcsin(target)

    XVar = make_var_type(n, "SinVar")
    x_var = XVar(0)
    x0 = jnp.zeros(n)

    def residual(vals: jaxls.VarValues, var, target: jax.Array):
        return jnp.sin(vals[var]) - target

    jac_fn = make_lineaxpr_jac(residual)

    # Verify Jacobian at x0: should be diag(cos(x0)) = I  (cos(0) = 1)
    ref_jac = jax.jacfwd(lambda x: jnp.sin(x) - target)(x0)
    test_jac = jac_fn(jaxls.VarValues.make([x_var.with_value(x0)]), x_var, target)
    print(f"  Jacobian matches jax.jacfwd at x0: {jnp.allclose(test_jac, ref_jac, atol=1e-10)}")

    # Verify Jacobian at x_true: should be diag(cos(arcsin(target)))
    ref_jac2 = jax.jacfwd(lambda x: jnp.sin(x) - target)(x_true)
    test_jac2 = jac_fn(jaxls.VarValues.make([x_var.with_value(x_true)]), x_var, target)
    print(f"  Jacobian matches jax.jacfwd at x_true: {jnp.allclose(test_jac2, ref_jac2, atol=1e-10)}")

    @jaxls.Cost.factory(jac_custom_fn=jac_fn)
    def sin_cost(vals, var, target):
        return residual(vals, var, target)

    problem = jaxls.LeastSquaresProblem(
        costs=[sin_cost(x_var, target)],
        variables=[x_var],
    ).analyze()
    sol = problem.solve(verbose=False)
    x_sol = sol[x_var]

    print(f"  ||x_sol - x_true||_inf = {jnp.max(jnp.abs(x_sol - x_true)):.2e}")
    assert jnp.allclose(x_sol, x_true, atol=1e-5), "Nonlinear LSQ failed"
    print("  PASS")


# ---------------------------------------------------------------------------
# Problem 3: two-variable cost  r(x, y) = x + y - target
# ---------------------------------------------------------------------------

def run_two_variable(n: int = 3, seed: int = 2) -> None:
    print("=" * 60)
    print("Problem 3: two-variable cost  x + y = target")
    print(f"  n = {n}")

    key = jax.random.PRNGKey(seed)
    target = jax.random.normal(key, (n,))
    # Under-determined: fix one variable with a prior, solve for the other

    XVar = make_var_type(n, "TwoVarX")
    YVar = make_var_type(n, "TwoVarY")
    x_var = XVar(0)
    y_var = YVar(0)

    def sum_residual(vals: jaxls.VarValues, xv: XVar, yv: YVar, target: jax.Array):
        return vals[xv] + vals[yv] - target

    def prior_x(vals: jaxls.VarValues, xv: XVar):
        return vals[xv]

    jac_sum = make_lineaxpr_jac(sum_residual)
    jac_prior = make_lineaxpr_jac(prior_x)

    @jaxls.Cost.factory(jac_custom_fn=jac_sum)
    def coupling_cost(vals, xv, yv, target):
        return sum_residual(vals, xv, yv, target)

    @jaxls.Cost.factory(jac_custom_fn=jac_prior)
    def prior_cost(vals, xv):
        return prior_x(vals, xv)

    problem = jaxls.LeastSquaresProblem(
        costs=[coupling_cost(x_var, y_var, target), prior_cost(x_var)],
        variables=[x_var, y_var],
    ).analyze()
    sol = problem.solve(verbose=False)
    x_sol, y_sol = sol[x_var], sol[y_var]

    residual_norm = jnp.max(jnp.abs(x_sol + y_sol - target))
    print(f"  ||x + y - target||_inf = {residual_norm:.2e}")
    assert residual_norm < 1e-5, "Two-variable problem failed"
    print("  PASS")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_linear_regression()
    run_nonlinear_lsq()
    run_two_variable()
    print("=" * 60)
    print("All MWE tests passed.")
