"""End-to-end sweep over sif2jax CUTEst problems.

For each problem (bounded minimisation + unconstrained minimisation +
quadratic) with n ≤ `MAX_N` and 1D y0:

1. **Correctness**: `bcoo_jacobian(hvp, y) @ v` matches `hvp(v)` to
   `REL_TOL` on a random probe vector. This avoids materialising the
   dense n×n Hessian for every problem (the sweep is ~200 problems).
2. **nse regression**: `bcoo_jacobian(hvp, y).nse` must not exceed
   the value recorded in `tests/nse_manifest.json` for that problem.
   Decreases are allowed and require a manifest bump
   (`uv run python -m tests.update_nse_manifest`).

Problems that `NotImplementedError` in the walk (missing primitive) are
skipped, not failed — they're a separate tracking concern.

Marked `@pytest.mark.slow` because running all ~200 problems takes a
couple of minutes. Invoke with `pytest -m slow tests/test_sif2jax_sweep.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import sparse

from lineaxpr import bcoo_jacobian


MAX_N = 5000
REL_TOL = 1e-6

_MANIFEST_PATH = Path(__file__).parent / "nse_manifest.json"


def _load_manifest():
    with _MANIFEST_PATH.open() as f:
        data = json.load(f)
    return data.get("problems", {})


try:
    from sif2jax.cutest._bounded_minimisation import (
        bounded_minimisation_problems as B,
    )
    from sif2jax.cutest._quadratic_problems import quadratic_problems as Q
    from sif2jax.cutest._unconstrained_minimisation import (
        unconstrained_minimisation_problems as U,
    )

    _GROUPS = [("unconstrained", U), ("bounded", B), ("quadratic", Q)]
except ImportError:
    _GROUPS = []


def _collect():
    """Yield (group_name, problem_instance, problem_class_name) for every
    sweep-eligible problem."""
    for group, plist in _GROUPS:
        for p in plist:
            yield group, p, p.__class__.__name__


def _id(param):
    _group, _p, name = param
    return name


@pytest.mark.slow
@pytest.mark.skipif(not _GROUPS, reason="sif2jax not installed")
@pytest.mark.parametrize("param", list(_collect()) if _GROUPS else [], ids=_id)
def test_sif2jax_correctness_and_nse(param):
    group, p, name = param
    y = p.y0

    # Size gate: skip huge problems and multi-dim y0 (out of scope).
    if y.ndim != 1:
        pytest.skip(f"multi-dim y0 (ndim={y.ndim})")
    if y.shape[0] > MAX_N:
        pytest.skip(f"n={y.shape[0]} exceeds MAX_N={MAX_N}")

    n = int(y.shape[0])

    def f(z):
        return p.objective(z, p.args)

    try:
        _, hvp = jax.linearize(jax.grad(f), y)
        S = bcoo_jacobian(hvp, y)
    except NotImplementedError as e:
        pytest.skip(f"walk raised: {e}")

    # Correctness via random-vector matvec (avoids O(n²) memory).
    v = jax.random.normal(jax.random.key(0), y.shape, dtype=y.dtype)
    if isinstance(S, sparse.BCOO):
        Sv = S @ v
    else:
        Sv = S @ v  # ndarray fallback; same op
    hvp_v = hvp(v)
    denom = float(jnp.max(jnp.abs(hvp_v))) + 1e-30
    rel_err = float(jnp.max(jnp.abs(Sv - hvp_v))) / denom
    assert rel_err < REL_TOL, (
        f"{name}: Sv vs hvp(v) rel_err={rel_err:.2e} > {REL_TOL:.0e}"
    )

    # nse regression (only meaningful when S is a BCOO — dense returns
    # skip the check).
    if isinstance(S, sparse.BCOO):
        manifest = _load_manifest()
        entry = manifest.get(name)
        if entry is not None:
            recorded_nse = entry["nse"]
            recorded_n = entry["n"]
            if recorded_n != n:
                # Problem default-size may have changed upstream; skip
                # the regression check rather than crying wolf.
                pytest.skip(
                    f"{name}: manifest n={recorded_n} != current n={n} — "
                    f"update the manifest"
                )
            assert S.nse <= recorded_nse, (
                f"{name}: nse regressed {recorded_nse} → {S.nse} (n={n}). "
                f"If this increase is intentional (trade-off justified), "
                f"bump `tests/nse_manifest.json`."
            )
        # Un-manifested problems: silent. Run `update_nse_manifest.py`
        # to capture current state.
