"""End-to-end sweep over sif2jax CUTEst problems.

For each problem (bounded minimisation + unconstrained minimisation +
quadratic) with n ≤ `MAX_N` and 1D y0:

1. **Correctness**: `materialize(hvp, y, format='bcoo') @ v` matches
   `hvp(v)` to `REL_TOL` on a random probe vector. This avoids
   materialising the dense n×n Hessian for every problem (the sweep is
   ~200 problems).
2. **nse regression**: `.nse` of the BCOO Hessian must not exceed
   the value recorded in `tests/nse_manifest.json` for that problem.
   Decreases are allowed and require a manifest bump
   (`uv run python -m tests.update_nse_manifest`).

Problems that `NotImplementedError` in the walk (missing primitive) are
registered in `KNOWN_UNIMPLEMENTED` and explicitly skipped with a reason
— a silent try/except around the walk used to hide regressions when a
previously-working problem started hitting an unsupported primitive
path.

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

import lineaxpr


MAX_N = 5000
REL_TOL = 1e-6

# Problems whose walk raises `NotImplementedError` because a primitive
# we haven't implemented structurally appears in the linearized jaxpr.
# Listed explicitly so unexpected `NotImplementedError`s from other
# problems fail the test rather than silently skip — surfacing the
# regression instead of hiding it.
#
# Format: problem class name → short reason string.
KNOWN_UNIMPLEMENTED: dict[str, str] = {
    # (none currently — HADAMALS was covered by the cond + 2D-gather
    # rules in bc46c45. Populate as new problems hit new primitives.)
}

# Constructor kwargs for smaller variants of problems whose default size
# exceeds `MAX_N`. Used purely for correctness coverage — we don't run
# the nse-manifest regression check on overridden variants (nse scales
# with n and the manifest is keyed by default class name).
#
# Only problems with a size-parameter constructor are listed. Others
# (e.g. CVXQP1 which has only y0_iD as init-arg) keep skipping.
# Not all of these have been verified against each problem's actual
# signature — they reflect documented defaults. If a constructor kwarg
# is wrong, the instantiation raises `TypeError` at collection time
# and we fall back to skipping via the MAX_N gate.
SIZE_OVERRIDES: dict[str, dict] = {
    # Class : kwargs for constructor (must be a valid smaller variant)
    "BOX":       {"n": 500},
    "COSINE":    {"n": 500},
    "CURLY10":   {"n": 500, "k": 10},
    "CURLY20":   {"n": 500, "k": 20},
    "CURLY30":   {"n": 500, "k": 30},
    "DIXON3DQ":  {"n": 500},
    "POWER":     {"n": 500},
    "INDEFM":    {"n": 500},
    "YATP1LS":   {"N": 20},   # n = N*(N+2) = 440
    "YATP1CLS":  {"N": 20},
    # Torsion family: n = q*q
    "TORSION1":  {"q": 20}, "TORSION2":  {"q": 20},
    "TORSION3":  {"q": 20}, "TORSION4":  {"q": 20},
    "TORSION5":  {"q": 20}, "TORSION6":  {"q": 20},
    "TORSIONA":  {"q": 20}, "TORSIONB":  {"q": 20},
    "TORSIONC":  {"q": 20}, "TORSIOND":  {"q": 20},
    "TORSIONE":  {"q": 20}, "TORSIONF":  {"q": 20},
}

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
    """Yield `(group, instance, pytest_id, is_override)` for every
    sweep-eligible problem.

    For problems whose default `y0.shape[0]` exceeds `MAX_N` but have a
    `SIZE_OVERRIDES` entry, yield an additional smaller-n variant with
    `is_override=True`; that variant runs the correctness check only,
    skipping the nse manifest regression (which is keyed by default
    size). The original (too-big) problem is still yielded and hits the
    MAX_N skip, so it stays visible in the skip count.
    """
    for group, plist in _GROUPS:
        for p in plist:
            name = p.__class__.__name__
            yield group, p, name, False
            # Add a smaller variant when the default is too big AND we
            # know how to construct a smaller one.
            if name in SIZE_OVERRIDES:
                y = p.y0
                if y.ndim == 1 and y.shape[0] > MAX_N:
                    try:
                        small = type(p)(**SIZE_OVERRIDES[name])
                        yield group, small, f"{name}[override]", True
                    except TypeError:
                        # Kwargs don't match this class' constructor —
                        # silently fall through (the too-big one will
                        # skip via the MAX_N gate).
                        pass


def _id(param):
    _group, _p, pid, _is_override = param
    return pid


@pytest.mark.slow
@pytest.mark.skipif(not _GROUPS, reason="sif2jax not installed")
@pytest.mark.parametrize("param", list(_collect()) if _GROUPS else [], ids=_id)
def test_sif2jax_correctness_and_nse(param):
    group, p, pid, is_override = param
    name = p.__class__.__name__
    y = p.y0

    # Size gate: skip huge problems and multi-dim y0 (out of scope).
    if y.ndim != 1:
        pytest.skip(f"multi-dim y0 (ndim={y.ndim})")
    if y.shape[0] > MAX_N:
        pytest.skip(f"n={y.shape[0]} exceeds MAX_N={MAX_N}")

    n = int(y.shape[0])

    if name in KNOWN_UNIMPLEMENTED:
        pytest.skip(f"known-unimplemented: {KNOWN_UNIMPLEMENTED[name]}")

    def f(z):
        return p.objective(z, p.args)

    # NotImplementedError is NOT caught here — if a walk that previously
    # worked now raises, we want the test to FAIL (so a rule regression
    # shows up), not silently skip.
    S = lineaxpr.bcoo_hessian(f)(y)

    # Correctness via random-vector matvec (avoids O(n²) memory).
    # Compare S @ v against hvp(v) where hvp is jax's linearized gradient.
    _, hvp = jax.linearize(jax.grad(f), y)
    v = jax.random.normal(jax.random.key(0), y.shape, dtype=y.dtype)
    Sv = S @ v  # works for both BCOO and ndarray
    hvp_v = hvp(v)
    denom = float(jnp.max(jnp.abs(hvp_v))) + 1e-30
    rel_err = float(jnp.max(jnp.abs(Sv - hvp_v))) / denom
    assert rel_err < REL_TOL, (
        f"{name}: Sv vs hvp(v) rel_err={rel_err:.2e} > {REL_TOL:.0e}"
    )

    # nse regression (only meaningful when S is a BCOO — dense returns
    # skip the check). Also skip for size-overridden variants — the
    # manifest nse is keyed by default-n, which won't match the
    # smaller-variant's nse.
    if is_override:
        return
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
