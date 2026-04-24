"""Developer tool: regenerate `tests/nse_manifest.json` with current nse
values from the sweep.

Run when:
- You've made an intentional nse change (improvement or justified regression)
  and need to update the golden file.
- A problem's default size (`y0.shape`) changed upstream.
- A new sif2jax problem was added.

Output is sorted deterministically so diffs are easy to review.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax


jax.config.update("jax_enable_x64", True)


MAX_N = 5000
MANIFEST_PATH = Path(__file__).parent / "nse_manifest.json"


def _collect():
    """Yield `(group, instance, name)` for each sif2jax problem.

    For problems whose default n exceeds MAX_N but have a SIZE_OVERRIDES
    entry, substitute the smaller variant — its nse replaces the
    default's in the manifest under the same bare class name key.
    """
    from sif2jax.cutest._bounded_minimisation import (
        bounded_minimisation_problems as B,
    )
    from sif2jax.cutest._quadratic_problems import quadratic_problems as Q
    from sif2jax.cutest._unconstrained_minimisation import (
        unconstrained_minimisation_problems as U,
    )

    from tests.test_sif2jax_sweep import SIZE_OVERRIDES

    for group, plist in [("unconstrained", U), ("bounded", B), ("quadratic", Q)]:
        for p in plist:
            name = p.__class__.__name__
            y = p.y0
            if (y.ndim == 1 and y.shape[0] > MAX_N) and name in SIZE_OVERRIDES:
                p = type(p)(**SIZE_OVERRIDES[name])
            yield group, p, name


def main():
    from jax.experimental import sparse

    problems: dict[str, dict] = {}
    skipped_bigger = 0
    skipped_multi = 0
    skipped_error = 0
    dense = 0

    for group, p, name in _collect():
        y = p.y0
        if y.ndim != 1:
            skipped_multi += 1
            continue
        if y.shape[0] > MAX_N:
            skipped_bigger += 1
            continue
        n = int(y.shape[0])

        def f(z, p=p):
            return p.objective(z, p.args)

        try:
            import lineaxpr as _lx
            S = jax.jit(_lx.bcoo_hessian(f))(y)
        except Exception as e:
            skipped_error += 1
            print(f"  skip {name}: {type(e).__name__}: {str(e)[:60]}")
            continue

        if isinstance(S, sparse.BCOO):
            problems[name] = {"n": n, "nse": int(S.nse), "group": group}
        else:
            dense += 1

    # Sort by name for stable diffs.
    sorted_problems = {k: problems[k] for k in sorted(problems)}

    manifest = {
        "_format_version": 1,
        "_note": (
            "nse (number of stored entries) regression manifest for "
            "bcoo_jacobian over sif2jax CUTEst problems. Increases fail "
            "tests. Decreases require a manifest bump — run `uv run "
            "python -m tests.update_nse_manifest` to regenerate."
        ),
        "problems": sorted_problems,
    }

    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    print()
    print(f"Wrote {len(sorted_problems)} problems to {MANIFEST_PATH}")
    print(f"  dense-output (no nse recorded): {dense}")
    print(f"  skipped (n > {MAX_N}): {skipped_bigger}")
    print(f"  skipped (multi-dim y0): {skipped_multi}")
    print(f"  skipped (walk error): {skipped_error}")


if __name__ == "__main__":
    main()
