"""Coverage with correctness verification, capped at MAX_N."""

import sys
import time
from collections import Counter

import jax
import jax.numpy as jnp

from lineaxpr import materialize  # noqa: E402

from sif2jax.cutest._bounded_minimisation import (  # noqa: E402
    bounded_minimisation_problems as B,
)
from sif2jax.cutest._quadratic_problems import (  # noqa: E402
    quadratic_problems as Q,
)
from sif2jax.cutest._unconstrained_minimisation import (  # noqa: E402
    unconstrained_minimisation_problems as U,
)


MAX_N = 5000


def main():
    fail_reasons: Counter = Counter()
    correctness_fails: list = []
    ok = 0
    total = 0
    skipped = 0

    with open("/tmp/coverage_v2.txt", "w") as out:
        for group_name, plist in [("unconstrained", U), ("bounded", B), ("quadratic", Q)]:
            out.write(f"=== {group_name} ({len(plist)}) ===\n")
            for p in plist:
                total += 1
                name = p.__class__.__name__
                try:
                    y = p.y0
                    if y.ndim != 1:
                        out.write(f"  {name}: SKIP multi-dim\n")
                        skipped += 1
                        continue
                    if y.shape[0] > MAX_N:
                        out.write(f"  {name}: SKIP n={y.shape[0]}\n")
                        skipped += 1
                        continue

                    def f(z):
                        return p.objective(z, p.args)

                    _, hvp = jax.linearize(jax.grad(f), y)
                    H = materialize(hvp, y)
                    # Correctness: H @ v should match hvp(v) for a random v.
                    v = jnp.array(jax.random.normal(jax.random.key(0), y.shape, dtype=y.dtype))
                    err = float(jnp.max(jnp.abs(H @ v - hvp(v))))
                    rel = err / (float(jnp.max(jnp.abs(hvp(v)))) + 1e-30)
                    ok += 1
                    if rel > 1e-6:
                        correctness_fails.append((name, rel))
                        out.write(f"  {name}: WRONG rel_err={rel:.2e}\n")
                except NotImplementedError as e:
                    msg = str(e)
                    key = msg.replace("No materialize rule for primitive ", "NIE: ")[:80]
                    fail_reasons[key] += 1
                    out.write(f"  {name}: {key}\n")
                except Exception as e:
                    key = f"{type(e).__name__}: {str(e)[:60]}"
                    fail_reasons[key] += 1
                    out.write(f"  {name}: {key}\n")
                out.flush()

        attempted = total - skipped
        out.write(f"\n=== Summary ===\n")
        out.write(f"OK: {ok}/{attempted} attempted ({100*ok/max(attempted, 1):.1f}%)\n")
        out.write(f"SKIPPED: {skipped} (n > {MAX_N} or multi-dim)\n")
        out.write(f"Correctness failures: {len(correctness_fails)}\n")
        for n_, r_ in correctness_fails:
            out.write(f"  {n_}: rel_err={r_:.2e}\n")
        out.write(f"\nFail reasons (top):\n")
        for r, c in fail_reasons.most_common(20):
            out.write(f"  [{c:3d}] {r}\n")

    print("done")


if __name__ == "__main__":
    main()
