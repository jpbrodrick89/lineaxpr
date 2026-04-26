"""Hand-rolled problems that target specific structural rule paths.

These complement the sif2jax sweep — they exercise corner cases of
walker rules that no real CUTEst problem hits today, locking in
correctness so the rule doesn't regress silently.

Each class duck-types `sif2jax.AbstractUnconstrainedMinimisation`:
`y0`, `args`, and `objective(y, args)`. The sweep keys nse-manifest
entries by class name, so synthetic class names should be prefixed
with `_SYN_` to avoid collision with real CUTEst problems.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class _SYN_SCATTER_COMPACT_DUP:
    """SPARSINE-shaped objective with batched perms whose unique outputs
    form a contiguous range with uniform duplicate factor (dup=2 over
    [0, n/2)). Targets the contiguous + uniform-dup branch of
    `_scatter_add_rule` — none of the real CUTEst problems hit this
    path because their permutations are either bijections (SPARSINE
    k∈{3,7,11}) or strided non-contiguous (SPARSINE k∈{2,5})."""

    def __init__(self, n: int = 100):
        if n % 2 != 0:
            raise ValueError("n must be even for the dup=2 perm")
        self.n = n
        half = n // 2
        # Three batched perms each of length n, each writing dup=2 times to
        # [0, half) — different orderings to keep the linearization rich.
        self._perms = np.stack([
            np.repeat(np.arange(half), 2),                  # 0,0,1,1,...,h-1,h-1
            np.tile(np.arange(half), 2),                    # 0,1,...,h-1, 0,1,...
            np.concatenate([np.arange(half), np.arange(half)]),
        ])

    @property
    def y0(self):
        return jnp.linspace(0.1, 0.5, self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        del args
        sine = jnp.sin(y)
        alpha = sine + sine[self._perms].sum(axis=0)
        return jnp.sum(
            0.5 * jnp.arange(1, self.n + 1, dtype=y.dtype) * alpha**2
        )


SYNTHETIC_PROBLEMS: list = [_SYN_SCATTER_COMPACT_DUP()]
