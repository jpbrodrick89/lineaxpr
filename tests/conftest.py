"""Shared pytest config for the lineaxpr test suite."""

from __future__ import annotations

import jax


# Match sif2jax / benchmark conventions: f64 is the default for all tests.
# CUTEst tolerances (and our bit-exactness claims) assume 64-bit.
jax.config.update("jax_enable_x64", True)
