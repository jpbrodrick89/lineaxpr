# lineaxpr — notes for Claude

## What this is

A JAX transformation that extracts dense / sparse Hessian/Jacobian matrices
from linear callables (the output of `jax.linearize`). Walks the linear
jaxpr once with per-primitive structural rules.

Public API (jax-like, preferred):

- `hessian(f)(y)`, `bcoo_hessian(f)(y)` — matches `jax.hessian`.
- `jacfwd(f)(y)`, `bcoo_jacfwd(f)(y)` — matches `jax.jacfwd`.
- `jacrev(f)(y)`, `bcoo_jacrev(f)(y)` — matches `jax.jacrev`.
  All six accept `format='dense'|'bcoo'` as a kwarg too; the
  `bcoo_`-prefixed variants are shorthand for `format='bcoo'`.

Lower-level building blocks:

- `materialize(linear_fn, primal, format='dense'|'bcoo')` — core
  entry point. Default `'dense'`.
- `sparsify(linear_fn)(seed_linop) -> LinOp` — primitive transform.
  No implicit Identity cast; caller provides the seed.
- `to_dense(linop)` / `to_bcoo(linop)` — format conversion helpers
  that work uniformly on LinOp classes, BCOO, and plain ndarrays.
- `Identity(n, dtype=...)` — standard seed for `sparsify`.
- `ConstantDiagonal`, `Diagonal`, `BEllpack` — LinOp classes (exposed
  for tests/debugging and custom seeds; not a pytree-registered API).

Current limitations (TODO #9c/#9d): no argnums, no has_aux, single
input, single output, 1D primal. `jax.vmap` composition not yet
verified.

## Testing philosophy

Tests should call the public jax-like API (`hessian`, `jacfwd`,
`jacrev`, and `bcoo_*` variants) wherever possible. Reserve direct
`materialize` / `sparsify` calls for:

- `tests/test_ops.py` — LinOp class methods (unit).
- `tests/test_sparsify.py` — transform-level behavior (custom seeds,
  const-prop, missing-primitive errors). `sparsify` IS the thing under
  test; public API would just be indirection.
- `tests/test_public_api.py` — unit tests of the `format` kwarg,
  invalid-format error, default-dense behavior.

Everything else (Hessian correctness, sweep, hand-rolled problems)
should go through the public API since that's how users exercise the
library.

## Non-goals

- Coloring-based extraction (that's asdex's approach; we do per-linearization
  structural walk instead, yielding exact — not conservative — sparsity).
- Second AD pass (we stay inside `jax.linearize`'s output).
- A full linear-operator algebra library like lineax (name inspired by it,
  but we're narrower in scope).
- Migrating onto `jax.experimental.sparse.sparsify`'s framework — see
  `docs/RESEARCH_NOTES.md` §10. We mirror its patterns (direct-assignment
  rule registry, single-env walk) but keep our own mixed-format space.

## Key invariants

- The walk produces **bit-exact** output (vs `vmap(hvp)(eye)` reference).
  Any rule change must preserve this. `tests/test_correctness.py` is the
  ground truth; `tests/test_materialize.py` pins explicit expected
  matrices for hand-rolled problems.
- LinOp forms live in `lineaxpr/_base.py` with methods `.todense()`,
  `.to_bcoo()`, `.negate()`, `.scale_scalar(s)`, `.scale_per_out_row(v)`,
  `.primal_aval()` (and `BEllpack.pad_rows(before, after)`). Rules dispatch to
  these methods rather than branching on LinOp type inside every rule.
- Primitive rules are registered by direct dict assignment:
  `materialize_rules[lax.PRIM_p] = _rule_fn`. Each rule receives
  `(invals, traced, n, **params)` and returns a LinOp (or list for
  multi-result primitives). No `@register` decorator — it was ceremony.
- Walk env is `dict[Var, tuple[bool, Any]]`. The `bool` is `traced`; the
  `Any` is a LinOp (traced) or concrete array (closure). Walker emits
  `(False, value)` via constant-propagation for eqns with all-closure
  inputs — load-bearing for constant-H problems (DUAL, CMPC) where it
  produces a trace-time BCOO literal.
- nse regression is pinned in `tests/nse_manifest.json` — increases in
  `bcoo_jacobian(...).nse` fail the sweep; decreases require a manifest
  bump via `uv run python -m tests.update_nse_manifest`.

## Benchmarks

Run with the docker runner (see `benchmarks/run_in_container.sh`) to get
the EAGER_CONSTANT_FOLDING=TRUE regime that matches JAX's release config.

**Critical**: always pass the linearization point `y` as a `jit` INPUT,
never a closure. Otherwise `jax.hessian`'s output gets constant-folded and
spurious speedups appear. See `docs/RESEARCH_NOTES.md` §10 for the
full picture on how EAGER_CONSTANT_FOLDING + closure-y interact.

**Also critical**: macOS-native unfolded full-sweep runs have ~3×
cross-problem contamination on dense-pattern problems (TABLE8, EXPLIN,
EG2). Don't cite regression numbers from
`.benchmarks/Darwin-*/*_full.json` smaller than ~3×. For clean unfolded
numbers use `USE_CONTAINER=1 NO_EAGER=1 bash benchmarks/run_bench.sh
--full` (saves under `.benchmarks/Linux-*/`). Folded container runs are
clean by default. Details + diagnostic scripts in
`docs/BENCH_HARNESS_NOTES.md`.

**Container + full sweep OOM workaround**: `USE_CONTAINER=1 ...
--full` in a single invocation OOMs the Docker Desktop VM around
test #118 (accumulated JAX / XLA state from ~500 concurrent
parametrizations exceeds the default 12 GB VM limit). Work around by
**splitting the run by problem abstract class** — each chunk is a
separate container invocation and starts from a fresh process.
Template (exclude CHARDIS0, which OOMs even alone):

```bash
for CLS in "AbstractUnconstrainedMinimisation" \
           "AbstractBoundedMinimisation and not AbstractBoundedQuadraticProblem and not CHARDIS0" \
           "AbstractConstrainedQuadraticProblem or AbstractBoundedQuadraticProblem"; do
  USE_CONTAINER=1 NO_EAGER=1 bash benchmarks/run_bench.sh --full -- \
    -k "(test_materialize or test_bcoo_jacobian) and ($CLS)" \
    --benchmark-save="$(git rev-parse --short HEAD)_full_linux_$(echo $CLS | awk '{print $1}' | tr '[:upper:]' '[:lower:]')"
done
```

Each chunk saves as a separate JSON; plots.py merges them via
`_latest_matching` pattern discovery or you can merge manually.

**Sif2jax source**: the container uses the **local editable checkout**
(`$SIF2JAX_PATH`, default `~/pasteurcodes/sif2jax`), mounted into
`/sif2jax` and `pip install -e`d at container start. No build-time pin.
Whatever branch/commit is checked out locally is what the container
sees — useful for testing sif2jax PRs without a rebuild, but means
reproducibility across machines depends on the local checkout state.

**Which benchmark stats to cite**: benchmark timings are geometrically
distributed with long tails. **Always ignore `mean` and `max`** — they
are dominated by a handful of outlier iterations (GC, scheduler jitter,
XLA background work) and will mislead you. Also ignore any quantile
above the median (p75, p90, p99): the long tail makes upper quantiles
dominated by the same transient noise as `mean`/`max`. Use `min`,
`median`, and lower quantiles (p10–p50). `min` captures the warm floor
(what a tight-loop user sees); `median` captures typical runtime. If a
delta shows up in `mean` but not `min`/`median`, it's noise.

**Sweep vs isolated min — sweep-level cross-problem contamination is
still a factor on Linux container**. Even in the clean container, a
sweep's reported `min` can be 1.3–2× off from the true warm floor for
a problem, because the benchmark harness amortises warmup across
sibling tests. **Before declaring a sweep-level regression real,
re-measure isolated** with the snippet below (8 runs × N=50 iters,
min-of-minimums):

```python
fn = jax.jit(lineaxpr.bcoo_hessian(f))
for _ in range(5): jax.block_until_ready(fn(y))  # warmup
mins = []
for _ in range(8):
    N=50; t=time.perf_counter()
    for _ in range(N): o = fn(y); jax.block_until_ready(o)
    mins.append((time.perf_counter()-t)*1e6/N)
mins.sort()
# Cite mins[0] (best warm floor) and mins[3] (median-of-8)
```

We've routinely seen a problem flagged at +30µs in the sweep come out
within 2µs of baseline isolated. Don't commit a "fix" for a sweep-
flagged regression until isolated confirms it.

## Development loop (rule-change pattern)

When adding a small structural rule change (one primitive's rule, or a
narrow helper), follow this loop. Deviating is fine for docs-only or
purely additive changes, but the loop catches real regressions early
and keeps the commit graph clean.

1. **Small incremental change** — one primitive rule or helper per
   commit. If a rule extension exposes a latent bug elsewhere (e.g.
   a missing method on `BEllpack.pad_rows` that only batched operands
   hit), fix the bug in the same commit and mention it in the message
   — but don't fold in unrelated improvements.

2. **Validate correctness** — `uv run pytest tests/ -q` (unit,
   ~45 s). If the change touches anything that might be exercised by
   the full sif2jax surface (`_add_rule`, `_reshape_rule`, batched
   paths, sentinels, etc.), also run `uv run pytest
tests/test_sif2jax_sweep.py -m slow --tb=line -q` (~3 min). The
   slow sweep caught a CHARDIS0 correctness bug on a 2026-04-22
   broadcast_in_dim change the unit tests missed.

3. **Spot-check expected-win problems in isolation** — before the
   full bench, verify the change delivers on the problem(s) it was
   motivated by. If you don't see the expected win, stop and
   understand why. Either (a) the theory was wrong, (b) a later
   densifier is still blocking, or (c) the change is net-negative —
   make a wise call on whether to keep it. Accepting a change that
   doesn't deliver its expected win is a code smell; either adjust
   or drop.

4. **Curated bench regression check** — `USE_CONTAINER=1 bash
benchmarks/run_bench.sh` (~45 s, saves
   `.benchmarks/Linux-*/NNNN_<sha>_lineaxpr.json`). Diff against the
   previous `*_lineaxpr.json`; any regression >1.3× AND >15µs
   absolute is material, but always isolated-verify before
   concluding (see the sweep-vs-isolated note above). Sub-15µs or
   sub-1.3× deltas are usually noise.

5. **Commit** — descriptive subject + body explaining motivation,
   affected rule(s), expected win, measured win, any known
   trade-offs or follow-up work. Co-author tag for Claude.

6. **Full sweep** — kick off in the background via a detached nohup
   runner script (not a Bash tool with `run_in_background=true` —
   that tends to get killed by harness activity). Use the chunked
   template from the OOM workaround above. ~15–25 min. Monitor
   completion with a `tail -f | grep --line-buffered` Monitor tool
   call.

7. **Generate plots** — merge chunks into a combined `*_full.json`
   and `*_full_folded.json`, then
   `uv run python -m benchmarks.plots --tag full --platform Linux`
   (jax_min baseline) followed by the same command with
   `--baseline asdex_bcoo`. Produces 5 PNGs per commit under
   `benchmarks/plots/`: `abs`, `ratio_vs_jax_min`,
   `scatter_vs_jax_min`, `ratio_vs_asdex_bcoo`, `scatter_vs_asdex_bcoo`.
   It's easy to forget the second `--baseline`; if you only see 3
   plots for a given SHA, regenerate.

8. **Check full-sweep regressions** vs the previous commit's full
   sweep. Same isolate-to-confirm rule applies. Large sweep-level
   regressions on problems whose code paths the change didn't touch
   are almost always noise.

When the loop surfaces a correctness bug (step 2), revert the change
immediately and either diagnose or file as follow-up; don't try to
patch forward through correctness failures.

## Useful quick commands

```bash
# Fast tests (unit + hand-rolled + transform-level + curated CUTEst):
uv run pytest tests/ -v

# Slow sweep over all sif2jax problems (~200, ~2 min):
uv run pytest tests/test_sif2jax_sweep.py -m slow -v

# Regenerate the nse manifest after an intentional structural change:
uv run python -m tests.update_nse_manifest

# Per-commit lineaxpr bench (materialize + bcoo_jacobian only, commit-tagged):
bash benchmarks/run_bench.sh

# Reference bench (jax.hessian / asdex; tagged by JAX version, rerun only on upstream update):
bash benchmarks/run_bench.sh --refs

# Full sweep: lineaxpr on all ~275 sif2jax problems (~10 min, per-commit):
bash benchmarks/run_bench.sh --full

# Full refs sweep: jax.hessian + asdex across all problems (~30 min, rarely rerun):
bash benchmarks/run_bench.sh --full-refs

# View combined report with min(folded, unfolded) as the jax baseline:
uv run python -m benchmarks.report --tag lineaxpr   # curated table
uv run python -m benchmarks.report --tag full --summary  # full with aggregate stats

# Other modes: --curated (5-way), --highn (n>2500)
# Add USE_CONTAINER=1 for EAGER_CONSTANT_FOLDING=TRUE release parity.

# Monkeypatch experiment (pure-BCOO comparison):
uv run python -m experiments.run_monkeypatch
```

## Repo layout

```
lineaxpr/
  __init__.py           # public API exports
  _base.py              # LinOp classes + densify helpers
  materialize.py        # sparsify transform, rule registry, rules, public wrappers
experiments/
  sparsify_monkeypatch.py   # adds add_any/pad/scatter-add rules to jax.experimental.sparse
  run_monkeypatch.py        # reports pure-BCOO vs lineaxpr timings
tests/
  conftest.py               # enables jax_enable_x64
  test_ops.py               # unit tests for LinOp methods
  test_materialize.py       # hand-rolled problems with explicit expected matrices
  test_sparsify.py          # transform-level tests (seeds, const-prop, jit, errors)
  test_correctness.py       # curated CUTEst smoke tests
  test_sif2jax_sweep.py     # slow sweep + nse regression
  nse_manifest.json         # golden nse values per problem
  update_nse_manifest.py    # regenerator
benchmarks/
  run_bench.sh              # commit-tagged bench runner (lineaxpr/refs/curated/highn/full)
  test_curated.py           # 16-problem curated bench (5-way comparison)
  test_full.py              # all sif2jax scalar problems (DENSE_MAX=2000, BCOO_MAX=5000)
  test_highn.py             # n > 2500 sweep
  run_in_container.sh       # docker runner for EAGER_CONSTANT_FOLDING parity
docs/
  ARCHITECTURE.md           # walk algorithm, LinOp forms, rule pattern
  RESEARCH_NOTES.md         # empirical findings, sparsify comparison, DUAL deep-dive
  TODO.md                   # prioritized future work
```

## References

- Historical gist (pre-repo exploration):
  https://gist.github.com/jpbrodrick89/a3657522e7218d2cc98dae9f80258216
- Comparison library — asdex (coloring-based):
  https://github.com/adrhill/asdex
- Upstream reference — `jax.experimental.sparse.sparsify`:
  `.venv/lib/python3.13/site-packages/jax/experimental/sparse/transform.py`

## Documents to read

- `docs/ARCHITECTURE.md` — data structures + walk algorithm + LinOp methods
- `docs/RESEARCH_NOTES.md` — empirical findings (monkeypatch §9–§10 is key)
- `docs/TODO.md` — prioritized future work
