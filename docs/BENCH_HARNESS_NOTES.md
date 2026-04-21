# Benchmark harness notes

Accumulated wisdom on getting trustworthy lineaxpr bench numbers. Read
this before interpreting full-sweep (`run_bench.sh --full`) results.

## TL;DR

- **macOS-native unfolded runs have ~3× cross-problem contamination** on
  dense-pattern problems (TABLE8, EXPLIN, EG2, NONCVXU[N2]). Use them
  for dev iteration only; don't draw conclusions about regressions
  smaller than ~3× from macOS-native `_full.json` files.
- **Linux-container runs are clean.** Use `USE_CONTAINER=1 NO_EAGER=1
bash benchmarks/run_bench.sh --full` for trustworthy unfolded numbers.
- **Folded runs (`USE_CONTAINER=1 --full` with default EAGER) are clean**
  — they already ran in the container.
- Directory split is automatic: `.benchmarks/Darwin-CPython-*/` vs
  `.benchmarks/Linux-CPython-*/`. Report filters to the current platform
  unless explicitly pointed elsewhere.

## The contamination, in detail

Symptoms observed on macOS ARM64 (same machine, across JAX 0.9.2 and
0.10.0, same pytest-benchmark version):

- **Cold** (first problem in a fresh process): TABLE8 materialize min
  ≈ 91–125 µs. Independent of impl (scatter, `v[:, None] * eye`, and
  `jnp.where(eye_bool, v, 0)` all tie).
- **Hot** (same problem after ~17 other `test_full` compiles + runs):
  TABLE8 min ≈ 299–336 µs. All three impls still tie; contamination
  doesn't prefer one.

Ratio: ~3×. Persists across the rest of the process — cannot be
undone by `gc.collect()`, `jax.clear_caches()`, deprecated
`jax.clear_backends()`, `synchronize_all_activity()`, extra warmup
iterations (tried 0, 50, 500, 2000, 5000 — all equally stuck), or
`--benchmark-disable-gc` (shaves ~10%, not meaningful).

### What the contamination is NOT

- **Not a memory leak.** RSS grows 570 → 727 MB across the preamble
  and then plateaus; `gc.collect()` doesn't release more. Healthy for
  JAX + sif2jax loaded.
- **Not pytest-benchmark.** A manual `time.perf_counter` loop inside a
  plain pytest test (no `benchmark` fixture) shows the same 310 µs min.
- **Not pytest the runner.** The identical manual loop in a standalone
  Python script (no pytest at all) also shows 310 µs.
- **Not iteration count.** Even 2000 timed iterations in a single batch
  all show the elevated floor; min never dips back toward 100 µs.
- **Not impl choice.** scatter, `v*eye`, and `jnp.where` all degrade
  identically. Switching back to scatter does not help.
- **Not a JAX-version regression.** 0.9.2 and 0.10.0 contaminate equally
  on macOS. In the container, both are clean.

### What it IS (best hypothesis)

**macOS/ARM-specific working-set eviction** triggered by running several
unrelated n×n-allocating problems in the same process. Once a problem
"goes cold" (another comparable-size problem runs in between), its
dispatch floor stays elevated for the rest of the process. The eviction
is NOT RSS-driven (RSS is stable); more likely CPU cache or XLA internal
per-executable state. Neither `MALLOC_NANO_ZONE=0` nor
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` nor XLA CPU flags
(`--xla_cpu_use_thunk_runtime=false`, `--xla_cpu_multi_thread_eigen=false`,
`OMP_NUM_THREADS=1`) recover it.

The same code path in a Linux x86 container (both jax 0.9.2 and 0.10.0)
has no such degradation — hot min ≈ cold min, within noise.

No public JAX or XLA issue exactly matches these symptoms as of
2026-04-21; filing one with a minimal repro may be the long-term path
to a platform-level fix. Until then, trust the container.

## How to get trustworthy numbers today

### For release-parity (folded) numbers

Already clean; nothing changed here.

```bash
USE_CONTAINER=1 bash benchmarks/run_bench.sh --full
# → .benchmarks/Linux-CPython-*/NNNN_<sha>_full_folded.json
```

### For clean unfolded numbers

```bash
USE_CONTAINER=1 NO_EAGER=1 bash benchmarks/run_bench.sh --full
# → .benchmarks/Linux-CPython-*/NNNN_<sha>_full_linux.json
```

The `NO_EAGER=1` flag strips `EAGER_CONSTANT_FOLDING=TRUE`, so you get
JAX's unfolded path (what users see without the release flag) but in
the clean Linux environment.

### For clean reference numbers (jax.hessian, asdex)

Same `USE_CONTAINER=1 NO_EAGER=1` prefix works for all reference modes:

```bash
USE_CONTAINER=1 NO_EAGER=1 bash benchmarks/run_bench.sh --full-jaxhes
USE_CONTAINER=1 NO_EAGER=1 bash benchmarks/run_bench.sh --full-asdex-dense
USE_CONTAINER=1 NO_EAGER=1 bash benchmarks/run_bench.sh --full-asdex-bcoo
# → saved as `full-<method>-jax<version>_linux.json`
```

### Dev iteration (macOS-native is fine for _this_)

`--curated` and cold/isolated single-problem runs are fine on macOS —
contamination only kicks in with many problems queued. Use native for
rapid iteration; switch to container for any number you'll cite.

## The bench container

`benchmarks/docker/Dockerfile` — minimal python:3.12-slim image with
jax/jaxlib pinned (default 0.10.0). No pycutest — sif2jax has no
runtime CUTEst dep and pulling the pycutest image just slowed builds
and forced an in-container pip install jax each run.

Build: `bash benchmarks/docker/build.sh`. Override the jax pin:

```bash
JAX_VERSION=0.11.0 bash benchmarks/docker/build.sh
```

To use the old pycutest image (for reproducing historical numbers or
testing jax 0.9.2):

```bash
CONTAINER_IMAGE=johannahaffner/pycutest:latest \
  USE_CONTAINER=1 bash benchmarks/run_bench.sh --full
```

## Diagnostic scripts

Left in `benchmarks/` for future perf investigation, not part of the
regular sweep:

- `test_contamination_probe.py` — parametrized pytest matrix over
  (impl × cleanup × warmup × problem). Has a `PREAMBLE=0/1` env toggle
  to compare cold vs hot. Saves to a `_linux` or `_darwin` benchmark
  file depending on platform.
- `rss_probe.py` — plain-Python RSS tracker across the same preamble
  used by the probe. Use to rule out memory leaks.
- `test_harness_isolation.py` — single pytest test that measures the
  same function via manual-loop and `benchmark()` fixture side-by-side.
  Used to pin whether contamination is pytest/pytest-benchmark or
  deeper (it's deeper).

## What's safe to compare

Within a single run on a single platform, direction (better/worse) is
reliable. Absolute numbers across platforms are NOT comparable (ARM vs
x86, different JAX backends). Use the container consistently for
regression tracking.

When in doubt, run the same problem in isolation (just that one test,
cold process) — isolated numbers agree across platforms much better
than sweep numbers.
