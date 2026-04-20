#!/bin/bash
# Run pytest-benchmark with commit-tagged save files.
#
# Usage:
#   bash benchmarks/run_bench.sh                 # lineaxpr-only (per-commit default)
#   bash benchmarks/run_bench.sh --refs          # reference methods (jax.hessian / asdex)
#   bash benchmarks/run_bench.sh --curated       # full curated 5-way comparison
#   bash benchmarks/run_bench.sh --highn         # high-n sweep (n > 2500)
#   bash benchmarks/run_bench.sh --full          # test_full.py (all scalar problems)
#   bash benchmarks/run_bench.sh -- <pytest args>   # pass extra args through
#
# Saves go to .benchmarks/<platform>/<tag>.json
# where <tag> is:
#   lineaxpr    : <short_commit>_lineaxpr   (rerun per commit)
#   refs        : refs-<jax_version>        (rerun on upstream JAX/asdex update only)
#   curated     : <short_commit>_curated
#   highn       : <short_commit>_highn
#   full        : <short_commit>_full
#
# Env-var overrides:
#   DENSE_MAX=2000    largest n for materialize benches
#   BCOO_MAX=5000     largest n for bcoo_jacobian benches
#   USE_CONTAINER=1   run under benchmarks/run_in_container.sh
#                     (EAGER_CONSTANT_FOLDING=TRUE for release parity)
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="lineaxpr"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --refs)     MODE="refs"; shift ;;
    --curated)  MODE="curated"; shift ;;
    --highn)    MODE="highn"; shift ;;
    --full)     MODE="full"; shift ;;
    --lineaxpr) MODE="lineaxpr"; shift ;;
    --)         shift; EXTRA_ARGS+=("$@"); break ;;
    *)          EXTRA_ARGS+=("$1"); shift ;;
  esac
done

SHORT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'nogit')"
JAX_VERSION="$(uv run python -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'unknown')"

SELECT=()
case "$MODE" in
  lineaxpr)
    NAME="${SHORT_SHA}_lineaxpr"
    SELECT=(benchmarks/test_curated.py -k "test_materialize or test_bcoo_jacobian")
    ;;
  refs)
    NAME="refs-jax${JAX_VERSION}"
    # jax_hessian (unfolded + folded) and asdex (dense + bcoo). The
    # folded variant is the "fair headline" comparison; the unfolded one
    # is kept for visibility into how much EAGER_CONSTANT_FOLDING helps.
    SELECT=(benchmarks/test_curated.py -k "test_jax_hessian or test_jax_hessian_folded or test_asdex_dense or test_asdex_bcoo")
    ;;
  curated)
    NAME="${SHORT_SHA}_curated"
    SELECT=(benchmarks/test_curated.py)
    ;;
  highn)
    NAME="${SHORT_SHA}_highn"
    SELECT=(benchmarks/test_highn.py)
    ;;
  full)
    NAME="${SHORT_SHA}_full"
    SELECT=(benchmarks/test_full.py)
    ;;
esac

echo "=== bench mode=$MODE  save=$NAME ==="
echo "=== JAX: $JAX_VERSION  commit: $SHORT_SHA ==="
echo ""

if [ "${USE_CONTAINER:-0}" = "1" ]; then
  echo "Running under container (EAGER_CONSTANT_FOLDING=TRUE)..."
  bash "$SCRIPT_DIR/run_in_container.sh" "${SELECT[@]}" \
    --benchmark-only --benchmark-save="$NAME" \
    "${EXTRA_ARGS[@]:+${EXTRA_ARGS[@]}}"
else
  uv run pytest "${SELECT[@]}" \
    --benchmark-only --benchmark-save="$NAME" \
    --benchmark-columns=min,median,mean,stddev,rounds \
    "${EXTRA_ARGS[@]:+${EXTRA_ARGS[@]}}"
fi
