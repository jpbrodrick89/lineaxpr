#!/bin/bash
# Run pytest-benchmark with commit-tagged save files.
#
# Per-commit (rerun every commit, git-ignored locally):
#   bash benchmarks/run_bench.sh                 # lineaxpr on 16 curated problems
#   bash benchmarks/run_bench.sh --full          # lineaxpr on all ~200 sif2jax problems
#   bash benchmarks/run_bench.sh --curated       # curated 5-way (lineaxpr + jax + asdex)
#   bash benchmarks/run_bench.sh --highn         # high-n sweep (n > 2500)
#
# Reference (rerun only on upstream JAX/asdex update, git-tracked):
#   bash benchmarks/run_bench.sh --refs          # references on 16 curated problems
#   bash benchmarks/run_bench.sh --full-refs     # all refs on all problems (slow,
#                                                  prone to big-n pathologies)
#
# Per-method reference splits (preferred over --full-refs — each gets its own
# size cap and isolated Python process; combine via benchmarks/report.py):
#   bash benchmarks/run_bench.sh --full-jaxhes         # jax.hessian unfolded
#   bash benchmarks/run_bench.sh --full-jaxhes-folded  # jax.hessian folded
#   bash benchmarks/run_bench.sh --full-asdex-dense    # asdex dense
#   bash benchmarks/run_bench.sh --full-asdex-bcoo     # asdex bcoo
#
# Saves go to .benchmarks/<platform>/NNNN_<tag>.json where <tag> is:
#   lineaxpr    : <short_commit>_lineaxpr
#   full        : <short_commit>_full
#   curated     : <short_commit>_curated
#   highn       : <short_commit>_highn
#   refs        : refs-jax<version>               (git-tracked)
#   full-refs   : full-refs-jax<version>          (git-tracked)
#
# Env-var overrides:
#   DENSE_MAX=2000    largest n for materialize benches in test_full.py
#   BCOO_MAX=5000     largest n for bcoo_jacobian benches in test_full.py
#   USE_CONTAINER=1   run under benchmarks/run_in_container.sh
#                     (EAGER_CONSTANT_FOLDING=TRUE env var for release parity)
#
# Pass extra pytest args:
#   bash benchmarks/run_bench.sh -- -k "MYPROB"
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="lineaxpr"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --refs)                MODE="refs"; shift ;;
    --curated)             MODE="curated"; shift ;;
    --highn)               MODE="highn"; shift ;;
    --full)                MODE="full"; shift ;;
    --full-refs)           MODE="full-refs"; shift ;;
    --full-jaxhes)         MODE="full-jaxhes"; shift ;;
    --full-jaxhes-folded)  MODE="full-jaxhes-folded"; shift ;;
    --full-asdex-dense)    MODE="full-asdex-dense"; shift ;;
    --full-asdex-bcoo)     MODE="full-asdex-bcoo"; shift ;;
    --lineaxpr)            MODE="lineaxpr"; shift ;;
    --)                    shift; EXTRA_ARGS+=("$@"); break ;;
    *)                     EXTRA_ARGS+=("$1"); shift ;;
  esac
done

SHORT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'nogit')"
JAX_VERSION="$(uv run python -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'unknown')"

# lineaxpr methods: materialize + bcoo_jacobian (per-commit).
# reference methods: jax.hessian (folded+unfolded) + asdex (dense+bcoo).
LINEAXPR_FILTER='test_materialize or test_bcoo_jacobian'
REFS_FILTER='test_jax_hessian or test_jax_hessian_folded or test_asdex_dense or test_asdex_bcoo'

# Per-method bench caps (each reference method has a different practical
# ceiling; running them together in one pytest process hits the slowest
# method's wall clock on every iteration. Split runs let us tune each).
# Overridable via DENSE_MAX / BCOO_MAX env vars (read by test_full.py).
SELECT=()
case "$MODE" in
  lineaxpr)
    NAME="${SHORT_SHA}_lineaxpr"
    SELECT=(benchmarks/test_curated.py -k "$LINEAXPR_FILTER")
    ;;
  refs)
    NAME="refs-jax${JAX_VERSION}"
    SELECT=(benchmarks/test_curated.py -k "$REFS_FILTER")
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
    SELECT=(benchmarks/test_full.py -k "$LINEAXPR_FILTER")
    ;;
  full-refs)
    # Combined; prone to long wall-clock if any single method has a big-n
    # pathological problem. Prefer the per-method splits below.
    NAME="full-refs-jax${JAX_VERSION}"
    SELECT=(benchmarks/test_full.py -k "$REFS_FILTER")
    ;;
  full-jaxhes)
    # jax.hessian WITHOUT folding. Fast compile; runtime = n HVPs.
    # Default DENSE_MAX=2000 matches the lineaxpr bench.
    NAME="full-jaxhes-jax${JAX_VERSION}"
    SELECT=(benchmarks/test_full.py -k "test_jax_hessian and not test_jax_hessian_folded")
    ;;
  full-jaxhes-folded)
    # jax.hessian WITH eager_constant_folding. Folding cost explodes on
    # constant-H problems (CMPC1 at n=2550 takes 5s to fold). Tighter
    # cap by default — override DENSE_MAX to extend if you're patient.
    export DENSE_MAX="${DENSE_MAX:-1200}"
    NAME="full-jaxhes-folded-jax${JAX_VERSION}"
    SELECT=(benchmarks/test_full.py -k "test_jax_hessian_folded")
    ;;
  full-asdex-dense)
    # asdex dense: coloring cost grows with n; dense output is the
    # bandwidth-bound floor.
    export DENSE_MAX="${DENSE_MAX:-1500}"
    NAME="full-asdex-dense-jax${JAX_VERSION}"
    SELECT=(benchmarks/test_full.py -k "test_asdex_dense")
    ;;
  full-asdex-bcoo)
    # asdex bcoo: coloring cost grows with n but output stays sparse.
    export BCOO_MAX="${BCOO_MAX:-3000}"
    NAME="full-asdex-bcoo-jax${JAX_VERSION}"
    SELECT=(benchmarks/test_full.py -k "test_asdex_bcoo")
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
