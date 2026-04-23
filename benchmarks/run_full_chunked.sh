#!/bin/bash
# Chunked full-sweep runner — both folded (EAGER_CONSTANT_FOLDING=TRUE,
# release-parity) and unfolded (NO_EAGER=1, clean Linux) regimes.
#
# Splits the sweep by problem abstract class so each chunk is a fresh
# container invocation. Works around the Docker-VM OOM around test
# ~#118 observed on combined runs (see CLAUDE.md "Container + full
# sweep OOM workaround"). Saves chunks with names like:
#   <sha>_full_folded_<class>     # folded
#   <sha>_full_linux_<class>      # unfolded
# After completion, merge via `benchmarks/merge_chunks.py`:
#   uv run python -m benchmarks.merge_chunks <sha> folded
#   uv run python -m benchmarks.merge_chunks <sha> linux
#
# Run detached so it survives terminal drops:
#   nohup bash benchmarks/run_full_chunked.sh > /tmp/full_sweep.log 2>&1 &
# Monitor via:
#   tail -f /tmp/full_sweep.log | grep -E "^===|^##########|FAILED|passed"
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SHORT_SHA="$(git rev-parse --short HEAD)"
CHUNKS=(
    "AbstractUnconstrainedMinimisation|unconstrainedminimisation"
    "AbstractBoundedMinimisation and not AbstractBoundedQuadraticProblem and not CHARDIS0|boundedminimisation"
    "AbstractConstrainedQuadraticProblem or AbstractBoundedQuadraticProblem|constrainedquadraticproblem"
)

run_chunk() {
    local tag=$1 eager_env=$2 cls=$3 cls_name=$4
    local save_name="${SHORT_SHA}_full_${tag}_${cls_name}"
    echo ""
    echo "=== [${tag}] ${cls_name} — save=${save_name} ==="
    env USE_CONTAINER=1 ${eager_env} bash benchmarks/run_bench.sh --full -- \
        -k "(test_materialize or test_bcoo_jacobian) and (${cls})" \
        --benchmark-save="${save_name}" \
        || echo "CHUNK FAILED: ${save_name}"
}

echo "########## FOLDED (EAGER_CONSTANT_FOLDING=TRUE) ##########"
for e in "${CHUNKS[@]}"; do
    run_chunk "folded" "" "${e%%|*}" "${e##*|}"
done

echo ""
echo "########## UNFOLDED (NO_EAGER=1, clean Linux) ##########"
for e in "${CHUNKS[@]}"; do
    run_chunk "linux" "NO_EAGER=1" "${e%%|*}" "${e##*|}"
done

echo ""
echo "########## ALL DONE — ${SHORT_SHA} ##########"
