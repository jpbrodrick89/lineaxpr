#!/bin/bash
# Run benchmarks inside a Linux container.
#
# Two usage modes:
#   1. Folded / release parity — EAGER_CONSTANT_FOLDING=TRUE (default).
#      Reproduces JAX's release config where constants get folded at
#      staging time. Matches `--full-folded` output naming.
#   2. Unfolded / clean-Linux numbers — pass `--no-flags`. Gets around
#      the macOS-native ~3× cross-problem contamination on dense-pattern
#      problems (TABLE8-class); see docs/BENCH_HARNESS_NOTES.md.
#
# Images:
#   - Default (`lineaxpr-bench:latest`): built from benchmarks/docker/
#     with jax 0.10.0 baked in. Build with `bash benchmarks/docker/build.sh`.
#   - Override with CONTAINER_IMAGE env var (e.g. to use the
#     johannahaffner/pycutest:latest image with jax 0.9.2 for version
#     comparisons).
#
# sif2jax is mounted from ~/pasteurcodes/sif2jax (override via
# SIF2JAX_PATH) and pip-installed editable at container start.

CONTAINER_IMAGE="${CONTAINER_IMAGE:-lineaxpr-bench:latest}"
MOUNT_PATH="/workspace"
LOCAL_PATH="$(cd "$(dirname "$0")/.." && pwd)"
SIF2JAX_PATH="${SIF2JAX_PATH:-$(cd "$(dirname "$0")/../../sif2jax" 2>/dev/null && pwd)}"

ENV_FLAGS="-e EAGER_CONSTANT_FOLDING=TRUE"
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--no-flags" ]; then
        ENV_FLAGS=""
    else
        ARGS+=("$arg")
    fi
done

MOUNTS="-v ${LOCAL_PATH}:${MOUNT_PATH}"
INSTALL="pip install -e ."
if [ -n "$SIF2JAX_PATH" ] && [ -d "$SIF2JAX_PATH" ]; then
    MOUNTS="$MOUNTS -v ${SIF2JAX_PATH}:/sif2jax"
    INSTALL="pip install -e /sif2jax && pip install -e ."
fi

docker run --rm \
  $ENV_FLAGS \
  $MOUNTS \
  -w ${MOUNT_PATH} \
  ${CONTAINER_IMAGE} \
  bash -c "$INSTALL && pytest --benchmark-only \"\$@\"" -- "${ARGS[@]}"
