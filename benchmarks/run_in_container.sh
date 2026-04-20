#!/bin/bash
# Run benchmarks inside the pycutest container with EAGER_CONSTANT_FOLDING on.
#
# Benchmarks depend on sif2jax for test problems. If sif2jax is at
# ~/pasteurcodes/sif2jax (adjacent to this repo), it gets mounted and
# pip-installed in-container. Override with SIF2JAX_PATH env var.

CONTAINER_IMAGE="${CONTAINER_IMAGE:-johannahaffner/pycutest:latest}"
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
