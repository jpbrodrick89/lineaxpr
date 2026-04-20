#!/bin/bash
# Like run_in_container.sh but also pip-installs asdex for head-to-head benches.

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
if [ -n "$SIF2JAX_PATH" ] && [ -d "$SIF2JAX_PATH" ]; then
    MOUNTS="$MOUNTS -v ${SIF2JAX_PATH}:/sif2jax"
    INSTALL="pip install -e /sif2jax && pip install -e . asdex"
else
    INSTALL="pip install -e . asdex"
fi

docker run --rm \
  $ENV_FLAGS \
  $MOUNTS \
  -w ${MOUNT_PATH} \
  ${CONTAINER_IMAGE} \
  bash -c "$INSTALL && pytest --benchmark-only \"\$@\"" -- "${ARGS[@]}"
