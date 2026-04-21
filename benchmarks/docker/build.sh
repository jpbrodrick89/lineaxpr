#!/bin/bash
# Build the lineaxpr bench container.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JAX_VERSION="${JAX_VERSION:-0.10.0}"
TAG="${TAG:-lineaxpr-bench:jax${JAX_VERSION}}"

echo "Building $TAG (jax==$JAX_VERSION)..."
docker build \
  --build-arg JAX_VERSION="$JAX_VERSION" \
  -t "$TAG" \
  -t lineaxpr-bench:latest \
  "$SCRIPT_DIR"

echo "Built: $TAG"
