# Minimal Linux image for lineaxpr benchmarks.
#
# Rationale (see docs/BENCH_HARNESS_NOTES.md):
#   - macOS-native unfolded runs show ~3× cross-problem contamination on
#     dense-pattern problems (TABLE8-class). Linux runs are clean.
#   - The sif2jax pycutest image ships jax 0.9.2, so we used to pip-install
#     jax 0.10.0 on every run (~40s). This image bakes it in.
#   - sif2jax itself has no pycutest runtime dep, so we drop CUTEst and
#     just keep Python + jax + deps.
#
# Build:
#   bash benchmarks/docker/build.sh
#
# Override the jax pin at build time:
#   docker build --build-arg JAX_VERSION=0.11.0 \
#     -t lineaxpr-bench:latest benchmarks/docker/
FROM python:3.12-slim

ARG JAX_VERSION=0.10.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "jax==${JAX_VERSION}" \
      "jaxlib==${JAX_VERSION}" \
      jaxtyping \
      equinox \
      beartype \
      wadler_lindig \
      typing_extensions \
      asdex \
      pytest \
      pytest-benchmark \
      pytest-timeout \
      pytest-forked \
      hatchling

WORKDIR /workspace
CMD ["/bin/bash"]
