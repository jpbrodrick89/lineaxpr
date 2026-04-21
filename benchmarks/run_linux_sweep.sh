#!/bin/bash
# Sequential Linux-container unfolded-sweep runner.
#
# Runs references (once, shared across commits) and lineaxpr at HEAD
# and the prior "scatter" Diagonal.todense commit. All inside the
# baked-in jax-0.10.0 container for clean (no-contamination) numbers.
#
# Wall clock: ~50 min total. Progress goes to stdout; intermediate
# results land under .benchmarks/Linux-CPython-*/.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "$(date '+%H:%M:%S') $*"; }

export USE_CONTAINER=1
export NO_EAGER=1

log "=== references: jax.hessian unfolded ==="
bash benchmarks/run_bench.sh --full-jaxhes

log "=== references: asdex dense ==="
bash benchmarks/run_bench.sh --full-asdex-dense

log "=== references: asdex bcoo ==="
bash benchmarks/run_bench.sh --full-asdex-bcoo

log "=== lineaxpr: HEAD (v*eye) ==="
bash benchmarks/run_bench.sh --full

log "=== lineaxpr: scatter variant (patch only Diagonal.todense) ==="
# Patch just the Diagonal.todense method to scatter (matches 760e2b7
# impl). The bench infrastructure itself stays at HEAD so USE_CONTAINER /
# NO_EAGER work. Capture original line and restore after.
ORIGINAL_LINE='        return self.values[:, None] * jnp.eye(self.n, dtype=self.values.dtype)'
SCATTER_PATCH='        idx = jnp.arange(self.n); return jnp.zeros((self.n, self.n), self.values.dtype).at[idx, idx].set(self.values)'
sed -i.bak "s|${ORIGINAL_LINE}|${SCATTER_PATCH}|" lineaxpr/_base.py

# Tag the save name so it doesn't collide with HEAD's _full run.
SHA="$(git rev-parse --short HEAD)"
mv ".benchmarks/Linux-CPython-3.12-64bit/" ".benchmarks/Linux-CPython-3.12-64bit/" 2>/dev/null || true
# Use a custom tag by temporarily moving HEAD's run then running with
# the patch; the bench script will pick up HEAD's sha so we rename after.
bash benchmarks/run_bench.sh --full || true
# Rename the just-produced file to indicate scatter variant.
LATEST="$(ls -t .benchmarks/Linux-CPython-3.12-64bit/*_full.json | head -1)"
if [ -f "$LATEST" ]; then
  NEW="${LATEST%.json}_scatter.json"
  # Bench files are numbered; keep the number, swap the label.
  NEW="$(dirname "$LATEST")/$(basename "$LATEST" .json)_scatter.json"
  mv "$LATEST" "$NEW"
  log "saved scatter variant as $NEW"
fi

# Restore original file
mv lineaxpr/_base.py.bak lineaxpr/_base.py

log "=== DONE ==="
