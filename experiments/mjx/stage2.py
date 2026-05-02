"""Stage 2 — lineaxpr sparse Jacobian extraction on MJX collocation.

Deliverables:
  1. lineaxpr.bcoo_jacfwd on the collocation residual
  2. Spy plot of sparsity pattern saved to experiments/mjx/plots/
  3. Nonzero count vs dense matrix size
  4. Verify |lineaxpr_J - jax.jacobian_J| < 1e-10 (bit-exact)
  5. Bandwidth of extracted matrix

Run:
    uv run python -m experiments.mjx.stage2
    uv run python -m experiments.mjx.stage2 --model double_pendulum --T 20
"""
from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import lineaxpr
from experiments.mjx.models import MODELS
from experiments.mjx.collocation import make_collocation_residual


PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


def _nominal_z(info: dict) -> jnp.ndarray:
    z = jnp.zeros(info["z_dim"])
    return z.at[0].set(0.01)


def run(model_name: str = "cartpole", T: int = 20):
    xml, desc = MODELS[model_name]
    f, info = make_collocation_residual(xml, T)
    nq, nv, nu, nx = info["nq"], info["nv"], info["nu"], info["nx"]
    stride = info["stride"]
    z0 = _nominal_z(info)
    res_dim = info["residual_dim"]
    z_dim   = info["z_dim"]

    print(f"\n{'='*62}")
    print(f"  Stage 2 — lineaxpr Jacobian extraction")
    print(f"  model={model_name}  {desc}")
    print(f"  T={T}  Jacobian: {res_dim}×{z_dim}")
    dense_size = res_dim * z_dim
    expected_nz = (T - 1) * nx * (2 * stride - nu)   # each defect: left nx×stride + right nx×nx (no ctrl at t+1)
    print(f"  dense elements: {dense_size}")
    print(f"{'='*62}")

    # ------------------------------------------------------------------
    # 1. lineaxpr BCOO Jacobian
    # ------------------------------------------------------------------
    J_bcoo = lineaxpr.bcoo_jacfwd(f)(z0)
    print(f"\n1. lineaxpr.bcoo_jacfwd OK")
    print(f"   BCOO shape: {J_bcoo.shape}   nse={J_bcoo.nse}")
    print(f"   fill: {100*J_bcoo.nse/dense_size:.2f}%  vs dense {dense_size} elements")

    # Dense for comparison
    J_dense = lineaxpr.jacfwd(f)(z0)
    print(f"   dense shape: {J_dense.shape}")

    # ------------------------------------------------------------------
    # 2. Compare with jax.jacobian
    # ------------------------------------------------------------------
    J_ref = jax.jacobian(f)(z0)
    max_err = float(jnp.max(jnp.abs(J_dense - J_ref)))
    print(f"\n2. max |lineaxpr_J - jax.jacobian_J| = {max_err:.2e}")
    if max_err < 1e-10:
        print("   Bit-exact match ✓")
    else:
        print("   WARNING: mismatch above 1e-10")

    # ------------------------------------------------------------------
    # 3. Bandwidth analysis
    # ------------------------------------------------------------------
    J_np = np.array(J_dense)
    rows, cols = np.where(np.abs(J_np) > 1e-12)
    if len(rows):
        offsets = cols - rows
        bw_upper = int(offsets.max())
        bw_lower = int(-offsets.min())
        print(f"\n3. Bandwidth: upper={bw_upper}  lower={bw_lower}")
        print(f"   Expected ≈ nx+nu={nx+nu} (one block-row couples stride cols "
              f"at t and nx cols at t+1)")

    # ------------------------------------------------------------------
    # 4. Spy plot
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(PLOT_DIR, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Full Jacobian spy
        ax = axes[0]
        ax.spy(np.abs(J_np) > 1e-12, markersize=2, color="steelblue")
        ax.set_title(f"Jacobian sparsity — {model_name} T={T}\n"
                     f"{res_dim}×{z_dim}, nse={J_bcoo.nse}, "
                     f"{100*J_bcoo.nse/dense_size:.1f}% fill")
        ax.set_xlabel("z index (qpos,qvel,ctrl per knot)")
        ax.set_ylabel("defect index")

        # Block-structure zoom: per-block norms
        brows = T - 1
        bcols = T
        block_norms = np.zeros((brows, bcols))
        for br in range(brows):
            for bc in range(bcols):
                r0, r1 = br * nx, (br + 1) * nx
                c0, c1 = bc * stride, (bc + 1) * stride
                block_norms[br, bc] = np.linalg.norm(J_np[r0:r1, c0:c1])

        ax2 = axes[1]
        im = ax2.imshow(block_norms, aspect="auto", cmap="Blues",
                        interpolation="nearest")
        ax2.set_title(f"Block norms ({nx}×{stride} blocks)")
        ax2.set_xlabel("time index")
        ax2.set_ylabel("defect index")
        plt.colorbar(im, ax=ax2)

        path = os.path.join(PLOT_DIR, f"spy_{model_name}_T{T}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"\n4. Spy plot saved to {path}")
    except ImportError:
        print("\n4. matplotlib not available — skipping plot")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*62}")
    print(f"  Summary")
    print(f"  Jacobian {res_dim}×{z_dim} = {dense_size} dense elements")
    print(f"  lineaxpr nse = {J_bcoo.nse}  ({100*J_bcoo.nse/dense_size:.2f}% fill)")
    print(f"  Sparsity ratio: {dense_size // max(J_bcoo.nse, 1)}× fewer stored values")
    print(f"  jax.jacobian needs {z_dim} forward passes;  "
          f"lineaxpr needs 1 linearize + structural walk")
    print(f"{'='*62}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cartpole", choices=list(MODELS.keys()))
    parser.add_argument("--T", type=int, default=20)
    args = parser.parse_args()
    run(args.model, args.T)
