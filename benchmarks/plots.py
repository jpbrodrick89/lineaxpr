"""Plot lineaxpr + reference benchmarks.

Patterns modeled on sif2jax/plot_benchmarks_jax092.py. Inverse CDF
(sorted runtime / ratio) is the default style — best for showing "most
problems are fast, some are slow" distributions across ~275 problems.

Plot types:
  inverse_cdf_abs     : sorted runtime per method, log-y
  inverse_cdf_ratio   : sorted runtime/baseline ratio per method
  scatter             : baseline_us vs method_us on log-log

Baselines:
  jax_min             : min(jax.hessian unfolded, jax.hessian folded)
  jax_unfolded        : jax.hessian without EAGER_CONSTANT_FOLDING
  jax_folded          : jax.hessian with EAGER_CONSTANT_FOLDING
  asdex_dense         : asdex dense
  asdex_bcoo          : asdex sparse
  pycutest            : pycutest (Fortran) — not yet integrated

Usage:
    # Auto-discover latest full + full-refs JSONs, produce headline plots:
    uv run python -m benchmarks.plots

    # Specific comparison:
    uv run python -m benchmarks.plots --methods bcoo_jacobian,asdex_bcoo \
                                       --baseline jax_min \
                                       --kind inverse_cdf_ratio

    # All figures for a given tag set:
    uv run python -m benchmarks.plots --tag full --all

Outputs go to benchmarks/plots/<commit>_<kind>_<baseline>.png.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BENCH_DIR = Path(__file__).resolve().parent.parent / ".benchmarks"
PLOT_DIR = Path(__file__).resolve().parent / "plots"

# Canonical method names users can request (plus aliases).
METHOD_ALIASES = {
    "mat": "materialize",
    "materialize": "materialize",
    "bcoo": "bcoo_jacobian",
    "bcoo_jacobian": "bcoo_jacobian",
    "jaxhes": "jax_hessian",
    "jax_hessian": "jax_hessian",
    "jax": "jax_hessian",
    "jaxhes_folded": "jax_hessian_folded",
    "jax_hessian_folded": "jax_hessian_folded",
    "jax_folded": "jax_hessian_folded",
    "jax_min": "jax_min",  # synthesized
    "asdex_dense": "asdex_dense",
    "asdex_bcoo": "asdex_bcoo",
}

# Plot style registry — keep consistent across figures. Labels use the
# public API names (lineaxpr.hessian / lineaxpr.bcoo_hessian) even
# though the test function names are test_materialize / test_bcoo_jacobian —
# users recognize the API, not the internal test names.
METHOD_STYLE = {
    "materialize":        dict(label="lineaxpr.hessian",          color="C0", linestyle="-"),
    "bcoo_jacobian":      dict(label="lineaxpr.bcoo_hessian",     color="C1", linestyle="-"),
    "jax_hessian":        dict(label="jax.hessian (unfolded)",    color="C2", linestyle="--"),
    "jax_hessian_folded": dict(label="jax.hessian (folded)",      color="C2", linestyle="-."),
    "jax_min":            dict(label="jax.hessian (best of folded/unfolded)", color="C3", linestyle="-"),
    "asdex_dense":        dict(label="asdex.dense",               color="C4", linestyle=":"),
    "asdex_bcoo":         dict(label="asdex.bcoo",                color="C5", linestyle="-"),
}


# -------------------------- data loading ---------------------------------


def _list_bench_jsons():
    for plat_dir in sorted(BENCH_DIR.iterdir()):
        if not plat_dir.is_dir():
            continue
        for j in sorted(plat_dir.glob("*.json")):
            yield plat_dir, j


def _latest_matching(pattern: str):
    """Return the JSON with the largest pytest-benchmark counter that matches."""
    pat = re.compile(pattern)
    best = None
    for _, j in _list_bench_jsons():
        if pat.search(j.name):
            m = re.match(r"(\d+)_", j.name)
            counter = int(m.group(1)) if m else 0
            if best is None or counter > best[0]:
                best = (counter, j)
    return best[1] if best else None


# Normalize short method names from test_curated.py to full names.
# test_curated uses `test_{kind}` with kind in {mat, bcoo, jaxhes, ...};
# test_full uses the full function name. Normalize to the function name
# so both sources key identically in our index.
_METHOD_CANON = {
    "mat": "materialize",
    "bcoo": "bcoo_jacobian",
    "jaxhes": "jax_hessian",
    "jaxhes_folded": "jax_hessian_folded",
}


def _index(json_path: Path) -> dict[tuple[str, str], tuple[float, int]]:
    """Index a bench JSON → {(problem, method): (min_us, n)}. stat=min by default."""
    with json_path.open() as f:
        data = json.load(f)
    out = {}
    for b in data["benchmarks"]:
        name = b["name"]
        if "[" not in name:
            continue
        method = name.split("[")[0].replace("test_", "")
        method = _METHOD_CANON.get(method, method)
        # tests may use bare names or 'Class-PROB' — strip to the trailing name.
        problem = name.split("[")[1].rstrip("]").split("-")[-1]
        min_us = b["stats"]["min"] * 1e6
        n = b["extra_info"].get("dimensionality", 0)
        out[(problem, method)] = (min_us, n)
    return out


def load_runs(lineaxpr_tag: str, refs_tag: str | None = None):
    """Return (lineaxpr_json_path, refs_json_path_list, unified_idx).

    `unified_idx` is a single dict keyed by (problem, method) where
    method may be any of the canonical names (including `jax_min` which
    is synthesized from min(jaxhes, jaxhes_folded)).

    For --full, refs may live in per-method files (full-jaxhes,
    full-jaxhes-folded, full-asdex-dense, full-asdex-bcoo) as well as
    in the combined full-refs. All are loaded and merged."""
    lx_path = _latest_matching(rf"_{re.escape(lineaxpr_tag)}\.json$")

    refs_paths = []
    if refs_tag is not None:
        # Explicit refs tag — load just that one.
        p = _latest_matching(rf"{re.escape(refs_tag)}.*\.json$")
        if p:
            refs_paths.append(p)
    else:
        # Auto-discover. For full, look for any per-method + combined refs.
        patterns = (
            ["full-refs-jax", "full-jaxhes-jax", "full-jaxhes-folded-jax",
             "full-asdex-jax", "full-asdex-dense-jax", "full-asdex-bcoo-jax"]
            if "full" in lineaxpr_tag else ["refs-jax"]
        )
        for pat in patterns:
            p = _latest_matching(rf"{re.escape(pat)}.*\.json$")
            if p and p not in refs_paths:
                refs_paths.append(p)

    idx: dict[tuple[str, str], tuple[float, int]] = {}
    if lx_path:
        idx.update(_index(lx_path))
    for p in refs_paths:
        idx.update(_index(p))

    # Synthesize jax_min = min(jax_hessian, jax_hessian_folded).
    problems = {p for (p, _) in idx}
    for p in problems:
        u = idx.get((p, "jax_hessian"))
        f = idx.get((p, "jax_hessian_folded"))
        if u and f:
            idx[(p, "jax_min")] = (min(u[0], f[0]), u[1])
        elif u:
            idx[(p, "jax_min")] = u
        elif f:
            idx[(p, "jax_min")] = f

    return lx_path, refs_paths, idx


def times_for(idx, method: str) -> dict[str, tuple[float, int]]:
    """Return {problem: (time_us, n)} for a canonical method name."""
    canon = METHOD_ALIASES.get(method, method)
    return {p: v for (p, m), v in idx.items() if m == canon}


# -------------------------- plot helpers ---------------------------------


def plot_inverse_cdf_abs(ax, runs: dict[str, dict[str, tuple[float, int]]]):
    """X = sorted problem index, Y = runtime (log), one line per method."""
    for method, data in runs.items():
        if not data:
            continue
        vals = sorted(v[0] for v in data.values())
        style = METHOD_STYLE.get(method, {"label": method})
        ax.plot(np.arange(len(vals)), vals, linewidth=1.3, alpha=0.85, **style)
    ax.set_yscale("log")
    ax.set_xlabel("problem index (sorted per method)")
    ax.set_ylabel("runtime (µs, log scale)")
    ax.grid(True, alpha=0.15, which="both")
    ax.legend(fontsize=8, loc="upper left")


def plot_inverse_cdf_ratio(ax, runs: dict[str, dict[str, tuple[float, int]]],
                            baseline: dict[str, tuple[float, int]],
                            baseline_label: str):
    """X = sorted problem index, Y = method / baseline, one line per method.

    Only problems where both method and baseline have data are included.
    """
    for method, data in runs.items():
        if not data:
            continue
        ratios = []
        for prob, (v, _n) in data.items():
            b = baseline.get(prob)
            if b is None or b[0] == 0:
                continue
            ratios.append(v / b[0])
        if not ratios:
            continue
        ratios.sort()
        style = METHOD_STYLE.get(method, {"label": method})
        ax.plot(np.arange(len(ratios)), ratios, linewidth=1.3, alpha=0.85, **style)
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.4, lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("problem index (sorted per method)")
    ax.set_ylabel(f"method / {baseline_label}   (<1 = method faster)")
    ax.grid(True, alpha=0.15, which="both")
    ax.legend(fontsize=8, loc="upper left")


def plot_scatter(ax, runs, baseline, baseline_label):
    """X = baseline runtime, Y = method runtime. log-log."""
    for method, data in runs.items():
        if not data:
            continue
        xs, ys = [], []
        for prob, (v, _n) in data.items():
            b = baseline.get(prob)
            if b is None:
                continue
            xs.append(b[0])
            ys.append(v)
        if not xs:
            continue
        style = METHOD_STYLE.get(method, {"label": method})
        ax.scatter(xs, ys, s=8, alpha=0.45,
                   color=style.get("color"),
                   marker="o", edgecolors="none",
                   label=style.get("label", method))

    lims = [1, 1e7]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="parity")
    for k in (0.1, 10):
        ax.plot(lims, [k * l for l in lims], "r:", lw=0.5, alpha=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e7)
    ax.set_ylim(1, 1e7)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{baseline_label} (µs)")
    ax.set_ylabel("method (µs)")
    ax.grid(True, alpha=0.1, which="both")
    ax.legend(fontsize=8, loc="upper left")


# -------------------------- CLI --------------------------


def _commit_from_lx(lx_path: Path | None) -> str:
    if lx_path is None:
        return "nodata"
    # 0005_4562b8f_full.json -> 4562b8f
    m = re.match(r"\d+_([0-9a-f]+)_", lx_path.name)
    return m.group(1) if m else "nodata"


def _counter_from_lx(lx_path: Path | None) -> str:
    """Extract the pytest-benchmark NNNN prefix.

    0005_4562b8f_full.json -> '0005'. Used to prefix plot filenames so
    plots are uniquely tied to the specific bench run that produced
    them (multiple runs of the same commit get distinct NNNN).
    """
    if lx_path is None:
        return "0000"
    m = re.match(r"(\d+)_", lx_path.name)
    return m.group(1) if m else "0000"


def _canonicalise_methods(method_spec: str | None, default: list[str]) -> list[str]:
    raw = default if method_spec is None else [s.strip() for s in method_spec.split(",") if s.strip()]
    return [METHOD_ALIASES.get(m, m) for m in raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full",
                    help="Lineaxpr bench tag (default: full; also: lineaxpr, curated)")
    ap.add_argument("--refs-tag", default=None,
                    help="Refs tag (default: matches --tag; full→full-refs-jax, "
                         "else refs-jax)")
    ap.add_argument("--methods", default=None,
                    help="Comma-separated method list. Default: "
                         "bcoo_jacobian,mat,asdex_bcoo,jax_min")
    ap.add_argument("--baseline", default="jax_min",
                    help="Baseline method for ratio/scatter plots. "
                         "(jax_min, jaxhes, jaxhes_folded, asdex_bcoo, ...)")
    ap.add_argument("--kind", default="all",
                    choices=["abs", "ratio", "scatter", "all"],
                    help="Plot kind (default: all).")
    ap.add_argument("--out-dir", default=str(PLOT_DIR),
                    help=f"Output directory (default: {PLOT_DIR}).")
    ap.add_argument("--name", default=None,
                    help="Override output filename stem.")
    args = ap.parse_args()

    lx, refs_paths, idx = load_runs(args.tag, args.refs_tag)
    print(f"lineaxpr: {lx.name if lx else '(not found)'}")
    if refs_paths:
        for p in refs_paths:
            print(f"refs:     {p.name}")
    else:
        print("refs:     (not found)")
    if lx is None:
        return 1

    methods = _canonicalise_methods(
        args.methods,
        default=["bcoo_jacobian", "mat", "asdex_bcoo", "jax_min"],
    )
    runs = {m: times_for(idx, m) for m in methods}

    baseline_method = METHOD_ALIASES.get(args.baseline, args.baseline)
    baseline_times = times_for(idx, baseline_method)
    baseline_label = METHOD_STYLE.get(baseline_method, {}).get("label", baseline_method)

    PLOT_DIR.mkdir(exist_ok=True)
    commit = _commit_from_lx(lx)
    counter = _counter_from_lx(lx)
    # Default stem: NNNN_<commit>_<tag> — matches the bench JSON's
    # naming scheme so it's obvious which bench a plot came from
    # (and NNNN tie-breaks between multiple runs of the same commit).
    stem = args.name or f"{counter}_{commit}_{args.tag}"

    # Status summary to stderr.
    for m, t in runs.items():
        print(f"  {m:25s}: n_problems={len(t)}")
    print(f"  baseline {baseline_method:20s}: n_problems={len(baseline_times)}")

    kinds = (["abs", "ratio", "scatter"] if args.kind == "all" else [args.kind])

    for kind in kinds:
        fig, ax = plt.subplots(figsize=(10, 6))
        if kind == "abs":
            plot_inverse_cdf_abs(ax, runs)
            ax.set_title(f"Hessian extraction — absolute runtime (sorted), commit {commit}")
            path = Path(args.out_dir) / f"{stem}_abs.png"
        elif kind == "ratio":
            if not baseline_times:
                print(f"  skip ratio plot: no data for baseline {baseline_method!r}")
                plt.close(fig)
                continue
            plot_inverse_cdf_ratio(ax, runs, baseline_times, baseline_label)
            ax.set_title(f"Hessian extraction — runtime / {baseline_label} "
                         f"(sorted), commit {commit}")
            path = Path(args.out_dir) / f"{stem}_ratio_vs_{baseline_method}.png"
        elif kind == "scatter":
            if not baseline_times:
                print(f"  skip scatter plot: no data for baseline {baseline_method!r}")
                plt.close(fig)
                continue
            plot_scatter(ax, runs, baseline_times, baseline_label)
            ax.set_title(f"Hessian extraction — {baseline_label} vs method, commit {commit}")
            path = Path(args.out_dir) / f"{stem}_scatter_vs_{baseline_method}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
