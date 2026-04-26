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
    "mat_folded": "materialize_folded",
    "materialize_folded": "materialize_folded",
    "bcoo_folded": "bcoo_jacobian_folded",
    "bcoo_jacobian_folded": "bcoo_jacobian_folded",
    "mat_min": "materialize_min",  # synthesized
    "materialize_min": "materialize_min",
    "bcoo_min": "bcoo_jacobian_min",  # synthesized
    "bcoo_jacobian_min": "bcoo_jacobian_min",
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
    "materialize":           dict(label="lineaxpr.hessian (unfolded)",              color="C0", linestyle="--"),
    "bcoo_jacobian":         dict(label="lineaxpr.bcoo_hessian (unfolded)",         color="C1", linestyle="--"),
    "materialize_folded":    dict(label="lineaxpr.hessian (folded)",                color="C0", linestyle="-."),
    "bcoo_jacobian_folded":  dict(label="lineaxpr.bcoo_hessian (folded)",           color="C1", linestyle="-."),
    "materialize_min":       dict(label="lineaxpr.hessian (best of folded/unfolded)",      color="C0", linestyle="-"),
    "bcoo_jacobian_min":     dict(label="lineaxpr.bcoo_hessian (best of folded/unfolded)", color="C1", linestyle="-"),
    "jax_hessian":           dict(label="jax.hessian (unfolded)",                   color="C2", linestyle="--"),
    "jax_hessian_folded":    dict(label="jax.hessian (folded)",                     color="C2", linestyle="-."),
    "jax_min":               dict(label="jax.hessian (best of folded/unfolded)",    color="C3", linestyle="-"),
    "asdex_dense":           dict(label="asdex.dense",                              color="C4", linestyle=":"),
    "asdex_bcoo":            dict(label="asdex.bcoo",                               color="C5", linestyle="-"),
}


# -------------------------- data loading ---------------------------------


# Platform filter for _list_bench_jsons. Set by main() via --platform.
# Mac and Linux numbers are not comparable (different arch + JAX backend),
# so any plot should be single-platform.
_PLATFORM_FILTER: str | None = None


def _list_bench_jsons():
    for plat_dir in sorted(BENCH_DIR.iterdir()):
        if not plat_dir.is_dir():
            continue
        if _PLATFORM_FILTER and _PLATFORM_FILTER.lower() not in plat_dir.name.lower():
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


def load_runs(lineaxpr_tag: str, refs_tag: str | None = None,
              sha: str | None = None):
    """Return (lineaxpr_json_path, refs_json_path_list, unified_idx).

    `unified_idx` is a single dict keyed by (problem, method) where
    method may be any of the canonical names, including synthesized
    `jax_min`, `materialize_min`, `bcoo_jacobian_min` (best-of
    folded/unfolded per-method).

    For `--full`, this auto-loads both the unfolded lineaxpr run
    (`<sha>_full.json`) and the folded companion
    (`<sha>_full_folded.json`) if present. Entries from the folded
    JSON are re-keyed with `_folded` suffix so they coexist with the
    unfolded entries in the index.

    Refs may live in per-method files (full-jaxhes, full-jaxhes-folded,
    full-asdex-dense, full-asdex-bcoo) as well as in the combined
    full-refs. All are loaded and merged."""
    if sha is not None:
        # Pin to a specific commit's bench files (used to regenerate
        # historical plots after a load_runs fix).
        lx_path = _latest_matching(
            rf"_{re.escape(sha)}_{re.escape(lineaxpr_tag)}\.json$"
        )
        lx_folded_path = (
            _latest_matching(rf"_{re.escape(sha)}_full_folded\.json$")
            if lineaxpr_tag == "full" else None
        )
    else:
        lx_path = _latest_matching(rf"_{re.escape(lineaxpr_tag)}\.json$")
        lx_folded_path = None
        if lineaxpr_tag == "full":
            lx_folded_path = _latest_matching(r"_full_folded\.json$")

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
    if lx_folded_path:
        # Re-key folded entries: method → method_folded so they coexist
        # with the unfolded ones under distinct method names.
        folded_idx = _index(lx_folded_path)
        for (prob, method), val in folded_idx.items():
            if method in ("materialize", "bcoo_jacobian"):
                idx[(prob, f"{method}_folded")] = val
            else:
                idx[(prob, method)] = val
    # Refs JSONs are produced by sweeps that run lineaxpr's `materialize`
    # and `bcoo_jacobian` alongside `asdex_*` / `jax_hessian*`, so the
    # JSON contains stale lineaxpr entries (frozen at the commit the
    # refs sweep was captured at). Drop those at ingest — otherwise they
    # silently overwrite the just-loaded current-commit values, causing
    # the bcoo_jacobian_min/materialize_min synthesis to mix stale
    # unfolded with current folded. Refs files only contribute their
    # own method names (asdex_bcoo, asdex_dense, jax_hessian,
    # jax_hessian_folded) — anything else gets filtered.
    REFS_METHODS = {
        "asdex_bcoo", "asdex_dense", "jax_hessian", "jax_hessian_folded",
    }
    for p in refs_paths:
        for (prob, method), val in _index(p).items():
            if method in REFS_METHODS:
                idx[(prob, method)] = val

    # Synthesize min-of-{folded, unfolded} methods (per-method "best" metric).
    problems = {p for (p, _) in idx}
    for p in problems:
        for base in ("jax_hessian", "materialize", "bcoo_jacobian"):
            u = idx.get((p, base))
            f = idx.get((p, f"{base}_folded"))
            min_key = "jax_min" if base == "jax_hessian" else f"{base}_min"
            if u and f:
                idx[(p, min_key)] = (min(u[0], f[0]), u[1])
            elif u:
                idx[(p, min_key)] = u
            elif f:
                idx[(p, min_key)] = f

    return lx_path, refs_paths, idx


def times_for(idx, method: str) -> dict[str, tuple[float, int]]:
    """Return {problem: (time_us, n)} for a canonical method name."""
    canon = METHOD_ALIASES.get(method, method)
    return {p: v for (p, m), v in idx.items() if m == canon}


# Abstract-type filters for publication plots. Loaded lazily so plot
# invocations without `--problem-filter` don't need sif2jax imported.
_PROBLEM_TYPES: dict[str, str] | None = None

def _problem_types() -> dict[str, str]:
    """Map problem class name → abstract-type tag.

    Tags: 'unconstrained', 'bounded-min', 'bounded-quad', 'constrained-quad'.
    """
    global _PROBLEM_TYPES
    if _PROBLEM_TYPES is not None:
        return _PROBLEM_TYPES
    try:
        import sif2jax
        from sif2jax._problem import (
            AbstractUnconstrainedMinimisation, AbstractBoundedMinimisation,
            AbstractBoundedQuadraticProblem, AbstractConstrainedQuadraticProblem,
        )
    except ImportError:
        _PROBLEM_TYPES = {}
        return _PROBLEM_TYPES
    _PROBLEM_TYPES = {}
    for p in sif2jax.problems:
        name = p.__class__.__name__
        # Check specific subclasses before their parents.
        if isinstance(p, AbstractUnconstrainedMinimisation):
            _PROBLEM_TYPES[name] = "unconstrained"
        elif isinstance(p, AbstractConstrainedQuadraticProblem):
            _PROBLEM_TYPES[name] = "constrained-quad"
        elif isinstance(p, AbstractBoundedQuadraticProblem):
            _PROBLEM_TYPES[name] = "bounded-quad"
        elif isinstance(p, AbstractBoundedMinimisation):
            _PROBLEM_TYPES[name] = "bounded-min"
    return _PROBLEM_TYPES


def filter_problems(
    data: dict[str, tuple[float, int]],
    tag: str | None,
) -> dict[str, tuple[float, int]]:
    """Keep only problems whose abstract tag matches.

    `tag` values: 'unconstrained', 'bounded-min', 'bounded-quad',
    'constrained-quad', or None for no filtering. Problems absent from
    sif2jax's problem list (e.g. out-of-sweep benchmarks) are dropped
    when a filter is active — they can't be classified.
    """
    if tag is None:
        return data
    types = _problem_types()
    return {p: v for p, v in data.items() if types.get(p) == tag}


def filter_by_n(
    data: dict[str, tuple[float, int]],
    min_n: int,
) -> dict[str, tuple[float, int]]:
    """Keep only problems with dimensionality > min_n."""
    return {p: v for p, v in data.items() if v[1] > min_n}


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
    ap.add_argument("--sha", default=None,
                    help="Pin to a specific commit's bench files (e.g. "
                         "'2da443c'). Used to regenerate historical plots "
                         "without picking up newer JSONs via _latest_matching.")
    ap.add_argument("--platform", default=None,
                    help="Restrict to a platform dir, e.g. 'Linux' or 'Darwin'. "
                         "Mac/Linux numbers aren't comparable; pick one. "
                         "Defaults to the current OS.")
    ap.add_argument("--problem-filter", default=None,
                    choices=[None, "unconstrained", "bounded-min",
                             "bounded-quad", "constrained-quad"],
                    help="Restrict plots to problems of a given abstract "
                         "type. Useful for publication: 'unconstrained' "
                         "drops the 176 bounded/constrained problems from "
                         "the plot, leaving just the ~201 true "
                         "unconstrained-minimisation benchmarks. Output "
                         "filename includes the filter tag.")
    args = ap.parse_args()

    global _PLATFORM_FILTER
    import platform as _platform_mod
    _PLATFORM_FILTER = args.platform or _platform_mod.system()

    lx, refs_paths, idx = load_runs(args.tag, args.refs_tag, sha=args.sha)
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
        default=["bcoo_jacobian_min", "materialize_min", "asdex_bcoo", "jax_min"],
    )
    runs = {m: times_for(idx, m) for m in methods}

    baseline_method = METHOD_ALIASES.get(args.baseline, args.baseline)
    baseline_times = times_for(idx, baseline_method)
    baseline_label = METHOD_STYLE.get(baseline_method, {}).get("label", baseline_method)

    # Apply problem filter (post-method-split so each method-dict shrinks).
    if args.problem_filter is not None:
        runs = {m: filter_problems(d, args.problem_filter) for m, d in runs.items()}
        baseline_times = filter_problems(baseline_times, args.problem_filter)

    PLOT_DIR.mkdir(exist_ok=True)
    commit = _commit_from_lx(lx)
    counter = _counter_from_lx(lx)
    # Default stem: NNNN_<commit>_<tag>[_<filter>] — `_<filter>` suffix
    # added when a problem-filter is active, so publication plots don't
    # collide with the regression-tracking plots.
    filter_suffix = f"_{args.problem_filter}" if args.problem_filter else ""
    stem = args.name or f"{counter}_{commit}_{args.tag}{filter_suffix}"

    kinds = (["abs", "ratio", "scatter"] if args.kind == "all" else [args.kind])

    # Two passes: full set, then "large" (n > 16) to drop toy-dim problems
    # that hide large-n structure under setup/dispatch overhead.
    passes: list[tuple[str, str]] = [("", ""), ("_large", " [n>16]")]

    for size_suffix, size_title in passes:
        if size_suffix:
            runs_p = {m: filter_by_n(d, 16) for m, d in runs.items()}
            baseline_p = filter_by_n(baseline_times, 16)
        else:
            runs_p = runs
            baseline_p = baseline_times

        print(f"pass {size_suffix or '(all)'}:")
        for m, t in runs_p.items():
            print(f"  {m:25s}: n_problems={len(t)}")
        print(f"  baseline {baseline_method:20s}: n_problems={len(baseline_p)}")

        title_filter = (f" [{args.problem_filter}]"
                        if args.problem_filter else "") + size_title

        for kind in kinds:
            fig, ax = plt.subplots(figsize=(10, 6))
            if kind == "abs":
                plot_inverse_cdf_abs(ax, runs_p)
                ax.set_title(f"{counter} · Hessian extraction{title_filter} — absolute runtime (sorted), commit {commit}")
                path = Path(args.out_dir) / f"{stem}{size_suffix}_abs.png"
            elif kind == "ratio":
                if not baseline_p:
                    print(f"  skip ratio plot: no data for baseline {baseline_method!r}")
                    plt.close(fig)
                    continue
                plot_inverse_cdf_ratio(ax, runs_p, baseline_p, baseline_label)
                ax.set_title(f"{counter} · Hessian extraction{title_filter} — runtime / {baseline_label} "
                             f"(sorted), commit {commit}")
                path = Path(args.out_dir) / f"{stem}{size_suffix}_ratio_vs_{baseline_method}.png"
            elif kind == "scatter":
                if not baseline_p:
                    print(f"  skip scatter plot: no data for baseline {baseline_method!r}")
                    plt.close(fig)
                    continue
                plot_scatter(ax, runs_p, baseline_p, baseline_label)
                ax.set_title(f"{counter} · Hessian extraction{title_filter} — {baseline_label} vs method, commit {commit}")
                path = Path(args.out_dir) / f"{stem}{size_suffix}_scatter_vs_{baseline_method}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
