"""Tabular report combining lineaxpr per-commit data with reference data.

Loads the most recent .benchmarks/<platform>/NNNN_<tag>.json files and
prints a per-problem comparison.

The "vs jax" reference uses min(folded, unfolded) — both regimes are
reasonable depending on use case (closure-y vs traced-y, constant-H
vs y-dependent), so the headline is the better of the two.

Usage:
    uv run python -m benchmarks.report                         # auto-discover latest
    uv run python -m benchmarks.report --lineaxpr 0003 --refs 0004
    uv run python -m benchmarks.report --tag full              # use *_full.json files
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent.parent / ".benchmarks"


def _all_jsons():
    """Yield (platform_dir, json_path) for every saved bench JSON."""
    for plat in sorted(BENCH_DIR.iterdir()):
        if not plat.is_dir():
            continue
        for j in sorted(plat.glob("*.json")):
            yield plat, j


def _latest(tag_re: str):
    """Find the most recent JSON whose basename matches `tag_re`.

    Sorts by the leading NNNN_ counter pytest-benchmark assigns.
    """
    pat = re.compile(tag_re)
    candidates = []
    for _, p in _all_jsons():
        if pat.search(p.name):
            # leading 4-digit counter is the pytest-benchmark seq num
            counter_match = re.match(r"(\d+)_", p.name)
            counter = int(counter_match.group(1)) if counter_match else 0
            candidates.append((counter, p))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _index(json_path: Path):
    """Index a benchmark JSON by (problem, method) -> min_us."""
    with json_path.open() as f:
        data = json.load(f)
    out = {}
    for b in data["benchmarks"]:
        name = b["name"]
        if "[" not in name:
            continue
        method = name.split("[")[0].replace("test_", "")
        # problem id may be "Class-NAME" or just "NAME" depending on test file
        problem = name.split("[")[1].rstrip("]").split("-")[-1]
        n = b["extra_info"].get("dimensionality", 0)
        out[(problem, method)] = (b["stats"]["min"] * 1e6, n)
    return out


def _problems(idx):
    """Set of problem names in an index."""
    return {p for (p, _m) in idx}


def _get(idx, problem, method):
    v = idx.get((problem, method))
    return v[0] if v else None


def _n(idx, problem):
    for (p, _), (_, n) in idx.items():
        if p == problem:
            return n
    return 0


def _format_us(v):
    if v is None:
        return " " * 9
    if v >= 1e4:
        return f"{v/1000:9.1f}ms".rstrip()[-9:].rjust(9)
    return f"{v:9.1f}"


def _format_ratio(num, den):
    if num is None or den is None:
        return " " * 8
    r = num / den
    if r >= 100:
        return f"{r:7.0f}x"
    if r >= 10:
        return f"{r:7.1f}x"
    return f"{r:7.2f}x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lineaxpr",
                    help="Lineaxpr tag suffix to report (lineaxpr, full, curated)")
    ap.add_argument("--refs-tag", default=None,
                    help="Refs tag suffix to combine with (default: matches --tag — "
                         "'lineaxpr'/'curated' use refs-jax*; 'full' uses full-refs-jax*)")
    args = ap.parse_args()

    if args.refs_tag is None:
        args.refs_tag = "full-refs-jax" if args.tag == "full" else "refs-jax"

    lx = _latest(rf"_{re.escape(args.tag)}\.json$")
    refs = _latest(rf"{re.escape(args.refs_tag)}.*\.json$")
    if lx is None:
        print(f"no lineaxpr JSON matching tag={args.tag!r}")
        return
    if refs is None:
        print(f"warning: no refs JSON matching tag={args.refs_tag!r}; reporting lineaxpr only")

    print(f"lineaxpr: {lx.name}")
    if refs:
        print(f"refs:     {refs.name}")
    print()

    lx_idx = _index(lx)
    refs_idx = _index(refs) if refs else {}
    problems = sorted(_problems(lx_idx) | _problems(refs_idx),
                      key=lambda p: (_n(lx_idx, p) or _n(refs_idx, p), p))

    cols = ["mat", "bcoo", "jaxhes_min", "asdex_dense", "asdex_bcoo"]
    header = (f"{'problem':18s} {'n':>5s}  "
              f"{'mat µs':>10s} {'bcoo µs':>10s} {'jax min µs':>11s}  "
              f"{'asdex_d':>9s} {'asdex_b':>9s}    "
              f"{'mat/jax':>8s} {'bcoo/jax':>9s} {'bcoo/asdex':>10s}")
    print(header)
    print("-" * len(header))

    for p in problems:
        n = _n(lx_idx, p) or _n(refs_idx, p)
        mat = _get(lx_idx, p, "mat") or _get(lx_idx, p, "materialize")
        bcoo = _get(lx_idx, p, "bcoo") or _get(lx_idx, p, "bcoo_jacobian")
        jh_unfold = _get(refs_idx, p, "jaxhes") or _get(refs_idx, p, "jax_hessian")
        jh_fold = _get(refs_idx, p, "jaxhes_folded") or _get(refs_idx, p, "jax_hessian_folded")
        jh_min = min((v for v in [jh_unfold, jh_fold] if v is not None), default=None)
        ax_d = _get(refs_idx, p, "asdex_dense")
        ax_b = _get(refs_idx, p, "asdex_bcoo")

        print(
            f"{p:18s} {n:>5d}  "
            f"{_format_us(mat):>10s} {_format_us(bcoo):>10s} {_format_us(jh_min):>11s}  "
            f"{_format_us(ax_d):>9s} {_format_us(ax_b):>9s}    "
            f"{_format_ratio(jh_min, mat):>8s} {_format_ratio(jh_min, bcoo):>9s} "
            f"{_format_ratio(ax_b, bcoo):>10s}"
        )


if __name__ == "__main__":
    main()
