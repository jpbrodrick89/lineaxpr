"""Diff two full-sweep merged JSONs. Reports regressions (new > old)
and wins (new < old) for materialize + bcoo_jacobian per problem using
min-time. Thresholds match CLAUDE.md's sweep-level rule (>1.3× AND
>15µs absolute). Per CLAUDE.md, sweep-level regressions are often
false positives (cross-problem contamination) — isolate-verify before
acting.

Usage:
  uv run python -m benchmarks.diff_full <old.json> <new.json>
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def _index(path):
    d = json.load(open(path))
    out = {}
    for b in d["benchmarks"]:
        name = b["name"]
        if "[" not in name:
            continue
        method = name.split("[")[0].replace("test_", "")
        prob = name.split("[")[1].rstrip("]").split("-")[-1]
        n = b["extra_info"].get("dimensionality", 0)
        out[(method, prob)] = (b["stats"]["min"] * 1e6, n)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a.split("=")[0]: a.split("=", 1)[1] if "=" in a else True
             for a in sys.argv[1:] if a.startswith("--")}
    old_path, new_path = args[0], args[1]
    ratio_thr = float(flags.get("--ratio", 1.3))
    abs_thr = float(flags.get("--abs", 15))
    old, new = _index(old_path), _index(new_path)
    common = sorted(set(old) & set(new))
    only_new = sorted(set(new) - set(old))
    only_old = sorted(set(old) - set(new))

    regrs, wins = [], []
    for key in common:
        ot, on = old[key]
        nt, nn = new[key]
        ratio = nt / ot
        delta = nt - ot
        if ratio >= ratio_thr and delta >= abs_thr:
            regrs.append((key, ot, nt, ratio, nn))
        if ratio <= 1 / ratio_thr and -delta >= abs_thr:
            wins.append((key, ot, nt, ratio, nn))

    regrs.sort(key=lambda r: -r[3])
    wins.sort(key=lambda r: r[3])

    print(f"old={Path(old_path).name}  new={Path(new_path).name}")
    print(f"common={len(common)}  only_new={len(only_new)}  only_old={len(only_old)}")
    print()
    print(f"=== REGRESSIONS (>={ratio_thr}x AND >={abs_thr}us) — {len(regrs)} ===")
    for (m, p), ot, nt, r, n in regrs:
        print(f"  {m:16s} {p:18s} n={n:<6d}  {ot:8.1f}us -> {nt:8.1f}us  ({r:.2f}x)")
    print()
    print(f"=== WINS (<={1/ratio_thr:.3f}x AND <=-{abs_thr}us) — {len(wins)} ===")
    for (m, p), ot, nt, r, n in wins:
        print(f"  {m:16s} {p:18s} n={n:<6d}  {ot:8.1f}us -> {nt:8.1f}us  ({r:.2f}x)")
    if only_new:
        print(f"\nonly in new ({len(only_new)}):")
        for m, p in only_new[:10]:
            print(f"  {m:16s} {p}")
    if only_old:
        print(f"only in old ({len(only_old)}):")
        for m, p in only_old[:10]:
            print(f"  {m:16s} {p}")


if __name__ == "__main__":
    main()
