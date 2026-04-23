"""Merge chunked full-sweep JSONs into a single combined JSON that
`plots.py`'s `_latest_matching(r"_full(_folded)?.json$")` can discover.

Usage:
  uv run python -m benchmarks.merge_chunks <short_sha> folded   # folded run
  uv run python -m benchmarks.merge_chunks <short_sha> linux    # unfolded run

Discovers `NNNN_<sha>_full_<tag>_<class>.json` chunks in the Linux
platform dir produced by `run_full_chunked.sh` and concatenates their
`benchmarks` arrays. Writes with historical naming: `_full_folded.json`
for folded, `_full.json` (no `_linux` suffix) for unfolded — matches
the tags `plots.py` looks for."""
from __future__ import annotations
import json, re, sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent.parent / ".benchmarks"


def _plat_dir():
    for d in sorted(BENCH_DIR.iterdir()):
        if d.is_dir() and "Linux" in d.name:
            return d
    raise SystemExit("no Linux benchmark dir found")


def _next_counter(plat):
    n = 0
    for j in plat.glob("*.json"):
        m = re.match(r"(\d+)_", j.name)
        if m:
            n = max(n, int(m.group(1)))
    return n + 1


def main():
    sha, tag = sys.argv[1], sys.argv[2]  # tag = "folded" or "linux"
    plat = _plat_dir()
    chunks = sorted(plat.glob(f"*_{sha}_full_{tag}_*.json"))
    if not chunks:
        raise SystemExit(f"no chunks for {sha}/{tag}")
    print(f"merging {len(chunks)} chunks")
    for c in chunks:
        print(f"  {c.name}")
    merged = None
    for c in chunks:
        d = json.load(c.open())
        if merged is None:
            merged = d
            merged["benchmarks"] = list(d.get("benchmarks", []))
        else:
            merged["benchmarks"].extend(d.get("benchmarks", []))
    suffix = "_full_folded" if tag == "folded" else "_full"
    out = plat / f"{_next_counter(plat):04d}_{sha}{suffix}.json"
    out.open("w").write(json.dumps(merged, indent=2) + "\n")
    print(f"wrote {out.name} ({len(merged['benchmarks'])} benchmarks)")


if __name__ == "__main__":
    main()
