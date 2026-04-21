"""Track RSS across the preamble + repeated TABLE8 materialize calls.

If RSS climbs monotonically and gc.collect() can't bring it down,
we have a leak (python-side or jax-side). If RSS plateaus but TABLE8
still runs slow, the contamination is fragmentation/XLA-state, not leak.
"""
from __future__ import annotations
import gc
import os
import resource
import time

import jax
import sif2jax

from lineaxpr import materialize


def rss_mb():
    # macOS returns bytes; Linux returns KB. ru_maxrss is "max RSS ever",
    # so we pair with /proc or use psutil if available.
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback: ru_maxrss (high-water mark). Won't show drops but
        # shows growth.
        val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS: bytes. Linux: KB.
        if os.uname().sysname == "Darwin":
            return val / (1024 * 1024)
        return val / 1024


def build(problem, fmt="dense"):
    args_c = problem.args

    @jax.jit
    def fn(y):
        def f(z): return problem.objective(z, args_c)
        _, h = jax.linearize(jax.grad(f), y)
        return materialize(h, y, format=fmt)
    return fn


def block(out):
    for l in jax.tree_util.tree_leaves(out):
        jax.block_until_ready(l)
    return out


def time_call(c, y, n=100):
    # warm
    block(c(y))
    times = []
    for _ in range(n):
        t = time.perf_counter()
        block(c(y))
        times.append(time.perf_counter() - t)
    return min(times) * 1e6


def time_call_chunked(c, y, total=2000, chunks=20):
    """Reproduce pytest-benchmark's many-round behavior. Returns a list
    of per-chunk mins so we can see if min creeps up across chunks."""
    block(c(y))
    per_chunk_min = []
    n_per = total // chunks
    for _ in range(chunks):
        times = []
        for _ in range(n_per):
            t = time.perf_counter()
            block(c(y))
            times.append(time.perf_counter() - t)
        per_chunk_min.append(min(times) * 1e6)
    return per_chunk_min


by_name = {type(p).__name__: p for p in sif2jax.problems}
target = by_name["TABLE8"]

preamble_names = [
    "CHNROSNB", "GENROSE", "GAUSS1LS", "GAUSS2LS", "GAUSS3LS",
    "LUKSAN11LS", "LUKSAN12LS", "LUKSAN13LS", "LUKSAN14LS",
    "ARGLINA", "ARGLINB", "ARGLINC",
    "VESUVIOLS", "VESUVIOULS", "VESUVIALS",
    "FLETCHCR", "INTEQNELS",
]

print(f"{'phase':42s}  {'rss_mb':>8s}  {'table8_us':>10s}")
print(f"{'-'*42}  {'-'*8}  {'-'*10}")

# Baseline
c0 = build(target)
y0 = jax.device_put(target.y0)
t = time_call(c0, y0)
print(f"{'baseline (cold)':42s}  {rss_mb():8.1f}  {t:10.1f}")

# Preamble: for each problem, compile + run 10 times
for pname in preamble_names:
    p = by_name.get(pname)
    if p is None:
        continue
    try:
        c = build(p)
        y = jax.device_put(p.y0)
        block(c(y))
        for _ in range(10):
            block(c(y))
    except Exception as e:
        print(f"  [skip {pname}: {e}]")
        continue
    # Drop refs
    del c, y
    gc.collect()
    t = time_call(c0, y0)
    print(f"  after {pname:35s}  {rss_mb():8.1f}  {t:10.1f}")

# Now aggressive cleanup attempts
print()
print("--- cleanup attempts ---")
gc.collect()
t = time_call(c0, y0)
print(f"{'after gc.collect()':42s}  {rss_mb():8.1f}  {t:10.1f}")

jax.clear_caches()
gc.collect()
t = time_call(c0, y0)
print(f"{'after clear_caches + gc':42s}  {rss_mb():8.1f}  {t:10.1f}")

try:
    for d in jax.devices():
        d.synchronize_all_activity()
    gc.collect()
    t = time_call(c0, y0)
    print(f"{'after sync + gc':42s}  {rss_mb():8.1f}  {t:10.1f}")
except Exception as e:
    print(f"sync failed: {e}")

# Most aggressive — clear_backends rebuilds c0 lookups, so we need to rebuild
try:
    jax.clear_backends()
    gc.collect()
    c0 = build(target)
    y0 = jax.device_put(target.y0)
    t = time_call(c0, y0)
    print(f"{'after clear_backends + rebuild':42s}  {rss_mb():8.1f}  {t:10.1f}")
except Exception as e:
    print(f"clear_backends failed: {e}")

# Try malloc_trim on linux; on mac, try trying to release memory manually
try:
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
    t = time_call(c0, y0)
    print(f"{'after malloc_trim(0)':42s}  {rss_mb():8.1f}  {t:10.1f}")
except (OSError, AttributeError):
    pass

# Now check: does min creep up across many iterations?
print()
print("--- many-iterations TABLE8 min, in 20 chunks of 100 ---")
print(f"{'chunk':>5s}  {'rss_mb':>8s}  {'min_us':>8s}")
chunks = time_call_chunked(c0, y0, total=2000, chunks=20)
for i, m in enumerate(chunks):
    print(f"{i:5d}  {rss_mb():8.1f}  {m:8.1f}")
print(f"overall min across all 2000: {min(chunks):.1f}us")
