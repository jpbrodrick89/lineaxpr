# Research notes — findings from the exploration phase

These are the non-obvious insights from building lineaxpr. Preserve them;
they inform design decisions.

## 1. EAGER_CONSTANT_FOLDING matters enormously

For problems with **y-independent Hessians** (pure quadratics: DUAL, CMPC),
`jax.hessian` with `EAGER_CONSTANT_FOLDING=TRUE` folds the entire Hessian
to a literal constant. On same-kernel comparison, materialize ties with
jax.hessian at the constant-load floor (~5 µs).

Without eager folding, `jax.hessian` does `n` forward-over-reverse passes
at runtime and is **15-25× slower** than materialize on the same problems.

So our "quadratic Hessian" wins depend on the execution environment. For
**y-dependent Hessians** (anything non-quadratic: FLETCHCR, DIXMAAN,
LEVYMONT, GENROSE, ARGTRIGLS) folding can't help — our wins are real
regardless.

**Lesson for benchmarking**: always pass `y` as a jit INPUT, never a
closure. Closure + quadratic = spurious jax.hessian speedup from folding.

## 2. Memory bandwidth is the floor for dense output

At `n = 3000`, a dense float64 Hessian is 72 MB. Writing 72 MB on a
typical CPU (≈30 GB/s bandwidth) takes ≈2.4 ms. Both asdex and materialize
hit this wall when asked to produce a dense matrix. The BCOO path skips
the dense allocation entirely — that's where the 100-6000× wins come from.

Empirically: `jnp.zeros((3000, 3000))` on its own takes ~2.2 ms. Adding a
scatter of 13000 entries only adds ~200 µs. So almost all of the dense
materialize time is the zero allocation.

## 3. asdex comparison

asdex uses global conservative sparsity + coloring + a second AD pass.
We use per-linearization exact sparsity + structural walk.

Benchmark findings (EAGER_CONSTANT_FOLDING=TRUE, y as jit input):

| Problem | n | jax.hess | materialize | bcoo_jac | asdex_dn | asdex_bc |
|---|---|---|---|---|---|---|
| HS110 | 10 | 8.9 µs | 6.9 µs | 30.7 µs | (skip) | (skip) |
| QING | 100 | 27 µs | 8 µs | 10.2 µs | 7.6 µs | 8.2 µs |
| LEVYMONT | 100 | 156 µs | 45 µs | 50 µs | 42 µs | 24 µs |
| ARGTRIGLS | 200 | 322 µs | 141 µs | 141 µs | (skip) | (skip) |
| FLETCHCR | 1000 | 1335 µs | 197 µs | 17 µs | 202 µs | 26 µs |
| CMPC1 | 2550 | 1208 µs | 736 µs | 8.6 µs | 736 µs | 8.2 µs |
| DIXMAANE1 | 3000 | 67 ms | 1.58 ms | 30 µs | 1.93 ms | 60 µs |
| DIXMAANI1 | 3000 | 62 ms | 1.51 ms | 28 µs | 2.45 ms | 68 µs |
| EDENSCH | 2000 | 34.9 ms | 609 µs | 72 µs | 794 µs | 66 µs |

### Where asdex wins
- **LEVYMONT-bcoo (2×)**: their static symmetric star-coloring finds 3
  colors for a 298-nnz pattern. 3 HVPs via vmap + gather-to-BCOO ≈ 11 µs.
  Our structural walk + scatter ≈ 19 µs. Architectural trade-off.
- Small-n sparse problems where coloring's constant HVP cost beats our
  per-nse scatter overhead.

### Where we win
- Medium-to-large n (DIXMAANE1, DIXMAANI1, EDENSCH, FLETCHCR) where
  structural walk amortizes and asdex's vmap pays per-color costs.
- All correctness: we're bit-exact. asdex's `symmetric=True` flag is buggy
  on some problems (DUAL1, DIXMAANB produced off-by-15 off-diagonals);
  `symmetric=False` fixes it but costs more colors on dense Hessians.

### Where asdex fails
- Tiny problems (HS110 n=10, HART6 n=6, ARGTRIGLS n=200): their compile
  path errors or skips. We handle these fine via the `n<16` short-circuit.
- Fully dense Hessians: no speedup (needs `n` colors) — ARGTRIGLS is the
  standout case, 258 µs asdex vs 197 µs jax.hessian.

## 4. The `jax.linearize` overhead myth

We initially suspected `jax.linearize` itself had fixed overhead that
inflated our times. Tested directly:

```
1 HVP via jax.linearize:        8.2 µs
3 HVPs via vmap(linearize):    12.9 µs
3 jvps bare (no linearize):    12.7 µs   <- identical
```

So `jax.linearize` is essentially free. The ~6 µs gap from our structural
walk to asdex's 3-HVP approach (on LEVYMONT) is our per-eqn overhead, not
linearize's.

## 5. jnp.prod = binary tree unfolds into many Pivoted levels

HS110's `(∏x)^0.2` term, when linearized, becomes a **tree of pads + adds**.
Each tree level produces a Pivoted; our walk processes them level-by-level,
accumulating ~50 small BCOO entries. Short-circuit to vmap(eye) fires at
n=10 (threshold 16) to avoid this overhead.

Fix would require cross-eqn pattern matching — detect "whole prod tree
HVP" and emit the analytical `α·u·uᵀ + diag(c·u²)` form directly.
Substantial machinery, low priority.

## 6. Pivoted out_rows in practice

Enumerating empirically across DIXMAANB, FLETCHCR, LEVYMONT (20
add_any(Pivoted, Pivoted) calls):
- **15/20: same out_rows, different in_cols** — e.g. diagonal + band
  contributions to same rows. Concat produces duplicates-per-row. JAX's
  `scatter-add` at densification correctly sums them, but intermediate
  `nse` is bloated ~2×.
- **4/20: disjoint out_rows ranges** — two pads to different row blocks
  (e.g. DIXMAANB rows [0..999] and [2000..2999]). Middle 1000 rows get
  filled by a later add.
- **0/20: concat produces contiguous arange(out_size)**. Empirically
  never happens.

**Only scatter-add** produces genuinely arbitrary `out_rows`. All other
ops (pad, slice, add concat) produce unions of contiguous ranges. This
suggests a future "range-set" representation for `out_rows` when no
scatter has fired — see TODO.

## 7. The `same-in_cols + different-out_rows` pattern

For banded Hessians (LEVYMONT, FLETCHCR, DIXMAAN's diagonal + bands), the
dominant pattern in add_any is: **same out_rows, different in_cols**
(15/20 empirical). If we added a "banded Pivoted" form that holds
`(out_rows, list[(in_cols, values)])` instead of concatenating entries,
we'd cut `nse` in half on these problems. Naturally produces CSR-style
output. See TODO #12.

## 8. DIXMAAN split-slice observation

Multiple DIXMAAN problems (A1, E1, I1, M1 — the β=0 subset) use
simpler HVP structures with much sparser outputs (25-30 µs bcoo vs 85 µs
for full DIXMAAN). The coloring is the same (N colors) but the nnz is
much smaller because fewer cross-terms.

## 9. Short-circuit at n<16

Without the short-circuit, HS110 (n=10) costs ~15 µs (structural walk
overhead on 50 tree-prod intermediates) vs 5 µs for vmap(hvp)(eye). We
short-circuit at `primal.size < 16` to match the vmap floor.

The threshold is a hard-coded heuristic. Could become a parameter or
derived from a "walk-complexity" estimate (number of eqns / operand
shape), but 16 works empirically across CUTEst.
