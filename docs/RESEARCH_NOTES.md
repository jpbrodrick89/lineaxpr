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

## 9. Counterfactual: `sparsify(vmap(lin))(sparse.eye(n))`

Tested on the curated 16 as a potential "90% of `bcoo_jacobian` for free"
approach: `jax.experimental.sparse.sparsify(jax.vmap(lin))(sparse.eye(n))`
where `lin = jax.linearize(jax.grad(f), y)[1]`.

**Result**: 14/16 fail at trace time.
- **compiles** (2/16): DUAL1, DUAL3 — the constant-H quadratics where the
  linearized grad is trivially a dot_general with BCOO.
- **`sparse rule for add_any is not implemented`** (7): HS110, HART6, QING,
  CHNROSNB, LEVYMONT, GENROSE, EDENSCH.
- **`sparse rule for pad is not implemented`** (4): FLETCHCR, DIXMAANB,
  DIXMAANE1, DIXMAANI1.
- **`sparse rule for scatter-add is not implemented`** (2): CMPC1, CMPC2.
- **`Addition between sparse matrices of different shapes`** (1): ARGTRIGLS.

`add_any`, `pad`, and `scatter-add` are the adjoint primitives emitted by
AD when transposing `reduce_sum`, `slice`, and `gather`. They appear in
*every* non-trivial linearized-grad; sparsify covers the forward side
(`add`, `slice`, `gather`) but not their transposes. So the "just use
sparsify" shortcut fails on exactly the problems where sparsity matters.

## 10. Is `jax.experimental.sparse` a useful framework to migrate onto?

Considered: make Identity / ConstantDiagonal / Pivoted into sparsify
formats, register transpose-primitive rules, let sparsify drive the walk
and pick the most efficient format per eqn.

**Rejected.** Reasons:

1. **sparsify is hard-coded to BCOO/BCSR.** `SparsifyValue` is a
   NamedTuple `(shape, data_ref, indices_ref, indptr_ref)` — the format
   space is the presence/absence of `indices_ref` and `indptr_ref`.
   Adding Identity/Diagonal/Pivoted is not "register another format"; it
   requires rewriting `SparsifyEnv`, `SparsifyValue`, and every existing
   rule (44 BCOO + 27 BCSR) to understand the new forms. The framework's
   extension point is "add a rule for primitive P on formats (BCOO, BCSR)",
   not "add a format".

2. **Disjoint primitive sets.** sparsify's 44 BCOO rules cover forward
   programs: `add, mul, slice, gather, dot_general, reduce_sum, ...`. Our
   walk consumes the adjoint set: `add_any, pad, scatter-add, transpose`.
   Migration doesn't save us from writing the rules we already have; it
   just reframes where they live.

3. **We already ARE a mixed-sparsity framework.** `env[var]` is one of
   {ConstantDiagonal, Diagonal, Pivoted, BCOO, ndarray}. Rules promote
   forms (e.g. `slice(ConstantDiagonal) → Pivoted`) and demote when
   incompatible (`add_any(Pivoted, BCOO) → BCOO`). `bcoo_jacobian` is the
   boundary conversion. Wrapping this in `sparsify`'s API would add no
   capability; it would subtract flexibility because sparsify's format
   enum is closed.

4. **sparsify would densify eagerly on our ops.** `bcoo_dot_general` and
   friends assume you want to stay in BCOO. Our walk needs to produce
   `ConstantDiagonal * closure_vec → Diagonal` without ever touching BCOO
   — Diagonal is ~10× cheaper to carry. Putting sparsify between us and
   the primitives would break this.

5. **Architectural framing unchanged.** The `SparsifyEnv` buffer-refs
   pattern (values are pointers into a shared pool) is cleaner than our
   inline LinOp values if we ever want to share intermediates across
   branches; worth borrowing in isolation. But that's a local refactor,
   not a framework migration.

**Where sparsify stays relevant**:
- As a **contribution target**: our `add_any` / `pad` / `scatter-add`
  rules are upstream-able. Adding them to `sparse_rules_bcoo` would let
  `sparsify(vmap(linearize(...)[1]))` work for users who only want BCOO
  output and are fine with the BCOO-all-the-way cost. That's a
  complementary tool, not a replacement.
- As a **consumer of our output**: `BCOO` is the lingua franca we hand
  back; downstream code using `sparsify` consumes our result naturally.
- As a **reference implementation**: their rule-registry pattern
  (`sparse_rules_bcoo: dict[Primitive, Callable]`) is identical to ours.
  Convergence, not coincidence — it's the right structure.

**Future readiness of current design**: good. The LinOp class hierarchy
is the extension point; adding a new form (e.g. the "banded Pivoted" from
TODO #1) only requires new densification rules and new cases in
`_add_rule` — existing rules stay put. The alternative (sparsify
migration) would pay a framework-rewrite cost for zero capability gain.

### Monkeypatch experiment: pure-BCOO floor

`experiments/sparsify_monkeypatch.py` registers minimal rules for
`add_any`, `pad`, and `scatter-add` into `sparse_rules_bcoo`, enough to
get `sparsify(vmap(lin))(sparse.eye(n))` to compile on 14/16 curated
problems. Remaining 2 failures (HART6, ARGTRIGLS) are upstream sparsify
limits: `bcoo_broadcast_in_dim` can't create sparse dims of length > 1
from a scalar broadcast.

All 14 numerically match `bcoo_jacobian`. Timing (best-of-20 µs):

| Problem | n | sparsify-bcoo | lineaxpr bcoo | sparsify / lineaxpr |
|---|---|---|---|---|
| HS110 | 10 | 62.8 | 13.9 | 4.5× |
| QING | 100 | 8.9 | 8.1 | 1.1× |
| CHNROSNB | 50 | 9.6 | 8.9 | 1.1× |
| **DUAL1** | 85 | 49.3 | 5.2 | **9.4×** |
| **DUAL3** | 111 | 91.2 | 6.2 | **14.7×** |
| LEVYMONT | 100 | 16.8 | 20.2 | **0.83×** (sparsify wins) |
| GENROSE | 500 | 12.8 | 8.2 | 1.6× |
| FLETCHCR | 1000 | 20.3 | 10.6 | 1.9× |
| **CMPC1** | 2550 | 5937.5 | 8.3 | **712×** |
| **CMPC2** | 1530 | 2517.8 | 8.5 | **298×** |
| DIXMAANB | 3000 | 157.0 | 44.8 | 3.5× |
| DIXMAANE1 | 3000 | 86.8 | 23.6 | 3.7× |
| DIXMAANI1 | 3000 | 72.4 | 22.9 | 3.2× |
| EDENSCH | 2000 | 99.0 | 41.0 | 2.4× |

**Readings**:

1. **Constant-H detection is our biggest specific win.** CMPC1/CMPC2
   (constant-H quadratics, large n) are 300–700× faster with our
   `ConstantDiagonal` → `Diagonal` → closure-BCOO path. Sparsify builds
   BCOO intermediates for every `mul`/`add`/`scatter`; we evaluate those
   at trace time and emit a literal closure-BCOO. This is *the* use case
   where our walk saves a full `n×n`-ish amount of work.

2. **Dense-ish problems (DIXMAAN/EDENSCH) win 2–4×.** Not the 100–6000×
   we see vs `jax.hessian`; most of that gap is avoiding densification,
   which sparsify also avoids. The residual 2–4× is Pivoted vs BCOO
   overhead in the walk — less copying and fewer index-gather ops.

3. **Small banded: LEVYMONT goes the other way.** Pure-BCOO is actually
   17% *faster* than our walk at n=100 on a banded pattern. Our Pivoted
   machinery + `add_any` concat has fixed overhead that BCOO avoids.
   Consistent with §3's asdex finding for the same problem.

4. **Tied problems (QING, CHNROSNB, GENROSE) within 10–60%.** Small/
   medium problems where neither approach has much to exploit.

**What this reframes**:

- If we dropped Pivoted/Diagonal/ConstantDiagonal and relied on sparsify
  + the 4 missing rules, we'd be within 2–4× on ~8 problems, *behind* on
  LEVYMONT, *ahead* on the 2 upstream-sparsify-broken cases. We'd lose
  the 100–700× CMPC/DUAL wins — but those are dominated by
  EAGER_CONSTANT_FOLDING-friendly constant-H problems where `jax.hessian`
  is also already fast. So the *unique* wins of lineaxpr over both
  `jax.hessian` AND pure-BCOO sparsify are smaller than the headline
  numbers suggest: mostly DIXMAAN/EDENSCH in the 2–4× range, plus robust
  handling of tiny n (HS110, HART6, ARGTRIGLS) where sparsify errors.

- This motivates upstreaming the 4 rules regardless — even for users who
  don't need lineaxpr, pure-BCOO works on 14/16 CUTEst problems with
  those rules.

- BandedPivoted would likely close the LEVYMONT gap and widen the
  DIXMAAN lead, so it remains well-motivated.

**Sanity check on "tiny-n robustness" claim**: HS110 (n=10), HART6 (n=6),
and ARGTRIGLS (n=200) all succeed *without* the n<16 short-circuit too
— verified by setting `_SMALL_N_VMAP_THRESHOLD = 0` and re-running.
The walk genuinely handles these shapes, so robustness is real.
Short-circuit is a perf optimization, not a correctness crutch.

### Why DUAL is so fast: Python-trace-time BCOO literal

On DUAL3 (n=111, constant-H):

| variant | µs |
|---|---|
| `bcoo_jacobian(linearize(grad(f), y)[1], y)` | 5.8 |
| `jax.hessian(f)(y)` dense | 152.5 |
| `BCOO.fromdense(jax.hessian(f)(y))` | FAIL (ConcretizationTypeError: nse unknown at trace) |
| `sparsify(vmap(lin))(sparse.eye(n))` | 88.7 |

(No EAGER_CONSTANT_FOLDING env var set in this run; with it, `jax.hessian`
folds to a dense literal and hits its own ~3–5µs floor on this size.)

The 5.8µs comes from a Python-level eager eval: the walk sees all-closure
eqns, evaluates them concretely at trace time, emits a `BCOO((data_np,
indices_np), shape=...)` as a jit-literal. The runtime cost is just
"return a precomputed constant."

**Can JAX constant-fold a BCOO end-to-end?** No. `BCOO.fromdense`
requires nse at trace time, so the dense→sparse conversion cannot be
lifted inside a jit that closes over a constant dense Hessian. And XLA's
HLO folder does not traverse `jax.experimental.sparse`'s custom
primitives (`bcoo_todense_p`, `bcoo_fromdense_p`, `bcoo_dot_general_p`),
so even if the inputs are literals the folder won't reduce them through
sparse ops.

**Workarounds that don't work**:
- `BCOO.fromdense(jax.hessian(f)(y))`: fails — nse unknown at trace.
- Closure y + `jax.hessian`: folds dense, but `BCOO.fromdense` still
  fails on the trace-side.
- `sparsify(vmap(lin))(sparse.eye(n))`: traces a BCOO through, never
  reduces to a literal.

**What would work**: add a jax feature "trace-time BCOO literal from
concrete dense array" — essentially exposing `BCOO.fromdense(x)` as a
constant when x is a literal. This is a small jax PR in principle; in
practice, unclear the team would accept it. Until then, a walk that
eagerly evaluates closure eqns (ours) is the only way to get
compile-time-constant BCOO output. This is a non-trivial advantage for
constant-H problems and a legitimate reason to preserve our walk's
const-propagation path — it's not dead code when the whole walk folds.

### Addendum: EAGER_CONSTANT_FOLDING + y-closure closes part of the gap

Turning on eager folding (`config.eager_constant_folding._set(True)`)
AND passing `y` as a closure (not a jit input):

**DUAL3 (n=111, near-dense constant H)**:

| variant | µs | output | nse |
|---|---|---|---|
| `lineaxpr bcoo` (y input) | 6.1 | ndarray (Hessian too dense) | — |
| `jax.hessian(f)(y)` (y input) | 6.4 | ndarray | — |
| `jax.hessian(f)(y)` (y closure) | 5.7 | ndarray | — |
| `BCOO.fromdense(jax.hessian(f)(y))` (y closure) | 10.2 | BCOO | 12105 |
| `BCOO.fromdense(jax.hessian(f)(y))` (y traced input) | FAIL — ConcretizationTypeError | — | — |

**CMPC1 (n=2550, truly sparse constant H)**:

| variant | µs | output | nse | compile |
|---|---|---|---|---|
| `lineaxpr bcoo` (y input) | 7.5 | BCOO | 1192 (w/ dupes) | fast |
| `jax.hessian dense` (y input) | 1511.8 | ndarray (50MB load) | — | slow |
| `BCOO.fromdense(jax.hessian(f)(y))` (y closure) | 7.0 | BCOO | **596 (deduped)** | **~5 s** XLA fold |

**Findings**:

1. With eager folding, `jax.hessian` matches lineaxpr at the literal-load
   floor for dense output, and `BCOO.fromdense(jax.hessian(f)(y_closure))`
   *matches or beats* lineaxpr on compile-time-known sparse H — and even
   has better nse (XLA folds away structural zeros our walk preserves).

2. **But only with `y` as a closure.** With `y` as a jit input,
   `BCOO.fromdense` requires nse at trace and fails
   (`ConcretizationTypeError`). There is no workaround: nse fundamentally
   cannot be known at trace without evaluating H on a concrete y.

3. **Compile-time cost on large sparse H is non-trivial.** CMPC1 (n=2550)
   spent ~5 seconds inside XLA constant-folding a scatter of a
   2550×1×2550 dense intermediate. lineaxpr builds the BCOO at Python
   trace time in milliseconds.

**What this means for lineaxpr's unique value on constant-H problems**:

- ⚠️ "lineaxpr is the only way to get a compile-time BCOO" is overstated.
  With eager folding + closure y, existing JAX gets there.
- ✅ lineaxpr still wins when `y` is a jit input (the optimization-loop
  case: varying y at each iteration, no re-jit). This is the important
  case for sif2jax-style benchmarks and for any iterative solver that
  calls `H(y)` repeatedly.
- ✅ lineaxpr still wins on compile time for large sparse constant-H —
  no multi-second XLA fold.
- ⚠️ Our "quadratic Hessian" benchmark wins over jax.hessian depend on
  EAGER_CONSTANT_FOLDING being *off* (the default). With it on, the gap
  closes dramatically for closure-y. `benchmarks/run_in_container.sh`
  uses EAGER_CONSTANT_FOLDING=TRUE precisely to prevent spurious wins.

## 11. Short-circuit at n<16

Without the short-circuit, HS110 (n=10) costs ~15 µs (structural walk
overhead on 50 tree-prod intermediates) vs 5 µs for vmap(hvp)(eye). We
short-circuit at `primal.size < 16` to match the vmap floor.

The threshold is a hard-coded heuristic. Could become a parameter or
derived from a "walk-complexity" estimate (number of eqns / operand
shape), but 16 works empirically across CUTEst.
