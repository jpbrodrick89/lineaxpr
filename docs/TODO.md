# TODO — prioritized

Empirically-grounded; see `RESEARCH_NOTES.md` for reasoning / history.

## Handoff context

Key files:

- `lineaxpr/materialize.py` — rules + public API
- `lineaxpr/_base.py` — LinOp classes (ConstantDiagonal, Diagonal, BEllpack)
- `tests/test_sif2jax_sweep.py` — sweep test with SIZE_OVERRIDES + KNOWN_UNIMPLEMENTED + `nse <= n*n` invariant
- `tests/nse_manifest.json` — per-problem nse baselines (regen via
  `uv run python -m tests.update_nse_manifest`)
- `benchmarks/plots.py` — `--problem-filter unconstrained` for
  publication plots

Rule patterns that have been load-bearing (keep in mind when writing
new rules):

- **Sparsity-preserving structural paths** — `mul(BE, dense)` etc.
  preserve per-batch-element nse; replicate pattern over a new batch
  when size-1 axes are expanded. See `_mul_rule` batch-expand and
  `_add_rule` batch-broadcast.
- **`_densify_if_wider_than_dense`** — fires when `k >= in_size`.
  Prevents BE from carrying effectively-dense state. Applied at
  `_reduce_sum_rule` (out-axis) and `_add_rule` (band-widen) emissions.
- **`_bcoo_concat` uses np.concatenate** for indices when possible —
  avoids XLA iota-decompression of tile-able constants.
- **CLAUDE.md rule**: never Python-loop over jnp arrays. Exception:
  `_ellpack_to_bcoo` loop-form stays for large nrows (k-gate at
  `_BE_TO_BCOO_LOOP_K_THRESHOLD = 3`) — see commit e82ab33.

Sweep-contamination awareness: sweep-level "regressions" are often
false positives — always isolate-verify before believing. See
CLAUDE.md "Sweep vs isolated min". EIGEN\*, DMN15102, LUKSAN16LS have
all been false positives this session.

## Current losses to `asdex_bcoo` (sweep 57c3093)

Problems where `bcoo_hessian` is meaningfully slower than `asdex_bcoo`
(>1.1× ratio, >5µs absolute). Check before committing new work.

| Problem    |    n | ratio |   bcoo | asdex | presumed cause                                 |
| ---------- | ---: | ----: | -----: | ----: | ---------------------------------------------- |
| BROYDN7D   | 5000 |  128× | 26.7ms | 210µs | dense fallback — 7-diag stencil densifies      |
| BDQRTIC    | 5000 |   40× |  6.3ms | 157µs | dense fallback; `D` quartic → wide-k BE        |
| LUKSAN16LS |  100 |  4.8× |   91µs |  19µs | effectively-dense Hessian; BE overhead         |
| GENHUMPS   | 5000 |  2.7× |  536µs | 202µs | dense fallback in HVP chain                    |
| NONMSQRT   | 4900 |  2.1× |  649µs | 308µs | k=71 emit path — partially structural but slow |
| DQDRTIC    | 5000 |  1.9× |   16µs |   8µs | small absolute; call-overhead dominance        |
| BROYDN3DLS | 5000 |  1.8× |  149µs |  81µs | similar stencil-densify as BROYDN7D            |
| SBRYBND    | 5000 |  1.8× |  1.6ms | 894µs | banded system walker produces wide-k BE        |
| LIARWHD    | 5000 |  1.4× |   49µs |  35µs | closed mostly (2026-04-22); residual small     |
| LUKSAN17LS |  100 |  1.3× |   50µs |  38µs | partial structural coverage                    |
| ARWHEAD    | 5000 |  1.3× |   53µs |  40µs | small absolute                                 |

Dominant patterns: (1) 5-diagonal / 7-diagonal stencil densification
(BROYDN7D/BDQRTIC/BROYDN3DLS — likely the 0c transpose / 0d 2D-gather
territory), (2) wide-k BE emission where smart-densify could be more
aggressive (SBRYBND, NONMSQRT), (3) small-n call-overhead where
coloring might win (LUKSAN16LS, DQDRTIC, 0f covers this).

## Priority 0 — open

### 0c. ~~Structural `_transpose_rule` for BEllpack~~ **DONE** (commit ac1a462)

Landed with an additional `_add_rule` batched-BCOO-concat branch to
stop row-range-mismatched batched BEs from rank-collapsing via the
flat `_to_bcoo` path. Followups below.

### 0j. ~~Unify `_ellpack_to_bcoo{,_batched,_keep_batch}` → always-batched~~ **DONE**

Landed post-0c. `_ellpack_to_bcoo_batched` now preserves `n_batch` (no
longer flattens to unbatched 2D). The semantic asymmetry is gone —
batched BEllpack → batched BCOO, unbatched BEllpack → unbatched BCOO.
Walker's existing final reshapes (`_reshape_rule` batched-BE →
unbatched-BE at line 1293 and batched-BCOO → unbatched-BCOO at line 1411) handle the final collapse at the correct point.

Sweep: 323 passed, 54 skipped, 0 failed (no walker-level regression —
every problem's final reshape correctly unbatches). Manifest diff:
none (DRCAV etc. stay at their existing nse since the batched BCOO
gets flattened by the walker, not by conversion).

### 0k. ~~`_cond_rule` tracer-index support~~ **DONE**

Turned out the HADAMALS cond is a `lax.platform_dependent`, whose
branches are semantically equivalent by contract. The eqn carries
`branches_platforms` — we detect it and pick the `None` (default)
branch as a fallback when concrete indexing fails. Zero densification.
Also dropped the redundant `int(np.asarray(...))` in favour of direct
tuple indexing via `__index__`. `UNFOLDED_UNSUPPORTED` is now empty.

### 0l. ~~Structural `_dot_general_rule` for BEllpack × closure-matrix~~ **DONE**

Landed (commit TBD). `_be_dot_closure_matrix` handles out-axis and
trailing-batch-axis contracts; respects `traced_is_first` (transposes
canonical output for closure-first operand). Upfront gate
`k_old * A >= in_size` returns `None` (→ dense fallback) when the
structural form would be no smaller than dense.

Isolated perf (clean serial runs, min-of-5 × 10 iter):

| Problem  | fold Δ | unfold Δ |
| -------- | -----: | -------: |
| HADAMALS |   -17% |      +8% |
| EIGENA2  |   -19% |     -24% |
| EIGENALS |   -44% |     -38% |
| EIGENBLS |   -44% |     -40% |
| EIGENCLS |   -42% |     -43% |
| MSQRTALS |   -53% |     -52% |
| MSQRTBLS |   -51% |     -52% |

The EIGEN/MSQRT family benefits most — their `Q.T @ Q`-style pattern
hits two `BE × closure_matrix` dots per iteration. Wins persist in
both folded and unfolded. HADAMALS unfolded +8% is within noise band
(n=400 is tiny). No correctness regressions (323 pass / 0 fail).

### 0d. Structural 2D point-gather / scatter-add

**Current**: `_gather_rule` + `_scatter_add_rule` have dense fallback
paths for `collapsed_slice_dims=(0, 1)` / `inserted_window_dims=(0, 1)`
(commit bc46c45). Correct but lossy on sparsity.

**Fix**: when operand is BEllpack with batch shape, `M[b[i,0], b[i,1]]`
= pick batch=`b[i,0]`, out-idx=`b[i,1]`, keep input axis. Emit a new
BEllpack with `batch_shape=()`, `out_size=N` (gather length),
values/cols gathered along the (batch, out) axes. Scatter-add is the
dual.

**Dependency**: only useful after 0c (transpose structural) lands —
currently the 2D gather's operand arrives dense because `transpose`
upstream densifies. If 0c doesn't happen, 0d has no beneficiary.

### 0f. Tiny-n `hessian`/`bcoo_hessian` overhead (~1µs vs `jax.hessian`)

**Observed** (2026-04-23 sweep): 147 problems with n ≤ 16 are ~1µs
slower than `jax.hessian`. All have <10µs baseline runtimes, so the
ratio looks bad (1.2×) but absolute gap is tiny.

**Root**: our `hessian(f)` calls `jax.linearize(jax.grad(f), y)` before
`materialize(lin, y, format)`. For n < `_SMALL_N_VMAP_THRESHOLD=16`,
materialize short-circuits to `vmap(linear_fn)(eye)` — same as
jax.hessian internally. The linearize step itself is pure overhead
for tiny-n.

**Fix**: in `hessian`/`bcoo_hessian` public wrappers, bypass linearize
for n < threshold and call `jax.hessian(f)(y)` directly (+ `to_bcoo`
for bcoo flavour). ~10 LoC. May also allow relaxing the threshold.

**Also**: `bcoo_hessian` has ~2-3µs floor vs `hessian` even after the
short-circuit — attributable to BCOO pytree return overhead at the
jit-dispatch boundary. Non-compile-amortisable. Possibly a case to
just document rather than fix.

### 0h. `_add_rule` batch-broadcast side-effects on tridiag-class problems

**Context**: the batch-broadcast path in `_add_rule` (4ff0881, mirrors
`_mul_rule`'s batch-expand) gave DMN15103LS −18% (52→43ms) but
introduced real isolated regressions on tridiagonal-structure problems
vs a743104:

| Problem    | a743104 | 57c3093 | Δ    |
| ---------- | ------- | ------- | ---- |
| DIXMAANB   | 32.8 µs | 39.3 µs | +20% |
| DIXMAANL   | 32.6 µs | 40.0 µs | +23% |
| BDEXP      | 68.0 µs | 80.8 µs | +19% |
| LUKSAN17LS | 50.6 µs | 55.5 µs | +10% |
| ARGTRIGLS  | 87.4 µs | 94.1 µs | +8%  |

All are small absolute (~5-12µs) but a consistent direction. Geomean
still improves (bcoo/jax: 1.30 → 1.35) because DMN/EIGEN/LUKSAN11-15
wins dominate. Still worth tracking down — batch-broadcast might be
firing for operands where the old non-matching-batch-shape fallback
was faster.

**Investigate**: instrument `_add_rule` to log when batch-broadcast
fires; check what shapes DIXMAANB passes. If the path fires on
already-matching batch shapes (shouldn't, per check), that's a bug.
If it fires on genuinely-mismatched shapes that were previously
densifying cleanly, gate more tightly.

### 0g. Extend SIZE_OVERRIDES upstream in sif2jax

**Context**: 45 problems still skip the sweep on size with no
override. Most are constrained-quadratic problems (CVXQP1-3,
NCVXQP1-9, NCVXBQP1-3, CVXBQP1, OBSTCL\*, QPBAND/QPNBAND, A0E\*/A0N\*,
GOULDQP2/3, JUNKTURN, 10FOLDTRLS) where the class hardcodes `n` via a
`@property` — no size constructor arg. Adding `n: int = default` as a
constructor kwarg per class would unlock all of them.

**Fix**: sif2jax PR adding constructor size kwargs to the CVXQP /
NCVXQP / bounded-quad families. Mechanical — each class's `n`
property returns a hardcoded int; change to `n: int = <current>` field.

### 0i. True `scan` structural support (promoted from Priority-4)

**Context**: some CUTEst problems emit `lax.scan` in their linearized
HVP. Currently the walker raises NotImplementedError on scan, which
surfaces as walk errors in the sweep. Short-circuit (`n < 16` →
`vmap(linear_fn)(eye)`) hides most cases but not all.

**Proposal**: add a `_scan_rule` that either (a) unrolls the scan body
into a single-iter walk and post-multiplies, or (b) lifts to a
batched BE over the scan length. Needs prototyping.

**Why priority-0**: real user problems hit this; sif2jax growth
increases exposure. User moved this from Priority-4 (2026-04-23).

## Priority 1 — structural improvements

### 1. Deferred `Sum` LinOp for order-independent add-chain bucketing

**Motivation**: the heat-equation test in `tests/test_materialize.py`
bloats a 3n-2 tridiagonal to 2.66× dense. `pad(X, (0,1)) + pad(X, (1,0))`
hits different `(start_row, end_row)` ranges, forces an early BCOO
promote, and the remaining chain piles up concat duplicates. Walk
becomes order-sensitive.

**Proposal**: `Sum(operands: tuple[LinOp, ...])` LinOp that `_add_rule`
returns instead of eagerly BCOO-promoting mismatched kinds. Unary
rules (mul/neg/pad/slice/squeeze/rev) distribute through Sum by
pushing into each operand. Densifying rules (dot_general, scatter_add,
reshape, broadcast_in_dim, reduce_sum, dense fallbacks, final walk
output) flush via `_flush_sum` — bucket operands by matrix shape +
`(start_row, end_row)`, emit per-bucket Ellpacks, BCOO-concat.

**Expected win**: heat-equation 56 → 28 for n=8. Plus potential
recovery of FREUROTH (+51%) and BDEXP (+25%) Ellpack regressions
(2026-04-20 — different-cols widen emits 2 broadcast_in_dim + concat
that Old Pivoted did in 1 op).

**Affects**: new class in `_base.py`, new `_flush_sum` helper,
~8 rule touches. Any path reading operand attributes directly (e.g.
`_linop_matrix_shape`) must handle Sum or be flushed first.

**Ordering vs CSR (#2)**: start with CSR — it partially subsumes Sum
(disjoint-range union is natural in CSR).

### 2. Internal CSR LinOp for arrowhead / disjoint-row adds

**Motivation**: `Ellpack`'s `(start_row, end_row)` + uniform `k` forces
padding or BCOO promotion when rows have very different entry counts
(classic arrowhead: one dense row, many unit rows → padded `k = n`).
`_scatter_add_rule` and mismatched-range `_add_rule` fall through to
BCOO concat, dropping structural info downstream rules could exploit.

**Proposal**: `Csr(row_ptr, col_ind, values, out_size, in_size)` that
stays structural through arbitrary scatter-add, disjoint-range add,
and arrowhead patterns. Also natural output target for
`materialize(..., format='csr')` — direct hand-off to cuDSS without a
reconstruction pass.

**Affects**: new class in `_base.py`, `_to_csr` helpers,
`_scatter_add_rule` emits Csr instead of BCOO, `_add_rule` gets a
mixed-Ellpack → Csr path, `materialize` accepts `format='csr'`.

**Win**: unlocks arrowhead problems that currently densify; cuDSS-
native output.

### 3. Test infra follow-ups

- **Sweep runtime**: the slow sweep (~4 min) now exercises 323
  problems; grows as overrides are added. If it becomes a bottleneck,
  split into a fast subset (LUKSAN, DMN, NONCVX, NONMSQRT canaries)
  and a full mode.
- **Publication plot refinement**: `--problem-filter bounded-min`,
  `--problem-filter bounded-quad`, etc. already supported by the
  current CLI. Consider a single-shot `--all-filters` variant for
  batch-generating all 4 variants.

## Priority 2 — JAX idiom alignment

### 6. `safe_map` / `safe_zip` from `jax._src.util`

Replace raw `zip()` calls to catch length mismatches early.

**Why not `strict=True`**: unknown. Candidate reasons — `safe_map` /
`safe_zip` may predate Python 3.10's `strict=True` (added then); JAX
internals widely use the `jax._src.util` helpers and matching the
style keeps diffs vs upstream small; or `safe_zip` may provide a
better error message ("lengths: x vs y") than strict-zip's
`ValueError`. No strong reason to prefer one over the other — if
you're touching a zip call, either is fine; don't churn existing
code unless fixing a real bug.

### 8a. jax-style kwargs on jacfwd/jacrev/hessian

Match `jax.jacfwd` / `jax.jacrev` / `jax.hessian`'s full signature:

- `argnums: int | Sequence[int]` — differentiate w.r.t. multiple args
- `has_aux: bool` — allow `f` to return `(output, aux)`; aux passes
  through untouched
- `holomorphic: bool` — complex-valued inputs/outputs
- `allow_int: bool` — allow integer inputs (result identically zero)

Today our wrappers hard-code `argnums=0`, single-output, float, no aux.
Low priority until a user hits one; the building block (`materialize`
on a traced linear_fn) doesn't care.

### 8b. Multi-input / multi-output + vmap composition

Currently `sparsify(linear_fn)(seed)` rejects multi-input and
multi-output linear fns, and the walk hasn't been tested against
`jax.vmap`-composed callers. Either extend the walk to flatten
multi-output → 1D → walk → reshape, OR document as a hard limit.
Also: verify `vmap(hessian(f))(batched_y)` works (should — walk runs
once at trace time, vmap batches the compiled result).

### 9. `custom_jvp_call_p` rule

Audit sif2jax for `@jax.custom_jvp` usage; add a rule that `lift_jvp`s
the custom-JVP'd function and re-runs through our walk. Confirmed
2026-04-20: sif2jax had 0 `custom_jvp` calls then, so this is not
blocking. Revisit if a user reports it or a new problem family hits it.

### 10. Primitive-coverage gaps (PALMER family, BQP1VAR)

**Historical context**: --full sweep at commit `4562b8f` found 18
`bcoo_jacobian` compile failures in PALMER1A/1C/1D, PALMER2A/2C/2E,
PALMER3A/3C/3E, PALMER4C/4E, PALMER6C/6E, PALMER7C/7E, PALMER8C/8E
and BQP1VAR (n=1 degenerate shape).

**Status check needed** (2026-04-23): PALMER\* problems appear in the
current sweep (PALMER3C, PALMER4C, PALMER6C, PALMER8C in regression
lists — they're passing correctness). Check whether the current
manifest has PALMER entries — if yes, coverage is addressed. If no,
the n<16 short-circuit may be hiding them (all PALMERs are n≤15).
BQP1VAR (n=1) similarly hidden by short-circuit. Verify by setting
`_SMALL_N_VMAP_THRESHOLD=0` and running PALMER3C + BQP1VAR; if they
fail, the missing primitive is named in the error.

## Priority 3 — memory / hygiene

### 11. Last-use analysis in `_walk_jaxpr`

`env` retains every intermediate LinOp until walk ends. For long
jaxprs with dense fallbacks, retained dense tensors can OOM. Compute
`last_use` per var, `del env[v]` after its last read. No current
reports of OOM; pre-emptive hygiene.

### 12. Per-rule unit tests

**Status check needed** (2026-04-23): `tests/test_ops.py` covers
LinOp methods (unit); `tests/test_materialize.py` is hand-rolled
end-to-end; `tests/test_sparsify.py` covers transform-level.
No dedicated per-rule test file exists — but end-to-end tests are
fairly granular. Verdict: partially addressed. Worth revisiting only
if a rule refactor causes a cross-problem regression that would have
been caught by isolated rule tests.

## Priority 4 — deferred / low ROI

### 15. Cross-eqn pattern matching for prod-tree (HS110)

HS110's `(∏x)^k` HVP fragments into many small Ellpacks that our walk
processes one-by-one. asdex also doesn't fix this. Analytical form is
`α · uuᵀ + diag(...)`. Detecting this from the jaxpr structure would
need multi-eqn lookahead — substantial new machinery. Currently
handled acceptably by the `n<16` short-circuit. Wait for a large-n
prod-tree problem to justify.

### 16. Symmetric output option

`materialize(fn, primal, symmetric=True)` that computes lower triangle
and mirrors. ~30% dense-path savings. No BCOO win. Requires detecting
or asserting symmetry — for Hessians it's guaranteed, but adds a new
API surface. Low priority.

## Priority 5 — densifying rules remaining audit

Per `RESEARCH_NOTES.md` §10 audit of rules that unconditionally
densify where a structural alternative is plausible:

- **`_dot_general`**: currently densifies when both operands traced.
  Could stay BE×BE → BE via sparse outer-product bucketing when
  operands are row-orthogonal; hard case.
- **`_broadcast_in_dim`**: has several structural paths already.
  Remaining case: leading-dim broadcast on BCOO operand (currently
  dense fallback). Rare.
- **`_reshape_rule`**: has batch+out flatten + inverse-flatten
  structural paths. Remaining edge cases around reshapes that cross
  the in_size axis don't have natural BE representation.

None of these have a clear isolated beneficiary yet. Wait for a
problem to motivate the work.

## Recently landed (2026-04-23, `a96520a..57c3093`)

- `_mul_rule` batch-expand + `_add_rule` batch-broadcast (mirror
  pair). Structural `mul(BE, dense)` and `add(BE, BE)` when one side
  has size-1 batch axes.
- `_densify_if_wider_than_dense` helper. Gates wide-k BE emissions at
  `_reduce_sum_rule` and `_add_rule`.
- `_bcoo_concat` np.concatenate for static indices (avoids XLA
  iota-decompression).
- `_ellpack_to_bcoo` k-gate for traced-cols branch.
- `_cond_rule` for closure-index cond (HADAMALS prep).
- 2D point-gather/scatter-add dense fallback (HADAMALS coverage).
- `_split_rule` batched-BE axis=n_batch.
- Test infra: SIZE_OVERRIDES (28 problems covered), KNOWN_UNIMPLEMENTED
  (CURLY/SCURLY conv_general_dilated), drop-skip-on-NotImplementedError,
  `assert nse <= n*n` invariant, `--problem-filter unconstrained`
  for plots.

Main wins vs baseline (e8fd7d5): LUKSAN11-15 (3-6×), DIXMAAN B/C/D
(1.7-1.9×), HADAMALS newly covered, DMN15102/3 fully recovered with
ac3c7a6+4ff0881. No significant regressions after isolate-verification.

---

Anything not listed here that comes up in sweep analysis: isolate-
verify first (CLAUDE.md), then check if it's a new pattern or a
re-occurrence of one of the closed items above.
