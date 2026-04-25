# BEllpack → BCOO / dense conversion audit

Living inventory of every site in `lineaxpr/materialize.py` that turns a
structural LinOp (BEllpack / Diagonal / ConstantDiagonal) into BCOO or
dense. Categorised by user's three accepted conversion criteria:

- **Rule 2**: `dot_general` → dense when `k_new ≥ in_size` (dense would
  be no larger).
- **Rule 3**: `add / reduce_sum` → dense when `n_unique_cols ≥ in_size`.
- **Tuple-deferred**: mixed BE + BCOO operands where a `tuple[BEllpack]`
  (SumLinop) representation would avoid promotion. Deferred pending
  the tuple-of-BE refactor.

Everything else is a **gap** — a potential optimisation where the rule
falls to dense for shapes we could in principle handle structurally.

| Rule site                         | Line          | Category      | Notes                                                                                                                          |
| --------------------------------- | ------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `_add_rule` densify gate          | 738, 760, 800 | ✅ Rule 3     | `_densify_if_wider_than_dense` in all-BE same-range                                                                            |
| `_add_rule` batched mix           | 881           | ⏸ tuple       | `_ellpack_to_bcoo_batched(v)` in batched BE+BCOO concat                                                                        |
| `_add_rule` mixed types           | 903           | ⏸ tuple       | `_to_bcoo` on `{CD, D, BE, BCOO}` shape-match fallback                                                                         |
| `_add_rule` linear form           | 921, 923      | specialised   | aval-() normalise to 1D                                                                                                        |
| `_add_rule` dense fallback        | 930           | inherent      | dense+BE = dense. ~1100 hits but unavoidable when one operand is genuinely dense matrix. Reduced indirectly by upstream fixes. |
| `_dot_general_rule` outer product | 1081          | inherent      | outer product is truly dense                                                                                                   |
| `_dot_general_rule` fallback      | 1109          | ✅ Rule 2     | `k_new ≥ in_size` gate in `_be_dot_closure_matrix`                                                                             |
| `_mul_rule` dense fallback        | 308           | ❌ gap        | complex scale broadcast shapes                                                                                                 |
| `_slice_rule` axis=1              | 1219          | ❌ gap        | input-axis slice (filter `in_cols` at trace time)                                                                              |
| `_pad_rule` interior              | 1292-1293     | ❌ edge       | interior-pad (`step>1`), rare                                                                                                  |
| `_pad_rule` multi-dim             | 1303          | ❌ edge       | multi-dim pad on non-batched BE                                                                                                |
| `_squeeze_rule`                   | 1377          | ❌ edge       | only 1 BCOO dims=(0,) sweep hit; very low impact                                                                               |
| `_rev_rule`                       | 1481          | ✅ structural | BE/BCOO unbatched dim 0: flip cols + values along row axis (BE) or remap row indices (BCOO). Dense fallback for other dims.    |
| `_reshape_rule` leading singleton | ~1670         | ✅ structural | aval `(N,)` → `(1, ..., 1, N)` — adds size-1 leading batch axes (LUKSAN16LS, commit `e5bf59c`).                                |
| `_broadcast_in_dim` linear-norm   | 1640          | specialised   | aval-() → (n,) 1D                                                                                                              |
| `_broadcast_in_dim` BE-tile       | ~1748         | ✅ structural | 1-row BE → (N,) tile: broadcast values + cols along new row axis. ~22 sweep hits.                                              |
| `_broadcast_in_dim` Diag stretch  | ~1804         | ✅ structural | `Diagonal(n)` → `(*pre_singletons, n, *post_singletons)` (commit `2da443c`). Unblocks PENALTY3 (12× faster).                   |
| `_broadcast_in_dim` fallback      | 1781          | ❌ edge       | residual unhandled shapes are rare; most 6-hit Diag patterns now structural                                                    |
| `_reduce_sum_rule` densify        | 1989, 1994    | ✅ Rule 3     | `_densify_if_wider_than_dense` in out-axis path                                                                                |
| `_reduce_sum_rule` BCOO row-sum   | ~2169         | ✅ structural | BCOO axes=(0,) static np indices: dedup cols → BE row-vector. ~24 sweep hits.                                                  |
| `_reduce_sum_rule` Diag fallback  | 2177          | intentional   | `_to_dense + jnp.sum` lets XLA DCE the (n,n) intermediate. ~14 hits; comment cites ARGTRIGLS 2.25× regress if changed          |
| `_concatenate_rule` fallback      | 2205          | inherent      | only HEART8LS; inputs are 6 dense + 2 BE so necessarily dense                                                                  |
| `_split_rule` BCOO/diag           | 2311          | edge          | non-BE operands via mask path                                                                                                  |
| `_split_rule` dense               | 2329          | edge          | axis != 0                                                                                                                      |
| `_select_n_rule` BCOO mask        | ~2540         | ✅ structural | row-mask for BCOO operand, multi-BCOO concat                                                                                   |
| `_select_n_rule` fallback         | 2583          | bit-exact     | TOINTGOR has tol=0.0; CD/D mismatched cases must use lax.select_n for bit-exactness                                            |
| `_cumsum_rule`                    | 2605          | inherent      | cumsum-on-Diagonal = lower-triangular, k=n hits densify gate anyway. Only 3 sweep hits (INTEQNELS, TENFOLDTRLS).               |
| `_transpose_rule` fallback        | 2631          | ✅ structural | no dense fallbacks observed across sweep                                                                                       |
| `_gather_rule` 1D BE              | ~2860         | ✅ structural | unbatched BE row gather: structural pick of rows + cols. Helps TOINTGOR.                                                       |
| `_gather_rule` other fallbacks    | 2699, 2737    | ❌ edge       | unhandled `dnums` shapes; rare                                                                                                 |
| `_scatter_add_rule`               | 2764+         | complex       | multiple conversions for scatter logic                                                                                         |

## Biggest remaining levers

**Tuple-deferred (would need SumLinop)** — unblocks SBRYBND (18× mix-adds),
GENHUMPS, DQDRTIC, LIARWHD, ARWHEAD. Biggest impact on top-10 losses.

**Always-dense paths** — `_rev_rule` now structural for axis=0 (commit
`faa6dfd`). `_cumsum_rule` left as densify because cumsum-on-Diagonal
becomes lower-triangular which hits the densify gate anyway.

## How the per-band inner loop works

Every non-densifying rule today is vectorised over bands except
`_add_rule` partial-match (issue #1). See that issue for the trade-off
between CPU fusion and potential GPU regression.
