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

| Rule site                         | Line          | Category      | Notes                                                                                               |
| --------------------------------- | ------------- | ------------- | --------------------------------------------------------------------------------------------------- |
| `_add_rule` densify gate          | 738, 760, 800 | ✅ Rule 3     | `_densify_if_wider_than_dense` in all-BE same-range                                                 |
| `_add_rule` batched mix           | 881           | ⏸ tuple       | `_ellpack_to_bcoo_batched(v)` in batched BE+BCOO concat                                             |
| `_add_rule` mixed types           | 903           | ⏸ tuple       | `_to_bcoo` on `{CD, D, BE, BCOO}` shape-match fallback                                              |
| `_add_rule` linear form           | 921, 923      | specialised   | aval-() normalise to 1D                                                                             |
| `_add_rule` dense fallback        | 930           | ❌ gap        | arbitrary-mix last resort                                                                           |
| `_dot_general_rule` outer product | 1081          | inherent      | outer product is truly dense                                                                        |
| `_dot_general_rule` fallback      | 1109          | ✅ Rule 2     | `k_new ≥ in_size` gate in `_be_dot_closure_matrix`                                                  |
| `_mul_rule` dense fallback        | 308           | ❌ gap        | complex scale broadcast shapes                                                                      |
| `_slice_rule` axis=1              | 1219          | ❌ gap        | input-axis slice (filter `in_cols` at trace time)                                                   |
| `_pad_rule` interior              | 1292-1293     | ❌ edge       | interior-pad (`step>1`), rare                                                                       |
| `_pad_rule` multi-dim             | 1303          | ❌ edge       | multi-dim pad on non-batched BE                                                                     |
| `_squeeze_rule`                   | 1377          | ❌ gap        | non-trivial squeeze dims                                                                            |
| `_rev_rule`                       | 1398          | ❌ **always** | no structural reverse path — cheap to add (metadata-only: flip `in_cols` order + values along axis) |
| `_reshape_rule`                   | 1599          | ❌ gap        | unhandled reshape patterns                                                                          |
| `_broadcast_in_dim` linear-norm   | 1640          | specialised   | aval-() → (n,) 1D                                                                                   |
| `_broadcast_in_dim` fallback      | 1781          | ❌ gap        | unhandled patterns                                                                                  |
| `_reduce_sum_rule` densify        | 1989, 1994    | ✅ Rule 3     | `_densify_if_wider_than_dense` in out-axis path                                                     |
| `_reduce_sum_rule` fallback       | 2016          | ❌ gap        | non-0-axis reduction, mixed operand types                                                           |
| `_concatenate_rule` fallback      | 2205          | ❌ gap        | mixed-structural concat                                                                             |
| `_split_rule` BCOO/diag           | 2311          | edge          | non-BE operands via mask path                                                                       |
| `_split_rule` dense               | 2329          | edge          | axis != 0                                                                                           |
| `_select_n_rule` fallback         | 2583          | ❌ edge       | higher-rank pred (`pred.ndim > 1`)                                                                  |
| `_cumsum_rule`                    | 2605          | ❌ **always** | no structural cumsum — add for INTEQNELS/HS91/HS92/TENFOLDTRLS                                      |
| `_transpose_rule` fallback        | 2631          | ❌ gap        | unhandled permutation (non out↔batch swap)                                                          |
| `_gather_rule` fallbacks          | 2699, 2737    | ❌ gap        | unhandled `dnums` shapes                                                                            |
| `_scatter_add_rule`               | 2764+         | complex       | multiple conversions for scatter logic                                                              |

## Biggest remaining levers

**Tuple-deferred (would need SumLinop)** — unblocks SBRYBND (18× mix-adds),
GENHUMPS, DQDRTIC, LIARWHD, ARWHEAD. Biggest impact on top-10 losses.

**Always-dense paths** (`_cumsum_rule`, `_rev_rule`) — cheap to add
structural paths, trigger on specific problems (INTEQNELS/HS91/HS92/
TENFOLDTRLS for cumsum; to be checked for rev).

## How the per-band inner loop works

Every non-densifying rule today is vectorised over bands except
`_add_rule` partial-match (issue #1). See that issue for the trade-off
between CPU fusion and potential GPU regression.
