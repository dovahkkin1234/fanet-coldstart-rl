# results/ — status index

Result files accumulate faster than they expire, and several files here are
**void** (produced by code with a known bug) or **superseded** (correct at the
time, but measured at an operating point the project has since moved off).
Nothing in a filename indicates which. This index does.

Written after the pre-regeneration audit, which found nine fixes (v12–v22)
uncommitted and results from three different operating points sitting side by
side with no marking.

---

## CURRENT — safe to cite

| file | what it is |
|---|---|
| `band_dense_slow_1000s.json` | Usable band + queue ablation, 1000 s, battery 8000. Usable rates 60/80/100. |
| `band_very_dense_1000s.json` | Same, very_dense. Usable rates 60/80. |
| `band_medium_slow_1000s.json` | Same, medium_slow. Usable rates 40/60/80. |
| `band_sparse_fast_1000s.json` | Same, sparse_fast. Usable rates 80/100 — **see caveat below**. |
| `band_sink50_1000s.json` | Convergecast band, 1000 s, battery 8000. Only rate 30 usable. |
| `energy_range.json` | Battery sweep that established `INITIAL_ENERGY = 8000`. |
| `panel_cc_v2_corrected.json` | Convergecast oracle panel, after the v16 delta sign-flip fix. |

### Caveat on `band_sparse_fast_1000s.json`

Its two "usable" cells have elasticity **0.80 and 0.85**, both pressed against
the `ELASTIC_HI = 0.85` ceiling, while every other scenario's usable cells sit
at **0.05–0.47**. Its `link_error` is ~0.007 (near zero) against a PDR of only
~0.30, so its dominant loss is neither link nor queue — most likely
route-unavailability, consistent with the original connectivity-limited
characterisation of this scenario. Treat its cells as qualitatively different
from the other four scenarios', not as peers.

---

## SUPERSEDED — correct when measured, wrong operating point now

These were produced at 20 s, 40 s, or 200 s durations, and/or at the default
100 battery, before the 1000 s / battery-8000 operating point was established.
The *directions* they found largely replicated at the final operating point;
the *rates* did not transfer.

| file | measured at | superseded by |
|---|---|---|
| `usable_band.json` | 20 s, default battery | `band_dense_slow_1000s.json` |
| `band_very_dense.json` | 20 s | `band_very_dense_1000s.json` |
| `band_medium_slow.json` | 20 s | `band_medium_slow_1000s.json` |
| `band_very_dense_200s.json` | 200 s, battery 4000 | `band_very_dense_1000s.json` |
| `congestion_band_convergecast.json` | 40 s | `band_sink50_1000s.json` |
| `hops_check.json`, `hops_check2.json` | 20 s | diagnostic only; not a result |
| `energy_1000s.json`, `energy_gap.json` | partial battery sweeps | `energy_range.json` |
| `regimes_1000s.json`, `operating_regimes.json` | pre-battery-fix regime maps | `energy_range.json` |
| `panel_cc_v12.json` | stale `[0.02–0.40]` rate grid | `panel_cc_v2_corrected.json` |
| `panel_cc_v2.json` | pre-v16 delta sign-flip | `panel_cc_v2_corrected.json` |

---

## VOID — produced by code with a known bug; do not cite

| file | why void |
|---|---|
| `panel_cc_small.json` | Pre-v12. Delivery-queue leak active — up to 88.4% of recorded drops were phantom. |
| `scan_congestion_window.json` | Verdicts void: the script printed `q_ovf` and `link_error` but hid `energy_depleted`, which dominated at the rates tested. It was measuring battery death, not congestion. |
| `oracle_congestion.json` | Measured entirely inside the saturation plateau (delivery pinned at a constant, independent of offered load), so its three "rates" are one operating point reported three times. |
| `probe_rate_grid_smoke.json` | Smoke-test output, never a result. |

---

## Reading any file's operating point

`provenance` inside each file reports `config_v2.BASE` — the **parity
reference** (40 s, rates `[0.5, 2.0, 4.0]`), *not* what the run used. Files
processed by `fix_result_provenance_v23.py` carry a `run_params` block with the
actual operating point. Files without one predate that fix; check the top-level
`duration` field and the generating command.

---

## ORACLE PANELS (added after the 8-teacher panel series)

All at 1000 s, battery 8000, 10 paired seeds. Full analysis in
`docs/Panel_Results_Report.md`.

| file | status | notes |
|---|---|---|
| `panel_oracle_generalized.json` | **SUPERSEDED — means valid, verdict not** | 9 teachers, 990 episodes. Per-cell means are sound. Its verdict used the incumbent-replacement rule, and the run did NOT apply the `sparse_fast` filter. Stores means only, no per-seed data. |
| `panel_contenders_v2.json` | **SUPERSEDED — use `_corrected`** | 5 contenders, 550 episodes, first file with `per_seed_pdr`. Its printed `oracle_assignment` is WRONG in 4 of 9 cells: the v2 leading-group rule let underpowered, materially worse teachers into the tied group. |
| `panel_contenders_v2_corrected.json` | CURRENT | Same data, oracle recomputed under the v25 rule (leading group = within 1 pp of top). No re-run. |
| `panel_extended_v3.json` | **CURRENT — the oracle of record** | Merges `lq_dijkstra` + `etx_dijkstra` (220 new episodes) into the contender data. Equivalence control reproduced stored values exactly before merging. Carries the final `oracle_assignment`. |
| `oracle_agreement.json` | CURRENT | Three oracles queried at every decision point on identical state. Overall 3-way agreement 0.649. Its override counter is null (wrong attribute name in the probe); the agreement figures are unaffected. |

### Final oracle assignment (from `panel_extended_v3.json`)

| cells | oracle |
|---|---|
| medium_slow @ 40, 60, 80 | dijkstra |
| dense_slow @ 60 | da_gpsr |
| dense_slow @ 80, 100 | gpsr |
| very_dense @ 60, 80 | da_gpsr |
| sink_50 @ 30 | da_gpsr |
| sparse_fast @ 80, 100 | flagged — no oracle set |

Whether the dataset uses these per-cell teachers or a single global teacher is
an open design decision; see the report, section 7.
