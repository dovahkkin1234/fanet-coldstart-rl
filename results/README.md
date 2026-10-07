# results/ — status index

Result files accumulate faster than they expire, and several files here are
**void** (produced by code with a known bug) or **superseded** (correct at the
time, but measured at an operating point the project has since moved off).
Nothing in a filename indicates which. This index does.

Written after the pre-regeneration audit, which found nine fixes (v12–v22)
uncommitted and results from three different operating points sitting side by
side with no marking.

> **v28 (2026-10-04) — mobility fix.** Every result from episodes longer than ~40 s produced before v28 ran on networks that freeze (drones trapped at their waypoints: 37–51% of the 5–15 m/s drones and all sparse_fast / sink_50 drones by 1000 s; DATASET_V3_SPEC §0 #15). That covers the `band_*_1000s*.json` files, `grid_verification.json`, `teacher_choice/` and `bc_data_scaling/` (300 s), `dataset_v3/` and `g4_v3/`. Read them as pre-v28 until re-measured; files written after v28 record `mobility: v28-arrive-on-pass` (the re-measured bands are `band_*_v28.json`; the pre-v28 band files are kept beside them). The 40 s results are essentially unaffected except in sparse_fast. The sparse_fast caveat below (low delivery, loss neither link nor queue) is what a frozen, partitioned network looks like — re-check it on the re-measured band before citing it.

> **v29 (2026-10-06).** Bands, grid verification, teacher choice and Point-1 were re-measured on the fixed simulator; the CURRENT table lists those files, and the pre-v28 ones moved to **PRE-V28** below (kept as the record). The sparse_fast caveat was the bug: on the fixed simulator sparse_fast is congestion-limited like the others.

---

## CURRENT — safe to cite

| file | what it is |
|---|---|
| `band_dense_slow_1000s_v28.json` | Usable band (phase 1), 1000 s, battery 8000, 100–300 m, v28 mobility. Usable rates 60/80/100. |
| `band_very_dense_1000s_v28.json` | Same, very_dense. Usable rates 60/80 (100 and 120 lose ~8 nodes to the battery). |
| `band_medium_slow_1000s_v28.json` | Same, medium_slow. Usable rates 40/60/80; 80 is the top of the sweep, so the upper edge is unmeasured. |
| `band_sparse_fast_1000s_v28.json` | Same, sparse_fast. Usable rates 60/80/100 — congestion-limited (see the resolved caveat below). |
| `band_sink50_1000s_z100_300_v28.json` | Convergecast band, same operating point. Usable rates 30/40. |
| `energy_range.json` | Battery sweep that established `INITIAL_ENERGY = 8000`. |
| `teacher_choice_v28/summary.json` | Teacher-choice test on the fixed simulator (2026-10-05): pre-registered verdict SINGLE da_gpsr, unchanged. 300 s, 100–300 m, battery 8000, 10 paired seeds. |
| `grid_verification.json` | Dataset V3 grid gate (`verify_dataset_grid_v3.py`), v28 mobility; the generator refuses cells that are not PASS here and treats a file from another mobility version as verifying nothing. Re-run after any `DATASET_GRID` change (v29: 18 cells). |
| `bc_data_scaling_v28/scaling.json` | Point-1 on the fixed simulator (2026-10-06): DATA-LIMITED on the same single trigger as before (DATASET_V3_SPEC §12 item 8). |
| `panel_cc_v2_corrected.json` | Convergecast oracle panel, after the v16 delta sign-flip fix. |

### sparse_fast caveat — resolved by v28

Before v28, sparse_fast's "usable" cells had elasticity 0.80–0.86, pressed
against the `ELASTIC_HI = 0.85` ceiling, with near-zero link errors (~0.007) and
little queue overflow: its loss was neither link nor queue but unreachable
destinations. That was the trapping bug — by 300 s every sparse_fast drone was
frozen and the network stayed partitioned. On the fixed simulator its usable
cells have elasticity 0.45–0.55 and queue overflow 0.17–0.26: it is
congestion-limited like the other four scenarios, and its cells are peers of theirs.

## PRE-V28 — the trapping simulator (kept as the record, not current)

| file | what it is | current replacement |
|---|---|---|
| `band_dense_slow_1000s.json`, `band_very_dense_1000s.json`, `band_medium_slow_1000s.json`, `band_sparse_fast_1000s.json` | 1000 s bands with the phase-2 queue ablation | `band_*_1000s_v28.json` (phase 1) |
| `band_sink50_1000s_z100_300.json` | sink_50 band at 100–300 m, with the phase-2 ablation (candqueue +3.51 pp, additive +4.08 pp) | `band_sink50_1000s_z100_300_v28.json` |
| `band_sink50_1000s.json` | sink_50 band at 50–150 m (BASE), not the operating point | — |
| `teacher_choice/` | teacher-choice test, 2026-10-01: SINGLE da_gpsr | `teacher_choice_v28/` |
| `bc_data_scaling/scaling.json` | Point-1, 2026-10-04: DATA-LIMITED | `bc_data_scaling_v28/` |
| `dataset_v3/v3_*` | Dataset V3 v26 run: G3.5 v3 and audit PASS | the regeneration into `data\v3m` |
| `g4_v3/gate.json` | restated G4 on the v26 dataset: PASS, held-out cells inside the margin | `g4_v3m/` |

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
| `panel_extended_v3.json` | **CURRENT — the oracle of record** | Merges `lq_dijkstra` + `etx_dijkstra` (220 new episodes) into the contender data. Equivalence control reproduced stored values exactly before merging. Carries the final `oracle_assignment`. Ran at altitude **50–150 m** (BASE). Evaluation reference only — the Dataset V3 label is da_gpsr everywhere. |
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
