# Dataset V3 — Specification

**Status:** v26 implemented and run in full (2026-10-04: 800 episodes, G3.5 v3 and audit PASS) — then **invalidated by the waypoint-trapping bug (§0 #15), fixed in v28.** v29 (2026-10-06): the bands were re-measured on the fixed simulator and `DATASET_GRID` follows them (18 cells, §2); teacher choice and Point-1 were re-run. **Regenerate before any use (§10).**
**Written:** 2026-10-02/03. **Supersedes:** `docs/DATASET_V2_MASTER_SPEC.md` (in repo),
`M3.5_Dataset_Schema.md` and `LOAD_DENSITY_DESIGN.md` (never committed). Where this document
and an older one disagree, this one is current; §11 lists every statement it overturns.

**Code:** `src/generate_dataset_v3.py` (generator) · `src/teacher_pickers_v3.py` (labels,
behaviour, votes) · `src/preflight_dataset_v3_check.py` (G3.5 v3) · `src/audit_dataset_v3.py`
(independent audit) · `src/verify_dataset_grid_v3.py` (grid gate) · `src/export_phaseb_v3.py`
(training export) · `src/rollout_gate_v3.py` (G4 check 4, restated) ·
`src/test_bc_data_scaling.py` (how much imitation data is worth training on) ·
edits to existing files via `apply_dataset_v3_v26.py`.

---

## 0. Why V3 — the measurements that changed the plan

Every row was measured (sandbox runs at the new operating point unless stated); each one
invalidated part of the V2 plan.

| # | Finding | Measurement | Consequence |
|---|---|---|---|
| 1 | **One teacher, da_gpsr** | Teacher-choice test (2026-10-01, 300 s, 10 paired seeds): the student trained on dijkstra labels is **3.97 pp worse** than the one trained on da_gpsr labels in medium_slow; mixing teachers had **no** effect on the dense cells (+0.10 / +0.02 pp); a da_gpsr student matches its teacher (+0.45 / +0.10 / +0.02 pp); holding medium_slow out costs nothing (−0.04 pp). **Re-run on the fixed simulator (v28, 2026-10-05): same verdict** — dijkstra student −4.75 pp; mixing +0.12 / +0.03 pp; a da_gpsr student vs its teacher +0.47 / −0.05 / −0.01 pp; holdout −0.00 pp | label = da_gpsr everywhere; no `label_teacher` column; per-cell table stays an *evaluation* reference |
| 2 | **Episodes are huge** | 0.7–1.2 M routing decisions per 1000 s episode at the band rates (V2's whole dataset: 533 k). V2 keeps ~2.1 KB of Python per decision in the parent process | per-episode shards written by the workers; nothing accumulates in the parent |
| 3 | **Most decisions are copies** | 77–86% of decisions repeat a context (frame, node, destination, hop count, candidate set) with an identical label and identical features **except the live own queue (#14)**; 132–268 k distinct contexts per episode; 52–71% of distinct contexts occur once, and 91–100% of those come from ε detours (recovery states) | keep every distinct context once + its multiplicity + its own-queue histogram (together lossless for imitation); a 3% packet sample would have kept only 10–16% of the contexts |
| 4 | **Own queue was dead** | frame-start snapshot non-empty in 6–36% of decisions vs 68–89% live (very_dense 10% vs 76%, sink_50 6% vs 75%); the query column was a byte-for-byte copy of `node_feat[current].queue_occupancy` | `own_queue_live` read at decision time (schema v6) |
| 5 | **Live neighbour queues would change the teacher** | da_gpsr picks a different neighbour on 10–18% of decisions (39% in sparse_fast) if it reads live occupancy instead of the snapshot every panel measured | neighbour / 2-hop queue signals stay the frame-start snapshot (also what a real node learns by beacon) |
| 6 | **The drop question is real** | destination unreachable at decision time: medium_slow@60 10.6%, sparse_fast@100 **70%** (queue overflow 0.42 there); 84% of dijkstra's medium_slow edge was dropping such packets | drop action encoded (−1) and validated, not generated; decide drop/hold in the M5 environment contract |
| 7 | **Bands under the behaviour policy** | under restricted da_gpsr at 100–300 m / 1000 s: dense_slow@60 queue overflow 0.019, sink_50@30 0.015 (reference actor: 0.099 / 0.144); very_dense@60 0.041 | the band stays defined under the reference actor (§2); da_gpsr overflow is reported, never used to move cells |
| 8 | **sink_50 band at the wrong altitude** | `band_sink50_1000s.json` provenance: z 50–150 (the convergecast band script had no altitude flags) | sink cells UNVERIFIED until re-measured at 100–300 m; the script now has `--z_min/--z_max` |
| 9 | **`BUFFERED_REF` sized on leaky data** | sized at 500 on pre-v12 data (phantom slots); post-fix raw maximum 209 (medium_slow@80; p99.9 ≤ 171; dense_slow@100 186, sparse_fast@100 187, very_dense@80 101, sink_50@30 44 — one 1000 s seed each): 500 used ≤ 42% of the range | `BUFFERED_REF_V6 = 300` (~40% headroom; v5 keeps 500) |
| 10 | **Link-lifetime saturation is structural** | share of edges at the 60 s cap stays 45–73% at every time of a 1000 s episode — `estimate_link_lifetime` returns the cap whenever the pair is not separating | the "1000 s unblocks `estimated_link_lifetime`" claim is withdrawn; the feature is a censored value by design; `LIFETIME_REF` unchanged |
| 11 | **The planned action check was wrong** | a uniform ε pick equals the label 6.5–51% of the time (16.6% in a real generator run) → the planned `behaviour_deviated == (action != label)` check fails on ~1.7% of correct rows | store `eps_fired`; check non-ε actions against the behaviour choice |
| 12 | **Gates that did not do what they said** | G3.5 v2 check 7 "reproducibility" only checked the seed list; audit C would flag correct da_gpsr labels as INCONSISTENT; per-episode metadata was collected and discarded; the node-id invariant was defined but never called; strict queue-slot conservation was never enabled | all fixed in the v3 gates / generator |
| 13 | **Isolated-node packets lose their queue slot** (simulator quirk, reported, not fixed) | a packet at a node with no neighbours is removed from its queue by the run loop and never re-enqueued; it bypasses FIFO and does not count toward occupancy. Rare: 13 packets of 70 k (0.02%) in sparse_fast@100, 0 in dense_slow@80 (150 s) | documented; changing the simulator would invalidate every panel — leave until a decision is taken |
| 14 | **The live own queue differs between a context's copies** (found in the v26 recheck) | the context key cannot contain `own_queue_live`: with it, 53–81% of decisions would be distinct (3–7× the contexts). Storing only the first occurrence's value (the first v26 draft) is biased: the first decision of a frame meets the node's fullest queue — in the smoke (all 460 k decisions) the own queue differs from the first occurrence's in 57–74% of decisions per cell, the first occurrence is higher by 4–12 packets on average, and the decision-weighted mean would have been 22.7 packets instead of the true 14.0 | every context also stores the exact histogram of the own-queue lengths its occurrences saw (`o_ctx/o_ownq/o_count`, ~5 B per pair); the export draws the own queue from it (§8); the first-occurrence value in `c_query` is never trained on |
| 15 | **Drones get trapped at their waypoints** (found 2026-10-04, after the full v26 run; in `mobility.py` since the first commit) | a mobility step is speed × 0.5 s (2.5–25 m) but arrival needs < 1 m, so a step that passes the waypoint without landing within 1 m starts an endless overshoot oscillation. Trapped at 300 s / 1000 s: 12–19% / 37–51% of the 5–15 m/s drones, 100% / 100% of sparse_fast, 91% / 100% of sink_50; a trapped drone jumps a full step every frame (fake velocities, flickering links). In the v26 dataset 38–46% of sparse_fast seeds deliver < 5% of packets. Same episode with the fix: sparse_fast@40 seed 106 PDR 0.021 → 0.352, seed 101 0.486 → 0.425; sink_50@30 seed 136 0.223 → 0.728 (link-error drops 55% → 26%); dense_slow@60 seed 101 0.682 → 0.684. At the 40 s parity point the slow scenarios are essentially unchanged (≤ 0.14 pp; G2 anchors identical); sparse_fast is not (+10.8 pp on average) | v28: a step that would pass the waypoint arrives; `MOBILITY_VERSION` and a code signature are recorded by the band scripts, the shards and the tests, and every gate refuses a mismatch. Bands re-measured, dataset regenerated (§10) |

---

## 1. Decisions (locked 2026-10-02 unless marked open)

| ID | Decision |
|---|---|
| V3-1 | **Label teacher: da_gpsr, globally**, through `teacher_pickers_v3.da_gpsr_pick` (full-graph semantics, choice restricted to unvisited candidates, ties broken in `G.neighbors` order = ascending node id). No destination short-circuit — exactly as the real teacher (it scores an adjacent destination like any candidate; skipped in 0.66% of adjacent-destination decisions in a very_dense smoke run, counted per episode). The per-cell oracle table (`panel_extended_v3.json`) remains the **evaluation** reference for D2, not a label source. |
| V3-2 | **Feature schema v6.** `own_queue_live` (query column 2) = the deciding node's queue occupancy at decision time, excluding the packet being forwarded. Everything about neighbours stays the frame-start snapshot. `neigh_buffered_packets` normalised by `BUFFERED_REF_V6`. v5 is still readable, only when a reader asks (`accept_legacy=True`) and then with v5 semantics. |
| V3-3 | **Drop action:** code −1 reserved in the schema and validated by the checks; no data-generating policy drops. |
| V3-4 | **Behaviour mix:** 70% of episodes driven by da_gpsr, 30% by gpsr / spbp / dijkstra_nodrop (10% each), assigned by **seed**, stratified per split (§6). ε = 0.10 uniform detours in every episode. |
| V3-5 | **Grid:** usable band + one low-load anchor per scenario, 16 cells (§2). Band defined under the band search's reference actor; buckets scenario-relative. |
| V3-6 | **Storage:** per-episode shards; *contexts* (every distinct context once, with multiplicity and the histogram of the own-queue lengths its occurrences saw) + *steps* (every hop of a 10% packet sample) + sampled *packets* + every *frame*. |
| V3-7 | **Operating point explicit:** `config_v2.OPERATING_POINT` (1000 s, 100–300 m, battery 8000). `BASE` stays the 40 s parity reference until v8b (D-12 unchanged; v8b's anchors verified still to match). |
| V3-8 | **medium_slow held out** by scenario (generalisation); D1–D5 are in-distribution — state it. Its labels are da_gpsr like everywhere else. |
| V3-9 | **sparse_fast labelled with da_gpsr** for consistency; the stored votes (dijkstra / gpsr / spbp) allow a different labelling later without regenerating. |
| V3-10 | **Suite C (sink_50) generated like any cell.** The frozen "30% of training" is a training-time weight, re-decided once sink_50's band is re-measured — **open**. |
| V3-11 | **No simulator checkpoints in the dataset.** Episodes are deterministic (reproducibility gate, §7), so any mid-episode state can be recreated from (scenario, rate, seed, behaviour). Checkpointing becomes an M5 environment feature with its own resume-equivalence gate. `checkpoint_id` dropped. |
| V3-12 | **Imitation weighting:** context multiplicity by default (= the raw decision stream the teacher-choice test validated); uniform-over-contexts is an ablation. **How many rows to train on** is set by `test_bc_data_scaling.py`: DATA-LIMITED → as many as the RAM ceiling allows (~45–50 M rows at `--max_gb 22`, §9); SATURATED → the smallest size it names — **open until it runs**. The dataset on disk stays lossless either way. |
| V3-13 | **G4 check 4 restated** (pre-registered, §8): paired non-inferiority against restricted da_gpsr, 1 pp margin, every training cell. |

---

## 2. Operating point and grid

`OPERATING_POINT = {z 100–300 m, duration 1000 s, drain 10 s, interference on, battery 8000}`.

| scenario | suite | anchor (bucket *low*) | band rates (bucket *medium* … *high* = top) | verification |
|---|---|---|---|---|
| dense_slow | A | 40 | 60, 80, **100** | PASS |
| very_dense | A | 40 | 60, **80** | PASS |
| medium_slow (held out) | A | 30 | 40, 60, **80** | PASS |
| sparse_fast | A | 40 | 60, 80, **100** | PASS (60 added in v29) |
| sink_50 | C | 20 | 30, **40** | PASS (40 added in v29) |

* **band** = rates marked usable in `results/band_*_1000s.json` (criteria: elasticity 0.05–0.85,
  queue overflow ≥ 0.02, energy share ≤ 0.05, no dead nodes) under `spbp_ab_noqueue`.
  It is a property of the *load*, fixed with a fixed reference policy; re-defining it under the
  behaviour policy would let a better policy move the band (and D2 must "happen naturally").
* **anchor** = the lowest rate of that scenario's band sweep — below the band on purpose, for
  the low-vs-high contrast the cold-start claim needs.
* `rate_index`, `rate_role`, `load_bucket_rel` live in `config_v2`. Bucket labels are not
  comparable with any pre-v26 result (absolute thresholds 0.5 / 2.0 on the old grid).
* `verify_dataset_grid_v3.py` writes `results/grid_verification.json`; the generator refuses
  any cell that is not PASS unless `--allow_unverified_cells` (recorded in the manifest).
* **sink_50:** measured at 100–300 m twice — before v28 only rate 30 was usable; on the fixed
  simulator 30 and 40 are (v29).
* **v28:** a band file counts only if it records `mobility = v28-arrive-on-pass`; the verifier refuses older files and prints the five re-measure commands (`--phase1_only`: which rates are usable is all the grid needs). They write `band_*_v28.json` beside the pre-v28 files, which are kept for comparison. `grid_verification.json` records the mobility too, and the generator treats a pre-v28 one as verifying nothing. If a band moves, `DATASET_GRID` follows it by the rule above.
* **v29 (2026-10-06):** the five bands re-measured on the fixed simulator (`results/band_*_v28.json`, phase 1, 3 map seeds): dense_slow, very_dense and medium_slow unchanged; **sparse_fast gained 60** — its elasticity fell from 0.80–0.86 (against the 0.85 ceiling, near-zero link errors: unreachable destinations in frozen, partitioned networks) to 0.45–0.55 with queue overflow 0.17–0.26, i.e. it is now congestion-limited like the other four; **sink_50 gained 40**. `DATASET_GRID` follows: 18 cells, 900 episodes. Buckets are scenario-relative, so sink_50@30 is now *medium*. medium_slow's top swept rate (80) is still usable, so its band's upper edge remains unmeasured. Re-run the verifier on the 18-cell grid before generating.

---

## 3. Files

```
data/v3/
  manifest.json                    versions, feature lists, grid, behaviour table, codebooks,
                                   per-scenario norm constants, episode index, totals
  shards/<scenario>/<scenario>_r<rate>_s<seed>.npz   one episode (all arrays, §4)
  shards/<scenario>/<scenario>_r<rate>_s<seed>.json  episode meta: config, norm constants,
                                   metrics, drop shares, counts, timings, SHA-256 per array
  preflight_v3.json, audit_v3.json gate outputs
results/dataset_v3/
  <data folder>_preflight_v3.json, <data folder>_audit_v3.json, <data folder>_manifest.json
                                   tracked copies (data/ is git-ignored; the gate evidence is not)
```
Shards are written atomically (`.tmp` + rename); generation is resume-aware and order-free.

---

## 4. Shard schema (`record_schema_version = 1`, `feature_schema_version = 6`)

**Frames** — every frame of the episode (stride 1).

| key | dtype | shape | meaning |
|---|---|---|---|
| `f_node_feat` | f32 | (n_frames·N, 9) | `NODE_FEATURES`, rows = node id (ids 0..N−1; node-id invariant asserted every frame) |
| `f_n_nodes` | i16 | (n_frames,) | N |
| `f_edge_index` / `f_edge_feat` | i16 / f32 | (2, ΣE) / (ΣE, 4) | undirected edges once; `EDGE_FEATURES` |
| `f_n_edges` | i32 | (n_frames,) | per-frame edge count |
| `f_t` | f32 | (n_frames,) | frame start time (s) |
| `f_cum` | i64 | (n_frames+1, 9) | cumulative generated / delivered / dropped + 6 drop reasons at each frame start, last row = episode end |
| `f_mean_occ`, `f_max_occ`, `f_mean_activity`, `f_inflight`, `f_mean_lq` | f32/i32 | (n_frames,) | simulator per-frame diagnostics |

**Contexts** — every distinct decision context, once.

| key | dtype | meaning |
|---|---|---|
| `c_frame`, `c_current`, `c_dst`, `c_hops` | i32/i16/i16/i8 | the context key (with the candidate list) |
| `c_k`, `c_cands` | i16 | candidate count; candidates (canonical order, visited excluded), flat |
| `c_cand_feat` | f32 (Σk, 4) | `CANDIDATE_FEATURES` |
| `c_query` | f32 (n, 6) | `QUERY_FEATURES` (v6). Its `own_queue_live` is the FIRST occurrence's (= `c_ownq_first`/50, checked) and is biased high (#14) — training rows take it from the histogram below (§8) |
| `c_label` | i16 | index of da_gpsr's choice |
| `c_scores` | f32 (Σk) | da_gpsr's score of every candidate (soft labels, residual policies, near-tie analysis) |
| `c_votes` | i16 (n, 3) | dijkstra / gpsr / spbp choice; dijkstra −1 = "would drop here" |
| `c_beh` | i16 | the episode's behaviour policy's choice (== label on da_gpsr episodes) |
| `c_ownq_first` | i8 | live own-queue length of the first occurrence |
| `c_mult` | i32 | how many decisions this context stands for (Σ = recorded decisions) |

**Own-queue histograms** — the one feature that differs between a context's occurrences.

| key | dtype | meaning |
|---|---|---|
| `o_ctx` | i32 | context id; pairs sorted by (context, length), each pair once |
| `o_ownq` | i8 | a live own-queue length (0–50) seen by occurrences of that context |
| `o_count` | i32 | how many of the context's decisions saw it (Σ per context = `c_mult`) |

Contexts + histograms reproduce the raw decision stream exactly: decision features = the
context's features with `own_queue_live` = length/50, label = the context's label.

**Steps** — every decision of a sampled packet (`h64(seed, pid) < 0.10`), contiguous chains.

| key | meaning |
|---|---|
| `s_ctx` | context id → features, label, behaviour choice |
| `s_pid`, `s_hop` | packet, hop index (chain order) |
| `s_action` | candidate index taken; −1 reserved (drop), never generated |
| `s_eps` | the ε draw fired (`deviated` = `action != label` is derived, not stored) |
| `s_ownq` | live own-queue length at this decision (exact, per step; always one of its context's histogram values). **An RL state built from a step takes `own_queue_live = s_ownq/50`, not the context's stored value.** |
| `s_t` | slot time (s) |
| `s_outcome` | moved / delivered / link_error / queue_overflow / energy_depleted / policy_drop |
| `s_attempts` | ARQ attempts used by this hop (0 if it never transmitted) |

**Packets** — the sampled packets: `p_pid, p_flow, p_src, p_dst, p_gen_t, p_fate`
(delivered / queue_overflow / link_error / ttl_expired / no_route / energy_depleted /
episode_end), `p_deliv_t` (NaN unless delivered), `p_hops`, `p_delay_ms`.

---

## 5. Generation

`DatasetSimulatorV3` overrides only `_select_next_hop`, `_try_forward`, `_finish_packet` and
`_build_graph`; ARQ, queues, energy and the drop taxonomy are the validated simulator, with
**strict** queue-slot conservation (a phantom slot raises).

Per decision: canonical candidates (per-frame cache, pinned equal to `canonical_candidates`)
→ empty ⇒ base class drops (`no_route`), counted → context key lookup → new context: label +
scores, three votes, behaviour choice, features (one extraction per *context*, not per
decision) → multiplicity += 1 and the live own-queue length into the context's histogram
(5 bytes per decision until the episode ends) → ε draw → action → if the packet is sampled, a step row whose
hop outcome and ARQ attempts are filled when `_try_forward` returns.

**Determinism contract.** The only random stream consumed is `ds_rng` (seed + 900 000): one
draw per decision, one more when ε fires — the same as V2. Packet sampling and behaviour
assignment are pure functions of (seed, pid) / seed (splitmix64), so recording more or fewer
packets never changes a trajectory and a smaller packet sample is a strict subset of a larger
one. Checked: scalar and vectorised hashes bit-identical; G3.5 v3 check 7 regenerates an
episode and compares every array's SHA-256.

---

## 6. Behaviour assignment

Within each split, seeds are ordered by `h64(seed, 0, salt=2)`; the first round(0.3·n) run an
"other" teacher, cycled through (gpsr, spbp, dijkstra_nodrop) with a per-split offset. The same
seed → policy map applies to every cell (paired across cells).

| split | seeds | other-teacher seeds |
|---|---|---|
| train 101–135 | 35 | 11 — gpsr 101 108 112 115 · spbp 114 116 117 120 · dijkstra_nodrop 119 125 128 |
| val 136–142 | 7 | 2 — dijkstra_nodrop 138 · gpsr 141 |
| test 143–150 | 8 | 2 — dijkstra_nodrop 147 · spbp 150 |

Overall 35 da_gpsr / 5 / 5 / 5. Imitation accuracy should be reported per behaviour policy.

---

## 7. Gates

| gate | script | what it proves |
|---|---|---|
| grid | `verify_dataset_grid_v3.py` | every cell is in (band) or below (anchor) a band measured at the operating point |
| G3.5 v3 | `preflight_dataset_v3_check.py` | 0 schema (v28: and the code signature — the dataset must come from the current simulator / feature / generator code, every shard from the same code) · 1 structure · 2 labels carry the max score · 3 trivial/hard shares (diagnostic) · 4 behaviour: seed→policy, non-ε action == behaviour choice, no −1, ε per episode and pooled · 5 coverage · 6 feature ranges / dead columns · **7 reproducibility (byte-identical regeneration)** · 8 redundancy · 9 dedup (unique keys, Σ multiplicity = decisions; every own-queue histogram sums to its multiplicity and holds the first-occurrence value that `c_query` stores; every step's own queue is in its context's histogram) · 10 chains (contiguous from the source, s→a→s′ linkage, final outcome = fate, hop count) · 11 sampled set == hash rule exactly · 12 frame counters == metrics |
| audit v3 | `audit_dataset_v3.py` | independent re-derivation from stored frames (own code, own splitmix64): A candidates are neighbours in canonical order · B label & scores · C every feature · D votes and the behaviour choice · E splits · F hashing · G own-queue histograms recounted in plain Python |

**Every per-shard check and every audit re-derivation is first run on a deliberately corrupted
copy of a real shard and must flag it**, or the gate aborts. Smoke result (sandbox, 10 episodes,
final code): G3.5 v3 PASS with 10/10 corruptions detected and all 47 arrays reproduced
byte-identically; audit v3 PASS with 15 000 contexts re-derived, label agreement 1.00000,
features 1.00000, 0 own-queue histogram mismatches.

---

## 8. Training and the restated G4

* `export_phaseb_v3.py` writes the PhaseB format `train_supervised_v2` already reads.
  `--ctx_frac` selects a **nested** fraction of contexts (hash of episode × context index), so
  learning curves need no extra generation. `--weighting multiplicity` (default) writes a
  `weight` column; `PhaseB(..., use_weights=manifest['use_weights'])` turns it into a weighted
  loss (the unweighted path is byte-identical to before).
* `--ownq` decides the one feature that differs between a context's occurrences:
  **`sample`** (default) — one row per context, own queue drawn from the context's histogram
  with probability count/multiplicity (deterministic hash, salt 4), so the weighted rows have
  the raw stream's own-queue distribution in expectation; **`expand`** — one row per
  (context, own queue) pair weighted by its count: the raw decision stream exactly, at 3–7×
  the rows (0.53–0.81 of all decisions in the smoke). Smoke check: decision-weighted mean own
  queue 14.00 (`sample`) / 13.97 (`expand`) vs 13.97 in the raw stream (first occurrence: 22.74). The biased first-occurrence value is never exported;
  rows that happen to equal it are asserted byte-identical to the stored `c_query`. The trainer
  batches by frames (48 per batch), so `expand` also multiplies rows per batch (and GPU memory)
  3–7×; lower `frames_per_batch` if it runs out of memory.
* **G4 check 4 (v3), pre-registered here** — `rollout_gate_v3.py`: reference = restricted
  da_gpsr (ε 0) at the operating point; student = masked GNN (mask `hop`, tuned M4 config)
  trained on the export; paired over evaluation seeds 1–5 (outside the dataset seeds), mean over
  model seeds. **PASS iff in every training cell the 95% CI lower bound of (student − reference)
  is above −1.0 pp.** Held-out medium_slow cells are reported, not gated. A cell whose CI
  straddles −1 pp with a half-width above 1 pp is **INCONCLUSIVE** (too few seeds, not a
  failure): re-run with eval seeds 1–10 and decide on those; a cell whose whole CI is below
  −1 pp, or whose narrower CI straddles it, **FAILS**. Power: the teacher-choice test measured a
  per-seed sd of (student − da_gpsr) of 0.09–0.42 pp at 300 s, so 5 seeds at 1000 s should give
  half-widths well under 1 pp. All four outcomes (and INCOMPLETE) were exercised on synthetic
  results.
* **D2 reference** must be re-measured at 100–300 m on the D2 evaluation seeds — every oracle
  panel ran at 50–150 m.

---

## 9. Size and cost

**Actual v26 run** (2026-10-04, Z8, 12 workers, with the trapping bug): 10.0 h for 800 episodes; 120.1 M contexts, 471 M own-queue pairs, 71.5 M steps, 714 M decisions, 11.5 GB. Export at `--ctx_frac 0.6` of train + val: 40.1 M rows, 997 k frames, 20.2 GB in 155 s. The v28 run will differ (the networks keep moving).

**v29 grid (18 cells, 900 episodes):** sparse_fast@60 (~2–3 min per episode) and sink_50@40 (~15 min) add about 1.2 h of generation and ~12% more contexts. The 16-cell export at `--ctx_frac 0.6` was 20.2 GB, so the 18-cell one will likely exceed `--max_gb 22`: the exporter refuses before writing anything and prints the size — then use `--ctx_frac` ≈ 0.6 × 21.5 / that size. Point-1 measured delivery flat from 13 k to 1.7 M rows (its pre-registered reading, DATA-LIMITED, rests on hard-row accuracy — §12 item 8), so the fraction is set by RAM.

Measured in the sandbox (60 s smoke episodes, 150 s probes, one 1000 s run per scenario) and
extrapolated to 1000 s:

| | per 1000 s episode | 16 cells × 50 seeds (800 episodes) |
|---|---|---|
| decisions | 0.7–1.2 M | ~0.7 billion (not stored) |
| contexts | 60–270 k | ~100–150 M |
| (context, own queue) pairs | 0.53–0.81 × decisions, 5 B each | ~0.5 billion pairs, ~2–3 GB of the shards |
| steps (10% packets) | 70–120 k | ~75 M |
| shard on disk (compressed) | ~10–40 MB | **~15–30 GB** |
| simulation time | ~5–40 min per episode (sparse_fast fastest; very_dense@80 slowest — 32 min for the bare 1000 s simulation) | **≈ 200 CPU-hours → ~17 h at 12 workers, ~25 h at 8** |

Per-core speed: the teacher-choice run on the Z8 (12 workers) took ~300 s per 300 s very_dense@60
episode, the same as a sandbox core, so these sandbox timings transfer. Timing-probe the first
real episodes before trusting them (the last linear extrapolation in this project was 13×
wrong). Workers default to 8 in every v3 script, because 16 spawn workers stalled the
convergecast band script on this machine (project report); `test_teacher_choice.py` ran 310
episodes with 12 workers without trouble, so the long runs below use 12.

**RAM — how much of the lossless dataset the trainer can hold** (measured on smoke exports):
an export row costs ~370 B at ~11 candidates (+ ~5.6 KB per frame), and `PhaseB` holds about the
same in RAM (~450 B/row). Train + val (504 episodes, 60–70 M contexts) would be ~25–30 GB — it
does not fit next to the trainer on 32 GB. `export_phaseb_v3.py --max_gb 22` (~70% of RAM, the
default) holds ~45–50 M rows, i.e. `--ctx_frac` ≈ 0.6–0.7 of train + val with `--ownq sample`
(`expand` is 3–7× bigger). The exporter sizes every array first and fills them in place, so the
export itself peaks at ~1× its size, not 2×. GPU time scales with rows: GLOBAL (449 k rows)
took ~8.5 s per epoch on the A4000, so ~45 M rows ≈ 14 min per epoch, ~5–7 h per model seed.
`test_bc_data_scaling.py`'s verdict decides whether that is worth it (V3-12).

---

## 10. Execution order

**v28/v29 re-run** (after `apply_mobility_fix_v28.py` / `verify_mobility_fix_v28.py`, then `apply_grid_followup_v29.py` / `verify_grid_followup_v29.py`):

```
1  DONE 2026-10-06: bands, phase 1 only, into results\band_*_v28.json (--skip-self-test: the
     self-tests check the code, not the band; verify_mobility_fix_v28 check 5 runs them)
2  DONE: verify_dataset_grid_v3.py -> 16/16 PASS, BAND MOVED sparse_fast (+60), sink_50 (+40)
     -> v29: DATASET_GRID follows (18 cells). Re-run on the 18-cell grid (18/18 PASS):
     python src\verify_dataset_grid_v3.py     (seconds; add --diagnose_seeds 3 --max_workers 12
     for da_gpsr's congestion per cell, ~1 h -- reported, never used to move cells)
3  python src\generate_dataset_v3.py --smoke ; G3.5 v3 + audit on data\v3_smoke
4  python src\generate_dataset_v3.py --out data\v3m --max_workers 12   (a NEW folder; ~11 h)
5  python src\preflight_dataset_v3_check.py --data data\v3m ; python src\audit_dataset_v3.py --data data\v3m
6  python src\export_phaseb_v3.py --data data\v3m --out data\phaseB_v3m --splits train val --ctx_frac 0.6
     (refused above --max_gb 22, with the size printed: then --ctx_frac = 0.6 x 21.5 / size)
7  python src\rollout_gate_v3.py --data data\phaseB_v3m --out results\g4_v3m --max_workers 12
     (v29: the held-out medium_slow cells are rolled out and reported by default)
```
Teacher choice on the fixed simulator: DONE 2026-10-05 (`results\teacher_choice_v28`) — SINGLE da_gpsr, unchanged (§0 #1). Point-1: DONE 2026-10-06 (`results\bc_data_scaling_v28`) — DATA-LIMITED again, on the same weak trigger (§12 item 8).

**Original v26 order** (run 2026-10-03/04 on the trapping simulator; kept for the record):

```
0  python src\test_bc_data_scaling.py --stage all --extend --rollout --max_workers 12
   (Point 1: how much imitation data is worth training on; uses results\teacher_choice only,
    reads that run's own settings from its config.json, works before or after step 1
    because it requests schema v5 explicitly)
1  python apply_dataset_v3_v26.py --src src --dry-run ; then without --dry-run ;
   python verify_dataset_v3_v26.py --src src ; python apply_docs_v3_v27.py --root .
2  sink_50 band at 100-300 m (one line):
     python src\find_congestion_band_convergecast.py --scenario sink_50 --duration 1000 --initial_energy 8000 --rates 20 30 40 50 60 --measure-seeds 15 --max_workers 8 --out results\band_sink50_1000s_z100_300.json
3  python src\verify_dataset_grid_v3.py --diagnose_seeds 3      (all 16 cells PASS)
4  plumbing (writes data\v3_smoke):
     python src\generate_dataset_v3.py --smoke
     python src\preflight_dataset_v3_check.py --data data\v3_smoke
     python src\audit_dataset_v3.py --data data\v3_smoke
5  timing probe (its shard is kept by the full run): python src\generate_dataset_v3.py --cells very_dense@80 --seeds 101
6  full run:     python src\generate_dataset_v3.py --out data\v3 --max_workers 12
7  python src\preflight_dataset_v3_check.py --data data\v3 ; python src\audit_dataset_v3.py --data data\v3
8  python src\export_phaseb_v3.py --data data\v3 --out data\phaseB_v3 --splits train val --ctx_frac <from step 0>
   python src\rollout_gate_v3.py --data data\phaseB_v3 --out results\g4_v3     (retrain + restated G4)
```
The dataset depends on the simulator and `OPERATING_POINT`, not on the RL environment, so it
can be generated while `run_iter()` / `FANETEnvV2` / the parity gate are built (v8b still waits
for that gate). Risk accepted: a simulator bug found later would mean regenerating.

---

## 11. Statements this document overturns

| statement | where | now |
|---|---|---|
| one global rate grid `[0.02 … 0.40]` | DATASET_V2_MASTER_SPEC §4.2, R-16 | void (v12); per-scenario band + anchor (§2) |
| stride 10 / `--frame_stride 10` | FILE2 §2 | stride 1 for frames; packet-sampled steps; deduplicated contexts |
| SP-BP oracle, `ORACLE_TEACHER='spbp'`, "wins all 12 cells" | generate_dataset_v2, FILE2, V2 spec | da_gpsr (V3-1); V2 generator is legacy (`--legacy_v5`) |
| per-cell oracle + `label_teacher` column | M3.5_Dataset_Schema §5.3 | single da_gpsr label |
| "feature schema carries forward unchanged" | M3.5_Dataset_Schema §5, §9 | schema v6 |
| `behaviour_deviated == (action != label)` check | V2 spec R-23, schema §5.8 | `eps_fired` + behaviour-choice check (§7) |
| "under the 70/30 mix ~30% of transitions deviate" | schema §4.2 | only where the other teacher disagrees (gpsr agrees with da_gpsr 75–97%) |
| `checkpoint_id` per decision; 50 s checkpoints in the dataset | schema §5.6, D-19 | dropped (V3-11) |
| "1000 s unblocks `estimated_link_lifetime` saturation" | FILE1, report §11.1, v8b docstring | withdrawn (§0 #10) |
| very_dense sign flip at rate 100 inside the band; 5-rate grids | LOAD_DENSITY_DESIGN | rate 100 has dead nodes at 1000 s; bands give 2–3 rates |
| panels ran at 100–300 m | Panel_Results_Report §2.1, report §12 | they ran at 50–150 m (BASE); bands (except sink_50) at 100–300 m |
| dequeue-before-decision explains the zero own queue | audit_dataset_v2 | frame-start stamping (corrected in v26) |
| G4 check 4 = ≥ 90% of SP-BP | rollout_eval_v2 | non-inferiority vs restricted da_gpsr (§8) |
| drones follow Random Waypoint for the whole episode | `mobility.py` since the first commit | they froze at a waypoint after a missed arrival (§0 #15); fixed in v28 |
| sparse_fast is connectivity-limited; its band cells are not peers of the other scenarios' (`results/README.md` caveat) | `results/README.md` | the trapping bug: on the fixed simulator it is congestion-limited (elasticity 0.45–0.55, queue overflow 0.17–0.26; v29) |
| the 1000 s bands; "at 1000 s only rate 30 is usable in sink_50"; the v26 dataset and its per-cell PDRs | report §11.2, `results/band_*_1000s.json`, `data/v3` | measured on freezing networks; re-measured / regenerated after v28 |
| link-lifetime saturation is purely structural (§0 #10) | this spec | partly confounded: a trapped drone reports a velocity that flips every frame; re-check on the v28 dataset |

## 12. Open items

1. sink_50 band at 100–300 m: done (v28 mobility: 30 and 40 usable, v29). The Suite C training weight (V3-10) remains open.
2. Training size from `test_bc_data_scaling.py` (V3-12).
3. Drop/hold in the M5 action space (environment contract) — the dataset already encodes it.
4. D2 reference re-measured at 100–300 m.
5. `BUFFERED_REF_V6`: checked on the v26 run (largest value 73% of the range, nothing clipped); re-confirm on the v28 run.
6. The isolated-node queue quirk (§0 #13): fix or keep, before the M5 environment is frozen.
7. **v28 re-run** (§10): bands and grid done (v29: 18 cells), teacher choice and Point-1 re-run → regeneration → gates → export → G4 (held-out cells reported by default).
8. Point-1 (`test_bc_data_scaling.py`, 2026-10-04): the pre-registered verdict is DATA-LIMITED, recorded as such. Its only trigger (very_dense hard rows +0.77 pp, 30% → 100%) is within model-seed noise (two seeds differ by up to 1.0 pp), delivery moved +0.04 pp, and the 4× step reversed it in every cell; delivery is flat from 13.7 k to 1.8 M rows. Report both (V3-12). **Re-run on the fixed simulator (2026-10-06, `results/bc_data_scaling_v28`): DATA-LIMITED again, on the same single trigger** — very_dense hard rows +1.01 pp (30% → 100%), smaller than the two model seeds' spread at that point (1.92 pp); delivery −0.04 pp (se 0.03); the 4× step lowered hard-row accuracy in all three cells (−0.28 / −0.08 / −0.53 pp); delivery vs da_gpsr flat from 13 k to 1.74 M rows (dense −0.02…+0.06, very_dense −0.08…−0.01 pp). New on the moving networks: small-data students are worse on hard rows (very_dense at 3%: 92.1% vs 95.0% before the fix), converging by 100%. The medium_slow student-vs-teacher offset (+0.5–0.6 pp in every run) is not significant (p 0.12–0.18, n = 5).
