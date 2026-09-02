"""apply_doc_v2_corrections.py -- FILE1/FILE2 v2: everything since the original upload.

Assertion-guarded str.replace, same convention as every code patch in this project.
Anchors must match EXACTLY ONCE per file; edits are staged in memory; nothing is
written unless every anchor in every file matches. Idempotent (checks a v2 guard
string before doing anything).

SUPERSEDES apply_doc_corrections_v10.py -- that patch was never applied to any file
that exists on disk (both originals were confirmed byte-identical to the initial
upload before this patch was written). This one folds its corrections in directly
plus everything found afterward, in one pass.

WHAT CHANGES, AND WHY

FILE 1 (5 edits):
  1. Header -- HEAD ebc013d -> 41ea7a4 (pushed, current), v2 marker.
  2. SS1 "why Approach 2 exists" -- cites docs/PHASE5_REPORT.md properly (the
     warmstart result is real and positive; the ablation-null result is the part
     explained by no congestion); notes the Approach-1 artifact quarantine and why
     (scope mismatch, not a defect).
  3. SS6.2 headroom row + a correction note -- 1.6/18.1/35.6 -> 1.4/17.5/35.1,
     attributed to the SAME id(G) defect class as the M4 check-6 bug (error 9),
     found in a different file. Conclusion survives: the two ~100%-reachable
     scenarios moved exactly zero.
  4. SS9 patch history -- v9's status enriched (staged, deliberately unapplied,
     not just "NOT applied"); v10 and v11 rows added.
  5. SS12 headroom mention -- same correction, plus a note that the figures are
     still at the OLD operating point and about to change.

FILE 2 (10 edits):
  1. Header -- HEAD, v2 marker.
  2. SS0 "WHERE YOU ARE" -- v10/v11 applied; v8/v9 status enriched with why they're
     deliberately unapplied; the current-state table's RATES row now points at the
     probe instead of presenting the five-rate proposal as the target.
  3. SS1.5 -- flagged superseded at the top (reasoning kept, conclusion isn't).
  4. SS1.5.1 (NEW) -- the actual three-quantity probe replacing the old proposal,
     including the experiment_headroom.py CLI-flag prerequisite (v11.2, not yet
     written) and the separate decision rule needed for policy spread.
  5. SS2.1 -- load-bucket note now points at SS1.5.1, not the superseded grid.
  6. SS3 -- sequencing correction: v8/v9 no longer "Step 2", moved after the new
     environment's parity gate, with the reasoning stated inline.
  7. SS7 -- baselines no longer parallel with regeneration; Gate G-A's detailed
     acceptance criteria folded in from docs/PRE_M5_POSITIONING_PLAN.md Part 2,
     including the PPORouter go/no-go, the missing itemised cost estimate, and the
     "12-cell" wording's soft dependency on the probe's output shape.
  8. SS10.2 -- was a bare header naming the question with no body. Filled in:
     what's resolved (observation structure, from ModelActorSimulator's proven
     pattern, plus its replay-buffer consequence) and what's genuinely open
     (continuing-task window, credit assignment).
  9. SS10.5 (NEW) -- the missing .pt checkpoint gap, and the checkpoint-load
     validation step this motivates (step 6b in the new ordering).
  10. SS13 -- full ordering rewrite reflecting all of the above; original ordering
      kept as SS13.1 for reference, with the two structural reasons it was wrong
      stated explicitly.
  11. SS14 -- same headroom correction as FILE 1, plus the new-operating-point
      caveat.

USAGE
    python apply_doc_v2_corrections.py --dir . --dry-run
    python apply_doc_v2_corrections.py --dir .
"""
import argparse, io, os, sys

F1 = 'FILE1_APPROACH2_RECORD.md'
F2 = 'FILE2_PRE_M5_EXECUTION_PLAN.md'
GUARD = 'Revised:** v2'

# ============================================================================
# FILE 1
# ============================================================================
EDITS_F1 = [
    ("""**Project:** FANET Lifelong-RL Routing · Shirish Giroti (CS23B2041), IIITDM Kancheepuram
**Guide:** Dr. Preeth Raguraman · **Repo:** `github.com/dovahkkin1234/fanet-coldstart-rl`
**HEAD at handoff:** `ebc013d` · **Written:** 2026-08-21""",
     """**Project:** FANET Lifelong-RL Routing · Shirish Giroti (CS23B2041), IIITDM Kancheepuram
**Guide:** Dr. Preeth Raguraman · **Repo:** `github.com/dovahkkin1234/fanet-coldstart-rl`
**HEAD:** `41ea7a4` (pushed, current) · **Written:** 2026-08-21 · **Revised:** v2, post-M4 pre-M5 investigation"""),

    ("""**Why Approach 2 exists:** Approach 1 (single-packet DQN) proved warmstart eliminates
cold-start risk, but its Dueling / reward-shaping / curriculum ablation returned uniformly
null — because single-packet routing means queue occupancy is identically zero, so no
congestion can form. Approach 2 is multi-packet and congestion-coupled specifically to fix
that.""",
     """**Why Approach 2 exists:** Approach 1 (single-packet DQN, `docs/PHASE5_REPORT.md`) is a
complete, gate-passed 30-seed study. Its warmstart result is genuinely positive and stands
on its own terms — up to 6× T90 speedup, a 5.1× fast/slow init-PDR ratio, near-zero
converged-PDR variance (0.010) across all 30 seeds — because single-packet episodes give
correct per-transition credit assignment by construction, a deliberate design choice, not
an oversight. But its Dueling / reward-shaping / curriculum ablation returned uniformly
null, because that same single-packet design makes queue occupancy identically zero, so no
congestion can ever form. Approach 2 is multi-packet and congestion-coupled specifically to
test what Approach 1 structurally could not.

**Note on the Approach-1 artifacts** (`results/checkpoints/*.pt`, `rl_env_v4.py` and its
siblings): the underlying study is cited above and stands. The code and checkpoints
themselves are quarantined from reuse in Approach 2 — wrong feature space (a flat 22-column
vector, not `features_v2.py`'s graph-structured schema) and, by design, no congestion. This
is a scope mismatch, not a defect, recorded once here so it is not re-investigated."""),

    ("""| **Headroom** | **1.6% at low load, 35.6% at high load** |

**Two limitations the gates report about themselves — put both in the methodology section:**""",
     """| **Headroom** | **1.4% at low load, 17.5% medium, 35.1% at high load** (overall 18.0%) |

**Headroom correction [MEASURED, v10, 7/7 verified]:** the original 1.6/18.1/35.6 figures
came from `experiment_headroom.py`, which had the same `id(G)`-keyed cache defect as the M4
check-6 bug (§7.5, error 9 below) — in a different file, found independently during pre-M5
config-duplication cleanup. Fixed via the identical frame-counter pattern. The correction is
small and the conclusion survives: `dense_slow` and `very_dense` (both ~100% reachable) moved
**exactly zero**; the entire delta is `no_route` in the two partitioned cells. Current
figures are in `results/headroom_v10.json`. The pooled 18.45% → 18.0% shift is consistent
with a correction, not a reversal.

**Two limitations the gates report about themselves — put both in the methodology section:**"""),

    ("""| **v9** | Pins the Suite C sink stationary at the area centre at `z_min`, with a construction-time assertion | **NOT applied** |""",
     """| **v9** | Pins the Suite C sink stationary at the area centre at `z_min`, with a construction-time assertion | **staged, NOT applied — deliberately, pending the new RL environment's SP-BP-parity gate; see FILE 2 §3** |
| **v10** | `HeadroomSimulator._reach_cache` in `experiment_headroom.py` — same `id(G)` defect class as v9 patched in a different file (`rollout_eval_v2.py`), found independently. Frame-counter key; `--cache_mode {fixed,legacy}` for attribution; verified 7/7 including an allocator-independent negative control | **applied** |
| **v11** | `src/config_v2.py` — single shared `SCENARIOS`/`RATES`/`BASE` source. Migrates 8 scripts off duplicated local constants (the same defect class that let v8 miss `experiment_headroom.py` entirely); verified with a positive control (mutate `config_v2.BASE`, confirm every module observes it) | **applied** |"""),

    ("""The escape is load: **headroom is 1.6% at low load and 35.6% at high load.** The thesis
lives at high load, and **that must be confirmed at M6 before M8–M11 consume months.**""",
     """The escape is load: **headroom is 1.4% at low load and 35.1% at high load** (§6.2). The
thesis lives at high load, and **that must be confirmed at M6, at the new operating point
and the new rate grid, before M8–M11 consume months** — the current 1.4/17.5/35.1 figures
are still at the OLD 40 s / altitude-50–150 operating point and the old rate grid; both are
about to change (FILE 2 §1.5.1, §3)."""),
]

# ============================================================================
# FILE 2
# ============================================================================
EDITS_F2 = [
    ("""**Project:** FANET Lifelong-RL Routing · **Repo:** `github.com/dovahkkin1234/fanet-coldstart-rl`
**Precondition HEAD:** `ebc013d` · **Written:** 2026-08-21
**Companion:** FILE 1 (the record — what was built and measured). Read it first.""",
     """**Project:** FANET Lifelong-RL Routing · **Repo:** `github.com/dovahkkin1234/fanet-coldstart-rl`
**HEAD:** `41ea7a4` (pushed, current) · **Written:** 2026-08-21 · **Revised:** v2
**Companion:** FILE 1 (the record — what was built and measured). Read it first."""),

    ("""**Applied:** patches v1–v7. M4 complete, all six G4 checks resolved.
**Built but NOT applied:** `apply_sim_changes_v8.py`, `apply_sink_fix_v9.py`.
**Not written yet:** the rates patch and the frame-stride patch (§2).

**Current state of the things you are about to change:**

| | current value | target |
|---|---|---|
| `RATES` | `[0.5, 2.0, 4.0]` | `[0.05, 0.15, 0.4, 0.7, 1.0]` |
| `BASE` duration | 40.0 s | 1000.0 s (v8) |
| `BASE` altitude | `z_min=50, z_max=150` | `z_min=100, z_max=300` (v8) |
| frame striding | **does not exist** | every 10th frame recorded |
| `SEEDS` | `range(101,151)` — 50 seeds | **unchanged, keep all 50** |""",
     """**Applied:** patches v1–v7 (M4 complete, all six G4 checks resolved); **v10** (the
`experiment_headroom.py` cache fix — a second instance of the same `id(G)` defect v9
patched elsewhere); **v11** (`src/config_v2.py`, the single shared config source, 8 files
migrated).
**Staged but deliberately NOT applied:** `apply_sim_changes_v8.py`, `apply_sink_fix_v9.py` —
held until the new RL environment's SP-BP-parity gate passes at the CURRENT 40 s / altitude
50–150 operating point (§3). Applying v8 first would remove the only fixed reference for
telling "the environment is right" apart from "the operating point changed underneath it."
**Not written yet:** the rates patch and the frame-stride patch (§2) — both now BLOCKED on
the three-quantity rate probe (§1.5.1) rather than ready to write directly.

**Current state of the things you are about to change:**

| | current value | target |
|---|---|---|
| `RATES` | `[0.5, 2.0, 4.0]` | **NOT FINALISED — pending §1.5.1's probe**, superseding the `(0.05, 0.15, 0.4, 0.7, 1.0)` proposal in §1.5 |
| `BASE` duration | 40.0 s | 1000.0 s (v8, deferred — see above) |
| `BASE` altitude | `z_min=50, z_max=150` | `z_min=100, z_max=300` (v8, deferred) |
| frame striding | **does not exist** | every 10th frame recorded |
| `SEEDS` | `range(101,151)` — 50 seeds | **unchanged, keep all 50** |
| config location | scattered, 8 copies (pre-v11) | **single source, `src/config_v2.py`** |"""),

    ("""## 1.5 Load sweep → `(0.05, 0.15, 0.4, 0.7, 1.0)`

The old `(0.5, 2.0, 4.0)` was calibrated for 40 s episodes. **[MEASURED] at 1000 s,""",
     """## 1.5 Load sweep → `(0.05, 0.15, 0.4, 0.7, 1.0)` — SUPERSEDED, see §1.5.1

> **This section's proposed grid is FLAGGED AS LIKELY WRONG and superseded by §1.5.1.**
> The reasoning below (the old grid saturates completely at 1000 s) is correct and worth
> keeping — what changed is HOW to pick the replacement. This five-rate proposal was
> designed before recognising that headroom is expected to be an inverted U in load, and
> its top three rates (0.4, 0.7, 1.0) all sit past the measured 0.25-rate point in
> `dense_slow` — the second-easiest scenario — plausibly landing most cells in or near
> collapse. Kept here for the historical reasoning; **do not implement this exact grid.**

The old `(0.5, 2.0, 4.0)` was calibrated for 40 s episodes. **[MEASURED] at 1000 s,"""),

    ("""Five rates instead of three gives a proper decline curve rather than three points.

## 1.6 Seeds: 101–150, continuous, all 50""",
     """Five rates instead of three gives a proper decline curve rather than three points.
**Superseded — see §1.5.1.**

## 1.5.1 The actual replacement — three-quantity probe [NOT YET RUN]

Measure three separate quantities, not one, across 4 scenarios × ~6 candidate rates
(0.02, 0.05, 0.10, 0.15, 0.25, 0.40) × ~5 seeds × 3 actors (`spbp`, `gpsr`, `dijkstra`,
plus `random` free on the same runs):

1. **Policy spread** — `spbp` PDR minus `gpsr`/`dijkstra` PDR (both congestion-blind).
   This is the inverted-U curve. **Picks the grid** — the five rates bracketing its peak,
   possibly per scenario class, since `sparse_fast` is expected to pass its knee earlier
   than `very_dense`. Record `spbp − random` too (free, same runs) — it bounds "does
   routing matter here at all", a separate question from "does congestion-awareness
   matter."
2. **Carried load** — offered vs delivered pkt/s. Locates the collapse knee; bounds the
   top of the grid.
3. **Existing addressable-share headroom metric** (`experiment_headroom.py`, unchanged
   definition) — continuity only. M6 needs it at the new operating point. **Do NOT use it
   to pick rates** — `ttl_expired` is exactly 0.000 in all measured cells, so the metric is
   essentially pure `queue_overflow` share, which climbs monotonically toward 1.0 past
   capacity by construction. It would report the most collapsed cell as having the most
   headroom.

**Fix a separate decision rule for policy spread before running the probe** — the existing
>20%/5–20%/<5% rule in §9 belongs to addressable-share, not spread, and is the wrong scale
(a 5% PDR spread between best and worst teacher is a large effect on this metric).

**Can run ahead of v8**, since `experiment_headroom.py` now reads `config_v2.BASE` (v11)
rather than a hardcoded local copy — point it at 1000 s / 100–300 m via CLI flags without
touching the shared module (**v11.2, not yet written**: `--duration`, `--z_min`, `--z_max`,
`--drain_time`, defaulting to `BASE[...]`, plus wiring `config_v2.provenance()` into the
output JSON so the operating point is recorded, not left to git archaeology again).

At 12-wide parallel this is roughly **55 minutes**, not the ~31 hours full regeneration
costs — run it first.

## 1.6 Seeds: 101–150, continuous, all 50"""),

    ("""**Load-bucket mapping must be checked.** `teacher_panel.load_bucket()` maps rate → low /
medium / high. With five rates it needs an explicit mapping — suggest
`0.05, 0.15 → low`; `0.4 → medium`; `0.7, 1.0 → high`, but **[TO VERIFY]** read the
function and confirm it does not hard-code the old three values.""",
     """**Load-bucket mapping must be checked — and now depends on §1.5.1, not this section.**
`teacher_panel.load_bucket()` maps rate → low / medium / high. The specific rate values,
and possibly the number of buckets per scenario class, come out of the probe, not the
superseded five-rate proposal in §1.5. **[TO VERIFY]** read the function and confirm it
does not hard-code the old three values regardless of which grid is chosen."""),

    ("""# 3. STEP 2 — APPLY v8 AND v9 (10 min)

```powershell
python apply_sim_changes_v8.py --src src --dry-run""",
     """# 3. APPLY v8 AND v9 — MOVED, no longer "Step 2" (10 min)

> **Sequencing correction, supersedes the original early-apply ordering (§13.1).** Do NOT
> run this until the new RL environment's SP-BP-parity gate (§10, built on
> `FANETSimulatorV2`) passes **12/12 exact at the CURRENT 40 s / altitude-50–150 operating
> point**. The teachers were measured at that exact point; it is the only fixed reference
> for telling "the environment is built correctly" apart from "the operating point changed
> underneath it." Moving both at once removes the ability to attribute a discrepancy to
> either cause. See §13 for the corrected position in the overall order.

```powershell
python apply_sim_changes_v8.py --src src --dry-run"""),

    ("""# 7. STEPS 6–8 — THE THREE BASELINES (~6 d, IN PARALLEL WITH STEP 3)

**These do not depend on the dataset.** They need only the configs to exist. Build them
while the 31-hour regeneration runs — the compute costs nothing if you are coding during it.""",
     """# 7. THE THREE BASELINES (~6 d) — REPOSITIONED, see §13

> **Sequencing correction.** No longer parallel with regeneration — regeneration itself now
> runs later (§3). Gate G-A needs a correct, parity-verified environment to test baselines
> in, so this whole section runs AFTER the new environment's SP-BP-parity gate passes,
> positioned alongside or just before warmstart transfer, since `QMRRouter` / `DQNRouter` /
> warmstart are the co-built competing D1–D5 arms.
>
> **Gate G-A's detailed acceptance criteria** are in `docs/PRE_M5_POSITIONING_PLAN.md`
> Part 2 — fold in, don't re-derive: (1) each baseline beats `random` at every load, (2)
> each has an equivalence control verified to FAIL on a deliberately broken variant, (3)
> panel comparison across all 12 cells, 30 seeds, Holm-corrected — **"12 cells" assumes the
> current 4-scenario × 3-load-bucket structure; if §1.5.1's probe produces per-scenario-
> class grids of a different shape, this criterion's wording needs updating, not only its
> numbers**, (4) cold-start curve for QMR and DQN, PDR vs episode index, 30 seeds — this is
> the raw material for the absorbing-trap claim itself.
>
> **`PPORouter` is explicitly optional in the source doc — "skip if it threatens M5–M11."**
> No hard go/no-go exists; if skipped, cite HCPMR/GNNPPOR and state that direct comparison
> is invalid across simulators.
>
> **No itemised cost estimate exists for the three baselines individually** in the source
> doc — only a package-level "~1 week, highest priority" for all of Work Package A. The
> day-counts below (1/3/2) are this file's own estimate, not confirmed against it.

**These do not depend on the dataset.** They need only the configs to exist."""),

    ("""## 10.2 The RL environment contract

Observation and action spaces, and **how an episodic simulator becomes a continuing-task
environment without leaking episode boundaries into the learning signal.**

## 10.3 The mask must be applied in the environment""",
     """## 10.2 The RL environment contract

Observation and action spaces, and **how an episodic simulator becomes a continuing-task
environment without leaking episode boundaries into the learning signal.**

**Status: genuinely open, not partially decided.** An earlier reading of this exact section
mistook it for having already settled the continuing-task design. It had not — this is the
header naming the question, with no body. Must be decided and frozen in Phase 0, before the
`run_iter()` refactor (§13 step 5).

**Sub-questions — what's resolved vs what's open:**

- **RESOLVED — the observation structure.** Proven by `ModelActorSimulator`'s existing
  pattern (v9's fix, check 6 verified deterministic): `model.encode()` runs once per frame
  on the full node/edge/adjacency tensors, cached; `model.score()` runs per decision using
  `extract_decision`'s query/candidate output plus that cached encoding. An observation is
  NOT a flat vector — it is `extract_decision`'s output plus a reference to the current
  frame's cached encoding, which updates on its own slower clock. The environment's
  observation space must expose this as a first-class structure, not a standard flat-vector
  Gym interface. **Consequence for the replay buffer, not yet designed:** the encoder is
  part of the Q-network and its weights change during training, so a cached encoding stored
  at collection time goes stale the moment the network updates. The buffer must store raw
  node/edge/adjacency tensors per transition and re-encode at sample time — a memory-cost
  design constraint to size in Phase 0, not an optimisation to discover late.
- **OPEN — episode boundary / continuing-task window.** Full 1000 s segments, or short
  windows sampled mid-trajectory (so congestion has already developed rather than every
  training episode starting cold)? Proposed default, not yet decided: 60–120 s windows
  sampled from mid-trajectory states of the 1000 s dataset episodes. **This decision also
  fixes D1's compute cost** — 30 seeds × 2 arms × 200 episodes at full 1000 s segments is
  roughly 5 days GPU-bound; short windows would collapse that materially.
- **OPEN — per-decision credit assignment** with many packets simultaneously in flight. No
  prior decision exists. `train_supervised_v2.py`'s loss-weighting already has vocabulary
  for this — `eventual_delivered`, currently "ablated, not assumed" — a plausible starting
  point for a delayed-outcome reward rather than only immediate per-hop reward, not a
  default to adopt without testing.

## 10.3 The mask must be applied in the environment"""),

    ("""G2's primary anchor runs `dijkstra`, which ignores `link_quality` and is therefore
**empirically blind** to link-model changes — the M-4 collision flip moved it by exactly
zero (87/280 both ways) while the `spbp` anchor moved 111→112. M5 puts a learned model in
the actor slot, which makes link-quality-sensitive paths the thing under test. Keep both
anchors armed.

---

# 11. WORKING CONVENTIONS""",
     """G2's primary anchor runs `dijkstra`, which ignores `link_quality` and is therefore
**empirically blind** to link-model changes — the M-4 collision flip moved it by exactly
zero (87/280 both ways) while the `spbp` anchor moved 111→112. M5 puts a learned model in
the actor slot, which makes link-quality-sensitive paths the thing under test. Keep both
anchors armed.

## 10.5 The warmstart checkpoint does not exist as a loadable artifact yet — KNOWN GAP

`results/m5_masked/` holds `check4_summary.json` and per-seed rollout JSONs only. No `.pt`
files. `rollout_eval_v2.py` evidently doesn't persist per-seed weights in masked mode.
**Phase 4 (warmstart transfer) needs an explicit checkpoint-save step added** — most likely
a small targeted re-run once the final architecture and post-regeneration config are
locked. Not solved here; noted so it is not discovered mid-Phase-4.

**Validate the load path before it is the only test that matters.** Insert a step (§13,
step 6b) between the new environment's parity gate and v8/regeneration: load the EXISTING
masked GNN checkpoint (once the save-step gap above is closed) into the new environment at
the CURRENT 40 s operating point and reproduce its known 98.0% rollout PDR within noise.
This tests the encode/score plumbing (§10.2) while a valid reference still exists, rather
than the first integration test being after v8, regeneration, and a full retrain — which
makes a failure ambiguous between "the load path is wrong" and "the retrained checkpoint is
bad."

---

# 11. WORKING CONVENTIONS"""),

    ("""# 13. ORDERING SUMMARY

```
1. write rates + frame-stride patch, with the subsample acceptance test   ~1 h    BLOCKING
2. apply v8 + v9, verify configs                                         10 min  BLOCKING
3. regenerate Suite A -> data/phaseB_1000s, gate it                      ~31 h   BLOCKING
   |
   +-- IN PARALLEL: 6. StatefulActor + G-A        1 d   BLOCKING
                    7. QMRRouter + control        3 d   BLOCKING
                    8. DQNRouter (= M7 scratch)   2 d   BLOCKING
4. 40 s vs 1000 s feature comparison                                     20 min  BLOCKING
5. retrain masked warmstart on the new data                              ~1 h    BLOCKING
9. Suites B and C (timing probe first)                                   TBD     not blocking
--> M5 design spec, then M5
```

**Everything marked BLOCKING must be done before M5. Suites B and C serve the comparative
analysis and can follow.**""",
     """# 13. ORDERING SUMMARY — v2, supersedes §13.1 in full

```
1.  git lfs pull; confirm results/checkpoints/*.pt provenance          CLOSED, FILE 1 SS1
2.  config_v2 unification (v11)                                       DONE -- 8 files, verified
2b. v11.2: --duration/--z_min/--z_max/--drain_time on
    experiment_headroom.py + wire config_v2.provenance() into
    its output JSON                                          ~30 min  BLOCKS 3
3.  three-quantity rate probe (SS1.5.1) at 1000s/100-300m,
    via experiment_headroom.py reading config_v2                ~1 h  sets the real rate grid
4.  Phase 0 close, frozen, not revisited after seeing results:
    - environment contract (SS10.2): continuing-task window design
    - per-decision credit assignment (SS10.2)
    - replay-buffer representation (SS10.2, encode/score consequence)
    - D1-D5 -> Suite-A/load-bucket mapping, using the probe's rates
5.  run_iter() refactor of the simulator (steppable), bit-identical
    equivalence gate across all 12 Suite A cells x 30 seeds            BLOCKING
6.  new environment (FANETEnvV2) over run_iter(), built on
    FANETSimulatorV2 via the _select_next_hop extension point;
    SP-BP-parity gate AT THE CURRENT 40s/50-150m operating point,
    12/12 exact                                                        BLOCKING, decisive test
6b. load the EXISTING masked GNN checkpoint (once SS10.5's save-step
    gap is closed) at 40s, reproduce 98.0% rollout PDR within noise    validates the load path
    while a reference still exists
7.  NOW apply v8 + v9 (informed by step 3's rates), regenerate
    Suite A -> data/phaseB_1000s, gate it                      ~31 h   BLOCKING
    |
    +-- IN PARALLEL, now that a parity-verified env exists:
        StatefulActor + Gate G-A          1 d
        QMRRouter + equivalence control   3 d
        DQNRouter (= M7 scratch arm)      2 d
        (PPORouter: optional, skip under schedule pressure)
8.  40s vs 1000s feature comparison                             20 min  BLOCKING
9.  retrain masked warmstart on the new data, re-gate against
    its own new-operating-point rollout PDR (not against 98.0%,
    which was the OLD operating point's number)                  ~1 h   BLOCKING
10. reward implementation; queue-occupancy-distribution check
    (doubles as closing the very_dense zero-occupancy anomaly,
    FILE 1 SS11 -- matters because D2 runs in exactly those cells)
11. Suites B and C (timing probe first)                          TBD    not blocking
--> smoke test, then D1
```

**Everything without "not blocking" must be done before D1. Suites B and C serve the
comparative analysis and can follow.**

---

## 13.1 The original ordering — superseded, kept for reference only

```
1. write rates + frame-stride patch, with the subsample acceptance test   ~1 h    BLOCKING
2. apply v8 + v9, verify configs                                         10 min  BLOCKING
3. regenerate Suite A -> data/phaseB_1000s, gate it                      ~31 h   BLOCKING
   |
   +-- IN PARALLEL: 6. StatefulActor + G-A        1 d   BLOCKING
                    7. QMRRouter + control        3 d   BLOCKING
                    8. DQNRouter (= M7 scratch)   2 d   BLOCKING
4. 40 s vs 1000 s feature comparison                                     20 min  BLOCKING
5. retrain masked warmstart on the new data                              ~1 h    BLOCKING
9. Suites B and C (timing probe first)                                   TBD     not blocking
--> M5 design spec, then M5
```

**Why this was wrong:** step 2 moves duration and altitude before anything has proven the
new RL environment reproduces the teachers at the OLD operating point — a discrepancy found
later could not be attributed to "the environment" vs "the operating point changed." Steps
6–8 (baselines) are scheduled in parallel with regeneration, but Gate G-A needs a correct
environment to test baselines IN, and that environment does not exist until after the
parity gate this ordering never mentions."""),

    ("""The thesis begins at M5. And the risk that most threatens it is that **routing decisions
barely matter at low load** — k=1 and k=∞ horizons disagree on 2–13% of decisions while
delivering identically. The escape is that headroom is **1.6% at low load and 35.6% at
high load**, so the story must be told at high load.

**Confirm that at M6 before M8–M11 consume months.**""",
     """The thesis begins at M5. And the risk that most threatens it is that **routing decisions
barely matter at low load** — k=1 and k=∞ horizons disagree on 2–13% of decisions while
delivering identically. The escape is that headroom is **1.4% at low load and 35.1% at
high load** at the CURRENT operating point (corrected from 1.6/35.6, FILE 1 §6.2) — so the
story must be told at high load, and **must be reconfirmed at M6 at the NEW operating point
and rate grid**, both of which are about to change (§1.5.1, §3).

**Confirm that at M6 before M8–M11 consume months.**"""),
]


def apply(text, edits, label):
    staged, ok = text, True
    for i, (old, new) in enumerate(edits, 1):
        n = staged.count(old)
        if n != 1:
            print(f"  [{label}] anchor {i}: matched {n} times, expected 1  <-- ABORT")
            ok = False
        else:
            print(f"  [{label}] anchor {i}: OK")
            staged = staged.replace(old, new, 1)
    return staged, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='.')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    out, allok = {}, True
    for fname, edits in [(F1, EDITS_F1), (F2, EDITS_F2)]:
        path = os.path.join(a.dir, fname)
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found"); return 1
        text = io.open(path, encoding='utf-8').read()
        if GUARD in text:
            print(f"  {fname}: ALREADY AT v2, skipping"); continue
        staged, ok = apply(text, edits, fname.split('_')[0])
        allok &= ok
        out[path] = staged

    if not allok:
        print("\n  NO FILES WRITTEN."); return 1
    if not out:
        print("\n  Nothing to do."); return 0
    if a.dry_run:
        print(f"\n  DRY RUN OK -- {len(out)} file(s) would change. Nothing written.")
        return 0
    for path, staged in out.items():
        io.open(path, 'w', encoding='utf-8').write(staged)
        print(f"  WROTE {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
