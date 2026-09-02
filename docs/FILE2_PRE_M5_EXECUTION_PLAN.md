# FILE 2 — PRE-M5 EXECUTION PLAN
## What to do next, in order, with acceptance criteria

**Project:** FANET Lifelong-RL Routing · **Repo:** `github.com/dovahkkin1234/fanet-coldstart-rl`
**HEAD:** `41ea7a4` (pushed, current) · **Written:** 2026-08-21 · **Revised:** v2
**Companion:** FILE 1 (the record — what was built and measured). Read it first.

> **Provenance convention:** **[MEASURED]** came out of a run · **[COMPUTED]** derived from
> parameters · **[ESTIMATED]** extrapolated and NOT verified · **[TO VERIFY]** must be
> confirmed before relying on it.
>
> **Standing rule for this project:** verify by execution, not inspection. A script that
> aborts before its write step still compiles.

---

# 0. WHERE YOU ARE

**Applied:** patches v1–v7 (M4 complete, all six G4 checks resolved); **v10** (the
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
| config location | scattered, 8 copies (pre-v11) | **single source, `src/config_v2.py`** |

---

# 1. DECISIONS ALREADY LOCKED — DO NOT RE-LITIGATE

## 1.1 Fully decentralised architecture

**GNN (`mixer='attention'`, L=2) with `hop_distance_to_dst`, `cand_hop_distance` and
`cand_reachable` masked at training AND rollout.** Config `lr=1e-3, attn_dropout=0.1,
max_epochs=100`.

Justification, in the order it should appear in the paper:
1. Our architecture has no ground control centre, so global BFS has no natural provider.
2. Obtaining hop-distance in deployment needs a controller or a distance-vector overlay
   with flooding, convergence delay and staleness under mobility.
3. A GNN's L message-passing layers are the **distributed analogue** — L rounds of
   neighbour exchange, obtainable from HELLO messages already being sent.
4. All three competitors use this same state class (geometry + learned neighbourhood
   embedding, no BFS features).
5. **Masking is free or slightly better in PDR [MEASURED]:** attention +0.39 pp
   CI[+0.14,+0.63] p=3.5e-03; mlp −0.98 pp CI[−1.26,−0.71] p=5.7e-08.

**The SP-BP teacher stays genie-aided.** It uses global BFS for labels; the student never
sees those features. **Name this "learning from privileged information" (teacher–student)**
— it is an established technique, and naming it pre-empts the obvious reviewer question.

## 1.2 Episode duration 40 s → 1000 s

Matches HCPMR / CQMR / IQMR (all 1000 s). Unblocks two features that are currently dead
**by construction, not by defect**:

| feature | current state **[MEASURED]** | why 1000 s fixes it |
|---|---|---|
| `energy` | mean 0.9899, **std 0.0205** | 25× longer drain |
| `estimated_link_lifetime` | **60.4% at the ceiling** | `LIFETIME_REF = 60 s` exceeds a 40 s episode, so the cap is unreachable |

**DO NOT also change `LIFETIME_REF`.** The saturation is an episode-length artifact.
Changing both at once makes the improvement unattributable. Fix the cause; leave the
constant alone.

`drain_time` stays at 10 s — a 2–3 hop packet completes in well under a second, so it never
needed to scale with duration.

## 1.3 Altitude 50–150 m → 100–300 m (span 100 → 200)

Matches all three competitors, which all place nodes at 100–300 m.

**Measured cost: ~8% of density. [COMPUTED]**

| scenario | span 100 | **span 200** | span 300 | span 500 |
|---|---|---|---|---|
| very_dense | 17.16 | **15.75** | 13.47 | 9.55 |
| dense_slow | 8.66 | **7.95** | 6.80 | 4.82 |
| medium_slow | 4.14 | **3.87** | 3.42 | 2.49 |
| sparse_fast | 2.34 | **2.21** | 1.99 | 1.48 |

**Why so cheap:** in a thin slab (h ≪ 2R) the communication sphere is effectively a
cylinder, so neighbour volume and total volume both scale linearly with h and the ratio is
independent of altitude. Only once h approaches 2R does the sphere saturate and density
fall as 1/h. With 2R = 500–600 m, span 200 is well inside the slab regime.

**Honest caveat for the paper:** span 200 against 2R = 500–600 is **still slab-like**
(ratio 0.33–0.40). It matches the literature; it does **not** make connectivity genuinely
3-D. That needs span ≈ 2R, which costs ~40% of density — and **no competitor does it
either**.

## 1.4 Altitude probe cells at span 500 — REPORTED SEPARATELY

v8 adds `SCENARIOS_TALL` (`*_tall`, band 100–600). Genuinely 3-D connectivity; no
competing paper evaluates this.

**NEVER fold these into a mean with Suite A.** Density falls ~40%, so PDR falls with it for
reasons that have nothing to do with routing quality. v8 keeps them in a separate dict and
`SCENARIOS` still contains only the four standard cells.

**How to present them:** as a scope probe demonstrating more realistic 3-D data collection
— something the competing papers avoid. Report as its own table.

## 1.5 Load sweep → `(0.05, 0.15, 0.4, 0.7, 1.0)` — SUPERSEDED, see §1.5.1

> **This section's proposed grid is FLAGGED AS LIKELY WRONG and superseded by §1.5.1.**
> The reasoning below (the old grid saturates completely at 1000 s) is correct and worth
> keeping — what changed is HOW to pick the replacement. This five-rate proposal was
> designed before recognising that headroom is expected to be an inverted U in load, and
> its top three rates (0.4, 0.7, 1.0) all sit past the measured 0.25-rate point in
> `dense_slow` — the second-easiest scenario — plausibly landing most cells in or near
> collapse. Kept here for the historical reasoning; **do not implement this exact grid.**

The old `(0.5, 2.0, 4.0)` was calibrated for 40 s episodes. **[MEASURED] at 1000 s,
`dense_slow`:**

| rate | 40 s PDR | 1000 s PDR |
|---|---|---|
| 0.05 | — | **0.6385** |
| 0.25 | — | **0.1995** |
| 0.5 | 0.6286 | — |
| 2.0 | 0.3286 | **0.0250** |
| 4.0 | 0.1917 | — |

At 1000 s the network saturates completely at the old rates. Rate 0.05 at 1000 s
reproduces roughly what rate 0.5 gave at 40 s — **offered load must drop ~10×**.

Five rates instead of three gives a proper decline curve rather than three points.
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

## 1.6 Seeds: 101–150, continuous, all 50

Already the default `SEEDS`. The split ranges (train 101–135, val 136–142, test 143–150)
already partition it. Statistically clean, easy to script, guarantees unseen environments.

An earlier suggestion of 16 seeds drawn from separate blocks was a **workaround for the
dataset-size problem**, not a verification step. Frame striding (§2.2) solves that problem
properly, so the compromise is unnecessary.

## 1.7 Frame striding ≈ 10

**Episode duration and frame *recording* rate are independent knobs.** The simulation still
runs the full 1000 s — congestion buildup, energy drain and link-lifetime dynamics are all
fully realised — we simply do not record every 0.5 s frame.

**[COMPUTED / ESTIMATED]** for 4 scenarios × 5 rates × 50 seeds = **1000 episodes**:

| stride | frames | size **[ESTIMATED]** | decisions **[ESTIMATED]** | vs current dataset |
|---|---|---|---|---|
| 1 | 2,000,000 | ~5.53 GB | ~22.2 M | 41.7× |
| 5 | 400,000 | ~1.11 GB | ~4.4 M | 8.3× |
| **10** | **200,000** | **~0.55 GB** | **~2.2 M** | **4.2×** |
| 20 | 100,000 | ~0.28 GB | ~1.1 M | 2.1× |

Size is extrapolated from 136 MB at 48,000 frames — **verify on the first run**.

**Why this matters beyond disk:** the independent audit loads `frames.npz` fully into
memory. 5.5 GB is an operational limit on a 32 GB machine, not just a storage cost.

---

# 2. STEP 1 — WRITE THE RATES + FRAME-STRIDE PATCH (~1 h)

Neither exists yet. Both go in `generate_dataset_v2.py`.

## 2.1 Rates — trivial

```python
RATES = [0.05, 0.15, 0.4, 0.7, 1.0]        # recalibrated for 1000 s episodes
```

**Load-bucket mapping must be checked — and now depends on §1.5.1, not this section.**
`teacher_panel.load_bucket()` maps rate → low / medium / high. The specific rate values,
and possibly the number of buckets per scenario class, come out of the probe, not the
superseded five-rate proposal in §1.5. **[TO VERIFY]** read the function and confirm it
does not hard-code the old three values regardless of which grid is chosen.

## 2.2 Frame striding — needs care

**Structure [VERIFIED IN SOURCE]:** `DatasetSimulator._note_frame(G)` is called once per
frame; it increments `self._frame_id`, clears `self._hop_cache`, calls
`F.extract_frame(G, self.nc)` and stores the result in `self.frames[self._frame_id]`.
Decisions carry `'frame_id': self._frame_id`. The aggregation loop remaps
`sorted(frames.items())` and does `d['frame_id'] = remap[d['frame_id']]`.

**Therefore frames and decisions MUST be gated together.** A decision referencing an
unrecorded frame will `KeyError` in the remap.

**Required behaviour:**

```python
def _note_frame(self, G):
    self._frame_id += 1
    self._cur_G = G
    self._hop_cache = {}
    self._frame_recorded = (self._frame_id % self.frame_stride == 0)
    if not self._frame_recorded:
        return                      # skip extract_frame entirely
    ids, nf, ei, ef = F.extract_frame(G, self.nc)
    self.frames[self._frame_id] = {...}
```

and in the decision-recording path: **if `not self._frame_recorded`, still choose and
return a next hop normally, but do not append the decision record.**

**THE CORRECTNESS CONDITION — this is the one that matters:**

> A stride-*k* dataset must be a **strict subsample** of a stride-1 dataset from the same
> seed. Same routing decisions, same trajectories, just fewer records.

That means **the routing choice and the ε-deviation logic must be unchanged on skipped
frames.** Skipping `extract_frame` and skipping the 8-teacher vote is safe *if* neither
drives the choice — **[TO VERIFY]** confirm the actor's next-hop selection does not depend
on the recorded vote before skipping it.

**ACCEPTANCE TEST — run this, do not assume:**

```
generate one episode at stride 1 and the same seed at stride 5
assert: the stride-5 decision records are a subset of the stride-1 records
assert: episode-level PDR is IDENTICAL between the two runs
```

If PDR differs, the striding changed the trajectory and the patch is wrong.

## 2.3 Also expose these as CLI arguments

`--frame_stride` (default 10) and `--rates`, both recorded in `manifest.json` so a dataset
can never be mistaken for one generated under different settings.

---

# 3. APPLY v8 AND v9 — MOVED, no longer "Step 2" (10 min)

> **Sequencing correction, supersedes the original early-apply ordering (§13.1).** Do NOT
> run this until the new RL environment's SP-BP-parity gate (§10, built on
> `FANETSimulatorV2`) passes **12/12 exact at the CURRENT 40 s / altitude-50–150 operating
> point**. The teachers were measured at that exact point; it is the only fixed reference
> for telling "the environment is built correctly" apart from "the operating point changed
> underneath it." Moving both at once removes the ability to attribute a discrepancy to
> either cause. See §13 for the corrected position in the overall order.

```powershell
python apply_sim_changes_v8.py --src src --dry-run
python apply_sim_changes_v8.py --src src
python apply_sink_fix_v9.py --src src --dry-run
python apply_sink_fix_v9.py --src src
```

**v8** sets duration 1000 s, altitude 100–300, and adds `SCENARIOS_TALL`,
`SCENARIOS_DENSITY`, `SCENARIOS_CONVERGECAST`. `SCENARIOS` still holds only the four
Suite A cells, so nothing is silently averaged across suites.

**v9** pins the Suite C sink stationary at the area centre at `z_min`, with a
construction-time assertion. **Do not skip v9** — without it the "ground station" is a
flying drone, which invalidates the one thing Suite C controls for.
**[MEASURED before the fix]:** the sink moved 156.4 m in 10 s from (517, 424, 215).

**Verification after applying:**

```powershell
python -c "import sys;sys.path.insert(0,'src');import generate_dataset_v2 as G;print({k:G.BASE[k] for k in ('z_min','z_max','duration','drain_time')});print(list(G.SCENARIOS));print(list(G.SCENARIOS_DENSITY));print(list(G.SCENARIOS_CONVERGECAST))"
```

Expect `z_min=100, z_max=300, duration=1000.0, drain_time=10.0`, four Suite A names, four
`density_*`, four `sink_*`.

---

# 4. STEP 3 — REGENERATE SUITE A (~31 h, unattended)

```powershell
python src\generate_dataset_v2.py --out data\phaseB_1000s --frame_stride 10
```

**Use a NEW output directory.** Do not overwrite `data/phaseB` — the M4 results and the
current masked warmstart were produced from it, and you will want it for comparison.

**[MEASURED] cost basis:** 1000 s episodes at 30 nodes took 99 s (rate 0.05), 112 s
(rate 0.25), 124 s (rate 2.0). All five new rates are ≤ 1.0, so **~110 s/episode
[ESTIMATED]** → 1000 episodes ≈ **30.6 h**.

**Then gate it — both, once each:**

```powershell
python src\preflight_dataset_v2_check.py --data data\phaseB_1000s
python src\audit_dataset_v2.py --data data\phaseB_1000s --sample 20000
```

**What to check in the output:**

| | expectation |
|---|---|
| G3.5 | 8/8 PASS |
| audit | 7/7 PASS |
| check 8 (redundancy) | no offending pairs — the earlier fixes should hold |
| **saturation diagnostic** | `estimated_link_lifetime` should **drop well below 60.4% at max**. If it does not, the episode-length diagnosis was wrong |
| `label_fallback` | 0.0000 |
| split | sums to the total; no empty split |

**If a decision-structure statistic moves unexpectedly — label distribution, trivial
baseline, split proportions — stop and find out why before proceeding.**

---

# 5. STEP 4 — THE 40 s vs 1000 s FEATURE COMPARISON (20 min)

**This validates the entire premise for extending duration. Do not skip it.**

Compare the feature-distribution table between `data/phaseB` (40 s) and
`data/phaseB_1000s`, specifically:

| feature | at 40 s **[MEASURED]** | expectation at 1000 s |
|---|---|---|
| `energy` | mean 0.9899, **std 0.0205** | std should rise materially |
| `estimated_link_lifetime` | **60.4% at max** | should fall substantially |
| `neigh_buffered_packets` | max 0.882 after v6 | should stay unclipped |

**If neither moves, the diagnosis in §1.2 was wrong** and 1000 s bought nothing but
compute. Report that honestly rather than proceeding as though it worked.

---

# 6. STEP 5 — RETRAIN THE MASKED WARMSTART (~1 h)

The current masked checkpoints in `results/m5_masked/` were **trained on the 40 s,
altitude-50–150 dataset**. They do not match the M5 environment.

```powershell
python src\rollout_eval_v2.py --data data\phaseB_1000s --out results\m5_warmstart --seeds 3 --mask hop --lr 1e-3 --attn_dropout 0.1 --max_epochs 100 --mixers attention
```

**Expectation:** rollout PDR **will differ** from the 98.0% measured on the old data — the
operating point has changed (longer episodes, lower rates, higher altitude). That is not a
regression; it is a different measurement. **Report both, do not compare them as though
they were the same experiment.**

The checkpoint records its own mask, so it cannot later be loaded under the wrong
condition.

---

# 7. THE THREE BASELINES (~6 d) — REPOSITIONED, see §13

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

**These do not depend on the dataset.** They need only the configs to exist.

## 7.1 `StatefulActor` protocol + gate G-A (1 d) — BLOCKING

The teacher panel's stateless `(G, current, destination)` signature cannot express an agent
that carries state within an episode. Needs:

- `reset(config)` at episode start
- `select(G, pkt, cands)` per decision
- `update(outcome)` after a transmission result

**Gate G-A:** each baseline must beat `random` at every load — the same floor G3 uses — and
each must have an **equivalence control that is verified to FAIL on a deliberately broken
variant**. Three separate SP-BP reimplementations have already diverged in this project;
this one is not going to be a fourth.

## 7.2 `QMRRouter` — online Q-learning (3 d) — BLOCKING

Per-node Q-table over neighbours; reward combining delay, link quality and energy;
ε-greedy; updated online from transmission outcomes.

**Why it matters twice:** it answers *"how do you compare to prior DRL work?"* **and** its
cold-start behaviour **is** phenomenon F1. The baseline and the demonstration are the same
experiment.

## 7.3 `DQNRouter` — plain DQN, no warmstart (2 d) — BLOCKING

Same feature set and architecture as the masked GNN (`mixer='attention'`, d=128, L=2),
trained online with the M5 RL loop, **randomly initialised**.

**This is simultaneously M7's scratch arm and D1's arm (a).** Build it once, use it twice.
Share `SEARCH_SPACE` with the warmstart arm so M-14 parity is structural rather than
remembered.

---

# 8. THE THREE TEST SUITES

**Suite A — the difficulty case.** The four existing scenarios, at 1000 s and altitude
100–300. Purpose: **graceful degradation under partition**. This is where D5 lives, and
where the harsh-conditions claim is made. **Highest priority — get good results here.**

**Suite B — clean density sweep.** Fixed area 1000×1000, altitude 100–300, range 250,
speed 10–30; **vary node count only**, N ∈ {50, 100, 150, 200}. Removes the four-factor
confound. Expected neighbours **8.59 → 34.91 [COMPUTED]**, matching HCPMR's range.
**Second priority.**

**Suite C — convergecast comparability.** Identical to B **plus one fixed ground sink**.
**Suite B alone is NOT enough** — matching density leaves the traffic pattern different
(random UAV pairs vs many-to-one). **C is the only configuration producing a genuinely
comparable number.** Put resources here **after** A and B are complete.

**Cost warning [ESTIMATED]:** Suites B and C run at 50–200 nodes. Cost scales roughly with
N² per slot, so N=200 will be several times the N=30 measurement. **Run a timing probe
before committing** — the last extrapolation of this kind was 13× wrong.

---

# 9. PRE-REGISTRATION — FIXED BEFORE ANY M5 RUN

**Do not revise these after seeing results.** Every threshold below is fixed now.

## 9.1 The scoping rule — read this before designing any run

- **D2 compares agent vs ORACLE**, so it needs headroom → **high-load dense cells only**.
- **D1 compares agent vs AGENT** (warmstart vs scratch), so it **can** be demonstrated in
  partitioned cells and may be **most dramatic there** — a collapsed scratch agent's floor
  has nothing to do with the oracle's ceiling.

An earlier version of this advice said partitioned cells were unusable for demonstrating a
win. That applies to D2 only, and was wrong about D1.

## 9.2 The five criteria

| | Design | PASS condition |
|---|---|---|
| **D1 ESCAPE** (F1) | High load in `dense_slow`, `very_dense` **and** `sparse_fast`. Arms: (a) scratch DQN, random init; (b) warmstart DQN from the masked GNN checkpoint. Metric: network PDR on held-out evaluation episodes at training episode 1 and at episode **K = 200** *(K fixed now)*. 30 paired seeds | Scratch's improvement CI **contains zero** AND warmstart's **excludes zero and is positive**. **PARTIAL:** both improve but the between-arm DoD is significantly positive → "warmstart accelerates escape" rather than "scratch is trapped". **FAIL:** scratch improves and the DoD CI contains zero → **the absorbing-trap premise is wrong and must be reported as such.** Record the full trajectory, not just endpoints — a slow climb and a flat line are different phenomena |
| **D2 EXCEED ORACLE** | High-load `dense_slow` and `very_dense` **only**. Metric: RL PDR ÷ SP-BP PDR on identical held-out episodes. 30 paired seeds. One-sample t against 1.00 | Mean ratio **> 1.00** with the 95% CI excluding 1.00. Reference: the unmasked M4 warmstart sits at 0.989. **On FAIL,** report honestly that RL matched but did not exceed a hand-tuned expert — and note SP-BP's queue term alone is worth **+0.0645 PDR**, so the agent must first rediscover congestion avoidance before it can add anything |
| **D3 RETENTION** (F2) | Train A (`dense_slow` high load) → B (`sparse_fast` high load) → re-evaluate A. Arms: naive fine-tuning vs a CL method (EWC or replay). Metric: PDR on A after B minus PDR on A immediately after A. 30 paired seeds | CL arm's forgetting significantly smaller (CI on the between-arm difference excludes zero). **PRECONDITION, check at M6:** naive fine-tuning must **actually forget**. If forgetting is negligible, D3 is unmeasurable and the CL leg has nothing to fix — report that rather than working around it |
| **D4 RECOVERY** (F3) | A → B → back to A. Metric: episodes to return to **95%** of the PDR previously achieved on A. Arms: with/without CBR. 30 paired seeds | CBR arm needs significantly fewer episodes. **PRECONDITION:** without CBR, recovery must take a **measurable** number of episodes. If it is near-instant, there is nothing for CBR to improve |
| **D5 GRACEFUL DEGRADATION** | `sparse_fast`, all loads (21.4% reachability). **Robustness claim, explicitly NOT superiority** | Agent maintains **≥ 90% of SP-BP** under partition — no collapse. **STRONGER PASS:** scratch DQN **collapses** under partition while warmstart does not, with a significant paired gap. **No competing paper evaluates a partitioned regime at all**, so even a null-superiority result here is a novel robustness datapoint |

## 9.3 The M6 decision gate

**Before investing in M8–M11, verify that F1, F2 and F3 are each separately measurable and
non-zero.** If forgetting turns out negligible in these regimes, one third of the thesis
has nothing to fix — far better to learn that at M6 than at M10.

**Precedent, not hypothetical:** Approach 1's component ablation already returned uniformly
null once, for a structural reason.

**Also confirm at M6:** that headroom really is load-dependent (1.6% low / 35.6% high). The
whole thesis lives at high load. If that does not hold, stop and re-plan.

## 9.4 The M10 outcome rule

| outcome | permitted claim |
|---|---|
| Removing any one mechanism causes disproportionate collapse | *"only the combination survives"* — the strong thesis |
| Effects are additive and separable | *"each addresses a distinct failure mode; complementary but independent"* |
| One mechanism dominates, others ≈ 0 | Report honestly as a negative result |

---

# 10. M5 DESIGN REQUIREMENTS

Write the design spec **before** any M5 code, per standing rule. Three things it must
settle:

## 10.1 The warmstart loading protocol — NOT a `load_state_dict` call

The M4 checkpoint is a **softmax imitation policy over candidates**, not a Q-function. The
spec must state exactly what transfers and what is randomly initialised. The encoder and
the key/query MLPs are reusable; the output head's *semantics* are not — a logit that means
"probability this is the oracle's choice" is not a Q-value.

## 10.2 The RL environment contract

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

## 10.3 The mask must be applied in the environment

The policy was trained with three columns zeroed. **If the M5 environment supplies their
true values, the policy sees columns it never learned to use.** It will not error — it will
route worse, silently. v7 provides the masking in the rollout path; M5's environment must
use the same mechanism, and the checkpoint's recorded mask is the source of truth.

## 10.4 A second regression anchor becomes load-bearing at M5

G2's primary anchor runs `dijkstra`, which ignores `link_quality` and is therefore
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

# 11. WORKING CONVENTIONS — THESE ARE NOT OPTIONAL

- **Design spec approved before implementation.**
- **Pass/fail gate on each milestone's own premise** before building on it.
- **Verify by execution, not inspection.**
- **Patches are assertion-guarded `str.replace`** with anchors matching **exactly once**;
  edits staged in memory and committed only if every anchor matches; each has an
  idempotency guard.
- **Equivalence controls must be verified to FAIL** on a deliberately broken variant. A
  checker that only ever passes proves nothing.
- **Pre-registered predictions are never revised post hoc.**
- **30+ seeds and Holm correction** for headline claims.
- **Report effect size and CI before the p-value** — at 30 seeds p-values shrink almost
  regardless of effect size.
- **Publish negative results**, including your own refuted hypotheses.
- **Never quote the grand-mean PDR alone.**
- **PowerShell, not bash.**
- **Commit and push at each verified checkpoint**, not at milestone boundaries — three
  staleness regressions in this project came from work sitting uncommitted.

---

# 12. GOTCHAS THAT WILL BITE YOU

| | |
|---|---|
| **Schema version** | v8 does not bump `FEATURE_SCHEMA_VERSION`; the feature *lists* are unchanged. Both checkers assert the manifest matches the live module, so an old dataset will correctly abort against new code — **that is the boundary working, not a failure** |
| **Split ranges** | Assignment is by seed range (train 101–135, val 136–142, test 143–150). Any reduced seed set must draw from **all three** ranges or the v5 assertion aborts generation |
| **`--out` directories** | Always use a fresh one when config changes. Results files are tagged by hyperparameters and mask, but not by dataset |
| **Verifier version pins** | `verify_v5_fixes.py` asserts schema `>= 5` after a fix. Pin behaviour, not version numbers |
| **`load_bucket` mapping** | Currently assumes three rates. **[TO VERIFY]** before regenerating with five |
| **Suite C sink capacity** | At N=200 the sink receives 50 flows. At rate 1.0 that is 50 pkt/s against a per-node capacity of 100 pkt/s **[COMPUTED]** — comfortable. At the *old* rate 2.0 it would have been exactly at capacity and saturated by construction. The rate recalibration fixes this too |
| **Timing extrapolation** | The last one was **13× wrong**. Probe before committing to a long run |

---

# 13. ORDERING SUMMARY — v2, supersedes §13.1 in full

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
parity gate this ordering never mentions.

---

# 14. THE ONE THING TO KEEP IN VIEW

Everything measured so far is **pre-RL**. The 98% figure, the reversal, all of it, is a
hand-designed rule plus a network that learned to **imitate** it. **That is a floor, not a
ceiling.**

The thesis begins at M5. And the risk that most threatens it is that **routing decisions
barely matter at low load** — k=1 and k=∞ horizons disagree on 2–13% of decisions while
delivering identically. The escape is that headroom is **1.4% at low load and 35.1% at
high load** at the CURRENT operating point (corrected from 1.6/35.6, FILE 1 §6.2) — so the
story must be told at high load, and **must be reconfirmed at M6 at the NEW operating point
and rate grid**, both of which are about to change (§1.5.1, §3).

**Confirm that at M6 before M8–M11 consume months.**
