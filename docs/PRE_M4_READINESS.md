# PRE-M4 READINESS ASSESSMENT
## Verdict: **NOT YET.** Four blockers, all cheap. Nine vulnerabilities, three of them new.

**Date:** 2026-08-07 · **Repo:** `fanet-coldstart-rl` @ `f6a2e54`
**Basis:** full G1/G2/G3 re-run post-collision-flip, both mechanism experiments, direct code inspection

---

## PART 1 — WHERE WE ACTUALLY ARE

Phase 0 is roughly 70% done. What's genuinely closed:

| | Status |
|---|---|
| Two unpushed commits | **Pushed.** `bb1f0a1 → 36881dd → 1769e57 → f6a2e54` |
| G2 regression anchor | **Armed**, two pairs, actor-pinned: `dijkstra 87/280`, `spbp 112/280` |
| Second lq-sensitive anchor (§3.4 open item) | **Closed**, and proved load-bearing (see V-7) |
| G1 collision-model parity | **Fixed** (F3 RNG parity, F5 lq fold-in) |
| Pre-M4 feature fix pass | **Applied and verified**, 26/26 |
| G3 re-baseline post-flip | **PASS**, 6/6, Holm 12/12 |
| `queue_weight` JSON | **Written and committed** (superseded — see B-1) |
| `spbp_mechanism` JSON | **Not committed** — correctly withheld |

What has **not** happened: the dataset has not been regenerated, G3.5 and the audit have not run since the schema change, `LOCAL_HORIZON` is undecided, and no M4 file exists.

---

## PART 2 — BLOCKERS (must clear before `model_gnn_attn.py`)

### B-1. Both mechanism experiments need re-running with v3 applied

`spbp_ab_full` was not reproducing panel SP-BP. Measured divergence: **1050 of 1704 decisions (38.4%)** on real mid-rollout graphs from `medium_slow` and `sparse_fast` (mean reachable fraction 0.56). The v3 patch fixes it and adds the control that was missing.

Note the asymmetry: 38.4% of *decisions* differ but mean PDR moves only 0.0067. That is the same phenomenon Part B reports — many distinct routes are near-equally good here — and it is the sharpest available statement of risk §9.1.

`results/queue_weight.json` is committed but was produced on the coarse asymmetric grid. It must be superseded, not amended.

**Expected after re-run:** `spbp_ab_full` mean rises 0.4061 → ~0.4128 and matches G3's per-cell `spbp` exactly; queue-term cost rises 0.0646 → roughly **0.071**. `candqueue → +0.0000` is a within-ablation comparison and should be **unchanged** — if it moves, something else is wrong.

### B-2. `LOCAL_HORIZON` is still undecided

Default is `2`. Regeneration is one-shot. This is a scoping decision about what the paper claims, not a correctness question — see the plan §0.3. **Nothing downstream can start until this is settled.**

Note V-8 below weakens one of the arguments I originally gave for `2`.

### B-3. Dataset regeneration + G3.5 + audit

The existing 533k dataset is schema-skewed; both checkers will correctly abort on it (verified: 4 problems reported, `rc=1`). Regenerate once, re-gate once.

**Watch for:** decision count, label distribution, trivial baseline, split sizes should all reproduce. None of the changed features enter the label. If a decision-structure statistic moves, stop.

### B-4. Recorded numbers that are now stale

`0.0615` (queue-term cost) appears in the spec §5.2 and anywhere it was carried forward. After B-1 it becomes ~0.071. Also `9 distinct orderings` in G3's verdict prose vs `10` in check 4's detail — hardcoded and now wrong.

---

## PART 3 — VULNERABILITY REGISTER

### NEW — found this session, not in any prior list

**V-1. `spbp_khop_next_hop(k=None)` also diverges from panel SP-BP. ⚠ HIGH**

Measured: **279/345 agreement (66 disagreements) — and exactly 66 partitioned cases.** The divergence is *entirely* on partitioned graphs. Same defect class as `spbp_ab_full`, in a third file (`routing_teachers_v3_local.py`), still unfixed.

This matters because `spbp_kinf` underpins two published claims:
- **Part B** ("k=1 and k=inf disagree on 2–13% of decisions yet PDR is unchanged")
- **The locality result** (`experiment_locality_cost.py`: global BFS worth **+0.0005 PDR**)

Both are *within-family* comparisons — `spbp_k1` vs `spbp_kinf`, both from the same k-hop implementation — so the bias cancels and the comparisons remain internally valid. But the **cross-family claim does not hold**: "global topology knowledge is worth nothing **to SP-BP**" requires `spbp_kinf ≈ spbp`, and it doesn't.

**Consequence for the plan:** I cited +0.0005 as evidence for defaulting `LOCAL_HORIZON = 2`. That argument survives but must be restated as *"within the k-hop SP-BP family, horizon buys +0.0005"* — which is weaker support for a claim about SP-BP itself. Fix the same way as `spbp_ab_full`, then re-run locality.

**V-2. Unreachable and far-away candidates are indistinguishable in the feature set. ⚠ MEDIUM**

`cand_hop_distance = min(h_map.get(u, HOP_CAP) / HOP_CAP, 1.0)` with `HOP_CAP = 10`. An unreachable candidate gets exactly `1.0` — identical to a genuine 10-hop candidate. Meanwhile SP-BP, the label source, **excludes** unreachable candidates entirely rather than scoring them.

Mitigating: Part A measures mean hops at 1.73–3.15, so a legitimate 10-hop candidate essentially never occurs. In practice `1.0` *is* a de facto unreachability flag, and the model can learn it. But the property is implicit, undocumented, and scenario-dependent — it would break in any topology with longer paths.

Given 38.4% of decisions in medium/sparse involve an unreachable candidate, this is a large share of the decision space resting on an accidental encoding. Two options, both one-cycle-compatible if decided now:
- **(a)** Leave it, document that `cand_hop_distance == 1.0` functions as an unreachability indicator in these scenarios, and state the scope limit.
- **(b)** Add `cand_reachable` as a 4th candidate feature in the same regeneration. Cheap, explicit, removes the conflation.

**V-3. A whole class of teacher re-implementations is unverified. ⚠ MEDIUM**

Three re-implementations of SP-BP now exist: panel `spbp_next_hop`, `_spbp_generic` (mechanism), `_spbp_qscale` (queue-weight), plus `spbp_khop_next_hop`. **Two of the four diverged**, both on the same branch, both caught only by cross-experiment comparison rather than by a gate.

The pattern is the actual finding: any experiment that re-implements a teacher rather than importing it is a candidate for the same defect. Add an equivalence control to every such script as a standing rule, not one at a time.

### CARRIED FORWARD — known, still open

**V-4. Oracle label degeneracy.** SP-BP wins 12/12 cells. Oracle labeling reduces to single-teacher imitation of SP-BP; the selection mechanism is validated but never exercised. Gate flags this itself. **Methodology-section item — do not let it hide behind a passing gate.**

**V-5. Backpressure's queue differential is inoperative** (zero-gradient on 64% of decisions; the packet is dequeued before its decision is recorded, so `Q_v ≈ 0` always). G3 check 2 must **not** be reported as evidence that backpressure-style routing wins — SP-BP passes it while performing no backpressure at all. Gate flags this.

**V-6. Loop-override contamination, now precisely measured:**

```
dijkstra 0.000 | spbp 0.088–0.109 | gpsr ~0.179 | da_gpsr 0.194–0.258
car 0.195–0.261 | spbp_lookahead 0.220–0.344 | dpp ~0.447 | backpressure 0.449–0.462
```

Oracle table margins run +0.013 to +0.060. DA-GPSR has ~2.5× SP-BP's override rate at high load, and an override replaces the teacher's choice with `unvisited[0]` in arbitrary NetworkX order. That is a live alternative explanation for the residual SP-BP–DA-GPSR gap the queue-weight experiment calls structural.

**Does not reach the dataset** — `DatasetSimulator._select_next_hop` returns a member of `canonical_candidates(...)`, which already excludes visited nodes. It contaminates the M3 teacher comparison, which selected the oracle. Test: condition outcomes on override/no-override, compare on non-overridden decisions only.

**V-7. G1 has no regression anchor at all.** G2 now has two; G1 has none, and its numbers *just changed materially* (F5 lq fold-in: mean lq at a=0.20 moved 0.295 → 0.345 across the sweep). It is a shape gate by design, but there is now nothing to catch drift in it. Cheap fix: record the five-row lq table as integers-scaled anchors.

The value of anchors was demonstrated concretely this session: the M-4 flip moves `dijkstra` by **exactly zero** (87/280 both ways) and `spbp` by 111→112. A single-actor anchor would have slept through a deliberate physics change.

**V-8. Reachability makes the generalisation claim weaker than it reads.** `medium_slow` — the held-out set — is **54.7% reachable**; `sparse_fast` is **21.4%**. G4 check 5 will therefore be dominated by partition structure rather than routing quality. Report reachability alongside the number or it will read as a much worse generalisation result than it is. Compounded by the existing confound: `sparse_fast` is the only *fast* scenario in training, so the low-degree end differs in mobility regime too.

**V-9. M-14 (equal hyperparameter budget for the MLP baseline) undecided.** Must be fixed in code before G4 runs, not after. Deciding post-hoc is the definition of an unfair baseline.

---

## PART 4 — WHAT SURVIVED, AND IT'S A LOT

Not everything is a problem. Post-flip, all of this reproduced:

- **12/12 Holm-significant**, SP-BP tops every cell, 10 distinct orderings, vote agreement 0.646–0.659, all six G3 checks PASS
- **Queue-term cost 0.0615 → 0.0646** — a +0.005 shift, consistent with a −0.7% margin change
- **`candqueue` = exactly 0.0000** across all 12 cells, unchanged
- **Part A correlations 0.889–0.930**, matching the recorded 0.89–0.93
- **G1 shape**: lq 0.972 → 0.345 monotone across the load sweep
- **No teacher took a fallback path on any decision** (0.000 across the panel), with the instrument confirmed live via nonzero flat rates

The collision-model flip changed nothing qualitative. That is a real result and it should be stated plainly in the paper: a documented physics correction moved the competitive margin by −0.7% and left every ranking claim intact.

---

## PART 5 — ORDERED PLAN TO CLEAR THE BLOCKERS

**Step 1 — apply v3, verify, re-run both experiments** (~1–2 h, mostly the 7560-run sweep)

Expected: `full` matches G3 per-cell; queue cost → ~0.071; `candqueue` still exactly 0.0000; the new *sweep optima* line answers whether DA-GPSR's reference sits above or below its optimum.

**Step 2 — fix `spbp_khop_next_hop` the same way, re-run locality** (~30 min)
Then restate the +0.0005 locality claim with its correct scope.

**Step 3 — decide `LOCAL_HORIZON` and V-2 option (a) or (b)**
Both change the regenerated dataset. Decide together, regenerate once.

**Step 4 — regenerate, G3.5, audit** (once each)

**Step 5 — commit everything, then build `model_gnn_attn.py`**

Steps 1–4 are about a day. Nothing here is architectural; it is finishing the audit trail on work already done.

---

## PART 6 — THE HONEST META-POINT

Two of four SP-BP re-implementations diverged from the panel, on the same branch, and neither was caught by a gate — both surfaced from comparing two experiments' printed reference values by hand. That is the eleventh audit-fix cycle, and the standing instruction is that the marginal return is now in the model rather than the data.

I think that instruction is still right, with one amendment: **V-1 and V-3 are the last of this class worth chasing**, because they touch published claims (the locality result) rather than internal consistency. After V-1 is fixed and the equivalence-control rule is standing, further searching should stop even if more is findable. The reason to stop is not that the code is clean — it is that a defect which moves 38% of decisions and 0.0067 of PDR is below the noise floor of every claim M4 will make, and the model is where the remaining uncertainty actually lives.
