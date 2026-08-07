# M4 EXECUTION PLAN — GNN + Attention Supervised Warmstart
## Design spec. Approve before implementation, per the project's standing rule.

**Prepared:** 2026-08-03 · **Applies to:** `fanet-coldstart-rl` @ `bb1f0a1` + local unpushed work
**Precondition:** the pre-M4 fix pass (`apply_pre_m4_fixes_v1.py`) is applied and verified.

---

## 0. Standing constraints this plan is written under

1. **The feature fix is ONE cycle.** Apply → regenerate once → re-gate once → build. If G3.5
   check 8 surfaces further redundancy, fold the fix into `apply_pre_m4_fixes_v1.py` and re-run
   the *whole* pass. Do not open a second audit cycle. Ten have already run; the marginal
   return is now in the model, not the data.
2. **Ordering is enforced:** features → model → training → G4. No stage begins before the prior
   gate passes.
3. **Pre-registered predictions are not revised post hoc.** G4 check 3 may legitimately fail.
4. **Verification is by execution.** A script that aborts before its write step still compiles.

---

## PHASE 0 — Unblock (est. 2–4 h, mostly waiting on runs)

### 0.1 Push the two outstanding commits — do this first

`origin/main` is at `bb1f0a1` and nothing follows it. Two semantic changes exist only on the lab
machine, and I confirmed both from the pushed tree:

| What the spec says | What is actually pushed |
|---|---|
| §4.2 `collision_model` default is `'unsaturated'` | `simulator_v2.py:222` → `config.get('collision_model', 'saturated')` |
| §3.4 integer anchor `REGRESSION_DELIVERED/GENERATED` | **The constants do not exist.** Check 6 is `abs(pdr1-pdr2) < 1e-12` between two runs *in the same process* — within-session determinism, not cross-session drift |

Line counts corroborate: spec says `link_model_v2.py` 371 / `simulator_v2.py` 820; pushed are
366 / 808. Every other file matches its stated count exactly. The divergence is precisely the two
files those commits touch.

```powershell
git status
git log --oneline -5
git push origin main
git log --oneline origin/main -5    # confirm both commits landed
```

This is the fourth instance of the staleness pattern. Working-method rule 5 exists for it.

### 0.2 Apply and verify the fix pass

```powershell
python apply_pre_m4_fixes_v1.py --src src --dry-run   # 23 anchors, all must match once
python apply_pre_m4_fixes_v1.py --src src
python verify_pre_m4_fixes.py                          # 26 checks, all must pass
```

The patch stages every edit in memory and commits only if *every* anchor matched exactly once.
A non-matching anchor raises `PatchError` and writes nothing.

**What changed, and why each was necessary:**

| # | Change | Justification |
|---|---|---|
| A1 | `EDGE_FEATURES` drops `snr` (6→5) | The graph's `snr` is `sinr_clean` from `_measured_link` at zero interference. Both directions pass the same `dist_ij`, so `0.5*(s_ij+s_ji)` collapses to a deterministic invertible function of `distance`. Linear r ≈ −0.93 — a Pearson-only screen would have passed it. **Change of plan:** there is no measured SINR to substitute; the interference model is a Bernoulli lethal-interferer indicator, not a power sum (M-8). Synthesising one from `link_quality` would create a fresh redundancy. Dropping is the honest fix. |
| A2 | `QUERY_FEATURES` drops `hops_so_far` (7→6) | `ttl_left + hops_so_far == 1.0` exactly on every row; max observed hops 19 vs TTL 20, so neither clip engages. Pearson r = −1.000. |
| A3 | `n_inflight` → `neigh_buffered_packets`, `network_mean_occupancy` → `neigh_mean_occupancy`, both over `LOCAL_HORIZON` hops | Resolves the observability gap (§0.3). Renamed because a column called `network_mean_occupancy` holding a 2-hop mean is the same naming trap as `snr`. |
| A4 | `degree` normaliser → persisted `degree_ref` | Was `max(N-1,1)` read off the live graph and never persisted, so the persist-and-reuse guarantee did not cover it. Value unchanged, now auditable. |
| B1 | **G3.5 check 8** — pairwise redundancy, per block | Check 6 catches *dead* columns; a duplicated pair passes it with full marks because both copies have healthy variance. That is how `queue_len` survived until a hand comparison caught it. Pearson for linear duplication, Spearman for monotone-nonlinear. |
| C1 | Manifest schema gate in **both** checkers | Every lookup is name-based, so nothing breaks by index arithmetic — what breaks is **version skew**: names resolve against the new module while the array has the old layout. Silent, plausible, wrong. The manifest already persisted all four lists and no checker ever read them back. |
| D1 | `--out` + `json.dump` on `experiment_spbp_mechanism.py` and `experiment_queue_weight.py` | Verified: those two have `json.dump:0 --out:0`; headroom / locality_cost / collision_model / calibration_sensitivity all have both. The Findings 2 and 4 numbers were never writable. |
| E1 | Persist `feature_schema_version` + `local_horizon` in the manifest | Makes the compatibility boundary real and the scoping decision traceable. |

**Check 8 is not tuned to pass.** The harness runs it against planted redundancies and it must
fail: an exact linear duplicate (|r|=1.000) and a monotone-nonlinear one whose linear |r| is
0.976 — *below* the 0.98 Pearson threshold — which only Spearman catches. It reports no
offenders on independent noise.

### 0.3 ⚠ THE ONE DECISION NEEDED BEFORE REGENERATION

`features_v2.LOCAL_HORIZON` currently defaults to **2**. Two defensible choices; pick one now,
because regeneration is a one-shot.

- **`LOCAL_HORIZON = 2` (current default).** Both load aggregates are computed over the 2-hop
  neighbourhood, so every query feature is computable by the forwarding node from state that
  OLSR-family protocols already exchange via HELLO messages. Consistent with the measured result
  that global BFS hop-distance is worth **+0.0005 PDR** — little evidence global aggregation buys
  anything, and a large deployability cost to claiming it. Matches the shallow end of the depth
  sweep and the `spbp_k1..k4` pattern.
- **`LOCAL_HORIZON = None`.** Whole-network aggregation. The **controller-assisted** claim. Must
  be stated as such in the paper, and pairs with the unmasked full-graph encoder.

Whichever you pick, it is persisted in the manifest and asserted by both checkers, so the paper's
deployability claim is traceable to the dataset rather than asserted.

### 0.4 Re-establish the baseline, then regenerate once

```powershell
python src\preflight_simulator_v2_check.py                  # 6/6; anchor 87/280
python src\preflight_teachers_v2_check.py --max_workers 16  # 6/6; Holm 12/12
python src\experiment_spbp_mechanism.py --out results\spbp_mechanism.json
python src\experiment_queue_weight.py   --out results\queue_weight.json
```

Then regenerate and re-gate — **once each**:

```powershell
python src\generate_dataset_v2.py --out data\phaseB
python src\preflight_dataset_v2_check.py --data data\phaseB      # now 8 checks
python src\audit_dataset_v2.py --data data\phaseB --sample 20000 # 7 checks
```

**Expected before regeneration:** both checkers **ABORT** on the existing 533k dataset with a
schema-skew message (verified: 4 problems reported). That is the boundary working, not a failure.

**Expected after:** decision count, labels, `label_fallback`, measured ε, trivial baseline, and
the split sizes should all reproduce their §4.11 values — those are decision-structure statistics
and none of the changed features enter the label. **Feature columns will differ. If a
decision-structure statistic moves, stop and find out why before proceeding.** This is the same
distinction that made the M-4 collision-flip regeneration interpretable.

### 0.5 Commit and push at each verified checkpoint — not at milestone boundaries

Three staleness regressions originated from work sitting uncommitted while a sandbox copy drifted.

**PHASE 0 EXIT GATE:** G3.5 8/8, audit 7/7, `results/spbp_mechanism.json` and
`results/queue_weight.json` on disk and committed, HEAD pushed.

---

## PHASE 1 — M4 build (est. 1–2 weeks)

### 1.1 Pre-commitments — decide now, record now, do not revisit after seeing results

| Item | Decision | Why now |
|---|---|---|
| **M-14 MLP budget** | Identical search budget for GNN and MLP: same grid, same trial count, same early-stop rule, same 3 seeds. Write the grid into `train_supervised_v2.py` as a constant before the first run. | Deciding after seeing results is the definition of an unfair baseline. This is the reviewer item most likely to sink check 2. |
| **Default depth** | `L = 2`, not 3. | §3.6 prediction 1 says shallow wins. Shipping `L=3` as default contradicts the project's own pre-registration (M-16). The sweep covers `{0,1,2,3}` regardless. |
| **Check 2 statistics** | 3 seeds is a **screening** test, not a headline claim. The headline GNN-vs-MLP number comes from the depth sweep at ≥10 seeds. | The project bar is 30 seeds for headline claims; a 3-point paired t-test on the architecture claim is inconsistent with it and near-powerless. |
| **Loss weighting** | Default unweighted. `vote_agreement` and `eventual_delivered` are **ablated, not assumed**. | Already in the spec; restated so it does not drift. |
| **Load balancing** | Report per-bucket metrics from the first run, not as a follow-up. | 59/32/9% split means the pooled number will look flat even if the high-load number is not. |

### 1.2 STEP 1 — `src/model_gnn_attn.py`

Three stages, plain `torch.nn`, no external GNN library.

```
Stage 1  encoder, ONCE per frame, CACHED
    h = Linear(9 → d),  d = 128
    repeat L times (default L = 2):
        h = h + DenseGATLayer(h, adj_mask, edge_feat)    # 4 heads, d_head 32
        h = LayerNorm(h)
Stage 2  q = MLP([h_current, h_dst, query_feat(6)]) → d
Stage 3  k_u = MLP([h_u, edge_feat(v,u), h_dst − h_u])
         logit_u = <q, k_u> / sqrt(d);  invalid → −inf;  p = softmax
```

Cross-entropy against the oracle label. **Defer the dueling value head to M5** — M4 is pure
classification.

**Two assertions that go in the model file, not the gate:**

1. **Adjacency symmetry.** `extract_frame` stores each edge once with `i<j`. If the dense mask is
   built from `edge_index_flat` without symmetrising, attention sees a triangular adjacency and
   the encoder silently becomes a directed message passer that still trains and still reports a
   plausible accuracy. That is the project's recurring signature — *a plausible number sitting on
   a dead mechanism*. Assert `(mask == mask.T).all()`.
2. **Encoder cache hit rate.** Caching is correctness-adjacent, not an optimisation: without it
   the encoder re-runs per decision, up to ~11× in `very_dense` where `num_flows = N//4 = 11`.
   Assert encoder calls per episode ≈ frame count, not decision count.

**k-hop-masked encoder variant.** Add `encoder_horizon: int | None` that masks attention to nodes
within k hops of the current node. If `LOCAL_HORIZON` is set, this is what makes the
decentralisation claim uniform across encoder *and* query features — masking the encoder alone
would leave two globally-aggregated scalars in the query. Reuse the `spbp_k1..k4` pattern; it
folds into the existing `L` sweep as extra columns. **Not a new experiment programme.**

### 1.3 STEP 2 — `src/train_supervised_v2.py`

- Split by `(scenario, seed)` per §4.11. **Never random rows** — all decisions in an episode share
  topology, flows and seed.
- Adam `lr=3e-4`, batch 256 decisions, dropout 0.1, ≤50 epochs, early stop on val top-1,
  grad clip 1.0. Seeds 42/123/7.
- Baselines on the **identical** dataset and split:
  1. trivial rule (slot 0 = nearest-to-destination) — the accuracy floor;
  2. MLP — flat per-candidate features, no graph encoder, **same query-key scoring head** so the
     only difference is message passing. Equal budget per M-14;
  3. SP-BP itself — 100% top-1 by construction; its value is the **rollout PDR ceiling**.
- **Profile one rollout episode before committing to the full evaluation grid.** With per-frame
  caching that is ~80 encoder calls + ~1000 cheap scoring calls per episode. If rollout is too
  slow, that changes the whole G4 shape and it is better to know on day one (risk §9.4).

### 1.4 STEP 3 — GATE G4 (`src/preflight_model_v2_check.py`)

| # | Check | Bar |
|---|---|---|
| 1 | Beats trivial baseline | Against the **59.5% contested floor**, not 71.1%. 28.8% of decisions are free wins where the label *is* the destination; they inflate every model equally and mask the GNN–MLP difference. |
| 2 | GNN beats MLP on test top-1 | Paired t across 3 training seeds. **Screening only** — see 1.1. |
| 3 | GNN advantage **grows with load** | The falsifiable form of "congestion is a multi-hop property." **May FAIL. Pre-registered honest negative result (§3.6 prediction 2), not a bug to engineer around.** M3 evidence points against it: `car` loses to `da_gpsr` in all 12 cells and `spbp_lookahead`'s rank degrades with load. |
| 4 | **Rollout PDR ≥ 90% of SP-BP** on held-out seeds, model as the simulator's actor | **DECISIVE.** Accuracy is a proxy; PDR is what M5 inherits. |
| 5 | Generalises to held-out `medium_slow` | Report the drop whatever it is. State it as **interpolation** (degree 4.37 between 9.20 and 2.51) — and note the confound: `sparse_fast` is the only *fast* scenario in training, so the low-degree end differs in mobility regime too. |
| 6 | Reproducible under fixed seed | — |

**Pre-register the awkward case now:** if the MLP *also* clears check 4's 90%, the GNN wins on
accuracy while its practical value looks unsupported. Decide the reporting line before running,
not after — deciding after is exactly what §3.6 exists to prevent. Suggested line: report both,
state that message passing improves label agreement without improving realised PDR at this
network scale, and let the depth sweep adjudicate.

### 1.5 STEP 4 — depth ablation

`L ∈ {0,1,2,3}` (L=0 = no message passing = the MLP control) × `encoder_horizon ∈ {2, None}`.
Test accuracy and rollout PDR vs depth, **split by load bucket**. With `hop_distance_to_dst`
supplied explicitly, depth measures **congestion-lookahead radius only**, not the model's ability
to approximate BFS — that is what makes the ablation interpretable.
Pre-registered prediction: **shallow wins**.

**PHASE 1 EXIT GATE:** G4 checks 1, 2, 4, 5, 6 pass; check 3 reported honestly either way; depth
sweep complete with per-bucket numbers.

---

## PHASE 2 — M5 onward (scope, not schedule)

M5 is the RL env + DQN loop with the warmstart from M4. Two things carried forward:

- **The second regression anchor** (§3.4) becomes a blocker here, not at M4. M4 touches no
  simulator code; M5 puts a learned model in the actor slot, which makes link-quality-sensitive
  paths the thing under test. G2's current anchor runs `dijkstra`, which ignores `link_quality`,
  so it would not have caught the M-4 collision change. Build the second anchor before M5 code.
- **Loop-override contamination** (§4.2). Correction to an earlier assessment: it does **not**
  reach the dataset — `DatasetSimulator._select_next_hop` returns a member of
  `canonical_candidates(...)`, which already excludes visited nodes, so the base class's override
  at `simulator_v2.py:509` can never fire during generation. It contaminates the **M3 teacher
  comparison** (backpressure 0.44–0.50 vs spbp ~0.10 vs da_gpsr 0.20–0.33), and that comparison
  chose the oracle. Bound it before citing backpressure's numbers, and test whether it explains
  Finding 4's unexplained ~0.029 PDR gap: DA-GPSR has 2–3× as many decisions replaced by
  `unvisited[0]` in arbitrary NetworkX order. Condition outcomes on override/no-override and
  compare on non-overridden decisions only — cheaper than the scaling hypothesis already falsified.

**Risk §9.1 remains the single biggest threat.** Headroom is 1.6% at low load and 35.6% at high.
Everything depends on the high-load regime. **Measure at M6 before investing in M8–M11.**

---

## Stop conditions

Reasons to pause and re-plan rather than push forward:

1. A decision-structure statistic (decision count, label distribution, trivial baseline, split
   sizes) moves after regeneration. Those should not change.
2. Rollout profiling says a full evaluation grid is infeasible. Restructure G4 before building it.
3. G4 check 4 fails badly (rollout PDR well under 90% of SP-BP). M5 inherits this model; a weak
   warmstart makes the entire warmstart-vs-scratch headline (M7) uninterpretable.
4. Check 8 keeps finding new redundancy after the single fix cycle. That would mean the feature
   set has a systemic problem worth designing rather than patching.
