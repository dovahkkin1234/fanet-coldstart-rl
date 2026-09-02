# FILE 1 — APPROACH 2 RECORD: M3 → M4 COMPLETE
## What was built, what was measured, what was learned

**Project:** FANET Lifelong-RL Routing · Shirish Giroti (CS23B2041), IIITDM Kancheepuram
**Guide:** Dr. Preeth Raguraman · **Repo:** `github.com/dovahkkin1234/fanet-coldstart-rl`
**HEAD:** `41ea7a4` (pushed, current) · **Written:** 2026-08-21 · **Revised:** v2, post-M4 pre-M5 investigation

> **This file is the RECORD.** It covers everything up to and including M4, before the
> pre-M5 fixes. The forward plan is in **FILE 2**.
>
> **Provenance convention used throughout:**
> **[MEASURED]** = came out of a run · **[COMPUTED]** = derived from parameters ·
> **[ESTIMATED]** = extrapolated, not verified · **[VERIFIED IN SOURCE]** = read from code.
> Anything without a tag is a design statement, not a result.
>
> Several errors in this project came from believing documentation over source. If a claim
> matters to a decision, re-run or re-read it.

---

# 1. THE THESIS

DRL routing policies have **three lifetime failure modes** that the field's standard
protocol (train to convergence, evaluate on the training distribution) structurally
cannot detect:

| | Failure mode | Proposed mechanism |
|---|---|---|
| **F1** | **Cold start.** A fresh policy routes badly; under load, bad routing creates congestion, congestion degrades links, degraded links make routing harder. Claimed to be an **absorbing trap**. | Warmstart from supervised imitation |
| **F2** | **Catastrophic forgetting.** Adapting to a new regime destroys competence on earlier ones. | Continual learning (EWC / replay) |
| **F3** | **No episodic recall.** A previously-solved situation must be relearned. | Case-based reasoning / episodic memory |

Thesis-defining claim: only the *combination* survives (leave-one-out, M10).

**NONE OF F1, F2 OR F3 HAS BEEN MEASURED.** M1–M4 are infrastructure. This is the single
most important framing statement in this document.

**Why Approach 2 exists:** Approach 1 (single-packet DQN, `docs/PHASE5_REPORT.md`) is a
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
is a scope mismatch, not a defect, recorded once here so it is not re-investigated.

---

# 2. ENVIRONMENT

Windows 11 Pro for Workstations · PowerShell, conda env `fanet`
**bash syntax does not work** — no `&&`, no `unzip`
HP Z8 G5 · Xeon Gold 6426Y 16C/32T · 32 GB RAM · **RTX A4000 16 GB**
Python 3.11.15 · PyTorch 2.11.0+cu128 · working dir `C:\Users\PREETH\FANET_sim\`
PyTorch Geometric / DGL deliberately **excluded** — graphs are 20–45 nodes; dense
`torch.nn` attention is simpler and faster at that size.

---

# 3. MILESTONE STATUS

| Milestone | Deliverable | Gate | Status |
|---|---|---|---|
| M1 | Interference-aware link model | G1 | **PASS** |
| M2 | Multi-packet congestion simulator | G2 6/6 | **PASS** |
| M3 | 8-teacher panel + oracle table | G3 6/6, Holm 12/12 | **PASS** |
| M3.5 | Phase-B dataset | G3.5 8/8 + independent audit 7/7 | **PASS** |
| M4 | Supervised warmstart policy | G4 6/6 | **PASS** |
| M5–M11 | RL loop, cold-start study, warmstart-vs-scratch, CL, CBR, LOO ablation, write-up | — | **NOT STARTED** |

---

# 4. THE SIMULATOR — VERIFIED SPECIFICATION

## 4.1 It is 3-D **[VERIFIED IN SOURCE]**

`mobility.py` header: *"Random Waypoint 3D mobility model… Each drone moves
independently. **No coordination between drones.**"*

`DroneRWP` holds `x, y, z` and `vx, vy, vz`; `_pick_new_waypoint` resamples `dest_x`,
`dest_y`, `dest_z`; vertical motion is capped at 30% of total; `pos()` returns a
3-vector; all distances are 3-D `np.linalg.norm`.

**An earlier report draft claimed "open 2-D space". That was wrong.**

## 4.2 The connectivity graph is a UNIT-DISK GRAPH **[VERIFIED IN SOURCE]**

`link_model_v2.link_exists()` passes `shadowing_db=0.0`, with its own docstring:

> *"shadowing affects quality/PER, not existence, to keep topology construction stable."*

Every graph-building call site passes zero shadowing; only per-slot transmission draws
`rng.normal(0, σ)`, and that draw sits **inside the ARQ retry loop** — a fresh sample per
retry.

**Therefore: edge existence is a deterministic function of Euclidean distance.**

**Consequences, and this bounds a paper claim:**
- Hop distance is essentially a function of Euclidean distance **by construction**.
- The measured geo-vs-hop correlation **r = 0.89–0.93 [MEASURED]** reflects that
  construction, not the channel. It falls short of 1.0 only because hop count is
  integer-valued and depends on where relays happen to sit.
- **Log-normal shadowing is NOT obstacle modelling** on three counts: it does not gate
  connectivity, it is not spatially correlated (independent draw per link), and it does
  not persist (redrawn per retry). Real obstacles do the opposite on all three.

**Correct scope sentence for Paper A:** *"the connectivity graph is a unit-disk graph in
which hop distance is a deterministic function of Euclidean distance"* — **not** "we have
no obstacles". What would falsify the claim is **persistent geometric blockage**, which is
the natural robustness experiment if time allows.

## 4.3 Scenario grid (Suite A) — as of `ebc013d`, before v8

| scenario | UAVs | area (m) | range | speed (m/s) | reachability **[MEASURED]** | mean candidates **[MEASURED]** |
|---|---|---|---|---|---|---|
| `very_dense` | 45 | 700² | 250 | 5–15 | 100.0% | 17.92 |
| `dense_slow` | 30 | 800² | 250 | 5–15 | 99.2% | 9.14 |
| `medium_slow` | 30 | 1300² | 280 | 5–15 | **54.7%** | 4.01 |
| `sparse_fast` | 20 | 1500² | 300 | **35–50** | **21.4%** | 2.68 |

Altitude band **50–150 m** (span 100). Load sweep **0.5 / 2.0 / 4.0** pkt/s per flow.
Flows per episode `N // 4`. Seeds 101–150. Episode **40 s** with a **10 s drain**.

**KNOWN METHODOLOGICAL WEAKNESS.** These four cells vary node count, area, comm range AND
speed *together*, so no cross-scenario result is attributable to any single factor. HCPMR
varies node count alone at fixed geometry — their density sweep is cleanly attributable,
ours is not. Suite B (FILE 2 §6) exists to fix this.

## 4.4 PHY / MAC **[VERIFIED IN SOURCE]**

TX 20 dBm · noise −95 dBm · 2.4 GHz · **path-loss exponent 2.0 (free space, justified in
source as "UAVs fly above obstacles")** · RX sensitivity −85 dBm · log-normal shadowing
σ = 5 dB · 1024-byte packets · Bianchi CW_MIN 16 / 6 backoff stages · carrier-sense 1×
and interference 2× comm range · **queue 50 packets** · TTL 20 hops · **ARQ 5 retries** ·
0.5 s topology frame / 0.01 s channel slot · `SERVICE_RATE = 1` packet per slot
(= 100 pkt/s per node) · `ACT_ALPHA = 1.0`.

**Collision model:** `unsaturated` Bianchi is the default since the M-4 flip. The
`saturated` path contains a quantisation defect — it rounds summed activity to an integer
station count, and Bianchi returns exactly 0 for n ≤ 1, so `p_collision` was **exactly
zero** across much of the measured operating range (activity 0.02–0.09) then jumped
discontinuously. `saturated` is retained only for reproducing pre-flip results.

## 4.5 Flow generation — the choice that dominates absolute PDR **[VERIFIED IN SOURCE]**

```python
src, dst = self.rng.choice(self.N, size=2, replace=False)
```

**Random UAV pairs, no reachability check.** In `sparse_fast` (21.4% reachable) most flows
are impossible by construction. This is deliberate — it is how partition stress is created.
**Every competitor instead routes to a single fixed ground station.**

## 4.6 PDR definitions **[VERIFIED]**

`network_pdr = n_delivered / n_generated`. `pdr_predrain` excludes packets generated in the
final `drain_time`. **Verified identical to four decimals in all 12 scenario × rate
cells** — no in-flight packet is miscounted as a failure.

---

# 5. THE DATASET (M3.5)

**533,237 decisions · 48,000 frames · 600 episodes** (4 scenarios × 3 rates × 50 seeds).
Oracle **SP-BP**, ε = 0.10 (measured 0.0992), `label_fallback = 0.0000` — no teacher ever
abandoned its own rule.

**Split by (scenario, seed), NEVER by row:** train 236,541 / val 48,012 / test 57,206 /
generalisation 191,478 (held-out scenario `medium_slow`). Zero frame leakage, verified
independently. Split assignment is by seed range — train 101–135, val 136–142,
test 143–150 — with a generation-time assertion that every decision lands in some split.

**Trivial baseline [MEASURED]:** always choosing the candidate nearest the destination
scores **71.14%**. Excluding the 28.8% of decisions where the label *is* the destination
gives the **contested floor of 59.46%**. All accuracy figures in this project are contested
accuracy.

**Per-scenario contested floors differ sharply [MEASURED]** — driven by candidate count:
`very_dense` 0.2529 · `dense_slow` 0.4147 · `medium_slow` 0.6832 · `sparse_fast` 0.8035.
**Score check 5 per-scenario, never against the pooled floor** — doing otherwise makes
generalisation appear to *beat* test, which is a candidate-count artifact.

**Feature schema v5:** NODE 9 · EDGE 4 · QUERY 6 · CANDIDATE 4 · `LOCAL_HORIZON = 2` ·
`BUFFERED_REF = 500` · `LIFETIME_REF = 60`.

---

# 6. MEASUREMENTS

## 6.1 SP-BP oracle PDR, per cell, 30 seeds **[MEASURED]**

| scenario | rate 0.5 | rate 2.0 | rate 4.0 |
|---|---|---|---|
| `dense_slow` | **0.798** | 0.479 | 0.273 |
| `very_dense` | **0.752** | 0.410 | 0.242 |
| `medium_slow` | 0.538 | 0.326 | 0.192 |
| `sparse_fast` | 0.406 | 0.324 | 0.214 |

**Range 0.192–0.798. Grand mean 0.4128.**

**NEVER QUOTE THE GRAND MEAN ALONE.** It averages a deliberately designed stress grid — two
partitioned scenarios and a load level at 8× the lowest — and is the least flattering number
the data produces. Always pair a PDR figure with its scenario and reachability.

## 6.2 Mechanism findings **[MEASURED]**

| finding | value |
|---|---|
| Global BFS over a 3-hop horizon | **+0.0065 PDR** — locality is approximately free |
| SP-BP queue term | **+0.0645 PDR** — the only structural feature that matters |
| Queue *differential* → candidate-only | **+0.0000 exactly**, all 12 cells |
| Link quality multiplicative → additive | **−0.0000** |
| BFS hops → geographic progress (scale-corrected) | **+0.0024** — explains 6.7% of the residual |
| DA-GPSR queue-weight sweep, 7560 runs | w = 1 optimal, bracketed both sides |
| **Headroom** | **1.4% at low load, 17.5% medium, 35.1% at high load** (overall 18.0%) |

**Headroom correction [MEASURED, v10, 7/7 verified]:** the original 1.6/18.1/35.6 figures
came from `experiment_headroom.py`, which had the same `id(G)`-keyed cache defect as the M4
check-6 bug (§7.5, error 9 below) — in a different file, found independently during pre-M5
config-duplication cleanup. Fixed via the identical frame-counter pattern. The correction is
small and the conclusion survives: `dense_slow` and `very_dense` (both ~100% reachable) moved
**exactly zero**; the entire delta is `no_route` in the two partitioned cells. Current
figures are in `results/headroom_v10.json`. The pooled 18.45% → 18.0% shift is consistent
with a correction, not a reversal.

**Two limitations the gates report about themselves — put both in the methodology section:**

1. **Oracle degeneracy.** SP-BP wins all 12 cells, so oracle labelling reduces to
   single-teacher imitation. The selection mechanism is validated but never exercised.
2. **Backpressure's queue differential is inoperative** — zero-gradient on 64% of
   decisions, because the packet is dequeued before its decision is recorded so `Q_v ≈ 0`
   always. **G3's check 2 must NOT be cited as evidence that backpressure-style routing
   wins** — SP-BP passes it while performing no backpressure at all.

## 6.3 Congestion coupling — proven, and unique to this work **[MEASURED]**

G2 load sweep, fixed `dijkstra` actor, 40 s episodes:

| rate | mean occupancy | activity | attempts/hop | congestion share of drops | link_error drops |
|---|---|---|---|---|---|
| 0.25 | 0.022 | 0.021 | 1.33 | 0.0% | 0 |
| 0.50 | 0.043 | 0.036 | 1.71 | 25.9% | 4 |
| 1.00 | 0.071 | 0.055 | 2.30 | 56.0% | 25 |
| 2.00 | 0.102 | 0.071 | 2.93 | 77.6% | 101 |
| 4.00 | 0.138 | 0.086 | 3.35 | 88.5% | 269 |

**Offered load rose 8×; link errors rose ~70×.** That disproportion is the signature of a
feedback loop rather than additive overload.

**Isolation experiment**, same load, interference on vs off:
`PDR 0.311, link_error 25` → `PDR 0.375, link_error **exactly 0**`.

**The mechanical chain:** queue fills → `ACT_ALPHA = 1.0` makes transmission activity a
direct function of queue occupancy → more transmission raises Bianchi collision
probability and neighbourhood interference → links degrade → more retries → queue fills
further.

**STILL UNPROVEN:** that a *worse routing policy* at the *same* offered load causes more
congestion than a better one. That is F1 and it is entirely ahead.

---

# 7. M4 — THE COMPLETE RECORD

## 7.1 The two architectures

| | parameters | can a node read its neighbours? |
|---|---|---|
| GNN (`mixer='attention'`), 2-layer dense graph attention | 336,168 | **yes**, exactly 2 hops |
| Matched control (`mixer='mlp'`), node-wise, same depth and width | 335,872 | **no** |

**0.1% apart.** Verified by **gradient reachability, not inspection**: a node 2 hops away
has |∂output/∂input| = **2.53e-03** for the GNN and **exactly 0.00e+00** for the control.

Also available: `mixer='attention_edgekey'` (338,176 params) which puts edge features into
keys and values rather than a per-head scalar bias — built as a GNN-vs-GNN control for
"was the GNN underpowered?", **not run at scale**.

## 7.2 The reversal — ACCURACY **[MEASURED, n=30 paired seeds]**

Tuned config: `lr=1e-3, attn_dropout=0.1, max_epochs=100`.

| arm | GNN | MLP | GNN − MLP | 95% CI | p |
|---|---|---|---|---|---|
| All features supplied | 0.9124 | 0.9229 | **−0.0105** | [−0.0141, −0.0069] | 1.8e-06 |
| Global BFS features masked | 0.7559 | 0.7160 | **+0.0399** | [+0.0363, +0.0435] | 5.2e-20 |
| **Difference of differences** | | | **+0.0504** | **[+0.0460, +0.0548]** | **2.3e-20** |

Cohen's *d* for the interaction = **+4.27**.

**Masked features:** `hop_distance_to_dst`, `cand_hop_distance`, `cand_reachable` — all
three computed by global BFS from the destination.

**Mechanism [MEASURED]:** cost of masking is GNN **−0.1565**, MLP **−0.2069**. The MLP
depends on those features **32% more** — it cannot recompute them; the GNN partly can.
**Message passing and hand-computed graph features are substitutes, not complements.**

**Robustness:**
- MLP wins **6/6 hyperparameter configs, 60/60 seeds** unmasked; 5/6 survive Holm.
- Interaction replicates at **+0.0472 / +0.0472 / +0.0504** across two hyperparameter
  settings and two mask definitions — all CIs overlap.
- Holds out of distribution: **−0.0189 → +0.0162** on held-out `medium_slow`, both p<1e-13.
- **Notable null:** masking two additional *locally computable* congestion features
  (`neigh_buffered_packets`, `neigh_mean_occupancy`) on top of the three global ones
  changed nothing (**+0.0003, p = 0.64**). The entire effect comes from the three
  global-BFS features.

## 7.3 The reversal — PDR **[MEASURED, n=30 paired seeds]**

This closed the last evidentiary gap; before it, the masked result was accuracy-only.

| | attention | mlp | winner |
|---|---|---|---|
| unmasked | 97.8% of SP-BP | 98.9% | MLP |
| **masked (`hop`)** | **98.0%** (0.4283) | 97.5% (0.4263) | **GNN** |

```
PAIRED  mlp − attention  (negative = GNN ahead)
  unmasked        +0.90 pp   CI[+0.67, +1.13]   d=+1.48   p=6.1e-09
  masked          −0.47 pp   CI[−0.78, −0.15]   d=−0.55   p=5.4e-03
  DoD             −1.37 pp   CI[−1.73, −1.01]   d=−1.41   p=1.6e-08   flips 27/30 seeds

COST OF MASKING, per model
  attention       +0.39 pp   CI[+0.14, +0.63]   p=3.5e-03   ← a small GAIN
  mlp             −0.98 pp   CI[−1.26, −0.71]   p=5.7e-08   ← real cost
```

**Masking did not cost the GNN — it gained it.** A prior prediction that masked PDR would
fall below the unmasked figures was **wrong**.

**Per-cell scope limit [MEASURED]:** trained scenarios **−0.53 pp (p=2.5e-03)**; held-out
`medium_slow` **−0.23 pp (p=0.613, NOT significant)**. Trained-vs-held-out DoD =
**−0.30 pp, CI [−1.30, +0.70], p = 0.546.**

**So the honest claim is:** the PDR reversal is established **on trained scenarios**; out
of distribution it is directionally consistent but inconclusive at n=30. The *accuracy*
reversal does hold OOD. Do not claim "regime-dependent" — that would assert the
between-regime difference is real, and this test says it is not distinguishable from noise.

## 7.4 The methodological finding — transferable beyond this project

In the masked arm the **accuracy gap (+5.08 pp) is 10.8× the PDR gap (+0.47 pp)
[MEASURED]**.

**Top-1 imitation accuracy overstates the routing cost of information removal by roughly an
order of magnitude.** A model that cannot reproduce the oracle's *exact* choice can still
pick an equally good alternative; PDR does not care which.

This resolves an apparent contradiction. The earlier ~1:1 accuracy→PDR translation holds
for **architecture comparison at fixed information**, where disagreements reflect genuinely
different routing quality. It does **not** hold for **information removal**. Both are true;
they measure different things.

Independently corroborated: k=1 and k=∞ horizons disagree on **2–13% of decisions while
producing identical PDR**.

## 7.5 All six G4 checks

| check | result |
|---|---|
| 1 — beats trivial floor | **PASS** both, ~+0.29/+0.30 above the contested floor |
| 2 — GNN beats matched MLP | **Answered:** MLP wins with features; **reverses under masking** |
| 3 — advantage grows with load | original **no**; replacement inconclusive (arms disagree), reported honestly |
| 4 — rollout PDR ≥ 90% SP-BP | **PASS** — 98.0% / 97.5% masked; 97.8% / 98.9% unmasked |
| 5 — generalises to held-out scenario | **PASS**, scored per-scenario |
| 6 — bit-reproducible under fixed seed | **PASS**, 0.00e+00 on all six runs |

## 7.6 Artefacts M4 produced

- **Masked GNN warmstart checkpoints**: `results/m5_masked/attention_<seed>_*_masked-*.pt`
  Each checkpoint **records its own mask** in the state dict, so it cannot be loaded under
  the wrong condition. **Trained on the 40 s / altitude-50–150 dataset** — see FILE 2 §5.
- `results/m4_*`, `results/m5_masked/*.json` — all run records.

---

# 8. THE COMPETITION — ALL THREE FULL TEXTS READ

| | **CQMR** (PMC 2025) | **IQMR** (TNSM 2025) | **HCPMR** (IEEE Access 2026) | **OURS** |
|---|---|---|---|---|
| Learning | Coordinated Q(λ) | Independent Q(λ) | MAPPO + GNN, CTDE | imitation → DQN (M5) |
| Simulator | **MATLAB** | **MATLAB** | NS-3.35 + ns3-gym | purpose-built Python |
| Geometry | cylinder R1000 H300 | cylinder R1000 H300 | box 1000×1000×300, **fixed across all terrains** | varies by scenario |
| UAVs | 50 | 50 | 50–200 | 20–45 |
| **Destination** | **1 TBS** | **1 TBS, cylinder base centre** | **1 GCC** | **random UAV pairs** |
| Neighbours **[COMPUTED, span-corrected]** | **2.74** | **2.74** | 8.59 (N=50) – 34.91 (N=200) | 2.34 – 17.16 at span 100 |
| Speed | 10–30 m/s | 10–30 m/s | 10–30 m/s | 5–15 (35–50 in one) |
| Mobility | Gauss-Markov | 3-D Gauss-Markov | RWP + Gauss-Markov | 3-D RWP |
| Propagation | Nakagami-k + PPP | Nakagami-*m* | Nakagami-*m* | free-space + shadowing |
| **Queue model** | **NONE** | FSM trigger only, no capacity | buffer in state, not in reward | **50 pkt + overflow drop** |
| **"Collision" means** | **physical UAV** | **physical UAV** | not modelled as such | **802.11 packet (Bianchi)** |
| Sim time | 8000 episodes | 8000 episodes | **1000 s** | **40 s** |
| Reported delivery | **96.7%** | **83.3%** | **>92%** | 0.4128 mean, 0.798 best cell |
| Seeds / CIs reported | not stated | not stated | not stated | **30–50 paired, Holm** |
| Non-graph control | n/a | n/a | **absent** | **present, matched capacity** |

## 8.1 Five findings from the full texts

1. **All three route to ONE ground station.** CQMR/IQMR state *"number of TBS: 1"*, placed
   at the centre of the cylinder base; HCPMR states *"Number of GCC = 1."*
2. **Neither CQMR nor IQMR models a finite packet buffer.** CQMR mentions queues **zero**
   times. IQMR's `q_r`/`q_t` are state-machine triggers with no capacity, no occupancy
   state variable and no overflow drop. **Congestion collapse cannot occur in their
   models.** *This is the strongest differentiator available and it is checkable from
   their own text.*
3. **Their "collision" is physical UAV collision** — `P^coll = 1 − exp(−r²/2ξˣξʸ)` derived
   from trajectory divergence. Mid-air separation, not channel contention.
4. **CQMR/IQMR use MATLAB with analytical reliability** — coverage probability is
   `E[SIR ≥ SIR_th]`, a closed-form outage expression, and IQMR states outright that SNR
   "is not explicitly computed". **Our packet-level ARQ + Bianchi model is more detailed
   than theirs.** Only HCPMR (NS-3) beats us on simulator fidelity.
5. **CQMR/IQMR at 2.74 neighbours are SPARSER than HCPMR** and comparable to our *sparsest*
   scenarios — yet report 96.7%. **Density is not the explanation; the single sink is.**

## 8.2 The positioning insight — the strongest argument available

HCPMR's state vector (their Eq. 2) is `s_i = [E_i, B_i, p_i(x,y,z), z_i, T_i]` — energy,
buffer, 3-D position, GNN embedding, terrain index. **No BFS hop distance. No
reachability.**

**That is exactly our masked arm** — the condition where our GNN wins.

And HCPMR's *stated rationale* for the GNN (their §II) is that embeddings enable routing
*"without requiring frequent network-wide updates or global topology dissemination."*

> **We are not contradicting HCPMR. We supply the control experiment that validates their
> design rationale — which their own paper lacks.** Their Fig 6(c) ablates GNN embedding
> *dimension* (64/128/256) but never the GNN's presence, and both baselines are Q-learning
> rather than a matched-capacity non-graph model.

## 8.3 Delivery lineage — and why it is not comparable

Q-FANET 50% → QMR 60% → IQMR 83.3% (2.5×10⁴ / 3×10⁴) → CQMR 96.7% (packet loss 3.33%);
HCPMR >92% in NS-3.

**Not comparable to ours.** Different traffic pattern (single sink vs random pairs),
different queue modelling (none vs full), different reliability computation (analytical
vs packet-level).

**Metric ambiguity worth flagging, not accusing:** HCPMR's delivery-success reward counts
forwarding *"to its cluster leader or the GCC"* as success. Whether the *reported* PDR uses
that definition or strict end-to-end is not stated.

---

# 9. PATCH HISTORY

Every patch is an assertion-guarded `str.replace` script: anchors must match **exactly
once**, edits are staged in memory, and nothing is written unless every anchor matches.
Each has an idempotency guard.

| | What it did | Status |
|---|---|---|
| **v1** | Dropped `snr` (invertible function of `distance`) and `hops_so_far` (`= 1 − ttl_left` exactly); k-hop observability scoping (`LOCAL_HORIZON`); G3.5 **check 8** pairwise redundancy (Pearson + Spearman); manifest schema compatibility gate; JSON output for two experiment scripts that could not write results | applied |
| **v2** | Armed the G2 regression anchors; **added a second, link-quality-sensitive anchor** (`spbp`) because the `dijkstra` anchor is empirically blind to link-model changes; RNG parity between collision models in G1; folded MAC contention into G1's `lq` | applied |
| **v3** | Fixed `spbp_ab_full` diverging from panel SP-BP on partitioned graphs (0.4061 → 0.4128, 12/12 cells exact); ported the equivalence control; mirrored the queue-weight sweep grids | applied |
| **v4** | Fixed `spbp_khop(k=∞)` diverging from panel SP-BP (279/345 → 345/345); added `cand_reachable`; geo-progress ablation variant | applied |
| **v5** | Dropped `packet_error_rate` (≈ 1 − `link_quality`, \|r\| = 0.998, found by check 8 on real data); split-completeness assertion | applied |
| **v6** | Fixed `neigh_buffered_packets` clipped on 38.8% of decisions (raw mean 102 against a `/100` normaliser left over from the global feature) → `BUFFERED_REF = 500`; added the G3.5 **saturation diagnostic** | applied |
| **v7** | **`--mask` support in the rollout path** — masking in `_encode` (node/edge) and `_select_next_hop` (query/candidate), presets imported from the trainer so they cannot drift, startup assertion on a real frame and real decision, mask recorded in filenames and checkpoints | **applied** |
| **v8** | Episode duration 40 s → 1000 s; altitude 50–150 → 100–300; `SCENARIOS_TALL` (span 500) probe; `SCENARIOS_DENSITY` (Suite B); `SCENARIOS_CONVERGECAST` (Suite C) | **NOT applied** |
| **v9** | Pins the Suite C sink stationary at the area centre at `z_min`, with a construction-time assertion | **staged, NOT applied — deliberately, pending the new RL environment's SP-BP-parity gate; see FILE 2 §3** |
| **v10** | `HeadroomSimulator._reach_cache` in `experiment_headroom.py` — same `id(G)` defect class as v9 patched in a different file (`rollout_eval_v2.py`), found independently. Frame-counter key; `--cache_mode {fixed,legacy}` for attribution; verified 7/7 including an allocator-independent negative control | **applied** |
| **v11** | `src/config_v2.py` — single shared `SCENARIOS`/`RATES`/`BASE` source. Migrates 8 scripts off duplicated local constants (the same defect class that let v8 miss `experiment_headroom.py` entirely); verified with a positive control (mutate `config_v2.BASE`, confirm every module observes it) | **applied** |

---

# 10. ERRORS FOUND — RECORDED SO THEY ARE NOT REPEATED

| | Error | Lesson |
|---|---|---|
| 1 | Report claimed the simulator uses "open 2-D space" | It is 3-D. **Verify against source, not documentation** |
| 2 | Grand-mean PDR (0.4128 / 0.4370) quoted as a performance figure | It averages a designed stress grid |
| 3 | Claimed HCPMR's 92% PDR figure "does not exist in the paper" | It does — their §V.B. **Abstract-only searching is not reading the paper** |
| 4 | Claimed HCPMR repositions UAVs to maintain connectivity | It does not — Table 4 gives RWP + Gauss-Markov |
| 5 | "We have shadowing, so we model obstacles" | Shadowing does not gate connectivity, is not spatially correlated, does not persist |
| 6 | Predicted masked rollout PDR would fall below unmasked | It **rose** for the GNN (+0.39 pp, p=3.5e-03) |
| 7 | v8 documented a pinned ground sink; the code never pinned it | Sink moved 156 m in 10 s from (517,424,215). **Fixed in v9 with a construction assertion** |
| 8 | Estimated 1000 s episode cost by linear scaling from 40 s | Actual **124 s vs 9 s predicted — 13× wrong**. Measure, do not extrapolate |
| 9 | `id(G)` used as an encoder cache key | CPython reuses freed addresses; one frame was routed with the *previous* frame's encoding. Signature: every "random" drift was identically 2.86e-02 = exactly one packet |
| 10 | `warn_only=True` in the determinism setup | Made the call cosmetic — nondeterministic kernels ran anyway. Two silent no-ops stacked (that, plus setting `CUBLAS_WORKSPACE_CONFIG` after CUDA init) |
| 11 | A verifier asserted schema version `== 4` after v6 made it 5 | Pin behaviour, not version numbers. Use `>=` where the check is "did the bump happen" |
| 12 | Predicted `cand_hop_distance` ↔ `cand_reachable` would be collinear and pre-emptively exempted them | Measured \|r\|=0.839, \|ρ\|=0.260 — not redundant. **Never exempt on a prediction, only on a measurement** |

**Defects found by gates rather than by inspection:** a feature clipped on 38.8% of
decisions; a `link_quality`/`packet_error_rate` pair that were complements (\|r\| = 0.998);
**three separate SP-BP re-implementations that diverged from the panel on partitioned
graphs**; the stale-cache bug above.

**The pattern worth internalising:** at least four defects were in code written to check
other code. A checker that only ever passes proves nothing — every equivalence control in
this project is verified to **fail** on a deliberately broken implementation.

---

# 11. OPEN QUESTIONS

- **The residual SP-BP–DA-GPSR gap (0.0358)** is not explained by queue structure
  (+0.0000), queue weight (w=1 optimal), or progress metric (+0.0024). Untested:
  DA-GPSR normalises progress by *current distance* rather than comm_range, and may
  restrict candidates to those making positive progress.
- **Loop-override contamination.** Override rates: `spbp` 0.088–0.109 vs `da_gpsr`
  0.194–0.258. Does **not** reach the dataset (`canonical_candidates` already excludes
  visited nodes) but contaminates the M3 teacher comparison, which selected the oracle.
- **G1 has no regression anchor.** G2 has two; G1's numbers changed materially with the
  v2 `lq` fold-in and nothing would catch future drift.
- **The `very_dense` zero-occupancy anomaly** (0.000% nonzero `current_queue_occupancy`
  across 129,302 decisions) is explained by the dequeue-before-decision ordering, but that
  explanation is still labelled a hypothesis pending M4 feature analysis.

---

# 12. THE BIGGEST RISK

k=1 and k=∞ horizons disagree on **2–13% of decisions while producing identical PDR**, and
both learned models sit within 1–2% of the oracle. **If routing decisions barely matter at
low load, harsher test conditions buy nothing there either.**

The escape is load: **headroom is 1.4% at low load and 35.1% at high load** (§6.2). The
thesis lives at high load, and **that must be confirmed at M6, at the new operating point
and the new rate grid, before M8–M11 consume months** — the current 1.4/17.5/35.1 figures
are still at the OLD 40 s / altitude-50–150 operating point and the old rate grid; both are
about to change (FILE 2 §1.5.1, §3).
