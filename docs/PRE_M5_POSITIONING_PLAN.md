# PRE-M5 POSITIONING PLAN
## Making the contribution defensible against the 2024–2026 DRL-FANET literature

**Date:** 2026-08-10 · **Status:** M4 complete (5/6 checks; check 6 pending re-verify)
**Purpose:** close the four gaps that would otherwise make this work unpublishable at TNSM/TMC/Ad Hoc Networks, *before* M5 consumes the remaining time.

---

## PART 0 — WHAT THE SEARCH ACTUALLY FOUND, INCLUDING A CORRECTION

### 0.1 The competitive landscape

| Work | Venue / year | Method | Simulator |
|---|---|---|---|
| **HCPMR** | 2025/26 | MAPPO + GNN, hierarchical clustering, CTDE | NS-3 / ns3-gym |
| **GNNPPOR** | IEEE Networking Letters 2025 | GNN + PPO, multi-factor joint routing | — |
| **QLR-FANET** | ETRI Journal 2025 | Q-learning + link bitrate adaptation + retransmission penalty | OMNeT++ |
| Li et al. | IEEE Trans. Comm. 2024 | RL energy-efficient fast routing | — |
| **QMR** | Comput. Commun. 2020 | Q-learning multi-objective | — |
| Q-FANET / CQMR / IQMR | 2021–24 | Q-learning variants | NS-3 |
| Arafat & Moh | IEEE IoT J. 2021 | Q-learning topology-aware | — |

HCPMR reports +6.5% PDR over CQMR and +16% over IQMR, +13%/+58% residual energy,
−20%/−38.5% control overhead. **It is the paper to position against.**

### 0.2 CORRECTION TO AN EARLIER CLAIM — CBR IS NOT AN EMPTY GAP

I previously said case-based reasoning in routing looked unoccupied. A targeted
search says otherwise, and the claim must be narrowed:

- **CBR in network management exists, and is old.** LSP restoration in MPLS
  networks via CBR (2003); CBR for smart-grid network reconfiguration; CBR
  similarity functions for bandwidth-allocation switching under dynamic traffic.
  The 4R cycle (retrieve / reuse / revise / retain) is standard vocabulary.
- **Episodic memory in RL is a mature subfield.** Model-Free Episodic Control
  (Blundell et al. 2016), Neural Episodic Control (Pritzel et al. 2017),
  Generalizable Episodic Memory (2021), retrieval-augmented and
  hypernetwork-based task-adaptive retrieval (2023–2026).

**What this changes.** The CBR component cannot be claimed as a novel
mechanism. It must be positioned as *applying established episodic-control
ideas to FANET routing, inside a lifelong protocol* — and it must cite
MFEC/NEC/GEM and the CBR-networking line explicitly. Claiming novelty here
without those citations is the single fastest way to get desk-rejected by a
reviewer who knows the episodic-control literature.

### 0.3 What survives as genuinely novel

| Claim | Status | Evidence |
|---|---|---|
| GNN + DRL routing | **NOT NOVEL — abandon** | HCPMR, GNNPPOR, GraphSAGE-MAPPO all do it |
| CBR / episodic recall mechanism | **NOT NOVEL — reposition as application** | MFEC, NEC, GEM; CBR-MPLS, CBR-smartgrid |
| Continual learning in wireless | **NOT NOVEL, but not in routing** | Davaslioglu 2024 (jamming, PackNet); STCL/G-MAPPO 2026 (UAV edge) |
| **Cold start as an absorbing trap in routing** | **NOVEL — no hit in 6 searches** | term appears only in recsys / serverless / AlphaZero-MCTS |
| **Lifetime failure-mode decomposition for routing policies** | **NOVEL** | every FANET paper found evaluates train→converge→report, stationary |
| **The three-way leave-one-out ablation** | **NOVEL** | no combined warmstart+CL+CBR work in routing |
| **A matched-capacity control showing GNN ≤ MLP** | **NOVEL as a negative result** | no FANET GNN paper found runs a matched control |

Nearest conceptual neighbour: *UAV-Aided Lifelong Learning for AoI and Energy
Optimization in Non-Stationary IoT Networks* — lifelong RL with warmstart
policies for unseen environments, ~8.3% gain. Not routing, no
forgetting/recall decomposition. **Must be cited and differentiated.**

---

## PART 1 — THE CENTRAL STRATEGIC DECISION

**Do not compete on PDR. You will lose, and for the wrong reason.**

Papers in this space report 90%+ PDR. Your oracle SP-BP reports **0.4370**.
That gap is an artifact of operating point, not quality: `sparse_fast` is
**21.4% reachable** and `medium_slow` is **54.7%**. Roughly half your node
pairs have no physical path at all. No routing protocol can deliver a packet
across a partition.

Placed naively in a comparison table, "43% PDR" against a row of 95% figures
reads as a broken system. **This is the single biggest presentation risk in
the paper.**

### The reframe

> Every FANET routing paper surveyed evaluates in a stationary regime: train,
> converge, report. None asks what happens to the learned policy over an
> operational lifetime. We show that DRL routing policies — including recent
> GNN-based designs — fail in three lifetime-specific ways that stationary
> evaluation cannot detect, and that only the combination of warmstart,
> continual learning, and case-based recall survives all three.

This is a **failure-mode-revelation** paper. Its currency is *robustness over
time*, not peak PDR. Your methodological rigour (pre-registration, 30–50 seed
designs, Holm correction, independent audit, published negative results)
becomes the selling point rather than overhead — almost nothing in the table
above does any of it.

---

## PART 2 — WORK PACKAGE A: LEARNED BASELINES  *(~1 week, highest priority)*

**Problem:** every teacher in the panel is classical. "How does this compare
to prior DRL work?" currently has no answer in the artifact.

**Insight that shapes the design:** QMR-style online Q-learning *learns during
the episode from scratch*. Its cold-start behaviour is not merely a
comparison — **it is the phenomenon under study.** The baseline doubles as
the demonstration.

### A.1 `QMRRouter` — online Q-learning (QMR, Comput. Commun. 2020)

Per-node Q-table over neighbours, reward combining delay, link quality and
energy; ε-greedy; updated online from transmission outcomes.

- Runs as an **actor**, not a teacher: it has state that evolves within an
  episode, so it cannot use the stateless `(G, current, destination)` signature.
- Add a `StatefulActor` protocol alongside `TEACHERS` — `reset(config)` at
  episode start, `select(G, pkt, cands)` per decision, `update(outcome)` after.
- **Equivalence control, mandatory.** Three SP-BP reimplementations have
  already diverged in this project. Assert QMR's Q-update against a
  hand-computed reference on a fixed 5-node graph.

### A.2 `DQNRouter` — plain DQN, no warmstart

Same feature set and architecture as the M4 MLP (`mixer='mlp'`, d=128, L=2),
trained online with the M5 RL loop, initialised randomly.

- This is **exactly the scratch arm of M7's warmstart-vs-scratch comparison.**
  Building it now means M7 is a re-run, not a new build.
- Shares `SEARCH_SPACE` with the warmstart arm, so M-14 parity is structural.

### A.3 `PPORouter` — optional, only if time permits

A single-agent PPO with the same encoder. Stands in for the
GNNPPOR/HCPMR family. **Skip if it threatens M5–M11.** Cite those papers and
state that direct comparison is invalid across simulators.

### A.4 Gate G-A (must pass before M5 code)

1. Each baseline beats `random` at every load — the same floor G3 uses.
2. Each has an equivalence control that **fails** on a deliberately broken variant.
3. Panel comparison across all 12 cells, 30 seeds, Holm-corrected.
4. Cold-start curve recorded for QMR and DQN: PDR vs episode index, 30 seeds.
   **This is the raw material for the absorbing-trap claim.**

---

## PART 3 — WORK PACKAGE B: THE OPERATING-POINT SECTION  *(~2 hours, do today)*

Draft text, numbers already measured:

> **Operating point and comparability.** Reported PDR in FANET routing spans a
> wide range, driven primarily by node density and connectivity rather than by
> protocol quality. Our scenario grid deliberately includes heavily partitioned
> regimes: mean end-to-end reachability is 100% in `very_dense`, 99.2% in
> `dense_slow`, 54.7% in `medium_slow` and 21.4% in `sparse_fast`. Mean
> candidate count per forwarding decision correspondingly falls from 17.9 to
> 2.7. Absolute PDR is therefore bounded well below unity by topology alone —
> our oracle teacher SP-BP achieves 0.4370 network PDR averaged over the grid.
> Numbers here are **not** comparable with studies reporting 90%+ PDR in fully
> connected scenarios; all comparisons in this paper are made against baselines
> re-implemented inside the same simulator, at the same operating point.

Add a reachability column to every results table. Report accuracy above the
**per-scenario** contested floor (very_dense 0.2529, dense_slow 0.4147,
medium_slow 0.6832, sparse_fast 0.8035) — never the pooled 0.5946, which
makes generalisation appear to beat test purely through candidate count.

---

## PART 4 — WORK PACKAGE C: RELATED-WORK POSITIONING  *(~1 day)*

Four subsections, each ending in an explicit differentiation sentence.

1. **RL/DRL routing in FANETs** — QMR, Q-FANET, CQMR, IQMR, QLR-FANET,
   Arafat & Moh, Li et al. *Differentiator: all evaluate stationary; none
   measures policy degradation over a lifetime.*
2. **GNN-based routing** — HCPMR, GNNPPOR, GraphSAGE-MAPPO. *Differentiator:
   none runs a matched-capacity non-graph control; we do, and the graph
   encoder does not win.*
3. **Continual learning in wireless** — Davaslioglu et al. 2024 (PackNet,
   jamming); STCL/G-MAPPO 2026 (UAV edge). *Differentiator: applied to
   jamming and coverage, not routing; no cold-start or recall component.*
4. **Episodic memory and CBR** — MFEC, NEC, GEM, retrieval-augmented RL;
   CBR-MPLS, CBR-smartgrid. *Differentiator: we apply established episodic
   control to routing inside a lifelong protocol; the mechanism is not claimed
   as novel.*

---

## PART 5 — WORK PACKAGE D: NS-3 CROSS-CHECK  *(~3 days, scope tightly)*

**Goal is narrow: convert "custom simulator" from a fatal objection into a
stated limitation.** Do NOT attempt to reproduce the whole study.

Validate **one** thing — the link model — at **one** operating point:

- NS-3 with 802.11 PHY, `LogDistancePropagationLossModel`, matched
  transmit power / frequency / comm range.
- Static 30-node topology, one packet flow, sweep offered load.
- Compare PDR-vs-distance and PDR-vs-load curve **shapes**, not absolute values.
- **Pre-register the criterion before running:** monotone decreasing in both,
  and the load at which PDR falls below 0.5 agreeing within a factor of two.

If it matches, one paragraph and a figure. If it doesn't, that is a finding
about your link model and better learned now than from a reviewer.

---

## PART 6 — SCHEDULE AND ORDERING

| # | Work package | Effort | Blocks M5? |
|---|---|---|---|
| 0 | Finish check 6 re-verify; commit M4 | 2 h | no |
| 1 | **B — operating-point section** | 2 h | no |
| 2 | **A.1 QMR baseline + control + G-A** | 3 d | **YES** |
| 3 | **A.2 DQN baseline** | 2 d | **YES** — it *is* M7's scratch arm |
| 4 | C — related work | 1 d | no |
| 5 | D — NS-3 cross-check | 3 d | no (parallelisable) |
| 6 | A.3 PPO baseline | 3 d | **skip unless time allows** |

**A.1 and A.2 block M5** and nothing else does. Everything else can run
alongside or after.

---

## PART 7 — THE REVISED CONTRIBUTION STATEMENT

Replace the old Tier-1/Tier-2 novelty list with:

1. **A lifetime failure-mode taxonomy for learned routing** — cold start,
   catastrophic forgetting, no episodic recall — each operationalised as a
   measurable quantity with a pre-registered falsification condition.
2. **Evidence that cold start is an absorbing trap under load** in a
   congestion-coupled simulator, and that warmstart escapes it. *(M5–M7)*
3. **A leave-one-out ablation** showing no single mechanism suffices. *(M10)*
4. **A negative architectural result**: a matched-capacity MLP control matches
   or beats a graph-attention encoder on both next-hop accuracy (n=50, Holm)
   and rollout PDR (n=30), in a literature that assumes the opposite.
5. **A reproducible artefact**: gates, independent audit, pre-registered
   predictions, determinism controls, and published negative results.

Point 4 is worth more than it looks. It is the only claim you can make
**today**, it contradicts the field's prevailing assumption, and it is backed
by a control that essentially nobody in this literature runs.
