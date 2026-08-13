# NOVELTY AUDIT AND EXPERIMENT PLAN
## Answering four questions honestly, then a plan that makes each claim defensible

**Date:** 2026-08-10 · **Basis:** 8 literature searches + code-level review of the M4 artefacts

---

# PART 1 — NOVELTY AUDIT

## 1.1 The verdict table

| # | Claim | Verdict | Nearest prior work | What it needs to survive review |
|---|---|---|---|---|
| N1 | GNN + DRL routing | **DEAD** | HCPMR, GNNPPOR, GraphSAGE-MAPPO | Abandon entirely |
| N2 | CBR / episodic recall mechanism | **NOT NOVEL** | MFEC 2016, NEC 2017, GEM 2021; CBR-MPLS 2003, CBR-smartgrid | Reframe as application; cite all six |
| N3 | Continual learning method | **NOT NOVEL** | EWC, PackNet; Davaslioglu 2024 (jamming), STCL 2026 (UAV edge) | Reframe as application to routing regimes |
| N4 | **Cold start as an absorbing trap in routing** | **NOVEL** | none found in 8 searches | M5–M6 must MEASURE it, not assume it |
| N5 | **Lifetime failure-mode decomposition for routing** | **NOVEL** | surveys name forgetting as an open challenge; nobody measures it in FANET routing | Verify by reading HCPMR + GNNPPOR in full |
| N6 | **Three-way leave-one-out ablation** | **NOVEL** | no combined system exists in routing to ablate | Must survive the null-result risk (§3.4) |
| N7 | **Matched-capacity control: GNN ≤ MLP** | **PROMISING, UNDER-EVIDENCED** | no FANET GNN paper runs a matched control | Three experiments — see Part 2. **Currently not publishable.** |
| N8 | Reproducible artefact (gates, audit, pre-registration, determinism) | **Differentiator, not a contribution** | — | Report as methodology, don't claim as novelty |

## 1.2 The single most useful citation found

An IoT RL-routing survey explicitly names **catastrophic forgetting and
non-stationary data distributions** among the open challenges for RL-based
routing. This is the perfect motivating citation: the field *acknowledges* the
problem in surveys while no FANET routing paper *measures* it.

Your framing sentence writes itself:

> Surveys of RL-based routing identify catastrophic forgetting and
> non-stationarity as open challenges [survey]. To our knowledge, no FANET
> routing study measures policy retention across regime changes; evaluation
> protocols uniformly train to convergence and report on the training
> distribution.

## 1.3 Honest caveat on the audit

This audit is built from abstracts, search snippets and reference lists — not
full-text reads. **Before making any "to our knowledge, no work..." claim in
the paper, read HCPMR and GNNPPOR end to end.** If either runs a non-graph
control or a sequential-regime evaluation, N5 and N7 weaken immediately. That
is a half-day of reading that protects two of your three surviving claims.

---

# PART 2 — FORENSIC REVIEW OF THE GNN ≤ MLP RESULT

You asked whether both models were up to standard and whether the result is
flawed. **I found five problems. One of them could reverse the conclusion.**

## 2.1 CRITICAL — the hop-distance features pre-empt the GNN's entire job

`QUERY_FEATURES` contains `hop_distance_to_dst`. `CANDIDATE_FEATURES` contains
`cand_hop_distance` and `cand_reachable`. **All three are computed by global
BFS from the destination** (`hop_distances_to(G, dst)`).

Propagating topological distance information is precisely what message passing
is *for*. You are handing it to **both** models as a pre-computed feature.

So the experiment as run answers:

> *Given explicit global BFS topology features, does message passing add
> anything?* → **No.**

It does **not** answer:

> *Do graph neural networks help FANET routing?*

A reviewer will spot this in the feature table and ask the obvious question.
Worse — your own locality experiment already predicts the answer: global hop
knowledge is worth only +0.0065 PDR, and geographic distance correlates
0.89–0.93 with hop distance. You have independently established that the
information the GNN would propagate is already available cheaply.

**THE DECISIVE EXPERIMENT — a 2×2:**

| | with hop features | without hop features |
|---|---|---|
| GNN (attention) | measured: 0.8869 | **?** |
| MLP (matched) | measured: 0.8977 | **?** |

- **If the GNN wins without hop features:** the finding becomes *"explicit
  topology features substitute for message passing at a fraction of the
  compute"* — practical, actionable, explains the mechanism, and is a
  **substantially better paper** than a flat negative.
- **If the GNN still loses:** the negative result becomes close to bulletproof,
  because the most obvious objection has been pre-empted.

Either outcome is publishable. The current single cell is not.

**Cost: near zero.** Masking feature columns is a train-time operation. No
regeneration. ~2.6 h per configuration × 2 = ~5 h.

## 2.2 MAJOR — no hyperparameter search ever happened

M-14 says "equal hyperparameter budget." The budget was **one fixed config for
both**: `d=128, layers=2, heads=4, dropout=0.1, lr=3e-4`.

That is symmetric, but symmetric-at-zero. Attention models routinely need a
different learning rate and often warmup relative to plain MLPs. A reviewer
will say the GNN was never given a fair chance, and they will be right.

**Fix:** a small shared grid — `lr ∈ {1e-4, 3e-4, 1e-3}` × `d ∈ {64, 128}` —
6 configs, both models, 10 seeds each, select on val, then re-run the headline
comparison at 30 seeds with each model's own best config. ~120 short runs.

## 2.3 MAJOR — the dropout asymmetry is still unresolved

`DenseGATLayer` applies dropout to the **attention weights**; `NodeMLPLayer`
applies it to a **256-dim hidden layer**. Same nominal p=0.1, radically
different severity: with node degree 2.7–17.9, dropping 10% of attention
weights removes a meaningful share of a node's neighbourhood, while dropping
10% of 256 hidden units is mild regularisation.

**Fix:** add `dropout ∈ {0.0, 0.1}` to the shared grid in §2.2. Do **not** zero
it for the GNN only — that breaks parity.

## 2.4 MODERATE — the GNN may be underpowered by my implementation

`DenseGATLayer` uses edge features only as a per-head scalar bias:

```python
self.e = nn.Linear(edge_dim, heads)   # one scalar per head
logits = logits + self.e(edge_dense).permute(0, 3, 1, 2)
```

Stronger GAT variants concatenate edge features into the key/value projections
rather than adding a scalar. If the GNN loses partly because *my* attention
layer uses edge features weakly, the negative result is about my
implementation, not about message passing.

**Fix:** one variant with edge features concatenated into keys, run at the same
budget. If it doesn't change the outcome, say so explicitly in the paper — it
closes the objection.

## 2.5 MODERATE — the depth sweep was specified but never run

The plan pre-registered `L ∈ {0,1,2,3}`. Only `L=2` was run. `L=1` may suit a
network whose locality result says a 1-hop horizon is nearly free.

## 2.6 What is genuinely sound

To be fair to the result, these are real strengths and should be stated:

- **Parameter parity is excellent**: 336,168 vs 335,872, 0.1% apart.
- **Message passing verified by gradient reachability**, not by inspection:
  2-hop |grad| = 2.53e-03 for the GNN, exactly 0.00e+00 for the control.
- **The epoch-cap confound runs the wrong way for the GNN**: the MLP trains
  longer (36.5 vs 28.3) and hits the cap more (4 vs 2), so more training could
  only widen the MLP's lead.
- **Two independent metrics agree**: accuracy (n=50, Holm) and rollout PDR
  (n=30, p=8e-9), at roughly 1:1 translation.
- **Determinism now verified** to 16 digits on CUDA.

## 2.7 Verdict on N7

**The result is real but the claim is currently far too broad.** What you can
defend today is:

> With explicit BFS hop-distance features supplied, a graph-attention encoder
> does not outperform a matched-capacity node-wise MLP on next-hop imitation
> or rollout PDR, at a single untuned hyperparameter configuration.

That is narrow and hedged. After §2.1–2.3 it becomes:

> Message passing provides no measurable benefit over a matched-capacity
> node-wise control, with or without explicit topology features, across a
> shared hyperparameter grid.

**That second version is a standalone GLOBECOM/ICC paper.** The first is not.

---

# PART 3 — THE LEAVE-ONE-OUT ABLATION

## 3.1 Is it valid?

**Yes.** Leave-one-out component ablation is standard ML methodology. With
three mechanisms you need four arms — full, −warmstart, −CL, −CBR — plus
optionally a none arm as the floor.

## 3.2 Is it doable?

**Yes.** 4 arms × 30 seeds = 120 RL training runs. Given M4 supervised
training runs at ~100 s, RL runs will be slower but the same order. Budget
roughly a week of unattended compute. **Feasibility is not the risk.**

## 3.3 Does it hold major significance?

**Conditionally, and this is where you should be careful.**

The thesis-defining claim is *"only the combination survives."* That requires
**superadditivity** — removing any one mechanism must cause disproportionate
failure. If instead the three contribute independently and additively, the
ablation produces a tidy linear decomposition and the headline collapses to
*"warmstart helps, CL helps a bit, CBR helps a bit."* That is a normal paper,
not a thesis-defining one.

**Pre-register now, before M5, what you conclude in each case:**

| Outcome | Claim you may make |
|---|---|
| Removing any one causes disproportionate collapse | "Only the combination survives" — the strong thesis |
| Effects are additive and separable | "Each addresses a distinct failure mode; contributions are complementary but independent" — weaker, still publishable |
| One mechanism dominates, others ~0 | Report honestly. The paper becomes "warmstart is what matters, and here is the evidence the other two do not" — a legitimate negative result |

## 3.4 Why hasn't it been done? And the risk nobody will tell you

Two reasons, both mundane:

1. **The combination doesn't exist in routing to ablate.** You cannot ablate a
   system nobody has built.
2. **Genre convention.** Networking papers overwhelmingly report "ours vs
   baselines" tables; component LOO ablations are ML convention. Importing that
   rigour is a modest methodological strength, not a technical barrier.

**The risk you should take seriously:** this project has *already run* a
component ablation that returned uniformly null. Approach 1's
Dueling/reward-shaping/curriculum ablation found nothing, for a structural
reason (single-packet ⇒ zero queue occupancy ⇒ no congestion to be dueling
about). That is a precedent, not a coincidence.

**Mitigation — do this at M5/M6, not M10:** verify that the three mechanisms
are addressing genuinely different failure modes *before* betting the thesis
on the ablation. Concretely: show cold start, forgetting, and recall failure
are separately measurable and separately non-zero. If forgetting turns out to
be negligible in your regimes, the CL leg has nothing to fix and the three-way
story is dead — better to learn that at M6 than at M10.

---

# PART 4 — WHAT "EVERY FANET PAPER IS STATIONARY" MEANS

Precisely, the evaluation protocol in every FANET routing paper I found is:

1. Train on a scenario or mixture until convergence.
2. Evaluate on the same (or a similar) distribution.
3. Report PDR, delay, overhead, energy.

**What is absent:** a *sequence* of regimes, trained one after another, with
**retention on earlier regimes re-measured after later training**. That
measurement is what defines catastrophic forgetting, and it is the standard
protocol in continual-learning work (backward transfer, forgetting rate).

**An important nuance you must not overstate.** HCPMR *does* evaluate across
urban, desert and mountainous scenarios. But cross-scenario **evaluation** is
not sequential **training with retention measurement**. Generalisation asks
"does a fixed policy work elsewhere?" Forgetting asks "after adapting
elsewhere, does it still work *here*?" Different questions.

Your differentiating sentence must therefore be surgical:

> Prior FANET routing work evaluates learned policies across scenarios but
> trains them once; we sequence regimes and measure retention on earlier
> regimes after adaptation to later ones.

Do **not** write "no prior work evaluates across scenarios" — that is false and
easily checked.

---

# PART 5 — IS REGIME-BASED CONTINUAL LEARNING STILL VALID?

**Valid: yes. Novel as a method: no. Novel as an application: yes, narrowly.**

- The *methods* (EWC, PackNet, replay, policy consolidation) are established.
- PackNet has already been applied to a wireless problem — jamming mitigation,
  with a systematic methodology for sequential task learning.
- Continual learning has already been applied to UAV networks — STCL/G-MAPPO,
  for coverage/energy under user-distribution shifts.
- **Applying it to routing regimes appears unoccupied.**

**Framing that survives review:** the contribution is the *problem
formulation* — treating FANET density/mobility/jamming regimes as a task
sequence for a routing policy — not the CL algorithm. Use an off-the-shelf
method (EWC or replay), cite it plainly, and do not dress it up. Reviewers
punish overclaimed method novelty far harder than they punish honest
application work.

**On its own this leg is weak.** It is only strong as one of three.

---

# PART 6 — VENUE ASSESSMENT

| Venue class | Examples | Realistic? | Why |
|---|---|---|---|
| Tier-1 conference | MobiCom, INFOCOM, SIGCOMM, NSDI | **No** | Simulation-only, custom simulator, no hardware. Your spec already ruled these out — correctly. |
| Tier-1 journal | IEEE/ACM ToN, TMC, JSAC | **Hard but not impossible** | Needs the NS-3 cross-check, learned baselines, and a positive LOO result |
| **Tier-2 journal** | **TNSM, IoT-J, TVT, Ad Hoc Networks, Computer Networks** | **Yes — this is the target** | Matches scope, rigour, and simulation-only status |
| Tier-2 conference | GLOBECOM, ICC, WCNC, VTC | **Yes, comfortably** | Ideal home for the GNN≤MLP negative result as a standalone short paper |

## 6.1 The strategic recommendation

**Split the work into two papers.**

**Paper A (short, submit within ~6 weeks):** the architecture negative result.
After §2.1–2.3, you have: matched-capacity control, 50 paired seeds,
Holm-corrected, two independent metrics agreeing 1:1, hop-feature ablation,
shared HP grid, determinism verified. Target GLOBECOM/ICC.

**Paper B (the thesis):** the lifetime failure-mode work. Targets TNSM or Ad
Hoc Networks after M5–M11.

Why this matters: it converts a single all-or-nothing submission into a
publication in hand plus a stronger main paper. It also de-risks the real
possibility that the LOO ablation comes back null.

---

# PART 7 — THE PLAN

## Phase 1 — Make N7 defensible (~1.5 weeks, mostly unattended)

| # | Experiment | Cost | Decides |
|---|---|---|---|
| 1.1 | **Hop-feature ablation 2×2** (mask `hop_distance_to_dst`, `cand_hop_distance`, `cand_reachable` at train time), 30 seeds | ~5 h | Whether N7 is narrow or broad |
| 1.2 | **Shared HP grid**: lr ∈ {1e-4,3e-4,1e-3} × d ∈ {64,128} × dropout ∈ {0,0.1}, 10 seeds screening, both models | ~15 h | Closes "the GNN was never tuned" |
| 1.3 | **Depth sweep** L ∈ {0,1,2,3} at the winning config, 30 seeds | ~10 h | Pre-registered; closes "wrong depth" |
| 1.4 | **Edge-feature-in-key GAT variant**, 30 seeds | ~3 h | Closes "your GAT was weak" |
| 1.5 | Re-run headline comparison at each model's own best config, 30 seeds | ~3 h | The publishable number |

**Gate G-N7:** N7 is publishable only if the MLP still wins, or ties, after all
five. If the GNN wins in any cell, the paper becomes the *conditional* result —
better, but a different paper.

## Phase 2 — Blockers for M5 (~1 week)

| # | Task | Blocks M5? |
|---|---|---|
| 2.1 | `StatefulActor` protocol + gate G-A scaffolding | YES |
| 2.2 | `QMRRouter` (online Q-learning) + equivalence control | YES |
| 2.3 | `DQNRouter` (= M7's scratch arm) | YES |
| 2.4 | Cold-start curves for both, 30 seeds — raw material for N4 | YES |

## Phase 3 — Parallelisable, non-blocking

| # | Task | Effort |
|---|---|---|
| 3.1 | Operating-point / reachability section (draft text already written) | 2 h |
| 3.2 | **Read HCPMR + GNNPPOR in full** — protects N5 and N7 | 4 h |
| 3.3 | Related-work section, four subsections | 1 d |
| 3.4 | NS-3 link-model cross-check, one operating point, pre-registered criterion | 3 d |

## Phase 4 — De-risk the thesis before M10

| # | Task | Why |
|---|---|---|
| 4.1 | At M6, verify all three failure modes are **separately measurable and non-zero** | If forgetting is negligible in your regimes, the CL leg has nothing to fix and the three-way story dies. Learn it at M6, not M10. |
| 4.2 | Pre-register the three LOO outcome interpretations (§3.3) | Prevents post-hoc rescue of a null result |

## Ordering

**1.1 first.** It is five hours, it is the cheapest experiment in the plan, and
it determines whether your only currently-defensible claim is narrow or broad.
Everything else can follow.
