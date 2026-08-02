# Approach 2 — Complete Notes
### From premise to present: what was built, what broke, what it proves, and why it is publishable

---

## Part 0. How to use this document

Three audiences, one text.

**To study from:** read Parts I–VII in order. Each milestone is told as: *what
question it had to answer → what was built → what went wrong → what the
numbers actually say*. The failures are not asides; they are where the
understanding is.

**To defend from:** Part X is a reviewer drill — the twenty questions most
likely to be asked, each with the honest answer and the evidence that backs
it. Do not memorise the answers. Understand the measurements behind them,
because a Tier-1 reviewer will follow up.

**To convince your guide with:** Part IX. It makes the case, and it makes it
honestly, including what is *not* yet proven. An overclaimed pitch that
collapses under one question is worse than a modest one that holds.

One rule throughout: **every number in this document was measured, not
estimated.** Where something is a hypothesis rather than a result, it says so.

---

## Part I. The premise, and why Approach 1 could not carry the paper

### The research question

A deep-RL routing agent deployed in a real FANET faces three distinct failure
modes over its operational lifetime:

1. **Cold start** — it begins stupid, and while it learns, packets die.
2. **Catastrophic forgetting** — conditions shift, and competence earned in
   one regime is destroyed by adaptation to the next.
3. **No episodic recall** — it re-derives from scratch situations it has
   already solved before.

The thesis: **warmstart, continual learning, and case-based reasoning address
these three respectively, and only the combination survives deployment.** The
leave-one-out ablation (warmstart-only vs +CL vs +CBR vs all-three) is the
result that would make this a systems contribution rather than three
techniques bolted together.

### What Approach 1 established, and what it could not

Approach 1 was single-packet DRL routing. One packet, routed to completion,
then the next. It proved cold-start is real and that warmstart removes it —
with a 30-seed variance study, documented incidence, and confidence intervals.

Then it ran an ablation: does adding Dueling architecture, reward shaping, or
curriculum learning improve on warmstart alone? **All three were null.** And
the reasons were structural, not tuning:

- **Reward shaping was inert.** Its shaping term was weighted by
  `link_quality`, and a pre-flight check found `link_quality` saturated near
  1.0 on ~100% of hops. Nothing to shape.
- **Dueling had no headroom.** 3–6 valid actions, 3–8 step episodes. The
  value/advantage decomposition had nothing to exploit.
- **Curriculum cost more than it gave** — a mild scratch-only penalty from
  reallocating a fixed episode budget.

That null result is honest and publishable as a conference ablation. But it
exposed the ceiling: **in a single-packet world, queue occupancy is
structurally zero.** No packet ever contends with another. The `p_q` congestion
penalty in the reward never fired. `link_quality` never varied because nothing
interfered. The environment could not exhibit the phenomenon the thesis is
about.

### Why Approach 2 exists

Congestion is not an add-on. It is the mechanism that makes cold-start
*dangerous* rather than merely slow. The claim Approach 2 is built to test:

> Under realistic load, cold start is not a transient — it is a trap. A badly
> initialised agent creates congestion, congestion degrades link quality,
> degraded links make routing harder, and that produces more congestion. The
> environment becomes adversarial to the agent's own learning.

For that loop to exist, three things had to be built and *validated before
being built upon*: interference physics (M1), a multi-packet simulator (M2),
and a teacher panel good enough to warmstart from (M3).

**The gate discipline.** Each milestone ends in a pass/fail gate (G1, G2, G3,
G3.5) that tests the *premise* of the next milestone. This is the single most
important methodological decision in the project, and Part IX argues it is
itself a contribution.

---

## Part II. M1 — Interference physics (Gate G1)

### The question G1 had to answer

Everything downstream assumes link quality depends on **network load**, not
just distance. If it doesn't, there is no congestion coupling, no regime
structure for continual learning to adapt across, and no novel situations for
a knowledge base to recall. **One week of work could kill four months of
plan.** So it was tested first.

### What was built

`link_model_v2.py` replaces the log-distance path-loss model with three pieces
of physics:

**1. SINR instead of SNR.** Signal-to-Interference-plus-Noise:

```
SINR = P_signal / (N₀ + Σ_k P_interferer,k)
```

Every concurrent transmitter within interference range raises the noise floor
for everyone else. This is what turns *load* into *link degradation*.

**2. Bianchi's CSMA/CA collision model** (Bianchi 2000, IEEE JSAC). MAC-layer
contention: the more stations contending in carrier-sense range, the higher
the collision probability, solved from the standard two-equation fixed point
in (τ, p). This captures loss that pure SINR misses.

**3. Log-normal shadowing** (σ = 5 dB). Breaks the pure
distance→quality determinism, so `packet_error_rate` carries information
beyond geometry — fixing the Approach-1 finding that the feature was inert.

**The backward-compatibility property that makes the later ablation clean:**
at zero interference, zero contention, zero shadowing, the model reduces
**exactly** to the Approach-1 model — verified to 3.5×10⁻¹⁵. That is what
licenses an interference-on/off ablation later: any difference is attributable
to interference, not to a reparameterisation.

### A modelling decision that mattered

The first formulation was too lethal: a single interferer at 200 m dropped
SINR from 31 dB to 2.5 dB — packet dead. If any nearby transmitter annihilates
any link, the network becomes *unroutable* rather than *harder*, and there is
no graded structure to learn around.

The fix was physical, not a fudge: **CSMA carrier-sensing means nearby nodes
defer — they do not transmit concurrently.** The dominant interference comes
from the **hidden-terminal annulus**: nodes outside carrier-sense range (so
they don't defer) but inside interference range. Those are farther, weaker,
and produce *graded* degradation.

This matters conceptually: distributed load keeps local activity low and
survivable; concentrated load craters SINR locally. **That asymmetry is the
congestion-collapse mechanism**, and it fell out of the physics rather than
being imposed.

### G1 result — PASSED

| offered load | mean link_quality | std | frac < 0.90 | mean PER |
|---|---|---|---|---|
| 0.00 | 0.972 | 0.034 | **0.1%** | 0.000 |
| 0.20 | 0.375 | 0.382 | **80.2%** | 0.655 |

The quantity that was 0.1% in Approach 1 — and made reward shaping inert — is
now 80.2%. Reproduced byte-identically on two machines.

**A second finding, used later:** dense and sparse topologies fail
*differently*. Dense is **congestion-limited** (near-perfect at zero load,
craters under load). Sparse is **range/partition-limited** (already degraded
at zero load, degrades gently since there are few interferers). Two physically
distinct regimes — independent support for the continual-learning thesis.

---

## Part III. M2 — The multi-packet simulator (Gate G2)

### Design, locked before coding

`docs/M2_SIMULATOR_V2_DESIGN.md` was written and approved before
implementation. Key decisions:

**Two-level time.** Frames (0.5 s) for mobility and topology; slots (0.01 s)
for the channel — 50 slots per frame. Interference is a slot-level phenomenon;
mobility is a frame-level one. Rebuilding the O(N²) graph every slot would
cost 50× for no physical benefit.

**Real bounded queues.** Approach 1 drained every queue to empty each step —
three lines that forced occupancy to zero. Deleted. Nodes now serve at most
`SERVICE_RATE = 1` packet per slot (one radio), and queues tail-drop at 50.

**The load → activity → interference loop.** A node's per-slot transmission
probability derives from its own queue occupancy:

```
activity(node) = clip(α·occupancy + β·[queue nonempty], 0, A_MAX)
```

Idle nodes contribute no interference; congested nodes contribute most. **This
closes the feedback loop** the whole thesis depends on:

```
bad routing → packets pile up → occupancy ↑ → activity ↑
     ↑                                            ↓
worse options ← link quality craters ← interference ↑
```

**Per-packet trajectories.** Each packet owns its own transition sequence.
Transitions from different packets are never interleaved — this is the
Approach-1 credit-assignment bug fixed and generalised.

### Two calibration problems found during bring-up

Neither was a code bug. Both were physics calibration, and both are worth
understanding because reviewers will ask about them.

**Problem 1: PER was a near-step function of SINR.** One hidden terminal
firing = guaranteed packet death. Interference therefore behaved as
*spatially-uniform random loss* that no routing policy could steer around.
Measured actor spread between good and bad teachers: **~0.008**. Effectively
zero. The environment could not distinguish good routing from bad.

**The fix: ARQ retransmissions.** Real 802.11-style MACs retry (limits of 4–7
are standard). On link error, retransmit up to N times, re-sampling the bursty
interference each attempt, dropping only after all fail. This converts
interference from a **binary packet-killer** into a **graded delay/throughput
cost** — which is exactly the lever a smart router can pull. Actor spread
appeared immediately (up to +0.06 PDR).

*This is the largest discretionary intervention in the environment, and it was
introduced specifically because a result was unusable. Part IX addresses how to
present that honestly; Part VIII notes it is now being swept for sensitivity.*

**Problem 2: interference was too diffuse.** With `β > 0`, any node holding a
packet radiated interference, producing a uniform floor. Setting **β = 0** ties
interference strictly to real backlog, concentrating it at genuinely congested
nodes — creating *avoidable hot-spots* rather than ambient noise. Also the
cleaner causal story: congestion causes interference.

### G2 result — PASSED (all six checks)

| check | result |
|---|---|
| Queues nonzero, load-dependent | mean occ 0.022 → 0.138, peaks at 1.00 |
| PDR degrades smoothly with load | 0.514 → 0.127, monotone, no cliff |
| Drop taxonomy shifts to congestion | congestion share **0% → 88.5%** |
| Activity in G1's graded band | 0.021 → 0.086 |
| Interference ON vs OFF differ | 25 vs **0** link errors |
| Bit-reproducible | PDR **0.310714** identical |

That PDR figure — 0.310714 — became the project's regression constant. Every
subsequent change to the simulator is validated by reproducing it exactly.

---

## Part IV. M3 — The teacher panel (Gate G3)

This milestone took eight rounds. It is also where the methodology proved
itself, so it is worth reading closely.

### Why teachers matter more than they look

Warmstart quality **is** teacher quality. The RL agent is initialised by
imitating these teachers, so any bias in the labels becomes the agent's prior.
Approach 1 used five congestion-*blind* teachers — harmless when there was no
congestion, fatal now that congestion is the dominant failure mode (G2:
congestion-caused drops 0% → 88.5%).

### The correlation-bloc problem

A naive majority-vote label fails in a specific, silent way. Congestion-blind
shortest-path-family teachers (Dijkstra, GPSR, AODV, Stable-Path,
Link-Lifetime) compute *near-identical* next hops — they are all minimising
variants of the same objective. They form a **correlated voting bloc**: five
votes that are not five independent opinions.

Congestion-aware teachers use genuinely different metrics and *disagree with
each other*. So under high load — exactly where congestion-awareness matters —
the blind bloc out-votes the aware minority and the label reverts to
shortest-path. **And the failure is invisible**, because vote-agreement looks
*high* (the bloc agrees with itself).

**The fix: oracle labeling.** Drop voting for the label entirely. Run each
teacher standalone, measure achieved network PDR per regime, and label with
the action of the *empirically best* teacher there. Headcount never enters, so
the scheme is structurally immune. Vote agreement is retained only as a
confidence weight.

### The eight rounds

Each round is a lesson. The pattern is identical throughout: **a
plausible-looking number sitting on top of a dead mechanism.**

**Round 1 — teachers were blind to congestion.** Four of six returned
byte-identical PDR. Cause: `_build_graph` constructed edges with
`interference_mw = 0.0`, so `packet_error_rate` was *exactly zero on every
link*. ETX-Dijkstra degenerated into plain hop-count Dijkstra; DA-GPSR's
quality term read a constant.

**Round 2 — the first fix was mathematically wrong.** Using *expected*
interference power collapsed `link_quality` to 0.12. Reason: link quality is
**convex** in interference, so evaluating at the mean badly underestimates the
mean (Jensen's inequality). The correct quantity — matching G1's finding that
PER is near-binary — is **P(at least one lethal interferer fires)**. Result:
0.67 at low load → 0.50 at high load. Physically sane.

**Round 3 — backpressure had silently become GPSR.** Byte-identical PDR in
every cell. Cause: the implementation required a *strictly positive* queue
differential before acting. Measured occupancy is 0.02–0.14, so
`Q_v − Q_u = 0` almost always, and it fell through to greedy progress on
~100% of decisions. **I had added that gate defensively**, to prevent
backpressure's known low-load wandering — and the safety mechanism ate the
entire algorithm. Tassiulas-Ephremides picks the max-weight link
unconditionally; the throughput-optimality proof depends on it.

**Round 4 — the diagnostic was measuring the wrong thing.** It reported a
"58% fallback rate" from sampling random node pairs on a *frozen graph after
the episode ended* — disconnected from the decisions that actually produced
the PDR. Rebuilt to count live, during the rollout.

**Round 5 — a hypothesis falsified by adding a third candidate.** ETX-Dijkstra
lost to plain Dijkstra everywhere. Hypothesis: path-stretching from ETX's
`1/(1−PER)` blow-up. So it was replaced with a bounded LQ-weighted variant.
Then a three-way head-to-head including `arq_etx` — bounded *and physically
exact* for this simulator's ARQ — showed **mean hop count identical
(1.89–1.95) across all candidates.** Nobody was stretching paths. The
hypothesis was wrong, and plain Dijkstra beat all three.

The real mechanism: `link_quality` is genuinely informative (correlation −0.38
with retry count) but **noisy and frame-stale**. A noiseless objective (hop
count) beats a weakly-correlated noisy one when optimised *globally* over a
path, because estimation error compounds hop-over-hop. The same signal *helps*
in a **one-hop greedy** decision, where it is freshest and never compounds.

**Round 6 — a scenario-classification bug.** `medium_slow` had expected degree
7.4, landing in the same "dense" band as `dense_slow` (9.2). The grid spanned
two classes, not three, and the "ranking changes across scenario class" check
had been passing on thin evidence.

**Round 7 — the wrong statistical test.** Every teacher in a cell runs on the
*same seeds*, and a seed fixes topology *and* flows. A hard seed is hard for
everyone, so per-seed PDR is strongly correlated across teachers (**measured
r = 0.89–0.98**). The comparison is **paired**; an unpaired Welch test
discards the pairing and most of the power. Measured cost: at n=30 it reported
**1 of 12 cells** as robust, with margins as large as +0.060 PDR scoring
p=0.16. Under the correct paired test: **12 of 12**, p ≤ 0.0007.

**Round 8 — DPP repeated Round 3's mistake.** A newly added Drift-Plus-Penalty
teacher collapsed to GPSR 85% of the time via the *same* defensive positivity
gate. Caught in smoke-testing because it produced byte-identical PDR to GPSR.

### G3 result — PASSED

Panel of eight, spanning four decision principles: hop-count (Dijkstra),
geometry (GPSR), queue-differential (Backpressure, SP-BP, DPP), and
congestion-field geometry (DA-GPSR, CAR), plus a 2-hop lookahead variant.

**SP-BP wins all 12 cells.** Paired t-test, Holm-corrected: **12/12
significant**. Vote agreement ~0.65 — the panel genuinely disagrees at the
decision level even though one policy wins on outcome, which confirms the
anti-bloc protection is working and the labeling problem was real.

**Rank statistics across 12 cells:**

| teacher | mean rank | #1 finishes | last-place |
|---|---|---|---|
| spbp | **1.00** | 12 | 0 |
| da_gpsr | 2.42 | 0 | 0 |
| dijkstra | 3.67 | 0 | 0 |
| spbp_lookahead | 3.67 | 0 | 0 |
| gpsr | 4.25 | 0 | 0 |
| car | 6.25 | 0 | 0 |
| backpressure | 7.08 | 0 | 3 |
| dpp | 7.67 | 0 | **9** |

**The regime structure lives in the runner-up slot,** not the winner: in
congestion-capable topologies the #2 is congestion-*aware* (`da_gpsr`); in the
sparsest it is congestion-*blind* (`dijkstra`) — tracking exactly the
congestion-limited vs range-limited split G1 found.

**Deeper lookahead consistently hurts.** `car` (neighbourhood congestion field)
loses to `da_gpsr` (single-hop occupancy) in all 12 cells — the *only*
difference between them is lookahead breadth. `spbp_lookahead` (2-hop) loses
to `spbp` in all 12, and the gap *widens with load* (rank 2 at low load → 5 at
high). Its override rate climbs 0.221 → 0.343: the 2-hop term steers it toward
nodes whose neighbours are empty, pulling it into already-visited regions.

**This is a pre-registered prediction for M4's GNN-depth ablation: shallow
should beat deep in this environment.**

---

## Part V. M3.5 — The Phase-B dataset

### Purpose

Turn the oracle table into training data: run SP-BP as the driving policy
across a wide grid, record every routing decision with its oracle label.

### Two design decisions, both defended

**Ragged candidate storage, not a fixed cap.** Approach 1 used
`MAX_NEIGHBORS = 15`. A degree audit measured `very_dense`'s maximum at 23 —
and the full run then hit **39**. A fixed cap would have truncated real
candidate lists, and if the truncated candidate were the *labelled* one, the
row would either crash or be **silently mislabelled**. Ragged storage removes
that failure mode entirely; the model uses masked attention, so variable
counts cost nothing architecturally.

**Full 8-teacher votes, not a subsample.** Rejected subsampling because the
risk is asymmetric: the dataset is generated once and trained against many
times, and a 10% vote subsample cannot be repaired without full regeneration.
Paying a small certain cost to remove a large contingent one.

### The label-correctness problem (the subtlest issue in the project)

`simulator_v2` applies loop-avoidance **after** the actor picks: an
already-visited choice is replaced by `unvisited[0]` — arbitrary iteration
order. Measured SP-BP override rate: **9–11% of decisions**.

Three wrong ways to label those:
- SP-BP's raw pick → teaches the model to choose *already-visited* nodes
- `unvisited[0]` → teaches an arbitrary iteration-order rule
- drop them → biases the dataset away from exactly the recovery states DAgger
  exists to capture

**The correct answer:** the candidate set *is* the visited-excluded set, and
the label is SP-BP **re-scored on that set**.

The first implementation of that got it wrong too — by deleting visited
*nodes* from the graph and re-running SP-BP's BFS on the pruned subgraph.
That changes the algorithm: SP-BP's hop-distance term is a BFS **rooted at the
destination**, and pruning severs `current` from `dst` in sparse topologies.
**22.6% of labels fell back to a nearest-to-destination heuristic** — a fifth
of the dataset not labelled by the oracle at all. G3.5 caught it.

The fix separates two things that were conflated: **hop distances must be
computed on the full graph** (that is what G3 validated), and **only the
choice may be restricted**. Stress-tested at 0 failures in 3000 random graphs,
and guarded by an import-time assertion that pins the restricted scorer against
the real `spbp_next_hop`.

### Dataset and validation

533,200 decisions across 48,000 frames. `label_fallback = 0.0000`, measured
ε = 0.0992.

**Two independent validators**, because a gate only checks what its author
thought to check. G3.5 checks internal structure; a separate audit
**cross-validates the two files against each other** — rebuilding each frame's
graph from raw stored data and re-deriving labels without trusting the
generator.

The audit found two things G3.5 missed: an oracle-vote inconsistency (label and
`votes['spbp']` computed by different code paths), and a **42,000× performance
bug** — `numpy`'s lazy `.npz` loader decompresses an entire array *on every
access*, and the audit indexed it inside a per-decision loop. That one was
caught by reading a Task Manager screenshot: 4% CPU with scattered single-core
spikes is not what a compute-bound job looks like.

**A finding that matters for M4:** the trivial nearest-to-destination baseline
scores **71.1%** top-1 — but 28.8% of decisions are *free* (destination is a
direct neighbour, everyone agrees). On genuinely contested decisions the floor
is **59.5%**. M4 must report accuracy above that floor, or free wins inflate
every model equally and mask the effect being measured.

Two independently-computed statistics cross-validate this: `is_destination`
mean among candidates (0.035) × mean candidates (8.13) = 0.2846, versus the
directly-measured destination-label rate of 0.2881. Agreement to 0.0035, from
entirely different arrays.

---

## Part VI. The adversarial review

At this point the work was deliberately reviewed as a hostile Tier-1 referee
would — looking for the load-bearing assumption that breaks it. Nineteen
findings, two rated as potentially fatal.

**M-1: god's-eye information with no control-plane cost.** Every teacher
queried a graph with complete, instantaneous, zero-cost global state —
including exact BFS hop-distance from any node to any destination, recomputed
every 0.5 s. No distributed FANET router has that without continuous
link-state flooding. **The project modelled interference from *data* traffic
rigorously and charged nothing for the *control* traffic that would buy the
global knowledge.** As written, SP-BP was presented as a deployable baseline.
It is not.

**M-2: M4's own design reintroduced M-1 into the "novel" architecture.** The
GNN was to be handed that same global hop-distance as an explicit input
feature — so the model gets the hardest signal for free regardless of depth,
and the planned GNN-depth ablation would be confounded.

Plus: no multiple-comparisons correction across 12 tests (~46% family-wise
error), Bianchi applied outside its saturation regime, calibration decisions
chosen because they produced the wanted effect with no sensitivity check, and
an "information superset" explanation for SP-BP that contradicted the project's
own Lookahead-SPBP result.

---

## Part VII. The post-review experiments

### 1. Locality cost — M-1 and M-2, answered with a measurement

Rather than add a disclaimer, the question was made empirical: **how much is
SP-BP's global knowledge actually worth?**

`spbp_khop` restricts SP-BP to a *k*-hop information horizon — exact
hop-distance only within *k* hops, geographic proxy beyond (destination
position from the packet header: GPSR's standard assumption). This maps
one-to-one onto both findings, because **a k-layer GNN propagates information
exactly k hops.**

Control: k=∞ reproduces the real `spbp_next_hop` exactly across 80 random
graphs, so any measured difference is the horizon, not a reimplementation
artifact.

**Result: gap(k=3) = +0.0005 PDR.** Locality is essentially free. M-1 and M-2
resolved.

**But the reason is not what it appears.** `spbp_k1` *won* several cells — and
more information should never hurt. Working out what k=1 computes:

```
h_cur − h_n = [1 + geo_cur/R] − [1 + geo_n/R] = (geo_cur − geo_n)/R
```

**The horizon constant cancels.** At k=1, SP-BP's "hop-distance" term *is*
plain geographic progress with zero topology knowledge — and it matches exact
BFS. Confirmed: geo-hop correlation **r = 0.89–0.93**.

**This falsifies M3's stated explanation.** SP-BP's dominance was attributed to
being an *information superset* combining Dijkstra's global hop-distance with
backpressure's local queue state. If global hop-distance is worth 0.0005 PDR,
that cannot be the mechanism. **The paper was stating the wrong reason for its
most prominent result.**

**Scope limit that must be stated:** nodes are placed uniformly at random in an
open 2D area with no obstacles, where hop ≈ geo/R almost by construction. With
voids or obstacles the proxy breaks — which is exactly why GPSR needs perimeter
routing. The conclusion is **scenario-specific, not general**.

### 2. Mechanism ablation — what SP-BP's advantage is actually made of

| ablation | cost |
|---|---|
| remove queue term entirely | **−0.0615 PDR** |
| queue differential → candidate-only | **0.0000** |
| link-quality multiplicative → additive | 0.0005 |

The exact zero is not coincidence. The two differ by `lq · Q_v`, which vanishes
only if **Q_v = 0** — and the packet is *dequeued before its decision is
recorded*, so the current node's queue is essentially always empty (measured
**exactly 0.000% nonzero** in `very_dense`).

**Therefore backpressure's defining feature — the queue differential — is
inoperative in this environment.** One side of the gradient is always zero.
This is the *same* fact behind classical backpressure's measured 69%
zero-gradient rate. **Three independent measurements, one cause.**

Consequence: G3's check 2 ("backpressure family tops congested high load") was
renamed. SP-BP passes it *while performing no backpressure at all* — accurate
about lineage, misleading about mechanism.

### 3. Queue-weight test — refuting a tuning artifact

If SP-BP's residual edge over DA-GPSR were just a better-scaled queue weight
(`queue_len` 0–50 vs `queue_occupancy` 0–1 — same signal, 50× the weight),
"our best teacher wins on a hyperparameter" would be a much weaker claim.

Tested both directions. **UP: increasing DA-GPSR's queue weight monotonically
destroys it** (0.377 → 0.299 → 0.205 → 0.151 → 0.133). DA-GPSR is already at
or below its optimum. **The tuning-artifact objection is refuted** — the panel
comparison is fair, established rather than assumed.

*Honest note: the asymmetry (SP-BP tolerates a strong queue term, DA-GPSR
doesn't) is real and my scaling explanation does not account for it. At w=15
their queue:progress ratios match, yet DA-GPSR scores 0.205 vs SP-BP's 0.412.
Unresolved, and stated as such.*

### 4. Headroom — the most consequential result so far

Three signals suggested the environment might have little room for routing to
matter: k=1 and k=∞ disagree on 2–13% of decisions with *identical* PDR; the
trivial baseline is 59.5%; and reachability is low in sparse regimes.

So it was measured directly — decomposing every packet's fate into what
routing *can* and *cannot* influence.

**Reachability (the connectivity ceiling):**

| scenario | node degree | reachable pairs |
|---|---|---|
| very_dense | 18.0 | 100% |
| dense_slow | 9.2 | 99.2% |
| medium_slow | 4.4 | **54.7%** |
| sparse_fast | 2.0 | **21.4%** |

In `sparse_fast`, **79% of node pairs cannot communicate at all.** PDR there is
dominated by partition, not routing skill.

**Routing-addressable headroom** (share of *routable* packets lost to causes
routing can plausibly fix — `link_error` deliberately excluded, since a packet
lost to an interference burst on a good link is not a routing mistake):

| load | headroom |
|---|---|
| low | **1.6%** |
| medium | **18.1%** |
| high | **35.6%** |

Pooled: 18.4% — a figure that hides the actual finding.

**At low load there is essentially nothing to win.** At high load, 30–42% is up
for grabs. **And this is exactly what the congestion-collapse thesis
predicts** — routing matters when the network is stressed. That is a strong,
non-obvious confirmation, and it dictates that M4/M5 must weight experiments
toward high load rather than pooling.

---

## Part VIII. Where things stand

### Complete and validated

| milestone | gate | status |
|---|---|---|
| M1 interference physics | G1 | PASSED, reproduced on two machines |
| M2 multi-packet simulator | G2 | PASSED, all six checks |
| M3 teacher panel + oracle labeling | G3 | PASSED, all six, Holm-corrected |
| M3.5 Phase-B dataset | G3.5 + independent audit | PASSED, 7+7 checks |

### Reviewer findings closed

M-1, M-2 (locality quantified), M-3 (superset story falsified, replaced),
M-6/M-18 (check-2 reframed; `very_dense` anomaly explained), M-7 (Holm
correction — 12/12 survive).

**In progress:** M-5, broad calibration sensitivity — sweeping `ACT_BETA`,
`max_retx`, `ACT_MAX`, and interference range, testing whether four qualitative
claims survive. Design is one-at-a-time, with the inability to detect
interactions stated as a limitation rather than glossed.

**Open:** M-4 (Bianchi applied outside its saturation regime — a known-literature
correctness issue), and the MODERATE tier (spatial shadowing correlation,
ε-exploration realism, dataset diversity, panel representativeness, MLP-baseline
fairness, scenario sampling).

### Not yet started

M4 (GNN + attention, supervised pretraining) through M11. **The three headline
claims — warmstart, continual learning, CBR — are unproven.** That is the honest
position and Part IX does not pretend otherwise.

### The bug ledger

Roughly **23 distinct defects** found and fixed, each caught by a gate, an
audit, or an assertion — not by inspection. The recurring signature: *a
plausible number sitting on a dead mechanism.* Several were caught only because
a second, independent checker existed. At least four were in code I had written
to check the first code.

**This ledger is an asset, not an embarrassment.** Part IX explains why.

---

## Part IX. Why this is Tier-1 viable — the case, made honestly

Read this section as two separate arguments, because conflating them is how
pitches collapse under questioning.

### Argument 1: the methodology is already publication-grade

This is true **now**, independent of how the RL results land.

**1. Gate-driven validation is rare and defensible.** Most simulation papers
build the environment and report results. This project validated the *premise*
of each milestone before building on it: G1 tested whether interference makes
link quality load-dependent *before* the simulator was written; G2 tested
whether congestion actually forms *before* teachers were built; G3 tested
whether the panel was sound *before* a dataset was generated; G3.5 and an
independent audit tested the dataset *before* any model touched it.

Every one of those gates caught at least one real defect that would otherwise
have propagated silently into every downstream result. **State this plainly to
a reviewer.** It is the difference between "we believe our simulator is
correct" and "here are the four independent tests that would have failed if it
weren't."

**2. Negative results are reported, not hidden.** Five separate dynamic-metric
teachers lost to plain Dijkstra. Deeper congestion lookahead consistently
*hurts* (two independent confirmations). Approach 1's entire ablation was null.
SP-BP's originally-stated mechanism was falsified — twice — by this project's
own experiments. **A reviewer who sees a paper report its own falsified
hypotheses trusts the positive results more, not less.**

**3. The statistics are unusually careful, including where they were wrong.**
The paired-vs-unpaired correction is the strongest example: measured r=0.89–0.98
between teachers, showing the comparison was paired, and the unpaired test was
costing 11 of 12 significant cells. Then Holm-Bonferroni across the family, with
Benjamini-Hochberg reported for context. Thirty seeds. Pre-registered
predictions with stated failure conditions.

**4. The environment is a contribution in its own right.** A physically
grounded (SINR + Bianchi + log-normal shadowing), multi-packet,
congestion-coupled FANET benchmark with a validated reduction to the
interference-free case and bit-level reproducibility. **Most FANET-RL papers use
toy link models where link quality is a function of distance alone** — this
project's own Approach 1 did, and that is precisely why its reward shaping was
inert. If open-sourced, benchmark papers get cited heavily.

**5. Two methodological findings generalise beyond this paper.** The
correlated-bloc problem in multi-teacher imitation (and oracle labeling as the
fix) applies to any majority-vote imitation setup. The
**routing-addressable headroom decomposition** — separating connectivity ceiling
from physics loss from routing-addressable loss — is a measurement other
routing papers do not perform and arguably should.

### Argument 2: the research claims are novel — but not yet proven

Be precise about this distinction with your guide.

**Tier-1 novelty, if the results land:**

1. **CBR-augmented DRL routing triggered by recurring interference
   signatures.** Case-based reasoning combined with deep RL for network routing
   is a thin literature. "The agent recognises it has seen this jammer signature
   before and retrieves a known-good routing pattern instead of re-learning" has
   not been done in a FANET context.
2. **Congestion-collapse as an absorbing cold-start failure mode.** The claim
   that a badly-initialised agent under load enters a state it cannot escape
   within the training horizon is a genuinely new *empirical* claim. The
   headroom result already supports its precondition: routing leverage rises
   monotonically with load (1.6% → 18.1% → 35.6%).
3. **Continual learning across physically distinct mobility regimes**, with
   proper backward/forward-transfer metrics.

**Tier-2, solid:** oracle-labeled multi-teacher warmstart with the bloc fix;
the GNN-depth-vs-lookahead ablation (with a **pre-registered prediction from
M3 that shallow will win** — a falsifiable claim, not a fishing expedition);
the benchmark environment.

**The composability claim** — *"only warmstart+CL+CBR together survive lifelong
deployment"* — is what elevates this from three techniques to a thesis, and it
requires the leave-one-out ablation as a first-class scheduled experiment.

### The honest risk assessment — say this to your guide

**Realistic venue ceiling: IEEE TNSM / TMC / IoT Journal / Ad Hoc Networks /
TVT.** These are genuine A-grade venues. Pure-simulation work is a stretch at
MobiCom/INFOCOM regardless of quality, because they weight real deployment
heavily. Physically-grounded simulation is what makes simulation-only work
*trusted* at the journal tier — which is exactly what M1 bought.

**Three outcomes, all publishable, at different strengths:**

- **Best case:** warmstart+CL+CBR shows a large composable effect at high load,
  congestion-collapse is demonstrated as an absorbing state. Strong TNSM/TMC
  paper.
- **Middle:** effects are real but modest. Still publishable — the headroom
  measurement lets you report a modest gain *against what was achievable*
  rather than against 1.0, which is a far stronger framing than most papers
  manage.
- **Worst:** RL fails to beat SP-BP meaningfully. You still have a benchmark
  and negative-results paper with unusually rigorous methodology. Lower tier,
  but not a wasted year — and the negative result is *interesting* because it
  would say something real about the limits of learned routing.

**The single biggest risk** is effect size. Approach 1's warmstart-vs-scratch
gap was single-digit percentages, often inside noise. The headroom result says
that at low load there is essentially nothing to win. **Everything depends on
whether the high-load regime, where 30–42% is addressable, produces a large
enough effect.** That is measurable early — M6's cold-start-under-load study —
and it should be measured before investing in M8–M11.

### The line to use with your guide

> "The environment and methodology are already at journal standard — four
> validation gates, ~23 caught defects, corrected statistics, and reported
> negative results. What is not yet proven is the three-part thesis. The next
> milestone tells us whether the effect exists at a size worth writing about,
> and we will know that before committing to the remaining six milestones."

That is a defensible research position. It does not overclaim, and it shows you
know exactly where the risk sits — which is what a guide actually wants to
hear.

---

## Part X. Reviewer drill — twenty questions and honest answers

**Q1. Your baselines assume global network state. Is any of this
implementable?**
No, not as literally implemented — and the paper says so. SP-BP is presented as
a **genie-aided upper bound**, and we quantified what that assumption is worth:
restricting it to a realistic *k*-hop horizon costs +0.0005 PDR, because
geographic distance (available in every packet header, GPSR's standard
assumption) proxies hop distance at r=0.89–0.93 in these topologies. The
idealisation is therefore not load-bearing. Scope limit: that proxy holds in
open 2D placement without obstacles; with voids it would break.

**Q2. Why is your baseline PDR so low? Is your MAC model broken?**
It is deliberate, and decomposed. Raw PDR conflates three things: connectivity
(21–100% of node pairs are reachable depending on scenario), interference
physics, and routing quality. On *routable* packets the best teacher achieves
0.767 at low load in the sparsest scenario. We report routable PDR alongside raw
PDR precisely so the router is not blamed for the topology.

**Q3. You applied Bianchi's model to a non-saturated network.**
Correct, and it is currently an open item. Bianchi (2000) assumes saturation;
our measured node activity is 0.02–0.09. We will either adopt a non-saturated
extension (Malone/Duffy/Leith), justify the approximation at our activity
levels, or state it as a limitation with a sensitivity check. It is on the
punch list, not hidden.

**Q4. You chose calibration constants that produced your effect.**
True, and being tested. `ACT_BETA=0` and ARQ were both adopted because they
produced usable dynamics — ARQ especially, since without it *no* routing policy
could beat any other (actor spread 0.008). We are sweeping all four constants
one-at-a-time and testing whether four qualitative claims survive. OAT cannot
detect interactions; that limitation is stated.

**Q5. Twelve significance tests with no correction?**
Corrected. Holm-Bonferroni (family-wise) is primary, Benjamini-Hochberg
reported alongside. All 12 cells survive: raw 12/12, Holm 12/12, BH 12/12.

**Q6. Why is one teacher winning every cell? Is the panel degenerate?**
No. Vote agreement is 0.65, so the teachers genuinely disagree at the decision
level; and the *runner-up* varies by regime (congestion-aware in dense,
congestion-blind in sparse). We also added three teachers specifically to test
whether the dominance was one-sided; none competed. We report the degeneracy
explicitly: oracle labeling reduces to single-teacher imitation *in this
environment*, and that is stated as a limitation, not hidden behind a passing
gate.

**Q7. Is SP-BP winning because it is better tuned?**
No — tested and refuted. Sweeping DA-GPSR's queue weight upward degrades it
monotonically, so it is already at its optimum. The comparison is fair.

**Q8. Why does SP-BP win, then?**
Its queue term: removing it costs −0.0615 PDR, while changing its *form* costs
0.0000. We initially attributed the win to an information superset; our own
locality experiment falsified that, and we report the correction. The precise
structural reason DA-GPSR's queue term does not work as well remains
unresolved, and we say so.

**Q9. You call it backpressure, but is it?**
No, and we renamed the check. The queue *differential* is inoperative here —
the packet is dequeued before its decision, so `Q_v ≈ 0` and `(Q_v − Q_u)` is
identically `−Q_u`. Replacing the differential with candidate-only queue costs
exactly 0.0000 PDR across all 12 cells. SP-BP passes the check while performing
no backpressure at all. Accurate about lineage, misleading about mechanism —
so we changed the claim.

**Q10. Your dataset is imitation of one teacher. How can the agent exceed it?**
It cannot, via imitation — M4 is explicitly warmstart quality, not SOTA. The
argument for exceeding SP-BP rests on signal categories *no* panel teacher can
use: energy, link lifetime, packet urgency (TTL), multi-flow context, and —
critically — **temporal history**. Every classical teacher here is a
memoryless, fixed-weight function of one graph snapshot. The failures of
Lookahead-SPBP, CAR, ETX and LQ-Dijkstra all trace to the same cause: a fixed
formula cannot down-weight information that has gone stale. A learned model
can. That is also the foundation for CL and CBR, which are explicitly about
memory across time.

**Q11. How do you know the dataset labels are correct?**
Two independent validators. A gate checks internal structure; a separate audit
rebuilds each frame's graph from raw stored data and **re-derives labels
without trusting the generator** — agreement 0.9998 (residual is float32
quantisation). Referential integrity: 0 violations in 20,000 sampled decisions.
An import-time assertion pins the label scorer against the real teacher.

**Q12. Your top-1 accuracy sounds high. Is the task trivial?**
Partly, and we corrected for it. The trivial nearest-to-destination rule scores
71.1% — but 28.8% of decisions are free wins where the destination is a direct
neighbour. On contested decisions the floor is 59.5%, and that is the figure we
report against.

**Q13. Do your scenarios generalise to real FANETs?**
Four scenario classes spanning expected node degree 2.0–18.0, but they are four
*fixed points* in parameter space, not draws from a distribution. Claims are
scoped to these regimes. Only RWP mobility so far; Gauss-Markov, RPGM and
Flocking arrive at M8 where continual learning requires a regime sequence.

**Q14. Why not use PyTorch Geometric?**
Graphs are 20–45 nodes. A dense N×N attention matmul is simpler and faster than
sparse-scatter machinery designed for 10⁴⁺ nodes, and it avoids a real
dependency risk on this platform. Stated as a scalability scope limit.

**Q15. Is your GNN-depth ablation confounded by the explicit hop-distance
feature?**
It was going to be — that was reviewer finding M-2 and we caught it before
building. The locality experiment then showed the feature is a convenience
rather than a crutch (a 3-hop horizon costs +0.0005 PDR), so the ablation
remains valid. We also **pre-registered the prediction that shallow will beat
deep**, based on M3 evidence that deeper lookahead hurts.

**Q16. Your simulator is single-threaded Python. How do you get statistical
power?**
Embarrassing parallelism across seeds. The 30-seed, 8-teacher, 12-cell grid is
2,880 runs in ~10 minutes on 16 cores. Dataset generation is 600 episodes in
127 seconds.

**Q17. How do you know a fix did not break something else?**
A regression constant. G2's PDR is 0.310714, and every simulator change is
validated by reproducing it exactly. It has caught silent behaviour changes
more than once.

**Q18. What is the biggest threat to your conclusions?**
Effect size. Approach 1's warmstart gap was single-digit percentages, often
inside noise, and our headroom measurement shows only 1.6% of routable packets
are routing-addressable at low load. Everything depends on the high-load regime
(35.6% addressable). We measure that at M6, before committing to the remaining
milestones.

**Q19. Your audits were written by the same person who wrote the code.**
True, and stated. They are independent in *code path* — the audit rebuilds
state from disk and re-derives results without calling the generator's
functions — but not in authorship, so a shared blind spot could pass both. The
mitigation is that they use different mechanisms and have in fact caught each
other's bugs repeatedly.

**Q20. Why should we believe the simulator at all?**
Four reasons, each checkable: it reduces exactly to a simpler validated model
at zero interference (3.5×10⁻¹⁵); it is bit-reproducible under fixed seeds; its
physics components are standard and cited (Bianchi 2000; log-normal shadowing);
and every premise it rests on was tested by a gate that could have failed. The
23 defects in the ledger are the evidence that those tests do real work.

---

## Appendix. Key figures, one page

| quantity | value |
|---|---|
| G1: link_quality, load 0 → 0.20 | 0.972 → 0.375 |
| G1: fraction below 0.90 | 0.1% → **80.2%** |
| G1: reduction to interference-free model | exact to 3.5×10⁻¹⁵ |
| G2: regression constant (PDR) | **0.310714** |
| G2: congestion-caused drop share | 0% → **88.5%** |
| G3: SP-BP cell wins | **12 / 12** |
| G3: paired-t correlation between teachers | r = 0.89–0.98 |
| G3: cells surviving Holm correction | **12 / 12** |
| G3: vote agreement (panel disagreement) | ~0.65 |
| M3.5: dataset size | 533,200 decisions / 48,000 frames |
| M3.5: label fallback rate | **0.0000** |
| M3.5: trivial baseline (raw / contested) | 71.1% / **59.5%** |
| M3.5: max candidate degree (vs old cap 15) | **39** |
| Locality: gap at 3-hop horizon | **+0.0005 PDR** |
| Locality: geo-hop correlation | r = 0.89–0.93 |
| Mechanism: cost of removing queue term | **−0.0615 PDR** |
| Mechanism: cost of changing its form | **0.0000** |
| Reachability (very_dense → sparse_fast) | 100% → **21.4%** |
| Headroom by load (low / med / high) | **1.6% / 18.1% / 35.6%** |
| Defects found and fixed | **~23** |
