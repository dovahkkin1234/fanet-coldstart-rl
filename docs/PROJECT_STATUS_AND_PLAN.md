# FANET Lifelong-RL Routing — Project Status & Plan

**Last updated:** current session. M1-M3.5 complete and pushed to GitHub
(`origin/main` @ `98a0291`); M4 (GNN + attention architecture) in progress.
**Purpose of this document:** a complete, self-contained account of the
research goal, the milestone plan, and everything built, found, and fixed so
far — including the full debugging history of M3 (six rounds, then two more
to fix the significance test) and M3.5 (five further bugs, caught and
verified in sequence). Written to be read end-to-end, not skimmed, and kept
as the record of the failure-to-success path, not just the current state.

---

## 1. The research goal

**Thesis:** A DRL routing agent deployed in a real FANET faces three distinct
failure modes over its lifetime — it starts stupid (**cold start**), it forgets
when conditions shift (**catastrophic forgetting**), and it re-derives from
scratch situations it has already solved (**no episodic recall**). **Warmstart
+ Continual Learning + CBR/knowledge-base maintenance** address these three
failure modes respectively, and the strongest claim the paper can make is that
**only the combination survives realistic lifelong deployment** — removing any
one degrades performance, validated by a leave-one-out ablation.

**Target venue:** realistic ceiling is IEEE TNSM / TMC / IoT Journal / Ad Hoc
Networks (Elsevier) / IEEE TVT — genuine A-grade journals reachable with a
physically-grounded simulation methodology. True A* systems venues (MobiCom,
INFOCOM) are a stretch without real hardware deployment.

### Full novelty inventory (ranked by defensibility)

**Tier 1 — could carry the paper alone:**
1. CBR-augmented DRL routing triggered by recurring, spatially-localized
   interference signatures (jamming as the concrete test case).
2. Congestion-collapse as an absorbing cold-start failure mode: a badly-
   initialized agent under load creates congestion, which degrades link
   quality, which makes routing harder, which creates more congestion — a
   feedback loop that a single-packet environment (Approach 1) cannot exhibit.
3. Continual learning across physically-distinct mobility regimes, evaluated
   with proper backward/forward-transfer metrics.

**Tier 2 — solid, non-trivial methodology:**
4. Oracle-labeled multi-teacher warmstart: label by *measured achieved
   performance per regime*, not by majority vote — avoids a congestion-blind
   teacher bloc silently winning by headcount under load.
5. GNN-depth-vs-congestion-lookahead ablation (deferred to M4).
6. A physically-grounded (SINR + Bianchi + shadowing), multi-packet,
   congestion-coupled FANET RL benchmark environment — a secondary
   contribution in its own right if open-sourced.

**Tier 3 — necessary but not differentiating alone:**
7. Cold-start quantification in single-packet routing (Approach 1's result).
8. Query-key attention for packet-to-candidate scoring (M4, engineering not
   novelty per se).

**The composability claim** (leave-one-out ablation: warmstart-only vs
+CL vs +CBR vs all-three) is what elevates this from "three techniques
applied to FANET routing" to an actual thesis, and must be an explicit,
scheduled experiment — not an afterthought.

---

## 2. Approach 1 (complete, prior work — context for why Approach 2 exists)

Single-packet DRL routing ablation. Base: `train_dqn_v3_ddqn_per.py`
(Double DQN + Prioritized Experience Replay). Tested reward shaping, Dueling
architecture, and curriculum scenario sampling, individually and combined
(v4a/v4b/v4c), each against a Stage-0 warmstart-vs-scratch baseline, with a
30-seed variance study for statistical rigor.

**Result:** warmstart reliably beats scratch (documented cold-start incidence,
PDR gap, AULC loss), but **none of Dueling, reward shaping, or curriculum
learning improved on warmstart-only** — reward shaping was structurally inert
(link quality saturates near 1.0 with no interference), Dueling had no
headroom (3-6 actions, 3-8 step episodes), curriculum gave a mild scratch-only
cost with no benefit anywhere. Effect sizes throughout were single-digit
percentages on AULC, often inside seed noise.

**Why this isn't enough alone:** queue occupancy is structurally zero (no
concurrent packets to contend), there's no coupling between routing decisions,
and reward shaping is provably inert in this regime. **Approach 2 exists to
fix exactly this** — multi-packet + interference makes congestion, and
therefore the whole warmstart/CL/CBR thesis, actually possible to observe.

---

## 3. Approach 2 milestone map

| # | Milestone | Status | Gate |
|---|---|---|---|
| M1 | Interference-aware link model | DONE | G1 PASSED |
| M2 | Multi-packet interference-coupled simulator | DONE | G2 PASSED |
| M3 | Teacher panel + oracle labeling | **DONE** | **G3 PASSED** |
| M3.5 | Phase-B dataset generation | **DONE** | **G3.5 PASSED + independent audit PASSED** |
| M4 | GNN + attention architecture, supervised pretrain | **IN PROGRESS** | G4 (accuracy vs MLP, rollout PDR) |
| M5 | RL environment + DQN training loop | not started | G5 (agent approaches teachers) |
| M6 | Cold-start study under load (30-seed) | not started | congestion-collapse test |
| M7 | Warmstart vs scratch across load sweep | not started | headline result 1 |
| M8 | Continual learning across mobility regimes | not started | headline result 2 |
| M9 | CBR + jamming regime | not started | headline result 3 |
| M10 | Leave-one-out composability ablation | not started | the thesis-defining result |
| M11 | Ablations (GNN depth, teacher quality), writeup | not started | - |

**Deliberate scope fence (set at M2/M3 design time, still in force):** extra
mobility models (Gauss-Markov, RPGM, Flocking), jamming, CBR, and continual
learning are explicitly **out of scope until M8+**. This is not something
being skipped — it's the plan. Mobility-regime diversity is only needed once
continual learning is being tested; M3/M4 don't need it. **RWP-only through
M1-M4 is by design, not a gap.** See section 7 for the direct answer to
whether this is still the right call.

---

## 4. M1 - Interference-aware link model (DONE, G1 PASSED)

**Objective:** Replace the pure log-distance path-loss model with one where
link quality depends on **network load**, not distance alone. This is the
single premise the entire Approach-2 plan rests on: without load-dependent
link quality, there is no congestion coupling, no regime structure for
continual learning to adapt across, and no novel situations for a CBR
knowledge base to recognize.

**Built:** `link_model_v2.py` -
- **SINR** in place of SNR: `P_signal / (noise + sum of interferers)`.
- **Bianchi (2000) CSMA/CA collision model**: MAC-layer contention probability
  as a function of carrier-sense-range contender count, solved via the
  standard two-equation fixed point.
- **Log-normal shadowing** (sigma = 5 dB), breaking the pure distance-determinism
  of v1.
- Reduces **exactly** to the v1 model at zero interference (verified to
  3.5e-15), which is what makes the later interference on/off ablation
  clean.

**Gate G1** (`preflight_interference_check.py`): sweeps offered load
(`activity` 0 to 0.2) and requires the link_quality distribution to spread and
degrade with load — the exact quantity that was ~0% saturated in Approach 1.

**Result — PASSED decisively**, reproduced byte-for-byte on both the original
laptop and later the new lab machine:

| activity | mean lq | std | frac < 0.90 | mean PER |
|---|---|---|---|---|
| 0.00 | 0.972 | 0.034 | 0.1% | 0.000 |
| 0.20 | 0.375 | 0.382 | 80.2% | 0.655 |

**Bonus finding used later in M3:** dense and sparse topologies fail
*differently* — dense scenarios are **congestion-limited** (near-perfect at
zero load, craters under load), sparse scenarios are **range/partition-
limited** (already degraded at zero load from raw distance, degrades gently
under load since there are few potential interferers). This physical
distinction became the basis for M3's four-band scenario classification.

---

## 5. M2 - Multi-packet interference-coupled simulator (DONE, G2 PASSED)

**Objective:** Replace the single-packet, total-queue-drain simulator with one
supporting genuinely concurrent packets, real bounded queues, and a slotted
channel where transmissions in the same slot interfere via M1's physics —
fixing Approach 1's structural inability to exhibit congestion at all.

**Design spec written first** (`M2_SIMULATOR_V2_DESIGN.md`), locked before any
code, covering:
- **Two-level time**: `FRAME_DT=0.5s` (mobility/topology rebuild) containing
  `SLOT_DT=0.01s` slots (channel/forwarding) — 50 slots per frame.
- **Real bounded queues**: `SERVICE_RATE=1` packet/slot, tail-drop on overflow
  (deleting the old total-drain-every-step behavior that forced queue
  occupancy to ~0 in Approach 1).
- **Load -> activity -> interference feedback loop**: each node's per-slot
  transmission probability derives from its own queue occupancy, so only
  genuinely congested nodes contribute interference to their neighbors — the
  mechanism that makes "bad routing leads to congestion leads to worse link
  quality leads to worse routing" (congestion-collapse) physically possible.
- **Per-packet trajectory tracking**: fixes Approach 1's cross-packet
  credit-assignment risk by keeping each packet's own transition sequence,
  never interleaving transitions from different packets.
- **Drop taxonomy**: `queue_overflow / link_error / ttl_expired / no_route /
  energy_depleted / episode_end`.

**Calibration problems found and fixed during bring-up** (not part of the
original design spec — discovered by testing):
1. **ARQ retransmissions** (`max_retx=5`, locked default). Without this, PER
   was a near-step function of SINR (one hidden-terminal firing = almost
   guaranteed packet death), so interference acted as spatially-uniform random
   loss that no routing policy could avoid — zero measurable actor spread. ARQ
   converts interference into a graded delay/energy cost (real 802.11-style
   MACs retry 4-7 times), restoring the ability for smart routing to actually
   help.
2. **`ACT_BETA = 0`** (locked default). Interference is driven purely by real
   queue backlog (`occupancy`), not by mere packet presence — a node that
   receives and immediately forwards one packet isn't congested and shouldn't
   radiate interference like a genuinely backed-up node. This concentrates
   interference at real hot-spots (the mechanism congestion-aware routing
   needs to exploit) and gives the clean causal story "congestion causes
   interference."
3. **Operating point kept "hard"** (not softened): baseline PDR sits in the
   0.3-0.5 range rather than 0.6-0.8. This was a deliberate call — the harder
   regime is the only one where congestion-collapse and warmstart's advantage
   *growing* with load can actually be observed; a softer regime would hide
   the paper's most novel claims.

**Gate G2** (`preflight_simulator_v2_check.py`) — six checks, **all PASSED**,
reproduced exactly on the new lab machine:

| Check | Result |
|---|---|
| Queues nonzero & load-dependent | mean occ 0.022 to 0.138, peak max 1.00 |
| PDR degrades smoothly with load | 0.514 to 0.127 (rate 0.25 to 4.0), monotone |
| Drop taxonomy shifts to congestion | congestion share 0% to 88.5% |
| Activity in graded band | 0.021 to 0.086 |
| Interference ON vs OFF differ | ON: PDR 0.311, 25 link errors; OFF: PDR 0.375, 0 link errors |
| Bit-reproducible | PDR 0.310714 == 0.310714 |

---

## 6. M3 - Teacher panel + oracle labeling (DONE, G3 PASSED)

### 6.1 Objective

Build a panel of routing algorithms ("teachers") spanning genuinely different
design principles — not just shortest-path variants — and design a labeling
scheme for supervised warmstart pretraining that selects, **per regime**
(scenario class x load bucket), the teacher that **empirically achieves the
highest network PDR** there. This is the **oracle labeling** scheme, and it
exists specifically to avoid a subtle failure mode:

**The correlation-bloc problem.** A naive majority-vote label lets
congestion-*blind* shortest-path-family teachers (Dijkstra, GPSR, AODV,
Stable-Path, Link-Lifetime — all effectively computing near-identical
next-hops) form a correlated voting bloc that can outvote genuinely
congestion-*aware* teachers purely by headcount, exactly under the high-load
conditions where congestion-awareness matters most — and the failure is
invisible, because vote-agreement looks *high* (the bloc agrees with itself).
Labeling by **measured achieved performance in the matching regime** instead
of by vote count is structurally immune to this, because headcount never
enters the label.

### 6.2 Design spec and gate

`M3_TEACHERS_ORACLE_DESIGN.md`, locked before implementation. **Gate G3**
(`preflight_teachers_v2_check.py`) requires, before Phase B (dataset
generation) is allowed to proceed at all:

1. All panel teachers beat a random-neighbor policy at every load.
2. A backpressure-family teacher tops the ranking at **high load, where
   congestion is measurably the bottleneck** (excluding sparse/range-limited
   regimes on principle, since congestion-aware routing has nothing to
   exploit there).
3. The **winner** changes across regimes (not degenerate to one teacher
   everywhere).
4. The winner changes across **scenario class** specifically.
5. Teachers genuinely disagree under load (vote agreement < 0.90) — confirms
   the labeling problem is real, not moot.
6. Bit-reproducible.

**Explicit rule from the design spec, still in force:** *do not generate the
Phase-B dataset before G3 passes* — a dataset built on a broken oracle table
would silently poison every result downstream (M4 onward), the same way a
broken premise would have poisoned M2/M3 if G1/G2 hadn't caught it first.

### 6.3 The full debugging history (six rounds — why this took as long as it did)

This is the part worth reading carefully, because the pattern across all six
rounds is the same: **a plausible-looking number sitting on top of a dead or
broken mechanism**, caught only by testing the mechanism directly rather than
trusting the printed output. This is the identical failure class that nearly
slipped through at M1 (saturated link_quality) and M2 (zero queue occupancy).

**Round 1 — teachers were blind to congestion.**
Initial full run: 4 of 6 teachers returned byte-identical PDR. Root cause:
`_build_graph()` constructed edges with `interference_mw=0.0` unconditionally,
so `packet_error_rate` was exactly zero on every link regardless of load.
ETX-Dijkstra degenerated to plain hop-count Dijkstra; DA-GPSR's quality term
read a constant. Fixed by making the graph reflect *measured* channel state
(what a real node would observe) instead of a static interference-free value.

First fix attempt used **expected interference power** — this collapsed
`link_quality` to ~0.12 everywhere (too pessimistic), because link quality is
*convex* in interference power, so evaluating at the mean badly underestimates
the mean quality (Jensen's inequality). Second, correct fix: model
**P(at least one lethal interferer fires this slot)** — matches G1's own
finding that PER is a near-step function of SINR, so this is the physically
right quantity. Result: physically sane, load-dependent `link_quality`
(0.67 at low load to 0.50 at high load).

**Round 2 — backpressure silently degenerated to GPSR.**
Full grid run failed check 3 (SP-BP won every cell). Investigation found
`backpressure` and `gpsr` returned byte-identical PDR in every cell. Root
cause: the backpressure implementation required a **strictly positive** queue
differential before acting; since measured queue occupancy sits at 0.02-0.14
(per G2), this condition almost never held, so backpressure fell through to
plain greedy-progress routing on ~100% of decisions. A post-hoc diagnostic
(sampled on a frozen graph *after* the episode ended) wrongly reported this as
a "58% fallback rate" — a number completely disconnected from what actually
happened during the real rollout. Fixed by removing the gate entirely
(matching the literature: Tassiulas-Ephremides picks the max-weight link
unconditionally, however small), and rebuilding the diagnostic to count live,
during the actual simulation, via module-level counters.

Also in this round: **ETX-Dijkstra lost to plain Dijkstra in every cell.**
Hypothesis at the time: path-stretching from ETX's `1/(1-PER)` blow-up.
Replaced with LQ-weighted Dijkstra (bounded via an `eps` floor) on that
theory — **this hypothesis turned out to be wrong** (see Round 3).

Also found: `medium_slow`'s expected node degree (7.4) accidentally landed in
the same "dense" classification band as `dense_slow` (9.2), so the scenario
grid only spanned two distinct classes instead of the intended three — check 4
("ranking changes across scenario class") had been passing on thin evidence.
Fixed by widening `medium_slow`'s area (1000 to 1300) and adding a fourth
"very-dense" class band.

**Round 3 — the LQ-Dijkstra fix was itself wrong, and a new bug was
introduced.**
Re-running the fixed diagnostic showed it was *still* reporting 0.000 at
every load — a bug introduced in Round 2's own rewrite: the rewrite of
`backpressure_next_hop` deleted the lines incrementing the module-level
`_BP_STATS` counters the harness reads, so the counters existed but nothing
populated them. Fixed and cross-verified against the simulator's own live
per-run counters so the two paths can't silently diverge again.

More importantly: a three-way head-to-head (LQ-Dijkstra, ETX-Dijkstra, and a
third candidate `arq_etx` — bounded and *physically exact* for the
simulator's actual ARQ mechanism) showed **mean hop count was identical
(1.89-1.95) across every candidate** — nobody was path-stretching. Plain
Dijkstra beat **all three** dynamic-metric variants in 9-12 of 12 cells. The
real mechanism: `link_quality` is genuinely informative (correlation -0.38
with ARQ retry count) but noisy and frame-stale; a **noiseless** objective
(hop count) beats a **weakly-correlated, noisy** one when optimized globally
over a multi-hop path, because estimation error compounds hop-over-hop. The
*same* signal helps when used for a **one-hop greedy** decision (SP-BP,
DA-GPSR already do exactly this), because the estimate is freshest there and
never compounds.

**Decision:** the dynamic-metric panel slot was dropped (6 to 5 teachers:
`dijkstra, gpsr, backpressure, spbp, da_gpsr`), and the negative finding —
"global path optimization on a noisy signal loses to greedy one-hop use of the
same signal under bursty interference" — is reported directly as a
methodology-section contribution rather than papered over.

**Round 4 — regime-dependence was itself mostly noise.**
Re-running G3 with the 5-teacher panel still failed checks 3/4. Closer
inspection: at the 2-3 seed counts then in use, only **3 of 12 cells** had a
winning margin above even a generous 0.02 threshold, and **all three** were
won by SP-BP. Every *other* apparent "winner" (`da_gpsr`, `gpsr`, `dijkstra`
in a few cells) came from cells with margins of 0.000-0.011 — statistically
indistinguishable from noise at that seed count. The "regime-dependence" the
checks had been certifying was largely an artifact of too few seeds.

**Decision:** raise seed count to **30** (matching Approach 1's established
rigor bar), and **parallelize** Phase A via `ProcessPoolExecutor` across the
new lab machine's 16 cores, so a 30-seed x 5-teacher x 4-scenario x 3-rate
grid (1,800 runs) completes in minutes rather than hours.

**Round 5 — the full 30-seed run: SP-BP dominance is real, not noise.**
Ran cleanly, diagnostics matched exactly across the parallel and serial code
paths (`bp_zerodiff` stable at 0.635-0.639 across all three loads — a
genuine, load-independent property of the algorithm). **Result: SP-BP won all
12 cells outright, and 10 of 12 cleared the (then-fixed) 0.02 robust-margin
bar.** This is no longer a seed-count artifact — it's confirmed.

**Mechanistic explanation, fully worked out:** SP-BP is the only panel member
that combines **exact global hop-distance-to-destination** (the same
information Dijkstra has, via BFS) with **local queue-awareness** (the same
information backpressure has). It is a strict information superset of every
other panel teacher, so its dominance is structural, not incidental.
Backpressure's collapse is precisely explained by two compounding
measurements: **~64% zero-queue-gradient rate** (no signal to act on most of
the time, since queues rarely differ) and **~45% loop-override rate** (by far
the highest of any teacher — the mechanical signature of wandering without a
destination bias, getting vetoed by the loop-avoidance rule, and settling for
whatever's left).

**Why checks 3/4 failing here is honest, not a bug to chase further:** the
anti-correlation-bloc protection is validated as working correctly — vote
agreement sits at ~0.68 (well below the 0.90 threshold), meaning the panel
genuinely disagrees at the decision level even though one policy consistently
wins on outcome. What happened is a *third* failure mode the checks weren't
built to distinguish from the first two: one teacher being a genuine,
structural information superset of the others. That's a legitimate empirical
result.

**Also identified:** the fixed `THIN_MARGIN=0.02` heuristic (set back when
only 2-3 seeds existed) was itself methodologically weak — an arbitrary
threshold rather than an actual significance test, now that n=30 provides
real variance estimates.

**Round 6 (current, most recent, NOT YET RUN AT FULL SCALE) — testing
whether SP-BP's dominance is one-sided.**

Per your explicit request: added three more hybrid teachers, each designed to
isolate a different question about *why* SP-BP dominates:

| Teacher | Family | Tests |
|---|---|---|
| **DPP** (Drift-Plus-Penalty / Lyapunov) | Same hybrid-backpressure family as SP-BP, but penalizes retransmission cost (not hop distance), unweighted by link quality, tunable `V=2.0` | Does *any* well-formed backpressure hybrid compete, or specifically SP-BP's formula? |
| **CAR** (Congestion-Aware Routing) | Genuinely different lineage: pure geometric progress steered by a **neighborhood** congestion field (mean occupancy across a candidate's neighbors), no backpressure term at all | Can non-backpressure congestion-awareness compete at all? |
| **Lookahead-SPBP** | SP-BP extended with 2-hop backlog visibility | Does deeper lookahead help (more information) or hurt (staler, second-hand data) — directly previewing the M4 GNN-depth ablation question |

**A bug caught during smoke-testing, before it reached a full run:** DPP
initially collapsed to GPSR's fallback ~85% of the time via a defensive
`score > 0` gate — **the identical mistake already made and fixed once in
backpressure** (Round 2). Since the retransmission-cost penalty is always a
pure cost (never a bonus), and real queue differentials are usually 0 or a
small integer, the gate fired almost every time. Fixed the same way:
no gate, always argmax over all neighbors.

**Also found and cleaned up:** a fully separate, silently-shadowed duplicate
definition of `lq_dijkstra_next_hop` — two functions with the same name in
the file, where only the second was ever actually callable. The dead first
definition's docstring claimed an empirical result (that an *additive*-bounded
LQ weighting beats Dijkstra at every load) that could not be independently
reproduced on a quick check, and appears to be a remnant from before an
earlier session compaction. Quarantined as clearly-marked, unverified dead
code rather than revived, since the dynamic-metric slot question was already
separately settled on solid evidence (Round 3).

**Statistical upgrade:** the fixed `THIN_MARGIN=0.02` heuristic is now
supplemented by a proper **Welch's t-test** (alpha=0.05) computed from each
teacher's real per-cell mean/std/n (n=30 makes this meaningful), used as the
*primary* robustness criterion for checks 2-4. The margin is retained only as
a quick-glance display value alongside the p-value.

**Smoke-tested** at reduced scale (6 seeds, 15s duration, single-threaded) —
no crashes, no hidden collapses across any of the 8 teachers (verified
pairwise), and the Welch's test is doing genuine work: several cells that
*would have* cleared the old fixed-margin heuristic are correctly flagged
**not significant** given the real variance at this small scale. Preliminary,
non-conclusive signal worth noting: `spbp_lookahead` was closely competitive
with `spbp` in several cells; `car` sat consistently below `da_gpsr`; `dpp`
and `backpressure` remained the two weakest teachers throughout.

**This is where things stand.** The code is delivered and believed correct,
but has **not yet been run at the full 30-seed scale with the 8-teacher
panel** — that run is what actually answers the question Round 6 was built
to ask.

### 6.4 How M3 actually resolved

Two more rounds happened after Round 6, both concerning statistics rather than
the panel itself, and they closed M3 out.

**Round 7 — the significance test itself was wrong.** The Round-6 harness used
an *unpaired* Welch's t-test between each cell's #1 and #2 teacher. But every
teacher in a cell is evaluated on the *same* 30 seeds — same topology, same
flows — so per-seed PDR is strongly correlated across teachers (observed
r = 0.89-0.98). Welch discards that pairing and loses most of the statistical
power: at n=30 it reported only 1 of 12 cells as significant, with margins as
large as +0.060 PDR scoring p=0.16. Verified on synthetic data matching the
observed variance structure: the same +0.062 margin scored p=0.085 unpaired
versus p=7x10^-12 paired, and a null control correctly stayed non-significant
(p=0.69) under the paired test — so the fix wasn't loosening the bar, it was
using the correct one. Re-run with a proper paired t-test: **12 of 12 cells
significant**, p <= 0.0007 in every one.

**Round 8 — the checks were reframed, not weakened.** With significance
corrected, the picture was unambiguous: SP-BP wins all 12 cells, confirmed
rather than uncertain. The original checks 3/4 ("does the WINNER change across
regimes") were standing in for two specific failure modes — a degenerate
panel, or the correlated-bloc problem (congestion-blind teachers outvoting
aware ones by headcount). Neither is present (vote agreement ~0.65; oracle
labels come from measured performance, so headcount never enters the label).
Checks 3/4 were rewritten to test what actually gates Phase B directly: is the
oracle's pick statistically justified in enough cells (>=90%; achieved 100%),
and does the panel carry real structure below the #1 slot (>=2 distinct
orderings, runner-up varies by scenario class; achieved 9 orderings, runner-up
flips cleanly between `da_gpsr`/`spbp_lookahead` in congestion-capable
topologies and `dijkstra` in the sparsest one). Both new checks were verified
to genuinely fail on synthetic noise-only and identical-ranking data before
being trusted — not tuned to pass on the real run.

**A caught-and-fixed contradiction, kept as part of the record rather than
smoothed over:** the first post-reframing run's own PASS message still claimed
"regime-dependence holds," directly contradicting the ORACLE LABEL DEGENERACY
note printed one line above it. Fixed to state plainly that the winner is
confirmed constant, not unresolved — the honest framing the new checks were
built to produce.

**Final G3 state, all six checks PASS:**

| Check | Result |
|---|---|
| 1. All teachers beat random at every load | PASS |
| 2. Backpressure family tops congested high load | PASS (3/3 robust congestion-limited cells, all won by SP-BP) |
| 3. Oracle pick statistically justified | PASS (12/12 cells, p <= 0.0007) |
| 4. Panel non-degenerate | PASS (9 distinct orderings; runner-up varies by scenario class) |
| 5. Teachers disagree under load | PASS (vote agreement ~0.65, well under the 0.90 threshold) |
| 6. Reproducible | PASS |

**The finding, stated for the paper:** SP-BP dominates because it is the only
panel member combining exact global hop-distance (Dijkstra's information) with
local queue-awareness (backpressure's information) — a structural information
superset, not an accident. Confirmed after adding three more teachers (`dpp`,
`car`, `spbp_lookahead`) specifically designed to test whether the dominance
was one-sided; none competed. Oracle labeling therefore reduces to
single-teacher imitation of SP-BP in this environment — reported directly as a
limitation, not hidden behind a passing gate.

### 6.5 Git / version-control status (current, superseding the account below)

All of M1 through M3.5 is now committed and pushed to `origin/main`
(`98a0291` as of this writing). The paragraph immediately below is preserved
as it was written mid-M3, before that push happened, because it is itself
part of the record this document exists to keep — including the moment
`git status` showed six rounds of uncommitted M3 work sitting only on disk.

> Reconstructed from the conversation, since this matters for tomorrow:
> GitHub (`origin/main`) had M1 complete and the M2 design spec, but not the
> M2 code itself. Locally, one commit sat ahead of `origin/main` (M2's
> simulator + G2 gate) with a failed push (no credentials available at the
> time). All of M3's work — six rounds of `routing_teachers_v2.py`,
> `teacher_panel.py`, `preflight_teachers_v2_check.py` — had been run locally
> but never `git add`ed. It existed purely as overwritten files on disk.

---

### 6.6 M3.5 - Phase-B dataset generation (DONE, G3.5 PASSED + independent audit PASSED)

### Objective

M3 established *which* teacher to imitate and validated the panel. M3.5 turns
that into actual training data for M4: run SP-BP as the driving policy across
a wide seed/scenario/rate grid, record every routing decision with its oracle
label, and package it as a supervised-learning dataset.

Per the M3 design spec's explicit order-of-work rule — *teachers -> G3 gate ->
dataset only if G3 passes* — this could not start until M3 closed. It also
surfaced three real prerequisites the M4 design doc had not accounted for: no
dataset generator existed anywhere, `simulator_v2._make_obs` was still a
6-field placeholder, and no MLP baseline existed for this environment. All
three are resolved by M3.5's deliverables.

### Design decisions made explicit before coding

**Ragged candidate storage, not a fixed `MAX_NEIGHBORS` cap.** Approach-1 used
`MAX_NEIGHBORS=15`. A pre-generation degree audit measured `very_dense`'s true
maximum at 23 (later 39 in the full run, since the audit sampled only 10 seeds
at the lowest rate) — already above the old cap. A fixed cap would have
silently truncated real candidate lists, and if the truncated candidate were
the labelled one, the row would be either a crash or a silent mislabel.
Candidates are stored as a flat buffer plus an offsets array instead; the
model already handles a variable candidate count via masked attention, so this
costs nothing architecturally.

**Full 8-teacher vote recording by default**, not a subsample. Rejected
subsampling specifically because it is *asymmetric* risk: the dataset is
generated once and trained against repeatedly, and a 10%-vote subsample cannot
be repaired later without full regeneration, whereas computing full votes now
costs a one-off few minutes on 16 cores.

### Bugs found and fixed, in the order they were caught

The pattern repeats from M3: each fix was verified, not just asserted, and
several fixes uncovered the next problem.

1. **Save-path crash.** `np.array(list_of_arrays, dtype=object)` does not
   reliably build an object array when the inner arrays share a leading
   dimension — `edge_index` is always `(2, E)` with varying `E`, so numpy
   tried to build a regular array and raised
   `ValueError: could not broadcast (2,255) into (2,)`. Reproduced in
   isolation before fixing. Fixed by storing frames the same way as
   candidates: flat buffers plus offsets, verified with an exact bit-level
   round-trip on mismatched-shape frames.

2. **22.6% of labels were not the oracle's.** G3.5 check 2 caught this on the
   first full generation run. Root cause: the teacher was restricted to legal
   (unvisited) candidates by deleting visited *nodes* from the graph and
   re-running SP-BP's BFS on the pruned subgraph — which changes SP-BP's
   actual algorithm, since its hop-distance term is a BFS rooted at the
   destination. In sparse topologies (`sparse_fast`, mean degree 1.95),
   removing two or three visited nodes routinely severed `current` from
   `dst` entirely, returning `None`, which fell back to a
   nearest-to-destination heuristic instead of the real oracle. Fixed with
   `spbp_pick_restricted()`: computes hop distances on the *full* graph
   (the quantity G3 actually validated), and restricts only the *choice* to
   legal candidates. Cannot return `None` with non-empty candidates —
   stress-tested at 0 failures in 3000 random graphs. Guarded by
   `assert_no_drift()`, which runs at import and pins the restricted scorer
   against the real `spbp_next_hop` so any future divergence aborts
   generation immediately rather than silently mislabelling data.
   Regenerated: `label_fallback` dropped from 0.2257 to exactly 0.0000.

3. **Oracle-vote self-inconsistency**, caught by a purpose-built independent
   audit (`audit_dataset_v2.py`) rather than by G3.5 — because G3.5 only
   checks what its author thought to check, and a second, adversarial
   checker exists specifically to catch what the first one doesn't. The
   label used `spbp_pick_restricted` (full-graph BFS); `votes['spbp']` used
   the old restricted-view mechanism (BFS on a *modified* graph) — two
   different code paths for the same teacher, which could disagree for
   purely mechanical reasons and bias `vote_agreement` downward. Fixed by
   having the vote reuse the label's own computation, so it agrees by
   construction. Confirmed at exactly 1.0000 after the fix.

4. **A genuine performance bug, caught by reading a Task Manager
   screenshot rather than assuming a long runtime was normal.** 4% overall
   CPU utilization with only brief, scattered single-core spikes is not what
   a compute-bound audit looks like. Root cause: `numpy`'s lazy `.npz`
   loader decompresses an entire member array from the zip archive on
   *every* access, with no caching — and the audit indexed it inside the
   per-decision sample loop. Measured directly: 300 such calls against a
   10.5 MB file took 25.8s; loading the same arrays into memory once and
   slicing took 0.0006s — a ~42,000x difference. Fixed with a `FrameStore`
   that decompresses each array exactly once at startup; verified at
   20,000-sample scale completing in 9.1 seconds against a file a quarter
   the real dataset's size, down from a projected 29+ minutes.

5. **A plain `NameError`** (`buckets` referenced but never assigned) in a
   later-added per-regime analysis section. Caught immediately on the next
   run, fixed with one line, and — the discipline that mattered here — the
   fix was verified with an actual end-to-end re-run against real
   graph-structured test data, not just a recompile, before being shipped.

### Two findings recorded for M4, cross-checked independently of the audit

**The trivial baseline is inflated by free decisions.** G3.5 reported 71.1%
accuracy for "always pick the nearest-to-destination candidate." But 28.8% of
decisions have the destination as a direct neighbour, where every sane rule
(including SP-BP itself, via its sink short-circuit) picks the same thing.
Two independently-computed statistics cross-validate this exactly:
`is_destination` mean among candidates (0.035) times mean candidates per
decision (8.13) predicts 0.2846; the directly-measured "label is the
destination" rate is 0.2881 — agreement to within 0.0035, computed from
entirely different arrays. On the genuinely contested 71.2% of decisions, the
trivial-rule floor is **59.5%, not 71.1%**. M4 must report accuracy on
contested decisions specifically, or the free wins inflate every model
equally and hide the GNN-vs-MLP difference the milestone exists to measure.

**`current_queue_occupancy` is regime-dependent in a way the global mean
hid, and the effect is a clean physical result.** Global mean was 0.004,
initially read as a near-dead feature. Split by scenario, the nonzero rate is
perfectly monotonic in node degree: 0.000% in `very_dense` (degree 18),
0.069% in `dense_slow` (degree 9.2), 0.424% in `medium_slow` (degree 4.4),
3.055% in `sparse_fast` (degree 2.0). Dense networks have enough path
diversity that packets essentially never queue behind each other at a
forwarding node; sparse networks funnel traffic through the few available
relays, so queues build there specifically. The feature is a sparse
congestion-event flag — silent 97-100% of the time, informative when it
fires — kept for M4 rather than dropped.

**One unresolved characterisation question, stated rather than guessed at:**
`very_dense` shows nonzero *network*-mean occupancy (0.0965, ~4.8 packets
queued network-wide) alongside *exactly* zero `current_queue_occupancy` across
all 129,302 of its decisions. Leading hypothesis — packets stuck at nodes with
no valid onward neighbour return early, before a decision row is recorded, so
congested-but-stuck nodes contribute to the network mean without ever
generating a decision — is plausible but unverified. Does not affect dataset
validity (no label, candidate, or feature is incorrect); worth confirming
during M4 feature analysis rather than before.

### Final dataset

533,200 decisions across 48,000 frames. `label_fallback=0.0000`, measured
epsilon=0.0992. G3.5's seven checks and the independent audit's seven checks
both PASS in full — including 0 referential-integrity violations across
20,000 sampled decisions, label re-derivation from raw stored data at 1.0000,
and 0 frames leaking across the train/val/test/generalisation split. Split:
236,531 train / 47,983 val / 57,044 test / 191,642 held out entirely as
`medium_slow` for generalisation (36% of the dataset — a large holdout,
recorded in the manifest with the reasoning: `medium_slow` sits *between*
dense and sparse in node degree, so this tests interpolation to an unseen
density, not extrapolation beyond the training range — a weaker
generalisation claim than holding out an extreme, worth stating plainly
rather than left for a reviewer to notice).

---

## 7. Direct answers to two questions asked mid-M3 (preserved as historical
   record — both are now resolved; see 6.4 and 6.6 for what actually
   happened)

### "Should we move on after implementing DPP/CAR/Lookahead-SPBP?"

**Not quite yet — one specific thing has to happen first, and it's small.**
Run the full 30-seed, 8-teacher G3 grid (the exact command is already in your
hands from the last exchange). That single run is what actually determines:

- Whether checks 2-4 pass for real this time, with proper statistical backing
  (Welch's test, not a guessed threshold).
- Whether the oracle table that will drive Phase-B dataset generation selects
  one dominant teacher everywhere, or genuinely different teachers in
  different regimes.

This is worth doing before moving on precisely because **the RL agent's
warmstart quality is a direct function of the oracle table's quality** — you
said this yourself, and it's correct. Getting this run in before M4 is the
single highest-leverage thing left to do in M3.

**What I want to be clear about, so this doesn't become an open-ended
detour:** this should be the *last* round of teacher expansion, not the start
of an indefinite search for "more regime-dependence." If the 8-teacher run
still shows SP-BP dominant — now backed by three additional, independently-
motivated hybrids rather than the original five — that is itself a complete,
statistically rigorous, and entirely legitimate finding for the paper (section
6.4 already explains the mechanism cleanly). The only thing that would justify
another iteration is a *new bug* surfacing in the run, which — per the last
six rounds — is a fast diagnose-and-fix, not a redesign.

**So, concretely: run the grid once. Whichever way it comes out (absent a new
bug), M3 is done and M4 starts next.**

### "We're RWP-only — should we add other mobility models before moving on?"

**No — RWP-only through M3 (and M4) is the plan, not a gap.** The M2 and M3
design specs both explicitly scoped additional mobility models (Gauss-Markov,
Random Point Group Mobility, Flocking) **out** of these milestones on
purpose. The reasoning, unchanged since it was first laid out: mobility-regime
diversity is only needed once **continual learning across regimes** is being
tested (M8), because that is what CL needs a regime-*shift sequence* for.
M3's job is just to establish that the teacher panel and oracle-labeling
scheme work correctly *within* a single mobility model first — adding
mobility-model diversity now would confound "is the oracle table correct"
with "does the oracle table generalize across mobility models," which are
better tested separately and in that order.

**Bottom line: finish the one remaining G3 run, commit and push everything to
GitHub, then proceed straight to M4 (GNN + attention architecture, supervised
pretraining) on the RWP-only environment — exactly as originally planned.**
The extra mobility models come into the plan at M8, not before.

---

## 8. Hardware and environment status

- **Development machine:** HP Z8 G5 Workstation, Intel Xeon Gold 6426Y
  (16 cores / 32 threads @ 2.5 GHz), 32 GB RAM, NVIDIA RTX A4000 (16 GB),
  Windows 11 Pro. Environment: Miniconda, Python 3.11.15, PyTorch 2.11.0+cu128
  (CUDA 12.8 build, compatible with the driver's CUDA 12.9), confirmed GPU-
  visible (`torch.cuda.is_available() == True`, device name confirmed).
- **Why this machine is sufficient for M1-M7:** the simulator is CPU-bound
  (pure Python/NetworkX), not GPU-bound — the 16-core CPU is what matters for
  the parallelized 30-seed studies, not the GPU. The A4000's 16 GB is also
  double the original laptop's 8 GB, comfortably fitting the M4 GNN+attention
  model.
- **A100 status:** not yet available (arrives ~2 weeks from when this was
  discussed). Not currently a bottleneck. Its actual value kicks in at M8+
  (heavy seed studies, continual-learning sweeps) — and only *after* the
  simulator core is vectorized into tensor operations, which is separate,
  planned work. Developing on the A4000 now and moving the heavy studies to
  the A100 once vectorization is done remains the right sequencing.

---

## 9. M3 / M3.5 completion record, and what's needed before M4 results count

**M3 — all complete:**

- [x] Full 30-seed, 8-teacher, 4-scenario, 3-rate G3 grid run (2,880 runs).
- [x] G3's six checks evaluated on a properly paired significance test
      (corrected from an initially-wrong unpaired Welch's t-test — see 6.4).
- [x] All M1-M3 code and design docs committed and pushed to GitHub.
- [x] The oracle table driving Phase-B generation is the final, paired-test-
      validated one (SP-BP, confirmed dominant across 12/12 cells).

**M3.5 — all complete:**

- [x] Phase-B dataset generated: 533,200 decisions, 48,000 frames.
- [x] G3.5's seven checks pass; a second, independent audit (built
      specifically because every prior gate in this project passed at least
      once with a real defect still present) also passes all seven.
- [x] All M3.5 code, the dataset generator, both gates, and updated design
      docs committed and pushed to GitHub.

**Before M4's results should be trusted, still to do:**

- [ ] Confirm the `very_dense` zero-occupancy-at-decision-time question
      (section 6.6) — doesn't block starting M4, but should be understood
      before drawing conclusions from congestion-feature importance.
- [ ] When reporting M4 accuracy, use the *contested-decision* baseline
      (59.5%), not the raw trivial baseline (71.1%) — the two independently-
      cross-checked numbers in section 6.6 are both correct; the second is
      the one that matters for judging whether a model adds anything.
- [ ] Respect the load-bucket imbalance (high 59% / medium 32% / low 9%) —
      either balance sampling or report per-bucket metrics, or the low-load
      regime is invisible to an unweighted loss.
- [ ] Keep committing at each verified checkpoint, not just at milestone
      boundaries — this section's predecessor existed because six rounds of
      M3 work once sat uncommitted; the habit adopted at the end of M3.5 is
      to push after each fix is verified, not after each milestone completes.

Once M4's own gate (G4: GNN beats MLP, rollout PDR >= 90% of SP-BP,
generalises to the held-out scenario) is passed, M5 — the actual RL training
loop — begins.
