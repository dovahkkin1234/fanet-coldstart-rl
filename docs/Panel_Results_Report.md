# Oracle Panel Results & the Revised Understanding of Congestion-Awareness

**FANET Lifelong-RL Cold-Start Routing** · Shirish Giroti (CS23B2041) · Guide: Dr. Preeth Raguraman
**Scope:** the 8-teacher generalized panel and every experiment that followed it — contender panel, link-metric extension, and oracle-agreement probe.
**Status of the numbers:** every figure below is taken directly from the saved result files, not re-typed from memory.

---

## 1. Executive Summary

The project began from the premise that **SP-BP** — shortest-path routing combined with a backpressure-style queue term — was the correct routing oracle, and that congestion-awareness was the property that made it strong. Four panel experiments, totalling **1,769 simulated 1,000-second episodes**, overturned that premise and replaced it with something more precise.

**What we now know, stated carefully:**

1. **SP-BP is not the best teacher anywhere it can be measured.** It is displaced in every one of the nine oracle-setting cells.
2. **No single routing policy is best across all regimes.** The winner depends on network density: shortest-path wins in sparser swarms, geographic routing in denser ones.
3. **Congestion-awareness is neither simply helpful nor simply harmful. Its effect depends on *how* the congestion signal is formulated and on *how dense* the network is.** Three formulation effects are established with 9-of-9 consistency; one density effect is established with a clear but non-universal pattern.
4. **Link-quality information is also context-dependent.** Used as a path weight in shortest-path routing it is harmful in every cell; used as one term in a local geographic score, it is part of a combination that helps.
5. **In this simulator, congestion shows up mainly as degraded links rather than overflowing queues** — link errors outnumber queue drops by 1.3× to 6× in every measured cell. This is the most important single fact for interpreting everything else.

The honest bottom line: **the project did not lose its congestion-awareness claim. It traded a vague claim ("congestion-awareness helps") for a sharper, better-evidenced one ("the value of a congestion signal depends on its formulation and on network density, and the standard backpressure formulation is the wrong one in sparse-to-moderate FANETs").**

> **A note on this report's honesty.** During this analysis I made and later retracted three interpretive claims. They are documented in §8 rather than silently removed, because a reader deserves to see which conclusions survived scrutiny and which did not.

---

## 2. Experimental Setup

### 2.1 Operating point

All panels ran at the operating point established by the earlier band-search and energy work:

| parameter | value | why |
|---|---|---|
| Episode duration | 1,000 s | matches published comparators; long enough for energy dynamics |
| Initial battery | 8,000 | smallest value at which long episodes and real congestion coexist |
| Altitude band | 100–300 m | matches published comparators |
| Seeds per cell | 10, paired | every teacher faces byte-identical traffic |

### 2.2 The cells

Each scenario was tested only at rates where the band-search had confirmed genuine, non-saturated congestion.

| scenario | nodes | rates tested | role |
|---|---|---|---|
| `medium_slow` | 30 | 40, 60, 80 | oracle-setting |
| `dense_slow` | 30 | 60, 80, 100 | oracle-setting |
| `very_dense` | 45 | 60, 80 | oracle-setting |
| `sink_50` | 50 | 30 | oracle-setting (convergecast) |
| `sparse_fast` | 20 | 80, 100 | **flagged** — reported only |

`sparse_fast` is flagged because its usable cells sit against the elasticity ceiling (0.80 and 0.85, versus 0.05–0.47 everywhere else) and its losses are neither link- nor queue-driven. It is measured and reported, but never used to choose an oracle.

### 2.3 The teachers, and what each actually computes

This table was built by **reading each teacher's source code**, not by inferring from its name. That distinction turned out to matter: two earlier conclusions in this analysis were wrong precisely because they were drawn from names and comments rather than code.

| teacher | base family | congestion signal | link quality | score (simplified) |
|---|---|---|---|---|
| `dijkstra` | shortest path | none | none | minimum hop count |
| `gpsr` | geographic | none | none | neighbour closest to destination |
| `da_gpsr` | geographic | **absolute**, candidate's own queue | yes, additive term | `progress − queue(u) + 0.5·lq` |
| `car` | geographic | **absolute**, candidate's *neighbourhood* average | yes, additive term | `progress − field(u) + 0.5·lq` |
| `spbp` | shortest path | **differential**, `Q_self − Q_u` | yes, multiplier | `lq · ((Q_self − Q_u) + v·hop_progress)` |
| `backpressure` | **none** | differential | yes, multiplier | `lq · (Q_self − Q_u)` |
| `dpp` | **none** | differential | via expected attempts | `(Q_self − Q_u) − V·(attempts − 1)` |
| `spbp_lookahead` | shortest path | differential | yes | SP-BP with lookahead |
| `lq_dijkstra` | shortest path | none | yes, **bounded** path weight | shortest path, `w = 1/max(lq, ε)` |
| `etx_dijkstra` | shortest path | none | yes, **unbounded** path weight | shortest path, `w = 1/max(1−PER, ε)` |

Plus three SP-BP ablations used as controlled probes: `spbp_ab_noqueue` (queue term removed), `spbp_ab_candqueue` (differential replaced by candidate-absolute), and `spbp_ab_additive`.

**Two structural facts in this table are essential:**

- `backpressure` and `dpp` have **no destination term at all**. They choose a neighbour purely by queue difference, so in a lightly loaded network they have no idea which way the destination lies. Their source code documents this as expected: *"an empty network gives all-zero weights and the argmax degenerates to the first neighbour — effectively a random walk."*
- `car` and `da_gpsr` use **identical weights and identical structure**. They differ in exactly one respect: `da_gpsr` penalises the candidate's *own* queue, `car` penalises the *average queue of the candidate's neighbours*. That makes them a clean controlled comparison.

### 2.4 Statistical method

Every comparison is **paired by seed**, so competing teachers face identical traffic and traffic variance drops out of the contrast. Multiplicity is controlled with **Holm–Bonferroni** correction across the whole panel.

The oracle for each cell is chosen by a **leading-group rule**: every teacher whose mean PDR lies within **1 percentage point** of the top scorer is treated as tied, and the **simplest** member of that group is chosen (parsimony). The history of this rule, including a bug in its first version, is in §3.2.

---

## 3. Results, Experiment by Experiment

### 3.1 Experiment 1 — The 8-teacher generalized panel

**Design:** 9 teachers (the 8 originals plus `spbp_ab_candqueue`), 11 cells, 10 seeds = **990 episodes, ~20.4 hours**.

**Mean PDR by cell:**

| cell | dijkstra | gpsr | backpr. | spbp | da_gpsr | dpp | car | spbp_la | candq |
|---|---|---|---|---|---|---|---|---|---|
| medium_slow@40 | **0.5458** | 0.5050 | 0.3125 | 0.5109 | 0.5115 | 0.3029 | 0.4877 | 0.4928 | 0.5281 |
| medium_slow@60 | **0.4057** | 0.3666 | 0.2162 | 0.3570 | 0.3655 | 0.2083 | 0.3379 | 0.3337 | 0.3839 |
| medium_slow@80 | **0.3156** | 0.2811 | 0.1643 | 0.2708 | 0.2750 | 0.1570 | 0.2489 | 0.2456 | 0.2965 |
| dense_slow@60 | 0.5947 | 0.6187 | 0.3903 | 0.5946 | **0.6337** | 0.3679 | 0.5879 | 0.5597 | 0.6209 |
| dense_slow@80 | 0.4733 | 0.4986 | 0.3040 | 0.4621 | **0.5062** | 0.2824 | 0.4599 | 0.4193 | 0.4941 |
| dense_slow@100 | 0.3940 | **0.4205** | 0.2520 | 0.3713 | 0.4156 | 0.2284 | 0.3753 | 0.3267 | 0.4046 |
| very_dense@60 | 0.5140 | 0.5489 | 0.3154 | 0.5483 | **0.5614** | 0.2864 | 0.5386 | 0.5379 | 0.5549 |
| very_dense@80 | 0.3966 | 0.4294 | 0.2403 | 0.4135 | **0.4458** | 0.2131 | 0.4146 | 0.3893 | 0.4277 |
| sink_50@30 | 0.6724 | 0.6848 | 0.4708 | 0.7038 | **0.7164** | 0.4265 | 0.6842 | 0.6832 | 0.7095 |
| sparse_fast@80 ⚑ | **0.1791** | 0.1346 | 0.1150 | 0.1586 | 0.1379 | 0.1312 | 0.1427 | 0.1478 | 0.1700 |
| sparse_fast@100 ⚑ | **0.1682** | 0.1142 | 0.0905 | 0.1464 | 0.1209 | 0.0830 | 0.1131 | 0.1327 | 0.1561 |

**Observations:**

- **SP-BP was displaced in 7 of 11 cells** under the incumbent-replacement rule, overturning the M3 oracle decision.
- **`backpressure` and `dpp` lost in every cell, by 10–28 percentage points.** As §2.3 explains, this reflects their lack of any destination term, not a verdict on congestion-awareness.
- **`spbp_ab_candqueue` beat `spbp` in every cell** yet **never won a cell outright** — in every case a simpler teacher beat it.
- **A regime split was already visible:** `dijkstra` led every `medium_slow` cell; geographic teachers led the denser ones.

**Two problems with this run,** both since corrected: `spbp_ab_noqueue` was never included, so the most plausible competitor never got a head-to-head; and only per-cell means were saved, not per-seed values, which made the result impossible to extend without re-running it.

### 3.2 Experiment 2 — The contender panel

**Design:** the 5 strongest contenders (`dijkstra`, `gpsr`, `da_gpsr`, `spbp_ab_noqueue`, `spbp`), 11 cells, 10 seeds = **550 episodes, ~10 hours**. Four teachers were dropped because they lost in every cell in Experiment 1. **Per-seed PDRs were stored for the first time.**

**Question it answered:** is SP-BP's problem its queue term, or its whole approach?

- *H1:* remove the queue term and SP-BP becomes competitive.
- *H2:* even without its queue term, SP-BP loses to simpler teachers.

**Result: H2.** `spbp_ab_noqueue` was recommended in **0 of 9** oracle cells. But the detail is revealing: it was the **top scorer in all three `medium_slow` cells**, beating `dijkstra` by only 0.10–0.17 pp. Stripped of its queue term, SP-BP converges to shortest-path behaviour and becomes statistically indistinguishable from it.

**A decision-rule bug, found and fixed.** The first version of the leading-group rule excluded a teacher only if it was *both* significantly worse *and* at least 1 pp worse. That sounds sensible but is wrong: with only 10 seeds, a teacher that is genuinely 4 pp worse often fails to reach significance, so it stays in the "tied" group — and parsimony then picks it for being simplest. Four of nine cells were affected:

| cell | wrongly picked | actually worse by |
|---|---|---|
| dense_slow@60 | dijkstra | −3.90 pp vs da_gpsr |
| dense_slow@80 | dijkstra | −3.29 pp vs da_gpsr |
| dense_slow@100 | dijkstra | −2.65 pp vs gpsr |
| sink_50@30 | dijkstra | −4.40 pp vs da_gpsr |

**"Not significantly worse" at n = 10 does not mean "equally good."** The corrected rule decides membership by effect size — within 1 pp of the top — and treats significance as confirmation rather than as the gate. Because per-seed data had just been stored, this was fixed by recomputation, with no re-run.

### 3.3 Experiment 3 — The link-metric extension

**Design:** the two link-metric teachers that had never been tested, `lq_dijkstra` and `etx_dijkstra`, **merged** into the stored contender data. 220 new episodes, ~4.2 hours, saving 550 episodes of re-running.

Before merging, an **equivalence control** re-simulated one already-measured teacher at one cell and required it to reproduce the stored per-seed values exactly. It matched to ten decimal places, confirming the merge combined data from one consistent simulator state rather than two.

**Why these two mattered:** `lq_dijkstra` is shortest-path routing weighted purely by link quality, with no queue term. Since link errors dominate losses, it directly tests whether link quality — rather than queue information — is the active ingredient in `da_gpsr`'s success. Its own source code states it was written to replace ETX in the panel; it was then never added. `etx_dijkstra` is the same idea with an unbounded metric, forming a controlled pair.

**Result: both lost in every one of the 11 cells.**

| cell | dijkstra | lq_dijkstra | Δ | etx_dijkstra | Δ |
|---|---|---|---|---|---|
| medium_slow@40 | 0.5458 | 0.5380 | −0.78 | 0.5292 | −1.66 |
| medium_slow@60 | 0.4057 | 0.3977 | −0.81 | 0.3923 | −1.34 |
| medium_slow@80 | 0.3156 | 0.3089 | −0.67 | 0.3039 | −1.17 |
| dense_slow@60 | 0.5947 | 0.5868 | −0.78 | 0.5685 | −2.61 |
| dense_slow@80 | 0.4733 | 0.4628 | −1.05 | 0.4511 | −2.22 |
| dense_slow@100 | 0.3940 | 0.3857 | −0.83 | 0.3793 | −1.47 |
| very_dense@60 | 0.5140 | 0.5051 | −0.89 | 0.4855 | −2.85 |
| very_dense@80 | 0.3966 | 0.3832 | −1.34 | 0.3699 | −2.67 |
| sink_50@30 | 0.6724 | 0.6312 | −4.12 | 0.6145 | −5.79 |

The ordering is **monotone in every cell**: plain hop count beats bounded link-quality weighting, which beats unbounded ETX weighting. The more heavily paths are weighted by link quality, the worse the result.

### 3.4 Experiment 4 — The oracle-agreement probe

**Design:** 9 episodes. In each cell the assigned oracle drove the simulation, but **at every single routing decision all three candidate oracles were queried on the identical state**. Up to 20,000 decisions logged per cell.

**Why:** the corrected oracle assignment uses three different teachers across nine cells. Behaviour cloning trains one network on all of them. If two cells present similar observations but their oracles choose different next hops, the network receives contradictory supervision — a silent failure that would only show up as a higher training-loss floor.

| cell | assigned oracle | 3-way agree | dij–gpsr | dij–da_gpsr | gpsr–da_gpsr | mean degree |
|---|---|---|---|---|---|---|
| medium_slow@40 | dijkstra | 0.652 | 0.802 | 0.664 | 0.830 | 5.3 |
| medium_slow@60 | dijkstra | 0.592 | 0.789 | 0.606 | 0.770 | 4.8 |
| medium_slow@80 | dijkstra | 0.544 | 0.756 | 0.562 | 0.746 | 4.5 |
| dense_slow@60 | da_gpsr | 0.747 | 0.775 | 0.757 | 0.947 | 10.7 |
| dense_slow@80 | gpsr | 0.695 | 0.792 | 0.707 | 0.866 | 10.0 |
| dense_slow@100 | gpsr | 0.649 | 0.801 | 0.676 | 0.796 | 9.5 |
| very_dense@60 | da_gpsr | 0.647 | 0.680 | 0.660 | 0.918 | 19.9 |
| very_dense@80 | da_gpsr | 0.593 | 0.648 | 0.622 | 0.855 | 18.6 |
| sink_50@30 | da_gpsr | 0.723 | 0.738 | 0.732 | 0.965 | 12.9 |

**Overall three-way agreement: 0.649** — the oracles disagree on roughly one decision in three.

---

## 4. Controlled Comparisons: Isolating Each Mechanism

Comparing whole teachers mixes several differences at once. The most informative evidence comes from pairs of teachers that **differ in exactly one respect**. Each row below isolates one mechanism across the nine oracle-setting cells.

| # | comparison | isolates | positive | mean |
|---|---|---|---|---|
| **A** | `da_gpsr` − `gpsr` | adding queue + link-quality terms to a geographic base | 6 / 9 | +0.86 pp |
| **B** | `spbp` − `spbp_ab_noqueue` | adding a queue **differential** to a shortest-path base | **3 / 9** | −1.33 pp |
| **C** | `spbp_ab_candqueue` − `spbp` | **absolute** vs **differential** queue signal | **9 / 9** | +2.09 pp |
| **D** | `lq_dijkstra` − `dijkstra` | link-quality **path weighting** on shortest path | **0 / 9** | −1.25 pp |
| **E** | `etx_dijkstra` − `lq_dijkstra` | unbounded vs bounded link metric | **0 / 9** | −1.17 pp |
| **F** | `car` − `da_gpsr` | neighbourhood-average vs candidate's-own queue | **0 / 9** | −3.29 pp |
| **G** | `spbp_ab_noqueue` − `dijkstra` | SP-BP minus its queue term, vs plain shortest path | 8 / 9 | +0.45 pp |

**Per-cell detail** (cells in order: medium_slow 40/60/80, dense_slow 60/80/100, very_dense 60/80, sink_50 30):

```
A  da_gpsr − gpsr        +0.65  −0.11  −0.62 | +1.49  +0.76  −0.49 | +1.25  +1.65 | +3.16
B  spbp − noqueue        −3.65  −5.01  −4.59 | −1.26  −2.00  −3.60 | +2.39  +1.19 | +4.55
C  candqueue − spbp      +1.72  +2.69  +2.58 | +2.63  +3.20  +3.33 | +0.66  +1.42 | +0.56
D  lq_dijk − dijk        −0.78  −0.81  −0.67 | −0.78  −1.05  −0.83 | −0.89  −1.34 | −4.12
E  etx − lq_dijk         −0.88  −0.54  −0.50 | −1.83  −1.17  −0.64 | −1.96  −1.32 | −1.68
F  car − da_gpsr         −2.38  −2.76  −2.60 | −4.57  −4.63  −4.03 | −2.27  −3.12 | −3.22
G  noqueue − dijkstra    +0.17  +0.13  +0.10 | +1.25  +0.88  +1.32 | +1.05  +0.51 | −1.41
```

**Four of these are unanimous (9/9 or 0/9) — the strongest evidence in the entire project.** They are C, D, E and F.

---

## 5. How Our Understanding of Congestion-Awareness Changed

### 5.1 Where we started

> *"SP-BP is the right oracle, and congestion-awareness is what makes it strong."*

This was the founding premise, inherited from the original M3 panel. That panel had been run on the simulator before the delivery-queue leak was fixed, at an operating point that was later shown not to produce genuine congestion. It was never a valid test of congestion-awareness.

### 5.2 Where we are now — four established findings

**Finding 1 — Formulation matters more than presence. (C: 9/9)**
Replacing SP-BP's queue *differential* (`Q_self − Q_u`) with an *absolute* penalty on the candidate's own queue improves delivery in **every one of nine cells**, by +2.09 pp on average. This is the cleanest result in the project, and it holds even in the cells where the differential itself helps (see Finding 3).

*Plausible reason:* a differential is a gradient — "move toward emptier neighbours." At the measured queue occupancies of 2–20%, most adjacent nodes have nearly equal queues, so the gradient is small and noisy, while the detours it induces are real and costly. An absolute penalty acts as a filter — "avoid clearly full neighbours" — and only activates when it matters.

**Finding 2 — The congestion signal must be local to the candidate. (F: 0/9)**
`car` and `da_gpsr` are identical except that `car` penalises the *average queue of the candidate's neighbours*, while `da_gpsr` penalises the *candidate's own queue*. The candidate's own queue wins in **every cell**, by 2.3–4.6 pp. Averaging congestion over a neighbourhood dilutes precisely the information that matters for choosing the next hop.

**Finding 3 — Whether congestion-awareness helps depends on network density. (B: 3/9)**
This is the most important correction in this report. Adding SP-BP's queue differential to a shortest-path base:

| scenario | mean degree | effect of queue differential |
|---|---|---|
| `medium_slow` | 4.9 | **hurts:** −3.65, −5.01, −4.59 pp |
| `dense_slow` | 10.1 | **hurts:** −1.26, −2.00, −3.60 pp |
| `very_dense` | 19.2 | **helps:** +2.39, +1.19 pp |
| `sink_50` | 12.9 | **helps:** +4.55 pp |

Across Suite A the effect runs monotonically with density — harmful when nodes have few neighbours, helpful when they have many. The same density dependence appears in comparison A, where adding queue and link-quality terms to a geographic base is roughly neutral in `medium_slow` but helps in `very_dense` and most in `sink_50`.

*Working hypothesis, not yet demonstrated:* congestion-awareness works by steering traffic onto alternative paths. In a dense network those alternatives exist and are cheap, so spreading load pays off. In a sparse network the alternatives are few, long, or absent, so the detours cost more than they save. `sink_50` fits unusually well because convergecast concentrates all traffic toward one node, which makes load-spreading near the sink especially valuable. **This mechanism fits the data but has not been tested directly** — see §9.

**Finding 4 — The base routing family matters more than any added signal.**
Across Experiments 2 and 3, every attempt to improve shortest-path routing by adding a signal — a queue differential, link-quality weighting, or ETX weighting — made it worse in most cells. Meanwhile `da_gpsr`, which carries both a queue and a link-quality term on a *geographic* base, wins four cells. What separates winners from losers is not so much which signals are used as which base family they are attached to, and how.

### 5.3 The revised claim

| before | after |
|---|---|
| Congestion-awareness helps routing. | The value of a congestion signal depends on its **formulation** and on **network density**. |
| SP-BP is the oracle. | No single policy is best; the best teacher is **regime-dependent**. |
| (implicit) Backpressure is a sound basis. | The backpressure-style **differential** is the wrong formulation for sparse-to-moderate FANETs; an **absolute, candidate-local** signal is consistently better. |

This is a **sharper** claim than the one the project started with, and it is directly relevant to the comparator literature, much of which builds on backpressure-style formulations.

---

## 6. Link Quality and Its Interaction with Congestion

### 6.1 The most important fact: congestion degrades links more than it fills queues

From the band-search at the final operating point, the share of generated packets lost to each cause:

| cell | queue overflow | link error | link ÷ queue |
|---|---|---|---|
| medium_slow@40 | 0.081 | 0.117 | 1.4× |
| medium_slow@60 | 0.141 | 0.215 | 1.5× |
| medium_slow@80 | 0.198 | 0.258 | 1.3× |
| dense_slow@60 | 0.099 | 0.252 | 2.5× |
| dense_slow@80 | 0.158 | 0.322 | 2.0× |
| dense_slow@100 | 0.192 | 0.362 | 1.9× |
| very_dense@60 | 0.069 | 0.416 | 6.0× |
| very_dense@80 | 0.117 | 0.490 | 4.2× |

**Link errors exceed queue overflows in every cell.** And they are not independent of congestion: the validated interference model (gate G1) couples link quality to transmission activity — mean link quality falls from 0.97 to 0.35 as neighbouring activity rises. **In this simulator, congestion manifests primarily as interference-degraded links, and only secondarily as full queues.**

This reframes the whole problem. A routing policy that watches only queues is watching the *smaller* of the two ways congestion causes loss.

### 6.2 How link quality behaves as a routing signal

Link quality turns out to be exactly as context-dependent as queue information:

| how link quality is used | example | result |
|---|---|---|
| as a **path weight** in shortest-path routing | `lq_dijkstra` | **harmful in 9/9 cells** (D) |
| as a **stronger, unbounded** path weight | `etx_dijkstra` | **worse still, 9/9** (E) |
| as a **multiplier** on hop progress | `spbp_ab_noqueue` | slightly helpful in dense cells (G) |
| as an **additive term** in a local geographic score | `da_gpsr` | part of a combination that helps in 6/9 (A) |

The same information helps or hurts depending on where it enters the decision.

*Why path-weighting may fail:* routing decisions use link quality as observed *at decision time*, but a packet experiences link quality *at transmission time*. With drones moving and interference fluctuating as neighbouring nodes switch on and off, the two may differ. Weighting whole paths by a momentary snapshot distorts route choice away from the shortest path without reliably buying better links. The monotone ordering — unbounded ETX worst, bounded link quality next, no weighting best — is consistent with this: the more strongly the snapshot is trusted, the worse the outcome. **This is a hypothesis;** measuring how well decision-time link quality predicts transmission-time outcome would test it (§9).

### 6.3 How the two signals interact

The winning teacher in dense regimes, `da_gpsr`, combines **three** terms: geographic progress, the candidate's own queue, and link quality. Neither signal succeeds alone on a shortest-path base — queue differentials hurt in most cells (B), link-quality weighting hurts in every cell (D). They succeed together, locally, on a geographic base, in dense networks (A).

The honest reading is that **the combination does the work, and no single ingredient can be credited on its own**. Experiment 3 was designed to test whether link quality alone could explain `da_gpsr`'s advantage; it could not.

---

## 7. What Else These Results Affect

**The oracle assignment.** Final, after the corrected decision rule and the link-metric extension:

| cells | oracle |
|---|---|
| medium_slow @ 40, 60, 80 | `dijkstra` |
| dense_slow @ 60 | `da_gpsr` |
| dense_slow @ 80, 100 | `gpsr` (tied with `da_gpsr`) |
| very_dense @ 60, 80 | `da_gpsr` |
| sink_50 @ 30 | `da_gpsr` |
| sparse_fast @ 80, 100 | flagged — no oracle set (best: `dijkstra`) |

**Label consistency for behaviour cloning.** The three oracles disagree on about one decision in three (§3.4). Almost all of that disagreement sits on one boundary — shortest-path versus geographic — because `gpsr` and `da_gpsr` agree on 75–97% of decisions. Mean local degree separates `medium_slow` (4.9) from the other scenarios (10.1–19.2) by roughly 2×, but the ranges overlap, so degree alone may not resolve every conflict.

**Agreement falls as load rises — in every scenario.** `dense_slow` goes 0.747 → 0.695 → 0.649; `medium_slow` 0.652 → 0.592 → 0.544; `very_dense` 0.647 → 0.593. Under light load, reasonable routing strategies converge on similar paths; under congestion they diverge. The choice of policy matters most exactly where the network is most stressed.

**The price of using a single teacher.** Averaged over the nine oracle cells:

| labelling scheme | mean PDR | cost vs per-cell | worst cell |
|---|---|---|---|
| per-cell oracles (ceiling) | 0.5048 | — | — |
| `da_gpsr` everywhere | 0.4923 | −1.25 pp | medium_slow −4.07 pp |
| `gpsr` everywhere | 0.4837 | −2.11 pp | medium_slow −4.07 pp |
| `dijkstra` everywhere | 0.4791 | −2.57 pp | very_dense −4.93 pp |

**The reinforcement-learning target.** Because no fixed teacher is best everywhere, the gap between the best single teacher and the per-cell ceiling is a concrete, measurable amount of improvement that a regime-aware policy could capture. If the warmstart uses `da_gpsr` everywhere, a known ~4 pp improvement exists in `medium_slow`, demonstrated to be achievable because `dijkstra` achieves it. That turns criterion D2 — *exceed the oracle* — into a target with a known location and size.

**Hop count is not the explanation.** In `medium_slow`@40, `dijkstra` (3.06 hops) beats `gpsr` (3.04 hops) by 4.08 pp despite essentially identical path lengths. In `dense_slow`@60, `da_gpsr` takes *more* hops than `dijkstra` (2.27 vs 2.09) and wins by 3.90 pp. Path length explains neither.

**Energy does not confound these results.** At the 8,000 battery, every oracle-setting cell showed zero energy-related drops and zero dead nodes.

---

## 8. Claims Made and Retracted During This Analysis

Recorded deliberately. Each was stated with confidence and later shown to be wrong.

**Retraction 1 — "Congestion-awareness hurts routing; the more a policy has, the worse it does."**
Made after Experiment 1. Wrong on two counts. It was drawn from the SP-BP family and from `backpressure`/`dpp` without reading their code; those two lose because they have *no destination term*, not because of their queue term. And it ignored `da_gpsr`, which is explicitly congestion-aware and wins four cells.

**Retraction 2 — "PDR rank is inverse hop rank; fewest hops wins."**
Made from four SP-BP ablations, where hop count was genuinely the only thing varying. It does not generalise across routing families: counterexamples appear in every scenario (§7).

**Retraction 3 — "Queue differentials are harmful; absolute penalties help."**
Made during the teacher audit. **Half right.** Absolute beats differential in 9/9 cells (C) — that part holds. But the differential is *not* uniformly harmful: it helps in `very_dense` and `sink_50` (B). The accurate statement is that the differential is *inferior* to the absolute form everywhere, and *harmful* only in sparse-to-moderate networks.

**The common cause of all three:** generalising from one teacher family, or from names and comments, instead of from the full set of controlled comparisons. The controlled-comparison table in §4 exists specifically to prevent a fourth.

---

## 9. Limitations and Open Questions

- **The density mechanism is a hypothesis.** Finding 3 is well-supported but the explanation — path diversity making detours cheap — has not been tested directly. A density sweep holding everything else fixed would establish it.
- **The link-quality staleness explanation is a hypothesis.** Measuring the correlation between decision-time link quality and transmission-time success would confirm or refute it.
- **Results may depend on interference calibration.** The finding that congestion appears mainly as link degradation depends on the interference model's parameters. A sensitivity sweep over interference strength would show where the balance tips toward queue-driven loss, and whether congestion-awareness gains value there.
- **`da_gpsr`'s weights were never tuned.** Its three weights (1.0, 1.0, 0.5) are fixed constants. A better-tuned version might raise the bar that the learned policy must clear — the more honest baseline.
- **Only one convergecast operating point.** `sink_50` had a single usable rate, so convergecast conclusions rest on one cell.
- **Ten seeds per cell.** Sufficient for the unanimous findings; some individual cell-level differences remain underpowered.
- **`sparse_fast` is unresolved.** Its losses are dominated by something other than links or queues, and no teacher separates significantly there.

---

## 10. Implications for the Project

**For the thesis narrative.** The contribution moves from "a congestion-aware router" to a characterisation of *when and how* congestion signals help FANET routing, with four controlled findings, three of them unanimous across nine cells. That is a stronger basis for a paper than the original premise would have provided even if it had held.

**For the reinforcement-learning stage.** The results supply a principled reason for learning: no hand-designed policy is best across regimes, the best one depends on locally observable density, and the winning hand-designed policy is a fixed, untuned linear combination of three signals. A policy that learns how to weight progress, congestion and link quality according to local conditions has a clear, measurable target.

**For the dataset.** The oracle is now settled per cell. The open design decision — per-cell teachers or a single global teacher — trades about 1.25 pp of warmstart quality against label consistency, and is set out in §7.

---

### Appendix — Experiment log

| experiment | teachers | cells | episodes | wall time | output file |
|---|---|---|---|---|---|
| 1. Generalized panel | 9 | 11 | 990 | ~20.4 h | `panel_oracle_generalized.json` |
| 2. Contender panel | 5 | 11 | 550 | ~10.0 h | `panel_contenders_v2.json` |
| 2b. Decision-rule correction | — | 11 | 0 | recompute | `panel_contenders_v2_corrected.json` |
| 3. Link-metric extension | +2 | 11 | 220 | ~4.2 h | `panel_extended_v3.json` |
| 4. Oracle-agreement probe | 3 | 9 | 9 | ~29 min | `oracle_agreement.json` |
| **Total** | | | **1,769** | **~35 h** | |
