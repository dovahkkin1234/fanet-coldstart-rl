# M3 Design Spec — Congestion-Aware Teachers + Oracle Labeling

**Status:** design locked before implementation.
**Gate to clear before M4:** G3 (defined in §7).
**Depends on:** M1 (`link_model_v2`, G1 passed), M2 (`simulator_v2`, G2 passed).

---

## 0. Why this milestone decides the fate of the warmstart contribution

Warmstart quality *is* teacher quality. The RL agent is initialized by imitating
these teachers, so whatever bias exists in the labels becomes the agent's prior.
Approach 1 used five congestion-blind teachers, which was harmless because the
single-packet environment had no congestion. In the M2 environment congestion is
the dominant failure mode (G2: congestion-caused drops rise 0% → 88.5% with
load), so congestion-blind labels would warmstart the agent into exactly the
behaviour the paper claims to fix.

This milestone therefore has two deliverables, and the second is the one that
carries research weight:

1. A teacher set that contains genuinely congestion-aware policies (§2).
2. A labeling scheme that does not let congestion-blind teachers dominate the
   label by correlation rather than by quality (§4).

---

## 1. The correlation-bloc problem (why not just add more teachers)

Naive majority vote over a large heterogeneous teacher set fails in a specific,
silent way.

Congestion-blind shortest-path-family teachers (Dijkstra, GPSR, AODV,
Stable-Path, Link-Lifetime) compute *near-identical* next hops, because they are
all minimizing variants of the same distance/hop objective. They form a
**correlated voting bloc**: 5 votes that are not 5 independent opinions.

Congestion-aware teachers (Backpressure, SP-BP, DA-GPSR, WCETT, ABC) use
genuinely different metrics (queue differential vs geometric avoidance vs
expected transmission time vs available bandwidth), so they *disagree with each
other* far more often.

Under high load — exactly where congestion-awareness matters most — the blind
bloc out-votes the congestion-aware minority and the majority-vote label reverts
to shortest-path. Worse, the failure is invisible to the obvious diagnostic:
`consensus_confidence` looks *high*, because the bloc agrees with itself.

This is the same class of silent failure as the saturated `link_quality` in
Approach 1: plausible-looking numbers, dead mechanism underneath.

**Two mitigations, both applied:**
- Prune the set so no family dominates by headcount (§2).
- Replace majority-vote labeling with oracle labeling (§4).

---

## 2. The pruned 6-teacher set

Two congestion-blind references (needed as the comparison baseline and because
at low load they *are* near-optimal), four congestion-aware.

| # | Teacher | Family | Congestion metric | Status |
|---|---------|--------|-------------------|--------|
| 1 | Dijkstra | Shortest-path | none (hop count) | **exists** (`routing_teachers.py`) |
| 2 | GPSR | Geometric | none (geographic progress) | **exists** (`routing_teachers.py`) |
| 3 | Backpressure | Backpressure | queue differential ΔQ | **exists** (`routing_teachers_v2.py`), needs revision (§3.1) |
| 4 | SP-BP | Hybrid backpressure | ΔQ + hop distance bias | **NEW** (§3.2) |
| 5 | DA-GPSR | Geometric congestion-aware | progress − queue delay | **exists in substance** as `queue_aware_greedy`, formalize (§3.3) |
| 6 | ETX/ABC | Dynamic-metric Dijkstra | expected transmissions / PER | **NEW** (§3.4) |

**Excluded and why:**
- *AODV, Stable-Path, Link-Lifetime* — all shortest-path-family; add bloc weight
  without adding decision diversity. (Retained in the codebase; simply not part
  of the M3 teacher panel.)
- *Drift-plus-penalty* — a generalization of SP-BP with an extra tunable V; the
  marginal label diversity does not justify the extra hyperparameter to defend
  to a reviewer. Candidate for the journal extension.
- *CAR* — overlaps heavily with DA-GPSR in this environment (both are
  "geometric progress penalized by local congestion"). Keeping one avoids a
  second correlated pair.

Rationale to state in the paper: the panel spans four *decision principles*
(hop-count, geometry, queue-differential, link-cost), with congestion-aware
teachers deliberately in the majority so that oracle labeling is not fighting
the panel composition.

---

## 3. Teacher specifications

All teachers keep the established signature:
`f(G, current, destination) -> next_hop | None`, and are pure functions of the
graph snapshot (no persistent state) unless stated. Node attrs available:
`x, y, z, vx, vy, vz, energy, queue_occupancy, queue_len`. Edge attrs:
`distance, relative_velocity, link_quality, base_link_quality,
estimated_link_lifetime, rssi, snr, packet_error_rate`.

### 3.1 Backpressure (revise existing)

Current implementation weights by `link_quality * (Q_current - Q_n)` and falls
back to progress when no positive differential exists. Two problems observed in
M2 bring-up: it underperformed `queue_aware_greedy` at every load tested, and
its low-load behaviour is dominated by the fallback rather than by backpressure
itself.

Revision:
- Keep the destination-neighbour short-circuit (deliver directly if `dst` is a
  neighbour) — this is standard practice, not a fudge.
- Weight: `w(u) = link_quality(v,u) * max(Q_v - Q_u, 0)`.
- **Loop-avoidance is the caller's job**, not the teacher's; `simulator_v2`
  already masks already-visited nodes. Teachers must not duplicate this or the
  two mechanisms interact confusingly.
- Fallback when all weights are zero: keep progress-ward choice, but **record
  that the fallback fired** (see §5, `bp_fallback` flag) so we can quantify how
  often "backpressure" is really acting as greedy. This is a reviewer-facing
  honesty measure and a diagnostic for §7's G3 check.

### 3.2 SP-BP — Shortest-Path Backpressure (NEW)

Fixes classical backpressure's known low-load pathology (packets random-walk
because an empty network has no queue gradient to follow).

```
score(u) = link_quality(v,u) * [ (Q_v - Q_u) + V_bias * (h(v) - h(u)) ]
```
where `h(n)` = hop distance from `n` to destination (BFS on the current graph,
computed once per decision), and `V_bias` converts hop-progress into queue
units. Pick `u` maximizing `score`.

- `V_bias` default **1.0** — i.e. one hop of progress is worth one packet of
  backlog. Chosen for interpretability, not tuned. Sensitivity is an ablation
  candidate, not an M3 concern.
- If `h(u)` is unreachable (disconnected), treat `h(u) = +inf` → excluded.
- Expected behaviour: matches Dijkstra at low load, matches Backpressure at
  high load. **This is the teacher most likely to top the oracle ranking**, and
  §7's G3 tests exactly that.

### 3.3 DA-GPSR — Delay-Aware GPSR (formalize existing)

`queue_aware_greedy_next_hop` already implements the substance. M3 renames it
`da_gpsr_next_hop` (keeping a backward-compatible alias so M2's G2 harness
still runs) and pins the weights explicitly:

```
score(u) = w_prog * progress(u) - w_queue * queue_occupancy(u) + w_lq * link_quality(v,u)
w_prog = 1.0, w_queue = 1.0, w_lq = 0.5     (as validated in M2)
```

Do **not** retune these in M3. They produced the actor spread that G2
validated (+0.04–0.06 PDR over Dijkstra); changing them now would invalidate
that result without new evidence.

### 3.4 ETX-Dijkstra (NEW) — the dynamic-metric teacher

WCETT and ABC both need per-link bandwidth/channel-diversity state the M2
simulator does not model. **ETX (expected transmission count)** is the
well-established member of that family that this environment *does* support
exactly — `packet_error_rate` is computed per link by `link_model_v2`, and ARQ
means retransmission count is the physically meaningful cost.

```
etx_weight(v,u) = 1 / max(1 - per(v,u), eps)        eps = 1e-3
next_hop = shortest_path(G, v, dst, weight=etx_weight)[1]
```

This is interference-reactive by construction: as load raises PER on congested
links, their ETX weight rises and the path routes around them. It is also the
closest classical analogue to what the RL agent should learn, which makes it a
strong teacher and an honest baseline.

Note for the paper: call this **ETX-Dijkstra**, not WCETT. WCETT's defining
feature is its channel-diversity term, which is not modelled here; claiming
WCETT would be inaccurate.

---

## 4. Oracle labeling (the core methodological contribution of M3)

### 4.1 Why not majority vote
See §1. Additionally: majority vote optimizes *agreement*, but the quantity we
care about is *achieved network PDR*. These come apart precisely under load.

### 4.2 The scheme

**Phase A — Teacher benchmarking (produces the oracle ranking).**
Run each of the 6 teachers standalone as the simulator actor across the full
grid: `scenarios × packet_rates × seeds`. Record network PDR, mean delay, and
drop taxonomy per cell. Aggregate to an **oracle table**:

```
oracle[(scenario_class, load_bucket)] = ranked list of teachers by mean network PDR
```

`load_bucket` ∈ {low, medium, high} from `packet_rate`; `scenario_class` from
the existing regime clustering (dense / sparse / medium-fast) so the table
generalizes rather than memorizing individual scenarios.

**Phase B — Trajectory generation with oracle labels.**
For each decision point encountered:
- Query **all 6** teachers for their proposed next hop.
- The label `is_best` marks the action proposed by the **oracle-best teacher for
  this (scenario_class, load_bucket) cell** — i.e. the teacher empirically shown
  in Phase A to achieve the highest network PDR in this regime.
- Retain the full vote vector (all 6 proposed actions) and compute
  `vote_agreement` = fraction of teachers agreeing with the oracle-best action.
  **This becomes a confidence weight, not the label.**

### 4.3 Behaviour policy (whose states do we learn on?)

The state distribution depends on who drives the simulation. Choice:
**drive with the oracle-best teacher for that regime, with ε-random deviation
(ε = 0.1) for state diversity.**

- Driving with the oracle-best teacher keeps the visited-state distribution
  close to the policy we are labeling toward (avoids labeling states the good
  policy would never reach).
- ε-random deviation prevents the dataset from being a single narrow trajectory
  and gives the agent recovery experience from suboptimal states — the standard
  DAgger-style argument.
- Record `behaviour_deviated` per row so the ε-steps can be filtered or weighted
  in M4 if they prove harmful.

### 4.4 What this buys, stated plainly for the paper

The label is defined by *measured achieved performance in the matching regime*,
not by teacher popularity. If the congestion-blind teachers genuinely route best
at low load, they win the label there — correctly. If backpressure-family
teachers route best under load, they win there — also correctly. The scheme
cannot be fooled by correlated blocs because headcount never enters the label.

---

## 5. Dataset schema

One row per routing decision. Extends the Approach-1 schema with congestion and
oracle-labeling fields.

| Field group | Fields |
|---|---|
| Identity | `episode, scenario_id, scenario_class, seed, packet_id, hop_index, t` |
| Load context | `packet_rate, load_bucket, n_inflight, network_mean_occupancy` |
| Candidate features | per candidate: `distance, link_quality, per, snr, est_lifetime, relative_velocity, progress, queue_occupancy, queue_len, energy, is_destination` |
| Packet context | `ttl_left, hops_so_far, dist_to_dest, current_queue_occupancy` |
| Teacher votes | `action_dijkstra, action_gpsr, action_backpressure, action_spbp, action_dagpsr, action_etx` |
| Oracle label | `oracle_teacher, is_best (per candidate), vote_agreement` |
| Diagnostics | `bp_fallback, behaviour_deviated` |
| Outcome | `hop_succeeded, tx_attempts, eventual_delivered, drop_reason` |

`eventual_delivered` requires backfilling once the packet terminates — the
per-packet trajectory tracking built in M2 makes this straightforward, and it
enables outcome-weighted training in M4 if wanted.

---

## 6. Known issue carried from M2 that M3 must fix

G2 showed a large, load-independent `episode_end` drop count (34–112 packets),
i.e. packets still in flight when the 40 s episode ends. These are **not
congestion drops** and they currently dilute PDR comparisons between teachers.

Fix in the M3 benchmark harness (not the simulator core):
- Add a **drain phase**: stop generating new packets at `T - drain_time`
  (default `drain_time = 10 s`), continue simulating so in-flight packets can
  terminate naturally.
- Report **PDR over packets generated before the drain cutoff** as the headline
  metric; report `episode_end` separately as a diagnostic.

This makes teacher comparison cleaner without touching the validated M2 physics.

---

## 7. GATE G3 — what must be true before M4

Run the Phase-A benchmark grid and confirm ALL of:

1. **All 6 teachers run without error** across the full grid, and each beats a
   random-neighbour policy at every load. (Sanity floor.)
2. **Backpressure-family (Backpressure or SP-BP) tops the ranking at high
   load.** This is the substantive check: it validates that the M2 congestion
   model rewards congestion-awareness. If shortest-path wins at every load,
   either the congestion coupling is too weak or the backpressure
   implementations are wrong — investigate before proceeding.
3. **Ranking changes with load.** The oracle table must not be constant across
   load buckets; if one teacher wins everywhere, oracle labeling degenerates to
   single-teacher imitation and the multi-teacher framing is unjustified.
4. **Ranking changes across scenario classes** (dense vs sparse). Supports the
   CL-across-regimes thesis with independent evidence.
5. **Vote agreement is materially below 1.0 under load** — confirms the teachers
   genuinely disagree where it matters, i.e. the labeling problem is real.
6. **Reproducible** under fixed seed, as with G1/G2.

Harness: `src/preflight_teachers_v2_check.py`, PASS/FAIL output in the same
style as the G1/G2 gates.

---

## 8. Files this milestone produces

| File | Purpose |
|---|---|
| `src/routing_teachers_v2.py` (extended) | + `spbp_next_hop`, `etx_dijkstra_next_hop`, `da_gpsr_next_hop` (alias), revised backpressure |
| `src/teacher_panel.py` | the 6-teacher panel, oracle table construction, vote collection |
| `src/preflight_teachers_v2_check.py` | **G3 gate harness** |
| `src/generate_dataset_v2.py` | Phase-B trajectory generation with oracle labels |

Order of work: teachers → G3 gate → (only if G3 passes) dataset generation.
**Do not generate the dataset before G3 passes** — a dataset built on a broken
oracle table would silently poison M4 and every result downstream.

---

## 9. Explicitly NOT in M3 (scope fence)

- GNN/attention architecture and supervised pretraining → M4.
- RL training, reward design → M5.
- Extra mobility models (Gauss-Markov, RPGM, Flocking) → after M4.
- Jamming, CBR, continual learning → M9–M11.
- Simulator vectorization / A100 port → after G4 profiling.

If any of these start appearing in M3 code, stop.
