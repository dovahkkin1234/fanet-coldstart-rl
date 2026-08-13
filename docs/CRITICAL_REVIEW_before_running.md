# CRITICAL REVIEW — before spending 15 hours

Written as a critic of my own design, before anything runs.

## FLAW 1 (severe) — `--mask hop` does not remove the GNN's job

Masking `hop_distance_to_dst`, `cand_hop_distance`, `cand_reachable` leaves TWO
things a 2-layer GNN would otherwise have to compute:

    neigh_buffered_packets   } k-hop (LOCAL_HORIZON = 2) aggregates of
    neigh_mean_occupancy     } neighbour queue occupancy

These ARE the congestion lookahead a 2-layer GNN produces. They were added in
v1 precisely because a distributed router needs local load state — and in doing
so they pre-empted the GNN's second job, independently of the hop features.

Had the 2x2 run as specified, a null result would have been uninterpretable:
"message passing adds nothing" versus "message passing's remaining job was
still handed over pre-computed" are indistinguishable.

**FIXED.** New preset `--mask gnnjob` masks all five.

## FLAW 2 (structural, unfixable) — geometry cannot be masked

`dist_to_dest`, `progress` and node x/y/z must remain or neither model knows
where the destination is. Part A measured geographic-vs-hop correlation at
**0.89-0.93** in this scenario generator (uniform-random, open 2D, no
obstacles).

So geometry remains a strong topology proxy under any mask. **No masking
experiment in this simulator can support a general claim that graph networks do
not help FANET routing.** The defensible claim is bounded to open-2D
uniform-random scenarios — the same scope limit `experiment_spbp_mechanism`
already prints for Part A.

State this in the paper. A reviewer who reads Part A will otherwise find it.

## FLAW 3 — hyperparameter selection at 10 seeds may be noise

Validation SD across seeds is ~0.0084, so at n=10 the CI half-width is ~0.006.
Six configs will very likely land inside that band. Picking a "winner" per model
from noise, then running 30 seeds on it, injects a selection bias that can point
either way.

**RULE, fixed in advance:** deviate from the default config only if a config
beats it by more than its own CI. Otherwise keep the default and report "no
configuration significantly beat the default", which is an honest and complete
answer to the tuning objection.

## FLAW 4 — power is adequate but not generous

The 2x2's key quantity is whether the GNN-MLP gap CHANGES between arms — a
difference-of-differences. Seeds are shared across arms, so the arms are paired:

    rho = 0.0 (worst case), n=30 -> DoD CI half-width 0.0055
    rho = 0.5 (likely),     n=30 -> DoD CI half-width 0.0039

The gap to explain is 0.0108. So a HALVING is detectable; a quarter-change is
not. Adequate for the pre-registered threshold, marginal beyond it.

## FLAW 5 — masking makes ~8% of candidate slots undecidable

With `cand_reachable` masked, neither model can identify unreachable candidates
beyond its receptive field, while the SP-BP label excludes them. 92.1% of
candidates are reachable, so roughly 8% of slots become partly undecidable.

This caps achievable accuracy in the masked arm and adds noise. It does NOT
bias the comparison — both models face it — but it reduces power. Note that a
2-layer GNN can partially infer reachability within 2 hops where the MLP cannot,
so this is legitimately part of "the GNN's job" rather than a confound.

## THE BIGGER QUESTION — is any of this required?

**For M5, no.** Both architectures pass check 4 (97.6% and 98.5% of SP-BP).
Either works as a warmstart. Use the MLP: simpler, slightly better, already
validated. **The GNN-vs-MLP question does not gate the thesis at all.**

The entire 15-hour programme serves Paper A (the standalone negative result),
not Paper B (the thesis). M5-M11 is seven milestones and is completely
unstarted.

### Recommended scope cut

| Tier | Work | Cost | When |
|---|---|---|---|
| **1 — do** | pilot (1 seed, `--mask gnnjob`) then 2x2 at 30 seeds | 5 min + 3 h | now |
| **2 — conditional** | HP screening, 6 configs x 10 seeds | 4 h | only if tier 1 leaves the negative result standing |
| **3 — defer** | depth sweep, edgekey control | 7.5 h | only for a revise-and-resubmit |

**3 hours now, not 15.** Tiers 2 and 3 are confirmatory; their value depends
entirely on which way tier 1 lands, and running them first risks tuning for an
experiment whose framing then changes.

## PRE-REGISTERED PREDICTION (recorded before running)

Baseline, unmasked: MLP - GNN = **+0.0108** contested accuracy.

| Outcome under `--mask gnnjob` | Conclusion |
|---|---|
| gap drops below +0.005, or reverses | The pre-computed features were substituting for message passing. Claim: *explicit topology and load features substitute for a graph encoder at a fraction of the compute.* Stronger, mechanistic paper. |
| gap stays near +0.0108 or grows | Message passing adds nothing even when its job is removed. Negative result is broad **within the stated scenario scope**. |
| GNN wins masked but by less than the MLP wins unmasked | Report both cells; the honest claim is conditional. |

Also record the ABSOLUTE accuracy drop from masking. If both models fall toward
the trivial floor, the features carried nearly all the signal and neither
architecture matters much — itself a finding worth reporting.
