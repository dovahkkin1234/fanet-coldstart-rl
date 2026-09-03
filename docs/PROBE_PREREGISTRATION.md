# Rate-grid probe — PRE-REGISTRATION

**Status: FROZEN.** Commit this file before running the probe. Nothing below may be
revised after seeing probe output. If a result contradicts a rule here, report the
contradiction — do not adjust the rule (FILE 1 §11, errors 12 and 13).

**Purpose.** Choose the packet-rate grid for the regenerated 1000 s dataset. The old
`(0.5, 2.0, 4.0)` was calibrated for 40 s episodes and saturates completely at 1000 s
(FILE 2 §1.5). The five-rate proposal `(0.05, 0.15, 0.4, 0.7, 1.0)` that replaced it was
flagged as likely wrong and superseded (FILE 2 §1.5.1). This probe picks the grid from
measurement instead.

---

## 1. What is measured

Four scenarios × 6 candidate rates `(0.02, 0.05, 0.10, 0.15, 0.25, 0.40)` × 5 seeds ×
6 actors, at the **new** operating point (1000 s, altitude 100–300 m), aggregated
`--by rate`.

**Seeds are PAIRED across actors** — the same seed list runs for every actor in every
cell, so every actor-difference is a paired difference. Unpaired, between-seed variance
(`routable_frac` ranged 0.74–0.87 across two seeds in a recorded run) would swamp a 3 pp
effect at n=5.

### Actors and what each one isolates

| actor | role |
|---|---|
| `spbp` | the oracle; upper reference |
| `spbp_ab_full` | same rule via the ablation code path — **free regression check** on the v3 fix (must equal `spbp` exactly; v3 closed a 0.4061→0.4128 divergence) |
| `spbp_ab_noqueue` | **PRIMARY COMPARATOR** — matched control, identical code path with only the queue term neutralised. Produced the +0.0645 figure in FILE 1 §6.2 |
| `dijkstra` | hop-count shortest path; blind to queues *and* link quality |
| `gpsr` | greedy geographic; separates queue-blindness from geometry-blindness |
| `random` | floor — bounds "does routing matter here at all" |

### The three quantities (FILE 2 §1.5.1)

1. **QUEUE VALUE (primary)** — paired `PDR_routable(spbp_ab_full) −
   PDR_routable(spbp_ab_noqueue)`, in PDR points, with a paired 95% CI.
2. **CARRIED LOAD** — delivered pkt/s (`n_delivered / duration`). Locates the collapse
   knee and **bounds the top of the grid**.
3. **ADDRESSABLE SHARE** — the existing `experiment_headroom.py` metric. Recorded for
   continuity at the new operating point (M6 needs it). **Must not select rates.**

---

## 2. Why the primary metric changed — recorded reasoning

`spbp − dijkstra` was the originally proposed spread. It is **rejected** because it
conflates queue value with link-quality value, and the two move oppositely in load.
Measured at 40 s, 3 seeds:

| scenario | rate | queue value (matched) | spbp − dijkstra |
|---|---|---|---|
| dense_slow | 0.5 | 9.84 pp | 21.27 pp |
| dense_slow | 2.0 | 3.65 pp | 10.48 pp |
| dense_slow | 4.0 | 3.21 pp | 7.86 pp |
| medium_slow | 0.5 | 12.65 pp | 11.86 pp |
| medium_slow | 2.0 | 13.51 pp | 14.41 pp |
| medium_slow | 4.0 | 7.75 pp | 8.86 pp |

Also rejected, with reasons:

- **`R_rel = ΔPDR / addressable_headroom`** — divides by a quantity that is *designed* to
  approach zero at low load. Measured: `dense_slow` rate 0.5 has headroom exactly 0.000
  (**division by zero**); `medium_slow` rate 0.5 gives R_rel = 8.12. Every measured cell
  cleared a proposed ≥0.25 threshold, so the criterion selected 6/6 cells and
  discriminated nothing. **Reported as a diagnostic only, never used to select, and
  undefined when headroom < 0.01.**
- **Relative spread `(a−b)/a` as the primary** — has the mirror-image problem: as PDR → 0
  in collapse, it inflates on a shrinking base and would drag the grid past the knee.
  Recorded alongside; not the selector.
- **Per-cell filtering on spread** — would delete the low-load anchor. The grid is also
  the dataset and D1–D5 evaluation grid, and the thesis claim in FILE 1 §12 *is* the
  low-vs-high contrast. A rule that keeps only high-spread cells removes the evidence for
  its own headline claim.

**Independent confirmation that headroom must not pick rates:** at `dense_slow` rate 0.5,
queue value is 9.84 pp while addressable headroom is exactly 0.000 — the two quantities
point opposite ways in the same cell.

---

## 3. Recorded predictions (before the 1000 s run)

Written down so they can fail publicly.

- **P1.** `spbp` and `spbp_ab_full` agree to < 1e-9 in every cell. *(Held at 40 s across
  6 cells.)* Failure ⇒ a v3 regression; stop and fix before reading anything else.
- **P2.** Queue value is **scenario-dependent in shape**, not uniformly unimodal. *(At
  40 s: `dense_slow` falls monotonically, `medium_slow` peaks at 2.0. An earlier
  prediction that the matched ablation would rise with load in both was WRONG and is
  recorded as such.)*
- **P3.** `episode_end` share collapses toward zero at 1000 s in the low-load cells,
  where it was ~72% of all loss at 40 s. Failure ⇒ packets are being held by something
  the 10 s drain does not clear.
- **P4.** Rate 0.02 produces a usable packet count at 1000 s. *(At 40 s it produced
  exactly zero — the degenerate case the validity gate exists to catch.)*

**Smoke-run observations (40 s, plumbing check — NOT grid evidence):** P1 held, max drift
0.00e+00 across both scenarios. The §4.4 boundary rule fired on `dense_slow` (argmax at
the lowest probed rate) and correctly refused to emit a grid. The §4.1 validity gate
fired on `medium_slow` rate 0.5 (`n_routable`=169 < 200). Both behaved as designed on
real data.

---

## 4. Decision rule — FROZEN

### 4.1 Cell validity

A cell is **valid** only if both hold:

- `n_routable ≥ 200` summed across seeds, and
- `PDR(spbp) − PDR(random) ≥ 2 pp`

An invalid cell is excluded from grid selection entirely and reported as `INVALID` with
its reason. A cell failing the second test is degenerate — either disconnected or fully
choked — and no policy can learn there.

### 4.2 Effect-size bands for queue value

Calibrated to this project's measured magnitudes, not round numbers: the queue term is
worth +6.45 pp overall and the `spbp − da_gpsr` gap is 3.58 pp, so the meaningful scale is
single-digit pp.

| band | queue value (point estimate) |
|---|---|
| strong | ≥ 8 pp |
| usable | 3 – 8 pp |
| floor | < 3 pp |

A band is assigned from the **point estimate**, and additionally requires
**sign consistency ≥ 0.8** — the paired difference must be positive in at least 4 of 5
seeds (⌈0.8n⌉ generally) — for `usable` or `strong`. A cell failing consistency is
demoted to `floor` regardless of its point estimate.

**Why not the CI lower bound.** An earlier draft of this rule banded on the paired CI
lower bound. The smoke run showed that is unusable at this seed count: measured paired-
difference SD is ~13.5 pp at 40 s, so for a 13 pp effect the 95% CI half-width is 16.7 pp
at n=5 — every cell lands in `floor`, and 6/6 smoke cells did, despite effects of
4–15 pp. Reaching a usable CI lower bound needs n≈10+. Rather than inflate the probe to
30 seeds for what is a **scoping decision, not a headline claim**, the rule bands on the
point estimate and controls noise with paired sign consistency, which is robust at small
n. The CI is still computed and reported.

1000 s episodes average ~25× more traffic internally than the 40 s smoke, so between-seed
variance should fall substantially — but this rule deliberately does **not** depend on
that happening.

**Seeds: 5 minimum, 10 recommended.** At n=10 the CI becomes informative
(half-width ≈9.6 pp on the smoke's variance) at roughly 2.8 h instead of 1.4 h.

### 4.3 Grid selection

Per scenario:

1. **Upper bound (knee)** = the rate at which **carried load** peaks. Past it, offered
   load is being discarded and no routing policy recovers it. The grid may not extend
   above this rate.
   > **AMENDMENT 1 (A2), post-probe.** Superseded. `argmax carried load` returned 0.40
   > for `dense_slow`, `medium_slow` and `very_dense`, where the 0.25→0.40 increment is
   > +0.4%, +3.2% and +1.9% — a plateau, not a peak. **The knee is the highest rate whose
   > carried-load increment over the previous rate is ≥ 2%.** Beyond it, throughput has
   > saturated. Note this bounds where *throughput* saturates, **not** where the learning
   > signal stops — see §6.3.
2. **Lower anchor** = the lowest valid rate. **Retained even if its queue value is in the
   floor band** — the low-load contrast is required evidence, not a candidate for
   pruning.
3. **Peak** = argmax queue value among valid cells.
   > **AMENDMENT 1 (A1), post-probe.** Insufficient. **A peak may not be selected from a
   > cell whose band is `floor`.** If no valid cell clears `floor`, the scenario has **NO
   > MEASURABLE EFFECT** and no grid may be emitted for it.
   >
   > As written this rule emitted a grid for `sparse_fast` built on noise: every cell was
   > `floor`, and the selected peak (rate 0.05, +2.30 pp) had **sign consistency 0.1 — 9
   > of 10 paired seeds were negative**, with one outlier carrying the mean. §4.1 validity
   > tested packet counts and the random floor; it never tested whether the effect exists.
4. **Emit 5 rates** spanning `[lower anchor … knee]`, guaranteed to include the peak.

If fewer than 5 valid rates exist in that span, the probe range is **too narrow or too
coarsely sampled** — report the shortfall and add probe points before selecting, rather
than emitting an undersized grid. *(Observed in the smoke run: `medium_slow` yielded only
`[2.0, 4.0]`.)*

### 4.4 Boundary rule — the probe range can invalidate itself

**If the argmax queue value falls on either endpoint of the probed range, the range is
wrong.** The true peak lies outside it and any grid anchored on that endpoint is an
artifact of where probing started. In that case: extend the range in the indicated
direction and re-probe **before** selecting a grid. Do not select from a boundary argmax.

*(This is not hypothetical — at 40 s, `dense_slow`'s argmax sits at the lowest probed
rate.)*

> **AMENDMENT 1 (A3), post-probe.** This rule assumes a peak exists. It fired in 3 of 4
> scenarios at 1000 s, reporting `EXTEND RANGE LOWER`. That reading was wrong: **relative
> queue value is flat at 14–22% across the full 20× load range** (§6.1), so there is no
> peak to bracket and extending below 0.02 would chase one into an empty network.
>
> **Amended:** before acting on a boundary argmax, test flatness. If relative queue value
> across valid cells varies by less than a factor of 2 between its own max and min, the
> model is FLAT — report `NO PEAK — FLAT RESPONSE`, do **not** extend the range, and
> select the grid by coverage (§6.3) instead. Only a boundary argmax **with** a genuine
> gradient justifies extending the range.

### 4.5 One grid or four

Default is **one global grid** covering the union of per-scenario ranges.

Switch to per-scenario grids **only if** the per-scenario argmax rates differ by ≥ 2 grid
steps. That switch has a cost that must be paid deliberately: `load_bucket()` becomes
scenario-aware, which changes `teacher_panel` and the dataset schema, and every
"12-cell panel" reference (including Gate G-A's criterion 3) needs rewording, not just
renumbering.

---

## 6. RESULTS — 1000 s probe, 1440 episodes, 10 paired seeds

Run 2026-09-02. `results/probe_rate_grid.json`. All four predictions in §3 held: P1 drift
exactly 0.00e+00 in all four scenarios (v3 fix intact); P2 confirmed (shape is
scenario-dependent); P3 confirmed (`episode_end` fell from 0.231 at 40 s to 0.055–0.084);
P4 held, though `sparse_fast` at rate 0.02 returned only 225 routable packets against the
200 threshold.

### 6.1 The inverted-U model is refuted

Absolute queue value falls monotonically with load, but **relative** queue value is flat:

| scenario | 0.02 | 0.05 | 0.10 | 0.15 | 0.25 | 0.40 |
|---|---|---|---|---|---|---|
| dense_slow | 18.1% | 16.4% | 18.6% | 18.0% | 17.3% | 15.4% |
| medium_slow | 16.4% | 16.5% | 20.9% | 21.6% | 17.0% | 15.9% |
| very_dense | 18.7% | 15.3% | 15.0% | 15.7% | 14.2% | 16.8% |
| sparse_fast | 0.0% | 2.7% | 2.8% | 2.3% | 6.1% | −1.1% |

Congestion-awareness buys a roughly constant ~17% relative improvement regardless of load.
The absolute decline is PDR itself declining. **Grid selection cannot be peak-based.**

### 6.2 Two post-probe claims, RETRACTED

Recorded because the project's convention is that refuted claims are published in the same
place as confirmed ones.

**RETRACTED — "the 17% is mostly an episode-boundary artifact."** The argument was that the
gain is dominated by reduced `episode_end`, so it measures latency rather than delivery. It
rested on an adjustment that counts in-flight packets as delivered — which is **biased
toward whichever arm has more stuck packets**, i.e. the no-queue arm. Refuted by direct
measurement (`dense_slow`, 300 s, 3 paired seeds; the probe had not recorded
`mean_delay_ms`):

| rate | actor | delay_ms | hops | delivered | pdr_routable |
|---|---|---|---|---|---|
| 0.05 | `spbp_ab_noqueue` | 9.8 | 1.88 | 69.0 | 0.708 |
| 0.05 | `spbp_ab_full` | **11.1** | **2.14** | **86.3** | **0.886** |
| 0.15 | `spbp_ab_noqueue` | 10.3 | 1.79 | 156.3 | 0.522 |
| 0.15 | `spbp_ab_full` | **11.8** | **2.01** | **186.7** | **0.622** |

The queue term delivers **more** packets, over **longer** paths, at **higher** per-packet
delay. It is genuine congestion avoidance — routing around busy nodes, paying delay and
link-error exposure, getting far fewer packets stuck. The 17% stands.

**RETRACTED — "FILE 1 §12's high-load framing is contradicted."** The argument was that
absolute queue value falls with load, so the thesis should not live at high load. But that
measures what SP-BP already *captured*, not what remains. Unclaimed budget
(`addressable − queue value`) — the room a better policy could still compete for:

| scenario | 0.02 | 0.05 | 0.10 | 0.15 | 0.25 | 0.40 |
|---|---|---|---|---|---|---|
| dense_slow | −0.135 | 0.020 | 0.166 | 0.287 | 0.384 | **0.453** |
| medium_slow | −0.092 | 0.034 | 0.137 | 0.253 | 0.383 | **0.449** |
| very_dense | −0.125 | 0.020 | 0.124 | 0.185 | 0.264 | **0.309** |

It grows monotonically with load in all three connected scenarios. **The most room above
SP-BP is at high load**, exactly as §12 states. No documentation change is warranted.

*(The negative values at rate 0.02 are real: queue value exceeds the addressable budget
there, because the queue term also prevents losses outside it.)*

### 6.3 FROZEN GRID

Selected by **coverage**, not peak-finding, per A3:

```
RATES = [0.02, 0.05, 0.10, 0.25, 0.40]
```

Spans `spbp` routable PDR 0.76 → 0.095 in `dense_slow`, roughly log-spaced over probed
points, giving 2 low / 1 medium / 2 high buckets — the 2/1/2 shape the superseded FILE 2
§1.5 proposal intended.

Rate 0.40 is **deliberately retained despite sitting past the carried-load knee.** The knee
bounds throughput saturation, not learning signal: 0.40 carries the largest unclaimed
budget (0.45) and is the deepest congestion regime, which is what D2 needs.

### 6.4 `sparse_fast` — NO MEASURABLE EFFECT, separate decision required

`sparse_fast` **does not inherit the frozen grid.** Every cell is `floor` band; sign
consistency never exceeds 0.6; carried load is flat at 0.06 from rate 0.05 onward; PDR is
non-monotonic in load (0.750 at 0.02, 0.861 at 0.05); and the lowest cell has ~22 routable
packets per episode. It is **connectivity-limited, not congestion-limited** — its
addressable budget is the largest of any scenario (0.80) precisely because most loss is
`no_route`, which congestion-awareness cannot address.

D5 (graceful degradation) is written against this scenario. Its rate grid, and whether the
D5 criterion should reference congestion at all, are **open decisions**, deferred to Phase 0
alongside the D1–D5 mapping. Do not silently apply the global grid to it.

---

## 5. What this probe does NOT decide

- `load_bucket()` thresholds — a separate patch, after the grid is chosen. The current
  `≤0.5 low / ≤2.0 medium / else high` maps every probe rate to `low`, which is why
  `--by rate` exists (v11.2).
- Whether v8 applies. It does not, until the new environment's SP-BP-parity gate passes
  at the **current** 40 s operating point (FILE 2 §3).
