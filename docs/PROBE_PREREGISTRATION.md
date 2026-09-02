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
2. **Lower anchor** = the lowest valid rate. **Retained even if its queue value is in the
   floor band** — the low-load contrast is required evidence, not a candidate for
   pruning.
3. **Peak** = argmax queue value among valid cells.
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

### 4.5 One grid or four

Default is **one global grid** covering the union of per-scenario ranges.

Switch to per-scenario grids **only if** the per-scenario argmax rates differ by ≥ 2 grid
steps. That switch has a cost that must be paid deliberately: `load_bucket()` becomes
scenario-aware, which changes `teacher_panel` and the dataset schema, and every
"12-cell panel" reference (including Gate G-A's criterion 3) needs rewording, not just
renumbering.

---

## 5. What this probe does NOT decide

- `load_bucket()` thresholds — a separate patch, after the grid is chosen. The current
  `≤0.5 low / ≤2.0 medium / else high` maps every probe rate to `low`, which is why
  `--by rate` exists (v11.2).
- Whether v8 applies. It does not, until the new environment's SP-BP-parity gate passes
  at the **current** 40 s operating point (FILE 2 §3).
