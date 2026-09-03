"""apply_probe_amendment_v1.py -- amend the probe pre-registration and runner.

Assertion-guarded str.replace, same convention as every patch here. Anchors must
match EXACTLY ONCE per file; edits staged in memory; nothing written unless every
anchor in every file matches. Idempotent.

WHY AN AMENDMENT AND NOT A REWRITE

The pre-registration was frozen and the probe ran against it. The probe then
(a) refuted the model the rules were built to fit and (b) exposed three defects in
the rules themselves. The project convention is that a frozen rule contradicted by a
result is REPORTED, not quietly rewritten -- so the original §4.3/§4.4 text stays
visible with its defect stated, the amendment is dated and additive, and the two
retracted post-probe claims are recorded with the measurements that refuted them.

WHAT CHANGES

  docs/PROBE_PREREGISTRATION.md  (4 anchors)
    A1  §4.3 -- peak may not be selected from a floor-band cell. sparse_fast got a
        grid built on one outlier seed (peak cell consistency 0.1: 9 of 10 paired
        seeds NEGATIVE). Validity tested packet counts and the random floor, never
        whether the effect exists.
    A2  §4.3.1 -- knee redefined as the last MATERIAL rise (>=2% increment), not
        argmax carried load. As implemented it returned 0.40 for three scenarios
        where the 0.25->0.40 increment is +0.4%.
    A3  §4.4 -- boundary rule must distinguish "range is wrong" from "model is
        wrong". It fired in 3 of 4 scenarios reporting EXTEND RANGE when relative
        queue value is flat at 14-22% across 20x of load, i.e. there is no peak.
    A4  new §6 -- probe results, the frozen grid, the two retractions with the
        measurements that refuted them, and sparse_fast's separate status.

  src/probe_rate_grid.py  (3 anchors)
    B1  knee = last material rise (implements A2)
    B2  peak must clear the floor band; otherwise NO MEASURABLE EFFECT (implements A1)
    B3  boundary rule reports FLAT-vs-BOUNDARY correctly (implements A3)

USAGE
    python apply_probe_amendment_v1.py --root . --dry-run
    python apply_probe_amendment_v1.py --root .
"""
import argparse, io, os, sys

DOC = os.path.join('docs', 'PROBE_PREREGISTRATION.md')
SRC = os.path.join('src', 'probe_rate_grid.py')
# Per-file guards: the doc carries the literal amendment marker, the source
# carries a string unique to the amended code. A single shared guard silently
# broke idempotency -- the doc matched, the source did not, and a second run
# aborted on 0/4 anchors instead of reporting "already applied".
GUARDS = {DOC: 'AMENDMENT 1', SRC: 'FLAT RESPONSE'}

# ─────────────────────────────── doc edits ────────────────────────────────
EDITS_DOC = [
    # A1 + A2 -- grid selection
    ("""1. **Upper bound (knee)** = the rate at which **carried load** peaks. Past it, offered
   load is being discarded and no routing policy recovers it. The grid may not extend
   above this rate.
2. **Lower anchor** = the lowest valid rate. **Retained even if its queue value is in the
   floor band** — the low-load contrast is required evidence, not a candidate for
   pruning.
3. **Peak** = argmax queue value among valid cells.
4. **Emit 5 rates** spanning `[lower anchor … knee]`, guaranteed to include the peak.""",
     """1. **Upper bound (knee)** = the rate at which **carried load** peaks. Past it, offered
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
4. **Emit 5 rates** spanning `[lower anchor … knee]`, guaranteed to include the peak."""),

    # A3 -- boundary rule
    ("""**If the argmax queue value falls on either endpoint of the probed range, the range is
wrong.** The true peak lies outside it and any grid anchored on that endpoint is an
artifact of where probing started. In that case: extend the range in the indicated
direction and re-probe **before** selecting a grid. Do not select from a boundary argmax.

*(This is not hypothetical — at 40 s, `dense_slow`'s argmax sits at the lowest probed
rate.)*""",
     """**If the argmax queue value falls on either endpoint of the probed range, the range is
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
> gradient justifies extending the range."""),

    # A4 -- results section
    ("""---

## 5. What this probe does NOT decide""",
     """---

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

## 5. What this probe does NOT decide"""),
]

# ─────────────────────────────── code edits ───────────────────────────────
EDITS_SRC = [
    # B1 -- material knee
    ("""        knee = max(valid, key=lambda c: c['carried'])['rate']       # §4.3.1""",
     """        # §4.3.1 as AMENDED (A2): last MATERIAL rise, not argmax. argmax returned
        # 0.40 where the 0.25->0.40 carried-load increment is +0.4%.
        knee = valid[0]['rate']
        for prev, cur in zip(valid, valid[1:]):
            if prev['carried'] > 1e-12 and (cur['carried'] - prev['carried']) / prev['carried'] >= 0.02:
                knee = cur['rate']"""),

    # B2 -- peak must clear floor
    ("""        peak = max(valid, key=lambda c: c['queue_value'])['rate']    # §4.3.3""",
     """        # §4.3.3 as AMENDED (A1): a peak may not come from a floor-band cell.
        non_floor = [c for c in valid if c['band'] != 'floor']
        if not non_floor:
            res.update(decision='NO MEASURABLE EFFECT -- every valid cell is floor band',
                       grid=None, knee=knee, anchor=anchor, peak=None,
                       note='no cell clears the floor band; sign consistency never '
                            'reaches 0.8. Scenario needs its own decision (pre-reg §6.4).')
            out[sc] = res
            continue
        peak = max(non_floor, key=lambda c: c['queue_value'])['rate']"""),

    # B4 -- --reanalyse: re-run the AMENDED rules on saved rows, no re-simulation
    ("""    ap.add_argument('--smoke', action='store_true',
                    help='fast plumbing check at the OLD 40s operating point, '
                         '2 scenarios x 3 rates x 2 seeds')
    args = ap.parse_args()""",
     """    ap.add_argument('--smoke', action='store_true',
                    help='fast plumbing check at the OLD 40s operating point, '
                         '2 scenarios x 3 rates x 2 seeds')
    ap.add_argument('--reanalyse', metavar='JSON', default=None,
                    help='re-run the decision rule on a saved probe JSON without '
                         'any re-simulation. Use after amending the rules.')
    args = ap.parse_args()

    if args.reanalyse:
        with open(args.reanalyse) as f:
            prev = json.load(f)
        print('=' * 100)
        print(f'  RE-ANALYSIS of {args.reanalyse} under the CURRENT decision rule')
        print('  (no re-simulation; rows are replayed exactly as recorded)')
        print('=' * 100)
        print(f"  rows: {len(prev['rows_per_seed'])}  seeds: {prev['seeds']}")
        print(f"  operating point: {prev['provenance']['resolved_base']}")
        again = analyse(prev['rows_per_seed'], prev['rates'])
        report(again)
        outp = args.reanalyse.replace('.json', '_reanalysed.json')
        with open(outp, 'w') as f:
            json.dump({**prev, 'schema': 'probe_rate_grid_v1_reanalysed',
                       'analysis': again}, f, indent=2)
        print(f'\\n  saved to {outp}')
        return 0"""),

    # B3 -- flatness test before acting on a boundary argmax
    ("""        # §4.4 boundary rule
        probed = [c['rate'] for c in cells]
        boundary = peak in (min(probed), max(probed))
        res.update(knee=knee, anchor=anchor, peak=peak, boundary_peak=boundary)
        if boundary:""",
     """        # §4.4 boundary rule, as AMENDED (A3): a boundary argmax only means the
        # RANGE is wrong if there is a genuine gradient. If relative queue value is
        # flat, the MODEL is wrong -- there is no peak to bracket.
        probed = [c['rate'] for c in cells]
        boundary = peak in (min(probed), max(probed))
        rels = [c['rel_spread'] for c in valid
                if c['rel_spread'] == c['rel_spread'] and c['rel_spread'] > 0]
        flat = bool(rels) and (max(rels) / min(rels) < 2.0)
        res.update(knee=knee, anchor=anchor, peak=peak, boundary_peak=boundary,
                   flat_response=flat,
                   rel_spread_ratio=(max(rels) / min(rels)) if rels else float('nan'))
        if boundary and flat:
            res.update(decision='NO PEAK -- FLAT RESPONSE; select by coverage',
                       grid=None,
                       note=f'relative queue value varies by only '
                            f'{max(rels)/min(rels):.2f}x across valid cells; the '
                            f'inverted-U model is refuted, do NOT extend the range '
                            f'(pre-reg §4.4 A3, §6.1)')
            out[sc] = res
            continue
        if boundary:"""),
]


def apply(text, edits, label):
    staged, ok = text, True
    for i, (old, new) in enumerate(edits, 1):
        n = staged.count(old)
        if n != 1:
            print(f"  [{label}] anchor {i}: matched {n} times, expected 1  <-- ABORT")
            ok = False
        else:
            print(f"  [{label}] anchor {i}: OK")
            staged = staged.replace(old, new, 1)
    return staged, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    out, allok = {}, True
    for rel, edits, label in [(DOC, EDITS_DOC, 'doc'), (SRC, EDITS_SRC, 'src')]:
        path = os.path.join(a.root, rel)
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found"); return 1
        text = io.open(path, encoding='utf-8').read()
        if GUARDS[rel] in text:
            print(f"  {rel}: ALREADY AMENDED, skipping"); continue
        staged, ok = apply(text, edits, label)
        allok &= ok
        out[path] = staged

    if not allok:
        print("\n  NO FILES WRITTEN."); return 1
    if not out:
        print("\n  Nothing to do."); return 0
    if a.dry_run:
        print(f"\n  DRY RUN OK -- {len(out)} file(s) would change. Nothing written.")
        return 0
    for path, staged in out.items():
        io.open(path, 'w', encoding='utf-8').write(staged)
        print(f"  WROTE {path}")
    print("\n  NEXT: re-run the selection on the EXISTING probe data (no re-simulation):")
    print("        python src\\probe_rate_grid.py --reanalyse results\\probe_rate_grid.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
