"""
preflight_simulator_v2_check.py  —  GATE G2 for Approach 2 (M2).

Validates that simulator_v2 actually exhibits the multi-packet congestion
dynamics the whole Approach-2 thesis depends on. Same PASS/FAIL discipline as
the G1 interference gate.

THE SIX CHECKS (docs/M2_SIMULATOR_V2_DESIGN.md §10):
  1. Queues nonzero and load-dependent    (Approach 1 had flat ~0 occupancy)
  2. Network PDR degrades smoothly with load (graded curve, no cliff/flat line)
  3. Drop taxonomy shifts to congestion causes under load
  4. Realized node activity lands in the graded band G1 validated (~0-0.2)
  5. Interference-OFF degenerate mode differs from ON (coupling is real)
  6. Bit-reproducible under a fixed seed

PLUS a delay-sanity diagnostic (ARQ calibration check): mean retransmission
attempts and per-hop delay must stay physically reasonable, confirming
max_retx=5 is not inflating latency absurdly.

Usage:
    python src\\preflight_simulator_v2_check.py
    python src\\preflight_simulator_v2_check.py --num_drones 30 --duration 40
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator_v2 import FANETSimulatorV2, ACT_BETA, ACT_ALPHA, ACT_MAX, DEFAULT_MAX_RETX

CONGESTION_CAUSES = ('queue_overflow', 'link_error')

# EXPLICIT REGRESSION ANCHOR.
# Check 6 previously compared two runs of the SAME build to each other, which
# catches nondeterminism but NOT drift: if a change altered the physics, both
# runs would drift identically and the check would still pass. The constant was
# therefore tracked only in notes, outside the code. It is now asserted here.
#
# ANCHOR ON THE INTEGER COUNTS, NOT THE PDR FLOAT.
# The first version of this anchor stored PDR as a float with a 1e-9 tolerance.
# That failed immediately in practice: PDR is n_delivered/n_generated = 87/280 =
# 0.31071428571..., the harness displayed it rounded to 0.310714, and the
# rounded value copied back in differed from the true one by 2.9e-7 -- far
# outside the tolerance. The two values printed IDENTICALLY in the failure
# message, which made the report actively confusing.
#
# Integers are exact, unambiguous, and safe to copy by hand. There is no
# tolerance to tune and no rounding to get wrong.
#
# Leave as None to record new values (the harness prints exactly what to paste).
# Update ONLY alongside a deliberate, documented physics change.
#
# THE ANCHOR RUNS ARE ACTOR-PINNED. They previously used `args.actor`, so
# recording the anchor under one actor and re-running the gate under another
# reported drift that was not drift -- and the message did not say why. The
# anchors now ignore --actor entirely. Determinism (r1 vs r2) still runs under
# --actor, because that is a property of the build, not of the anchor.
REGRESSION_ACTOR = 'dijkstra'
REGRESSION_DELIVERED = 87
REGRESSION_GENERATED = 280

# SECOND ANCHOR, LINK-QUALITY-SENSITIVE.
# 'dijkstra' routes on hop count and ignores link_quality completely, so the
# primary anchor above CANNOT detect a change to the link model. The M-4
# collision-model flip is precisely such a change, and this gate would have
# slept through it. 'spbp' scores on link_quality AND queue occupancy, and is
# the oracle teacher the M3.5 labels come from, so drift here is drift in the
# thing every downstream milestone depends on.
# Two integers. No new machinery. Closes the second-anchor item.
#
# THIS IS NOT HYPOTHETICAL. Measured at this gate's default config:
#     dijkstra : 87/280 saturated  ->  87/280 unsaturated   (blind)
#     spbp     : 111/280 saturated -> 112/280 unsaturated   (sees it)
# The M-4 collision-model flip -- a deliberate, documented physics change --
# moves the dijkstra anchor by EXACTLY ZERO. Of the two anchors, only the
# link-quality-sensitive one can detect the change that was just made.
REGRESSION_LQ_ACTOR = 'spbp'
REGRESSION_LQ_DELIVERED = 112
REGRESSION_LQ_GENERATED = 280


def _anchor_block(label, res, delivered, generated, actor, fname):
    """Compare one run against one recorded pair. Returns (ok, armed).

    Unset => ok=True, armed=False. A gate that cannot pass on a fresh checkout
    is useless, so bootstrapping must not FAIL -- but an unarmed gate must
    never LOOK like a passing one, hence the banner and the armed flag that
    reaches the verdict line.
    """
    if delivered is None or generated is None:
        print(f"\n    ┌─ [{label} ANCHOR NOT SET] — check 6 is NOT protecting "
              f"against drift ─┐")
        print(f"    │ Copy these lines verbatim into {fname}:")
        print(f"    │     {label}_DELIVERED = {res['n_delivered']}")
        print(f"    │     {label}_GENERATED = {res['n_generated']}")
        print(f"    │ (actor={actor}; PDR {res['n_delivered']}/"
              f"{res['n_generated']} = {res['network_pdr']:.6f}, shown for")
        print(f"    │  reference only -- the anchor is the integer counts, so "
              f"there is")
        print(f"    │  no rounding to get wrong.)")
        print(f"    └{'─' * 68}┘")
        return True, False
    ok = (res['n_delivered'] == delivered and res['n_generated'] == generated)
    if not ok:
        print(f"\n    *** REGRESSION [{label}, actor={actor}]: "
              f"delivered/generated = {res['n_delivered']}/{res['n_generated']}"
              f", anchor = {delivered}/{generated} ***")
        print("    The simulator's behaviour changed. If intentional (a")
        print("    documented physics change), update BOTH pairs and say so in")
        print("    the commit. If not, something drifted silently.")
        if label == 'REGRESSION_LQ':
            print("    This anchor reads link_quality; the dijkstra anchor does")
            print("    not. If only this one moved, the change is in the LINK")
            print("    MODEL, not in topology or queueing.")
    return ok, True


def run(cfg):
    return FANETSimulatorV2(cfg).run()


def congestion_share(drops):
    tot = sum(drops.values())
    if tot == 0:
        return 0.0
    cong = sum(v for k, v in drops.items() if k in CONGESTION_CAUSES)
    return cong / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_drones', type=int, default=30)
    ap.add_argument('--area', type=float, default=1000.0)
    ap.add_argument('--comm_range', type=float, default=280.0)
    ap.add_argument('--duration', type=float, default=40.0)
    ap.add_argument('--speed_min', type=float, default=5.0)
    ap.add_argument('--speed_max', type=float, default=15.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rates', type=float, nargs='+',
                    default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument('--actor', default='dijkstra')
    args = ap.parse_args()

    base = dict(num_drones=args.num_drones, area_x=args.area, area_y=args.area,
                z_min=50, z_max=150, comm_range=args.comm_range,
                duration=args.duration, speed_min=args.speed_min,
                speed_max=args.speed_max, pause_max=5.0, seed=args.seed)

    print("\n" + "=" * 78)
    print("  GATE G2 — MULTI-PACKET CONGESTION DYNAMICS ARE REAL?")
    print("=" * 78)
    print(f"  {args.num_drones} drones, {args.area:.0f}x{args.area:.0f} m, "
          f"comm_range={args.comm_range:.0f} m, duration={args.duration:.0f}s")
    print(f"  actor={args.actor}, seed={args.seed}")
    print(f"  locked calibration: ACT_ALPHA={ACT_ALPHA} ACT_BETA={ACT_BETA} "
          f"ACT_MAX={ACT_MAX} max_retx={DEFAULT_MAX_RETX}")

    # ── Load sweep (interference ON) ─────────────────────────────────────────
    rows = []
    for pr in args.rates:
        m = run({**base, 'packet_rate': pr, 'interference_on': True,
                 'actor': args.actor})
        rows.append((pr, m))

    print("\n" + "-" * 78)
    print("  LOAD SWEEP (interference ON)")
    print("-" * 78)
    print(f"  {'rate':>6}{'PDR':>8}{'meanOcc':>9}{'maxOcc':>8}{'activity':>10}"
          f"{'cong%':>8}{'attempts':>10}{'ms/hop':>8}")
    for pr, m in rows:
        print(f"  {pr:>6.2f}{m['network_pdr']:>8.3f}{m['mean_queue_occ']:>9.3f}"
              f"{m['max_queue_occ']:>8.2f}{m['mean_activity']:>10.3f}"
              f"{100*congestion_share(m['drop_reasons']):>7.1f}%"
              f"{m['mean_tx_attempts']:>10.2f}{m['mean_delay_per_hop_ms']:>8.2f}")

    print("\n  drop taxonomy by load:")
    for pr, m in rows:
        print(f"    rate={pr:<5.2f} {m['drop_reasons']}")

    # ── CHECK 1: queues nonzero and load-dependent ───────────────────────────
    occs = [m['mean_queue_occ'] for _, m in rows]
    max_occs = [m['max_queue_occ'] for _, m in rows]
    c1 = (max(occs) > 0.01) and (occs[-1] > occs[0]) and (max(max_occs) > 0.5)

    # ── CHECK 2: PDR degrades smoothly with load ─────────────────────────────
    pdrs = [m['network_pdr'] for _, m in rows]
    monotone = all(pdrs[i] >= pdrs[i + 1] - 0.02 for i in range(len(pdrs) - 1))
    graded = (pdrs[0] - pdrs[-1]) > 0.05 and pdrs[0] > 0.05
    c2 = monotone and graded

    # ── CHECK 3: drop taxonomy shifts toward congestion causes ───────────────
    cong_lo = congestion_share(rows[0][1]['drop_reasons'])
    cong_hi = congestion_share(rows[-1][1]['drop_reasons'])
    c3 = (cong_hi > cong_lo) and (cong_hi > 0.40)

    # ── CHECK 4: realized activity in the graded band G1 validated ───────────
    acts = [m['mean_activity'] for _, m in rows]
    c4 = all(0.0 < a < 0.30 for a in acts) and acts[-1] > acts[0]

    # ── CHECK 5: interference OFF vs ON differ materially ────────────────────
    mid = args.rates[len(args.rates) // 2]
    m_on = run({**base, 'packet_rate': mid, 'interference_on': True,
                'actor': args.actor})
    m_off = run({**base, 'packet_rate': mid, 'interference_on': False,
                 'actor': args.actor})
    le_on = m_on['drop_reasons'].get('link_error', 0)
    le_off = m_off['drop_reasons'].get('link_error', 0)
    c5 = (le_off == 0) and (le_on > 0) and (m_off['network_pdr'] > m_on['network_pdr'])

    print("\n" + "-" * 78)
    print("  INTERFERENCE ON vs OFF  (rate=%.2f)" % mid)
    print("-" * 78)
    print(f"    ON : PDR={m_on['network_pdr']:.3f}  link_error={le_on}  drops={m_on['drop_reasons']}")
    print(f"    OFF: PDR={m_off['network_pdr']:.3f}  link_error={le_off}  drops={m_off['drop_reasons']}")

    # ── CHECK 6: reproducibility ─────────────────────────────────────────────
    r1 = run({**base, 'packet_rate': mid, 'interference_on': True, 'actor': args.actor})
    r2 = run({**base, 'packet_rate': mid, 'interference_on': True, 'actor': args.actor})
    determinism = (r1['n_delivered'] == r2['n_delivered'] and
                   r1['n_dropped'] == r2['n_dropped'] and
                   abs(r1['network_pdr'] - r2['network_pdr']) < 1e-12)

    # ---- drift, against two ACTOR-PINNED anchors ----
    # Pinned so that --actor cannot make a matching build look like a
    # regression. r1 is reused only when its actor already equals the pinned
    # one, which is the default path and saves a redundant episode.
    _fname = os.path.basename(__file__)
    ra = (r1 if args.actor == REGRESSION_ACTOR else
          run({**base, 'packet_rate': mid, 'interference_on': True,
               'actor': REGRESSION_ACTOR}))
    ok_a, armed_a = _anchor_block('REGRESSION', ra, REGRESSION_DELIVERED,
                                  REGRESSION_GENERATED, REGRESSION_ACTOR, _fname)
    rb = run({**base, 'packet_rate': mid, 'interference_on': True,
              'actor': REGRESSION_LQ_ACTOR})
    ok_b, armed_b = _anchor_block('REGRESSION_LQ', rb, REGRESSION_LQ_DELIVERED,
                                  REGRESSION_LQ_GENERATED, REGRESSION_LQ_ACTOR,
                                  _fname)
    anchored = ok_a and ok_b
    fully_armed = armed_a and armed_b
    c6 = determinism and anchored

    # ── DELAY SANITY (ARQ calibration diagnostic) ────────────────────────────
    attempts = [m['mean_tx_attempts'] for _, m in rows]
    per_hop = [m['mean_delay_per_hop_ms'] for _, m in rows
               if not np.isnan(m['mean_delay_per_hop_ms'])]
    delay_ok = (all(1.0 <= a <= DEFAULT_MAX_RETX + 1 for a in attempts) and
                all(p <= 2.0 * (DEFAULT_MAX_RETX + 1) for p in per_hop))

    # ── VERDICT ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    checks = [
        ("1. Queues nonzero & load-dependent",
         c1, f"meanOcc {occs[0]:.3f}->{occs[-1]:.3f}, peak max {max(max_occs):.2f}"),
        ("2. PDR degrades smoothly with load",
         c2, f"PDR {pdrs[0]:.3f}->{pdrs[-1]:.3f}, monotone={monotone}"),
        ("3. Drop taxonomy shifts to congestion",
         c3, f"congestion share {100*cong_lo:.1f}% -> {100*cong_hi:.1f}%"),
        ("4. Activity in graded band (0-0.30)",
         c4, f"activity {acts[0]:.3f}->{acts[-1]:.3f}"),
        ("5. Interference ON/OFF differ",
         c5, f"link_error {le_off}(off) vs {le_on}(on); PDR {m_off['network_pdr']:.3f} vs {m_on['network_pdr']:.3f}"),
        ("6. Bit-reproducible + no drift vs anchors",
         c6, f"determinism={determinism}; "
             f"{REGRESSION_ACTOR}={ra['n_delivered']}/{ra['n_generated']} "
             + ('(**UNSET**)' if not armed_a else 'vs anchor OK' if ok_a else 'DRIFT')
             + f"; {REGRESSION_LQ_ACTOR}={rb['n_delivered']}/{rb['n_generated']} "
             + ('(**UNSET**)' if not armed_b else 'vs anchor OK' if ok_b else 'DRIFT')
             + ('' if fully_armed else '  <-- NOT PROTECTING AGAINST DRIFT')),
    ]
    for name, ok, detail in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:<40} {detail}")
    print(f"    [{'OK  ' if delay_ok else 'WARN'}] {'ARQ delay sanity (diagnostic)':<40} "
          f"attempts {min(attempts):.2f}-{max(attempts):.2f}/hop, "
          f"{min(per_hop):.2f}-{max(per_hop):.2f} ms/hop")

    passed = all(c for _, c, _ in checks)
    print()
    if passed:
        print("    G2 PASS — multi-packet congestion dynamics are real and graded.")
        print("    Queues build under load, PDR degrades smoothly, drops shift to")
        print("    congestion causes, and the interference coupling is verified.")
        print("    The environment is usable for M3 (teachers) and beyond. PROCEED.")
    else:
        print("    G2 FAIL — do NOT build M3+ on this simulator yet.")
        print("    Inspect the failing checks above before proceeding.")
    print("=" * 78 + "\n")
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
