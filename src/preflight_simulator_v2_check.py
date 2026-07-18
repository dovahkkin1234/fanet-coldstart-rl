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
    c6 = (r1['n_delivered'] == r2['n_delivered'] and
          r1['n_dropped'] == r2['n_dropped'] and
          abs(r1['network_pdr'] - r2['network_pdr']) < 1e-12)

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
        ("6. Bit-reproducible (fixed seed)",
         c6, f"PDR {r1['network_pdr']:.6f} == {r2['network_pdr']:.6f}"),
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
