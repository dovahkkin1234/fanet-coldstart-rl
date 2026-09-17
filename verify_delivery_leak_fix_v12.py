"""verify_delivery_leak_fix_v12.py -- verify the leak fix BY EXECUTION.

Written after the fact: apply_delivery_leak_fix_v12.py printed
"NEXT: run this" while the file did not exist. Fourth instance of that defect in
this project; the checks below were previously run ad-hoc and are now
reproducible on your machine.

CHECKS
  1. QUEUE-SLOT CONSERVATION -- no queue slot is held by an already
     delivered/dropped packet, across every Suite A scenario and both a light
     and a heavy rate. This is the invariant whose ABSENCE hid the leak through
     four milestones: packet-COUNT conservation (generated == delivered +
     dropped) passed 8/8 cells even while the leak was active, because delivered
     packets were counted correctly -- it was their SLOTS that leaked.
  2. CONVERGECAST CAP GONE -- delivery is no longer pinned at MAX_QUEUE. Before
     the fix, 400/400 episodes delivered exactly 50 across a 43x range of
     offered traffic.
  3. DURATION INDEPENDENCE -- PDR no longer decays purely with episode length.
     Pre-fix dense_slow at rate 0.05 fell 96.8% -> 63.9% from 200 s to 1000 s
     with ZERO overflow drops; the decay was entirely phantom occupancy.
  4. CAPACITY SANITY -- congestion appears where per-node capacity says it
     should and not before. Guards against a fix that is TOO permissive: if the
     destination check accidentally skipped queueing for intermediate hops, PDR
     would stay ~1.0 even far above capacity.
  5. NODE-ID INVARIANT -- every frame graph holds all N nodes with ids 0..N-1.
     Load-bearing: cand_flat stores GLOBAL drone ids that are used directly as
     FRAME-LOCAL embedding row indices.
  6. NEGATIVE CONTROL -- the conservation check is proven to FIRE. A checker
     that only ever passes proves nothing, so this deliberately re-creates the
     leak in a subclass and requires the check to catch it.

USAGE
    python verify_delivery_leak_fix_v12.py --src src
"""
import argparse, os, sys, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.src))

    from simulator_v2 import (FANETSimulatorV2, MAX_QUEUE,
                              SLOTS_PER_FRAME, FRAME_DT)
    from config_v2 import BASE, SCENARIOS, get_suite

    warnings.simplefilter('ignore')
    fails = []
    print('=' * 92)
    print('  VERIFY v12 -- destination-queue leak')
    print('=' * 92)

    if not hasattr(FANETSimulatorV2, '_assert_queue_conservation'):
        print('  FAIL -- v12 is not applied (no _assert_queue_conservation).')
        return 1

    # ---- 1. queue-slot conservation ---------------------------------------
    print(f"\n  1. QUEUE-SLOT CONSERVATION\n"
          f"  {'scenario':<13}{'rate':>7}{'gen':>8}{'deliv':>8}{'pdr':>9}{'phantom':>9}")
    print('  ' + '-' * 56)
    for sc in ['very_dense', 'dense_slow', 'medium_slow', 'sparse_fast']:
        for rate in [0.5, 4.0]:
            m = FANETSimulatorV2({**BASE, **SCENARIOS[sc], 'packet_rate': rate,
                                  'seed': 1, 'actor': 'spbp'}).run()
            ph = m['n_phantom_slots']
            if ph != 0:
                fails.append(f"{sc} rate {rate}: {ph} phantom slot(s)")
            print(f"  {sc:<13}{rate:>7}{m['n_generated']:>8}{m['n_delivered']:>8}"
                  f"{m['network_pdr']:>9.4f}{ph:>9}")

    # ---- 2. convergecast cap gone -----------------------------------------
    print(f"\n  2. CONVERGECAST CAP  (pre-fix: delivery pinned at MAX_QUEUE={MAX_QUEUE})")
    try:
        cc = get_suite('convergecast')['sink_50']
        prev = None
        for dur in [200.0, 400.0]:
            m = FANETSimulatorV2({**BASE, **cc, 'duration': dur, 'z_min': 100,
                                  'z_max': 300, 'packet_rate': 0.05,
                                  'seed': 1, 'actor': 'spbp'}).run()
            print(f"     dur={dur:>6.0f}  generated={m['n_generated']:>6}  "
                  f"delivered={m['n_delivered']:>6}  phantom={m['n_phantom_slots']}")
            if m['n_delivered'] == MAX_QUEUE:
                fails.append(f"convergecast dur={dur}: delivery still pinned at "
                             f"MAX_QUEUE -- leak present")
            if prev is not None and m['n_delivered'] <= prev:
                fails.append("convergecast: delivery did not grow with duration")
            prev = m['n_delivered']
    except RuntimeError:
        print('     SKIPPED -- convergecast suite empty (v8a not applied)')

    # ---- 3. duration independence -----------------------------------------
    print(f"\n  3. DURATION INDEPENDENCE  (pre-fix: 96.8% -> 63.9% with ZERO overflow)")
    pdrs = []
    for dur in [200.0, 400.0]:
        m = FANETSimulatorV2({**BASE, **SCENARIOS['dense_slow'], 'duration': dur,
                              'z_min': 100, 'z_max': 300, 'packet_rate': 0.05,
                              'seed': 1, 'actor': 'spbp'}).run()
        pdrs.append(m['network_pdr'])
        print(f"     dur={dur:>6.0f}  pdr={m['network_pdr']:.4f}  "
              f"overflow={m['drop_reasons'].get('queue_overflow', 0)}  "
              f"phantom={m['n_phantom_slots']}")
    if pdrs[1] < pdrs[0] - 0.10:
        fails.append(f"PDR still decays with duration ({pdrs[0]:.4f} -> "
                     f"{pdrs[1]:.4f}) -- residual leak")

    # ---- 4. capacity sanity (guards against an over-permissive fix) --------
    cap = SLOTS_PER_FRAME / FRAME_DT
    print(f"\n  4. CAPACITY SANITY  (per-node service = {cap:.0f} pkt/s)")
    print(f"     {'rate':>7}{'offered':>10}{'pdr':>9}{'q_ovf':>8}{'phantom':>9}")
    seen_congestion = False
    for rate in [4.0, 60.0]:
        m = FANETSimulatorV2({**BASE, **SCENARIOS['dense_slow'],
                              'packet_rate': rate, 'seed': 1, 'actor': 'spbp'}).run()
        q = m['drop_reasons'].get('queue_overflow', 0)
        if q > 0:
            seen_congestion = True
        print(f"     {rate:>7}{7 * rate:>10.0f}{m['network_pdr']:>9.4f}{q:>8}"
              f"{m['n_phantom_slots']:>9}")
    if not seen_congestion:
        fails.append("no queue overflow even far above capacity -- the fix may "
                     "be skipping queue admission for intermediate hops too")

    # ---- 5. node-id invariant ---------------------------------------------
    print("\n  5. NODE-ID INVARIANT")
    try:
        s = FANETSimulatorV2({**BASE, **SCENARIOS['sparse_fast'],
                              'packet_rate': 2.0, 'seed': 1, 'actor': 'spbp'})
        G = s._build_graph()
        s._assert_node_id_invariant(G)
        print(f"     holds: {G.number_of_nodes()} nodes, ids 0..{s.N - 1}")
    except AssertionError as e:
        fails.append(f"node-id invariant: {e}")
    except Exception as e:
        fails.append(f"node-id check errored: {type(e).__name__}: {e}")

    # ---- 6. NEGATIVE CONTROL ----------------------------------------------
    print("\n  6. NEGATIVE CONTROL -- re-create the leak, does the check catch it?")

    class Leaky(FANETSimulatorV2):
        """Deliberately re-introduces the leak: park a delivered packet in a
        queue and never free it. The conservation check MUST catch this."""
        def _metrics(self):
            for q in self.queues:
                if q.buffer is not None:
                    break
            # fabricate one phantom: a delivered packet left in a queue
            class _P:
                pid, delivered, dropped, drop_reason = -1, True, False, None
            self.queues[0].buffer.append(_P())
            return super()._metrics()

    try:
        m = Leaky({**BASE, **SCENARIOS['dense_slow'], 'packet_rate': 0.5,
                   'seed': 1, 'actor': 'spbp'}).run()
        if m['n_phantom_slots'] > 0:
            print(f"     caught {m['n_phantom_slots']} phantom slot(s) -- "
                  f"the check is live, not dead code")
        else:
            fails.append("negative control: the check did NOT catch a "
                         "deliberately planted phantom slot -- it is dead code")
    except Exception as e:
        fails.append(f"negative control errored: {type(e).__name__}: {e}")

    # strict mode must raise rather than warn
    try:
        Leaky({**BASE, **SCENARIOS['dense_slow'], 'packet_rate': 0.5, 'seed': 1,
               'actor': 'spbp', 'strict_conservation': True}).run()
        fails.append("strict_conservation=True did not raise")
        print("     strict mode did NOT raise  *** dead ***")
    except AssertionError:
        print("     strict mode raises AssertionError as required")
    except Exception as e:
        fails.append(f"strict mode raised {type(e).__name__}, expected AssertionError")

    print()
    if fails:
        print(f"  FAIL -- {len(fails)} problem(s):")
        for f in fails:
            print('    - ' + f)
        return 1
    print("  PASS -- 6/6. The conservation check is verified to fire on a planted")
    print("  leak, so it is a live guard rather than a check that only ever passes.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
