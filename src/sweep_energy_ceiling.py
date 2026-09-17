"""sweep_energy_ceiling.py -- what battery opens a congestion window at 1000 s?

THE BLOCKER THIS RESOLVES

At 1000 s (the dataset's target duration) the energy budget caps sustainable
load at ~6.6 per-flow, while congestion needs ~40-80. So 1000 s episodes and a
congestion regime are currently mutually exclusive, and no rate grid satisfies
both. That is the ONLY remaining blocker on dataset regeneration -- every other
flaw found is either fixed or one patch away.

The cause is calibration: INITIAL_ENERGY=100 with TX_COST=0.02 kills a node
after 5,000 transmits. A real UAV radio draws milliwatts against a watt-hour
battery, so that figure is low by orders of magnitude and raising it is more
defensible than keeping it.

WHAT THIS FINDS

For each candidate battery, at the TARGET duration, it locates:
  * whether ANY rate produces real congestion with negligible energy death
  * the usable band (congested, responsive, no dead nodes)
  * the smallest battery that opens a workable window

It reports the FULL loss breakdown so an energy-limited cell can never be
mistaken for a congested one -- the error that cost several days earlier.

DESIGNED TO RUN IN PARALLEL. Use --max_workers 4 and run alongside the other
pre-regeneration tests; four such jobs fit comfortably on 16 cores.

USAGE
    python src\\sweep_energy_ceiling.py --max_workers 4
    python src\\sweep_energy_ceiling.py --max_workers 4 --duration 1000 \\
        --energies 100 500 1000 2000 --rates 20 40 60 80
"""
import argparse, json, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator_v2 import FANETSimulatorV2, INITIAL_ENERGY, EnergyModel
from config_v2 import BASE, SCENARIOS, provenance

MIN_QOVF = 0.02      # congestion must be real
MAX_ENERGY = 0.05    # energy death must be negligible


def _run(job):
    sc, dur, rate, energy, seed, alt = job
    cfg = {**BASE, **SCENARIOS[sc], 'duration': dur,
           'z_min': alt[0], 'z_max': alt[1], 'packet_rate': rate,
           'seed': seed, 'actor': 'spbp', 'initial_energy': energy}
    s = FANETSimulatorV2(cfg)
    if abs(getattr(s, 'initial_energy', INITIAL_ENERGY) - energy) > 1e-9:
        raise RuntimeError(
            'initial_energy override had NO effect -- apply_energy_config_v14.py '
            'is not applied. This sweep would silently test the same battery '
            'at every setting.')
    m = s.run()
    ph = m.get('n_phantom_slots')
    if ph is None:
        raise RuntimeError('n_phantom_slots missing -- v12 not applied')
    if ph != 0:
        raise RuntimeError(f'{ph} phantom slot(s): leak present, results void')
    g = max(m['n_generated'], 1)
    d = m['drop_reasons']
    return {'scenario': sc, 'duration': dur, 'rate': rate, 'energy': energy,
            'seed': seed, 'pdr': m['network_pdr'],
            'generated': m['n_generated'], 'delivered': m['n_delivered'],
            'dead': sum(1 for e in s.energy if e <= 0.01), 'n_nodes': s.N,
            'q_ovf': d.get('queue_overflow', 0) / g,
            'energy_share': d.get('energy_depleted', 0) / g,
            'link': d.get('link_error', 0) / g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='dense_slow')
    ap.add_argument('--duration', type=float, default=1000.0)
    ap.add_argument('--energies', type=float, nargs='+',
                    default=[100., 250., 500., 1000., 2000.])
    ap.add_argument('--rates', type=float, nargs='+',
                    default=[10., 20., 40., 60., 80.])
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--z_min', type=float, default=100)
    ap.add_argument('--z_max', type=float, default=300)
    ap.add_argument('--max_workers', type=int, default=4)
    ap.add_argument('--out', default='results/energy_ceiling.json')
    args = ap.parse_args()

    alt = (args.z_min, args.z_max)
    seeds = list(range(1, args.seeds + 1))
    cfg = SCENARIOS[args.scenario]
    nflows = max(1, cfg['num_drones'] // 4)

    print('=' * 96)
    print('  ENERGY-CEILING SWEEP -- what battery opens congestion at the target duration?')
    print('=' * 96)
    print(f"  scenario {args.scenario}  N={cfg['num_drones']}  flows={nflows}  "
          f"duration {args.duration}s  alt {alt[0]:.0f}-{alt[1]:.0f}m")
    print(f"  current INITIAL_ENERGY = {INITIAL_ENERGY}  "
          f"(node dies after {INITIAL_ENERGY/EnergyModel.TX_COST:.0f} transmits)")
    print(f"  batteries tested: {args.energies}")
    print(f"  usable cell = q_ovf >= {MIN_QOVF}, energy_share <= {MAX_ENERGY}, dead == 0")

    jobs = [(args.scenario, args.duration, r, e, s, alt)
            for e in args.energies for r in args.rates for s in seeds]
    print(f"\n  {len(jobs)} episodes")
    t0, rows, done = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result()); done += 1
            if done % max(1, len(jobs) // 10) == 0:
                el = time.time() - t0
                print(f'    {done}/{len(jobs)}  ({el:.0f}s, '
                      f'~{el/done*(len(jobs)-done):.0f}s left)')

    agg = defaultdict(list)
    for r in rows:
        agg[(r['energy'], r['rate'])].append(r)

    print(f"\n  {'battery':>9}{'rate':>6}{'offer':>7}{'pdr':>7}{'deliv':>8}"
          f"{'q_ovf':>8}{'energy':>8}{'link':>7}{'dead':>6}  verdict")
    print('  ' + '-' * 82)
    grid, usable_by_energy = [], defaultdict(list)
    for e in args.energies:
        for r in args.rates:
            g = agg[(e, r)]
            if not g:
                continue
            def mean(k): return sum(x[k] for x in g) / len(g)
            cell = dict(energy=e, rate=r, offered=nflows * r, pdr=mean('pdr'),
                        delivered=mean('delivered'), q_ovf=mean('q_ovf'),
                        energy_share=mean('energy_share'), link=mean('link'),
                        dead=mean('dead'))
            cell['usable'] = bool(cell['q_ovf'] >= MIN_QOVF
                                  and cell['energy_share'] <= MAX_ENERGY
                                  and cell['dead'] == 0)
            grid.append(cell)
            if cell['usable']:
                usable_by_energy[e].append(r)
            v = ('USABLE' if cell['usable'] else
                 ('energy-limited' if cell['energy_share'] > MAX_ENERGY or cell['dead'] > 0
                  else 'no congestion'))
            print(f"  {e:>9.0f}{r:>6.0f}{cell['offered']:>7.0f}{cell['pdr']:>7.3f}"
                  f"{cell['delivered']:>8.0f}{cell['q_ovf']:>8.3f}"
                  f"{cell['energy_share']:>8.3f}{cell['link']:>7.3f}"
                  f"{cell['dead']:>6.1f}  {v}")

    print('\n' + '=' * 96)
    print('  VERDICT')
    print('=' * 96)
    opened = [e for e in args.energies if usable_by_energy[e]]
    if not opened:
        print(f"  NO battery tested opens a congestion window at {args.duration}s.")
        print("  Either extend --energies upward, or the target duration itself")
        print("  must change. Do NOT start regeneration until one of those resolves.")
        smallest = None
    else:
        smallest = min(opened)
        print(f"  Smallest battery that opens a window: INITIAL_ENERGY = {smallest:.0f}")
        print(f"  ({smallest/INITIAL_ENERGY:.1f}x the current {INITIAL_ENERGY:.0f})")
        for e in opened:
            print(f"     battery {e:>7.0f}  usable rates: "
                  f"{[int(r) for r in usable_by_energy[e]]}")
        print("\n  NOTE: raising INITIAL_ENERGY changes a simulator constant, so")
        print("  G1/G2 must be re-gated afterwards and the change recorded. It does")
        print("  NOT invalidate the v10/v11/v12 fixes, which are independent.")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'schema': 'energy_ceiling_v1', 'scenario': args.scenario,
                   'duration': args.duration, 'alt': list(alt),
                   'energies': args.energies, 'rates': args.rates,
                   'seeds': seeds, 'current_initial_energy': INITIAL_ENERGY,
                   'smallest_opening_battery': smallest,
                   'usable_by_energy': {str(k): v for k, v in usable_by_energy.items()},
                   'provenance': provenance(), 'grid': grid, 'rows': rows}, f, indent=2)
    print(f'\n  saved to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
