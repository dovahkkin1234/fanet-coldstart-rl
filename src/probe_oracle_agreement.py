"""probe_oracle_agreement.py -- do the three per-cell oracles conflict?

THE QUESTION THIS DECIDES

The extended panel assigns three different oracles across nine cells:

    medium_slow @40/60/80  -> dijkstra
    dense_slow  @60        -> da_gpsr
    dense_slow  @80/100    -> gpsr
    very_dense  @60/80     -> da_gpsr
    sink_50     @30        -> da_gpsr

Behaviour cloning trains ONE network on all of them. If two cells present
near-identical local observations but their oracles choose different next hops,
the network receives contradictory supervision and learns an average of both --
which can be worse than either teacher alone. That failure is silent: training
loss simply plateaus higher, with nothing pointing at the cause.

This measures the risk directly, before the dataset is built.

WHAT IT MEASURES

  1. RAW AGREEMENT -- at every decision point, all three oracles are queried on
     the SAME graph, current node and destination. How often do they pick the
     same next hop? High agreement => the multi-teacher question is largely
     moot and per-cell labelling is safe.

  2. CONTEXT SEPARABILITY -- the observable local state at each decision
     (degree, queue occupancy, neighbour count). If the scenarios occupy
     distinct regions of this space, the network can condition on it and
     resolve the conflict itself. `degree` is already one of the nine node
     features, and local degree is what physically distinguishes a dense swarm
     from a sparse one, so this is the feature most likely to carry the signal.

  3. OVERRIDE RATE -- simulator_v2 applies a loop-avoidance override AFTER the
     actor returns, and its own comment notes the override "picks the first
     unvisited neighbour rather than re-scoring with the teacher's own rule, so
     it is arbitrary" and "does not affect teachers equally". Any decision where
     the override fires is one where the recorded action is NOT the teacher's
     choice -- directly relevant to the dataset fix that records the true
     action taken.

READ THE RESULT AS

  agreement >= ~0.85  -> per-cell labelling is safe as-is; conflicts are rare
  agreement ~0.5-0.85 -> add explicit context features before cloning
  agreement <  ~0.5   -> per-cell teachers genuinely conflict; prefer a single
                         global teacher, or condition the policy explicitly

USAGE
    python src\\probe_oracle_agreement.py --max_workers 4
"""
import argparse, json, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulator_v2 as S
from simulator_v2 import FANETSimulatorV2, TEACHERS
from config_v2 import BASE, SCENARIOS, get_suite, provenance

ORACLES = ['dijkstra', 'gpsr', 'da_gpsr']

# the assignment from panel_extended_v3
CELLS = {
    ('dense_slow', 60.0):  ('suiteA', 'da_gpsr'),
    ('dense_slow', 80.0):  ('suiteA', 'gpsr'),
    ('dense_slow', 100.0): ('suiteA', 'gpsr'),
    ('very_dense', 60.0):  ('suiteA', 'da_gpsr'),
    ('very_dense', 80.0):  ('suiteA', 'da_gpsr'),
    ('medium_slow', 40.0): ('suiteA', 'dijkstra'),
    ('medium_slow', 60.0): ('suiteA', 'dijkstra'),
    ('medium_slow', 80.0): ('suiteA', 'dijkstra'),
    ('sink_50', 30.0):     ('convergecast', 'da_gpsr'),
}
DURATION = 1000.0
INITIAL_ENERGY = 8000.0
MAX_LOG = 20000          # cap decisions logged per cell


def scenario_cfg(suite, name):
    return SCENARIOS[name] if suite == 'suiteA' else get_suite('convergecast')[name]


def run_cell(job):
    (scen, rate), (suite, cell_oracle), seed = job
    log = []

    def probe(G, current, destination, **kw):
        """Drives with the cell's own oracle, and records what every oracle
        would have chosen at this identical decision point."""
        choices = {}
        for o in ORACLES:
            try:
                choices[o] = TEACHERS[o](G, current, destination)
            except Exception:
                choices[o] = None
        if len(log) < MAX_LOG:
            nbrs = list(G.neighbors(current))
            log.append({
                'choices': choices,
                'degree': len(nbrs),
                'occ': float(G.nodes[current].get('queue_occupancy', 0.0)),
                'n_cand': len(nbrs),
            })
        return choices[cell_oracle]

    # register under a neutral name so the dispatcher takes the plain branch
    # (not the 'random' rng branch, nor the 'backpressure' tuple branch)
    TEACHERS['__probe__'] = probe
    cfg = {**BASE, **scenario_cfg(suite, scen), 'duration': DURATION,
           'initial_energy': INITIAL_ENERGY, 'packet_rate': rate,
           'seed': seed, 'actor': '__probe__'}
    sim = FANETSimulatorV2(cfg)
    m = sim.run()
    n_override = getattr(sim, 'n_loop_override', None)

    # ---- agreement statistics ----
    pair, allthree, n = defaultdict(int), 0, len(log)
    degs, occs = [], []
    for row in log:
        c = row['choices']
        if c['dijkstra'] == c['gpsr'] == c['da_gpsr'] and c['dijkstra'] is not None:
            allthree += 1
        for a in range(len(ORACLES)):
            for b in range(a + 1, len(ORACLES)):
                x, y = ORACLES[a], ORACLES[b]
                if c[x] == c[y] and c[x] is not None:
                    pair[f'{x}|{y}'] += 1
        degs.append(row['degree']); occs.append(row['occ'])

    return {'scenario': scen, 'rate': rate, 'seed': seed,
            'cell_oracle': cell_oracle, 'n_decisions': n,
            'agree_all3': allthree / max(n, 1),
            'agree_pair': {k: v / max(n, 1) for k, v in pair.items()},
            'mean_degree': sum(degs) / max(len(degs), 1),
            'mean_occ': sum(occs) / max(len(occs), 1),
            'deg_min': min(degs) if degs else 0, 'deg_max': max(degs) if degs else 0,
            'pdr': m['network_pdr'], 'n_loop_override': n_override}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--max_workers', type=int, default=4)
    ap.add_argument('--out', default='results/oracle_agreement.json')
    args = ap.parse_args()

    jobs = [(k, v, s) for k, v in CELLS.items()
            for s in range(1, args.seeds + 1)]
    print('=' * 92)
    print('  ORACLE AGREEMENT PROBE -- do the per-cell teachers conflict?')
    print('=' * 92)
    print(f'  oracles: {ORACLES}')
    print(f'  cells: {len(CELLS)}  seeds: {args.seeds}  episodes: {len(jobs)}')
    print(f'  (all three oracles queried at every decision point on the SAME state)')

    t0, rows, done = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(run_cell, j) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result()); done += 1
            el = time.time() - t0
            print(f'    {done}/{len(jobs)}  ({el:.0f}s)', flush=True)

    print('\n' + '=' * 92)
    print('  PER-CELL AGREEMENT')
    print('=' * 92)
    print(f"  {'cell':<22}{'oracle':<11}{'all3':>7}{'dij|gps':>9}{'dij|dag':>9}"
          f"{'gps|dag':>9}{'deg':>7}{'n':>8}")
    rows.sort(key=lambda r: (r['scenario'], r['rate']))
    for r in rows:
        p = r['agree_pair']
        print(f"  {r['scenario']+'@'+str(int(r['rate'])):<22}{r['cell_oracle']:<11}"
              f"{r['agree_all3']:>7.3f}{p.get('dijkstra|gpsr',0):>9.3f}"
              f"{p.get('dijkstra|da_gpsr',0):>9.3f}{p.get('gpsr|da_gpsr',0):>9.3f}"
              f"{r['mean_degree']:>7.1f}{r['n_decisions']:>8}")

    overall = sum(r['agree_all3'] * r['n_decisions'] for r in rows) / \
              max(sum(r['n_decisions'] for r in rows), 1)

    print('\n' + '=' * 92)
    print('  CONTEXT SEPARABILITY -- can local degree distinguish the regimes?')
    print('=' * 92)
    byscen = defaultdict(list)
    for r in rows:
        byscen[r['scenario']].append(r)
    for scen, rs in sorted(byscen.items()):
        oracles = sorted({x['cell_oracle'] for x in rs})
        lo = min(x['deg_min'] for x in rs); hi = max(x['deg_max'] for x in rs)
        mean = sum(x['mean_degree'] for x in rs) / len(rs)
        print(f'  {scen:<14} oracle(s)={str(oracles):<24} '
              f'degree mean={mean:>5.1f}  range=[{lo},{hi}]')
    print('\n  If these degree ranges overlap heavily ACROSS scenarios with')
    print('  DIFFERENT oracles, degree alone cannot resolve the conflict and')
    print('  explicit context features are needed.')

    print('\n' + '=' * 92)
    print('  VERDICT')
    print('=' * 92)
    print(f'  overall 3-way agreement: {overall:.3f}')
    if overall >= 0.85:
        print('  -> HIGH. The per-cell oracles mostly choose the same hop anyway.')
        print('     Per-cell labelling is safe; conflicting labels are rare.')
    elif overall >= 0.5:
        print('  -> MODERATE. Real conflicts exist. Add explicit context features')
        print('     (local degree statistics) before behaviour cloning, or verify')
        print('     that degree already separates the regimes (table above).')
    else:
        print('  -> LOW. The oracles genuinely disagree. Per-cell labelling would')
        print('     feed the network contradictory supervision. Prefer a single')
        print('     global teacher, or condition the policy explicitly on regime.')

    ov = [r['n_loop_override'] for r in rows if r['n_loop_override'] is not None]
    if ov:
        print(f'\n  loop-avoidance override fired: {sum(ov)} times across {len(rows)} episodes')
        print('  (every such decision records an action that is NOT the teacher\'s')
        print('   choice -- the dataset must store the action actually taken)')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'schema': 'oracle_agreement_v1', 'oracles': ORACLES,
                   'duration': DURATION, 'initial_energy': INITIAL_ENERGY,
                   'run_params': {'duration': DURATION,
                                  'initial_energy': INITIAL_ENERGY,
                                  'seeds': args.seeds, 'max_log': MAX_LOG},
                   'provenance': provenance(),
                   'overall_agreement': overall, 'rows': rows}, f, indent=2)
    print(f'\n  saved to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
