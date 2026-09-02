"""
experiment_headroom.py  —  How much PDR is ROUTING actually able to affect?

WHY THIS IS THE PRIORITY
========================
Three independent measurements now point the same way, and together they
question whether this environment can demonstrate what M4/M5 are built to
demonstrate:

  1. Locality:   k=1 and k=inf disagree on 2-13% of decisions and produce
                 IDENTICAL PDR. Many routes are near-equally good.
  2. Baseline:   a trivial nearest-to-destination rule already scores 59.5%
                 top-1 on genuinely contested decisions (M3.5).
  3. Reachability: only 21.4% of node pairs are connected in sparse_fast and
                 54.7% in medium_slow. Most of the network cannot be routed to
                 at all.

If PDR is dominated by connectivity and physics rather than by routing choice,
then "the GNN beats the MLP" and "warmstart beats scratch" may be measuring
something with very little room to move -- and that must be known BEFORE the
architecture is built, not after G4 returns an ambiguous number.

WHAT THIS MEASURES
------------------
Decomposes every generated packet's fate into causes that routing CAN and
CANNOT influence.

  CONNECTIVITY CEILING
      Fraction of packets whose destination was reachable from the source at
      generation time. No routing policy can deliver the rest, so this is an
      upper bound on PDR for ANY algorithm, learned or classical.
      (Approximate: mobility can connect a pair later, so the true ceiling is
      slightly higher. Reported as such, never as exact.)

  ROUTABLE PDR
      PDR computed over routable packets only. This is the quantity routing
      research is actually about; raw PDR conflates it with connectivity.

  LOSS DECOMPOSITION over routable packets, by drop reason:
      link_error      -> interference physics, weakly routing-addressable
      queue_overflow  -> congestion, PARTLY routing-addressable (this is
                         exactly what congestion-aware routing should fix)
      ttl_expired     -> path too long / churn, routing-addressable
      no_route        -> local dead end despite global reachability,
                         routing-addressable
      episode_end     -> measurement artifact, not a real loss

  ROUTING-ADDRESSABLE HEADROOM
      Share of routable packets lost to causes routing can plausibly fix.
      THIS is the budget M4/M5 are competing over. If it is small, no
      architecture can show a large effect here and the experimental design
      must change rather than the model.

HOW TO READ THE RESULT (rule fixed before running)
--------------------------------------------------
  headroom > 20%  : plenty of room; M4/M5 as designed are well-posed.
  headroom 5-20%  : workable, but effect sizes will be small. Report per-
                    scenario, and expect the dense regimes to carry the result.
  headroom < 5%   : the environment cannot demonstrate routing quality.
                    Fix the EXPERIMENT (higher load, tighter queues, more
                    partition-free topologies), not the model.

Usage:
    python src\\experiment_headroom.py --max_workers 16
    python src\\experiment_headroom.py --quick
"""

import os, sys, json, argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator_v2 import FANETSimulatorV2
from teacher_panel import scenario_class, load_bucket

# Config now lives in config_v2.py -- see the note there on why eight
# independent copies of this block were a latent hazard.
from config_v2 import SCENARIOS, RATES, BASE, get_suite, provenance  # noqa: F401

# Causes routing can plausibly do something about. link_error is deliberately
# EXCLUDED: a packet lost to a burst of hidden-terminal interference on an
# otherwise good link is not a routing mistake. Including it would inflate the
# apparent headroom and flatter the very models this experiment exists to
# scope.
ROUTING_ADDRESSABLE = ('queue_overflow', 'ttl_expired', 'no_route')


class HeadroomSimulator(FANETSimulatorV2):
    """Records, per packet, whether the destination was reachable at generation.

    Overrides ONLY the no-op _on_packet_generated hook, so every piece of
    validated M2 machinery runs unchanged.
    """

    def __init__(self, config):
        # Assigned BEFORE super().__init__ so _build_graph is safe to call from
        # anywhere in the base constructor, now or after a future edit.
        self._frame_no = 0
        super().__init__(config)
        self.routable = {}          # pid -> bool
        self._reach_cache = {}      # (frame_no, src) -> reachable set
        self._cache_born = {}       # same key -> frame it was computed in
        # 'fixed'  : key on the frame counter (correct)
        # 'legacy' : key on id(G), reproducing the pre-v10 defect verbatim so
        #            archived results can be attributed rather than guessed at
        self.cache_mode = str(config.get('cache_mode', 'fixed'))
        assert self.cache_mode in ('fixed', 'legacy'), self.cache_mode
        self.cache_stats = dict(lookups=0, stale_hits=0, distinct_ids=set())

    def _build_graph(self):
        """Increment the frame counter HERE, not in the step loop, so it cannot
        drift out of step with the graph it labels whatever the caller does."""
        self._frame_no += 1
        return super()._build_graph()

    def _on_packet_generated(self, G, pkt):
        self.cache_stats['lookups'] += 1
        self.cache_stats['distinct_ids'].add(id(G))
        key = ((id(G), pkt.src) if self.cache_mode == 'legacy'
               else (self._frame_no, pkt.src))
        if key in self._reach_cache and self._cache_born.get(key) != self._frame_no:
            # A hit on an entry computed under a DIFFERENT topology.
            self.cache_stats['stale_hits'] += 1
        if key not in self._reach_cache:
            if len(self._reach_cache) > 4000:
                self._reach_cache.clear(); self._cache_born.clear()
            try:
                self._reach_cache[key] = set(
                    nx.single_source_shortest_path_length(G, pkt.src))
            except nx.NodeNotFound:
                self._reach_cache[key] = set()
            self._cache_born[key] = self._frame_no
        self.routable[pkt.pid] = pkt.dst in self._reach_cache[key]

    def run(self):
        m = super().run()
        routable_gen = routable_del = 0
        drops_routable = Counter()
        for tr in self.completed_trajectories:
            if not self.routable.get(tr['pid'], False):
                continue
            routable_gen += 1
            if tr['delivered']:
                routable_del += 1
            else:
                drops_routable[tr['drop_reason'] or 'unknown'] += 1
        m['n_routable'] = routable_gen
        m['routable_frac'] = routable_gen / max(m['n_generated'], 1)
        m['pdr_routable'] = routable_del / max(routable_gen, 1)
        m['drops_routable'] = dict(drops_routable)
        m['cache_mode'] = self.cache_mode
        m['cache_lookups'] = self.cache_stats['lookups']
        m['cache_stale_hits'] = self.cache_stats['stale_hits']
        m['cache_distinct_ids'] = len(self.cache_stats['distinct_ids'])
        m['n_frames'] = self._frame_no
        # REGRESSION TRIPWIRE, not a live check. Under the fixed key this is a
        # tautology (key[0] IS the frame number) and can never fire on today's
        # code. It exists to break loudly if a future edit reverts the key.
        # The real negative control lives in verify_headroom_fix_v10.py.
        if self.cache_mode == 'fixed':
            assert self.cache_stats['stale_hits'] == 0, (
                f"fixed cache produced {self.cache_stats['stale_hits']} "
                f"cross-frame hits -- the key is no longer frame-scoped")
        return m


def _run(job):
    sc, cfg, rate, seed, actor, cache_mode, collision_model, base = job
    # `base` is the RESOLVED operating point (config_v2.BASE plus any CLI
    # override), passed explicitly rather than read from the module so a worker
    # under Windows spawn cannot pick up a different one than the parent.
    full = {**base, **cfg, 'packet_rate': rate, 'seed': seed, 'actor': actor,
            'cache_mode': cache_mode, 'collision_model': collision_model}
    m = HeadroomSimulator(full).run()
    return {
        'scenario': sc, 'rate': rate, 'seed': seed,
        'scenario_class': scenario_class(full), 'load_bucket': load_bucket(rate),
        'pdr_raw': m['pdr_predrain'], 'pdr_routable': m['pdr_routable'],
        'routable_frac': m['routable_frac'], 'n_routable': m['n_routable'],
        'drops_routable': m['drops_routable'],
        'cache_mode': cache_mode, 'collision_model': collision_model,
        'cache_lookups': m['cache_lookups'],
        'cache_stale_hits': m['cache_stale_hits'],
        'cache_distinct_ids': m['cache_distinct_ids'], 'n_frames': m['n_frames'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 31)))
    ap.add_argument('--rates', type=float, nargs='+', default=RATES)
    ap.add_argument('--actor', default='spbp', help='best classical teacher')
    ap.add_argument('--cache_mode', default='fixed', choices=['fixed', 'legacy'],
                    help="'legacy' reproduces the pre-v10 id(G) defect on purpose")
    ap.add_argument('--collision_model', default='unsaturated',
                    choices=['unsaturated', 'saturated'],
                    help="'saturated' reproduces the pre-M-4-flip condition "
                         "under which results/headroom.json was produced")
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='results/headroom.json')
    # -- operating point. Defaults ARE config_v2.BASE, so an unflagged run is
    # -- unchanged; overriding here never mutates the shared module, which the
    # -- 40 s SP-BP-parity reference depends on staying put (FILE2 3).
    ap.add_argument('--duration', type=float, default=BASE['duration'],
                    help='episode seconds (default: config_v2.BASE)')
    ap.add_argument('--drain_time', type=float, default=BASE['drain_time'],
                    help='drain seconds (default: config_v2.BASE)')
    ap.add_argument('--z_min', type=float, default=BASE['z_min'],
                    help='altitude floor in m (default: config_v2.BASE)')
    ap.add_argument('--z_max', type=float, default=BASE['z_max'],
                    help='altitude ceiling in m (default: config_v2.BASE)')
    # -- aggregation key. 'bucket' reproduces pre-v11.2 behaviour exactly.
    # -- 'rate' is REQUIRED for the rate probe: load_bucket() is
    # -- (<=0.5 low, <=2.0 medium, else high), so every probe rate 0.02-0.40
    # -- collapses into 'low' and the spread curve averages away.
    ap.add_argument('--by', default='bucket', choices=['bucket', 'rate'],
                    help="aggregate cells by load bucket (default) or by "
                         "individual rate (use this for the rate probe)")
    args = ap.parse_args()

    base = {**BASE, 'duration': args.duration, 'drain_time': args.drain_time,
            'z_min': args.z_min, 'z_max': args.z_max}
    if base != BASE:
        print(f"\n  operating point OVERRIDDEN: duration={base['duration']}s "
              f"drain={base['drain_time']}s alt={base['z_min']}-{base['z_max']}m")
    if args.by == 'rate':
        n_cells = (1 if args.quick else len(SCENARIOS)) * len(args.rates)
        print(f"  aggregating by RATE -> {n_cells} cells "
              f"(by bucket this would collapse to "
              f"{len({load_bucket(r) for r in args.rates})} per scenario)")

    scen = {'medium_slow': SCENARIOS['medium_slow']} if args.quick else SCENARIOS
    jobs = [(sc, cfg, r, sd, args.actor, args.cache_mode, args.collision_model,
             base)
            for sc, cfg in scen.items() for r in args.rates for sd in args.seeds]

    print("\n" + "=" * 78)
    print("  HEADROOM EXPERIMENT — how much PDR can routing actually affect?")
    print("=" * 78)
    print(f"  actor = {args.actor} (best classical teacher)   runs = {len(jobs)}")
    print(f"  routing-addressable drop causes: {', '.join(ROUTING_ADDRESSABLE)}")
    print("  (link_error deliberately EXCLUDED -- a packet lost to an interference")
    print("   burst on an otherwise good link is not a routing mistake, and")
    print("   counting it would inflate the apparent headroom.)")

    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % max(len(jobs) // 10, 1) == 0:
                print(f"    {done}/{len(jobs)}")

    # aggregate per (scenario, load bucket) -- or per (scenario, rate) under
    # --by rate, which the rate probe requires; see the CLI note above.
    _key = (lambda r: (r['scenario'], r['rate'])) if args.by == 'rate' \
        else (lambda r: (r['scenario'], r['load_bucket']))
    _keyname = 'rate' if args.by == 'rate' else 'load_bucket'
    cells = {}
    for r in results:
        cells.setdefault(_key(r), []).append(r)

    print("\n" + "-" * 78)
    print("  CONNECTIVITY CEILING AND ROUTABLE PDR")
    print("-" * 78)
    print(f"  {'scenario':<13}{'load':<8}{'routable':>10}{'pdr_raw':>10}"
          f"{'pdr_routable':>14}{'lost(routable)':>16}")
    rows = []
    for key in sorted(cells):
        rs = cells[key]
        rf = float(np.mean([x['routable_frac'] for x in rs]))
        praw = float(np.mean([x['pdr_raw'] for x in rs]))
        prt = float(np.mean([x['pdr_routable'] for x in rs]))
        print(f"  {key[0]:<13}{key[1]:<8}{100*rf:>9.1f}%{praw:>10.3f}"
              f"{prt:>14.3f}{1-prt:>15.3f}")
        rows.append({'scenario': key[0], _keyname: key[1],
                     'routable_frac': rf, 'pdr_raw': praw, 'pdr_routable': prt})

    # loss decomposition over routable packets
    print("\n" + "-" * 78)
    print("  WHY ROUTABLE PACKETS ARE LOST  (share of routable packets)")
    print("-" * 78)
    causes = ['link_error', 'queue_overflow', 'ttl_expired', 'no_route',
              'energy_depleted', 'episode_end']
    print(f"  {'scenario':<13}{'load':<8}" + "".join(f"{c[:12]:>15}" for c in causes))
    headroom_rows = []
    for key in sorted(cells):
        rs = cells[key]
        tot = sum(x['n_routable'] for x in rs)
        agg = Counter()
        for x in rs:
            agg.update(x['drops_routable'])
        line = f"  {key[0]:<13}{key[1]:<8}"
        for c in causes:
            line += f"{agg.get(c, 0)/max(tot,1):>15.3f}"
        print(line)
        addressable = sum(agg.get(c, 0) for c in ROUTING_ADDRESSABLE) / max(tot, 1)
        headroom_rows.append({'scenario': key[0], _keyname: key[1],
                              'headroom': float(addressable)})

    print("\n" + "-" * 78)
    print("  ROUTING-ADDRESSABLE HEADROOM  (the budget M4/M5 compete over)")
    print("-" * 78)
    by_sc = {}
    for h in headroom_rows:
        by_sc.setdefault(h['scenario'], []).append(h['headroom'])
    print(f"  {'scenario':<15}{'headroom':>12}")
    for sc, v in sorted(by_sc.items()):
        print(f"  {sc:<15}{100*np.mean(v):>11.1f}%")
    overall = float(np.mean([h['headroom'] for h in headroom_rows]))
    print(f"  {'OVERALL':<15}{100*overall:>11.1f}%")

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    if overall > 0.20:
        print(f"  {100*overall:.1f}% of routable packets are lost to causes routing can")
        print("  plausibly address. There is ample room for a better policy to show")
        print("  a measurable effect. M4/M5 as designed are well-posed.")
    elif overall > 0.05:
        print(f"  {100*overall:.1f}% routing-addressable headroom. Workable, but effect")
        print("  sizes will be SMALL. Consequences to act on:")
        print("   - report per-scenario, not pooled: the dense regimes will carry")
        print("     any result, and sparse_fast may contribute mostly noise;")
        print("   - power the M4/M5 comparisons for a small effect (more seeds),")
        print("     rather than assuming a large one will be obvious;")
        print("   - state the headroom explicitly in the paper so a modest gain is")
        print("     read against what was achievable, not against 1.0.")
    else:
        print(f"  ONLY {100*overall:.1f}% routing-addressable headroom. This environment")
        print("  CANNOT demonstrate routing quality: PDR is dominated by")
        print("  connectivity and physics, not by decisions. Fix the EXPERIMENT,")
        print("  not the model -- e.g. raise offered load, tighten queues, or use")
        print("  topologies that stay connected -- before building M4, or G4 will")
        print("  return an ambiguous number that no architecture can improve.")
    print()
    print("  NOTE: the connectivity ceiling is approximate. A pair unreachable at")
    print("  generation may connect later through mobility, so the true ceiling is")
    print("  slightly higher and this headroom is a mild UNDER-estimate.")
    print("=" * 78 + "\n")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        stale_total = sum(r.get('cache_stale_hits', 0) for r in results)
        json.dump({'rows': rows, 'headroom_rows': headroom_rows,
                   'overall_headroom': overall, 'actor': args.actor,
                   'routing_addressable': list(ROUTING_ADDRESSABLE),
                   'seeds': list(args.seeds), 'rates': list(args.rates),
                   # provenance: results/headroom.json carried none of this,
                   # which is why dating it required git archaeology
                   'cache_mode': args.cache_mode,
                   'collision_model': args.collision_model,
                   'cache_stale_hits_total': int(stale_total),
                   'schema': 'headroom_v11_2',
                   # RESOLVED operating point + config fingerprint. provenance()
                   # existed since v11 but was imported and never called, which
                   # is why dating headroom.json needed git archaeology.
                   'provenance': {**provenance(), 'resolved_base': base,
                                  'aggregated_by': args.by},
                   # per-seed rows, so any future comparison can be PAIRED;
                   # the archive aggregated to 12 cells before writing
                   'rows_per_seed': results},
                  f, indent=2)
    print(f"  saved to {args.out}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
