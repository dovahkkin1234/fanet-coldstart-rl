"""apply_headroom_probe_v11_2.py -- unblock the three-quantity rate probe.

Assertion-guarded str.replace, same convention as every patch in this project:
anchors must match EXACTLY ONCE, edits are staged in memory, nothing is written
unless every anchor matches, idempotency-guarded.

WHY THIS EXISTS

FILE2 SS1.5.1 needs experiment_headroom.py pointed at 1000 s / 100-300 m to pick
the real rate grid. Three things stopped that:

  1. NO OPERATING-POINT FLAGS. After v11 the script reads config_v2.BASE, which
     is correct -- but the only exposed knob is --rates. Changing duration or
     altitude meant editing config_v2.BASE, which moves EVERY consumer including
     the 40 s reference the SP-BP-parity gate depends on (FILE2 SS3). Now
     overridable per-run without touching the shared module.

  2. NO PROVENANCE IN THE OUTPUT. config_v2.provenance() existed and was
     imported with a noqa: F401 -- imported, never called. Dating
     results/headroom.json previously required git archaeology; that is exactly
     the failure this function was written to prevent. Now written into the JSON,
     with the RESOLVED operating point (post-override), not the module defaults.

  3. THE PROBE WOULD HAVE BEEN SILENTLY COLLAPSED. This is the one that would
     have wasted the run. teacher_panel.load_bucket() is:
         <=0.5 -> low,  <=2.0 -> medium,  else high
     Every probe rate (0.02 .. 0.40) is <= 0.5, so ALL SIX map to 'low'.
     Aggregation is by (scenario, load_bucket), so 4 scenarios x 6 rates = 24
     cells would collapse into 4, averaging away the policy-spread curve that is
     the entire point of the probe. Fixed with --by rate.

     NOTE the split: this patch does NOT re-threshold load_bucket itself. The
     correct thresholds depend on the grid the probe has not chosen yet, so that
     belongs in the later patch that sets RATES (FILE2 SS2.1 -> SS1.5.1).

BACKWARD COMPATIBILITY. --by defaults to 'bucket' and the flags default to
config_v2.BASE, so an unflagged run is byte-identical to pre-patch behaviour.
Verified by re-running against the committed results/headroom_v10.json.

SCOPE. src/experiment_headroom.py only. Nothing else imports its _run (checked:
experiment_calibration_sensitivity.py and experiment_collision_model.py import
HeadroomSimulator and ROUTING_ADDRESSABLE, never _run), so the job-tuple arity
change is contained.

USAGE
    python apply_headroom_probe_v11_2.py --src src --dry-run
    python apply_headroom_probe_v11_2.py --src src
"""
import argparse, io, os, sys

TARGET = 'experiment_headroom.py'
GUARD = "--by"

EDITS = [
    # ---- 1. _run: accept a resolved base config + carry the group key --------
    ("""def _run(job):
    sc, cfg, rate, seed, actor, cache_mode, collision_model = job
    full = {**BASE, **cfg, 'packet_rate': rate, 'seed': seed, 'actor': actor,
            'cache_mode': cache_mode, 'collision_model': collision_model}
    m = HeadroomSimulator(full).run()
    return {
        'scenario': sc, 'rate': rate, 'seed': seed,
        'scenario_class': scenario_class(full), 'load_bucket': load_bucket(rate),""",
     """def _run(job):
    sc, cfg, rate, seed, actor, cache_mode, collision_model, base = job
    # `base` is the RESOLVED operating point (config_v2.BASE plus any CLI
    # override), passed explicitly rather than read from the module so a worker
    # under Windows spawn cannot pick up a different one than the parent.
    full = {**base, **cfg, 'packet_rate': rate, 'seed': seed, 'actor': actor,
            'cache_mode': cache_mode, 'collision_model': collision_model}
    m = HeadroomSimulator(full).run()
    return {
        'scenario': sc, 'rate': rate, 'seed': seed,
        'scenario_class': scenario_class(full), 'load_bucket': load_bucket(rate),"""),

    # ---- 2. CLI: operating point + aggregation key ---------------------------
    ("""    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='results/headroom.json')
    args = ap.parse_args()

    scen = {'medium_slow': SCENARIOS['medium_slow']} if args.quick else SCENARIOS
    jobs = [(sc, cfg, r, sd, args.actor, args.cache_mode, args.collision_model)
            for sc, cfg in scen.items() for r in args.rates for sd in args.seeds]""",
     """    ap.add_argument('--max_workers', type=int, default=None)
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
        print(f"\\n  operating point OVERRIDDEN: duration={base['duration']}s "
              f"drain={base['drain_time']}s alt={base['z_min']}-{base['z_max']}m")
    if args.by == 'rate':
        n_cells = (1 if args.quick else len(SCENARIOS)) * len(args.rates)
        print(f"  aggregating by RATE -> {n_cells} cells "
              f"(by bucket this would collapse to "
              f"{len({load_bucket(r) for r in args.rates})} per scenario)")

    scen = {'medium_slow': SCENARIOS['medium_slow']} if args.quick else SCENARIOS
    jobs = [(sc, cfg, r, sd, args.actor, args.cache_mode, args.collision_model,
             base)
            for sc, cfg in scen.items() for r in args.rates for sd in args.seeds]"""),

    # ---- 3. aggregation key ------------------------------------------------
    ("""    # aggregate per (scenario, load bucket)
    cells = {}
    for r in results:
        cells.setdefault((r['scenario'], r['load_bucket']), []).append(r)""",
     """    # aggregate per (scenario, load bucket) -- or per (scenario, rate) under
    # --by rate, which the rate probe requires; see the CLI note above.
    _key = (lambda r: (r['scenario'], r['rate'])) if args.by == 'rate' \\
        else (lambda r: (r['scenario'], r['load_bucket']))
    _keyname = 'rate' if args.by == 'rate' else 'load_bucket'
    cells = {}
    for r in results:
        cells.setdefault(_key(r), []).append(r)"""),

    # ---- 4. rows: label the group column by whatever it actually is ---------
    ("""        rows.append({'scenario': key[0], 'load_bucket': key[1],
                     'routable_frac': rf, 'pdr_raw': praw, 'pdr_routable': prt})""",
     """        rows.append({'scenario': key[0], _keyname: key[1],
                     'routable_frac': rf, 'pdr_raw': praw, 'pdr_routable': prt})"""),

    ("""        headroom_rows.append({'scenario': key[0], 'load_bucket': key[1],
                              'headroom': float(addressable)})""",
     """        headroom_rows.append({'scenario': key[0], _keyname: key[1],
                              'headroom': float(addressable)})"""),

    # ---- 5. provenance into the output -------------------------------------
    ("""                   'cache_mode': args.cache_mode,
                   'collision_model': args.collision_model,
                   'cache_stale_hits_total': int(stale_total),
                   'schema': 'headroom_v10',""",
     """                   'cache_mode': args.cache_mode,
                   'collision_model': args.collision_model,
                   'cache_stale_hits_total': int(stale_total),
                   'schema': 'headroom_v11_2',
                   # RESOLVED operating point + config fingerprint. provenance()
                   # existed since v11 but was imported and never called, which
                   # is why dating headroom.json needed git archaeology.
                   'provenance': {**provenance(), 'resolved_base': base,
                                  'aggregated_by': args.by},""")
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    path = os.path.join(a.src, TARGET)
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found"); return 1
    text = io.open(path, encoding='utf-8').read()

    if "ap.add_argument('--by'" in text:
        print("  ALREADY APPLIED (v11.2 guard hit). Nothing to do."); return 0

    staged, ok = text, True
    for i, (old, new) in enumerate(EDITS, 1):
        n = staged.count(old)
        if n != 1:
            print(f"  anchor {i}: matched {n} times, expected 1  <-- ABORT")
            ok = False
        else:
            print(f"  anchor {i}: OK")
            staged = staged.replace(old, new, 1)

    if not ok:
        print("\n  NO FILE WRITTEN. Fix the anchors and re-run."); return 1
    if a.dry_run:
        print(f"\n  DRY RUN OK -- {len(EDITS)}/{len(EDITS)} anchors matched. "
              f"Nothing written."); return 0

    io.open(path, 'w', encoding='utf-8').write(staged)
    print(f"\n  WROTE {path}")
    print("  NEXT: python verify_headroom_probe_v11_2.py --src " + a.src)
    return 0


if __name__ == '__main__':
    sys.exit(main())
