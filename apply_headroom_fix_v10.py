"""apply_headroom_fix_v10.py  (revision 2) -- fix the id(G) reachability cache.

Assertion-guarded str.replace. Anchors must match EXACTLY ONCE; edits are staged
in memory and nothing is written unless every anchor matches.

THREE-STATE DETECTION. The file may be:
  * pristine              -> apply the full change
  * revision 1 of v10     -> apply only the delta (rev 1 shipped three defects,
                             listed below; do not leave it in place)
  * revision 2            -> no-op

DEFECTS IN REVISION 1, FIXED HERE
  R1-a  _frame_no was assigned AFTER super().__init__(). Safe today only
        because FANETSimulatorV2.__init__ happens not to call _build_graph();
        one line moved there would have turned it into an AttributeError.
        Now assigned before the super() call.
  R1-b  the docstring claimed collision_model provenance and per-seed rows were
        written to the output JSON. main() wrote neither. That is the same
        defect class as the `# (frame_marker, src)` comment sitting above an
        id(G) key -- documentation describing behaviour the code does not have.
        Both are now actually written.
  R1-c  the fixed-mode assertion is a tautology: the key's first element IS the
        frame number, so _cache_born[key] == self._frame_no on every hit by
        construction. It can never fire. Retained deliberately as a REGRESSION
        TRIPWIRE against a future edit reverting the key, and relabelled as
        such -- but verify_headroom_fix_v10.py now carries a real negative
        control that forces the broken key under cache_mode='fixed' and
        confirms the assertion raises.

SCOPE. Touches src/experiment_headroom.py only. simulator_v2.py is not
modified, so the armed G1/G2 regression anchors are unaffected.

NOTE ON BLAST RADIUS. HeadroomSimulator is imported by two other experiments,
experiment_calibration_sensitivity.py and experiment_collision_model.py, both
of which consume n_routable and drops_routable. They pick up the fix
automatically (cache_mode defaults to 'fixed'), which means their archived
outputs -- results/calibration_sensitivity.json and results/collision_model.json
-- were produced under the defect and need regenerating too.

USAGE
    python apply_headroom_fix_v10.py --src src --dry-run
    python apply_headroom_fix_v10.py --src src
"""
import argparse, io, os, sys

TARGET = 'experiment_headroom.py'
GUARD_R1 = "self._cache_born = {}"
GUARD_R2 = "REGRESSION TRIPWIRE"

JSON_OLD = """        json.dump({'rows': rows, 'headroom_rows': headroom_rows,
                   'overall_headroom': overall, 'actor': args.actor,
                   'routing_addressable': list(ROUTING_ADDRESSABLE),
                   'seeds': list(args.seeds), 'rates': list(args.rates)},
                  f, indent=2)
"""
JSON_NEW = """        stale_total = sum(r.get('cache_stale_hits', 0) for r in results)
        json.dump({'rows': rows, 'headroom_rows': headroom_rows,
                   'overall_headroom': overall, 'actor': args.actor,
                   'routing_addressable': list(ROUTING_ADDRESSABLE),
                   'seeds': list(args.seeds), 'rates': list(args.rates),
                   # provenance: results/headroom.json carried none of this,
                   # which is why dating it required git archaeology
                   'cache_mode': args.cache_mode,
                   'collision_model': args.collision_model,
                   'cache_stale_hits_total': int(stale_total),
                   'schema': 'headroom_v10',
                   # per-seed rows, so any future comparison can be PAIRED;
                   # the archive aggregated to 12 cells before writing
                   'rows_per_seed': results},
                  f, indent=2)
"""

# -- pristine -> revision 2 -------------------------------------------------
EDITS_BASE = [
    ("""    def __init__(self, config):
        super().__init__(config)
        self.routable = {}          # pid -> bool
        self._reach_cache = {}      # (frame_marker, src) -> reachable set

    def _on_packet_generated(self, G, pkt):
        key = (id(G), pkt.src)
        if key not in self._reach_cache:
            if len(self._reach_cache) > 4000:
                self._reach_cache.clear()
            try:
                self._reach_cache[key] = set(
                    nx.single_source_shortest_path_length(G, pkt.src))
            except nx.NodeNotFound:
                self._reach_cache[key] = set()
        self.routable[pkt.pid] = pkt.dst in self._reach_cache[key]
""",
     """    def __init__(self, config):
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
        \"\"\"Increment the frame counter HERE, not in the step loop, so it cannot
        drift out of step with the graph it labels whatever the caller does.\"\"\"
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
"""),

    ("""        m['n_routable'] = routable_gen
        m['routable_frac'] = routable_gen / max(m['n_generated'], 1)
        m['pdr_routable'] = routable_del / max(routable_gen, 1)
        m['drops_routable'] = dict(drops_routable)
        return m
""",
     """        m['n_routable'] = routable_gen
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
"""),

    ("""def _run(job):
    sc, cfg, rate, seed, actor = job
    full = {**BASE, **cfg, 'packet_rate': rate, 'seed': seed, 'actor': actor}
    m = HeadroomSimulator(full).run()
    return {
        'scenario': sc, 'rate': rate, 'seed': seed,
        'scenario_class': scenario_class(full), 'load_bucket': load_bucket(rate),
        'pdr_raw': m['pdr_predrain'], 'pdr_routable': m['pdr_routable'],
        'routable_frac': m['routable_frac'], 'n_routable': m['n_routable'],
        'drops_routable': m['drops_routable'],
    }
""",
     """def _run(job):
    sc, cfg, rate, seed, actor, cache_mode, collision_model = job
    full = {**BASE, **cfg, 'packet_rate': rate, 'seed': seed, 'actor': actor,
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
"""),

    ("""    ap.add_argument('--actor', default='spbp', help='best classical teacher')
""",
     """    ap.add_argument('--actor', default='spbp', help='best classical teacher')
    ap.add_argument('--cache_mode', default='fixed', choices=['fixed', 'legacy'],
                    help="'legacy' reproduces the pre-v10 id(G) defect on purpose")
    ap.add_argument('--collision_model', default='unsaturated',
                    choices=['unsaturated', 'saturated'],
                    help="'saturated' reproduces the pre-M-4-flip condition "
                         "under which results/headroom.json was produced")
"""),

    ("""    jobs = [(sc, cfg, r, sd, args.actor)
            for sc, cfg in scen.items() for r in args.rates for sd in args.seeds]
""",
     """    jobs = [(sc, cfg, r, sd, args.actor, args.cache_mode, args.collision_model)
            for sc, cfg in scen.items() for r in args.rates for sd in args.seeds]
"""),

    (JSON_OLD, JSON_NEW),
]

# -- revision 1 -> revision 2 (delta only) ----------------------------------
EDITS_UPGRADE = [
    ("""    def __init__(self, config):
        super().__init__(config)
        self.routable = {}          # pid -> bool
        self._reach_cache = {}      # (frame_no, src) -> reachable set
        self._cache_born = {}       # same key -> frame it was computed in
        self._frame_no = 0
""",
     """    def __init__(self, config):
        # Assigned BEFORE super().__init__ so _build_graph is safe to call from
        # anywhere in the base constructor, now or after a future edit.
        self._frame_no = 0
        super().__init__(config)
        self.routable = {}          # pid -> bool
        self._reach_cache = {}      # (frame_no, src) -> reachable set
        self._cache_born = {}       # same key -> frame it was computed in
"""),

    ("""        # EQUIVALENCE CONTROL. Structurally impossible under the fixed key;
        # verified to FAIL on the legacy path by verify_headroom_fix_v10.py.
        if self.cache_mode == 'fixed':
            assert self.cache_stats['stale_hits'] == 0, (
                f"fixed cache produced {self.cache_stats['stale_hits']} "
                f"cross-frame hits -- the key is not frame-scoped")
""",
     """        # REGRESSION TRIPWIRE, not a live check. Under the fixed key this is a
        # tautology (key[0] IS the frame number) and can never fire on today's
        # code. It exists to break loudly if a future edit reverts the key.
        # The real negative control lives in verify_headroom_fix_v10.py.
        if self.cache_mode == 'fixed':
            assert self.cache_stats['stale_hits'] == 0, (
                f"fixed cache produced {self.cache_stats['stale_hits']} "
                f"cross-frame hits -- the key is no longer frame-scoped")
"""),

    (JSON_OLD, JSON_NEW),
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
    ap.add_argument('--src', default='src')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    path = os.path.join(a.src, TARGET)
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found"); return 1
    text = io.open(path, encoding='utf-8').read()

    if GUARD_R2 in text:
        print("  ALREADY AT REVISION 2. Nothing to do."); return 0
    if GUARD_R1 in text:
        print("  Detected REVISION 1 -- applying the delta (3 defects).")
        staged, ok = apply(text, EDITS_UPGRADE, 'upgrade')
    else:
        print("  Detected PRISTINE file -- applying the full change.")
        staged, ok = apply(text, EDITS_BASE, 'base')

    if not ok:
        print("\n  NO FILE WRITTEN. Fix the anchors and re-run."); return 1
    if a.dry_run:
        print("\n  DRY RUN OK. Nothing written."); return 0

    io.open(path, 'w', encoding='utf-8').write(staged)
    print(f"\n  WROTE {path}")
    print("  NEXT: python verify_headroom_fix_v10.py --src " + a.src)
    return 0


if __name__ == '__main__':
    sys.exit(main())
