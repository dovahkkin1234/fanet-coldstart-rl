"""verify_headroom_fix_v10.py (revision 3) -- verify the cache fix BY EXECUTION.

WHY REVISION 3 EXISTS
  Rev 2's check 4 compared the legacy run's routable verdicts against the fixed
  run's and required them to DIFFER in the partitioned scenarios. That check is
  not sound, because whether they differ depends on CPython's allocator
  recycling graph addresses -- which varies by platform, build and heap history.

  Measured, same seed, same scenarios:
      Linux sandbox : 36-48 distinct id(G) per 80 frames -> 17-30% stale hits
      Windows/conda : 57-60 distinct id(G) per 80 frames -> 0.0-1.7% stale hits

  On a low-reuse host the legacy run can produce zero wrong verdicts, making
  legacy == fixed and failing rev 2's check 4 even though the fix is correct.
  That is a false failure caused by the check, not by the code under test.

  Rev 3 replaces it with a DETERMINISTIC worst case: a constant cache key, so
  every lookup after the first per source is guaranteed stale regardless of the
  allocator. The immunity claim -- that a stale reachability set is still the
  CORRECT set wherever reachability is ~100% -- is then testable everywhere.

CHECKS
  1. legacy stale-hit rate + address reuse                       (INFORMATIONAL,
     never fails: allocator-dependent, see above)
  2. fixed produces exactly zero stale hits                      (bug is gone)
  3. fixed agrees with a fresh per-packet BFS on every packet     (fix is right)
  4. DETERMINISTIC SIGNATURE, pre-registered: under forced worst-case staleness,
     very_dense and dense_slow must STILL agree with fresh BFS (reachability
     ~100%, so a stale set is still correct); medium_slow and sparse_fast must
     NOT. Allocator-independent.
  5. raw PDR is bit-identical across all modes                    (fix is scoped)
  6. the tripwire fires when the key is deliberately reverted     (control works)
  7. BFS cost per episode                                         (informational)
"""
import argparse, os, sys, time
import networkx as nx

REACH = {'very_dense': '100.0%', 'dense_slow': '99.2%',
         'medium_slow': '54.7%', 'sparse_fast': '21.4%'}
IMMUNE = ('very_dense', 'dense_slow')


def _mk_forced(HeadroomSimulator):
    class ForcedStale(HeadroomSimulator):
        """Constant cache key -> every lookup after the first per source reuses
        a reachability set computed under an EARLIER frame's topology. This is
        the worst case the id(G) defect can produce, made deterministic."""
        def _on_packet_generated(self, G, pkt):
            self.cache_stats['lookups'] += 1
            self.cache_stats['distinct_ids'].add(id(G))
            key = ('FORCED', pkt.src)
            if (key in self._reach_cache
                    and self._cache_born.get(key) != self._frame_no):
                self.cache_stats['stale_hits'] += 1
            if key not in self._reach_cache:
                try:
                    self._reach_cache[key] = set(
                        nx.single_source_shortest_path_length(G, pkt.src))
                except nx.NodeNotFound:
                    self._reach_cache[key] = set()
                self._cache_born[key] = self._frame_no
            self.routable[pkt.pid] = pkt.dst in self._reach_cache[key]
    return ForcedStale


def run_with_truth(cls, cfg):
    """One run. Records ground truth computed fresh per packet, counts BFS."""
    truth, ctr = {}, {'bfs': 0}
    _orig = nx.single_source_shortest_path_length

    class T(cls):
        def _on_packet_generated(self, G, pkt):
            try:
                truth[pkt.pid] = pkt.dst in _orig(G, pkt.src)
            except nx.NodeNotFound:
                truth[pkt.pid] = False
            super()._on_packet_generated(G, pkt)

    def counting(G, src, **kw):
        ctr['bfs'] += 1
        return _orig(G, src, **kw)

    sim = T(cfg)
    nx.single_source_shortest_path_length = counting
    try:
        m = sim.run()
    finally:
        nx.single_source_shortest_path_length = _orig
    wrong = sum(1 for pid, t in truth.items() if sim.routable.get(pid, False) != t)
    return m, sim, wrong, ctr['bfs']


def negative_control(HeadroomSimulator, BASE, SCENARIOS):
    class BrokenKey(HeadroomSimulator):
        def _on_packet_generated(self, G, pkt):
            self.cache_stats['lookups'] += 1
            key = (id(G), pkt.src)          # the defect, deliberately restored
            if (key in self._reach_cache
                    and self._cache_born.get(key) != self._frame_no):
                self.cache_stats['stale_hits'] += 1
            if key not in self._reach_cache:
                try:
                    self._reach_cache[key] = set(
                        nx.single_source_shortest_path_length(G, pkt.src))
                except nx.NodeNotFound:
                    self._reach_cache[key] = set()
                self._cache_born[key] = self._frame_no
            self.routable[pkt.pid] = pkt.dst in self._reach_cache[key]

    # Constant-key variant guarantees the tripwire has something to catch even
    # on a host where the allocator never recycles a graph address.
    class BrokenKeyForced(BrokenKey):
        def _on_packet_generated(self, G, pkt):
            self.cache_stats['lookups'] += 1
            key = ('FORCED', pkt.src)
            if (key in self._reach_cache
                    and self._cache_born.get(key) != self._frame_no):
                self.cache_stats['stale_hits'] += 1
            if key not in self._reach_cache:
                try:
                    self._reach_cache[key] = set(
                        nx.single_source_shortest_path_length(G, pkt.src))
                except nx.NodeNotFound:
                    self._reach_cache[key] = set()
                self._cache_born[key] = self._frame_no
            self.routable[pkt.pid] = pkt.dst in self._reach_cache[key]

    cfg = {**BASE, **SCENARIOS['sparse_fast'], 'packet_rate': 2.0,
           'seed': 1, 'actor': 'spbp'}      # cache_mode defaults to 'fixed'
    try:
        BrokenKeyForced(cfg).run()
    except AssertionError as e:
        return True, str(e).split('\n')[0]
    return False, "no AssertionError -- the tripwire is DEAD CODE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src')
    ap.add_argument('--rate', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.src))
    from experiment_headroom import HeadroomSimulator, BASE, SCENARIOS
    Forced = _mk_forced(HeadroomSimulator)

    print("=" * 92)
    print("  VERIFY v10 rev3 -- reachability cache key")
    print("=" * 92)
    print("  PRE-REGISTERED, DETERMINISTIC (recorded before running):")
    print("    under FORCED worst-case staleness --")
    print("      very_dense / dense_slow   -> still 0 wrong (reachability ~100%)")
    print("      medium_slow / sparse_fast -> some wrong    (partitioned)")
    print("    'legacy' below is INFORMATIONAL: its stale rate is allocator-")
    print("    dependent and differs by platform, so nothing is asserted on it.")
    print()
    print(f"{'scenario':<13}{'reach':>8}{'mode':>8}{'stale':>7}{'ids':>5}"
          f"{'frames':>7}{'reuse':>7}{'wrong':>7}{'bfs':>7}"
          f"{'rt_frac':>9}{'pdr_raw':>9}{'s':>6}")
    print("-" * 92)

    fails, sig = [], {}
    for sc in ['very_dense', 'dense_slow', 'medium_slow', 'sparse_fast']:
        pm = {}
        for mode, cls, cm in [('legacy', HeadroomSimulator, 'legacy'),
                              ('fixed', HeadroomSimulator, 'fixed'),
                              ('forced', Forced, 'legacy')]:
            cfg = {**BASE, **SCENARIOS[sc], 'packet_rate': a.rate,
                   'seed': a.seed, 'actor': 'spbp', 'cache_mode': cm}
            t0 = time.time()
            try:
                m, sim, wrong, nbfs = run_with_truth(cls, cfg)
            except AssertionError as e:
                print(f"{sc:<13}{REACH[sc]:>8}{mode:>8}   ASSERTION FAILED: {e}")
                fails.append(f"{sc}/{mode}: {e}"); continue
            reuse = 1.0 - m['cache_distinct_ids'] / max(m['n_frames'], 1)
            pm[mode] = dict(m=m, wrong=wrong)
            print(f"{sc:<13}{REACH[sc]:>8}{mode:>8}{m['cache_stale_hits']:>7}"
                  f"{m['cache_distinct_ids']:>5}{m['n_frames']:>7}"
                  f"{100*reuse:>6.0f}%{wrong:>7}{nbfs:>7}"
                  f"{m['routable_frac']:>9.4f}{m['pdr_predrain']:>9.4f}"
                  f"{time.time()-t0:>6.1f}")

        if len(pm) != 3:
            continue
        if pm['fixed']['m']['cache_stale_hits'] != 0:
            fails.append(f"{sc}: fixed produced stale hits")
        if pm['fixed']['wrong'] != 0:
            fails.append(f"{sc}: fixed disagrees with fresh BFS on "
                         f"{pm['fixed']['wrong']} packets")
        if pm['forced']['m']['cache_stale_hits'] == 0:
            fails.append(f"{sc}: forced mode produced no stale hits -- the "
                         f"worst case is not being exercised")

        w = pm['forced']['wrong']
        sig[sc] = w
        if sc in IMMUNE and w != 0:
            fails.append(f"{sc}: SIGNATURE VIOLATED -- immune scenario got "
                         f"{w} wrong verdicts under forced staleness")
        if sc not in IMMUNE and w == 0:
            fails.append(f"{sc}: SIGNATURE VIOLATED -- partitioned scenario got "
                         f"0 wrong verdicts under forced staleness")

        pdrs = {k: v['m']['pdr_predrain'] for k, v in pm.items()}
        if len(set(pdrs.values())) != 1:
            fails.append(f"{sc}: raw PDR differs across modes {pdrs} -- the fix "
                         f"is not confined to bookkeeping")

    print()
    print("  CHECK 4 -- DETERMINISTIC SIGNATURE (forced worst-case staleness)")
    for sc, w in sig.items():
        want = '0 wrong' if sc in IMMUNE else '>0 wrong'
        ok = (w == 0) if sc in IMMUNE else (w > 0)
        print(f"    {sc:<13} predicted {want:<9} got {w:>3} wrong   "
              f"{'OK' if ok else 'MISMATCH'}")

    print()
    print("  CHECK 6 -- NEGATIVE CONTROL (broken key forced under cache_mode=fixed)")
    fired, msg = negative_control(HeadroomSimulator, BASE, SCENARIOS)
    print(f"    tripwire fired: {fired}   {msg}")
    if not fired:
        fails.append("negative control: the tripwire did not fire on a "
                     "deliberately reverted key -- it is dead code")

    print()
    if fails:
        print(f"  FAIL -- {len(fails)} problem(s):")
        for f in fails:
            print("    - " + f)
        return 1
    print("  PASS -- 7/7, and every assertion is allocator-independent.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
