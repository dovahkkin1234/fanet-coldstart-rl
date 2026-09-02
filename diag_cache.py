"""Instrument the LEGACY _reach_cache without changing its behaviour.

Counts, per episode:
  lookups        - calls to _on_packet_generated
  stale_hits     - cache hits whose entry was created in a DIFFERENT frame
  distinct_ids   - distinct id(G) values seen (tests the address-reuse claim)
  frames_seen    - frames in which at least one packet was generated
  disagreements  - packets whose routable verdict differs from a fresh BFS

Behaviour is byte-identical to the shipped code: the same key is used and the
same value is returned. Everything added is observation only.
"""
import sys, os, json, argparse

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--src', default='src')
_ap.add_argument('--out', default='results/diag_cache.json')
_known, _ = _ap.parse_known_args()
sys.path.insert(0, os.path.abspath(_known.src))   # cwd-relative, same as the
                                                    # apply/verify scripts --
                                                    # run this from FANET_sim,
                                                    # not from inside src\

import networkx as nx
from experiment_headroom import HeadroomSimulator, BASE, SCENARIOS


class DiagSimulator(HeadroomSimulator):
    def __init__(self, config):
        super().__init__(config)
        self._frame_no = 0
        self._created_in = {}      # key -> frame it was computed in
        self.diag = dict(lookups=0, stale_hits=0, frames_seen=set(),
                         distinct_ids=set(), disagreements=0, truth_routable=0)

    def _build_graph(self):
        self._frame_no += 1
        return super()._build_graph()

    def _on_packet_generated(self, G, pkt):
        self.diag['lookups'] += 1
        self.diag['frames_seen'].add(self._frame_no)
        self.diag['distinct_ids'].add(id(G))

        # ---- ground truth: reachability in THIS frame, computed fresh -------
        try:
            truth = pkt.dst in nx.single_source_shortest_path_length(G, pkt.src)
        except nx.NodeNotFound:
            truth = False
        self.diag['truth_routable'] += int(truth)

        # ---- the shipped code path, unchanged ------------------------------
        key = (id(G), pkt.src)
        if key in self._reach_cache and self._created_in.get(key) != self._frame_no:
            self.diag['stale_hits'] += 1
        if key not in self._reach_cache:
            if len(self._reach_cache) > 4000:
                self._reach_cache.clear(); self._created_in.clear()
            try:
                self._reach_cache[key] = set(
                    nx.single_source_shortest_path_length(G, pkt.src))
            except nx.NodeNotFound:
                self._reach_cache[key] = set()
            self._created_in[key] = self._frame_no
        self.routable[pkt.pid] = pkt.dst in self._reach_cache[key]

        if self.routable[pkt.pid] != truth:
            self.diag['disagreements'] += 1


def run(scenario, rate, seed):
    cfg = {**BASE, **SCENARIOS[scenario], 'packet_rate': rate,
           'seed': seed, 'actor': 'spbp'}
    s = DiagSimulator(cfg)
    m = s.run()
    d = s.diag
    return dict(scenario=scenario, rate=rate, seed=seed,
                lookups=d['lookups'], stale_hits=d['stale_hits'],
                frames=len(d['frames_seen']), distinct_ids=len(d['distinct_ids']),
                disagreements=d['disagreements'],
                routable_frac_buggy=m['routable_frac'],
                routable_frac_truth=d['truth_routable'] / max(d['lookups'], 1))


if __name__ == '__main__':
    print(f"{'scenario':<13}{'reach':>7}{'frames':>8}{'lookups':>9}"
          f"{'ids':>5}{'stale':>8}{'stale%':>8}{'wrong':>7}{'wrong%':>8}"
          f"{'rf_buggy':>10}{'rf_true':>9}")
    print("-" * 94)
    out = []
    reach = {'very_dense': '100.0%', 'dense_slow': '99.2%',
             'medium_slow': '54.7%', 'sparse_fast': '21.4%'}
    for sc in ['very_dense', 'dense_slow', 'medium_slow', 'sparse_fast']:
        r = run(sc, 2.0, 1)
        out.append(r)
        print(f"{sc:<13}{reach[sc]:>7}{r['frames']:>8}{r['lookups']:>9}"
              f"{r['distinct_ids']:>5}{r['stale_hits']:>8}"
              f"{100*r['stale_hits']/max(r['lookups'],1):>7.1f}%"
              f"{r['disagreements']:>7}"
              f"{100*r['disagreements']/max(r['lookups'],1):>7.2f}%"
              f"{r['routable_frac_buggy']:>10.4f}{r['routable_frac_truth']:>9.4f}")
    out_path = _known.out
    json.dump(out, open(out_path, 'w'), indent=1)
    print(f'\n  saved to {out_path}')
