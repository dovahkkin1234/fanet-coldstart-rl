"""
audit_dataset_v3.py  --  INDEPENDENT audit of Dataset V3 (docs/DATASET_V3_SPEC.md §7).

    python src\\audit_dataset_v3.py --data data\\v3
    python src\\audit_dataset_v3.py --data data\\v3 --per_shard 3000

Not the gate. G3.5 v3 checks what the generator's author thought to check, using
the generator's own records; this script rebuilds every sampled frame graph from
the STORED arrays (de-normalised with the manifest's constants) and re-derives
labels, scores, features and votes with code written separately from
teacher_pickers_v3 / features_v2 -- so a shared bug cannot pass both.

  A  candidates   are neighbours of the deciding node in the stored frame, in
                  canonical order (ascending distance to dst, ties by node id)
  B  label        da_gpsr re-derived from stored positions / snapshot queue
                  occupancy / link quality, ties broken by ascending node id
                  (= G.neighbors order); stored scores reproduced
  C  features     every candidate and query feature re-derived (own_queue_live
                  from the context's recorded live queue length)
  D  votes        gpsr = closest to dst; dijkstra = minimum-hop reachable
                  candidate, -1 exactly when no candidate reaches dst; spbp
                  re-derived from its formula; the behaviour choice of non-da_gpsr
                  episodes equals the corresponding teacher
  E  splits       seeds partition into train/val/test, the generalisation
                  scenario is held out by scenario, no shard shared
  F  hashing      behaviour table and packet sample recomputed with an
                  independent splitmix64 implementation
  G  own queue    (not re-derivable from frames: it is live) each sampled
                  context's own-queue histogram counts exactly its multiplicity
                  and contains its first-occurrence value; every step of a
                  sampled context has an own queue the histogram contains
Each re-derivation is first shown to FAIL on a corrupted copy of a real shard.
"""

import argparse
import json
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TTL_REF = 20.0
HOP_CAP = 10.0
W_PROG, W_QUEUE, W_QUAL = 1.0, 1.0, 0.5       # da_gpsr weights (routing_teachers_v2)
SPBP_V_BIAS = 1.0                              # routing_teachers_v2.SPBP_V_BIAS
HOP_UNREACHABLE = 999.0
LABEL_MIN_AGREE = 0.99
FEAT_TOL = 1e-4
FEAT_MIN_OK = 0.999


# ── independent splitmix64 (do NOT import generate_dataset_v3's) ────────────
def _sm64(a, b, salt):
    m = (1 << 64) - 1
    x = (((a & 0xFFFFFFFF) << 32) | (b & 0xFFFFFFFF)) & m
    x ^= (salt * 0xD1B54A32D192ED03) & m
    x = (x + 0x9E3779B97F4A7C15) & m
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & m
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & m
    return ((x ^ (x >> 31)) >> 11) / 2.0 ** 53


class Frame:
    """One stored frame, de-normalised, with adjacency and BFS helpers."""

    def __init__(self, d, f, nc, names):
        N = int(d['f_n_nodes'][f])
        nf = d['f_node_feat'][f * N:(f + 1) * N].astype(np.float64)
        ix, iy, iz = (names['node'].index(k) for k in ('x', 'y', 'z'))
        self.pos = np.stack([nf[:, ix] * nc['area_x'], nf[:, iy] * nc['area_y'],
                             nf[:, iz] * max(nc['z_max'] - nc['z_min'], 1e-6) + nc['z_min']], 1)
        self.occ = nf[:, names['node'].index('queue_occupancy')]
        self.N = N
        self.adj = {i: [] for i in range(N)}
        self.lq = {}

    def set_edges(self, ei, ef, lq_col):
        for k in range(ei.shape[1]):
            a, b = int(ei[0, k]), int(ei[1, k])
            self.adj[a].append(b)
            self.adj[b].append(a)
            self.lq[(a, b)] = self.lq[(b, a)] = float(ef[k, lq_col])
        for a in self.adj:
            self.adj[a].sort()                    # G.neighbors order = ascending id

    def bfs(self, src, cutoff=None):
        dist = {src: 0}
        dq = deque([src])
        while dq:
            u = dq.popleft()
            if cutoff is not None and dist[u] >= cutoff:
                continue
            for v in self.adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        return dist


def build_frames(d, nc, names):
    nfr = len(d['f_n_edges'])
    eoff = np.concatenate([[0], np.cumsum(d['f_n_edges'].astype(np.int64))])
    lq_col = names['edge'].index('link_quality')
    cache = {}

    def get(f):
        fr = cache.get(f)
        if fr is None:
            fr = Frame(d, f, nc, names)
            fr.set_edges(d['f_edge_index'][:, eoff[f]:eoff[f + 1]],
                         d['f_edge_feat'][eoff[f]:eoff[f + 1]], lq_col)
            if len(cache) > 64:
                cache.clear()
            cache[f] = fr
        return fr
    return get, nfr


def audit_contexts(d, meta, nc, names, idx):
    """Re-derive A-D for the context indices `idx`. Returns counters."""
    get, _ = build_frames(d, nc, names)
    off = np.concatenate([[0], np.cumsum(d['c_k'].astype(np.int64))])
    qn = names['query']
    diag = float(np.hypot(nc['area_x'], nc['area_y']))
    out = dict(n=0, a_bad=0, b_agree=0, b_score_maxdiff=0.0, c_rows=0, c_ok=0,
               d_gpsr_bad=0, d_dij_bad=0, d_spbp_bad=0, d_beh_bad=0)
    beh_name = meta['behaviour']
    for i in idx:
        i = int(i)
        fr = get(int(d['c_frame'][i]))
        c, dst, hops = int(d['c_current'][i]), int(d['c_dst'][i]), int(d['c_hops'][i])
        cands = [int(u) for u in d['c_cands'][off[i]:off[i + 1]]]
        out['n'] += 1
        # A
        nb = set(fr.adj[c])
        dd = {u: float(np.linalg.norm(fr.pos[dst] - fr.pos[u])) for u in cands}
        if not set(cands) <= nb or any(dd[cands[j]] > dd[cands[j + 1]] + 1e-3
                                       for j in range(len(cands) - 1)):
            out['a_bad'] += 1
        # B
        dcd = float(np.linalg.norm(fr.pos[dst] - fr.pos[c]))
        sc = {}
        for u in cands:
            prog = (dcd - dd[u]) / max(dcd, 1.0)
            sc[u] = W_PROG * prog - W_QUEUE * fr.occ[u] + W_QUAL * fr.lq[(c, u)]
        best = max(sorted(cands), key=lambda u: (sc[u], -u))   # max score, lowest id
        if best == cands[int(d['c_label'][i])]:
            out['b_agree'] += 1
        st = d['c_scores'][off[i]:off[i + 1]]
        out['b_score_maxdiff'] = max(out['b_score_maxdiff'],
                                     float(np.max(np.abs(st - np.array([sc[u] for u in cands])))))
        # C
        h = fr.bfs(dst)
        want_c = np.array([[np.clip((dcd - dd[u]) / max(dcd, 1.0), -1, 1),
                            min(h.get(u, HOP_CAP) / HOP_CAP, 1.0),
                            1.0 if u == dst else 0.0, 1.0 if u in h else 0.0] for u in cands])
        got_c = d['c_cand_feat'][off[i]:off[i + 1]]
        hood = fr.bfs(c, cutoff=2)
        occs = [fr.occ[u] for u in hood]
        qd = {'ttl_left': max(TTL_REF - hops, 0.0) / TTL_REF,
              'dist_to_dest': min(dcd / max(diag, 1e-6), 1.0),
              'own_queue_live': float(d['c_ownq_first'][i]) / nc['max_queue'],
              'neigh_buffered_packets': min(float(np.sum(occs)) * nc['max_queue'] / nc['buffered_ref'], 1.0),
              'neigh_mean_occupancy': float(np.mean(occs)),
              'hop_distance_to_dst': min(h.get(c, HOP_CAP) / HOP_CAP, 1.0)}
        want_q = np.array([qd[k] for k in qn])
        ok = (np.abs(want_c - got_c).max() <= FEAT_TOL
              and np.abs(want_q - d['c_query'][i]).max() <= FEAT_TOL)
        out['c_rows'] += 1
        out['c_ok'] += int(ok)
        # D
        v = d['c_votes'][i]               # vote order: dijkstra, gpsr, spbp
        gp = min(sorted(cands), key=lambda u: (dd[u], u))
        if int(v[1]) < 0 or cands[int(v[1])] != gp:
            out['d_gpsr_bad'] += 1
        if dst in cands:
            dj_ok = int(v[0]) >= 0 and cands[int(v[0])] == dst
        else:
            reach = [u for u in cands if u in h]
            if c not in h or not reach:
                dj_ok = int(v[0]) == -1
            else:
                dj_ok = int(v[0]) >= 0 and h[cands[int(v[0])]] == min(h[u] for u in reach)
        out['d_dij_bad'] += int(not dj_ok)
        if dst in cands:
            sp = dst
        else:
            qc = round(fr.occ[c] * nc['max_queue'])
            hc = h.get(c, HOP_UNREACHABLE)
            sp, bs = None, -float('inf')
            for u in cands:                                   # spbp iterates cands order
                s = fr.lq[(c, u)] * ((qc - round(fr.occ[u] * nc['max_queue']))
                                     + SPBP_V_BIAS * (hc - h.get(u, HOP_UNREACHABLE)))
                if s > bs:
                    bs, sp = s, u
        if int(v[2]) < 0 or cands[int(v[2])] != sp:
            out['d_spbp_bad'] += 1
        beh = cands[int(d['c_beh'][i])]
        if beh_name == 'gpsr':
            out['d_beh_bad'] += int(beh != gp)
        elif beh_name == 'spbp':
            out['d_beh_bad'] += int(beh != sp)
        elif beh_name == 'dijkstra_nodrop':
            want = cands[int(v[0])] if int(v[0]) >= 0 else cands[0]
            out['d_beh_bad'] += int(beh != want)
        else:
            out['d_beh_bad'] += int(beh != cands[int(d['c_label'][i])])
    return out


def audit_ownq(d, idx):
    """G: plain-Python recount of the own-queue histograms of the sampled contexts."""
    want = {int(i) for i in idx}
    hist = {}
    for c, v, k in zip(d['o_ctx'].tolist(), d['o_ownq'].tolist(), d['o_count'].tolist()):
        if c in want:
            hist.setdefault(c, {})[v] = hist.get(c, {}).get(v, 0) + k
    bad = 0
    for i in want:
        h = hist.get(i, {})
        if sum(h.values()) != int(d['c_mult'][i]) or int(d['c_ownq_first'][i]) not in h:
            bad += 1
    for c, v in zip(d['s_ctx'].tolist(), d['s_ownq'].tolist()):
        if c in want and v not in hist.get(c, {}):
            bad += 1
    return bad


def verdict(cnt):
    n = max(cnt['n'], 1)
    return {
        'A candidates': cnt['a_bad'] == 0,
        'B label': cnt['b_agree'] / n >= LABEL_MIN_AGREE and cnt['b_score_maxdiff'] < 1e-3,
        'C features': cnt['c_ok'] / max(cnt['c_rows'], 1) >= FEAT_MIN_OK,
        'D votes': (cnt['d_gpsr_bad'] + cnt['d_dij_bad'] + cnt['d_beh_bad']) == 0
                   and cnt['d_spbp_bad'] / n <= 1 - LABEL_MIN_AGREE,
        'G own queue': cnt.get('g_bad', 0) == 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join('data', 'v3'))
    ap.add_argument('--per_shard', type=int, default=1500)
    ap.add_argument('--max_shards', type=int, default=None)
    ap.add_argument('--report_dir', default=os.path.join('results', 'dataset_v3'),
                    help='tracked copy of the verdict (data/ is git-ignored)')
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.data, 'manifest.json')))
    names = {'node': man['node_features'], 'edge': man['edge_features'],
             'query': man['query_features'], 'cand': man['candidate_features']}
    print('=' * 78)
    print(f"  INDEPENDENT AUDIT v3 -- {args.data}   episodes={man['totals']['episodes']}")
    print('=' * 78)
    rng = np.random.default_rng(5)
    eps = man['episodes'][:args.max_shards] if args.max_shards else man['episodes']
    tot = None
    for k, e in enumerate(eps):
        meta = json.load(open(os.path.join(args.data, e['shard'].replace('.npz', '.json'))))
        d = dict(np.load(os.path.join(args.data, e['shard'])))
        nc = man['norm_constants_per_scenario'][meta['scenario']]
        n = len(d['c_mult'])
        idx = rng.choice(n, size=min(n, args.per_shard), replace=False) if n else []
        if k == 0:
            # negative controls on this real shard
            bad = {kk: v.copy() for kk, v in d.items()}
            multi = np.where(bad['c_k'] > 1)[0][:200]
            bad['c_label'][multi] = (bad['c_label'][multi] + 1) % bad['c_k'][multi]
            bad['c_cand_feat'][:, 0] += 0.01
            bad['c_votes'][multi, 1] = (bad['c_votes'][multi, 1] + 1) % bad['c_k'][multi]
            cb = audit_contexts(bad, meta, nc, names, multi)
            bad['o_count'][int(np.searchsorted(bad['o_ctx'], multi[0]))] += 1
            cb['g_bad'] = audit_ownq(bad, multi)
            v_bad = verdict(cb)
            dead = [kk for kk in ('B label', 'C features', 'D votes', 'G own queue') if v_bad[kk]]
            print(f"  negative controls: corrupted labels / features / votes / own-queue histogram "
                  f"{'ALL detected' if not dead else 'NOT detected: ' + str(dead)}")
            if dead:
                print('  ABORTED: an audit that cannot fail proves nothing.')
                return 1
        cnt = audit_contexts(d, meta, nc, names, idx)
        cnt['g_bad'] = audit_ownq(d, idx)
        if tot is None:
            tot = {kk: 0 for kk in cnt}
            tot['b_score_maxdiff'] = 0.0
        for kk, v in cnt.items():
            tot[kk] = max(tot[kk], v) if kk == 'b_score_maxdiff' else tot[kk] + v
        if (k + 1) % 25 == 0 or k + 1 == len(eps):
            print(f'    audited {k + 1}/{len(eps)} shards, {tot["n"]:,} contexts')
    res = verdict(tot)
    n = max(tot['n'], 1)
    # E splits
    seeds = {}
    for e in man['episodes']:
        seeds.setdefault(e['split'], set()).add(e['seed'])
    sp = list(seeds.values())
    disjoint = all(not (a & b) for i, a in enumerate(sp) for b in sp[i + 1:])
    plan = man['split_plan']
    in_range = all(plan[f'{s}_seeds'][0] <= sd <= plan[f'{s}_seeds'][1]
                   for s, ss in seeds.items() for sd in ss)
    res['E splits'] = disjoint and in_range
    # F hashing (independent splitmix64)
    tab = man['behaviour']['table']
    share, others = man['behaviour']['other_share'], man['behaviour']['other_teachers']
    ok_tab = True
    for k, (split, key) in enumerate((('train', 'train_seeds'), ('val', 'val_seeds'),
                                      ('test', 'test_seeds'))):
        lo, hi = plan[key]
        ss = list(range(lo, hi + 1))
        order = sorted(ss, key=lambda s: (_sm64(s, 0, man['hash']['salts']['behaviour']), s))
        n_o = int(share * len(ss) + 0.5)
        for i, s in enumerate(order):
            want = others[(i + 2 * k) % len(others)] if i < n_o else 'da_gpsr'
            ok_tab &= tab[str(s)] == want
    e0 = man['episodes'][0]
    d0 = dict(np.load(os.path.join(args.data, e0['shard'])))
    frac = man['rl_packet_frac'][0]
    ok_pk = all(_sm64(e0['seed'], int(p), man['hash']['salts']['packet']) < frac for p in d0['p_pid'])
    res['F hashing'] = bool(ok_tab and ok_pk)
    print(f"\n    contexts audited {tot['n']:,}")
    print(f"    A candidates not neighbours / not canonical: {tot['a_bad']}")
    print(f"    B label agreement {tot['b_agree'] / n:.5f} (min {LABEL_MIN_AGREE}); "
          f"max |stored - re-derived score| {tot['b_score_maxdiff']:.2e}")
    print(f"    C feature rows within {FEAT_TOL}: {tot['c_ok'] / max(tot['c_rows'], 1):.5f} (min {FEAT_MIN_OK})")
    print(f"    D vote mismatches: gpsr {tot['d_gpsr_bad']}  dijkstra {tot['d_dij_bad']}  "
          f"spbp {tot['d_spbp_bad']}  behaviour choice {tot['d_beh_bad']}")
    print(f"    G own-queue histogram / step mismatches: {tot['g_bad']}")
    print('\n  VERDICT')
    for k, v in res.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(res.values())
    print('\n  AUDIT v3 ' + ('PASS' if ok else 'FAIL'))
    with open(os.path.join(args.data, 'audit_v3.json'), 'w') as f:
        json.dump({'results': res, 'counts': tot}, f, indent=1)
    tag = os.path.basename(os.path.normpath(args.data))     # tracked copy (data/ is git-ignored)
    os.makedirs(args.report_dir, exist_ok=True)
    json.dump({'results': res, 'counts': tot},
              open(os.path.join(args.report_dir, f'{tag}_audit_v3.json'), 'w'), indent=1)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
