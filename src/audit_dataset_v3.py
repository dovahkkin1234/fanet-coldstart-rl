"""
audit_dataset_v3.py  --  INDEPENDENT audit of Dataset V3 (docs/DATASET_V3_SPEC.md §7).

    python src\\audit_dataset_v3.py --data data\\v3m --max_workers 12
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
  E  splits       seeds partition into train/val/test (disjoint, inside the
                  split plan's ranges) and the manifest names the configured
                  generalisation scenario (v31: the hold-out by scenario itself
                  is applied downstream, by export_phaseb_v3 and PhaseB)
  F  hashing      behaviour table and packet sample recomputed with an
                  independent splitmix64 implementation (v31: the first shard's
                  sampled packets must EQUAL the rule over every generated packet)
  G  own queue    (not re-derivable from frames: it is live) each sampled
                  context's own-queue histogram counts exactly its multiplicity
                  and contains its first-occurrence value; every step of a
                  sampled context has an own queue the histogram contains
Each re-derivation is first shown to FAIL on a corrupted copy of a real shard.

v30 -- FLAGGED CONTEXTS ARE RE-CHECKED ON THE EXACT FRAME. The stored frame is
float32, and two terms amplify its rounding far beyond 1e-7:
  * da_gpsr progress (d_cd - d_nd) / max(d_cd, 1) is unbounded; its error grows
    like d_nd / d_cd^2 when the deciding node is a few metres from the destination
    (measured: 3e-6 at d_cd > 50 m, 3e-5 at 10-25 m);
  * SP-BP scores are link quality x an integer (queue + hop gradient), so float32
    rounding can make two scores equal or unequal (on v3m: 15 of the 16 SP-BP flags
    were float64 scores one rounding step apart that float32 made exactly equal, 1
    was an exact float64 tie that float32 broke).
On data\\v3m (2026-10-08) that alone failed the v26 audit: max score difference
1.10e-3 against the 1e-3 tolerance, 16 SP-BP vote mismatches and, in the SP-BP
behaviour episodes, 3 behaviour-choice mismatches (the same SP-BP picks, counted
again at zero tolerance). The tolerances are NOT changed. Instead every context that
fails a precision-sensitive check on the float32 frame (B score or label, D gpsr or
spbp vote, behaviour choice) is re-derived -- with the same function and the same
tolerances -- from the exact float64 frame, recreated by replaying its episode
(episodes are deterministic: G3.5 check 7). A replay counts only if it reproduces
every stored frame array byte for byte; otherwise the context keeps its float32
result. The replay uses the simulator only to recreate positions, queues and link
quality; no teacher or feature code takes part in a re-derivation. Both sets of
counts are reported (counts_float32 = what v26 printed); the verdict uses the
re-checked counts. --exact_replay off restores the v26 verdict.
Also v30: shards are audited in parallel (--max_workers) with exactly the v26 sample
(same generator, same order), and every flagged context is written to the report.

v31 -- EVERY CHECK IS SHOWN TO FAIL. Corrupted copies now also exercise A (a candidate
list out of canonical order), D (dijkstra votes changed; spbp votes changed in every
sampled context, far above the 1% tolerance), E (a seed outside the split plan; a seed in
two splits) and F (a behaviour-table entry changed; one sampled packet removed). No
tolerance and no verdict on v3m changes.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AUDIT_VERSION = 'v31'
TTL_REF = 20.0
HOP_CAP = 10.0
W_PROG, W_QUEUE, W_QUAL = 1.0, 1.0, 0.5       # da_gpsr weights (routing_teachers_v2)
SPBP_V_BIAS = 1.0                              # routing_teachers_v2.SPBP_V_BIAS
HOP_UNREACHABLE = 999.0
LABEL_MIN_AGREE = 0.99
SCORE_TOL = 1e-3                               # max |stored - re-derived score| (v26, unchanged)
FEAT_TOL = 1e-4
FEAT_MIN_OK = 0.999
FRAME_KEYS = ('f_node_feat', 'f_n_nodes', 'f_edge_index', 'f_edge_feat', 'f_n_edges')
SUM_KEYS = ('a_bad', 'b_agree', 'c_ok', 'd_gpsr_bad', 'd_dij_bad', 'd_spbp_bad', 'd_beh_bad')
RECHECK_KEYS = ('a_bad', 'b_agree', 'd_gpsr_bad', 'd_dij_bad', 'd_spbp_bad', 'd_beh_bad')
MAX_RECORDED_PER_SHARD = 50


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


class ExactFrame(Frame):
    """v30: a frame of the REPLAYED episode, in float64 (positions in metres, queue
    occupancy, link quality), with the same adjacency and BFS helpers."""

    def __init__(self, pos, occ, edges):            # pylint: disable=super-init-not-called
        self.pos = np.asarray(pos, np.float64)
        self.occ = np.asarray(occ, np.float64)
        self.N = len(self.pos)
        self.adj = {i: [] for i in range(self.N)}
        self.lq = {}
        for a, b, lq in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)
            self.lq[(a, b)] = self.lq[(b, a)] = float(lq)
        for a in self.adj:
            self.adj[a].sort()


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


def derive(fr, d, i, off, nc, qn, diag, beh_name, features=True):
    """Re-derive A-D for context i on frame `fr` (stored float32 Frame or replayed
    ExactFrame -- the same arithmetic either way). Returns per-context results."""
    c, dst, hops = int(d['c_current'][i]), int(d['c_dst'][i]), int(d['c_hops'][i])
    cands = [int(u) for u in d['c_cands'][off[i]:off[i + 1]]]
    r = {'frame': int(d['c_frame'][i]), 'current': c, 'dst': dst, 'k': len(cands)}
    # A
    nb = set(fr.adj[c])
    dd = {u: float(np.linalg.norm(fr.pos[dst] - fr.pos[u])) for u in cands}
    r['a_bad'] = int(not set(cands) <= nb or any(dd[cands[j]] > dd[cands[j + 1]] + 1e-3
                                                 for j in range(len(cands) - 1)))
    # B
    dcd = float(np.linalg.norm(fr.pos[dst] - fr.pos[c]))
    sc = {}
    for u in cands:
        prog = (dcd - dd[u]) / max(dcd, 1.0)
        sc[u] = W_PROG * prog - W_QUEUE * fr.occ[u] + W_QUAL * fr.lq[(c, u)]
    best = max(sorted(cands), key=lambda u: (sc[u], -u))   # max score, lowest id
    r['b_agree'] = int(best == cands[int(d['c_label'][i])])
    st = d['c_scores'][off[i]:off[i + 1]]
    r['b_score_diff'] = float(np.max(np.abs(st - np.array([sc[u] for u in cands]))))
    r['d_cd'] = dcd
    # C
    h = fr.bfs(dst)
    if features:
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
        r['c_ok'] = int(np.abs(want_c - got_c).max() <= FEAT_TOL
                        and np.abs(want_q - d['c_query'][i]).max() <= FEAT_TOL)
    # D
    v = d['c_votes'][i]               # vote order: dijkstra, gpsr, spbp
    gp = min(sorted(cands), key=lambda u: (dd[u], u))
    r['d_gpsr_bad'] = int(int(v[1]) < 0 or cands[int(v[1])] != gp)
    if dst in cands:
        dj_ok = int(v[0]) >= 0 and cands[int(v[0])] == dst
    else:
        reach = [u for u in cands if u in h]
        if c not in h or not reach:
            dj_ok = int(v[0]) == -1
        else:
            dj_ok = int(v[0]) >= 0 and h[cands[int(v[0])]] == min(h[u] for u in reach)
    r['d_dij_bad'] = int(not dj_ok)
    spm = None
    if dst in cands:
        sp = dst
    else:
        qc = round(fr.occ[c] * nc['max_queue'])
        hc = h.get(c, HOP_UNREACHABLE)
        sp, bs, s2 = None, -float('inf'), -float('inf')
        for u in cands:                                   # spbp iterates cands order
            s = fr.lq[(c, u)] * ((qc - round(fr.occ[u] * nc['max_queue']))
                                 + SPBP_V_BIAS * (hc - h.get(u, HOP_UNREACHABLE)))
            if s > bs:
                bs, s2, sp = s, bs, u
            elif s > s2:
                s2 = s
        spm = bs - s2 if len(cands) > 1 else None
    r['d_spbp_bad'] = int(int(v[2]) < 0 or cands[int(v[2])] != sp)
    beh = cands[int(d['c_beh'][i])]
    if beh_name == 'gpsr':
        r['d_beh_bad'] = int(beh != gp)
    elif beh_name == 'spbp':
        r['d_beh_bad'] = int(beh != sp)
    elif beh_name == 'dijkstra_nodrop':
        want = cands[int(v[0])] if int(v[0]) >= 0 else cands[0]
        r['d_beh_bad'] = int(beh != want)
    else:
        r['d_beh_bad'] = int(beh != cands[int(d['c_label'][i])])
    vote = lambda j: cands[int(v[j])] if int(v[j]) >= 0 else None
    r['picks'] = {'label': cands[int(d['c_label'][i])], 'da_gpsr': best, 'gpsr_vote': vote(1),
                  'gpsr': gp, 'spbp_vote': vote(2), 'spbp': sp, 'dijkstra_vote': vote(0),
                  'behaviour': beh}
    r['spbp_margin'] = spm
    return r


def is_flagged(r, score_tol):
    """Precision-sensitive checks this context fails on the frame it was derived on."""
    return [k for k, bad in (('B score', r['b_score_diff'] >= score_tol),
                             ('B label', not r['b_agree']), ('D gpsr', r['d_gpsr_bad']),
                             ('D spbp', r['d_spbp_bad']), ('D behaviour', r['d_beh_bad'])) if bad]


def _new_counts():
    return dict(n=0, a_bad=0, b_agree=0, b_score_maxdiff=0.0, c_rows=0, c_ok=0, d_gpsr_bad=0,
                d_dij_bad=0, d_spbp_bad=0, d_beh_bad=0, b_score_maxdiff_unflagged=0.0)


def audit_contexts(d, meta, nc, names, idx, score_tol=None):
    """Re-derive A-D for the context indices `idx` on the STORED frames. Returns
    (counters, {context index: results} for the flagged contexts)."""
    score_tol = SCORE_TOL if score_tol is None else score_tol
    get, _ = build_frames(d, nc, names)
    off = np.concatenate([[0], np.cumsum(d['c_k'].astype(np.int64))])
    qn = names['query']
    diag = float(np.hypot(nc['area_x'], nc['area_y']))
    out = _new_counts()
    flagged = {}
    for i in idx:
        i = int(i)
        r = derive(get(int(d['c_frame'][i])), d, i, off, nc, qn, diag, meta['behaviour'])
        out['n'] += 1
        out['c_rows'] += 1
        for k in SUM_KEYS:
            out[k] += r[k]
        out['b_score_maxdiff'] = max(out['b_score_maxdiff'], r['b_score_diff'])
        if is_flagged(r, score_tol):
            flagged[i] = r
        else:
            out['b_score_maxdiff_unflagged'] = max(out['b_score_maxdiff_unflagged'], r['b_score_diff'])
    return out, flagged


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
        'B label': cnt['b_agree'] / n >= LABEL_MIN_AGREE and cnt['b_score_maxdiff'] < SCORE_TOL,
        'C features': cnt['c_ok'] / max(cnt['c_rows'], 1) >= FEAT_MIN_OK,
        'D votes': (cnt['d_gpsr_bad'] + cnt['d_dij_bad'] + cnt['d_beh_bad']) == 0
                   and cnt['d_spbp_bad'] / n <= 1 - LABEL_MIN_AGREE,
        'G own queue': cnt.get('g_bad', 0) == 0,
    }


# ── parallel per-shard audit (the v26 sample, unchanged) ─────────────────────
def _map(fn, jobs, workers):
    """Results of fn over jobs, as they complete; in-process when workers <= 1."""
    if workers <= 1:
        for j in jobs:
            yield fn(j)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for fu in as_completed(futs):
            yield fu.result()


def _norm_names(man):
    return {'node': man['node_features'], 'edge': man['edge_features'],
            'query': man['query_features'], 'cand': man['candidate_features']}


def audit_shard(job):
    data, e, idx, nc, names, score_tol = job
    meta = json.load(open(os.path.join(data, e['shard'].replace('.npz', '.json'))))
    d = dict(np.load(os.path.join(data, e['shard'])))
    if len(d['c_mult']) != int(e['n_contexts']):
        raise AssertionError(f"{e['shard']}: {len(d['c_mult'])} contexts on disk, "
                             f"{e['n_contexts']} in the manifest (the sample is drawn from the latter)")
    cnt, flagged = audit_contexts(d, meta, nc, names, idx, score_tol)
    cnt['g_bad'] = audit_ownq(d, idx)
    return e['shard'], meta['behaviour'], cnt, flagged


# ── v30: exact re-check of flagged contexts by replaying the episode ─────────
def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def replay_exact(data, e, frames):
    """Replay the episode of shard `e` and capture `frames` in float64. Returns
    ({frame: ExactFrame}, report). The frames are returned only if the replay
    reproduces every stored FRAME array byte for byte (report['exact'])."""
    import config_v2 as C                         # lazy: only flagged shards pay for these
    import generate_dataset_v3 as G
    meta = json.load(open(os.path.join(data, e['shard'].replace('.npz', '.json'))))
    rep = {'shard': e['shard'], 'key': meta['key'], 'exact': False}
    if meta.get('code_signature') != G.CODE_SIGNATURE:
        rep['why'] = (f"shard written by code {meta.get('code_signature')}, current code "
                      f"{G.CODE_SIGNATURE}: the replay would be another episode")
        return {}, rep
    cfg = C.dataset_episode_config(meta['scenario'], meta['rate'], meta['seed'],
                                   scenario_id=meta['scenario'], actor='da_gpsr',
                                   duration=float(meta['config']['duration']))
    if G._jsonable(cfg) != meta['config']:
        rep['why'] = 'the episode config rebuilt from the shard metadata differs from the recorded one'
        return {}, rep
    want = {int(f) for f in frames}
    cap = {}

    class _Replay(G.DatasetSimulatorV3):
        def _build_graph(self):
            g = super()._build_graph()
            if self._fid in want:                 # _fid = index of the frame just built
                ids = sorted(g.nodes())
                cap[self._fid] = ([[g.nodes[n]['x'], g.nodes[n]['y'], g.nodes[n]['z']] for n in ids],
                                  [g.nodes[n]['queue_occupancy'] for n in ids],
                                  [(u, v, a['link_quality']) for u, v, a in g.edges(data=True)])
            return g

    t0 = time.time()
    sim = _Replay(cfg, meta['behaviour'], epsilon=meta['epsilon'], rl_frac=meta['rl_packet_frac'])
    sim.run()
    arrs = sim.arrays()
    rep['seconds'] = round(time.time() - t0, 1)
    with np.load(os.path.join(data, e['shard'])) as stored:
        rep['arrays_differing_from_stored'] = sorted(k for k in arrs if k not in stored.files
                                                     or _sha(arrs[k]) != _sha(stored[k]))
    rep['missing_frames'] = sorted(want - set(cap))
    rep['exact'] = not (set(FRAME_KEYS) & set(rep['arrays_differing_from_stored'])) and not rep['missing_frames']
    if not rep['exact']:
        rep['why'] = ('replayed frames differ from the stored ones: '
                      f"{sorted(set(FRAME_KEYS) & set(rep['arrays_differing_from_stored']))}"
                      f"{' missing ' + str(rep['missing_frames']) if rep['missing_frames'] else ''}")
        return {}, rep
    return {f: ExactFrame(*cap[f]) for f in want}, rep


def recheck_shard(job):
    data, e, nc, names, flagged_ctx, score_tol = job
    try:
        frames, rep = replay_exact(data, e, sorted({r['frame'] for r in flagged_ctx.values()}))
    except Exception as ex:                              # noqa: BLE001 -- reported, never hidden
        return e['shard'], {'shard': e['shard'], 'key': e.get('key', e['shard']), 'exact': False,
                            'why': f'the replay raised {type(ex).__name__}: {ex}'}, {}
    if not rep['exact']:
        return e['shard'], rep, {}
    meta = json.load(open(os.path.join(data, e['shard'].replace('.npz', '.json'))))
    d = dict(np.load(os.path.join(data, e['shard'])))
    off = np.concatenate([[0], np.cumsum(d['c_k'].astype(np.int64))])
    diag = float(np.hypot(nc['area_x'], nc['area_y']))
    exact = {i: derive(frames[r['frame']], d, i, off, nc, names['query'], diag, meta['behaviour'],
                       features=False) for i, r in flagged_ctx.items()}
    rep['resolved'] = sum(not is_flagged(r, score_tol) for r in exact.values())
    return e['shard'], rep, exact


def _negative_controls(d, meta, nc, names):
    """v26 controls on a real shard (labels / features / gpsr votes / own-queue
    histogram corrupted), plus v30: a behaviour choice corrupted in an episode driven
    by each non-da_gpsr teacher. Returns the checks that did NOT fire."""
    bad = {kk: v.copy() for kk, v in d.items()}
    multi = np.where(bad['c_k'] > 1)[0][:200]
    bad['c_label'][multi] = (bad['c_label'][multi] + 1) % bad['c_k'][multi]
    bad['c_cand_feat'][:, 0] += 0.01
    bad['c_votes'][multi, 1] = (bad['c_votes'][multi, 1] + 1) % bad['c_k'][multi]
    cb, _ = audit_contexts(bad, meta, nc, names, multi)
    bad['o_count'][int(np.searchsorted(bad['o_ctx'], multi[0]))] += 1
    cb['g_bad'] = audit_ownq(bad, multi)
    v_bad = verdict(cb)
    dead = [kk for kk in ('B label', 'C features', 'D votes', 'G own queue') if v_bad[kk]]
    for teacher, slot in (('gpsr', 1), ('spbp', 2), ('dijkstra_nodrop', 0)):
        ok = dict(d, c_beh=d['c_beh'].copy())
        ok['c_beh'][multi] = np.where(ok['c_votes'][multi, slot] >= 0, ok['c_votes'][multi, slot], 0)
        clean, _ = audit_contexts(ok, dict(meta, behaviour=teacher), nc, names, multi)
        ok['c_beh'][multi] = (ok['c_beh'][multi] + 1) % ok['c_k'][multi]
        corrupt, _ = audit_contexts(ok, dict(meta, behaviour=teacher), nc, names, multi)
        # the clean copy may disagree only where the stored vote itself is re-derived differently
        if corrupt['d_beh_bad'] <= clean['d_beh_bad']:
            dead.append(f'D behaviour ({teacher})')
    # v31: A -- candidate lists out of canonical order
    x = dict(d, c_cands=d['c_cands'].copy())
    off = np.concatenate([[0], np.cumsum(d['c_k'].astype(np.int64))])
    for i in multi[:50]:
        x['c_cands'][off[i]:off[i + 1]] = x['c_cands'][off[i]:off[i + 1]][::-1].copy()
    ca, _ = audit_contexts(x, meta, nc, names, multi[:50])
    if ca['a_bad'] == 0:
        dead.append('A candidates')
    # v31: D -- dijkstra votes changed (a vote set to -1, a -1 vote set to slot 0)
    x = dict(d, c_votes=d['c_votes'].copy())
    v0 = x['c_votes'][multi, 0]
    x['c_votes'][multi, 0] = np.where(v0 >= 0, -1, 0).astype(x['c_votes'].dtype)
    cd, _ = audit_contexts(x, meta, nc, names, multi)
    if cd['d_dij_bad'] == 0:
        dead.append('D dijkstra votes')
    # v31: D -- spbp votes changed in every sampled context (far above the 1% tolerance)
    x = dict(d, c_votes=d['c_votes'].copy())
    x['c_votes'][multi, 2] = ((x['c_votes'][multi, 2] + 1) % x['c_k'][multi]).astype(x['c_votes'].dtype)
    cs, _ = audit_contexts(x, meta, nc, names, multi)
    if cs['d_spbp_bad'] == 0 or verdict(dict(cs, g_bad=0))['D votes']:
        dead.append('D spbp votes')
    return dead


def check_splits(man):
    """E: seeds partition into the splits (disjoint, inside the split plan's ranges) and the
    manifest names the configured generalisation scenario. The hold-out by scenario itself
    is applied downstream (export_phaseb_v3 gives that scenario its own split; PhaseB
    asserts the partition)."""
    import config_v2 as C
    seeds = {}
    for e in man['episodes']:
        seeds.setdefault(e['split'], set()).add(e['seed'])
    sp = list(seeds.values())
    disjoint = all(not (a & b) for i, a in enumerate(sp) for b in sp[i + 1:])
    plan = man['split_plan']
    in_range = all(s in ('train', 'val', 'test') and plan[f'{s}_seeds'][0] <= sd <= plan[f'{s}_seeds'][1]
                   for s, ss in seeds.items() for sd in ss)
    gen_ok = plan.get('generalisation_scenario') == C.GENERALISATION_SCENARIO
    return bool(disjoint and in_range and gen_ok)


def check_hashing(man, e0, meta0, d0):
    """F: the behaviour table and the first shard's packet sample, recomputed with the
    independent splitmix64 above. v31: the sampled set must EQUAL the rule over every
    generated packet (before, it only had to satisfy it)."""
    plan = man['split_plan']
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
    frac = man['rl_packet_frac'][0]
    salt = man['hash']['salts']['packet']
    n_gen = int(meta0['metrics']['n_generated'])
    want = {p for p in range(n_gen) if _sm64(e0['seed'], p, salt) < frac}
    got = {int(p) for p in d0['p_pid']}
    return bool(ok_tab and want == got)


def _negative_controls_manifest(man, e0, meta0, d0):
    """v31: E and F on corrupted copies. Returns the controls that did NOT fire."""
    dead = []
    bad = json.loads(json.dumps(man))
    bad['episodes'][0]['seed'] = bad['split_plan']['train_seeds'][0] - 1    # outside every range
    if check_splits(bad):
        dead.append('E splits (seed outside the plan)')
    bad = json.loads(json.dumps(man))
    eps = bad['episodes']
    a = next((e for e in eps if e['split'] == 'train'), None)
    b = next((e for e in eps if e['split'] != 'train'), None)
    if a is not None and b is not None:
        b['seed'] = a['seed']                                               # a seed in two splits
        if check_splits(bad):
            dead.append('E splits (seed in two splits)')
    bad = json.loads(json.dumps(man))
    k0 = next(iter(bad['behaviour']['table']))
    bad['behaviour']['table'][k0] = 'gpsr' if bad['behaviour']['table'][k0] != 'gpsr' else 'da_gpsr'
    if check_hashing(bad, e0, meta0, d0):
        dead.append('F hashing (behaviour table)')
    if len(d0['p_pid']) and check_hashing(man, e0, meta0, dict(d0, p_pid=d0['p_pid'][1:])):
        dead.append('F hashing (packet sample)')
    return dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join('data', 'v3'))
    ap.add_argument('--per_shard', type=int, default=1500)
    ap.add_argument('--max_shards', type=int, default=None)
    ap.add_argument('--max_workers', type=int, default=8,
                    help='processes for the per-shard audit and the replays (the sample does not '
                         'depend on it)')
    ap.add_argument('--exact_replay', choices=('flagged', 'off'), default='flagged',
                    help="'flagged' (default): re-check every flagged context on its exact frame by "
                         "replaying the episode; 'off': the v26 verdict on the float32 frames")
    ap.add_argument('--report_dir', default=os.path.join('results', 'dataset_v3'),
                    help='tracked copy of the verdict (data/ is git-ignored)')
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.data, 'manifest.json')))
    names = _norm_names(man)
    print('=' * 78)
    print(f"  INDEPENDENT AUDIT v3 ({AUDIT_VERSION}) -- {args.data}   episodes={man['totals']['episodes']}")
    print('=' * 78)
    eps = man['episodes'][:args.max_shards] if args.max_shards else man['episodes']
    # the v26 sample: one generator, shards in manifest order, drawn from each shard's size
    rng = np.random.default_rng(5)
    samples = []
    for e in eps:
        n = int(e['n_contexts'])
        samples.append(rng.choice(n, size=min(n, args.per_shard), replace=False) if n else np.zeros(0, np.int64))
    # negative controls on the first real shard
    e0 = eps[0]
    meta0 = json.load(open(os.path.join(args.data, e0['shard'].replace('.npz', '.json'))))
    d0 = dict(np.load(os.path.join(args.data, e0['shard'])))
    dead = _negative_controls(d0, meta0, man['norm_constants_per_scenario'][meta0['scenario']], names)
    dead += _negative_controls_manifest(man, e0, meta0, d0)
    print(f"  negative controls: corrupted labels / features / votes (gpsr, dijkstra, spbp) / "
          f"own-queue histogram / behaviour choice (gpsr, spbp, dijkstra_nodrop episodes) / "
          f"candidate order / splits / hashing "
          f"{'ALL detected' if not dead else 'NOT detected: ' + str(dead)}")
    if dead:
        print('  ABORTED: an audit that cannot fail proves nothing.')
        return 1
    # float32 pass, in parallel
    t0 = time.time()
    tot = None
    flagged, beh_of = {}, {}
    jobs = [(args.data, e, idx, man['norm_constants_per_scenario'][e['scenario']], names, SCORE_TOL)
            for e, idx in zip(eps, samples)]
    by_shard = {e['shard']: e for e in eps}
    for k, (shard, beh, cnt, fl) in enumerate(_map(audit_shard, jobs, args.max_workers), 1):
        beh_of[shard] = beh
        if fl:
            flagged[shard] = fl
        if tot is None:
            tot = {kk: 0 for kk in cnt}
        for kk, v in cnt.items():
            tot[kk] = max(tot[kk], v) if kk.startswith('b_score_maxdiff') else tot[kk] + v
        if k % 25 == 0 or k == len(jobs):
            print(f'    audited {k}/{len(jobs)} shards, {tot["n"]:,} contexts  ({time.time() - t0:.0f}s)',
                  flush=True)
    tot32 = {k: v for k, v in tot.items() if k != 'b_score_maxdiff_unflagged'}
    res32 = verdict(tot32)
    n = max(tot['n'], 1)
    n_fl = sum(len(v) for v in flagged.values())

    def summary(c, label):
        print(f"\n    {label}")
        print(f"    A candidates not neighbours / not canonical: {c['a_bad']}")
        print(f"    B label agreement {c['b_agree'] / n:.5f} ({n - c['b_agree']} disagree; min "
              f"{LABEL_MIN_AGREE}); max |stored - re-derived score| {c['b_score_maxdiff']:.2e} "
              f"(tolerance {SCORE_TOL:g})")
        print(f"    C feature rows within {FEAT_TOL}: {c['c_ok'] / max(c['c_rows'], 1):.5f} (min {FEAT_MIN_OK})")
        print(f"    D vote mismatches: gpsr {c['d_gpsr_bad']}  dijkstra {c['d_dij_bad']}  "
              f"spbp {c['d_spbp_bad']} (max {1 - LABEL_MIN_AGREE:.0%})  behaviour choice {c['d_beh_bad']}")
        print(f"    G own-queue histogram / step mismatches: {c['g_bad']}")

    print(f"\n    contexts audited {tot['n']:,}")
    summary(tot32, 'ON THE STORED float32 FRAMES (the v26 audit):')
    # v30: exact re-check
    tot_x = dict(tot32)
    records, replays = [], []
    if n_fl and args.exact_replay == 'flagged':
        print(f"\n    {n_fl} flagged context(s) in {len(flagged)} shard(s): replaying "
              f"{len(flagged)} episode(s) to re-check them on the exact frame ...", flush=True)
        t1 = time.time()
        rjobs = [(args.data, by_shard[s], man['norm_constants_per_scenario'][by_shard[s]['scenario']],
                  names, fl, SCORE_TOL) for s, fl in flagged.items()]
        exact_all = {}
        for k, (shard, rep, exact) in enumerate(_map(recheck_shard, rjobs,
                                                     min(args.max_workers, len(rjobs))), 1):
            replays.append(rep)
            exact_all[shard] = exact
            msg = (f"frames byte-identical, {rep['resolved']}/{len(exact)} resolved"
                   if rep['exact'] else f"NOT USED: {rep.get('why')}")
            other = [k2 for k2 in rep.get('arrays_differing_from_stored', []) if k2 not in FRAME_KEYS]
            print(f"      [{k}/{len(rjobs)}] {rep['key']:<24} {msg}"
                  f"{'  (other arrays differ: ' + str(other) + ')' if other else ''}  "
                  f"({rep.get('seconds', 0):.0f}s)", flush=True)
        mx = tot['b_score_maxdiff_unflagged']
        for shard, fl in flagged.items():
            ex_r = exact_all.get(shard, {})
            for i, r32 in fl.items():
                rx = ex_r.get(i)
                use = rx if rx is not None else r32
                for kk in RECHECK_KEYS:
                    tot_x[kk] += use[kk] - r32[kk]
                mx = max(mx, use['b_score_diff'])
        tot_x['b_score_maxdiff'] = mx
        print(f'      replays done ({time.time() - t1:.0f}s)')
        summary(tot_x, 'AFTER THE EXACT RE-CHECK OF THE FLAGGED CONTEXTS (v30 verdict):')
    elif n_fl:
        print(f"\n    {n_fl} flagged context(s) NOT re-checked (--exact_replay off): verdict on float32")
        exact_all = {}
    else:
        exact_all = {}
    for shard, fl in sorted(flagged.items()):
        for j, (i, r32) in enumerate(sorted(fl.items())):
            if j >= MAX_RECORDED_PER_SHARD:
                break
            rx = exact_all.get(shard, {}).get(i)
            records.append({'shard': shard, 'behaviour': beh_of[shard], 'context': i,
                            'frame': r32['frame'], 'current': r32['current'], 'dst': r32['dst'],
                            'k': r32['k'], 'd_cd_m': round(r32['d_cd'], 4),
                            'float32': {'fails': is_flagged(r32, SCORE_TOL), 'score_diff': r32['b_score_diff'],
                                        'picks': r32['picks'], 'spbp_margin': r32['spbp_margin']},
                            'exact': (None if rx is None else
                                      {'fails': is_flagged(rx, SCORE_TOL), 'score_diff': rx['b_score_diff'],
                                       'picks': rx['picks'], 'spbp_margin': rx['spbp_margin']})})
    res = verdict(tot_x)
    res['E splits'] = res32['E splits'] = check_splits(man)
    res['F hashing'] = res32['F hashing'] = check_hashing(man, e0, meta0, d0)
    print('\n  VERDICT' + (' (float32 frames only)' if args.exact_replay == 'off' or not n_fl else
                         '   (v26 float32 verdict in brackets)'))
    for k, v in res.items():
        extra = '' if (args.exact_replay == 'off' or not n_fl) else f"   [{'PASS' if res32[k] else 'FAIL'}]"
        print(f"    [{'PASS' if v else 'FAIL'}] {k}{extra}")
    ok = all(res.values())
    print('\n  AUDIT v3 ' + ('PASS' if ok else 'FAIL'))
    report = {'audit_version': AUDIT_VERSION, 'results': res, 'counts': tot_x,
              'results_float32': res32, 'counts_float32': tot32,
              'tolerances': {'label_min_agree': LABEL_MIN_AGREE, 'score': SCORE_TOL,
                             'feature': FEAT_TOL, 'feature_min_ok': FEAT_MIN_OK,
                             'spbp_max_mismatch': round(1 - LABEL_MIN_AGREE, 6)},
              'exact_replay': args.exact_replay, 'per_shard': args.per_shard,
              'n_flagged': n_fl, 'flagged': records, 'replays': replays}
    with open(os.path.join(args.data, 'audit_v3.json'), 'w') as f:
        json.dump(report, f, indent=1)
    tag = os.path.basename(os.path.normpath(args.data))     # tracked copy (data/ is git-ignored)
    os.makedirs(args.report_dir, exist_ok=True)
    with open(os.path.join(args.report_dir, f'{tag}_audit_v3.json'), 'w') as f:
        json.dump(report, f, indent=1)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
