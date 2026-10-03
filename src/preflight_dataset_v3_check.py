"""
preflight_dataset_v3_check.py  --  GATE G3.5 v3 for Dataset V3 (docs/DATASET_V3_SPEC.md §7).

    python src\\preflight_dataset_v3_check.py --data data\\v3
    python src\\preflight_dataset_v3_check.py --data data\\v3_smoke --skip_repro

Streams one episode shard at a time (memory-bounded). Every per-shard check is a
function that returns a list of problems, and every one of them is first run on a
deliberately corrupted copy of a real shard and must report the corruption --
a check that cannot fail is not run (project rule; it caught real bugs before).

CHECKS
  0  schema         record/feature schema versions, feature lists, local horizon,
                    norm constants identical across a scenario's episodes
  1  structure      offsets, shapes, id ranges, finiteness, meta counts
  2  labels         0 <= label < k, the label carries the maximum stored score,
                    votes / behaviour choice in range
  3  trivial        DIAGNOSTIC: slot-0 (nearest-to-destination = gpsr) share,
                    'hard' share (label != slot 0), per cell
  4  behaviour      the seed's behaviour policy matches the manifest table; on
                    da_gpsr episodes behaviour == label; every non-epsilon step's
                    action == the behaviour choice; the reserved drop action -1
                    never appears; epsilon fidelity per episode and pooled
  5  coverage       every grid cell x seed present and complete; mix per split
  6  features       finite, in range, no dead column (sampled; query.own_queue_live
                    drawn from each context's own-queue histogram, as the export does)
  7  reproducible   one episode regenerated from scratch has byte-identical arrays
                    (SHA-256) -- the check G3.5 v2 only claimed to make
  8  redundancy     no duplicated column inside a block (Pearson / Spearman)
  9  dedup          context keys unique within a frame; multiplicities sum to
                    the recorded decisions; each context's own-queue histogram
                    sums to its multiplicity and holds its first-occurrence value
                    (which is what c_query stores); every step's live own queue is
                    in its context's histogram
  10 chains         per sampled packet: hops contiguous from 0 at its source,
                    every non-final step 'moved', the next step starts where the
                    action sent the packet (s -> a -> s'), the final step agrees
                    with the packet's fate, hop count == successful hops
  11 sampling       the sampled packet set is EXACTLY {pid: h64(seed,pid) < frac}
  12 counters       cumulative frame counters monotone and equal to the episode
                    metrics
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_v2 as F                                         # noqa: E402
import generate_dataset_v3 as G                                 # noqa: E402
import config_v2 as C                                           # noqa: E402
from preflight_dataset_v2_check import (saturation_report,      # noqa: E402
                                        _corr_matrix, PEARSON_MAX, SPEARMAN_MAX)

EPS_TOL_EPISODE = 0.02
EPS_TOL_POOLED = 0.005
SAMPLE_PER_SHARD = 4000
MOVED, DELIVERED = G.HOP_OUTCOMES.index('moved'), G.HOP_OUTCOMES.index('delivered')
OWNQ_COL = F.QUERY_FEATURES.index('own_queue_live')


def _avg_ranks(col):
    """Ranks with TIES AVERAGED (standard Spearman). The v2 check breaks ties by
    row position, which gives two columns that are jointly zero on the same rows
    identical rank sequences: measured on a v5 dataset at the old operating point
    (99.3% of rows all-zero queue features) it reported rho = 0.9999 for pairs
    whose true rho is 0.833 -- a spurious redundancy FAIL."""
    order = np.argsort(col, kind='mergesort')
    vals = col[order]
    _u, first, counts = np.unique(vals, return_index=True, return_counts=True)
    r = np.empty(len(col), dtype=np.float64)
    r[order] = np.repeat(first + (counts - 1) / 2.0, counts)
    return r


def redundancy_report(block_name, x, names):
    """Pairs inside one block above PEARSON_MAX (linear) or SPEARMAN_MAX
    (monotone, average ranks). Same thresholds as G3.5 v2."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        return [], []
    pear = _corr_matrix(x)
    spear = _corr_matrix(np.stack([_avg_ranks(x[:, j]) for j in range(x.shape[1])], 1))
    off, lines = [], []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            p, r = abs(pear[i, j]), abs(spear[i, j])
            if p > PEARSON_MAX or r > SPEARMAN_MAX:
                off.append((block_name, names[i], names[j], round(p, 4), round(r, 4)))
                lines.append(f'    ** {block_name}: {names[i]} <-> {names[j]}  |r|={p:.4f} |rho|={r:.4f}')
    return off, lines


def load_shard(data, rel):
    """Eager load. np.load(.npz) is LAZY and re-decompresses a member on every
    access -- the 42,000x audit slowdown found in M3.5."""
    return dict(np.load(os.path.join(data, rel)))


def offsets(k):
    return np.concatenate([[0], np.cumsum(k.astype(np.int64))])


# ─────────────────────────────────────────────────────────────────────────────
# per-shard checks: each returns a list of problem strings (empty = pass)
# ─────────────────────────────────────────────────────────────────────────────
def check_structure(d, meta):
    p = []
    nfr = len(d['f_n_edges'])
    N = int(d['f_n_nodes'][0]) if nfr else 0
    if nfr and not (d['f_n_nodes'] == N).all():
        p.append('f_n_nodes not constant')
    if d['f_node_feat'].shape != (nfr * N, len(F.NODE_FEATURES)):
        p.append(f"f_node_feat shape {d['f_node_feat'].shape} != ({nfr * N}, {len(F.NODE_FEATURES)})")
    ne = int(d['f_n_edges'].sum())
    if d['f_edge_index'].shape != (2, ne) or d['f_edge_feat'].shape != (ne, len(F.EDGE_FEATURES)):
        p.append('edge arrays inconsistent with f_n_edges')
    if ne and (d['f_edge_index'].min() < 0 or d['f_edge_index'].max() >= N
               or (d['f_edge_index'][0] == d['f_edge_index'][1]).any()):
        p.append('edge index out of range or self-loop')
    if d['f_cum'].shape != (nfr + 1, 3 + len(G.DROP_REASONS)):
        p.append('f_cum shape')
    for k in ('f_t', 'f_mean_occ', 'f_max_occ', 'f_mean_activity', 'f_inflight', 'f_mean_lq'):
        if len(d[k]) != nfr:
            p.append(f'{k} length {len(d[k])} != {nfr} frames')
    n = len(d['c_mult'])
    for k in ('c_frame', 'c_current', 'c_dst', 'c_hops', 'c_k', 'c_label', 'c_beh',
              'c_ownq_first'):
        if len(d[k]) != n:
            p.append(f'{k} length != contexts')
    if d['c_query'].shape != (n, len(F.QUERY_FEATURES)):
        p.append('c_query shape')
    if d['c_votes'].shape != (n, len(G.T.VOTE_TEACHERS)):
        p.append('c_votes shape')
    kk = int(d['c_k'].sum())
    if not (len(d['c_cands']) == kk == len(d['c_scores']) == len(d['c_cand_feat'])):
        p.append('candidate arrays inconsistent with sum(c_k)')
    if d['c_cand_feat'].shape[1:] != (len(F.CANDIDATE_FEATURES),):
        p.append('c_cand_feat width')
    if n and (d['c_k'] < 1).any():
        p.append('a context with zero candidates')
    if n and ((d['c_frame'] < 0).any() or (d['c_frame'] >= nfr).any()
              or (np.diff(d['c_frame']) < 0).any()):
        p.append('c_frame out of range or not in frame order')
    for k in ('c_current', 'c_dst', 'c_cands'):
        if len(d[k]) and (d[k].min() < 0 or d[k].max() >= N):
            p.append(f'{k} out of node range')
    no = len(d['o_ctx'])
    if not (len(d['o_ownq']) == len(d['o_count']) == no):
        p.append('own-queue histogram arrays differ in length')
    elif no:
        key = d['o_ctx'].astype(np.int64) * G.OWNQ_BASE + d['o_ownq']
        if (d['o_ctx'].min() < 0 or d['o_ctx'].max() >= n or d['o_ownq'].min() < 0
                or d['o_ownq'].max() > G.MAX_QUEUE or (d['o_count'] < 1).any()):
            p.append('own-queue histogram out of range')
        if (np.diff(key) <= 0).any():
            p.append('own-queue histogram not strictly sorted by (context, length)')
    ns = len(d['s_ctx'])
    for k in ('s_pid', 's_hop', 's_action', 's_eps', 's_ownq', 's_t', 's_outcome', 's_attempts'):
        if len(d[k]) != ns:
            p.append(f'{k} length != steps')
    if ns and (d['s_ctx'].min() < 0 or d['s_ctx'].max() >= n):
        p.append('s_ctx out of range')
    if ns and (d['s_outcome'].min() < 0 or d['s_outcome'].max() >= len(G.HOP_OUTCOMES)):
        p.append('s_outcome out of codebook')
    npk = len(d['p_pid'])
    for k in ('p_flow', 'p_src', 'p_dst', 'p_gen_t', 'p_fate', 'p_deliv_t', 'p_hops', 'p_delay_ms'):
        if len(d[k]) != npk:
            p.append(f'{k} length != packets')
    for k, v in d.items():
        if v.dtype.kind == 'f' and k != 'p_deliv_t' and not np.isfinite(v).all():
            p.append(f'{k} has non-finite values')
    dl = d['p_fate'] == G.FATES.index('delivered')
    if npk and (not np.isfinite(d['p_deliv_t'][dl]).all() or np.isfinite(d['p_deliv_t'][~dl]).any()):
        p.append('p_deliv_t finite iff delivered violated')
    cn = meta['counts']
    for k, v in (('n_frames', nfr), ('n_contexts', n), ('n_steps', ns), ('n_packets_sampled', npk),
                 ('n_ownq_pairs', no)):
        if cn[k] != v:
            p.append(f'meta {k}={cn[k]} != arrays {v}')
    return p


def check_labels(d, meta=None):
    p = []
    n = len(d['c_mult'])
    if not n:
        return p
    k = d['c_k'].astype(np.int64)
    off = offsets(k)
    lab, beh = d['c_label'].astype(np.int64), d['c_beh'].astype(np.int64)
    if (lab < 0).any() or (lab >= k).any():
        p.append('label outside its candidate list')
        return p
    if (beh < 0).any() or (beh >= k).any():
        p.append('behaviour choice outside its candidate list')
    smax = np.maximum.reduceat(d['c_scores'], off[:-1])
    at = d['c_scores'][off[:-1] + lab]
    bad = int((at < smax - 1e-6).sum())
    if bad:
        p.append(f'{bad} labels do not carry their context maximum score')
    v = d['c_votes'].astype(np.int64)
    if (v < -1).any() or (v >= k[:, None]).any():
        p.append('vote outside [-1, k)')
    return p


def check_behaviour(d, meta, man):
    p = []
    want = man['behaviour']['table'].get(str(meta['seed']))
    if meta['behaviour'] != want:
        p.append(f"episode behaviour {meta['behaviour']} != manifest table {want}")
    if meta['behaviour'] == 'da_gpsr' and len(d['c_beh']) and (d['c_beh'] != d['c_label']).any():
        p.append('da_gpsr episode whose behaviour choice differs from its label')
    a = d['s_action'].astype(np.int64)
    if (a == G.ACTION_DROP).any():
        p.append(f'reserved drop action -1 appears {int((a == G.ACTION_DROP).sum())}x '
                 f'but no policy generates it (decision 2026-10-02)')
    if len(a):
        k = d['c_k'][d['s_ctx']].astype(np.int64)
        if ((a < 0) & (a != G.ACTION_DROP)).any() or (a >= k).any():
            p.append('action outside its candidate list')
        ne = ~d['s_eps']
        mism = int((a[ne] != d['c_beh'][d['s_ctx'][ne]]).sum())
        if mism:
            p.append(f'{mism} non-epsilon steps whose action != the behaviour choice')
    cn = meta['counts']
    rate = cn['n_eps_fired'] / max(cn['n_recorded_decisions'], 1)
    if abs(rate - meta['epsilon']) > EPS_TOL_EPISODE:
        p.append(f"epsilon {rate:.4f} vs configured {meta['epsilon']} (tol {EPS_TOL_EPISODE})")
    return p


def check_dedup(d, meta):
    p = []
    n = len(d['c_mult'])
    if not n:
        return p
    off = offsets(d['c_k'])
    keys = set()
    cands = d['c_cands']
    for i in range(n):
        keys.add((int(d['c_frame'][i]), int(d['c_current'][i]), int(d['c_dst'][i]),
                  int(d['c_hops'][i]), cands[off[i]:off[i + 1]].tobytes()))
    if len(keys) != n:
        p.append(f'{n - len(keys)} duplicate context keys within a frame')
    if (d['c_mult'] < 1).any():
        p.append('context multiplicity < 1')
    if int(d['c_mult'].sum()) != meta['counts']['n_recorded_decisions']:
        p.append(f"multiplicities sum to {int(d['c_mult'].sum())} != "
                 f"{meta['counts']['n_recorded_decisions']} recorded decisions")
    # own-queue histograms (the one feature that differs between a context's copies)
    oc = d['o_ctx'].astype(np.int64)
    if len(oc) and (oc.min() < 0 or oc.max() >= n):
        p.append('own-queue histogram context id out of range')
        return p
    tot = np.bincount(oc, weights=d['o_count'].astype(np.float64), minlength=n)
    bad = int((np.rint(tot).astype(np.int64) != d['c_mult'].astype(np.int64)).sum())
    if bad:
        p.append(f'{bad} contexts whose own-queue histogram does not sum to the multiplicity')
    pair = oc * G.OWNQ_BASE + d['o_ownq'].astype(np.int64)
    first = np.arange(n, dtype=np.int64) * G.OWNQ_BASE + d['c_ownq_first'].astype(np.int64)
    miss = int((~np.isin(first, pair)).sum())
    if miss:
        p.append(f'{miss} contexts whose first own-queue value is not in their histogram')
    col = F.QUERY_FEATURES.index('own_queue_live')
    want = (d['c_ownq_first'].astype(np.float64) / G.MAX_QUEUE).astype(np.float32)
    if not np.array_equal(d['c_query'][:, col], want):
        p.append('c_query own_queue_live != c_ownq_first / MAX_QUEUE')
    if len(d['s_ctx']):
        sp = d['s_ctx'].astype(np.int64) * G.OWNQ_BASE + d['s_ownq'].astype(np.int64)
        miss = int((~np.isin(sp, pair)).sum())
        if miss:
            p.append(f"{miss} steps whose own-queue length is not in their context's histogram")
    return p


def check_chains(d, meta):
    p = []
    ns = len(d['s_ctx'])
    pk = {int(x): i for i, x in enumerate(d['p_pid'])}
    if ns == 0:
        return p
    off = offsets(d['c_k'])
    order = np.lexsort((d['s_hop'], d['s_pid']))
    pid, hop = d['s_pid'][order], d['s_hop'][order].astype(np.int64)
    ctx, act, oc = d['s_ctx'][order], d['s_action'][order].astype(np.int64), d['s_outcome'][order]
    cur, dst = d['c_current'][ctx].astype(np.int64), d['c_dst'][ctx].astype(np.int64)
    chosen = d['c_cands'][off[ctx] + np.clip(act, 0, None)].astype(np.int64)
    starts = np.r_[0, np.where(np.diff(pid) != 0)[0] + 1]
    ends = np.r_[starts[1:], ns]
    n_bad = {'missing_packet': 0, 'not_from_source': 0, 'hop_gap': 0, 'nonfinal_not_moved': 0,
             'link_broken': 0, 'dst_changes': 0, 'fate_mismatch': 0, 'hop_count': 0}
    for s, e in zip(starts, ends):
        q = pk.get(int(pid[s]))
        if q is None:
            n_bad['missing_packet'] += 1
            continue
        if hop[s] != 0 or cur[s] != int(d['p_src'][q]):
            n_bad['not_from_source'] += 1
        if e - s > 1:
            if (np.diff(hop[s:e]) != 1).any():
                n_bad['hop_gap'] += 1
            if (oc[s:e - 1] != MOVED).any():
                n_bad['nonfinal_not_moved'] += 1
            if (cur[s + 1:e] != chosen[s:e - 1]).any():
                n_bad['link_broken'] += 1
        if (dst[s:e] != int(d['p_dst'][q])).any():
            n_bad['dst_changes'] += 1
        fate = G.FATES[int(d['p_fate'][q])]
        last = G.HOP_OUTCOMES[int(oc[e - 1])]
        if fate == 'delivered':
            ok = last == 'delivered'
        elif fate in ('link_error', 'queue_overflow', 'energy_depleted'):
            ok = last == fate          # a hop-time drop is the final step's outcome
        else:                          # no_route / ttl_expired / episode_end
            ok = last == 'moved'
        if not ok:
            n_bad['fate_mismatch'] += 1
        if int(d['p_hops'][q]) != int(((oc[s:e] == MOVED) | (oc[s:e] == DELIVERED)).sum()):
            n_bad['hop_count'] += 1
    p += [f'{v} packets: {k}' for k, v in n_bad.items() if v]
    return p


def check_sampling(d, meta):
    p = []
    seed, frac = meta['seed'], meta['rl_packet_frac']
    n_gen = int(meta['metrics']['n_generated'])
    expect = set(np.where(G.h64_np(seed, np.arange(n_gen), G.SALT_PACKET) < frac)[0].tolist())
    got = set(int(x) for x in d['p_pid'])
    if got != expect:
        p.append(f'sampled packets differ from the hash rule: {len(got - expect)} extra, '
                 f'{len(expect - got)} missing')
    if not set(int(x) for x in d['s_pid']) <= got:
        p.append('steps for packets that are not in the packet table')
    return p


def check_counters(d, meta):
    p = []
    cum = d['f_cum']
    if (np.diff(cum, axis=0) < 0).any():
        p.append('cumulative frame counters decrease')
    m = meta['metrics']
    last = cum[-1]
    want = [m['n_generated'], m['n_delivered'], m['n_dropped']] + \
           [int(m['drop_reasons'].get(r, 0)) for r in G.DROP_REASONS]
    if [int(x) for x in last] != [int(x) for x in want]:
        p.append(f'final frame counters {last.tolist()} != episode metrics {want}')
    return p


SHARD_CHECKS = (('1 structure', check_structure), ('2 labels', check_labels),
                ('9 dedup', check_dedup), ('10 chains', check_chains),
                ('11 sampling', check_sampling), ('12 counters', check_counters))


# ─────────────────────────────────────────────────────────────────────────────
def negative_controls(d, meta, man):
    """Each check must FAIL on a corrupted copy of a real shard."""
    rng = np.random.default_rng(0)
    out = {}

    def cp():
        return {k: v.copy() for k, v in d.items()}
    # labels: move one label off its maximum score
    x = cp()
    multi = np.where(x['c_k'] > 1)[0]
    if len(multi):
        i = int(multi[0])
        off = offsets(x['c_k'])
        s = x['c_scores'][off[i]:off[i + 1]]
        x['c_label'][i] = int(np.argmin(s))
        if s.min() < s.max():
            out['2 labels'] = bool(check_labels(x, meta))
    # behaviour: flip non-epsilon actions
    x = cp()
    ne = np.where(~x['s_eps'] & (x['c_k'][x['s_ctx']] > 1))[0]
    if len(ne):
        j = ne[:max(1, len(ne) // 100)]
        x['s_action'][j] = (x['s_action'][j] + 1) % x['c_k'][x['s_ctx'][j]]
        out['4 behaviour'] = bool(check_behaviour(x, meta, man))
    # behaviour: a reserved drop action
    x = cp()
    if len(x['s_action']):
        x['s_action'][0] = G.ACTION_DROP
        out['4 behaviour (drop)'] = bool(check_behaviour(x, meta, man))
    # dedup: duplicate the first context's key
    x = cp()
    if len(x['c_mult']) > 1:
        x['c_mult'][0] += 1
        out['9 dedup'] = bool(check_dedup(x, meta))
    # dedup: own-queue histogram that no longer sums to the multiplicity
    x = cp()
    if len(x['o_count']):
        x['o_count'][len(x['o_count']) // 2] += 1
        out['9 dedup (own-queue histogram)'] = bool(check_dedup(x, meta))
    # dedup: a step whose own queue its context never saw
    x = cp()
    if len(x['s_ctx']):
        c0 = int(x['s_ctx'][0])
        seen = set(x['o_ownq'][x['o_ctx'] == c0].tolist())
        x['s_ownq'][0] = next(v for v in range(G.MAX_QUEUE + 1) if v not in seen)
        out['9 dedup (step own queue)'] = bool(check_dedup(x, meta))
    # chains: point a mid-chain step at a different context
    x = cp()
    if len(x['s_ctx']) > 2:
        order = np.lexsort((x['s_hop'], x['s_pid']))
        pids = x['s_pid'][order]
        same = np.where(pids[1:] == pids[:-1])[0]
        if len(same):
            a = order[same[0] + 1]
            x['s_ctx'][a] = (x['s_ctx'][a] + 1) % len(x['c_mult'])
            out['10 chains'] = bool(check_chains(x, meta))
    # sampling: lose one packet row
    x = cp()
    if len(x['p_pid']):
        keep = np.ones(len(x['p_pid']), bool)
        keep[int(rng.integers(len(keep)))] = False
        for k in [k for k in x if k.startswith('p_')]:
            x[k] = x[k][keep]
        out['11 sampling'] = bool(check_sampling(x, meta))
    # counters
    x = cp()
    x['f_cum'][-1, 0] += 1
    out['12 counters'] = bool(check_counters(x, meta))
    # structure
    x = cp()
    x['c_k'][0] += 1
    out['1 structure'] = bool(check_structure(x, meta))
    return out


def check_schema(man):
    p = []
    if man.get('record_schema_version') != G.RECORD_SCHEMA_VERSION:
        p.append(f"record_schema_version {man.get('record_schema_version')} != {G.RECORD_SCHEMA_VERSION}")
    p += F.assert_manifest_compatible(man, context='G3.5v3')
    return p


def repro(data, meta):
    dur = meta['config']['duration'] if meta['config']['duration'] != C.OPERATING_POINT['duration'] else None
    t0 = time.time()
    _arrs, m2 = G.run_episode(meta['scenario'], meta['rate'], meta['seed'], meta['behaviour'],
                              meta['epsilon'], meta['rl_packet_frac'], duration=dur)
    diff = sorted(k for k in meta['sha256_16'] if meta['sha256_16'][k] != m2['sha256_16'].get(k))
    return diff, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join('data', 'v3'))
    ap.add_argument('--skip_repro', action='store_true')
    ap.add_argument('--repro_key', default=None, help="episode key, default: the smallest")
    ap.add_argument('--max_shards', type=int, default=None, help='debug: check only N shards')
    ap.add_argument('--report_dir', default=os.path.join('results', 'dataset_v3'),
                    help='tracked copy of the verdict and manifest (data/ is git-ignored)')
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.data, 'manifest.json')))
    print('=' * 78)
    print(f"  GATE G3.5 v3 -- {args.data}   episodes={man['totals']['episodes']}"
          + ('   [SMOKE dataset]' if man.get('smoke') else ''))
    print('=' * 78)
    results, details = {}, {}

    sch = check_schema(man)
    results['0 schema'] = not sch
    details['0 schema'] = sch
    if sch:
        print('  ABORTED: schema mismatch\n    ' + '\n    '.join(sch))
        return 1

    eps = man['episodes'][:args.max_shards] if args.max_shards else man['episodes']
    probs = {name: [] for name, _ in SHARD_CHECKS}
    probs['4 behaviour'] = []
    samples = {'node': [], 'edge': [], 'query': [], 'cand': []}
    trivial = {}
    ownq_diag = {}
    norm = {}
    eps_n = eps_f = 0
    rng = np.random.default_rng(1)
    nc_ok = True
    t0 = time.time()
    for i, e in enumerate(eps):
        meta = json.load(open(os.path.join(args.data, e['shard'].replace('.npz', '.json'))))
        d = load_shard(args.data, e['shard'])
        if i == 0:
            nc = negative_controls(d, meta, man)
            dead = [k for k, v in nc.items() if not v]
            print(f"  negative controls: {len(nc) - len(dead)}/{len(nc)} corruptions detected"
                  + (f'   DEAD CHECKS: {dead}' if dead else ''))
            if dead:
                print('  ABORTED: a check that cannot fail proves nothing.')
                return 1
        for name, fn in SHARD_CHECKS:
            for msg in fn(d, meta):
                probs[name].append(f"{e['key']}: {msg}")
        for msg in check_behaviour(d, meta, man):
            probs['4 behaviour'].append(f"{e['key']}: {msg}")
        if norm.setdefault(meta['scenario'], meta['norm_constants']) != meta['norm_constants']:
            nc_ok = False
        eps_n += meta['counts']['n_recorded_decisions']
        eps_f += meta['counts']['n_eps_fired']
        cell = f"{meta['scenario']}@{meta['rate']:g}"
        t = trivial.setdefault(cell, [0.0, 0.0, 0.0])
        w = d['c_mult'].astype(np.float64)
        t[0] += float((w * (d['c_label'] == 0)).sum()); t[1] += float(w.sum())
        t[2] += float((d['c_label'] != 0).sum()) / max(len(d['c_label']), 1)
        oq = ownq_diag.setdefault(cell, [0.0, 0.0, 0.0])
        nd = meta['counts']['n_recorded_decisions']
        oq[0] += nd
        oq[1] += meta['own_queue']['share_decisions_differing_from_first'] * nd
        oq[2] += meta['own_queue']['mean_first_minus_decision'] * nd
        n = len(d['c_mult'])
        if n:
            idx = rng.choice(n, size=min(n, SAMPLE_PER_SHARD), replace=False)
            q = d['c_query'][idx].copy()
            q[:, OWNQ_COL] = (G.ownq_sample(d, meta['key'], idx).astype(np.float64)
                              / G.MAX_QUEUE).astype(np.float32)
            samples['query'].append(q)
            off = offsets(d['c_k'])
            ci = np.concatenate([np.arange(off[j], off[j + 1]) for j in idx[:500]])
            samples['cand'].append(d['c_cand_feat'][ci])
        nn_ = len(d['f_node_feat'])
        samples['node'].append(d['f_node_feat'][rng.choice(nn_, size=min(nn_, SAMPLE_PER_SHARD), replace=False)])
        ne_ = len(d['f_edge_feat'])
        if ne_:
            samples['edge'].append(d['f_edge_feat'][rng.choice(ne_, size=min(ne_, SAMPLE_PER_SHARD), replace=False)])
        if (i + 1) % 25 == 0 or i + 1 == len(eps):
            print(f'    checked {i + 1}/{len(eps)} shards  ({time.time() - t0:.0f}s)')

    for name in probs:
        results[name] = not probs[name]
        details[name] = probs[name][:20]
    if not nc_ok:
        results['0 schema'] = False
        details['0 schema'].append('norm constants differ between episodes of a scenario')
    pooled = eps_f / max(eps_n, 1)
    if abs(pooled - G.EPSILON) > EPS_TOL_POOLED:
        results['4 behaviour'] = False
        details['4 behaviour'].append(f'pooled epsilon {pooled:.4f} vs {G.EPSILON}')

    # 5 coverage
    have = {(e['scenario'], float(e['rate']), int(e['seed'])) for e in man['episodes']}
    if man.get('smoke'):
        want = {(e['scenario'], float(e['rate']), int(e['seed'])) for e in man['episodes']}
    else:
        want = {(s, r, sd) for s, r in C.dataset_cells() for sd in C.DATASET_SEEDS}
    miss = sorted(want - have)
    results['5 coverage'] = not miss
    details['5 coverage'] = [f'missing {len(miss)} episodes, e.g. {miss[:5]}'] if miss else []
    mix = {}
    for e in man['episodes']:
        mix.setdefault(e['split'], {}).setdefault(e['behaviour'], 0)
        mix[e['split']][e['behaviour']] += 1

    # 6 features
    blocks = {'node': (F.NODE_FEATURES, samples['node']), 'edge': (F.EDGE_FEATURES, samples['edge']),
              'query': (F.QUERY_FEATURES, samples['query']),
              'cand': (F.CANDIDATE_FEATURES, samples['cand'])}
    lo_ok = {'progress': -1.0, 'vx': -1.0, 'vy': -1.0, 'vz': -1.0}
    fprob, red, sat = [], [], []
    for b, (names, arrs) in blocks.items():
        if not arrs:
            continue
        x = np.concatenate(arrs)
        if not np.isfinite(x).all():
            fprob.append(f'{b}: non-finite')
        for j, nm in enumerate(names):
            lo, hi = float(x[:, j].min()), float(x[:, j].max())
            if lo < lo_ok.get(nm, 0.0) - 1e-6 or hi > 1.0 + 1e-6:
                fprob.append(f'{b}.{nm} outside its range: [{lo:.3f}, {hi:.3f}]')
            if x[:, j].std() < 1e-9:
                fprob.append(f'{b}.{nm} is DEAD (zero variance)')
        o, lines = redundancy_report(b, x, names)
        red += o
        sat += saturation_report(b, x, names)
    results['6 features'] = not fprob
    details['6 features'] = fprob
    results['8 redundancy'] = not red
    details['8 redundancy'] = [str(r) for r in red]

    # 7 reproducibility
    if args.skip_repro:
        results['7 reproducible'] = None
        details['7 reproducible'] = ['SKIPPED (--skip_repro)']
    else:
        key = args.repro_key or min(man['episodes'], key=lambda e: e['n_decisions'])['key']
        e = next(x for x in man['episodes'] if x['key'] == key)
        meta = json.load(open(os.path.join(args.data, e['shard'].replace('.npz', '.json'))))
        print(f'  regenerating {key} for the reproducibility check ...')
        diff, secs = repro(args.data, meta)
        results['7 reproducible'] = not diff
        details['7 reproducible'] = ([f'{key}: arrays differ: {diff}'] if diff
                                     else [f'{key}: all {len(meta["sha256_16"])} arrays identical ({secs:.0f}s)'])

    # ---- report
    print('\n  DIAGNOSTICS')
    for cell, (s0, w, hard) in sorted(trivial.items()):
        n_ep = sum(1 for e in man['episodes'] if f"{e['scenario']}@{e['rate']:g}" == cell)
        o = ownq_diag.get(cell, [1.0, 0.0, 0.0])
        print(f'    {cell:<18} slot-0 (gpsr) share of decisions {s0 / max(w, 1):.3f}   '
              f'hard contexts {hard / max(n_ep, 1):.3f}   own queue != first occurrence in '
              f'{o[1] / max(o[0], 1):.3f} of decisions (first is {o[2] / max(o[0], 1):+.1f} pkts)')
    print(f'    behaviour mix per split: {mix}')
    print(f'    pooled epsilon {pooled:.4f}')
    if sat:
        print('    saturation:')
        for line in sat:
            print('  ' + line)
    print('\n  VERDICT')
    names = ['0 schema', '1 structure', '2 labels', '4 behaviour', '5 coverage', '6 features',
             '7 reproducible', '8 redundancy', '9 dedup', '10 chains', '11 sampling', '12 counters']
    for nm in names:
        r = results.get(nm)
        tag = 'SKIP' if r is None else ('PASS' if r else 'FAIL')
        first = details.get(nm) or []
        print(f'    [{tag}] {nm:<16} {first[0] if first else ""}')
        for extra in first[1:4]:
            print(f'           {"":<16} {extra}')
    passed = all(v for v in results.values() if v is not None)
    print('\n  G3.5 v3 ' + ('PASS' if passed else 'FAIL')
          + ('' if results.get('7 reproducible') is not None else '  (reproducibility not run)'))
    rep = {'results': results, 'details': details, 'mix': mix, 'pooled_epsilon': pooled,
           'own_queue': {c: {'share_differing_from_first': o[1] / max(o[0], 1),
                             'mean_first_minus_decision': o[2] / max(o[0], 1)}
                         for c, o in sorted(ownq_diag.items())}}
    with open(os.path.join(args.data, 'preflight_v3.json'), 'w') as f:
        json.dump(rep, f, indent=1)
    # a tracked copy: data/ is git-ignored, the gate evidence should not be
    tag = os.path.basename(os.path.normpath(args.data))
    os.makedirs(args.report_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(args.report_dir, f'{tag}_preflight_v3.json'), 'w'), indent=1)
    json.dump(man, open(os.path.join(args.report_dir, f'{tag}_manifest.json'), 'w'), indent=1)
    print(f'  written: {args.data}/preflight_v3.json and {args.report_dir}/{tag}_preflight_v3.json (+ manifest copy)')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
