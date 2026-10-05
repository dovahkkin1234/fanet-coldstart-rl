"""
export_phaseb_v3.py  --  Dataset V3 -> the PhaseB format train_supervised_v2 reads.

    python src\\export_phaseb_v3.py --data data\\v3 --out data\\phaseB_v3
    python src\\export_phaseb_v3.py --data data\\v3 --out data\\phaseB_v3_10 --ctx_frac 0.10

The v3 shards hold every distinct decision context once, with its multiplicity and
the histogram of the live own-queue values its occurrences saw. This writes a
training directory (frames.npz / decisions.npz / manifest.json) for
train_supervised_v2.PhaseB, selecting:
  * scenarios / splits      (default: everything; PhaseB splits by seed range and
                             holds medium_slow out by scenario, from split_plan)
  * --ctx_frac              a NESTED fraction of contexts: context i of an episode
                            is kept iff h64(episode, i) < frac, so the 10% export is
                            a subset of the 30% export (learning curves for free)
  * --ownq                  how the one feature that differs between a context's
                            occurrences (own_queue_live, schema v6) is written:
                            'sample' (default): one row per context, own queue drawn
                              from the context's histogram with probability
                              count / multiplicity (deterministic hash) -- the
                              weighted rows have the raw stream's own-queue
                              distribution in expectation;
                            'expand': one row per distinct (context, own queue) pair
                              -- reproduces the raw decision stream EXACTLY, at
                              3-7x the rows. The trainer batches by frames (48 per
                              batch), so rows per batch grow the same way: lower
                              frames_per_batch if the GPU runs out of memory.
                            The first-occurrence value stored in c_query is never
                            used: it is biased high (the first decision of a frame
                            meets the fullest queue).
  * --weighting             'multiplicity' (default): weight = how many raw decisions
                            the row stands for -> the loss equals training on the
                            raw decision stream, the regime the teacher-choice test
                            validated (99.4-99.5% held-out accuracy, PDR parity);
                            'uniform': every distinct context counts once in total
                            (emphasises rare recovery states) -- an ablation.
Train with PhaseB(dir, mask=..., use_weights=manifest['use_weights'])
(rollout_gate_v3.py does this from the manifest).

MEMORY. Two passes: the first sizes every output array exactly (reading only the
small index arrays of each shard), the second fills arrays preallocated at their
final size -- so the export's peak RAM is the export itself plus one shard, not
twice the export. The trainer (PhaseB) then holds about the same amount (measured
~450 bytes per row at ~11 candidates, plus the frames). --max_gb (default 22, ~70%
of a 32 GB machine) refuses anything bigger unless --force.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_v2 as F                                     # noqa: E402
import generate_dataset_v3 as G                             # noqa: E402

episode_int = G.episode_int
SIZING_KEYS = ('c_mult', 'c_k', 'c_frame', 'c_ownq_first', 'o_ctx', 'o_ownq', 'o_count',
               'f_n_edges', 'f_n_nodes')


def select_contexts(key, n, frac):
    if frac >= 1.0:
        return np.arange(n)
    return np.where(G.h64_np(episode_int(key), np.arange(n), G.SALT_CONTEXT) < frac)[0]


def ranges(starts, lens):
    """Concatenation of arange(starts[i], starts[i] + lens[i]) without a Python loop."""
    lens = np.asarray(lens, np.int64)
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    before = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return np.repeat(np.asarray(starts, np.int64) - before, lens) + np.arange(tot)


def rows_for_episode(d, key, sel, ownq, weighting):
    """-> (context index per row, own-queue length per row, weight per row)."""
    mult = d['c_mult'].astype(np.int64)
    if ownq == 'sample':
        ctx = np.asarray(sel, np.int64)
        ln = G.ownq_sample(d, key, ctx).astype(np.int64)
        w = mult[ctx].astype(np.float64) if weighting == 'multiplicity' else np.ones(len(ctx))
    else:
        keep = np.zeros(len(mult), bool)
        keep[sel] = True
        o_ctx = d['o_ctx'].astype(np.int64)
        pr = np.where(keep[o_ctx])[0]
        ctx, ln = o_ctx[pr], d['o_ownq'][pr].astype(np.int64)
        cnt = d['o_count'][pr].astype(np.float64)
        w = cnt if weighting == 'multiplicity' else cnt / mult[ctx]
    return ctx, ln, w


def plan_episode(d, e, args):
    """Everything the two passes must agree on, from the small index arrays only."""
    n = len(d['c_mult'])
    sel = select_contexts(e['key'], n, args.ctx_frac)
    if len(sel) == 0:
        return None
    ctx, ln, w = rows_for_episode(d, e['key'], sel, args.ownq, args.weighting)
    fr_used = np.unique(d['c_frame'][ctx])
    return {'sel': sel, 'ctx': ctx, 'ln': ln, 'w': w, 'fr_used': fr_used,
            'N': int(d['f_n_nodes'][0]), 'k': d['c_k'][ctx].astype(np.int64),
            'ne': d['f_n_edges'][fr_used].astype(np.int64)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join('data', 'v3'))
    ap.add_argument('--out', default=os.path.join('data', 'phaseB_v3'))
    ap.add_argument('--scenarios', nargs='+', default=None)
    ap.add_argument('--splits', nargs='+', default=None,
                    help="subset of train val test generalisation (default all)")
    ap.add_argument('--ctx_frac', type=float, default=1.0)
    ap.add_argument('--ownq', choices=('sample', 'expand'), default='sample')
    ap.add_argument('--weighting', choices=('multiplicity', 'uniform'), default='multiplicity')
    ap.add_argument('--max_gb', type=float, default=22.0,
                    help='refuse a bigger export (default 22 = ~70%% of 32 GB; the trainer '
                         'holds about the same)')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--allow_stale_code', action='store_true',
                    help='export a dataset written by other code (recorded in the manifest)')
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.data, 'manifest.json')))
    if man['feature_schema_version'] != F.FEATURE_SCHEMA_VERSION:
        raise SystemExit(f"dataset schema v{man['feature_schema_version']} != module "
                         f"v{F.FEATURE_SCHEMA_VERSION}")
    if man.get('record_schema_version') != G.RECORD_SCHEMA_VERSION:
        raise SystemExit(f"record schema v{man.get('record_schema_version')} != generator "
                         f"v{G.RECORD_SCHEMA_VERSION} (regenerate: shards without own-queue "
                         f"histograms cannot be exported)")
    if man.get('code_signature') != G.CODE_SIGNATURE:
        msg = (f"dataset written by other code (signature {man.get('code_signature') or 'none, pre-v28'} "
               f"!= current {G.CODE_SIGNATURE}): its episodes do not come from this simulator")
        if not args.allow_stale_code:
            raise SystemExit(msg + ' -- regenerate, or pass --allow_stale_code')
        print('  WARNING: ' + msg)
    col = man['query_features'].index('own_queue_live')
    gen_sc = man['split_plan']['generalisation_scenario']
    eps = []
    for e in man['episodes']:
        split = 'generalisation' if e['scenario'] == gen_sc else e['split']
        if args.scenarios and e['scenario'] not in args.scenarios:
            continue
        if args.splits and split not in args.splits:
            continue
        eps.append(e)
    if not eps:
        raise SystemExit('nothing selected')

    # ---- pass 1: exact sizes ------------------------------------------------
    t0 = time.time()
    n_rows = n_cand = n_frames = n_nodes = n_edges = 0
    widths = {'scenario': 1, 'load_bucket': 1, 'behaviour': 1}
    used = []
    for k, e in enumerate(eps):
        z = np.load(os.path.join(args.data, e['shard']))
        pl = plan_episode({key: z[key] for key in SIZING_KEYS}, e, args)
        if pl is None:
            continue
        used.append(e)
        n_rows += len(pl['ctx'])
        n_cand += int(pl['k'].sum())
        n_frames += len(pl['fr_used'])
        n_nodes += len(pl['fr_used']) * pl['N']
        n_edges += int(pl['ne'].sum())
        for c in widths:
            widths[c] = max(widths[c], len(str(e[c])))
        if (k + 1) % 100 == 0:
            print(f'    sizing {k + 1}/{len(eps)} episodes  ({time.time() - t0:.0f}s)')
    if not used:
        raise SystemExit('no context selected')
    nq, nv = len(man['query_features']), len(man['vote_teachers'])
    ncf, nnf, nef = (len(man['candidate_features']), len(man['node_features']),
                     len(man['edge_features']))
    str_b = 4 * sum(widths.values())
    by_dec = n_rows * (nq * 4 + 8 + nv * 2 + 4 * 7 + 2 + str_b) + n_cand * (4 + ncf * 4) + 8
    by_frm = n_nodes * (nnf * 4 + 4) + n_edges * (2 * 4 + nef * 4) + 2 * 8 * (n_frames + 1)
    gb = (by_dec + by_frm) / 1e9
    print(f'  exporting {len(used)} episodes, ctx_frac={args.ctx_frac}, ownq={args.ownq}, '
          f'weighting={args.weighting}: {n_rows:,} rows, {n_frames:,} frames -> '
          f'{gb:.2f} GB (decisions {by_dec / 1e9:.2f}, frames {by_frm / 1e9:.2f})  '
          f'[sized in {time.time() - t0:.0f}s]')
    if gb > args.max_gb and not args.force:
        raise SystemExit(f'  {gb:.1f} GB > --max_gb {args.max_gb}; lower --ctx_frac or select '
                         f'fewer splits/scenarios, or pass --force (the trainer needs about '
                         f'as much RAM again)')

    # ---- pass 2: fill arrays preallocated at their final size ---------------
    FR = {'node_feat_flat': np.empty((n_nodes, nnf), np.float32),
          'node_ids_flat': np.empty(n_nodes, np.int32),
          'node_offsets': np.zeros(n_frames + 1, np.int64),
          'edge_index_flat': np.empty((2, n_edges), np.int32),
          'edge_feat_flat': np.empty((n_edges, nef), np.float32),
          'edge_offsets': np.zeros(n_frames + 1, np.int64)}
    DE = {'cand_flat': np.empty(n_cand, np.int32),
          'cand_offsets': np.zeros(n_rows + 1, np.int64),
          'cand_feat_flat': np.empty((n_cand, ncf), np.float32),
          'votes': np.empty((n_rows, nv), np.int16),
          'frame_id': np.empty(n_rows, np.int32), 'current': np.empty(n_rows, np.int32),
          'dst': np.empty(n_rows, np.int32), 'query_feat': np.empty((n_rows, nq), np.float32),
          'label': np.empty(n_rows, np.int32),
          'scenario': np.empty(n_rows, f"<U{widths['scenario']}"),
          'seed': np.empty(n_rows, np.int32),
          'load_bucket': np.empty(n_rows, f"<U{widths['load_bucket']}"),
          'packet_rate': np.empty(n_rows, np.float32), 'weight': np.empty(n_rows, np.float32),
          'hard': np.empty(n_rows, bool), 'rate_index': np.empty(n_rows, np.int8),
          'behaviour': np.empty(n_rows, f"<U{widths['behaviour']}")}
    ro = co = fo = no = eo = 0
    oq_stats = np.zeros(4)      # sum w, sum w*exported len, sum stream len, sum w*first len
    n_first_rows = 0
    t1 = time.time()
    for k, e in enumerate(used):
        d = dict(np.load(os.path.join(args.data, e['shard'])))
        pl = plan_episode(d, e, args)
        ctx, ln, w, fr = pl['ctx'], pl['ln'], pl['w'], pl['fr_used']
        m, N, kk, ne = len(ctx), pl['N'], pl['k'], pl['ne']
        nfr, kt, et = len(fr), int(kk.sum()), int(ne.sum())
        # frames
        eoff = np.concatenate([[0], np.cumsum(d['f_n_edges'].astype(np.int64))])
        FR['node_feat_flat'][no:no + nfr * N] = d['f_node_feat'][ranges(fr * N, np.full(nfr, N))]
        FR['node_ids_flat'][no:no + nfr * N] = np.tile(np.arange(N, dtype=np.int32), nfr)
        FR['node_offsets'][fo + 1:fo + nfr + 1] = no + N * np.arange(1, nfr + 1)
        eidx = ranges(eoff[fr], ne)
        FR['edge_index_flat'][:, eo:eo + et] = d['f_edge_index'][:, eidx]
        FR['edge_feat_flat'][eo:eo + et] = d['f_edge_feat'][eidx]
        FR['edge_offsets'][fo + 1:fo + nfr + 1] = eo + np.cumsum(ne)
        # candidates of every row (a context's candidates repeat once per row)
        off = np.concatenate([[0], np.cumsum(d['c_k'].astype(np.int64))])
        cidx = ranges(off[ctx], kk)
        DE['cand_flat'][co:co + kt] = d['c_cands'][cidx]
        DE['cand_feat_flat'][co:co + kt] = d['c_cand_feat'][cidx]
        DE['cand_offsets'][ro + 1:ro + m + 1] = co + np.cumsum(kk)
        # query: the stored first-occurrence own queue replaced by the row's value
        q = d['c_query'][ctx].copy()
        q[:, col] = (ln.astype(np.float64) / G.MAX_QUEUE).astype(np.float32)
        same = ln == d['c_ownq_first'][ctx]
        if not np.array_equal(q[same], d['c_query'][ctx[same]]):
            raise AssertionError(f"{e['key']}: rebuilt own_queue_live differs from the stored "
                                 f'value on first-occurrence rows')
        n_first_rows += int(same.sum())
        r = slice(ro, ro + m)
        DE['query_feat'][r] = q
        DE['frame_id'][r] = fo + np.searchsorted(fr, d['c_frame'][ctx])
        DE['current'][r] = d['c_current'][ctx]
        DE['dst'][r] = d['c_dst'][ctx]
        DE['label'][r] = d['c_label'][ctx]
        DE['votes'][r] = d['c_votes'][ctx]
        DE['hard'][r] = d['c_label'][ctx] != 0
        DE['weight'][r] = w
        DE['scenario'][r] = e['scenario']
        DE['seed'][r] = e['seed']
        DE['load_bucket'][r] = e['load_bucket']
        DE['packet_rate'][r] = e['rate']
        DE['rate_index'][r] = e['rate_index']
        DE['behaviour'][r] = e['behaviour']
        # own-queue fidelity diagnostic over the selected contexts (multiplicity view)
        keep = np.zeros(len(d['c_mult']), bool)
        keep[pl['sel']] = True
        po = keep[d['o_ctx']]
        wm = (d['c_mult'][ctx].astype(np.float64) if args.ownq == 'sample'
              else d['o_count'][po].astype(np.float64))
        oq_stats += [wm.sum(), (wm * ln).sum(),
                     (d['o_count'][po].astype(np.float64) * d['o_ownq'][po]).sum(),
                     (d['c_mult'][pl['sel']].astype(np.float64) * d['c_ownq_first'][pl['sel']]).sum()]
        ro, co, fo, no, eo = ro + m, co + kt, fo + nfr, no + nfr * N, eo + et
        if (k + 1) % 50 == 0 or k + 1 == len(used):
            print(f'    {k + 1}/{len(used)} episodes  ({time.time() - t1:.0f}s)')
    if (ro, co, fo, no, eo) != (n_rows, n_cand, n_frames, n_nodes, n_edges):
        raise AssertionError('pass 2 filled a different size than pass 1 planned')

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, 'frames.npz'), **FR)
    np.savez(os.path.join(args.out, 'decisions.npz'), **DE)
    use_w = not (args.weighting == 'uniform' and args.ownq == 'sample')
    sw = max(oq_stats[0], 1.0)
    ownq_report = {'exported_mean_len': oq_stats[1] / sw, 'stream_mean_len': oq_stats[2] / sw,
                   'first_occurrence_mean_len': oq_stats[3] / sw,
                   'rows_equal_to_first_occurrence': n_first_rows}
    out_man = {
        'purpose': 'Dataset V3 export for train_supervised_v2.PhaseB',
        'source': os.path.abspath(args.data), 'source_created': man.get('created'),
        'ctx_frac': args.ctx_frac, 'ownq': args.ownq, 'weighting': args.weighting,
        'use_weights': use_w,
        'label_teacher': man['label_teacher'],
        'node_features': man['node_features'], 'edge_features': man['edge_features'],
        'query_features': man['query_features'], 'candidate_features': man['candidate_features'],
        'feature_schema_version': man['feature_schema_version'],
        'record_schema_version': man['record_schema_version'],
        'code_signature': man.get('code_signature'), 'mobility': man.get('mobility'),
        'stale_code_allowed': bool(args.allow_stale_code),
        'local_horizon': man['local_horizon'],
        'split_plan': man['split_plan'],
        'norm_constants_per_scenario': man['norm_constants_per_scenario'],
        'provenance': man['provenance'],
        'episodes': [e['key'] for e in used],
        'n_rows': int(n_rows), 'n_frames': int(n_frames),
        'decisions_represented': (float(DE['weight'].astype(np.float64).sum())
                                  if args.weighting == 'multiplicity' else None),
        'own_queue_check': ownq_report,
    }
    json.dump(out_man, open(os.path.join(args.out, 'manifest.json'), 'w'), indent=1)
    sz = sum(os.path.getsize(os.path.join(args.out, f)) for f in ('frames.npz', 'decisions.npz'))
    print(f"  wrote {args.out}: {n_rows:,} rows, {n_frames:,} frames, {sz / 1e9:.2f} GB "
          f"(planned {gb:.2f})  ({time.time() - t0:.0f}s)")
    print(f"  own queue (packets, decision-weighted): exported {ownq_report['exported_mean_len']:.2f}  "
          f"raw stream {ownq_report['stream_mean_len']:.2f}  "
          f"(first occurrence would have been {ownq_report['first_occurrence_mean_len']:.2f})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
