"""
train_supervised_v2.py — M4 Step 2. Supervised warmstart + the check-2 comparison.

    python src/train_supervised_v2.py --data data/phaseB --out results/m4 --seeds 50

DECISIONS FIXED IN CODE BEFORE THE FIRST RUN (plan §1.1). Deciding any of
these after seeing results is what §3.6 exists to prevent.

  M-14   SEARCH_SPACE below is shared verbatim by both models. There is one
         constant, not two, so the budgets cannot drift apart by an edit to
         one call site.
  M-16   Default depth L=2, not 3, matching the pre-registered "shallow wins".
  n=50   Paired across TRAINING seeds. See the caveat printed with the result:
         this samples initialisation and shuffle order, NOT the data split, so
         the claim it supports is conditional on this split.
  split  By (scenario, seed) from the manifest. NEVER random rows -- every
         decision in an episode shares topology, flows and seed, so a random
         row split leaks the same episode into train and test.

WHAT GETS RECORDED WHETHER OR NOT IT IS HEADLINED
-------------------------------------------------
Effect size, its 95% CI, and Cohen's d are computed and written alongside the
p-value. This costs nothing -- same numbers, same loop -- and it is not
recoverable later without re-running 100 trainings.

It matters here specifically: training-seed variance is small because the data
split is fixed, so at n=50 a 0.2pp difference will land at p<0.001. A p-value
alone would certify a difference nobody cares about. What to put in the paper
is a decision for when the numbers exist; what to MEASURE is a decision that
has to be made now.

Accuracy is reported three ways for the same reason:
  raw          comparable to the 0.7114 trivial floor
  contested    excludes the 28.8% of decisions where the label IS the
               destination; those are free wins that inflate every model
               equally and mask the GNN-MLP difference. Trivial floor 0.5945.
  per bucket   the load split is 59.4/31.6/9.0, so a pooled number can look
               flat while the high-load number moves.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

import features_v2 as F
from model_gnn_attn import FANETRouter, densify, set_determinism

# ─────────────────────────────────────────────────────────────────────────────
# M-14: ONE budget, shared by both models. Do not add a second.
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_SPACE = dict(
    d=128,
    layers=2,               # M-16
    heads=4,
    dropout=0.1,
    # GNN-only. Separate from `dropout` so the two mixers apply feature dropout
    # in structurally identical places (fix 3). Defaults to 0.0 so attention
    # dropout is something you SWEEP, never something the GNN inherits by
    # accident from a knob meant for feature maps.
    attn_dropout=0.0,
    lr=3e-4,
    weight_decay=1e-4,
    frames_per_batch=48,    # a batch is a set of FRAMES; the encoder runs once
                            # per frame and scores all of its decisions
    max_epochs=50,
    patience=6,             # early stop on val contested accuracy
    grad_clip=1.0,
)
MODELS = ('attention', 'mlp')       # GNN, and the matched-capacity control
# 'attention_edgekey' is available via --mixers. It is a GNN-vs-GNN control
# ("was the GNN underpowered by weak edge-feature use?"), not part of the
# matched-capacity comparison -- it carries ~0.6% more parameters.


# ─────────────────────────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE MASKING — the hop-feature ablation
# ─────────────────────────────────────────────────────────────────────────────
# WHY THIS EXISTS.
# hop_distance_to_dst, cand_hop_distance and cand_reachable are all computed by
# global BFS from the destination (features_v2.hop_distances_to). Propagating
# topological distance is precisely what message passing is FOR -- and the
# unmasked experiment hands all three to BOTH models as pre-computed features.
#
# So the unmasked comparison answers "given explicit BFS topology features, does
# message passing add anything?" -- NOT "do graph networks help FANET routing?".
# The locality experiment already predicted the first answer: global hop
# knowledge is worth +0.0065 PDR, and geographic distance correlates 0.89-0.93
# with hop distance.
#
# Masking these three columns removes the GNN's job from the feature set and
# asks whether the encoder can recover it. The 2x2 is the experiment; a single
# cell is not.
#
# ZERO, DON'T DROP. Both models keep identical architecture and parameter
# counts, so matched capacity is preserved exactly and only the INFORMATION
# changes. Dropping columns would alter input dims and break the 0.1% parameter
# parity that makes check 2 interpretable.
MASK_PRESETS = {
    # TOPOLOGY ONLY. Removes the pre-computed BFS distance features.
    'hop': ['hop_distance_to_dst', 'cand_hop_distance', 'cand_reachable'],

    # EVERYTHING A 2-LAYER GNN COULD OTHERWISE COMPUTE FOR ITSELF.
    # 'hop' alone is NOT sufficient, and using it alone would have produced a
    # misleading answer. Two things a GNN would derive by message passing are
    # ALSO handed over pre-computed:
    #   neigh_buffered_packets  } k-hop (LOCAL_HORIZON=2) aggregates of
    #   neigh_mean_occupancy    } neighbour queue occupancy -- i.e. exactly the
    #                             congestion lookahead a 2-layer GNN produces
    # Masking 'hop' but not these leaves the GNN's second job pre-empted, so a
    # null result would be uninterpretable.
    'gnnjob': ['hop_distance_to_dst', 'cand_hop_distance', 'cand_reachable',
               'neigh_buffered_packets', 'neigh_mean_occupancy'],

    'none': [],
}
# HARD CEILING, STATE IT IN THE PAPER.
# Geometry can never be masked: dist_to_dest, progress and the node x/y/z
# coordinates must remain or neither model knows where the destination is.
# Part A measured geographic-vs-hop distance correlation at 0.89-0.93 in this
# scenario generator (uniform-random placement, open 2D, no obstacles), so
# geometry remains a strong topology proxy no matter what is masked.
# CONSEQUENCE: no masking experiment in THIS simulator can support a general
# claim that graph networks do not help FANET routing. The defensible claim is
# bounded to open-2D uniform-random scenarios where geometry proxies topology
# -- the same scope limit experiment_spbp_mechanism already prints for Part A.

_BLOCKS = ('node', 'edge', 'query', 'cand')


def resolve_mask(names):
    """name -> (block, column index). Fails loudly on an unknown name.

    A typo here would silently mask nothing and produce a 'no difference'
    result that looks like a finding. Validate once, at construction.
    """
    if not names:
        return {}
    out = {b: [] for b in _BLOCKS}
    lists = {'node': F.NODE_FEATURES, 'edge': F.EDGE_FEATURES,
             'query': F.QUERY_FEATURES, 'cand': F.CANDIDATE_FEATURES}
    for nm in names:
        hit = [(b, l.index(nm)) for b, l in lists.items() if nm in l]
        if not hit:
            raise SystemExit(
                f"unknown feature {nm!r}. Known names:\n"
                + "\n".join(f"  {b}: {l}" for b, l in lists.items()))
        if len(hit) > 1:
            raise SystemExit(f"{nm!r} appears in several blocks: {hit}")
        out[hit[0][0]].append(hit[0][1])
    return {b: sorted(v) for b, v in out.items() if v}


class PhaseB:
    def __init__(self, data_dir, mask=None):
        self.frm = {k: v for k, v in np.load(
            os.path.join(data_dir, 'frames.npz')).items()}
        self.dec = {k: v for k, v in np.load(
            os.path.join(data_dir, 'decisions.npz')).items()}
        with open(os.path.join(data_dir, 'manifest.json')) as f:
            self.man = json.load(f)

        self.mask = resolve_mask(mask or [])
        self.mask_names = list(mask or [])

        skew = F.assert_manifest_compatible(self.man, context='M4 training')
        if skew:
            raise SystemExit(
                "SCHEMA SKEW — refusing to train.\n  "
                + "\n  ".join(skew)
                + "\n  Feature names would resolve against a column layout the "
                  "dataset does not have: no exception, plausible loss curves, "
                  "wrong model.")

        d = self.dec
        self.n = len(d['label'])
        cf, co = d['cand_flat'], d['cand_offsets']
        lab_node = cf[co[:-1] + d['label']]
        self.contested = lab_node != d['dst']          # label is not the dst
        self.k_sizes = (co[1:] - co[:-1]).astype(np.int64)
        self.trivial_raw = float((d['label'] == 0).mean())
        self.trivial_contested = float((d['label'][self.contested] == 0).mean())

        # split by (scenario, seed) — never by row
        plan = self.man['split_plan']
        heldout = plan['generalisation_scenario']
        sc, sd = d['scenario'], d['seed'].astype(int)
        tr_lo, tr_hi = plan['train_seeds']
        va_lo, va_hi = plan['val_seeds']
        te_lo, te_hi = plan['test_seeds']
        gen = sc == heldout
        self.idx = {
            'train': np.where(~gen & (sd >= tr_lo) & (sd <= tr_hi))[0],
            'val': np.where(~gen & (sd >= va_lo) & (sd <= va_hi))[0],
            'test': np.where(~gen & (sd >= te_lo) & (sd <= te_hi))[0],
            'generalisation': np.where(gen)[0],
        }
        assert sum(len(v) for v in self.idx.values()) == self.n, \
            "splits do not partition the dataset"
        # frames are disjoint across splits by construction (episode = seed);
        # assert rather than assume, since leakage here would be invisible
        fsets = {k: set(d['frame_id'][v].tolist()) for k, v in self.idx.items()}
        for a in fsets:
            for b in fsets:
                if a < b:
                    assert not (fsets[a] & fsets[b]), f"frame leak {a}/{b}"

        # decisions grouped by frame, so a batch of frames covers whole frames
        self.by_frame = {}
        for split, ids in self.idx.items():
            fid = d['frame_id'][ids]
            order = np.argsort(fid, kind='stable')
            ids, fid = ids[order], fid[order]
            bounds = np.searchsorted(fid, np.unique(fid), side='left')
            bounds = np.append(bounds, len(fid))
            self.by_frame[split] = [
                (int(fid[bounds[i]]), ids[bounds[i]:bounds[i + 1]])
                for i in range(len(bounds) - 1)]

    def frame_tensors(self, fid, device):
        f = self.frm
        a, b = f['node_offsets'][fid], f['node_offsets'][fid + 1]
        nf = torch.from_numpy(f['node_feat_flat'][a:b]).to(device)
        e0, e1 = f['edge_offsets'][fid], f['edge_offsets'][fid + 1]
        ei = torch.from_numpy(f['edge_index_flat'][:, e0:e1].astype(np.int64))
        ef = torch.from_numpy(f['edge_feat_flat'][e0:e1]).to(device)
        adj, dense = densify(ei.to(device), ef, nf.shape[0], device=device)
        return nf, adj, dense

    def batch(self, split, frame_slice, device):
        """One batch = several whole frames + all of their decisions."""
        d, f = self.dec, self.frm

        # VECTORISED FRAME ASSEMBLY.
        # The first version called frame_tensors() once per frame: at 48 frames
        # per batch and ~440 batches that is ~21,000 densify() calls per epoch,
        # each with its own small host->device copies. On a GPU those transfers
        # dominate. Everything below is gathered in numpy across the WHOLE
        # batch and transferred in three large copies.
        fids = np.array([fid for fid, _ in frame_slice], dtype=np.int64)
        dec_ids = [ids for _, ids in frame_slice]
        fmap = np.repeat(np.arange(len(fids), dtype=np.int64),
                         [len(x) for x in dec_ids])
        B = len(fids)

        n0, n1 = f['node_offsets'][fids], f['node_offsets'][fids + 1]
        ns = (n1 - n0).astype(np.int64)
        N = int(ns.max())
        Fn = f['node_feat_flat'].shape[1]
        Fe = f['edge_feat_flat'].shape[1]

        nrow = np.repeat(np.arange(B, dtype=np.int64), ns)
        ncol = np.arange(ns.sum(), dtype=np.int64) - np.repeat(
            np.cumsum(ns) - ns, ns)
        nsrc = np.repeat(n0.astype(np.int64), ns) + ncol
        nf_np = np.zeros((B, N, Fn), dtype=np.float32)
        nm_np = np.zeros((B, N), dtype=bool)
        nf_np[nrow, ncol] = f['node_feat_flat'][nsrc]
        nm_np[nrow, ncol] = True

        e0, e1 = f['edge_offsets'][fids], f['edge_offsets'][fids + 1]
        es = (e1 - e0).astype(np.int64)
        erow = np.repeat(np.arange(B, dtype=np.int64), es)
        ecol = np.arange(es.sum(), dtype=np.int64) - np.repeat(
            np.cumsum(es) - es, es)
        esrc = np.repeat(e0.astype(np.int64), es) + ecol
        ii = f['edge_index_flat'][0, esrc].astype(np.int64)
        jj = f['edge_index_flat'][1, esrc].astype(np.int64)
        efv = f['edge_feat_flat'][esrc]
        adj_np = np.zeros((B, N, N), dtype=bool)
        de_np = np.zeros((B, N, N, Fe), dtype=np.float32)
        # SYMMETRISE: extract_frame stores each edge once with i<j. Both
        # directions are written here; encode() asserts the result is symmetric.
        adj_np[erow, ii, jj] = True
        adj_np[erow, jj, ii] = True
        de_np[erow, ii, jj] = efv
        de_np[erow, jj, ii] = efv

        node_feat = torch.from_numpy(nf_np).to(device)
        node_mask = torch.from_numpy(nm_np).to(device)
        adj = torch.from_numpy(adj_np).to(device)
        dense = torch.from_numpy(de_np).to(device)

        ids = np.concatenate(dec_ids)
        fmap = torch.from_numpy(fmap).to(device)
        co = d['cand_offsets']
        ks = (co[ids + 1] - co[ids]).astype(np.int64)
        K = int(ks.max())
        n_dec = len(ids)

        # VECTORISED RAGGED ASSEMBLY.
        # The first version looped in Python over every decision, doing four
        # small host->device transfers each. At ~1000 decisions per batch and
        # ~440 batches that is ~440k iterations per epoch, and it left the GPU
        # idle for most of the epoch: 92 s/epoch on an A4000 for a 336k-param
        # model is not a compute limit, it is a data-loading limit.
        # Everything below is built once in numpy and transferred in a handful
        # of large copies instead.
        rows = np.repeat(np.arange(n_dec, dtype=np.int64), ks)
        starts = np.repeat(np.cumsum(ks) - ks, ks)
        cols = np.arange(len(rows), dtype=np.int64) - starts
        src = np.repeat(co[ids].astype(np.int64), ks) + cols

        ci_np = np.zeros((n_dec, K), dtype=np.int64)
        cm_np = np.zeros((n_dec, K), dtype=bool)
        cfeat_np = np.zeros((n_dec, K, d['cand_feat_flat'].shape[1]),
                            dtype=np.float32)
        ci_np[rows, cols] = d['cand_flat'][src]
        cm_np[rows, cols] = True
        cfeat_np[rows, cols] = d['cand_feat_flat'][src]

        cand_idx = torch.from_numpy(ci_np).to(device)
        cand_mask = torch.from_numpy(cm_np).to(device)
        cand_feat = torch.from_numpy(cfeat_np).to(device)

        # candidate edge features: one advanced-index gather on device, rather
        # than n_dec separate small gathers
        cur_np = d['current'][ids].astype(np.int64)
        cur_t = torch.from_numpy(cur_np).to(device)
        fr = fmap.unsqueeze(1).expand(n_dec, K)
        cu = cur_t.unsqueeze(1).expand(n_dec, K)
        cand_edge = dense[fr, cu, cand_idx]
        cand_edge = cand_edge * cand_mask.unsqueeze(-1)
        query_feat_t = torch.from_numpy(d['query_feat'][ids]).to(device)
        # Applied AFTER everything is assembled, including cand_edge, which is
        # gathered out of `dense` -- masking dense alone would leave the
        # candidate edge copy unmasked and the ablation would be half-applied.
        for col in self.mask.get('node', []):
            node_feat[..., col] = 0.0
        for col in self.mask.get('edge', []):
            dense[..., col] = 0.0
            cand_edge[..., col] = 0.0
        for col in self.mask.get('query', []):
            query_feat_t[..., col] = 0.0
        for col in self.mask.get('cand', []):
            cand_feat[..., col] = 0.0

        return dict(
            node_feat=node_feat, adj=adj, dense=dense, node_mask=node_mask,
            fmap=fmap,
            cur=cur_t,
            dst=torch.from_numpy(d['dst'][ids].astype(np.int64)).to(device),
            query=query_feat_t,
            cand_idx=cand_idx, cand_feat=cand_feat, cand_edge=cand_edge,
            cand_mask=cand_mask,
            label=torch.from_numpy(d['label'][ids].astype(np.int64)).to(device),
            ids=ids)


def run_batch(model, bt):
    """Encode each frame ONCE, then score every decision that belongs to it."""
    h = model.encode(bt['node_feat'], bt['adj'], bt['dense'], bt['node_mask'],
                     check=False)
    hh = h[bt['fmap']]
    nf = bt['node_feat'][bt['fmap']]
    return model.score(hh, nf, bt['cur'], bt['dst'], bt['query'],
                       bt['cand_idx'], bt['cand_feat'], bt['cand_edge'],
                       bt['cand_mask'])


# ─────────────────────────────────────────────────────────────────────────────
# evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, ds, split, device, frames_per_batch):
    model.eval()
    fs = ds.by_frame[split]
    correct, ids_all = [], []
    for i in range(0, len(fs), frames_per_batch):
        bt = ds.batch(split, fs[i:i + frames_per_batch], device)
        pred = run_batch(model, bt).argmax(-1)
        correct.append((pred == bt['label']).cpu().numpy())
        ids_all.append(bt['ids'])
    correct = np.concatenate(correct)
    ids = np.concatenate(ids_all)
    out = {'accuracy_raw': float(correct.mean())}
    con = ds.contested[ids]
    out['accuracy_contested'] = float(correct[con].mean()) if con.any() else float('nan')
    out['above_trivial_raw'] = out['accuracy_raw'] - ds.trivial_raw
    out['above_trivial_contested'] = out['accuracy_contested'] - ds.trivial_contested
    for b in np.unique(ds.dec['load_bucket'][ids]):
        m = ds.dec['load_bucket'][ids] == b
        out[f'accuracy_{b}'] = float(correct[m].mean())
        out[f'accuracy_contested_{b}'] = (
            float(correct[m & con].mean()) if (m & con).any() else float('nan'))
    out['n'] = int(len(correct))
    return out


def train_one(ds, mixer, seed, device, dst_encoding='encoded', hp=None,
              verbose=False):
    hp = dict(SEARCH_SPACE if hp is None else hp)
    set_determinism()          # G4 check 6: see model_gnn_attn.set_determinism
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FANETRouter(d=hp['d'], layers=hp['layers'], heads=hp['heads'],
                        dropout=hp['dropout'], mixer=mixer,
                        attn_dropout=hp.get('attn_dropout', 0.0),
                        dst_encoding=dst_encoding).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp['lr'],
                           weight_decay=hp['weight_decay'])
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    fs = list(ds.by_frame['train'])
    best, best_state, bad, epochs_run = -1.0, None, 0, 0
    for epoch in range(hp['max_epochs']):
        epochs_run = epoch + 1
        model.train()
        rng.shuffle(fs)
        for i in range(0, len(fs), hp['frames_per_batch']):
            bt = ds.batch('train', fs[i:i + hp['frames_per_batch']], device)
            logits = run_batch(model, bt)
            loss = lossf(logits, bt['label'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp['grad_clip'])
            opt.step()
        va = evaluate(model, ds, 'val', device, hp['frames_per_batch'])
        score = va['accuracy_contested']
        if verbose:
            print(f"      epoch {epoch:02d}  val_contested={score:.4f}")
        if score > best + 1e-5:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp['patience']:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.stats.assert_cached()
    return model, best, epochs_run


# ─────────────────────────────────────────────────────────────────────────────
# statistics
# ─────────────────────────────────────────────────────────────────────────────
def paired_stats(a, b):
    """Paired comparison with EFFECT SIZE, not just a p-value.

    At n=50 on a fixed data split, training-seed variance is small enough that
    a 0.2pp difference reaches p<0.001. The p-value alone would certify a
    difference of no practical size, so mean difference, its 95% CI and
    Cohen's d are always reported next to it.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    dif = a - b
    n = len(dif)
    mean = float(dif.mean())
    sd = float(dif.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else float('inf') if mean != 0 else 0.0
    try:
        from scipy import stats as _st
        p = float(2 * _st.t.sf(abs(t), df=n - 1)) if n > 1 else float('nan')
        crit = float(_st.t.ppf(0.975, df=n - 1)) if n > 1 else float('nan')
    except ImportError:
        p, crit = float('nan'), 1.96
    return dict(n=n, mean_diff=mean, sd_diff=sd, t=float(t), p=p,
                ci95=[mean - crit * se, mean + crit * se],
                cohens_d=(mean / sd if sd > 0 else float('nan')))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/phaseB')
    ap.add_argument('--out', default='results/m4')
    ap.add_argument('--seeds', type=int, default=50)
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('--dst_encoding', default='encoded',
                    choices=['encoded', 'raw'])
    ap.add_argument('--layers', type=int, default=None,
                    help='override depth for the sweep; default is M-16 L=2')
    # HYPERPARAMETER KNOBS (fix 2).
    # SEARCH_SPACE was one fixed config for both models: symmetric, but
    # symmetric at ZERO. "Equal budget" with a budget of one is not a fair
    # comparison, and attention models routinely want a different learning rate
    # from plain MLPs. Every knob below applies to BOTH models identically --
    # M-14 is preserved because there is still exactly one config per run, and
    # the tag records it so two settings can never be merged into one file.
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--d', type=int, default=None)
    ap.add_argument('--heads', type=int, default=None)
    ap.add_argument('--dropout', type=float, default=None)
    ap.add_argument('--attn_dropout', type=float, default=None,
                    help='GNN-only; the MLP control ignores it by construction')
    ap.add_argument('--weight_decay', type=float, default=None)
    ap.add_argument('--mixers', nargs='+', default=list(MODELS),
                    choices=['attention', 'attention_edgekey', 'mlp'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--mask', nargs='+', default=None,
                    help="feature names to ZERO, or a preset: 'hop' expands to "
                         "hop_distance_to_dst cand_hop_distance cand_reachable. "
                         "The hop-feature 2x2 ablation is --mask none vs "
                         "--mask hop, everything else held fixed.")
    ap.add_argument('--max_epochs', type=int, default=None,
                    help='override for smoke tests ONLY; the real budget is '
                         'SEARCH_SPACE and overriding it breaks M-14 parity '
                         'unless BOTH models use the same override')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    mask_names = []
    for m in (args.mask or []):
        mask_names += MASK_PRESETS.get(m, [m])

    hp = dict(SEARCH_SPACE)
    for _k in ('layers', 'lr', 'd', 'heads', 'dropout', 'attn_dropout',
               'weight_decay'):
        _v = getattr(args, _k)
        if _v is not None:
            hp[_k] = _v
    if args.max_epochs is not None:
        hp['max_epochs'] = args.max_epochs
        # Clamp patience only when the cap is LOWERED. Raising the cap must not
        # silently change the early-stopping rule, or grid configs would run
        # under a different stopping criterion from the default and M-14 parity
        # would be broken across configs rather than across models.
        hp['patience'] = min(hp['patience'], args.max_epochs)
        if args.max_epochs >= SEARCH_SPACE['max_epochs']:
            hp['patience'] = SEARCH_SPACE['patience']
    os.makedirs(args.out, exist_ok=True)
    tag = f"L{hp['layers']}_{args.dst_encoding}"
    # Any hyperparameter that differs from the default goes in the filename.
    # Without this, a swept run silently merges into the default run's
    # runs_*.json and the paired test compares two different configurations.
    _diff = {k: hp[k] for k in ('lr', 'd', 'heads', 'dropout', 'attn_dropout',
                                'weight_decay')
             if hp[k] != SEARCH_SPACE[k]}
    for _k in sorted(_diff):
        tag += f"_{_k}{_diff[_k]:g}"
    # The mask MUST be in the tag. Two runs that differ only in what the model
    # was allowed to see are not comparable, and silently merging them into one
    # runs_*.json would produce a meaningless paired test.
    if mask_names:
        tag += "_mask-" + "+".join(sorted(n[:12] for n in mask_names))
    if args.max_epochs is not None:
        tag += f"_ep{args.max_epochs}"      # never mix a smoke run with a real one
    resfile = os.path.join(args.out, f'runs_{tag}.json')
    # RESUMABLE. 100 trainings is long enough that an interruption must not
    # cost the whole run, and partial results must never be silently mixed
    # with a different configuration -- hence the tag in the filename.
    runs = json.load(open(resfile)) if os.path.isfile(resfile) else {}

    ds = PhaseB(args.data, mask=mask_names)
    print("=" * 78)
    print("  M4 STEP 2 — SUPERVISED WARMSTART")
    print("=" * 78)
    print(f"  schema v{ds.man['feature_schema_version']}  "
          f"local_horizon={ds.man['local_horizon']}  device={args.device}")
    print(f"  splits: " + "  ".join(f"{k}={len(v)}" for k, v in ds.idx.items()))
    if mask_names:
        print(f"  MASKED (zeroed) features: {mask_names}")
        print(f"    resolved to columns: {ds.mask}")
        _bt = ds.batch('train', ds.by_frame['train'][:4], args.device)
        for _blk, _key in (('node', 'node_feat'), ('edge', 'dense'),
                           ('query', 'query'), ('cand', 'cand_feat')):
            for _c in ds.mask.get(_blk, []):
                _mx = float(_bt[_key][..., _c].abs().max())
                assert _mx == 0.0, (
                    f"mask did NOT apply: {_blk} column {_c} has max |v|={_mx}")
        print(f"    VERIFIED zero in a real batch (all blocks, incl. cand_edge)")
    print(f"  trivial floor: raw={ds.trivial_raw:.4f}  "
          f"contested={ds.trivial_contested:.4f}")
    print(f"  shared budget (M-14): {hp}")
    print(f"  {args.seeds} seeds x {len(args.mixers)} models = "
          f"{args.seeds * len(args.mixers)} trainings, tag={tag}")
    print("=" * 78)

    t_first = None
    for s in range(args.seed0, args.seed0 + args.seeds):
        for mixer in args.mixers:
            key = f"{mixer}:{s}"
            if key in runs:
                continue
            t0 = time.time()
            model, val_best, eps_run = train_one(ds, mixer, s, args.device,
                                                 args.dst_encoding, hp,
                                                 args.verbose)
            rec = {'val_contested': val_best, 'seconds': time.time() - t0,
                   'epochs_run': eps_run}
            for split in ('test', 'generalisation'):
                for k, v in evaluate(model, ds, split, args.device,
                                     hp['frames_per_batch']).items():
                    rec[f'{split}_{k}'] = v
            runs[key] = rec
            json.dump(runs, open(resfile, 'w'), indent=2)
            print(f"  [{key:>14}] val={val_best:.4f} "
                  f"test_contested={rec['test_accuracy_contested']:.4f} "
                  f"({rec['seconds']:.0f}s)")
            if t_first is None:
                t_first = rec['seconds']
                per_epoch = t_first / max(eps_run, 1)
                n_train = args.seeds * len(args.mixers)
                print(f"  ---- COST: {per_epoch:.0f} s/epoch, "
                      f"{eps_run} epoch(s) in this training.")
                if args.max_epochs is not None and \
                        args.max_epochs < SEARCH_SPACE['max_epochs']:
                    # SMOKE MODE -- only when the cap is LOWERED.
                    # The first version fired on ANY --max_epochs, so raising
                    # the cap to 100 for the HP grid printed "this is NOT the
                    # real cost" on six perfectly real runs.
                    # Projecting from a truncated training would
                    # understate the real job by roughly the epoch ratio -- the
                    # first version of this printed "0.1 h" for a job that is
                    # actually tens of hours, because t_first covered one epoch
                    # instead of a full early-stopped training.
                    lo = per_epoch * 15 * n_train / 3600
                    hi = per_epoch * SEARCH_SPACE['max_epochs'] * n_train / 3600
                    print(f"       ** SMOKE MODE (--max_epochs {args.max_epochs}). "
                          f"This is NOT the real cost. **")
                    print(f"       A real run early-stops somewhere between ~15 "
                          f"and {SEARCH_SPACE['max_epochs']} epochs, so expect")
                    print(f"       {lo:.0f}-{hi:.0f} h for {n_train} trainings "
                          f"at {args.seeds} seeds, and 8x that for the full "
                          f"depth sweep.")
                else:
                    total = t_first * n_train
                    print(f"       PROJECTED TOTAL: {total/3600:.1f} h for this "
                          f"configuration ({n_train} trainings), assuming later "
                          f"seeds early-stop near {eps_run} epochs.")
                    print(f"       Full depth sweep (8 configurations): "
                          f"~{8*total/3600:.0f} h.")
                print(f"       Resumable: rerun the same command to continue.")

    # Compare the FIRST TWO mixers requested, not hardcoded attention/mlp --
    # otherwise `--mixers attention_edgekey mlp` KeyErrors after a 3 h run.
    if len(args.mixers) < 2:
        print(f"\n  Only one mixer requested ({args.mixers}); no paired "
              f"comparison. Results are in {resfile}.")
        return
    m_a, m_b = args.mixers[0], args.mixers[1]
    seeds_ok = [s for s in range(args.seed0, args.seed0 + args.seeds)
                if f'{m_a}:{s}' in runs and f'{m_b}:{s}' in runs]
    a = [runs[f'{m_a}:{s}']['test_accuracy_contested'] for s in seeds_ok]
    b = [runs[f'{m_b}:{s}']['test_accuracy_contested'] for s in seeds_ok]
    st = paired_stats(a, b)
    summary = {'tag': tag, 'hp': hp, 'n_seeds': len(seeds_ok),
               'mixer_a': m_a, 'mixer_b': m_b,
               'trivial_raw': ds.trivial_raw,
               'trivial_contested': ds.trivial_contested,
               'mean_a': float(np.mean(a)), 'mean_b': float(np.mean(b)),
               'paired': st}
    json.dump(summary, open(os.path.join(args.out, f'summary_{tag}.json'), 'w'),
              indent=2)

    print("\n" + "=" * 78)
    print(f"  CHECK 2 — {m_a} vs {m_b} (test, contested accuracy)")
    print("=" * 78)
    print(f"    {m_a:<18}{np.mean(a):.4f}      {m_b:<18}{np.mean(b):.4f}")
    print(f"    (difference is {m_a} MINUS {m_b})")
    print(f"    mean difference {st['mean_diff']:+.4f} "
          f"(95% CI {st['ci95'][0]:+.4f} .. {st['ci95'][1]:+.4f})")
    print(f"    Cohen's d {st['cohens_d']:+.2f}   t={st['t']:.2f}  "
          f"p={st['p']:.2e}   n={st['n']} paired seeds")
    print()
    print("    READ THE EFFECT SIZE FIRST. With the data split fixed, only")
    print("    initialisation and shuffle order vary across these seeds, so the")
    print("    p-value shrinks with n almost regardless of how small the")
    print("    difference is. The CI is the number that says whether it matters.")
    print()
    print("    SCOPE: this is conditional on ONE train/val/test split. It does")
    print("    not sample split variance, so the claim is 'on this split',")
    print("    not 'in general'. Say so in the paper.")
    print("=" * 78 + "\n")


if __name__ == '__main__':
    main()
