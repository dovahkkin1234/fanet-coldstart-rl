"""test_teacher_choice.py -- ONE global teacher, or PER-CELL teachers?

Settles the open label-teacher decision (report §17.1) by measurement instead of
argument, BEFORE the dataset patch is written.

THE QUESTION, AS THREE TESTABLE CLAIMS
  R  REALIZABILITY. In medium_slow the measured oracle is dijkstra (+4.0 pp over
     da_gpsr in the panel). dijkstra decides from GLOBAL hop distances, which
     the decentralised student never sees (they are masked). Can a masked
     student trained on dijkstra labels actually route better there than a
     masked student trained on da_gpsr labels? If not, per-cell labelling
     cannot help even in principle.
  C  CONTAMINATION. Training one student on per-cell labels mixes teachers that
     disagree on ~1 decision in 3. Does that hurt the dense cells, compared with
     one consistent da_gpsr teacher?
  M  MECHANISM. Is dijkstra's medium_slow advantage a ROUTING effect, or does it
     come from dijkstra DROPPING packets whose destination is unreachable
     (no_route) -- early admission control a masked student cannot copy? Tested
     with a 'dijkstra_nodrop' arm that routes exactly like dijkstra but forwards
     unreachable packets to the nearest-to-destination neighbour instead.

DESIGN
  Cells (one rate each, all inside their 1000 s usable bands; crossed design --
  dijkstra wins one, da_gpsr wins two, each by ~4 pp in the panel):
      medium_slow@60 (dijkstra)   dense_slow@60 (da_gpsr)   very_dense@60 (da_gpsr)
  Operating point: altitude 100-300 m, battery 8000 (the v8b point).
      NOTE: the oracle panels ran at the config_v2.BASE altitude 50-150 m
      (they override duration and battery only), so the teacher arms here also
      re-measure the panel gaps at the altitude the dataset will actually use.
  Duration: 300 s by default (1000 s is affordable with --duration 1000).
      The teacher arms are the precondition check: if the dijkstra/da_gpsr gaps
      do not reproduce at this duration, the run says INCONCLUSIVE.

  Stage gen      behaviour teacher (+ epsilon) drives the simulator; a random
                 subsample of decisions is recorded with features and THREE
                 labels (dijkstra, da_gpsr, gpsr). Recording uses its own RNG,
                 so trajectories do not depend on the recording rate.
  Stage train    five masked GNN students (M4 tuned config, mask='hop'):
                   S_dij_ms   medium_slow, dijkstra labels   (realizability)
                   S_da_ms    medium_slow, da_gpsr labels    (realizability control)
                   GLOBAL     all 3 cells, da_gpsr labels
                   PERCELL    medium_slow dijkstra + dense cells da_gpsr
                   GLOBAL_HO  dense cells only, da_gpsr (the planned medium_slow holdout)
  Stage rollout  every student and three teacher arms (dijkstra, dijkstra_nodrop,
                 da_gpsr) in the actor slot, PAIRED rollout seeds 1..N.
  Stage analyze  paired tests, Holm-corrected, pre-registered verdict.

PRE-REGISTERED DECISION RULE (fixed before running -- do not edit after)
  P0  precondition: dijkstra - da_gpsr > 0 in medium_slow, and da_gpsr -
      dijkstra > 0 in both dense cells (paired 95% CI excludes 0). If not:
      INCONCLUSIVE at this operating point.
  Primary family, Holm-corrected, practical threshold 1 pp:
      R1  S_dij_ms - S_da_ms           (medium_slow)
      C1  PERCELL  - GLOBAL            (dense_slow)
      C2  PERCELL  - GLOBAL            (very_dense)
      G1  PERCELL  - GLOBAL            (medium_slow)
  Verdict:
      PER-CELL JUSTIFIED  if R1 >= +1 pp significant AND G1 >= +1 pp significant
                          AND neither C1 nor C2 is <= -1 pp significant.
      SINGLE da_gpsr      if R1 is not significantly >= +1 pp (dijkstra is not
                          realizable enough to matter), OR per-cell contaminates
                          a dense cell by >= 1 pp.
      MIXED               otherwise -- reported with the numbers, no forced call.
  Secondary (reported, own Holm family): M1 dijkstra_nodrop - dijkstra (ms),
      M2 dijkstra_nodrop - da_gpsr (ms), imitation losses, GLOBAL_HO - GLOBAL.

USAGE (PowerShell)
    python src\\test_teacher_choice.py --smoke                     # ~10 min plumbing check
    python src\\test_teacher_choice.py --stage all --max_workers 8   # the real run
  Stages can be run one at a time (--stage gen|train|rollout|analyze); every
  stage resumes from what is already on disk in --out.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features_v2 as F
import routing_teachers_v2 as rt
from routing_teachers import dijkstra_next_hop
from simulator_v2 import FANETSimulatorV2, TTL
from config_v2 import BASE, SCENARIOS
from generate_dataset_v2 import canonical_candidates
import inspect
from mobility import MOBILITY_VERSION                     # v28

# v26 COMPATIBILITY. Everything in this test is feature schema v5 (frame-start
# snapshot own queue): its data, its models and its rollouts. After the v26 patch
# the live module is v6, so v5 semantics are requested explicitly; on a pre-v26
# checkout these helpers fall back to the module defaults (which are v5).
def _nc_v5(cfg):
    if 'schema_version' in inspect.signature(F.norm_constants).parameters:
        return F.norm_constants(cfg, schema_version=5)
    return F.norm_constants(cfg)


def _phaseb_v5(path, mask):
    from train_supervised_v2 import PhaseB
    if 'accept_legacy' in inspect.signature(PhaseB.__init__).parameters:
        return PhaseB(path, mask=mask, accept_legacy=True)
    return PhaseB(path, mask=mask)


V5_QUERY_FEATURES = getattr(F, 'LEGACY_FEATURE_LISTS', {}).get(5, {}).get(
    'query_features', F.QUERY_FEATURES)

# ─────────────────────────────────────────────────────────────────────────────
# experiment definition
# ─────────────────────────────────────────────────────────────────────────────
CELLS = {'medium_slow': 60.0, 'dense_slow': 60.0, 'very_dense': 60.0}
PANEL_REF = {   # panel means, 1000 s, battery 8000, altitude 50-150 m (BASE)
    'medium_slow': {'dijkstra': 0.4057, 'da_gpsr': 0.3655},
    'dense_slow':  {'dijkstra': 0.5947, 'da_gpsr': 0.6337},
    'very_dense':  {'dijkstra': 0.5140, 'da_gpsr': 0.5614},
}
GEN_COMBOS = [('medium_slow', 'dijkstra'), ('medium_slow', 'da_gpsr'),
              ('dense_slow', 'da_gpsr'), ('very_dense', 'da_gpsr')]
STUDENTS = {
    'S_dij_ms':  [('medium_slow', 'dijkstra')],
    'S_da_ms':   [('medium_slow', 'da_gpsr')],
    'GLOBAL':    [('medium_slow', 'da_gpsr'), ('dense_slow', 'da_gpsr'),
                  ('very_dense', 'da_gpsr')],
    'PERCELL':   [('medium_slow', 'dijkstra'), ('dense_slow', 'da_gpsr'),
                  ('very_dense', 'da_gpsr')],
    'GLOBAL_HO': [('dense_slow', 'da_gpsr'), ('very_dense', 'da_gpsr')],
}
TEACHER_ARMS = ['dijkstra', 'dijkstra_nodrop', 'da_gpsr']
STUDENT_CELLS = {'S_dij_ms': ['medium_slow'], 'S_da_ms': ['medium_slow'],
                 'GLOBAL': list(CELLS), 'PERCELL': list(CELLS),
                 'GLOBAL_HO': list(CELLS)}
LABEL_KEYS = ('dijkstra', 'da_gpsr', 'gpsr')
MIN_PP = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# restricted pickers: distances on the FULL graph, choice over unvisited cands
# (the semantics spbp_pick_restricted established; pruning the graph instead
#  produced a 22.6% fallback rate in M3.5)
# ─────────────────────────────────────────────────────────────────────────────
def dijkstra_pick(G, c, dst, cands, h_map):
    """dijkstra restricted to legal candidates. Returns None iff dst is
    unreachable from c (the real teacher then drops the packet: no_route)."""
    if dst in cands:
        return dst
    if c not in h_map:
        return None
    try:
        path = nx.shortest_path(G, c, dst)
        ref = path[1] if len(path) >= 2 else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        ref = None
    if ref is not None and ref in cands:
        return ref
    reach = [u for u in cands if u in h_map]
    if not reach:
        return None
    return min(reach, key=lambda u: (h_map[u], cands.index(u)))


def da_gpsr_pick(G, c, dst, cands):
    """da_gpsr's exact score over legal candidates, iterated in G.neighbors
    order so ties break exactly as in rt.da_gpsr_next_hop."""
    cset = set(cands)
    dpos = rt._pos(G, dst)
    dist_cd = float(np.linalg.norm(dpos - rt._pos(G, c)))
    best, best_s = None, -float('inf')
    for n in G.neighbors(c):
        if n not in cset:
            continue
        dist_nd = float(np.linalg.norm(dpos - rt._pos(G, n)))
        prog = (dist_cd - dist_nd) / max(dist_cd, 1.0)
        occ = float(G.nodes[n].get('queue_occupancy', 0.0))
        lq = float(G.edges[c, n].get('link_quality', 0.0))
        s = (rt.DAGPSR_W_PROGRESS * prog - rt.DAGPSR_W_QUEUE * occ
             + rt.DAGPSR_W_QUALITY * lq)
        if s > best_s:
            best_s, best = s, n
    return best


def _random_graph(rng):
    n = int(rng.integers(5, 13))
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, x=float(rng.integers(0, 900)), y=float(rng.integers(0, 900)),
                   z=float(rng.integers(100, 300)), energy=90.0,
                   queue_occupancy=float(rng.random()),
                   queue_len=float(rng.integers(0, 50)))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.4:
                G.add_edge(i, j, distance=float(rng.integers(50, 300)),
                           link_quality=float(rng.random()))
    return G


def drift_pins(dij=dijkstra_pick, da=da_gpsr_pick, trials=300):
    """With nothing visited, each restricted picker must equal its real
    teacher. Returns the number of disagreements (0 = pinned)."""
    rng = np.random.default_rng(7)
    bad = 0
    for _ in range(trials):
        G = _random_graph(rng)
        src, dst = 0, G.number_of_nodes() - 1
        nb = list(G.neighbors(src))
        if not nb:
            continue
        cands = canonical_candidates(G, src, dst, {src})
        h = F.hop_distances_to(G, dst)
        if nx.has_path(G, src, dst):
            if dij(G, src, dst, cands, h) != dijkstra_next_hop(G, src, dst):
                bad += 1
        if da(G, src, dst, cands) != rt.da_gpsr_next_hop(G, src, dst):
            bad += 1
    return bad


def assert_pins():
    """Pins must hold AND must be shown able to fail (project rule: a checker
    that only ever passes proves nothing)."""
    if drift_pins():
        raise AssertionError('restricted picker drifted from its real teacher')

    def broken_dij(G, c, dst, cands, h):            # picks the FARTHEST hop
        r = [u for u in cands if u in h]
        return max(r, key=lambda u: h[u]) if r else None

    def broken_da(G, c, dst, cands):                # ignores the queue term
        return cands[0]
    if drift_pins(dij=broken_dij) == 0 or drift_pins(da=broken_da) == 0:
        raise AssertionError('drift pin cannot detect a broken picker -- dead check')


# ─────────────────────────────────────────────────────────────────────────────
# simulator actor: a teacher driving the network, optionally recording
# ─────────────────────────────────────────────────────────────────────────────
def episode_config(scen, seed, args, **extra):
    return {**BASE, **SCENARIOS[scen], 'duration': float(args.duration),
            'z_min': args.z_min, 'z_max': args.z_max,
            'initial_energy': args.initial_energy,
            'packet_rate': CELLS[scen], 'seed': int(seed), **extra}


class TeacherSim(FANETSimulatorV2):
    """Overrides only _select_next_hop. behaviour in
    {'dijkstra','dijkstra_nodrop','da_gpsr'}."""

    def __init__(self, config, behaviour, epsilon=0.0, record_prob=0.0):
        super().__init__(config)
        self.behaviour = behaviour
        self.eps = float(epsilon)
        self.rec_p = float(record_prob)
        self.ds_rng = np.random.default_rng(self.seed + 900_000)   # epsilon draws
        self.rec_rng = np.random.default_rng(self.seed + 700_000)  # recording only
        self.nc = _nc_v5(config)
        self._fid = -1
        self._frames = {}
        self._used = set()
        self._hcache = {}
        self.dec = []
        self.n_unreachable = 0

    def _build_graph(self):
        G = super()._build_graph()
        self._fid += 1
        self._hcache = {}
        if self.rec_p > 0:
            ids, nf, ei, ef = F.extract_frame(G, self.nc)
            self._frames[self._fid] = (ids, nf, ei, ef)
        return G

    def _h(self, G, dst):
        if dst not in self._hcache:
            self._hcache[dst] = F.hop_distances_to(G, dst)
        return self._hcache[dst]

    def _select_next_hop(self, G, pkt, neighbors):
        c, dst = pkt.current, pkt.dst
        cands = canonical_candidates(G, c, dst, set(pkt.path))
        if not cands:
            return None
        h = self._h(G, dst)
        d_lab = dijkstra_pick(G, c, dst, cands, h)
        if self.behaviour == 'dijkstra':
            if d_lab is None:
                self.n_unreachable += 1
                return None                    # real dijkstra drops: no_route
            lab = d_lab
        elif self.behaviour == 'dijkstra_nodrop':
            lab = d_lab if d_lab is not None else cands[0]
        else:
            lab = da_gpsr_pick(G, c, dst, cands)
        act = lab
        if self.eps > 0 and self.ds_rng.random() < self.eps:
            act = cands[int(self.ds_rng.integers(len(cands)))]
        if self.rec_p > 0 and self.rec_rng.random() < self.rec_p:
            qf, cf = F.extract_decision(G, pkt, cands, self.nc, h, 0.0, 0.0,
                                        ttl_const=TTL)
            da_lab = lab if self.behaviour == 'da_gpsr' else da_gpsr_pick(G, c, dst, cands)
            self.dec.append(dict(
                frame=self._fid, current=c, dst=dst,
                cands=np.asarray(cands, np.int32), qf=qf, cf=cf,
                lab_dijkstra=(cands.index(d_lab) if d_lab is not None else -1),
                lab_da_gpsr=cands.index(da_lab),
                lab_gpsr=0,          # canonical order sorts by distance-to-dst
                action=cands.index(act)))
            self._used.add(self._fid)
        return act


# ─────────────────────────────────────────────────────────────────────────────
# stage: gen
# ─────────────────────────────────────────────────────────────────────────────
def _gen_job(job):
    scen, teacher, seed, args_d = job
    args = argparse.Namespace(**args_d)
    t0 = time.time()
    sim = TeacherSim(episode_config(scen, seed, args), teacher,
                     epsilon=args.epsilon, record_prob=args.record_prob)
    m = sim.run()
    if m.get('n_phantom_slots', 0) != 0:
        raise RuntimeError('phantom queue slots -- v12 leak present')
    used = sorted(sim._used)
    remap = {f: i for i, f in enumerate(used)}
    fr = [sim._frames[f] for f in used]
    D = sim.dec
    out = os.path.join(args.out, 'raw', f'{scen}_{teacher}_s{seed}.npz')
    kc = np.array([len(d['cands']) for d in D], np.int64)
    np.savez_compressed(
        out,
        node_feat=np.concatenate([f[1] for f in fr]) if fr else np.zeros((0, 9), np.float32),
        node_counts=np.array([f[1].shape[0] for f in fr], np.int64),
        edge_index=np.concatenate([f[2] for f in fr], axis=1) if fr else np.zeros((2, 0), np.int64),
        edge_feat=np.concatenate([f[3] for f in fr]) if fr else np.zeros((0, 4), np.float32),
        edge_counts=np.array([f[2].shape[1] for f in fr], np.int64),
        frame=np.array([remap[d['frame']] for d in D], np.int64),
        current=np.array([d['current'] for d in D], np.int32),
        dst=np.array([d['dst'] for d in D], np.int32),
        cand_flat=np.concatenate([d['cands'] for d in D]) if D else np.zeros(0, np.int32),
        cand_counts=kc,
        cand_feat=np.concatenate([d['cf'] for d in D]) if D else np.zeros((0, 4), np.float32),
        query=np.array([d['qf'] for d in D], np.float32),
        lab_dijkstra=np.array([d['lab_dijkstra'] for d in D], np.int32),
        lab_da_gpsr=np.array([d['lab_da_gpsr'] for d in D], np.int32),
        lab_gpsr=np.array([d['lab_gpsr'] for d in D], np.int32),
        action=np.array([d['action'] for d in D], np.int32))
    return dict(scen=scen, teacher=teacher, seed=seed, n_dec=len(D),
                n_frames=len(fr), pdr=m['network_pdr'],
                unreachable=sim.n_unreachable, seconds=time.time() - t0)


def stage_gen(args):
    os.makedirs(os.path.join(args.out, 'raw'), exist_ok=True)
    seeds = list(args.train_seeds) + list(args.val_seeds)
    jobs = [(s, t, sd, vars(args)) for s, t in GEN_COMBOS for sd in seeds
            if not os.path.isfile(os.path.join(args.out, 'raw', f'{s}_{t}_s{sd}.npz'))]
    print(f'\n  GEN: {len(jobs)} episodes to run '
          f'({len(GEN_COMBOS)} combos x {len(seeds)} seeds, resume-aware)')
    log = _load(args, 'gen_log.json', [])
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_gen_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            log.append(r)
            _save(args, 'gen_log.json', log)
            print(f'    {i}/{len(jobs)}  {r["scen"]:<12} {r["teacher"]:<9} '
                  f's{r["seed"]:<4} decisions={r["n_dec"]:>6}  pdr={r["pdr"]:.3f}  '
                  f'{r["seconds"]:.0f}s  (elapsed {time.time()-t0:.0f}s)')


# ─────────────────────────────────────────────────────────────────────────────
# stage: train -- assemble PhaseB-format datasets, reuse M4's trainer verbatim
# ─────────────────────────────────────────────────────────────────────────────
def _assemble(args, combos, seeds, out_dir):
    """Write frames.npz / decisions.npz / manifest.json that train_supervised_v2
    .PhaseB loads unchanged. Label column = the combo's own teacher. Rows with
    no label (dijkstra on an unreachable destination) are excluded."""
    if all(os.path.isfile(os.path.join(out_dir, n))
           for n in ('frames.npz', 'decisions.npz', 'manifest.json')):
        return
    os.makedirs(out_dir, exist_ok=True)
    NF, NC, EI, EF, EC = [], [], [], [], []
    fr_off = 0
    cols = {k: [] for k in ('frame_id', 'current', 'dst', 'query_feat', 'label',
                            'scenario', 'seed', 'load_bucket')}
    cand_flat, cand_feat, cand_counts = [], [], []
    dropped = 0
    for scen, teacher in combos:
        for sd in seeds:
            z = dict(np.load(os.path.join(args.out, 'raw', f'{scen}_{teacher}_s{sd}.npz')))  # eager: lazy npz re-decompresses per access
            nfr = len(z['node_counts'])
            NF.append(z['node_feat']); NC.append(z['node_counts'])
            EI.append(z['edge_index']); EF.append(z['edge_feat']); EC.append(z['edge_counts'])
            lab = z[f'lab_{teacher}']
            keep = lab >= 0
            dropped += int((~keep).sum())
            co = np.concatenate([[0], np.cumsum(z['cand_counts'])])
            for i in np.where(keep)[0]:
                cand_flat.append(z['cand_flat'][co[i]:co[i + 1]])
                cand_feat.append(z['cand_feat'][co[i]:co[i + 1]])
                cand_counts.append(z['cand_counts'][i])
            cols['frame_id'].append(z['frame'][keep] + fr_off)
            cols['current'].append(z['current'][keep])
            cols['dst'].append(z['dst'][keep])
            cols['query_feat'].append(z['query'][keep])
            cols['label'].append(lab[keep])
            n = int(keep.sum())
            cols['scenario'].append(np.array([scen] * n))
            cols['seed'].append(np.full(n, sd, np.int32))
            cols['load_bucket'].append(np.array(['band'] * n))
            fr_off += nfr
    node_counts = np.concatenate(NC)
    edge_counts = np.concatenate(EC)
    np.savez_compressed(
        os.path.join(out_dir, 'frames.npz'),
        node_feat_flat=np.concatenate(NF).astype(np.float32),
        node_ids_flat=np.concatenate([np.arange(k) for k in node_counts]).astype(np.int32),
        node_offsets=np.concatenate([[0], np.cumsum(node_counts)]).astype(np.int64),
        edge_index_flat=np.concatenate(EI, axis=1),
        edge_feat_flat=np.concatenate(EF).astype(np.float32),
        edge_offsets=np.concatenate([[0], np.cumsum(edge_counts)]).astype(np.int64))
    np.savez_compressed(
        os.path.join(out_dir, 'decisions.npz'),
        frame_id=np.concatenate(cols['frame_id']).astype(np.int32),
        current=np.concatenate(cols['current']).astype(np.int32),
        dst=np.concatenate(cols['dst']).astype(np.int32),
        cand_flat=np.concatenate(cand_flat).astype(np.int32),
        cand_offsets=np.concatenate([[0], np.cumsum(cand_counts)]).astype(np.int64),
        cand_feat_flat=np.concatenate(cand_feat).astype(np.float32),
        query_feat=np.concatenate(cols['query_feat']).astype(np.float32),
        label=np.concatenate(cols['label']).astype(np.int32),
        scenario=np.concatenate(cols['scenario']),
        seed=np.concatenate(cols['seed']),
        load_bucket=np.concatenate(cols['load_bucket']))
    man = {
        'purpose': 'test_teacher_choice', 'combos': combos, 'seeds': list(seeds),
        'node_features': F.NODE_FEATURES, 'edge_features': F.EDGE_FEATURES,
        'query_features': V5_QUERY_FEATURES, 'candidate_features': F.CANDIDATE_FEATURES,
        'feature_schema_version': 5,
        'local_horizon': F.LOCAL_HORIZON,
        'split_plan': {'train_seeds': [min(args.train_seeds), max(args.train_seeds)],
                       'val_seeds': [min(args.val_seeds), max(args.val_seeds)],
                       'test_seeds': [99999, 99999],
                       'generalisation_scenario': '__none__'},
        'rows_without_label_excluded': dropped,
    }
    json.dump(man, open(os.path.join(out_dir, 'manifest.json'), 'w'), indent=2)


def _hp():
    from train_supervised_v2 import SEARCH_SPACE
    hp = dict(SEARCH_SPACE)
    hp.update(lr=1e-3, attn_dropout=0.1, max_epochs=100)   # M4 tuned config
    return hp


def stage_train(args):
    import torch
    from train_supervised_v2 import train_one, evaluate, MASK_PRESETS
    mask = MASK_PRESETS['hop']
    hp = _hp()
    if args.smoke:
        hp.update(max_epochs=2, patience=1)
    res = _load(args, 'train.json', {})
    seeds_all = list(args.train_seeds) + list(args.val_seeds)
    for name, combos in STUDENTS.items():
        d = os.path.join(args.out, 'data', name)
        _assemble(args, combos, seeds_all, d)
        ds = _phaseb_v5(d, mask)
        for ms in range(args.model_seeds):
            seed = 2000 + ms
            key = f'{name}:{seed}'
            ck = os.path.join(args.out, 'models', f'{name}_{seed}.pt')
            if key in res and os.path.isfile(ck):
                continue
            os.makedirs(os.path.dirname(ck), exist_ok=True)
            t0 = time.time()
            model, val_best, eps = train_one(ds, 'attention', seed, args.device, hp=hp)
            model.eval()
            torch.save({'state_dict': model.state_dict(), 'mixer': 'attention',
                        'hp': hp, 'mask': mask, 'student': name, 'seed': seed,
                        'combos': combos}, ck)
            res[key] = {'val_contested': val_best, 'epochs': eps,
                        'n_train': int(len(ds.idx['train'])),
                        'n_val': int(len(ds.idx['val'])),
                        'seconds': time.time() - t0}
            _save(args, 'train.json', res)
            print(f'    [{key:<14}] val contested acc {val_best:.4f}  '
                  f'epochs {eps}  n_train {len(ds.idx["train"])}  '
                  f'({time.time()-t0:.0f}s)')
    # cross-evaluation: every student on every combo's validation episodes
    xe = _load(args, 'cross_eval.json', {})
    for scen, teacher in GEN_COMBOS:
        d = os.path.join(args.out, 'data', f'EVAL_{scen}_{teacher}')
        _assemble(args, [(scen, teacher)], list(args.val_seeds), d)
        ds = _phaseb_v5(d, mask)
        for name in STUDENTS:
            for ms in range(args.model_seeds):
                key = f'{name}:{2000+ms}|{scen}|{teacher}'
                if key in xe:
                    continue
                model = _load_model(os.path.join(args.out, 'models',
                                                 f'{name}_{2000+ms}.pt'), args.device)
                ev = evaluate(model, ds, 'val', args.device, hp['frames_per_batch'])
                xe[key] = {'acc': ev['accuracy_raw'],
                           'acc_contested': ev['accuracy_contested'], 'n': ev['n']}
        _save(args, 'cross_eval.json', xe)
    print('    cross-evaluation written (student x validation set)')


def _load_model(path, device):
    import torch
    from model_gnn_attn import FANETRouter
    ck = torch.load(path, map_location=device, weights_only=False)
    hp = ck['hp']
    m = FANETRouter(d=hp['d'], layers=hp['layers'], heads=hp['heads'],
                    dropout=hp['dropout'], mixer=ck['mixer'],
                    attn_dropout=hp.get('attn_dropout', 0.0)).to(device)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# stage: rollout
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_CACHE = {}


def _roll_job(job):
    arm, scen, seed, ckpt, args_d = job
    args = argparse.Namespace(**args_d)
    t0 = time.time()
    cfg = episode_config(scen, seed, args)
    if arm in ('dijkstra', 'da_gpsr'):
        m = FANETSimulatorV2({**cfg, 'actor': arm}).run()
    elif arm == 'dijkstra_nodrop':
        m = TeacherSim(cfg, 'dijkstra_nodrop').run()
    else:
        import torch
        torch.set_num_threads(1)
        from rollout_eval_v2 import ModelActorSimulator
        from train_supervised_v2 import MASK_PRESETS
        if ckpt not in _MODEL_CACHE:
            _MODEL_CACHE[ckpt] = _load_model(ckpt, 'cpu')
        kw = {'mask': MASK_PRESETS['hop']}
        if 'schema_version' in inspect.signature(ModelActorSimulator.__init__).parameters:
            kw['schema_version'] = 5
        sim = ModelActorSimulator(cfg, _MODEL_CACHE[ckpt], 'cpu', **kw)
        m = sim.run()
    if m.get('n_phantom_slots', 0) != 0:
        raise RuntimeError('phantom queue slots -- v12 leak present')
    gen = max(m['n_generated'], 1)
    return dict(arm=arm, scen=scen, seed=seed, pdr=m['network_pdr'],
                n_generated=m['n_generated'],
                drops={k: v / gen for k, v in m['drop_reasons'].items()},
                mean_hops=m.get('mean_hops'), seconds=time.time() - t0)


def stage_rollout(args):
    res = _load(args, 'rollout.json', {})
    jobs = []
    for scen in CELLS:
        for sd in args.rollout_seeds:
            for arm in TEACHER_ARMS:
                if f'{arm}|{scen}|{sd}' not in res:
                    jobs.append((arm, scen, sd, None, vars(args)))
    for name, cells in STUDENT_CELLS.items():
        for ms in range(args.model_seeds):
            arm = f'{name}:{2000+ms}'
            ck = os.path.join(args.out, 'models', f'{name}_{2000+ms}.pt')
            for scen in cells:
                for sd in args.rollout_seeds:
                    if f'{arm}|{scen}|{sd}' not in res:
                        jobs.append((arm, scen, sd, ck, vars(args)))
    # longest jobs (students, very_dense) first, so the tail is short
    jobs.sort(key=lambda j: (j[3] is None, j[1] != 'very_dense'))
    print(f'\n  ROLLOUT: {len(jobs)} episodes to run (resume-aware)')
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_roll_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            res[f'{r["arm"]}|{r["scen"]}|{r["seed"]}'] = r
            _save(args, 'rollout.json', res)
            el = time.time() - t0
            print(f'    {i}/{len(jobs)}  {r["arm"]:<16} {r["scen"]:<12} s{r["seed"]:<3} '
                  f'pdr={r["pdr"]:.4f}  {r["seconds"]:.0f}s  '
                  f'(~{el/i*(len(jobs)-i)/60:.0f} min left)')


# ─────────────────────────────────────────────────────────────────────────────
# stage: analyze
# ─────────────────────────────────────────────────────────────────────────────
def _paired(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    if n < 2:
        return dict(mean=float(d.mean()) if n else float('nan'), lo=float('nan'),
                    hi=float('nan'), p=float('nan'), n=n)
    se = d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    p = float(stats.ttest_rel(a, b).pvalue) if se > 0 else (0.0 if d.mean() else 1.0)
    return dict(mean=float(d.mean()), lo=float(d.mean() - tc * se),
                hi=float(d.mean() + tc * se), p=p, n=n, sd=float(d.std(ddof=1)))


def _holm(tests):
    keys = sorted(tests, key=lambda k: tests[k]['p'] if tests[k]['p'] == tests[k]['p'] else 1)
    m = len(keys)
    running = 0.0
    for i, k in enumerate(keys):
        p = tests[k]['p'] if tests[k]['p'] == tests[k]['p'] else 1.0
        running = max(running, min(1.0, (m - i) * p))
        tests[k]['p_holm'] = running
    return tests


def _cell_classifier(args):
    """Can the student's LOCAL observation tell medium_slow from the dense
    cells? Logistic regression on locally observable summaries, trained on
    train seeds, scored on val seeds. Uses da_gpsr-behaviour data in every
    cell so behaviour does not leak the answer."""
    import torch
    feats, ys, sds, conflict = [], [], [], []
    ql = list(F.QUERY_FEATURES)
    keep_q = [i for i, n in enumerate(ql) if n != 'hop_distance_to_dst']
    deg_col = F.NODE_FEATURES.index('degree')
    prog_col = F.CANDIDATE_FEATURES.index('progress')
    for scen in CELLS:
        for sd in list(args.train_seeds) + list(args.val_seeds):
            p = os.path.join(args.out, 'raw', f'{scen}_da_gpsr_s{sd}.npz')
            if not os.path.isfile(p):
                continue
            z = dict(np.load(p))
            noff = np.concatenate([[0], np.cumsum(z['node_counts'])])
            co = np.concatenate([[0], np.cumsum(z['cand_counts'])])
            for i in range(len(z['frame'])):
                fr = z['frame'][i]
                deg = z['node_feat'][noff[fr] + z['current'][i], deg_col]
                cf = z['cand_feat'][co[i]:co[i + 1]]
                feats.append(np.concatenate([z['query'][i][keep_q],
                                             [deg, len(cf) / 20.0,
                                              cf[:, prog_col].mean(),
                                              cf[:, prog_col].max()]]))
                ys.append(1.0 if scen == 'medium_slow' else 0.0)
                sds.append(sd)
                conflict.append(z['lab_dijkstra'][i] >= 0 and
                                z['lab_dijkstra'][i] != z['lab_da_gpsr'][i])
    if not feats:
        return None
    X = torch.tensor(np.array(feats), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    sds = np.array(sds)
    conflict = np.array(conflict, bool)
    tr = torch.tensor(np.isin(sds, args.train_seeds))
    va = ~tr
    mu, sdv = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sdv
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos = float(y[tr].mean())
    wts = torch.where(y[tr] > 0, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        z_ = Xn[tr] @ w + b
        loss = (wts * torch.nn.functional.binary_cross_entropy_with_logits(
            z_, y[tr], reduction='none')).mean() + 1e-3 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        pred = ((Xn @ w + b) > 0).float()
    def bal(mask):
        mask = torch.tensor(mask) if not torch.is_tensor(mask) else mask
        yy, pp = y[mask], pred[mask]
        if len(yy) == 0 or yy.min() == yy.max():
            return float('nan')
        return float(0.5 * ((pp[yy > 0] == 1).float().mean() + (pp[yy == 0] == 0).float().mean()))
    vmask = va.numpy()
    return {'balanced_acc_val': bal(va),
            'balanced_acc_val_on_conflict_rows': bal(vmask & conflict),
            'n_val': int(va.sum()), 'n_val_conflict': int((vmask & conflict).sum())}


def stage_analyze(args):
    R = _load(args, 'rollout.json', {})

    xe = _load(args, 'cross_eval.json', {})
    out = {'operating_point': {'duration': args.duration, 'z': [args.z_min, args.z_max],
                               'mobility': MOBILITY_VERSION,
                               'initial_energy': args.initial_energy,
                               'cells': CELLS}}
    seeds = list(args.rollout_seeds)

    def arm_vec(arm, scen):
        if ':' in arm or arm in STUDENTS:
            v = []
            for sd in seeds:
                xs = [R[f'{arm}:{2000+ms}|{scen}|{sd}']['pdr']
                      for ms in range(args.model_seeds)
                      if f'{arm}:{2000+ms}|{scen}|{sd}' in R]
                v.append(np.mean(xs) if xs else np.nan)
            return np.array(v)
        return np.array([R[f'{arm}|{scen}|{sd}']['pdr'] if f'{arm}|{scen}|{sd}' in R
                         else np.nan for sd in seeds])

    def drops(arm, scen):
        acc = {}
        rows = [r for k, r in R.items() if k.startswith(arm + '|' + scen + '|')
                or (k.startswith(arm + ':') and f'|{scen}|' in k)]
        for r in rows:
            for kk, v in r['drops'].items():
                acc[kk] = acc.get(kk, 0.0) + v / len(rows)
        return acc

    print('\n' + '=' * 96)
    print('  TEACHER-CHOICE TEST -- results')
    print('=' * 96)
    print(f'  operating point: {args.duration:.0f}s, altitude {args.z_min:.0f}-{args.z_max:.0f} m, '
          f'battery {args.initial_energy:.0f}, rollout seeds {seeds[0]}..{seeds[-1]} (paired)')
    arms = TEACHER_ARMS + list(STUDENTS)
    table = {}
    print(f'\n  {"arm":<16}' + ''.join(f'{c:>16}' for c in CELLS))
    for arm in arms:
        row = {}
        for scen in CELLS:
            v = arm_vec(arm, scen)
            row[scen] = float(np.nanmean(v)) if np.isfinite(v).any() else None
        table[arm] = row
        print(f'  {arm:<16}' + ''.join(f'{(row[c] if row[c] is not None else float("nan")):>16.4f}'
                                      for c in CELLS))
    print(f'  {"panel (50-150m,1000s)":<16}')
    for t in ('dijkstra', 'da_gpsr'):
        print(f'    {t:<14}' + ''.join(f'{PANEL_REF[c][t]:>16.4f}' for c in CELLS))
    out['mean_pdr'] = table

    print('\n  drop shares (fraction of generated), medium_slow:')
    for arm in ('dijkstra', 'dijkstra_nodrop', 'da_gpsr', 'S_dij_ms', 'S_da_ms'):
        d = drops(arm, 'medium_slow')
        print(f'    {arm:<16}' + '  '.join(f'{k}={v:.3f}' for k, v in sorted(d.items())))
    out['drops_medium_slow'] = {a: drops(a, 'medium_slow') for a in
                                ('dijkstra', 'dijkstra_nodrop', 'da_gpsr',
                                 'S_dij_ms', 'S_da_ms', 'GLOBAL', 'PERCELL')}

    def T(a1, a2, scen):
        x, y = arm_vec(a1, scen), arm_vec(a2, scen)
        ok = np.isfinite(x) & np.isfinite(y)
        return _paired(x[ok], y[ok])

    pre = {'P0_ms': T('dijkstra', 'da_gpsr', 'medium_slow'),
           'P0_ds': T('da_gpsr', 'dijkstra', 'dense_slow'),
           'P0_vd': T('da_gpsr', 'dijkstra', 'very_dense')}
    prim = _holm({'R1 S_dij_ms-S_da_ms (ms)': T('S_dij_ms', 'S_da_ms', 'medium_slow'),
                  'C1 PERCELL-GLOBAL (ds)': T('PERCELL', 'GLOBAL', 'dense_slow'),
                  'C2 PERCELL-GLOBAL (vd)': T('PERCELL', 'GLOBAL', 'very_dense'),
                  'G1 PERCELL-GLOBAL (ms)': T('PERCELL', 'GLOBAL', 'medium_slow')})
    sec = _holm({'M1 dijkstra_nodrop-dijkstra (ms)': T('dijkstra_nodrop', 'dijkstra', 'medium_slow'),
                 'M2 dijkstra_nodrop-da_gpsr (ms)': T('dijkstra_nodrop', 'da_gpsr', 'medium_slow'),
                 'I1 S_dij_ms-dijkstra (ms)': T('S_dij_ms', 'dijkstra', 'medium_slow'),
                 'I2 S_da_ms-da_gpsr (ms)': T('S_da_ms', 'da_gpsr', 'medium_slow'),
                 'I3 GLOBAL-da_gpsr (ds)': T('GLOBAL', 'da_gpsr', 'dense_slow'),
                 'I4 GLOBAL-da_gpsr (vd)': T('GLOBAL', 'da_gpsr', 'very_dense'),
                 'H1 GLOBAL_HO-GLOBAL (ms)': T('GLOBAL_HO', 'GLOBAL', 'medium_slow'),
                 'H2 GLOBAL_HO-da_gpsr (ms)': T('GLOBAL_HO', 'da_gpsr', 'medium_slow')})

    def show(title, tests, holm=True):
        print(f'\n  {title}')
        for k, t in tests.items():
            ph = f'  p_holm={t["p_holm"]:.3g}' if holm and 'p_holm' in t else ''
            print(f'    {k:<36} {100*t["mean"]:+6.2f} pp  CI[{100*t["lo"]:+.2f}, '
                  f'{100*t["hi"]:+.2f}]  p={t["p"]:.3g}{ph}  n={t["n"]}')
    show('PRECONDITION (teacher gaps at this operating point)', pre, holm=False)
    show('PRIMARY FAMILY (Holm)', prim)
    show('SECONDARY FAMILY (Holm)', sec)
    out['precondition'], out['primary'], out['secondary'] = pre, prim, sec

    # training-level evidence
    tr = _load(args, 'train.json', {})
    print('\n  student validation accuracy (own labels, contested):')
    for k, v in sorted(tr.items()):
        print(f'    {k:<16} {v["val_contested"]:.4f}   n_train={v["n_train"]}')
    if xe:
        print('\n  cross-evaluation (raw accuracy on each validation set, mean over model seeds):')
        for scen, teacher in GEN_COMBOS:
            cells = []
            for name in STUDENTS:
                xs = [xe[f'{name}:{2000+ms}|{scen}|{teacher}']['acc']
                      for ms in range(args.model_seeds)
                      if f'{name}:{2000+ms}|{scen}|{teacher}' in xe]
                cells.append(f'{name}={np.mean(xs):.3f}' if xs else f'{name}=--')
            print(f'    {scen}/{teacher}-labels: ' + '  '.join(cells))
        out['cross_eval'] = xe

    # label agreement on recorded states + regime identifiability
    agree = {}
    for scen in CELLS:
        a = []
        for sd in list(args.train_seeds) + list(args.val_seeds):
            p = os.path.join(args.out, 'raw', f'{scen}_da_gpsr_s{sd}.npz')
            if os.path.isfile(p):
                z = dict(np.load(p))
                ok = z['lab_dijkstra'] >= 0
                a.append((z['lab_dijkstra'][ok] == z['lab_da_gpsr'][ok]).mean())
        agree[scen] = float(np.mean(a)) if a else None
    print('\n  dijkstra vs da_gpsr label agreement on recorded states (da_gpsr behaviour):')
    for s, v in agree.items():
        print(f'    {s:<12} {v if v is None else round(v, 3)}')
    out['label_agreement'] = agree
    clf = _cell_classifier(args)
    if clf:
        print(f'\n  regime identifiability (local obs -> "is this medium_slow?"):  '
              f'balanced acc {clf["balanced_acc_val"]:.3f} overall, '
              f'{clf["balanced_acc_val_on_conflict_rows"]:.3f} on rows where the '
              f'two teachers disagree (n={clf["n_val_conflict"]})')
        out['regime_classifier'] = clf

    # verdict
    def sig_ge(t, thr):
        return t.get('p_holm', 1) < 0.05 and t['mean'] >= thr
    P0 = all(t['lo'] > 0 for t in pre.values() if t['n'] >= 2)
    r1 = prim['R1 S_dij_ms-S_da_ms (ms)']
    g1 = prim['G1 PERCELL-GLOBAL (ms)']
    c1, c2 = prim['C1 PERCELL-GLOBAL (ds)'], prim['C2 PERCELL-GLOBAL (vd)']
    contam = any(t.get('p_holm', 1) < 0.05 and t['mean'] <= -MIN_PP for t in (c1, c2))
    print('\n' + '=' * 96)
    if not P0:
        verdict = 'INCONCLUSIVE -- teacher gaps did not reproduce at this operating point'
    elif sig_ge(r1, MIN_PP) and sig_ge(g1, MIN_PP) and not contam:
        verdict = 'PER-CELL JUSTIFIED'
    elif (not sig_ge(r1, MIN_PP)) or contam:
        verdict = 'SINGLE da_gpsr'
    else:
        verdict = 'MIXED -- read the numbers'
    print(f'  PRE-REGISTERED VERDICT: {verdict}')
    print('=' * 96)
    out['verdict'] = verdict
    _save(args, 'summary.json', out)
    print(f'  saved {os.path.join(args.out, "summary.json")}\n')


# ─────────────────────────────────────────────────────────────────────────────
def _load(args, name, default):
    p = os.path.join(args.out, name)
    return json.load(open(p)) if os.path.isfile(p) else default


def _save(args, name, obj):
    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, name + '.tmp')
    json.dump(obj, open(tmp, 'w'), indent=1, default=float)
    os.replace(tmp, os.path.join(args.out, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['all', 'gen', 'train', 'rollout', 'analyze'])
    ap.add_argument('--out', default='results/teacher_choice')
    ap.add_argument('--duration', type=float, default=300.0)
    ap.add_argument('--z_min', type=float, default=100.0)
    ap.add_argument('--z_max', type=float, default=300.0)
    ap.add_argument('--initial_energy', type=float, default=8000.0)
    ap.add_argument('--epsilon', type=float, default=0.10)
    ap.add_argument('--record_prob', type=float, default=0.05)
    ap.add_argument('--train_seeds', type=int, nargs='+', default=list(range(101, 111)))
    ap.add_argument('--val_seeds', type=int, nargs='+', default=[136, 137])
    ap.add_argument('--rollout_seeds', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--model_seeds', type=int, default=2)
    ap.add_argument('--max_workers', type=int, default=8)
    ap.add_argument('--device', default=None)
    ap.add_argument('--smoke', action='store_true',
                    help='tiny run (30 s episodes, 1 seed each) -- plumbing only, '
                         'numbers are meaningless')
    args = ap.parse_args()
    if args.device is None:
        try:
            import torch
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            args.device = 'cpu'
    if args.smoke:
        args.duration, args.record_prob = 30.0, 0.3
        args.train_seeds, args.val_seeds = [101], [136]
        args.rollout_seeds, args.model_seeds = [1, 2], 1
        args.out = args.out.rstrip('/\\') + '_smoke'
    bad = [s for s in args.rollout_seeds if s in args.train_seeds or s in args.val_seeds]
    if bad:
        raise SystemExit(f'rollout seeds {bad} overlap training/validation seeds')

    print('=' * 96)
    print('  TEST: one global teacher vs per-cell teachers')
    print('=' * 96)
    print(f'  out={args.out}  device={args.device}  workers={args.max_workers}'
          f'{"   *** SMOKE: numbers meaningless ***" if args.smoke else ""}')
    print(f'  {args.duration:.0f}s episodes, altitude {args.z_min:.0f}-{args.z_max:.0f} m, '
          f'battery {args.initial_energy:.0f}, cells {CELLS}')
    assert_pins()
    print('  drift pins: restricted pickers == real teachers (300 graphs), and the '
          'pin FAILS on two broken pickers  OK')
    os.makedirs(args.out, exist_ok=True)
    # v28: never mix simulators in one output folder (the mobility fix changes every
    # trajectory longer than ~30 s)
    prev = _load(args, 'config.json', None)
    if prev is not None and prev.get('mobility') != MOBILITY_VERSION:
        raise SystemExit(f"  {args.out} was produced with mobility "
                         f"{prev.get('mobility') or 'pre-v28 (waypoint trapping)'}; this "
                         f'simulator is {MOBILITY_VERSION}. Use a new --out.')
    args.mobility = MOBILITY_VERSION
    _save(args, 'config.json', vars(args))

    t0 = time.time()
    if args.stage in ('all', 'gen'):
        stage_gen(args)
    if args.stage in ('all', 'train'):
        stage_train(args)
    if args.stage in ('all', 'rollout'):
        stage_rollout(args)
    if args.stage in ('all', 'analyze'):
        stage_analyze(args)
    print(f'  total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
