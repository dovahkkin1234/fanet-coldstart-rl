"""
generate_dataset_v2.py  —  M3.5 Phase-B dataset generation for M4 supervised pretraining.

Produces oracle-labelled routing decisions from the M2 multi-packet simulator,
driven by the M3 oracle teacher (SP-BP, which G3 showed wins all 12 cells).

=============================================================================
DESIGN DECISIONS, AND WHY (both were flagged as open questions before writing)
=============================================================================

1. RAGGED CANDIDATE STORAGE  -> ADOPTED.
   Candidates are stored as a flat buffer plus an offsets array, NOT fixed-width
   padded to a MAX_NEIGHBORS constant.

   Justification it cannot backfire: the model (M4 spec section 4) scores
   candidates by attention with masking, so it natively accepts a variable
   candidate count -- fixed width was only ever a storage convenience, never an
   architectural requirement. Meanwhile the risk of fixed width is concrete and
   measured: `very_dense` (45 nodes, comm_range 250, 700x700) has EXPECTED
   degree 18.03, already above Approach-1's MAX_NEIGHBORS=15, so a fixed cap
   would truncate real candidate lists in the densest scenario -- and if the
   truncated-away candidate were the labelled one, the decision would be either
   a crash or (worse) silently mislabelled. Ragged removes that failure mode
   entirely at the cost of one extra offsets array. Strictly safer, zero
   performance or integrity cost.

2. VOTE SUBSAMPLING  -> REJECTED. Full 8-teacher votes are recorded by default.
   The M4 spec floated computing votes on only ~10% of decisions to save the
   8x teacher-query cost, since `vote_agreement` is only an optional confidence
   weight and training defaults to unweighted.

   Rejected because it CAN backfire, asymmetrically: the dataset is generated
   once and then trained against repeatedly. If M4's confidence-weighting
   ablation (spec section 5.2) or any later analysis needs full votes, a 10%
   subsample cannot be repaired without regenerating everything -- whereas the
   cost of computing them now is a one-off few minutes on a 16-core machine.
   Paying a small certain cost to remove a large contingent one is the right
   trade here. `--vote_fraction` remains configurable (default 1.0) purely as
   an escape hatch if measured generation time proves prohibitive.

=============================================================================
LABEL CORRECTNESS -- the subtlest issue in this file
=============================================================================
simulator_v2 applies loop-avoidance AFTER the actor picks: if the chosen hop is
already on the packet's path, it is replaced by `unvisited[0]` (arbitrary
networkx iteration order). Measured SP-BP override rate in G3: ~9-11% of
decisions.

Labelling those decisions with SP-BP's raw pick would teach the model to choose
already-visited nodes; labelling them with `unvisited[0]` would teach an
arbitrary iteration-order rule; dropping them would bias the dataset away from
exactly the recovery states DAgger exists to capture.

Instead: the candidate set IS the visited-excluded set, and the label is SP-BP
RE-SCORED on that set (via a networkx subgraph view, so the teacher's own code
is reused verbatim -- the scoring formula is never duplicated here and cannot
drift from routing_teachers_v2). Because the behaviour policy uses the same
re-scored choice, the base-class override never fires during generation, so
labels equal actions except on deliberate epsilon steps.

=============================================================================
EPSILON-DEVIATION (M4 spec section 2.1, mechanism pinned down)
=============================================================================
With probability epsilon the packet is physically moved to a uniform-random
VALID candidate instead of SP-BP's choice. Three properties enforced here:
  (a) the deviation changes where the packet actually goes -- it is not an
      alternate label recorded on the side;
  (b) it flows through the same _try_forward machinery (ARQ, queue admission,
      energy) as any other hop, because it is returned from _select_next_hop;
  (c) the LABEL at every decision is SP-BP's choice at the CURRENT actual
      state, queried fresh -- never a stale pre-deviation intention. This is
      what makes post-deviation recovery states carry correct labels.

Usage:
    python src\\generate_dataset_v2.py --out data/phaseB --max_workers 16
    python src\\generate_dataset_v2.py --measure_only     # degree audit, no gen
"""

import os, sys, json, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator_v2 import FANETSimulatorV2, TEACHERS, PANEL, TTL
from teacher_panel import scenario_class, load_bucket
import routing_teachers_v2 as rt
import features_v2 as F

# Grid: deliberately DISJOINT seeds from G3's 1-30, so the gate's conclusions
# and the training data are not entangled.
# Config now lives in config_v2.py -- see the note there on why eight
# independent copies of this block were a latent hazard.
from config_v2 import SCENARIOS, RATES, BASE, get_suite, provenance  # noqa: F401
SEEDS = list(range(101, 151))          # 50 seeds, disjoint from G3

ORACLE_TEACHER = 'spbp'                # G3: wins all 12 cells
EPSILON = 0.10


def canonical_candidates(G, current, dst, visited):
    """VALID candidates in CANONICAL order.

    Valid  = neighbours not already on the packet's path (the model must never
             be trained to choose one, since routing would reject it).
    Canonical order = ascending distance-to-destination. Without this the order
             is networkx insertion order, which encodes nothing -- and G3.5's
             trivial-rule baseline ("always pick slot 0") would be measuring
             iteration order rather than a real heuristic. With it, slot 0 IS
             the nearest-to-destination baseline, making that check meaningful.
    """
    dpos = np.array([G.nodes[dst]['x'], G.nodes[dst]['y'], G.nodes[dst]['z']])
    cands = [n for n in G.neighbors(current) if n not in visited]
    def keyfn(u):
        upos = np.array([G.nodes[u]['x'], G.nodes[u]['y'], G.nodes[u]['z']])
        return (float(np.linalg.norm(dpos - upos)), u)   # u breaks exact ties
    return sorted(cands, key=keyfn)


HOP_UNREACHABLE = 999.0   # hop-distance stand-in for nodes dst cannot reach


def spbp_pick_restricted(G, current, dst, cands, h_map,
                         v_bias=rt.SPBP_V_BIAS):
    """SP-BP's own scoring rule, evaluated over a RESTRICTED candidate set.

    WHY THIS EXISTS (this is the fix for a 22.6% label_fallback rate that
    failed gate G3.5 on the first generation run):

    The first implementation restricted the teacher by deleting every visited
    NODE from the graph and re-running spbp_next_hop on the pruned subgraph.
    That is subtly but importantly the wrong algorithm: SP-BP's hop-distance
    term h comes from a BFS rooted at the destination, and pruning nodes
    changes that BFS. In sparse topologies (sparse_fast has mean degree 1.95)
    removing two or three visited nodes routinely severed `current` from `dst`
    entirely, so spbp_next_hop returned None and 22.6% of rows fell back to a
    nearest-to-destination heuristic -- i.e. a fifth of the dataset was
    labelled by something that is NOT the oracle teacher.

    The correct semantics separate two things the old code conflated:
      * hop distances must be computed on the FULL graph, because that is the
        quantity G3 actually validated SP-BP on;
      * only the CHOICE may be restricted to legal (unvisited) candidates.

    So this scores exactly SP-BP's formula using the caller's full-graph BFS
    map, and takes the argmax over `cands` only. Consequences:
      * it can never return None while `cands` is non-empty, so the fallback
        path effectively disappears;
      * it needs no BFS of its own -- it reuses h_map, which _select_next_hop
        already computes for the feature vector, making it FASTER than the
        subgraph approach it replaces.

    Unreachable candidates are scored with HOP_UNREACHABLE rather than skipped,
    so they rank last but remain selectable if nothing else is available.

    DRIFT RISK: this restates SP-BP's formula rather than calling the teacher.
    `assert_no_drift()` below pins it against routing_teachers_v2.spbp_next_hop
    on the unrestricted case, and is executed at import time.
    """
    if dst in cands:
        return dst                      # sink short-circuit, as in spbp_next_hop
    q_cur = float(G.nodes[current].get('queue_len', 0.0))
    h_cur = float(h_map.get(current, HOP_UNREACHABLE))
    best, best_score = None, -float('inf')
    for n in cands:
        q_n = float(G.nodes[n].get('queue_len', 0.0))
        lq = float(G.edges[current, n].get('link_quality', 0.0))
        h_n = float(h_map.get(n, HOP_UNREACHABLE))
        score = lq * ((q_cur - q_n) + v_bias * (h_cur - h_n))
        if score > best_score:
            best_score, best = score, n
    return best


def teacher_on_valid(teacher_name, G, current, dst, visited):
    """Run a NON-oracle panel teacher restricted to unvisited candidates.

    Used only to record the 8-teacher vote vector. Hides just the edges from
    `current` to visited neighbours (a networkx restricted_view, O(1), no copy)
    rather than deleting visited nodes globally -- so each teacher's internal
    graph computations still see essentially the whole graph, for the same
    reason spelled out in spbp_pick_restricted above.
    """
    hide = [(current, v) for v in visited if G.has_edge(current, v)]
    H = nx.restricted_view(G, [], hide)
    try:
        return TEACHERS[teacher_name](H, current, dst)
    except Exception:
        return None


def assert_no_drift():
    """Guard against spbp_pick_restricted drifting from the real SP-BP.

    With nothing visited, the restricted scorer must reproduce spbp_next_hop
    exactly. Runs at import so a divergence surfaces immediately rather than
    silently mislabelling a whole dataset."""
    import numpy as _np
    rng = _np.random.default_rng(0)
    for trial in range(60):
        n = int(rng.integers(5, 12))
        Gt = nx.Graph()
        for i in range(n):
            Gt.add_node(i, x=float(rng.integers(0, 900)), y=float(rng.integers(0, 900)),
                        z=100.0, energy=90.0,
                        queue_occupancy=float(rng.random()),
                        queue_len=float(rng.integers(0, 6)))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.45:
                    Gt.add_edge(i, j, distance=float(rng.integers(50, 300)),
                                link_quality=float(rng.random()),
                                packet_error_rate=float(rng.random()) * 0.3)
        src, dstn = 0, n - 1
        if not Gt.has_node(src) or not Gt.has_node(dstn):
            continue
        nbrs = list(Gt.neighbors(src))
        if not nbrs or not nx.has_path(Gt, src, dstn):
            continue
        h = F.hop_distances_to(Gt, dstn)
        ref = rt.spbp_next_hop(Gt, src, dstn)
        got = spbp_pick_restricted(Gt, src, dstn, nbrs, h)
        if ref is not None and got != ref:
            raise AssertionError(
                f"spbp_pick_restricted drifted from spbp_next_hop "
                f"(trial {trial}: got {got}, expected {ref})")


assert_no_drift()


class DatasetSimulator(FANETSimulatorV2):
    """FANETSimulatorV2 that records oracle-labelled decisions.

    Overrides ONLY _select_next_hop (the extension point added to the base
    class), so every other piece of validated M2 machinery -- ARQ retries,
    queue admission, energy accounting, drop taxonomy, per-packet trajectories
    -- runs completely unchanged.
    """

    def __init__(self, config):
        super().__init__(config)
        self.ds_epsilon = float(config.get('epsilon', EPSILON))
        self.ds_vote_fraction = float(config.get('vote_fraction', 1.0))
        self.ds_rng = np.random.default_rng(self.seed + 900_000)
        self.nc = F.norm_constants(config)

        self.frames = {}          # frame_id -> dict(node_ids,node_feat,edge_index,edge_feat)
        self.decisions = []       # list of dicts
        self._frame_id = -1
        self._cur_G = None
        self._hop_cache = {}      # (frame_id,dst) -> BFS map, reused across packets

    # -- frame bookkeeping -------------------------------------------------
    def _note_frame(self, G):
        """Called once per frame. Caches frame-level features and clears the
        per-frame BFS cache."""
        self._frame_id += 1
        self._cur_G = G
        self._hop_cache = {}
        ids, nf, ei, ef = F.extract_frame(G, self.nc)
        self.frames[self._frame_id] = {
            'node_ids': ids, 'node_feat': nf,
            'edge_index': ei, 'edge_feat': ef,
        }

    def _hops_to(self, G, dst):
        key = dst
        if key not in self._hop_cache:
            self._hop_cache[key] = F.hop_distances_to(G, dst)
        return self._hop_cache[key]

    # -- the only overridden behaviour -------------------------------------
    def _select_next_hop(self, G, pkt, neighbors):
        c, dst = pkt.current, pkt.dst
        visited = set(pkt.path)
        cands = canonical_candidates(G, c, dst, visited)
        if not cands:
            return None                      # no valid route; base class drops it

        # Full-graph BFS from the destination -- the SAME quantity SP-BP uses,
        # cached per (frame, dst) and reused for both the label and the feature
        # vector below.
        h_map = self._hops_to(G, dst)

        label_hop = spbp_pick_restricted(G, c, dst, cands, h_map)
        if label_hop is None or label_hop not in cands:
            # Should now be unreachable: spbp_pick_restricted always returns a
            # member of a non-empty `cands`. Retained as a tripwire so any
            # future regression is flagged rather than silently mislabelled.
            label_hop = cands[0]
            label_fallback = True
        else:
            label_fallback = False
        label_idx = cands.index(label_hop)

        # votes (full by default; see module docstring decision 2)
        votes = {}
        if self.ds_vote_fraction >= 1.0 or self.ds_rng.random() < self.ds_vote_fraction:
            for t in PANEL:
                if t == ORACLE_TEACHER:
                    # Use the SAME computation that produced the label. The
                    # oracle must agree with its own label by construction;
                    # routing it through teacher_on_valid instead would use a
                    # different mechanism (spbp_next_hop on a restricted_view,
                    # i.e. a BFS over a modified graph) and could disagree for
                    # purely mechanical reasons, biasing vote_agreement DOWNWARD
                    # and corrupting any confidence-weighted training variant.
                    votes[t] = label_hop
                else:
                    votes[t] = teacher_on_valid(t, G, c, dst, visited)
            valid_votes = [v for v in votes.values() if v is not None]
            agree = (sum(1 for v in valid_votes if v == label_hop) / len(valid_votes)
                     if valid_votes else 0.0)
            has_votes = True
        else:
            agree, has_votes = float('nan'), False

        # epsilon-deviation: physically changes where the packet goes
        deviated = bool(self.ds_rng.random() < self.ds_epsilon)
        action_hop = (cands[int(self.ds_rng.integers(len(cands)))]
                      if deviated else label_hop)

        n_inflight = self.ts_inflight[-1] if self.ts_inflight else 0
        net_occ = self.ts_mean_occ[-1] if self.ts_mean_occ else 0.0
        qf, cf = F.extract_decision(G, pkt, cands, self.nc, h_map,
                                    n_inflight, net_occ, ttl_const=TTL)

        self.decisions.append({
            'frame_id': self._frame_id,
            'packet_id': pkt.pid,
            'hop_index': pkt.hops,
            'current': c, 'dst': dst,
            'candidates': np.asarray(cands, dtype=np.int32),
            'label': label_idx,
            'label_fallback': label_fallback,
            'query_feat': qf,
            'cand_feat': cf,
            'votes': np.asarray([votes.get(t, -1) if votes.get(t) is not None else -1
                                 for t in PANEL], dtype=np.int32),
            'has_votes': has_votes,
            'vote_agreement': agree,
            'behaviour_deviated': deviated,
            'action': action_hop,
            # outcome fields backfilled after the episode
            'hop_succeeded': None, 'eventual_delivered': None, 'drop_reason': None,
        })
        return action_hop

    # -- hook frame notification into the main loop ------------------------
    def _build_graph(self):
        G = super()._build_graph()
        self._note_frame(G)
        return G

    # -- outcome backfill --------------------------------------------------
    def run(self):
        metrics = super().run()
        by_pid = {tr['pid']: tr for tr in self.completed_trajectories}
        for d in self.decisions:
            tr = by_pid.get(d['packet_id'])
            if tr is None:
                d['eventual_delivered'] = False
                d['drop_reason'] = 'unknown'
            else:
                d['eventual_delivered'] = bool(tr['delivered'])
                d['drop_reason'] = tr['drop_reason'] or ''
            d['hop_succeeded'] = True     # a recorded decision always attempted a hop
        return metrics


# ---------------------------------------------------------------------------
def _run_episode(job):
    """One episode -> (frames, decisions, meta). Module-level for pickling."""
    sc_name, cfg, seed, rate, eps, vote_frac = job
    full = {**BASE, **cfg, 'packet_rate': rate, 'seed': seed,
            'actor': ORACLE_TEACHER, 'epsilon': eps, 'vote_fraction': vote_frac,
            'scenario_id': sc_name}
    sim = DatasetSimulator(full)
    m = sim.run()
    meta = {
        'scenario': sc_name, 'seed': seed, 'packet_rate': rate,
        'scenario_class': scenario_class(full), 'load_bucket': load_bucket(rate),
        'pdr_predrain': m['pdr_predrain'], 'n_decisions': len(sim.decisions),
        'n_frames': len(sim.frames), 'override_rate': m.get('override_rate', 0.0),
    }
    return sim.frames, sim.decisions, meta


def measure_max_degree(max_workers=None):
    """PREREQUISITE AUDIT: measure true maximum node degree across the grid.

    Informational once ragged storage is adopted (no truncation can occur), but
    recorded because the M4 spec called for it and because a degree far above
    expectation would be worth knowing before training."""
    print("\n  Measuring true max node degree across the scenario grid...")
    out = {}
    for sc_name, cfg in SCENARIOS.items():
        mx, tot, cnt = 0, 0, 0
        for seed in SEEDS[:10]:
            full = {**BASE, **cfg, 'packet_rate': RATES[0], 'seed': seed,
                    'actor': ORACLE_TEACHER}
            sim = FANETSimulatorV2(full)
            for d in sim.drones:
                d.step(0.5)
            G = sim._build_graph()
            degs = [G.degree(n) for n in G.nodes()]
            if degs:
                mx = max(mx, max(degs)); tot += sum(degs); cnt += len(degs)
        out[sc_name] = {'max_degree': mx, 'mean_degree': tot / max(cnt, 1)}
        print(f"    {sc_name:<13} max_degree={mx:<4} mean_degree={tot/max(cnt,1):.2f}")
    print(f"\n    Approach-1 MAX_NEIGHBORS was 15; observed max here = "
          f"{max(v['max_degree'] for v in out.values())}.")
    print("    Ragged storage is used, so no truncation occurs regardless.")
    return out


def _split_sizes(all_dec, heldout='medium_slow'):
    """Actual decision counts per split, computed at generation time so the
    manifest records what was really produced rather than what was intended."""
    tr, va, te, ge = 0, 0, 0, 0
    for d in all_dec:
        sc, sd = d['scenario'], int(d['seed'])
        if sc == heldout:
            ge += 1
        elif 101 <= sd <= 135:
            tr += 1
        elif 136 <= sd <= 142:
            va += 1
        elif 143 <= sd <= 150:
            te += 1
    tot = max(tr + va + te + ge, 1)

    # FAIL FAST ON ORPHANED DECISIONS.
    # The split is assigned by HARDCODED seed ranges (train 101-135, val
    # 136-142, test 143-150) while --seeds is a free CLI argument. Seeds
    # outside 101-150 match no branch and are counted into no split, silently.
    # Observed with --seeds 1 2 3: 22,416 of 34,400 decisions (65.2%) orphaned.
    #
    # The independent audit does catch this ("Split viable, all decisions
    # assigned"), and G3.5 does not -- its check 5 and check 7 both pass. So
    # this was never a silent corruption path provided the audit is run. But
    # discovering it after a full generation plus two gates costs a great deal
    # more than discovering it here.
    assigned = tr + va + te + ge
    if assigned != len(all_dec):
        raise AssertionError(
            f"{len(all_dec) - assigned} of {len(all_dec)} decisions "
            f"({100 * (len(all_dec) - assigned) / max(len(all_dec), 1):.1f}%) "
            f"belong to NO split.\n"
            f"  The split is assigned by seed range: train 101-135, "
            f"val 136-142, test 143-150, and any seed for the held-out "
            f"scenario {heldout!r}.\n"
            f"  Seeds seen: {sorted({int(d['seed']) for d in all_dec})}\n"
            f"  Either pass --seeds within 101-150, or update the ranges in "
            f"_split_sizes to match.")

    return {'train': tr, 'val': va, 'test': te, 'generalisation': ge,
            'generalisation_share': round(ge / tot, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/phaseB')
    ap.add_argument('--max_workers', type=int, default=None)
    ap.add_argument('--epsilon', type=float, default=EPSILON)
    ap.add_argument('--vote_fraction', type=float, default=1.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--measure_only', action='store_true')
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print("  M3.5 — PHASE B DATASET GENERATION")
    print("=" * 78)

    degrees = measure_max_degree()
    if args.measure_only:
        return 0

    jobs = [(sc, cfg, sd, r, args.epsilon, args.vote_fraction)
            for sc, cfg in SCENARIOS.items() for r in RATES for sd in args.seeds]
    print(f"\n  {len(jobs)} episodes "
          f"({len(SCENARIOS)} scenarios x {len(RATES)} rates x {len(args.seeds)} seeds)")
    print(f"  oracle teacher = {ORACLE_TEACHER}   epsilon = {args.epsilon}   "
          f"vote_fraction = {args.vote_fraction}")

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    all_frames, all_dec, all_meta = [], [], []
    frame_offset = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_run_episode, j): i for i, j in enumerate(jobs)}
        done = 0
        for fut in as_completed(futs):
            frames, decs, meta = fut.result()
            remap = {}
            for fid, fr in sorted(frames.items()):
                remap[fid] = frame_offset
                fr['global_frame_id'] = frame_offset
                fr['episode_meta'] = meta
                all_frames.append(fr)
                frame_offset += 1
            for d in decs:
                d['frame_id'] = remap[d['frame_id']]
                d['scenario'] = meta['scenario']
                d['seed'] = meta['seed']
                d['packet_rate'] = meta['packet_rate']
                d['scenario_class'] = meta['scenario_class']
                d['load_bucket'] = meta['load_bucket']
                all_dec.append(d)
            all_meta.append(meta)
            done += 1
            if done % max(len(jobs) // 20, 1) == 0:
                print(f"    {done}/{len(jobs)} episodes")

    elapsed = time.time() - t0
    print(f"\n  generated in {elapsed:.0f}s — "
          f"{len(all_frames)} frames, {len(all_dec)} decisions")

    # ---- save: frames (RAGGED via flat buffers + offsets) -----------------
    # NOT object arrays. np.array(list_of_arrays, dtype=object) silently fails
    # when the inner arrays share a leading dimension -- edge_index is always
    # (2, E) with varying E, so numpy tries to build a regular (n_frames, 2, ?)
    # array and raises. Flat-buffer + offsets is the same idiom already used for
    # candidates below: consistent, pickle-free, faster to load and smaller.
    node_counts = np.array([f['node_feat'].shape[0] for f in all_frames], dtype=np.int64)
    edge_counts = np.array([f['edge_index'].shape[1] for f in all_frames], dtype=np.int64)
    node_off = np.concatenate([[0], np.cumsum(node_counts)])
    edge_off = np.concatenate([[0], np.cumsum(edge_counts)])
    np.savez_compressed(
        os.path.join(args.out, 'frames.npz'),
        node_feat_flat=np.concatenate([f['node_feat'] for f in all_frames], axis=0),
        node_ids_flat=np.concatenate([f['node_ids'] for f in all_frames]),
        node_offsets=node_off,
        edge_index_flat=np.concatenate([f['edge_index'] for f in all_frames], axis=1),
        edge_feat_flat=np.concatenate([f['edge_feat'] for f in all_frames], axis=0),
        edge_offsets=edge_off)

    # ---- save: decisions (RAGGED candidates via flat buffer + offsets) ----
    cand_flat, cand_off = [], [0]
    cf_flat = []
    for d in all_dec:
        cand_flat.append(d['candidates'])
        cf_flat.append(d['cand_feat'])
        cand_off.append(cand_off[-1] + len(d['candidates']))
    np.savez_compressed(
        os.path.join(args.out, 'decisions.npz'),
        frame_id=np.array([d['frame_id'] for d in all_dec], dtype=np.int32),
        packet_id=np.array([d['packet_id'] for d in all_dec], dtype=np.int32),
        hop_index=np.array([d['hop_index'] for d in all_dec], dtype=np.int16),
        current=np.array([d['current'] for d in all_dec], dtype=np.int32),
        dst=np.array([d['dst'] for d in all_dec], dtype=np.int32),
        cand_flat=np.concatenate(cand_flat) if cand_flat else np.zeros(0, np.int32),
        cand_offsets=np.array(cand_off, dtype=np.int64),
        cand_feat_flat=(np.concatenate(cf_flat) if cf_flat
                        else np.zeros((0, len(F.CANDIDATE_FEATURES)), np.float32)),
        query_feat=np.array([d['query_feat'] for d in all_dec], dtype=np.float32),
        label=np.array([d['label'] for d in all_dec], dtype=np.int32),
        label_fallback=np.array([d['label_fallback'] for d in all_dec], dtype=bool),
        votes=np.array([d['votes'] for d in all_dec], dtype=np.int32),
        has_votes=np.array([d['has_votes'] for d in all_dec], dtype=bool),
        vote_agreement=np.array([d['vote_agreement'] for d in all_dec], dtype=np.float32),
        behaviour_deviated=np.array([d['behaviour_deviated'] for d in all_dec], dtype=bool),
        eventual_delivered=np.array([d['eventual_delivered'] for d in all_dec], dtype=bool),
        scenario=np.array([d['scenario'] for d in all_dec]),
        seed=np.array([d['seed'] for d in all_dec], dtype=np.int32),
        packet_rate=np.array([d['packet_rate'] for d in all_dec], dtype=np.float32),
        load_bucket=np.array([d['load_bucket'] for d in all_dec]))

    # ---- save: manifest, incl. normalisation constants (MUST be reused) ---
    manifest = {
        'oracle_teacher': ORACLE_TEACHER,
        'epsilon': args.epsilon,
        'vote_fraction': args.vote_fraction,
        'seeds': list(args.seeds),
        'rates': RATES,
        'scenarios': {k: v for k, v in SCENARIOS.items()},
        'base': BASE,
        'panel': PANEL,
        'node_features': F.NODE_FEATURES,
        'edge_features': F.EDGE_FEATURES,
        'query_features': F.QUERY_FEATURES,
        'candidate_features': F.CANDIDATE_FEATURES,
        # Compatibility boundary. Both checkers assert these against the live
        # features_v2 module and abort on mismatch, so a dataset can never be
        # validated by code that disagrees with it about column layout.
        'feature_schema_version': F.FEATURE_SCHEMA_VERSION,
        # Observability scoping: k = neighbourhood radius over which the two
        # load aggregates are computed; None = whole-network
        # (controller-assisted). Recorded so the paper's deployability claim is
        # traceable to the dataset rather than asserted.
        'local_horizon': F.LOCAL_HORIZON,
        'norm_constants_per_scenario': {k: F.norm_constants({**BASE, **v})
                                        for k, v in SCENARIOS.items()},
        'max_degree_audit': degrees,
        # FIX 3: record the intended split and the ACTUAL holdout size at
        # generation time, so the scale of the generalisation set is documented
        # rather than rediscovered later (it is large -- see note below).
        'split_plan': {
            'train_seeds': [101, 135], 'val_seeds': [136, 142],
            'test_seeds': [143, 150],
            'generalisation_scenario': 'medium_slow',
            'note': ('The held-out scenario is excluded from train/val/test by '
                     'construction, not by seed range. medium_slow sits between '
                     'dense (degree 9.2) and sparse (2.5) at degree 4.4, so this '
                     'tests INTERPOLATION to an unseen density rather than '
                     'extrapolation beyond the training range. It is a sizeable '
                     'fraction of all decisions -- report the exact share in the '
                     'paper rather than leaving a reviewer to compute it.'),
        },
        'split_actual_decisions': _split_sizes(all_dec),
        'n_frames': len(all_frames), 'n_decisions': len(all_dec),
        'generation_seconds': elapsed,
        'storage': 'ragged candidates (flat buffer + offsets); no fixed MAX_NEIGHBORS cap',
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    sizes = {n: os.path.getsize(os.path.join(args.out, n)) / 1e6
             for n in ('frames.npz', 'decisions.npz')}
    print(f"  saved to {args.out}/  "
          f"frames={sizes['frames.npz']:.0f}MB  decisions={sizes['decisions.npz']:.0f}MB")
    _sp = manifest['split_actual_decisions']
    print(f"  split: train={_sp['train']} val={_sp['val']} test={_sp['test']} "
          f"generalisation={_sp['generalisation']} "
          f"({100*_sp['generalisation_share']:.0f}% held out as '{'medium_slow'}')")
    print("=" * 78 + "\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
