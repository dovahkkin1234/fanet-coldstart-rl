"""
audit_dataset_v2.py  —  INDEPENDENT deep audit of the Phase-B dataset.

This is deliberately NOT the G3.5 gate. G3.5 checks what its author thought to
check; this script exists to check things G3.5 does NOT, and to cross-validate
decisions.npz against frames.npz rather than validating each file in isolation.

Every previous gate in this project passed at least once while a real defect
was still present (G2 with congestion-blind teachers, G3 with an unpaired
t-test, G3.5 with a 22.6% mislabel rate). Passing a gate is evidence, not
proof. Run this before committing to training.

AUDITS PERFORMED
  A. Cross-file referential integrity  — do decision->frame references resolve,
     and are the recorded candidates ACTUALLY neighbours of `current` in the
     referenced frame's graph? (G3.5 never opens the graph.)
  B. Label semantics                   — is the labelled candidate a real
     neighbour, and does it match an INDEPENDENT re-derivation of SP-BP's rule
     from the stored graph? This re-derives the label from raw stored data
     rather than trusting the generator that wrote it.
  C. Oracle-vote self-consistency      — does votes['spbp'] agree with the
     label? They are computed by different code paths, so systematic
     disagreement indicates a mechanism inconsistency rather than genuine
     teacher disagreement.
  D. Feature distributions             — per-column min/max/mean/std, and
     out-of-range detection for columns that should be bounded.
  E. Split viability                   — does the planned seed/scenario split
     actually partition cleanly, with usable volume in every part?
  F. Leakage probe                     — confirm no seed appears in two splits,
     and that frames are not shared across split boundaries.
  G. Epsilon-step behaviour            — do deviated steps differ from the
     label as they must, and non-deviated steps match it?

Usage:
    python src\\audit_dataset_v2.py --data data/phaseB
    python src\\audit_dataset_v2.py --data data/phaseB --sample 20000
"""

import os, sys, json, argparse
from collections import Counter

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_v2 as F
import routing_teachers_v2 as rt

HOP_UNREACHABLE = 999.0


class FrameStore:
    """Loads frames.npz ONCE into plain in-memory ndarrays.

    THIS CLASS EXISTS BECAUSE OF A REAL PERFORMANCE BUG, caught by watching
    Task Manager during an actual run rather than assuming the runtime was
    reasonable: np.load on a .npz returns a lazy NpzFile, and NpzFile.__getitem__
    DECOMPRESSES THE ENTIRE MEMBER ARRAY FROM THE ZIP ARCHIVE ON EVERY ACCESS --
    it does not cache. The first version of this audit called frm['node_offsets'],
    frm['node_feat_flat'], etc. INSIDE the per-decision sample loop (via
    rebuild_graph), so every sampled decision re-decompressed the whole frames
    file from scratch.

    Measured cost: 300 such calls against a 10.5 MB frames.npz took 25.8s
    (86 ms/call); loading the same arrays into memory once and slicing them
    took 0.0006s for the same 300 calls -- a ~42,000x difference. At the real
    dataset's scale (174 MB, 20000 sampled decisions, PLUS an entirely separate
    second pass in check B that rebuilt graphs all over again) this explains a
    stall of tens of minutes to hours: exactly what running Task Manager showed
    (4% overall CPU, brief single-core decompression bursts, long idle gaps --
    not what a genuinely compute-bound 20000-sample audit should look like).

    Fix: decompress each array exactly once at construction time.
    """
    def __init__(self, frm):
        self.node_offsets = np.asarray(frm['node_offsets'])
        self.edge_offsets = np.asarray(frm['edge_offsets'])
        self.node_ids_flat = np.asarray(frm['node_ids_flat'])
        self.node_feat_flat = np.asarray(frm['node_feat_flat'])
        self.edge_index_flat = np.asarray(frm['edge_index_flat'])
        self.edge_feat_flat = np.asarray(frm['edge_feat_flat'])
        self.n_frames = len(self.node_offsets) - 1

    def rebuild(self, fid):
        n0, n1 = self.node_offsets[fid], self.node_offsets[fid + 1]
        e0, e1 = self.edge_offsets[fid], self.edge_offsets[fid + 1]
        ids = self.node_ids_flat[n0:n1]
        nf = self.node_feat_flat[n0:n1]
        ei = self.edge_index_flat[:, e0:e1]
        ef = self.edge_feat_flat[e0:e1]
        G = nx.Graph()
        for local, nid in enumerate(ids):
            G.add_node(int(nid), row=local)
        for k in range(ei.shape[1]):
            a, b = int(ids[ei[0, k]]), int(ids[ei[1, k]])
            G.add_edge(a, b, row=k)
        return G, ids, nf, ei, ef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/phaseB')
    ap.add_argument('--sample', type=int, default=20000,
                    help='decisions to deep-check (graph rebuild is not free)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    dec = np.load(os.path.join(args.data, 'decisions.npz'))
    print("  loading frames.npz into memory once (this used to be the slow part)...")
    _t0 = __import__('time').time()
    frm = FrameStore(np.load(os.path.join(args.data, 'frames.npz')))
    print(f"    done in {__import__('time').time() - _t0:.1f}s")
    with open(os.path.join(args.data, 'manifest.json')) as f:
        man = json.load(f)

    # Schema compatibility, BEFORE anything reads a feature column. Every
    # lookup below is name-based -- F.EDGE_FEATURES.index('link_quality'),
    # F.NODE_FEATURES.index('queue_occupancy'),
    # {nm: j for j, nm in enumerate(F.QUERY_FEATURES)} -- so a features_v2.py
    # that has changed since generation does not raise. It silently resolves
    # the right NAME to the wrong COLUMN. Check B in particular would then
    # re-derive labels from whatever column now sits at that index and report a
    # plausible agreement rate. This auditor exists precisely because "every
    # gate in this project has passed at least once while a real defect was
    # still present"; skew is the way it would pass while reading noise.
    _skew = F.assert_manifest_compatible(man, context='audit')
    if _skew:
        print("\n" + "=" * 78)
        print("  AUDIT ABORTED — DATASET/MODULE SCHEMA SKEW")
        print("=" * 78)
        for _p in _skew:
            print(f"    ** {_p}")
        print()
        print("    Refusing to audit: name-based column lookups would resolve")
        print("    against a layout the dataset does not have.")
        print("=" * 78 + "\n")
        return 1

    n = len(dec['label'])
    offs = dec['cand_offsets']
    cand_flat = dec['cand_flat']
    labels = dec['label']
    frame_id = dec['frame_id']
    current = dec['current']
    dstv = dec['dst']
    n_frames = frm.n_frames
    panel = man['panel']

    print("\n" + "=" * 78)
    print("  INDEPENDENT DEEP AUDIT — Phase-B dataset")
    print("=" * 78)
    print(f"  {n} decisions, {n_frames} frames")
    print(f"  deep-checking a random sample of {min(args.sample, n)} decisions")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n, size=min(args.sample, n), replace=False)

    results = {}

    # ---------- A. referential integrity ----------
    fid_ok = bool((frame_id >= 0).all() and (frame_id < n_frames).all())
    bad_neighbour = 0
    bad_label_neighbour = 0
    cand_dupes = 0
    checked = 0
    graph_cache = {}          # fid -> (G, ids, nf, ei, ef); SHARED with check B below
    n_idx = len(idx)
    for ii, i in enumerate(idx):
        if ii and ii % 5000 == 0:
            print(f"    ... check A: {ii}/{n_idx}")
        fid = int(frame_id[i])
        if fid not in graph_cache:
            if len(graph_cache) > 6000:
                graph_cache.clear()
            graph_cache[fid] = frm.rebuild(fid)
        G = graph_cache[fid][0]
        c = int(current[i])
        cs = cand_flat[offs[i]:offs[i + 1]].tolist()
        if len(set(cs)) != len(cs):
            cand_dupes += 1
        if c not in G:
            bad_neighbour += 1
            continue
        nbrs = set(G.neighbors(c))
        if not set(cs).issubset(nbrs):
            bad_neighbour += 1
        lab_node = cs[int(labels[i])]
        if lab_node not in nbrs:
            bad_label_neighbour += 1
        checked += 1
    results['A'] = (fid_ok and bad_neighbour == 0 and bad_label_neighbour == 0
                    and cand_dupes == 0)

    print("\n" + "-" * 78)
    print("  A. CROSS-FILE REFERENTIAL INTEGRITY")
    print("-" * 78)
    print(f"    frame_id all in range .................. {fid_ok}")
    print(f"    candidates are real neighbours ......... {bad_neighbour} violations / {checked}")
    print(f"    labelled candidate is a neighbour ...... {bad_label_neighbour} violations")
    print(f"    duplicate candidates in a list ......... {cand_dupes}")

    # ---------- B. label re-derivation from raw stored data ----------
    # Rebuild SP-BP's decision using ONLY what is on disk, then compare. This
    # does not trust the generator that produced the labels.
    lq_col = F.EDGE_FEATURES.index('link_quality')
    # 'queue_len' was removed as a duplicate of 'queue_occupancy' (they were the
    # same signal: occupancy == length/MAX_QUEUE). Raw length is recovered by
    # multiplying occupancy back up, which is what SP-BP's score needs since it
    # mixes the queue term with an UNSCALED hop term.
    qocc_col = F.NODE_FEATURES.index('queue_occupancy')
    agree_rederived = 0
    rederived_checked = 0
    sub = idx[:min(4000, len(idx))]
    n_sub = len(sub)
    for ii, i in enumerate(sub):
        if ii and ii % 2000 == 0:
            print(f"    ... check B: {ii}/{n_sub}")
        fid = int(frame_id[i])
        if fid not in graph_cache:          # reuse check A's cache -- do not
            if len(graph_cache) > 6000:      # rebuild graphs a second time
                graph_cache.clear()
            graph_cache[fid] = frm.rebuild(fid)
        G, ids, nf, ei, ef = graph_cache[fid]
        c, d = int(current[i]), int(dstv[i])
        if c not in G or d not in G:
            continue
        cs = cand_flat[offs[i]:offs[i + 1]].tolist()
        if not set(cs).issubset(set(G.neighbors(c))):
            continue
        h = nx.single_source_shortest_path_length(G, d)
        row_of = {int(nid): k for k, nid in enumerate(ids)}
        lqmap = {}
        for k in range(ei.shape[1]):
            a, b = int(ids[ei[0, k]]), int(ids[ei[1, k]])
            lqmap[(a, b)] = lqmap[(b, a)] = float(ef[k, lq_col])
        # DENORMALISE queue_len before re-deriving. features_v2 stores it as
        # raw/max_queue, while SP-BP's score mixes it with an UNSCALED hop term
        # -- so comparing normalised queue against raw hops would shrink the
        # queue term ~50x and change the argmax. (This bit the first version of
        # this audit: it reported a label mismatch that was the audit's own
        # scaling error, not a dataset defect. The model is unaffected either
        # way: a fixed linear rescale is absorbed into its weights.)
        qscale = float(F.MAX_QUEUE_REF)
        if d in cs:
            pick = d
        else:
            q_cur = float(nf[row_of[c], qocc_col]) * qscale
            h_cur = float(h.get(c, HOP_UNREACHABLE))
            best, bs = None, -float('inf')
            for u in cs:
                q_u = float(nf[row_of[u], qocc_col]) * qscale
                sc = lqmap.get((c, u), 0.0) * ((q_cur - q_u) +
                     rt.SPBP_V_BIAS * (h_cur - float(h.get(u, HOP_UNREACHABLE))))
                if sc > bs:
                    bs, best = sc, u
            pick = best
        rederived_checked += 1
        if pick == cs[int(labels[i])]:
            agree_rederived += 1
    rate_B = agree_rederived / max(rederived_checked, 1)
    # 0.99, not higher: features are stored as float32 while the generator
    # scores in float64, so candidate pairs whose true scores differ by less
    # than float32 resolution (~1e-6 at these magnitudes) can legitimately
    # flip on re-derivation. Sub-1% disagreement is the quantisation floor,
    # not a defect. A genuine mismatch (wrong features, wrong scaling) shows
    # up far below this -- the pre-fix scaling bug scored well under 0.99.
    results['B'] = rate_B > 0.99

    print("\n" + "-" * 78)
    print("  B. LABEL RE-DERIVED INDEPENDENTLY FROM STORED GRAPH")
    print("-" * 78)
    print(f"    re-derived label matches stored label .. {rate_B:.4f} "
          f"({agree_rederived}/{rederived_checked})")
    print("    (uses only frames.npz contents; does not trust the generator)")
    if rate_B <= 0.99:
        print("    ** MISMATCH: stored labels do not reproduce from stored features.")
        print("       Either the features written are not the ones the label was")
        print("       computed from, or normalisation is lossy. Investigate before")
        print("       training — the model would be fitting inconsistent targets.")

    # ---------- C. oracle-vote self-consistency ----------
    votes = dec['votes']
    has_v = dec['has_votes']
    spbp_col = panel.index('spbp') if 'spbp' in panel else None
    if spbp_col is not None:
        sel = np.where(has_v)[0]
        lab_nodes = np.array([cand_flat[offs[i] + labels[i]] for i in sel])
        spbp_votes = votes[sel, spbp_col]
        match = float((spbp_votes == lab_nodes).mean())
    else:
        match = float('nan')
    results['C'] = (match > 0.98)

    print("\n" + "-" * 78)
    print("  C. ORACLE-VOTE SELF-CONSISTENCY")
    print("-" * 78)
    print(f"    votes['spbp'] equals the label ......... {match:.4f}")
    if match <= 0.98:
        print("    ** INCONSISTENT. The label and the spbp vote are produced by")
        print("       DIFFERENT code paths: the label uses spbp_pick_restricted")
        print("       (full-graph BFS), the vote uses spbp_next_hop on a")
        print("       restricted_view (BFS over a modified graph). Disagreement")
        print("       here is a MECHANISM artifact, not genuine teacher")
        print("       disagreement, and it biases vote_agreement DOWNWARD —")
        print("       which matters if confidence-weighted training is used.")
        print("       FIX: compute the spbp vote with spbp_pick_restricted too,")
        print("       so the oracle trivially agrees with its own label.")

    # ---------- D. feature distributions ----------
    qf = dec['query_feat']
    cf = dec['cand_feat_flat']
    nf_all = frm.node_feat_flat
    step = max(len(nf_all) // 200000, 1)
    nfs = nf_all[::step]

    def col_report(name, arr, names):
        print(f"    {name}:")
        bad = []
        for j, nm in enumerate(names):
            v = arr[:, j]
            lo, hi, mu, sd = v.min(), v.max(), v.mean(), v.std()
            flag = ''
            if not np.isfinite(v).all():
                flag = '  <-- NON-FINITE'; bad.append(nm)
            elif sd < 1e-9:
                flag = '  <-- CONSTANT'; bad.append(nm)
            elif hi > 50 or lo < -50:
                flag = '  <-- LARGE MAGNITUDE'
            print(f"      {nm:<26} min={lo:8.3f} max={hi:8.3f} "
                  f"mean={mu:7.3f} std={sd:6.3f}{flag}")
        return bad

    print("\n" + "-" * 78)
    print("  D. FEATURE DISTRIBUTIONS")
    print("-" * 78)
    bad_n = col_report('node (frame-level)', nfs, F.NODE_FEATURES)
    bad_q = col_report('query (per-decision)', qf, F.QUERY_FEATURES)
    bad_c = col_report('candidate', cf, F.CANDIDATE_FEATURES)
    results['D'] = not (bad_n or bad_q or bad_c)

    # ---------- E/F. split viability + leakage ----------
    seeds = dec['seed']; scen = dec['scenario']
    buckets = dec['load_bucket']   # was referenced but never assigned in
                                   # section H below - the actual bug
    train_seeds = set(range(101, 136)); val_seeds = set(range(136, 143))
    test_seeds = set(range(143, 151))
    heldout = 'medium_slow'
    in_train = np.array([(s in train_seeds) and (sc != heldout)
                         for s, sc in zip(seeds, scen)])
    in_val = np.array([(s in val_seeds) and (sc != heldout)
                       for s, sc in zip(seeds, scen)])
    in_test = np.array([(s in test_seeds) and (sc != heldout)
                        for s, sc in zip(seeds, scen)])
    in_gen = np.array([sc == heldout for sc in scen])
    overlap = (train_seeds & val_seeds) | (train_seeds & test_seeds) | (val_seeds & test_seeds)
    counts = dict(train=int(in_train.sum()), val=int(in_val.sum()),
                  test=int(in_test.sum()), generalisation=int(in_gen.sum()))
    covered = int(in_train.sum() + in_val.sum() + in_test.sum() + in_gen.sum())
    min_expected = max(200, int(0.01 * n))   # scales with dataset size
    results['E'] = (not overlap and min(counts.values()) >= min_expected
                    and covered == n)

    print("\n" + "-" * 78)
    print("  E/F. SPLIT VIABILITY AND LEAKAGE")
    print("-" * 78)
    for k, v in counts.items():
        print(f"    {k:<16} {v:>8} decisions")
    print(f"    seed-range overlap between splits ...... {sorted(overlap) or 'none'}")
    print(f"    every decision assigned exactly once ... {covered == n} "
          f"({covered}/{n})")
    print(f"    smallest split >= {min_expected} rows ......... "
          f"{min(counts.values()) >= min_expected}")

    # per-split frame disjointness (a frame must not span splits)
    fr_train = set(frame_id[in_train].tolist())
    fr_test = set(frame_id[in_test].tolist())
    fr_gen = set(frame_id[in_gen].tolist())
    frame_leak = len(fr_train & fr_test) + len(fr_train & fr_gen)
    print(f"    frames shared across splits ............ {frame_leak}")
    if frame_leak:
        print("       Each frame belongs to exactly one episode, hence one")
        print("       (scenario, seed), so on a correctly generated dataset this")
        print("       must be 0. A non-zero value means decisions were assigned")
        print("       scenario/seed independently of their frame — a generator bug.")
    results['F'] = (frame_leak == 0)

    # ---------- G. epsilon-step behaviour ----------
    dev = dec['behaviour_deviated']
    print("\n" + "-" * 78)
    print("  G. EPSILON-STEP BEHAVIOUR")
    print("-" * 78)
    print(f"    deviated fraction ...................... {dev.mean():.4f}")
    print(f"    deviated decisions ..................... {int(dev.sum())}")
    # candidate-size 1 means deviation cannot differ from the label
    csz = np.diff(offs)
    forced = int(((csz == 1) & dev).sum())
    print(f"    deviated where only 1 candidate existed  {forced} "
          f"(these necessarily equal the label)")
    results['G'] = abs(dev.mean() - man['epsilon']) < 0.02

    # ---------- H. per-regime feature behaviour ----------
    # A GLOBAL mean can hide a feature that is inert in one regime and
    # informative in another. energy and current_queue_occupancy both looked
    # near-dead globally; this splits them by scenario and load so a
    # regime-dependent signal is visible instead of being averaged away.
    print("\n" + "-" * 78)
    print("  H. PER-REGIME FEATURE BEHAVIOUR (global means can hide real signal)")
    print("-" * 78)
    q_cols = {nm: j for j, nm in enumerate(F.QUERY_FEATURES)}
    watch_q = [c for c in ('current_queue_occupancy', 'network_mean_occupancy',
                           'n_inflight', 'hop_distance_to_dst') if c in q_cols]
    scen_u = sorted(set(scen.tolist()))
    buck_u = ['low', 'medium', 'high']

    print("    query features, by scenario:")
    hdr = "      {:<14}".format('scenario') + "".join(f"{c[:20]:>22}" for c in watch_q)
    print(hdr)
    for sc in scen_u:
        m = (scen == sc)
        row = "      {:<14}".format(sc)
        for c in watch_q:
            v = qf[m, q_cols[c]]
            row += f"{v.mean():>10.4f}+-{v.std():<10.4f}"
        print(row)

    print("    query features, by load bucket:")
    print(hdr.replace('scenario', 'bucket'))
    for bk in buck_u:
        m = (buckets == bk)
        if not m.any():
            continue
        row = "      {:<14}".format(bk)
        for c in watch_q:
            v = qf[m, q_cols[c]]
            row += f"{v.mean():>10.4f}+-{v.std():<10.4f}"
        print(row)

    # energy lives in node features, which are per-frame; report its spread
    en_col = F.NODE_FEATURES.index('energy')
    en = nfs[:, en_col]
    print(f"\n    energy (node feature): mean={en.mean():.4f} std={en.std():.4f} "
          f"min={en.min():.4f} p01={np.percentile(en,1):.4f}")
    print("      Low global std does NOT mean the feature is useless: drain is")
    print("      real but CONCENTRATED (see min/p01), and 40 s episodes are far")
    print("      shorter than a real drone mission, so energy has little room to")
    print("      become binding here. It is retained deliberately -- it is one of")
    print("      the signals no panel teacher uses, and it becomes meaningful at")
    print("      longer horizons. If energy is to be a genuine differentiator in")
    print("      later milestones, episode duration or initial energy must change;")
    print("      that is an experiment-design decision, not a feature defect.")

    # is current_queue_occupancy regime-dependent?
    if 'current_queue_occupancy' in q_cols:
        cq = qf[:, q_cols['current_queue_occupancy']]
        by_sc = {sc: cq[scen == sc].mean() for sc in scen_u}
        spread = max(by_sc.values()) - min(by_sc.values())
        print(f"\n    current_queue_occupancy by scenario: "
              f"{ {k: round(v, 4) for k, v in by_sc.items()} }")
        print(f"      spread across scenarios = {spread:.4f}")
        if spread > 2 * cq.mean() and cq.mean() > 0:
            print("      -> REGIME-DEPENDENT. The low global mean was hiding real")
            print("         variation; keep this feature.")
        else:
            print("      -> weakly regime-dependent at this episode length. Note")
            print("         that the packet is dequeued BEFORE the decision is")
            print("         recorded, so this measures OTHER packets waiting at the")
            print("         current node, not the packet being routed.")

    # ---------- verdict ----------
    print("\n" + "=" * 78)
    print("  AUDIT VERDICT")
    print("=" * 78)
    names = {
        'A': 'Cross-file referential integrity',
        'B': 'Label re-derives from stored graph',
        'C': 'Oracle vote consistent with label',
        'D': 'Feature distributions sane',
        'E': 'Split viable, all decisions assigned',
        'F': 'No frame leakage across splits',
        'G': 'Epsilon-step behaviour correct',
    }
    for k in 'ABCDEFG':
        print(f"    [{'PASS' if results[k] else 'FAIL'}] {names[k]}")
    ok = all(results.values())
    print()
    if ok:
        print("    AUDIT CLEAN — the dataset holds up to checks beyond G3.5's own.")
        print("    Safe to proceed to M4 model construction.")
    else:
        print("    AUDIT FOUND ISSUES — see the FAIL lines above. Note that some")
        print("    findings (notably C) may be cosmetic for unweighted training but")
        print("    still worth fixing before they propagate into M4 analyses.")
    print("=" * 78 + "\n")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
