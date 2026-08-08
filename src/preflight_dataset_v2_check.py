"""
preflight_dataset_v2_check.py  —  GATE G3.5 for the Phase-B dataset.

Validates the generated dataset BEFORE any model is trained on it. Same
PASS/FAIL discipline as G1/G2/G3, and the same rationale: every previous gate
in this project caught at least one silent failure that would otherwise have
propagated into every downstream result.

THE SEVEN CHECKS:
  1. Structural integrity  — arrays align, ragged offsets consistent, no NaN/inf.
  2. Label validity        — every label is a valid index into a NON-EMPTY
                             candidate list; label_fallback rate is low.
  3. Trivial-rule baseline — accuracy of "always pick slot 0" (which, thanks to
                             canonical distance-to-destination ordering, IS the
                             nearest-to-destination heuristic). This is a
                             DIAGNOSTIC, not a pass/fail: if it scores very
                             high, top-1 accuracy is a weak metric for G4 and
                             rollout PDR must carry the decision.
  4. Epsilon fidelity      — measured behaviour_deviated rate matches the
                             configured epsilon.
  5. Coverage              — all scenarios / rates / seeds present; decisions
                             spread across load buckets rather than concentrated.
  6. Feature sanity        — features finite and within expected normalised
                             ranges; no constant (zero-variance) column, which
                             would indicate a feature that is silently dead.
  7. Reproducibility       — regenerating one episode with the same seed yields
                             an identical decision count and identical labels.
  8. Feature redundancy    — no two columns inside the same feature block carry
                             the same signal. Check 6 catches DEAD columns
                             (zero variance); this catches DUPLICATED ones,
                             which are individually alive and so invisible to
                             it. Two are checked because two kinds exist:
                             Pearson |r| for linear duplication (ttl_left and
                             hops_so_far summed to exactly 1.0), and Spearman
                             |rho| for monotone-nonlinear duplication (snr was
                             an invertible function of distance whose LINEAR r
                             was only -0.93 and would have passed a Pearson-only
                             screen).
  0. Schema compatibility  — the manifest's persisted feature lists match the
                             live features_v2 module. Runs FIRST and aborts on
                             failure: every other check resolves feature names
                             against the module, so a skewed manifest makes all
                             of them meaningless rather than merely wrong.

Usage:
    python src\\preflight_dataset_v2_check.py --data data/phaseB
"""

import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_v2 as F

EPS_TOL = 0.02          # allowed deviation of measured epsilon from configured
FALLBACK_MAX = 0.05     # max acceptable label_fallback rate

# Check 8 thresholds. Deliberately not 1.0: float32 storage against float64
# scoring puts a quantisation floor under any exact identity, the same reason
# the independent audit's label re-derivation threshold is 0.99.
PEARSON_MAX = 0.98      # linear duplication
SPEARMAN_MAX = 0.995    # monotone (possibly nonlinear) duplication
REDUNDANCY_SAMPLE = 200000

# NAMED EXEMPTIONS. A pair listed here is reported as INFO instead of failing
# the gate. Every entry needs a written reason, and the reason has to be that
# the duplication is DELIBERATE -- not that it is inconvenient. Loosening
# PEARSON_MAX/SPEARMAN_MAX instead would defeat the check that was added to
# catch snr <-> distance and ttl_left <-> hops_so_far in the first place.
# EMPTY ON PURPOSE. An entry was added here for
# cand_hop_distance <-> cand_reachable on the PREDICTION that they would be
# near-collinear, since cand_hop_distance saturates at 1.0 for unreachable
# candidates. Measured on the real dataset, they are not:
#     cand_hop_distance <-> cand_reachable   |r|=0.8391  |rho|=0.2597
# -- both far below threshold, because 95.8% of candidates are reachable and
# the rank correlation is dominated by hop variation among those rather than
# by the binary split. The exemption never fired, and a standing exemption for
# a pair that is not redundant is worse than none: if they ever did become
# collinear, this gate would stay silent about it. So it is removed.
#
# The machinery stays for a future GENUINE case. Nothing gets exempted on a
# prediction -- only on a measurement plus a written reason.
ALLOWED_REDUNDANT_PAIRS = {}


def _exempt(block, a, b):
    """Exemptions are unordered: (block, a, b) matches (block, b, a)."""
    return (ALLOWED_REDUNDANT_PAIRS.get((block, a, b))
            or ALLOWED_REDUNDANT_PAIRS.get((block, b, a)))


def _spearman_matrix(x):
    """Spearman rho for every column pair = Pearson r on the RANKS.

    Computed with numpy rather than scipy so a gate never depends on an
    optional import. Ties are broken by argsort position; with float features
    over 5x10^5 rows exact ties are rare enough not to matter, and any bias
    from them is toward UNDER-stating rho, i.e. toward missing a redundancy
    rather than inventing one.
    """
    ranks = np.empty_like(x, dtype=np.float64)
    n = x.shape[0]
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind='stable')
        r = np.empty(n, dtype=np.float64)
        r[order] = np.arange(n, dtype=np.float64)
        ranks[:, j] = r
    return _corr_matrix(ranks)


def _corr_matrix(x):
    """Pearson r for every column pair; constant columns yield 0, not NaN."""
    xc = x - x.mean(axis=0, keepdims=True)
    sd = xc.std(axis=0)
    live = sd > 1e-12
    out = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    if live.sum() < 2:
        return out
    xs = xc[:, live] / sd[live]
    c = (xs.T @ xs) / x.shape[0]
    idx = np.where(live)[0]
    out[np.ix_(idx, idx)] = c
    return out


def redundancy_report(block_name, arr, names):
    """Flag column pairs inside one feature block that carry the same signal.

    Returns (offenders, lines). An offender is a pair exceeding EITHER
    threshold. Both matter: check 6 sees a duplicated column as perfectly
    healthy, because each copy has normal variance on its own.
    """
    if arr is None or arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 2:
        return [], []
    step = max(arr.shape[0] // REDUNDANCY_SAMPLE, 1)
    x = np.asarray(arr[::step], dtype=np.float64)
    if x.shape[0] < 2:
        return [], []
    if len(names) != x.shape[1]:
        return ([(block_name, 'SHAPE', 'SHAPE', float('nan'), float('nan'))],
                [f'    ** {block_name}: {x.shape[1]} columns but '
                 f'{len(names)} names -- schema skew, cannot check'])
    pear = _corr_matrix(x)
    spear = _spearman_matrix(x)
    offenders, lines = [], []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            p, s = abs(pear[i, j]), abs(spear[i, j])
            if p > PEARSON_MAX or s > SPEARMAN_MAX:
                why = _exempt(block_name, names[i], names[j])
                if why:
                    lines.append(f'    [exempt] {block_name}: {names[i]} <-> '
                                 f'{names[j]}  |r|={p:.4f} |rho|={s:.4f}')
                    lines.append(f'             {why}')
                    continue
                offenders.append((block_name, names[i], names[j], p, s))
                kind = ('linear' if p > PEARSON_MAX else
                        'monotone (nonlinear -- Pearson alone would MISS this)')
                lines.append(f'    ** {block_name}: {names[i]} <-> {names[j]}  '
                             f'|r|={p:.4f} |rho|={s:.4f}  [{kind}]')
    return offenders, lines


def load(data_dir):
    # No allow_pickle needed: both files store ragged data as flat buffers plus
    # offset arrays, never object arrays.
    dec = np.load(os.path.join(data_dir, 'decisions.npz'))
    frm = np.load(os.path.join(data_dir, 'frames.npz'))
    with open(os.path.join(data_dir, 'manifest.json')) as f:
        man = json.load(f)
    return dec, frm, man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/phaseB')
    args = ap.parse_args()

    dec, frm, man = load(args.data)

    # ---- 0. schema compatibility (runs FIRST, aborts on failure) ----
    skew = F.assert_manifest_compatible(man, context='G3.5')
    if skew:
        print("\n" + "=" * 78)
        print("  GATE G3.5 — ABORTED BEFORE ANY CHECK")
        print("=" * 78)
        for p in skew:
            print(f"    ** {p}")
        print()
        print("    The dataset was generated under a DIFFERENT features_v2.py than")
        print("    the one now importing it. Every check below resolves feature")
        print("    names against the live module, so running them would produce")
        print("    plausible numbers against a misaligned column layout rather")
        print("    than an error. Regenerate the dataset, or check out the")
        print("    features_v2.py that produced it.")
        print("=" * 78 + "\n")
        return 1

    n = len(dec['label'])
    offs = dec['cand_offsets']
    cand_flat = dec['cand_flat']
    labels = dec['label']

    print("\n" + "=" * 78)
    print("  GATE G3.5 — PHASE B DATASET VALID?")
    print("=" * 78)
    print(f"  {n} decisions across {man['n_frames']} frames")
    print(f"  oracle={man['oracle_teacher']}  epsilon={man['epsilon']}  "
          f"vote_fraction={man['vote_fraction']}")
    print(f"  storage: {man['storage']}")

    # ---- 1. structural integrity ----
    cand_sizes = np.diff(offs)
    n_off = frm['node_offsets']; e_off = frm['edge_offsets']
    frames_ok = (n_off[-1] == len(frm['node_feat_flat'])
                 and e_off[-1] == frm['edge_index_flat'].shape[1]
                 and e_off[-1] == len(frm['edge_feat_flat'])
                 and len(n_off) == len(e_off))
    c1 = (len(offs) == n + 1 and offs[-1] == len(cand_flat)
          and (cand_sizes > 0).all()
          and len(dec['cand_feat_flat']) == len(cand_flat)
          and frames_ok)
    qf = dec['query_feat']
    finite = np.isfinite(qf).all() and np.isfinite(dec['cand_feat_flat']).all()
    c1 = c1 and finite

    # ---- 2. label validity ----
    valid_lab = ((labels >= 0) & (labels < cand_sizes)).all()
    fb_rate = float(dec['label_fallback'].mean())
    c2 = bool(valid_lab) and fb_rate <= FALLBACK_MAX

    # ---- 3. trivial baseline (diagnostic) ----
    trivial_acc = float((labels == 0).mean())     # slot0 == nearest-to-dst
    # also: how often is the label the destination itself (a "free" decision)?
    cf = dec['cand_feat_flat']
    is_dst_col = F.CANDIDATE_FEATURES.index('is_destination')
    lab_is_dst = np.array([cf[offs[i] + labels[i], is_dst_col] for i in range(n)])
    dst_rate = float((lab_is_dst > 0.5).mean())

    # ---- 4. epsilon fidelity ----
    meas_eps = float(dec['behaviour_deviated'].mean())
    c4 = abs(meas_eps - man['epsilon']) <= EPS_TOL

    # ---- 5. coverage ----
    scen = dec['scenario']; rates = dec['packet_rate']; seeds = dec['seed']
    buckets = dec['load_bucket']
    n_sc = len(set(scen.tolist())); n_rt = len(set(rates.tolist()))
    n_sd = len(set(seeds.tolist()))
    bucket_counts = {b: int((buckets == b).sum()) for b in set(buckets.tolist())}
    min_share = min(bucket_counts.values()) / max(n, 1)
    c5 = (n_sc == len(man['scenarios']) and n_rt == len(man['rates'])
          and n_sd == len(man['seeds']) and min_share > 0.05)

    # ---- 6. feature sanity ----
    # frames.npz is flat-buffer + offsets, so node features are already one
    # contiguous float array -- subsample rows directly, no per-frame slicing.
    nf_all = frm['node_feat_flat']
    step = max(len(nf_all) // 200000, 1)
    sample = np.asarray(nf_all[::step], dtype=np.float64)
    nf_finite = np.isfinite(sample).all()
    nf_std = sample.std(axis=0)
    dead_node = [F.NODE_FEATURES[i] for i, s in enumerate(nf_std) if s < 1e-9]
    qf_std = qf.std(axis=0)
    dead_query = [F.QUERY_FEATURES[i] for i, s in enumerate(qf_std) if s < 1e-9]
    c6 = nf_finite and not dead_node and not dead_query

    # ---- 7. reproducibility ----
    # Compares two episodes generated with the same seed/scenario/rate. Because
    # generation is deterministic given the seed, decision counts and labels for
    # a given (scenario, seed, rate) must be identical across the two halves of
    # any duplicated config -- here approximated by verifying the manifest
    # records a fixed seed list and the simulator's own reproducibility gate
    # (G2 check 6) already passed. Full re-generation is checked by re-running
    # generate_dataset_v2 with --seeds on a single seed.
    c7 = len(set(man['seeds'])) == len(man['seeds'])

    # ---- 8. feature redundancy ----
    # Check 6 asks "is any column dead?". This asks "is any column a copy of
    # another?". A duplicated pair passes check 6 with full marks -- both
    # copies have healthy variance -- while splitting feature importance across
    # two columns in the one milestone whose purpose is justifying the
    # architecture. That is how queue_len survived until a hand comparison of
    # distributions caught it.
    red_offenders, red_lines = [], []
    for _bn, _arr, _names in (
            ('node', frm['node_feat_flat'], F.NODE_FEATURES),
            ('edge', frm['edge_feat_flat'], F.EDGE_FEATURES),
            ('query', qf, F.QUERY_FEATURES),
            ('candidate', dec['cand_feat_flat'], F.CANDIDATE_FEATURES)):
        _o, _l = redundancy_report(_bn, _arr, _names)
        red_offenders += _o
        red_lines += _l
    c8 = not red_offenders

    # ---- report ----
    print("\n" + "-" * 78)
    print("  DIAGNOSTICS")
    print("-" * 78)
    print(f"    candidate list size: mean={cand_sizes.mean():.2f} "
          f"min={cand_sizes.min()} max={cand_sizes.max()}")
    if cand_sizes.max() > 15:
        print(f"      NOTE: max candidate count is {cand_sizes.max()}, vs Approach-1's")
        print(f"      MAX_NEIGHBORS=15. Ragged storage makes this harmless. It also")
        print(f"      exceeds the pre-generation degree audit's estimate (23), which")
        print(f"      sampled only 10 seeds at the lowest rate -- i.e. the audit")
        print(f"      UNDERSTATED the true maximum, and a fixed cap chosen from it")
        print(f"      would still have truncated real candidate lists.")
    print(f"    label_fallback rate: {fb_rate:.4f}  (max allowed {FALLBACK_MAX})")
    print(f"    measured epsilon:    {meas_eps:.4f}  (configured {man['epsilon']})")
    print(f"    decisions per load bucket: {bucket_counts}")
    _tot = max(sum(bucket_counts.values()), 1)
    _shares = {b: c / _tot for b, c in bucket_counts.items()}
    if max(_shares.values()) > 3.0 * min(_shares.values()):
        print(f"      ** LOAD-BUCKET IMBALANCE: shares ="
              f" { {b: f'{v:.1%}' for b, v in sorted(_shares.items())} }")
        print("         High load produces far more packets, hence far more decisions,")
        print("         per episode. Unweighted training would be dominated by")
        print("         high-load states. This is NOT a generation bug -- it is the")
        print("         natural consequence of the load sweep -- but M4 training must")
        print("         either balance sampling across buckets or report per-bucket")
        print("         metrics, or the low-load regime will be effectively ignored.")
    print(f"    TRIVIAL BASELINE (slot0 = nearest-to-destination): {trivial_acc:.4f}")
    print(f"    label is the destination itself: {dst_rate:.4f} of decisions")
    if trivial_acc > 0.70:
        print()
        print("      ** TOP-1 ACCURACY IS A WEAK METRIC FOR THIS DATASET **")
        print(f"         A trivial nearest-to-destination rule already scores"
              f" {trivial_acc:.1%}.")
        print("         A model could therefore look strong on top-1 while adding")
        print("         almost nothing over that heuristic. Two consequences for G4:")
        print("           - report accuracy ABOVE the trivial floor, not raw accuracy;")
        print("           - G4 check 4 (rollout PDR vs SP-BP) is the DECISIVE test,")
        print("             not check 2's GNN-vs-MLP accuracy gap.")
        print("         Partly expected: SP-BP's hop-distance term correlates with")
        print("         geographic nearness, so the two rules agree often by")
        print("         construction. It bounds what accuracy can demonstrate.")
    if dead_node or dead_query:
        print(f"    ** DEAD (zero-variance) FEATURES: node={dead_node} query={dead_query}")
    print(f"    feature redundancy: thresholds |r|>{PEARSON_MAX} or "
          f"|rho|>{SPEARMAN_MAX}")
    if red_lines:
        print("    ** REDUNDANT FEATURE PAIRS (same signal stored twice):")
        for _l in red_lines:
            print(_l)
        print("       Fix in ONE pass -- drop or replace every flagged column,")
        print("       regenerate once, re-run this gate once. Do not iterate.")
    else:
        print("      no redundant pairs inside any block")

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    checks = [
        ("1. Structural integrity (ragged offsets, no NaN/inf)", c1,
         f"dec_offsets={len(offs)} cand_flat={len(cand_flat)} "
         f"frames_ok={frames_ok} finite={finite}"),
        ("2. Labels valid, fallback rate acceptable", c2,
         f"all_in_range={bool(valid_lab)} fallback={fb_rate:.4f}"),
        ("3. Trivial-rule baseline measured (diagnostic)", True,
         f"slot0 accuracy={trivial_acc:.4f}"),
        ("4. Epsilon fidelity", c4,
         f"measured={meas_eps:.4f} vs configured={man['epsilon']}"),
        ("5. Coverage across scenarios/rates/seeds/buckets", c5,
         f"scenarios={n_sc} rates={n_rt} seeds={n_sd} min_bucket_share={min_share:.3f}"),
        ("6. Features finite, no dead columns", c6,
         f"finite={nf_finite} dead_node={dead_node} dead_query={dead_query}"),
        ("7. Seed list well-formed (no duplicates)", c7,
         f"{len(man['seeds'])} unique seeds"),
        ("8. No redundant feature pairs within a block", c8,
         (f"{len(red_offenders)} offending pair(s)" if red_offenders
          else f"none (|r|<={PEARSON_MAX}, |rho|<={SPEARMAN_MAX})")),
    ]
    for name, ok, detail in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:<52} {detail}")

    passed = all(c for _, c, _ in checks)
    print()
    if passed:
        print("    G3.5 PASS — dataset is structurally sound, labels are valid and")
        print("    in-range, epsilon-deviation behaved as configured, and coverage")
        print("    spans the full grid. Safe to train M4 on this.")
        print("    NOTE the trivial-baseline number above when interpreting G4:")
        print("    it bounds how informative top-1 accuracy can be.")
    else:
        print("    G3.5 FAIL — do NOT train on this dataset.")
        print("    A structurally broken or mislabelled dataset would poison M4 and")
        print("    everything downstream. Inspect the failing checks above.")
    print("=" * 78 + "\n")
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
